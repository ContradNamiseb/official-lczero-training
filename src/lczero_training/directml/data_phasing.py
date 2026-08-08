"""Rotate which training files a run reads from, by step phase.

Long DirectML runs on the Iris Xe eventually OOM not from steady-state
memory but from the data loader indexing hundreds of tars at once: the
``ShufflingChunkPool`` metadata and the loader pipeline's queues grow with
the source count, and the DirectML allocator cannot reclaim the space once
the pool has walked the whole directory. Limiting the visible corpus to a
window of tars -- and advancing that window as training progresses -- keeps
the resident set small while still covering the full dataset over a run.

The mechanism is a symlink farm: a small directory containing links to only
the current phase's tar files, which the ``file_path_provider`` is pointed
at instead of the real corpus. The loader opens tars by content, so it
cannot tell a symlink from the file itself, and swapping a phase is an
atomic directory rebuild. No C++ changes are needed.

Phases are deterministic and derived from the step alone: tars are sorted
oldest-first (chronological curriculum), grouped into consecutive windows
of ``file_count``, and the phase index is
``(step // phase_step_interval) % num_groups``. Resuming from a checkpoint
at step N automatically lands on the same window that step would have used,
so the feature survives restarts and checkpoint resume.
"""

from __future__ import annotations

import dataclasses
import logging
import os
import pathlib

logger = logging.getLogger(__name__)

# Steps per phase when none is given. Deliberately in the range a single
# run reaches: the window is chosen once at startup and never rotates
# inside a process (rebuilding the farm under a running loader would mean
# deleting tars it holds open), so it only advances across restarts. A
# value far above a typical run length would pin every restart to the same
# window and quietly turn the whole corpus into one slice of it.
DEFAULT_PHASE_STEP_INTERVAL = 25_000


class DataPhasingError(Exception):
    """Raised when the requested phasing cannot be satisfied."""


@dataclasses.dataclass
class FarmStats:
    """How the farm entries were materialized."""

    symlinked: int = 0
    hardlinked: int = 0
    copied: int = 0


def _list_tars(directory: pathlib.Path) -> list[pathlib.Path]:
    """All training files in the corpus, oldest first.

    Sorts by name: the training-data naming scheme is a zero-padded
    timestamped sequence (``training-cleanvisits-20260724-0007.tar``), so a
    lexicographic name sort is also a chronological one. Case-insensitive
    on the extension so ``.TAR`` is not silently dropped.
    """
    if not directory.is_dir():
        raise DataPhasingError(f"data directory does not exist: {directory}")
    tars = sorted(
        (
            entry
            for entry in directory.iterdir()
            if entry.suffix.lower() == ".tar"
        ),
        key=lambda entry: entry.name,
    )
    if not tars:
        raise DataPhasingError(f"no .tar files found in {directory}")
    return tars


def phase_window(
    *,
    directory: pathlib.Path,
    file_count: int,
    phase_step_interval: int,
    step: int,
) -> list[pathlib.Path]:
    """The tar files for the phase that owns ``step``.

    Raises DataPhasingError if the window cannot be built. Returns a list of
    ``file_count`` paths, oldest first, chosen deterministically from the
    step so a resumed run reproduces the same window.
    """
    if file_count < 1:
        raise DataPhasingError(f"file_count must be >= 1, got {file_count}")
    if phase_step_interval < 1:
        raise DataPhasingError(
            f"phase_step_interval must be >= 1, got {phase_step_interval}"
        )
    tars = _list_tars(directory)
    groups = [
        tars[index : index + file_count]
        for index in range(0, len(tars), file_count)
    ]
    phase = (step // phase_step_interval) % len(groups)
    window = groups[phase]
    logger.info(
        "Data phase %d/%d for step %d: %d tar(s), %s .. %s",
        phase + 1,
        len(groups),
        step,
        len(window),
        window[0].name,
        window[-1].name,
    )
    return window


def build_phase_farm(
    *,
    farm_dir: pathlib.Path,
    window: list[pathlib.Path],
    allow_copy: bool = True,
) -> FarmStats:
    """Point ``farm_dir`` at exactly ``window``. Returns how it was done.

    Rebuilds the farm from empty: clears any prior contents, then links
    each tar. Only safe to call while no loader is reading the farm, which
    is why the daemon does it once before the pipeline starts.

    Symlinks on Windows need developer mode or the
    SeCreateSymbolicLinkPrivilege; without either, falls back to hardlinks
    (same volume, no privilege) and finally to copying. Copying is a real
    cost -- a window of 20 tars is several GB -- so it is counted, warned
    about, and can be refused outright with ``allow_copy=False``.
    """
    farm_dir.mkdir(parents=True, exist_ok=True)
    for stale in farm_dir.iterdir():
        if stale.is_symlink() or stale.is_file():
            stale.unlink()

    stats = FarmStats()
    for tar in window:
        destination = farm_dir / tar.name
        try:
            os.symlink(tar, destination)
            stats.symlinked += 1
            continue
        except OSError:
            pass
        try:
            # Hardlinks need the same volume. They also share the file's
            # data, so the farm costs directory entries and nothing else.
            os.link(tar, destination)
            stats.hardlinked += 1
            continue
        except OSError:
            pass
        if not allow_copy:
            raise DataPhasingError(
                f"cannot link {tar.name} into {farm_dir} and copying is "
                "disabled; enable Windows developer mode for symlinks, or "
                "put the farm on the same volume as the corpus for "
                "hardlinks"
            )
        import shutil

        shutil.copy2(tar, destination)
        stats.copied += 1

    if stats.copied:
        total_bytes = sum(
            (farm_dir / tar.name).stat().st_size for tar in window
        )
        logger.warning(
            "Phase farm %s copied %d of %d tar(s) -- roughly %.1f GB of "
            "duplicate data, rewritten on every start. Enable Windows "
            "developer mode (symlinks) or keep the farm on the corpus's "
            "volume (hardlinks) to avoid this.",
            farm_dir,
            stats.copied,
            len(window),
            total_bytes / 1e9,
        )
    logger.info(
        "Phase farm %s exposes %d tar(s) (%d symlink, %d hardlink, %d copy)",
        farm_dir,
        len(window),
        stats.symlinked,
        stats.hardlinked,
        stats.copied,
    )
    return stats
