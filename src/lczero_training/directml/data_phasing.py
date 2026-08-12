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

``shuffle_seed`` trades the chronological curriculum for variety. Passing it
does not make phase selection non-reproducible -- the same (seed, step)
still yields the same window, which is what a checkpoint resume depends on
-- it changes what "the same" means. Sequential grouping means phase 0 is
always the oldest tars and phase 1 is always the next-oldest, forever;
across a run with many restarts against a small ``file_count``, that is a
handful of fixed neighbor groups repeating on a fixed schedule. Shuffling
partitions the corpus into ``file_count``-sized groups after randomizing the
order, so a run instead see a fresh combination of tars, and a run long
enough to lap back to phase 0 (every ``num_groups * phase_step_interval``
steps) gets a freshly reshuffled partition rather than the exact one it saw
last time -- still a full, non-overlapping partition of the corpus within
any one lap, just not the same one twice.
"""

from __future__ import annotations

import dataclasses
import hashlib
import logging
import os
import pathlib
import random

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
    shuffle_seed: str | None = None,
) -> list[pathlib.Path]:
    """The tar files for the phase that owns ``step``.

    Raises DataPhasingError if the window cannot be built. Returns a list of
    ``file_count`` paths, chosen deterministically from ``step`` (and
    ``shuffle_seed``, if given) so a resumed run reproduces the same window.

    Oldest-first when ``shuffle_seed`` is None (the default): phase 0 is
    always the oldest tars, phase 1 the next-oldest, and so on -- a fixed
    chronological curriculum. With a seed, the corpus is instead partitioned
    after a pseudo-random shuffle, reseeded once per "epoch" (one full lap
    through every phase, ``num_groups * phase_step_interval`` steps) so a
    run that laps more than once does not see the exact same partition
    twice. Still fully deterministic: the same directory contents, seed and
    step always produce the same window.
    """
    if file_count < 1:
        raise DataPhasingError(f"file_count must be >= 1, got {file_count}")
    if phase_step_interval < 1:
        raise DataPhasingError(
            f"phase_step_interval must be >= 1, got {phase_step_interval}"
        )
    tars = _list_tars(directory)
    ordered = list(tars)
    epoch = None
    if shuffle_seed:
        num_groups = -(-len(tars) // file_count)  # ceil division
        epoch = step // (phase_step_interval * num_groups)
        # sha256 rather than Python's hash(): that one is salted per
        # process (PYTHONHASHSEED), so two runs -- or a run and the daemon
        # that resumed it -- would shuffle differently for what is meant to
        # be the identical window.
        digest = hashlib.sha256(
            f"{shuffle_seed}:{epoch}".encode("utf-8")
        ).hexdigest()
        random.Random(digest).shuffle(ordered)
    groups = [
        ordered[index : index + file_count]
        for index in range(0, len(ordered), file_count)
    ]
    phase = (step // phase_step_interval) % len(groups)
    window = groups[phase]
    if shuffle_seed:
        # Not a ".. " range: a shuffled window's first and last entries are
        # not its oldest and newest, so presenting them as a span would be
        # actively misleading. List a few instead.
        sample = ", ".join(tar.name for tar in window[:3])
        more = f" (+{len(window) - 3} more)" if len(window) > 3 else ""
        logger.info(
            "Data phase %d/%d for step %d (shuffled, epoch %d): "
            "%d tar(s): %s%s",
            phase + 1,
            len(groups),
            step,
            epoch,
            len(window),
            sample,
            more,
        )
    else:
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
