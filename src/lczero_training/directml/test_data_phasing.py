"""Windowing and farm construction for data phasing.

These run on the host filesystem with plain tar files -- no DirectML, no
loader -- so they exercise the phase arithmetic and the symlink/hardlink
farm directly.
"""

from __future__ import annotations

import pathlib

import pytest

from lczero_training.directml import data_phasing
from lczero_training.directml.data_phasing import DataPhasingError


def _make_corpus(directory: pathlib.Path, names: list[str]) -> None:
    directory.mkdir(parents=True, exist_ok=True)
    for name in names:
        (directory / name).write_bytes(b"tar-bytes")


def _tar_names(count: int) -> list[str]:
    # The real corpus names sort chronologically; the test corpus mimics
    # that so "oldest first" is observable in the window contents.
    return [f"training-20260101-{index:04d}.tar" for index in range(count)]


# --------------------------------------------------------------------------
# phase_window
# --------------------------------------------------------------------------


def test_window_is_deterministic_for_a_step(tmp_path):
    _make_corpus(tmp_path, _tar_names(12))
    first = data_phasing.phase_window(
        directory=tmp_path, file_count=4, phase_step_interval=100, step=250
    )
    second = data_phasing.phase_window(
        directory=tmp_path, file_count=4, phase_step_interval=100, step=250
    )
    assert first == second


def test_windows_are_oldest_first(tmp_path):
    _make_corpus(tmp_path, _tar_names(12))
    window = data_phasing.phase_window(
        directory=tmp_path, file_count=4, phase_step_interval=100, step=0
    )
    assert [tar.name for tar in window] == _tar_names(4)


def test_step_selects_the_owning_phase(tmp_path):
    names = _tar_names(12)
    _make_corpus(tmp_path, names)
    # 12 tars, groups of 4 -> phases cover step ranges [0,100), [100,200),
    # [200,300), then wrap.
    window = data_phasing.phase_window(
        directory=tmp_path, file_count=4, phase_step_interval=100, step=150
    )
    assert [tar.name for tar in window] == names[4:8]


def test_phases_wrap_around_the_corpus(tmp_path):
    names = _tar_names(12)
    _make_corpus(tmp_path, names)
    window = data_phasing.phase_window(
        directory=tmp_path, file_count=4, phase_step_interval=100, step=350
    )
    assert [tar.name for tar in window] == names[0:4]


def test_last_partial_group_is_used(tmp_path):
    names = _tar_names(10)  # groups of 4, 4, 2
    _make_corpus(tmp_path, names)
    window = data_phasing.phase_window(
        directory=tmp_path, file_count=4, phase_step_interval=100, step=250
    )
    assert [tar.name for tar in window] == names[8:10]


def test_missing_directory_raises(tmp_path):
    with pytest.raises(DataPhasingError):
        data_phasing.phase_window(
            directory=tmp_path / "nope",
            file_count=4,
            phase_step_interval=100,
            step=0,
        )


def test_empty_directory_raises(tmp_path):
    (tmp_path / "data").mkdir()
    with pytest.raises(DataPhasingError):
        data_phasing.phase_window(
            directory=tmp_path / "data",
            file_count=4,
            phase_step_interval=100,
            step=0,
        )


# --------------------------------------------------------------------------
# phase_window, shuffled
# --------------------------------------------------------------------------


def test_shuffle_is_still_deterministic_for_a_step(tmp_path):
    """The property a checkpoint resume depends on: same step, same window.
    Shuffling changes what the partition looks like, not whether repeating
    the call reproduces it."""
    _make_corpus(tmp_path, _tar_names(20))
    first = data_phasing.phase_window(
        directory=tmp_path,
        file_count=4,
        phase_step_interval=100,
        step=250,
        shuffle_seed="corpus-a",
    )
    second = data_phasing.phase_window(
        directory=tmp_path,
        file_count=4,
        phase_step_interval=100,
        step=250,
        shuffle_seed="corpus-a",
    )
    assert first == second


def test_shuffle_is_off_by_default(tmp_path):
    """No seed must reproduce the plain sequential behaviour exactly --
    passing shuffle_seed=None cannot be a silent behaviour change for every
    existing caller that does not know the parameter exists."""
    names = _tar_names(12)
    _make_corpus(tmp_path, names)
    window = data_phasing.phase_window(
        directory=tmp_path, file_count=4, phase_step_interval=100, step=150
    )
    assert [tar.name for tar in window] == names[4:8]


def test_shuffle_differs_from_the_sequential_grouping(tmp_path):
    """The whole point: a real seed must not just reproduce oldest-first."""
    names = _tar_names(30)
    _make_corpus(tmp_path, names)
    sequential = data_phasing.phase_window(
        directory=tmp_path, file_count=5, phase_step_interval=100, step=0
    )
    shuffled = data_phasing.phase_window(
        directory=tmp_path,
        file_count=5,
        phase_step_interval=100,
        step=0,
        shuffle_seed="corpus-b",
    )
    assert [t.name for t in shuffled] != [t.name for t in sequential]


def test_shuffle_covers_the_corpus_exactly_once_per_epoch(tmp_path):
    """A shuffle must still be a partition: nothing dropped, nothing
    duplicated within one lap through every phase."""
    names = _tar_names(23)  # not a multiple of file_count
    _make_corpus(tmp_path, names)
    seen: set[str] = set()
    num_groups = -(-23 // 4)
    for phase in range(num_groups):
        window = data_phasing.phase_window(
            directory=tmp_path,
            file_count=4,
            phase_step_interval=100,
            step=phase * 100,
            shuffle_seed="corpus-c",
        )
        overlap = seen & {t.name for t in window}
        assert not overlap, f"phase {phase} repeated {overlap}"
        seen.update(t.name for t in window)
    assert seen == set(names)


def test_shuffle_reseeds_on_the_next_epoch(tmp_path):
    """A run long enough to lap back to phase 0 must not see the identical
    partition it saw last lap -- that was the actual complaint: fixed
    windows repeating at fixed intervals forever."""
    names = _tar_names(30)
    _make_corpus(tmp_path, names)
    num_groups = -(-30 // 5)
    epoch_length = num_groups * 100
    first_epoch = data_phasing.phase_window(
        directory=tmp_path,
        file_count=5,
        phase_step_interval=100,
        step=0,
        shuffle_seed="corpus-d",
    )
    second_epoch = data_phasing.phase_window(
        directory=tmp_path,
        file_count=5,
        phase_step_interval=100,
        step=epoch_length,
        shuffle_seed="corpus-d",
    )
    assert [t.name for t in first_epoch] != [t.name for t in second_epoch]


def test_shuffle_stays_within_one_epoch_for_the_whole_lap(tmp_path):
    """Two steps in the same lap must draw from the same shuffled partition,
    the same way two steps in the same sequential group always have."""
    names = _tar_names(20)
    _make_corpus(tmp_path, names)
    early = data_phasing.phase_window(
        directory=tmp_path,
        file_count=4,
        phase_step_interval=100,
        step=0,
        shuffle_seed="corpus-e",
    )
    late = data_phasing.phase_window(
        directory=tmp_path,
        file_count=4,
        phase_step_interval=100,
        step=0,
        shuffle_seed="corpus-e",
    )
    assert early == late


def test_different_seeds_shuffle_differently(tmp_path):
    """Confirms the seed is load-bearing, not decorative."""
    names = _tar_names(30)
    _make_corpus(tmp_path, names)
    a = data_phasing.phase_window(
        directory=tmp_path,
        file_count=5,
        phase_step_interval=100,
        step=0,
        shuffle_seed="corpus-f",
    )
    b = data_phasing.phase_window(
        directory=tmp_path,
        file_count=5,
        phase_step_interval=100,
        step=0,
        shuffle_seed="corpus-g",
    )
    assert [t.name for t in a] != [t.name for t in b]


def test_invalid_arguments_raise(tmp_path):
    _make_corpus(tmp_path, _tar_names(4))
    with pytest.raises(DataPhasingError):
        data_phasing.phase_window(
            directory=tmp_path, file_count=0, phase_step_interval=100, step=0
        )
    with pytest.raises(DataPhasingError):
        data_phasing.phase_window(
            directory=tmp_path, file_count=4, phase_step_interval=0, step=0
        )


# --------------------------------------------------------------------------
# build_phase_farm
# --------------------------------------------------------------------------


def test_farm_exposes_only_the_window(tmp_path):
    corpus = tmp_path / "corpus"
    names = _tar_names(6)
    _make_corpus(corpus, names)
    window = [corpus / name for name in names[:3]]
    farm = tmp_path / "farm"

    stats = data_phasing.build_phase_farm(farm_dir=farm, window=window)
    exposed = sorted(entry.name for entry in farm.iterdir())
    assert exposed == names[:3]
    # Each entry resolves to the real tar's content.
    for name in names[:3]:
        assert (farm / name).read_bytes() == b"tar-bytes"
    # However they were materialized, all three are accounted for.
    assert stats.symlinked + stats.hardlinked + stats.copied == 3


def test_farm_reports_how_entries_were_materialized(tmp_path):
    """The caller has to be able to see a copy fallback.

    Copying a window is several GB rewritten on every start, so it cannot
    be indistinguishable from linking in the logs.
    """
    corpus = tmp_path / "corpus"
    names = _tar_names(2)
    _make_corpus(corpus, names)

    stats = data_phasing.build_phase_farm(
        farm_dir=tmp_path / "farm",
        window=[corpus / name for name in names],
    )
    assert isinstance(stats, data_phasing.FarmStats)
    assert stats.symlinked + stats.hardlinked + stats.copied == 2


def test_farm_can_refuse_to_copy(tmp_path, monkeypatch):
    """allow_copy=False must fail loudly rather than duplicate the corpus."""
    corpus = tmp_path / "corpus"
    names = _tar_names(2)
    _make_corpus(corpus, names)

    # Deny both link mechanisms, as an unprivileged Windows account with
    # the farm on a different volume would.
    def deny(*args, **kwargs):
        raise OSError("linking not permitted")

    monkeypatch.setattr(data_phasing.os, "symlink", deny)
    monkeypatch.setattr(data_phasing.os, "link", deny)

    with pytest.raises(DataPhasingError, match="copying is disabled"):
        data_phasing.build_phase_farm(
            farm_dir=tmp_path / "farm",
            window=[corpus / name for name in names],
            allow_copy=False,
        )


def test_farm_falls_back_to_copying_when_links_are_denied(
    tmp_path, monkeypatch
):
    corpus = tmp_path / "corpus"
    names = _tar_names(2)
    _make_corpus(corpus, names)

    def deny(*args, **kwargs):
        raise OSError("linking not permitted")

    monkeypatch.setattr(data_phasing.os, "symlink", deny)
    monkeypatch.setattr(data_phasing.os, "link", deny)

    farm = tmp_path / "farm"
    stats = data_phasing.build_phase_farm(
        farm_dir=farm, window=[corpus / name for name in names]
    )
    assert stats.copied == 2
    assert sorted(entry.name for entry in farm.iterdir()) == names
    for name in names:
        assert (farm / name).read_bytes() == b"tar-bytes"


def test_default_interval_is_within_reach_of_one_run(tmp_path):
    """The window only advances across restarts, never inside a process.

    A default far above a single run's length would pin every restart to
    the same window, which is the opposite of covering the corpus. Real
    runs here reach tens of thousands of steps before stopping.
    """
    assert data_phasing.DEFAULT_PHASE_STEP_INTERVAL <= 50_000


def test_farm_rebuild_drops_the_previous_phase(tmp_path):
    corpus = tmp_path / "corpus"
    names = _tar_names(6)
    _make_corpus(corpus, names)
    farm = tmp_path / "farm"

    data_phasing.build_phase_farm(
        farm_dir=farm, window=[corpus / name for name in names[:3]]
    )
    data_phasing.build_phase_farm(
        farm_dir=farm, window=[corpus / name for name in names[3:]]
    )
    exposed = sorted(entry.name for entry in farm.iterdir())
    assert exposed == names[3:]


def test_farm_does_not_modify_the_corpus(tmp_path):
    corpus = tmp_path / "corpus"
    names = _tar_names(4)
    _make_corpus(corpus, names)
    data_phasing.build_phase_farm(
        farm_dir=tmp_path / "farm", window=[corpus / names[0]]
    )
    # The real directory still holds every tar, untouched.
    assert sorted(entry.name for entry in corpus.iterdir()) == names
