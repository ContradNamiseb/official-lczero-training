"""Clearing the shader caches and restarting the driver between launches.

This runs on the path *to* starting a training run, so the property that
matters most is that nothing here can stop one: a reset that fails is a
tidy-up that did not happen, not a reason to refuse to train.
"""

import logging
from pathlib import Path

import pytest

from lczero_training.directml import gpu_reset


@pytest.fixture
def caches(tmp_path, monkeypatch):
    """Two fake cache directories with something in them."""
    paths = []
    for name in ("D3DSCache", "ShaderCache"):
        directory = tmp_path / name
        (directory / "nested").mkdir(parents=True)
        (directory / "shader.bin").write_bytes(b"x" * 2048)
        (directory / "nested" / "more.bin").write_bytes(b"y" * 1024)
        paths.append(directory)
    monkeypatch.setattr(gpu_reset, "CACHE_PATHS", tuple(paths))
    return paths


def test_it_clears_both_caches_but_keeps_the_directories(caches):
    """Contents, not the folders: the driver expects them to exist and
    recreates entries inside them on demand."""
    cleared = gpu_reset.clear_shader_caches()

    # 2048 + 1024 bytes in each of the two directories.
    assert cleared == pytest.approx(2 * 3072 / 1024**2, rel=0.01)
    for directory in caches:
        assert directory.is_dir(), "the directory itself must survive"
        assert list(directory.iterdir()) == []


def test_dry_run_measures_without_deleting(caches):
    cleared = gpu_reset.clear_shader_caches(dry_run=True)

    assert cleared > 0
    for directory in caches:
        assert (directory / "shader.bin").exists()


def test_a_missing_cache_directory_is_not_an_error(tmp_path, monkeypatch):
    monkeypatch.setattr(gpu_reset, "CACHE_PATHS", (tmp_path / "never-existed",))

    assert gpu_reset.clear_shader_caches() == 0.0


def test_off_does_nothing_at_all(caches, monkeypatch):
    monkeypatch.setattr(
        gpu_reset,
        "restart_display_driver",
        lambda: pytest.fail("off must not touch the driver"),
    )

    gpu_reset.reset("off")

    assert (caches[0] / "shader.bin").exists()


def test_cache_mode_leaves_the_driver_alone(caches, monkeypatch):
    monkeypatch.setattr(
        gpu_reset,
        "restart_display_driver",
        lambda: pytest.fail("cache mode must not reset the driver"),
    )

    gpu_reset.reset("cache")

    assert list(caches[0].iterdir()) == []


def test_full_mode_resets_the_driver_and_waits(caches, monkeypatch):
    """The reset is asynchronous. Building a DirectML device while the driver
    is still coming back fails a launch that would otherwise be fine."""
    calls = {"driver": 0, "slept": 0.0}
    monkeypatch.setattr(
        gpu_reset,
        "restart_display_driver",
        lambda: calls.__setitem__("driver", calls["driver"] + 1) or True,
    )
    monkeypatch.setattr(
        gpu_reset.time, "sleep", lambda s: calls.__setitem__("slept", s)
    )

    gpu_reset.reset("full")

    assert calls["driver"] == 1
    assert calls["slept"] == gpu_reset.SETTLE_SECONDS


def test_it_does_not_wait_when_the_keystroke_could_not_be_sent(
    caches, monkeypatch
):
    """No interactive desktop, no reset, nothing to wait for."""
    monkeypatch.setattr(gpu_reset, "restart_display_driver", lambda: False)
    monkeypatch.setattr(
        gpu_reset.time,
        "sleep",
        lambda s: pytest.fail("nothing was reset, so nothing to settle"),
    )

    gpu_reset.reset("full")


def test_a_failing_reset_never_blocks_a_launch(monkeypatch, caplog):
    """The supervisor calls this immediately before spawning the trainer.
    A tidy-up that throws would turn a working run into no run at all."""

    def explode(*_args, **_kwargs):
        raise OSError("the cache is on fire")

    monkeypatch.setattr(gpu_reset, "clear_shader_caches", explode)

    with caplog.at_level(logging.WARNING):
        gpu_reset.reset("full")  # must return normally

    assert any(
        "continuing to launch anyway" in record.message
        for record in caplog.records
    )


def test_the_cache_paths_are_the_two_real_ones():
    """Intel's is the larger of the two on this hardware and was missing from
    the first version of this."""
    names = [Path(p).name for p in gpu_reset.CACHE_PATHS]

    assert "D3DSCache" in names
    assert "ShaderCache" in names
