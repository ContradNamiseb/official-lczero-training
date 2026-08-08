"""The restart supervisor's loop, without launching a trainer.

Every test here fakes ``_run_daemon`` and drives the checkpoint directory by
hand, because the real child needs a DirectML adapter, four minutes of data
loader startup, and several gigabytes that CI does not have. What is worth
testing is the arithmetic and the decisions: which target each launch aims
at, when to stop relaunching, and that a crashed run is resumed from
wherever its recovery checkpoint landed.
"""

import contextlib
import ctypes
import pathlib

import pytest
from google.protobuf import text_format

from lczero_training.commands import directml_supervisor as supervisor
from proto.root_config_pb2 import RootConfig


@pytest.fixture
def config_file(tmp_path):
    """A config whose only interesting field is the checkpoint path."""
    config = RootConfig()
    config.training.checkpoint.path = str(tmp_path / "checkpoints")
    path = tmp_path / "config.textproto"
    path.write_text(text_format.MessageToString(config))
    return path


def _write_checkpoint(directory, step: int) -> None:
    """A file the supervisor's scan will find. Contents never matter -- it
    reads the step out of the name and never opens it."""
    path = pathlib.Path(directory)
    path.mkdir(parents=True, exist_ok=True)
    (path / f"checkpoint-{step:09d}.pt").write_bytes(b"")


class _FakeTrainer:
    """Stands in for ``_run_daemon``, recording the commands it was given.

    ``advances`` is the step gain each successive launch achieves; a gain of
    zero is a launch that died before its first checkpoint.
    """

    def __init__(self, directory, advances, code: int = 0):
        self.directory = directory
        self.advances = list(advances)
        self.code = code
        self.commands: list[list[str]] = []

    def __call__(self, command, job):
        self.commands.append(list(command))
        gain = self.advances.pop(0) if self.advances else 0
        if gain:
            from lczero_training.directml import checkpoint as checkpoint_io

            current = checkpoint_io.latest_step(self.directory) or 0
            _write_checkpoint(self.directory, current + gain)
        return self.code

    def target_steps(self) -> list[int]:
        return [
            int(item.split("=", 1)[1])
            for command in self.commands
            for item in command
            if item.startswith("--target-step=")
        ]


@pytest.fixture
def run(monkeypatch, config_file, tmp_path):
    """Drive main() with a faked trainer. Returns (exit code, trainer)."""
    directory = tmp_path / "checkpoints"

    def go(argv, advances, code=0, start_step=100000):
        if start_step is not None:
            _write_checkpoint(directory, start_step)
        trainer = _FakeTrainer(directory, advances, code)
        monkeypatch.setattr(supervisor, "_run_daemon", trainer)
        # No job object: it would be a real kernel handle for a child that
        # is never launched.
        monkeypatch.setattr(
            supervisor,
            "_kill_children_with_me",
            lambda: contextlib.nullcontext(None),
        )
        # Before *argv, which may carry its own `--`: a flag after the
        # separator would be passed to the trainer instead of read here.
        exit_code = supervisor.main(
            ["--config", str(config_file), "--backoff", "0", *argv]
        )
        return exit_code, trainer

    return go


def test_each_launch_aims_at_its_own_nearer_target(run):
    """The proactive restart, which is the whole point.

    A restart budget must cap each launch's target so the daemon exits
    cleanly -- checkpointing and exporting on the way out -- rather than
    running until the allocator refuses and taking the segment with it.
    """
    code, trainer = run(
        ["--target-step", "125000", "--restart-every", "10000"],
        advances=[10000, 10000, 5000],
    )

    assert code == 0
    assert trainer.target_steps() == [110000, 120000, 125000], (
        "the last launch must aim at the finish line, not past it"
    )


def test_a_crash_resumes_from_where_the_checkpoint_reached(run):
    """A run that dies partway is resumed from its recovery checkpoint, not
    from the start of the segment it was in."""
    code, trainer = run(
        ["--target-step", "120000", "--restart-every", "10000"],
        advances=[3500, 10000, 6500],
        code=1,
    )

    assert code == 0
    # 100000 -> 103500 (crashed early), then a fresh 10000-step budget from
    # 103500, and finally the finish line rather than 123500.
    assert trainer.target_steps() == [110000, 113500, 120000]


def test_no_restart_budget_runs_straight_at_the_finish_line(run):
    code, trainer = run(
        ["--target-step", "120000", "--restart-every", "0"],
        advances=[20000],
    )

    assert code == 0
    assert trainer.target_steps() == [120000]


def test_it_gives_up_after_repeated_launches_that_gain_nothing(run):
    """The circuit breaker. Without it a run that cannot start at all --
    a bad config, a missing corpus -- relaunches forever."""
    code, trainer = run(
        ["--target-step", "200000", "--max-stalls", "3"],
        advances=[0, 0, 0, 0, 0],
        code=1,
    )

    assert code == 1
    assert len(trainer.commands) == 3, "must stop at --max-stalls launches"


def test_progress_resets_the_stall_counter(run):
    """Occasional launches that die instantly must not add up across hours
    of successful training to end a healthy run."""
    code, trainer = run(
        [
            "--target-step",
            "130000",
            "--restart-every",
            "10000",
            "--max-stalls",
            "2",
        ],
        advances=[0, 10000, 0, 10000, 0, 10000],
    )

    assert code == 0
    assert len(trainer.commands) == 6


def test_max_launches_caps_a_healthy_run(run):
    """A hard ceiling on an unattended run, separate from --max-stalls: this
    one stops launches that are all making progress."""
    code, trainer = run(
        [
            "--target-step",
            "500000",
            "--restart-every",
            "10000",
            "--max-launches",
            "2",
        ],
        advances=[10000, 10000, 10000, 10000],
    )

    assert code == 0, "hitting the cap is a clean stop, not a failure"
    assert len(trainer.commands) == 2
    assert trainer.target_steps() == [110000, 120000]


def test_it_refuses_to_start_without_a_checkpoint(run):
    code, trainer = run(
        ["--target-step", "200000"], advances=[], start_step=None
    )

    assert code == 1
    assert not trainer.commands, "nothing to resume from, so nothing to launch"


def test_it_does_nothing_when_the_target_is_already_reached(run):
    code, trainer = run(["--target-step", "100000"], advances=[])

    assert code == 0
    assert not trainer.commands


def test_supervisor_flags_after_the_separator_are_rejected(run):
    """--target-step in the passthrough would reach the daemon twice, and
    argparse takes the last -- silently disabling the restart budget."""
    code, trainer = run(
        ["--target-step", "200000", "--", "--target-step=999999"],
        advances=[10000],
    )

    assert code == 2
    assert not trainer.commands


def test_passthrough_flags_reach_every_launch(run):
    code, trainer = run(
        [
            "--target-step",
            "120000",
            "--restart-every",
            "10000",
            "--",
            "--kda-chunk-size=8",
            "--gc-every=500",
        ],
        advances=[10000, 10000],
    )

    assert code == 0
    for command in trainer.commands:
        assert "--kda-chunk-size=8" in command
        assert "--gc-every=500" in command


# --- the flag boundary the TUI shares with this command --------------------


def test_partition_flags_splits_on_the_supervisor_s_own_flags():
    mine, theirs = supervisor.partition_flags(
        [
            "--config",
            "docs/kda_split.textproto",
            "--target-step=1000000",
            "--kda-chunk-size=8",
            "--restart-every",
            "15000",
            "--report-every=10",
        ]
    )

    assert mine == [
        "--config",
        "docs/kda_split.textproto",
        "--target-step=1000000",
        "--restart-every",
        "15000",
    ], "a `--flag value` pair has to travel together"
    assert theirs == ["--kda-chunk-size=8", "--report-every=10"]


def test_partition_flags_leaves_a_plain_daemon_list_alone():
    flags = ["--kda-chunk-size=8", "--gc-every", "500", "--eval-every=5000"]
    mine, theirs = supervisor.partition_flags(flags)

    assert mine == []
    assert theirs == flags


def test_the_tui_builds_a_supervised_command_with_the_separator():
    """--supervise has to move the supervisor's flags in front of the `--`;
    the TUI collects them all in one flat list from its own passthrough."""
    import argparse

    from lczero_training.tui.directml_app import DirectMlTuiApp

    args = argparse.Namespace(
        config="docs/kda_split.textproto",
        supervise=True,
        daemon_flags=[
            "--config",
            "docs/kda_split.textproto",
            "--target-step=1000000",
            "--kda-chunk-size=8",
        ],
    )
    command = DirectMlTuiApp(args=args)._child_command()

    assert command[1:3] == ["-m", supervisor.__name__]
    separator = command.index("--")
    assert "--target-step=1000000" in command[:separator]
    assert command[separator + 1 :] == ["--kda-chunk-size=8"]


def test_the_tui_spawns_the_daemon_directly_without_supervise():
    import argparse

    from lczero_training.tui.directml_app import DirectMlTuiApp

    args = argparse.Namespace(
        config="c.textproto",
        supervise=False,
        daemon_flags=["--config", "c.textproto", "--target-step=1000"],
    )
    command = DirectMlTuiApp(args=args)._child_command()

    assert command[2] == "lczero_training.commands.directml_daemon"
    assert "--" not in command


# --- the Win32 structures, whose field sizes are not checked by anything --


def test_the_job_object_limit_structures_have_the_documented_sizes():
    """A field declared one width too narrow shifts every field after it, so
    LimitFlags would be written where the kernel does not read it and the
    kill-on-close guarantee would silently not apply. These are the sizes
    Win32 documents for a 64-bit process."""
    if ctypes.sizeof(ctypes.c_void_p) != 8:
        pytest.skip("the documented sizes below are the 64-bit ones")

    assert ctypes.sizeof(supervisor._IoCounters) == 48
    assert ctypes.sizeof(supervisor._BasicLimits) == 64
    assert ctypes.sizeof(supervisor._ExtendedLimits) == 144
    # The field that actually carries the guarantee, and the one a mis-sized
    # neighbour above it would move.
    assert supervisor._BasicLimits.LimitFlags.offset == 16
