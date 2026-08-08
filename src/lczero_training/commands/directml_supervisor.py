"""Restart supervisor for DirectML training: the only real cure for the leak.

    uv run lc0-directml-supervisor --config CONFIG --target-step 1000000 \
        -- --kda-chunk-size=8 --report-every=10

``torch_directml`` reaches PyTorch through the ``PrivateUse1`` backend, and
its tensors are ``OpaqueTensorImpl`` -- invisible to PyTorch's caching
allocator. DX12 suballocates from coarse multi-megabyte heaps and a heap
stays live until every suballocation in it is freed, so fragmentation
strands memory that is nominally free. There is no escape hatch:
``empty_cache``, ``memory_allocated`` and ``synchronize`` are all absent
from the installed build, so nothing inside the process can hand blocks
back. **The OS reclaims a DX12 context only on process termination.**

So this does not try. The daemon already checkpoints every
``steps_per_network`` steps and resumes from the last one exactly, which
makes a process restart lossless; this turns that into the recovery
mechanism. It relaunches the daemon until the target step is reached --
after a crash, and proactively every ``--restart-every`` steps so the
scheduled restart arrives before the wall does. The best unsupervised run
managed 36,397 steps before dying; restarting well inside that turns an
unavoidable leak into a routine event.

Restarting costs one data-loader startup, about four minutes. At roughly
0.9 s/step the default 15,000-step budget is under 2% overhead.

The daemon inherits this process's stdin, stdout and stderr, so its JSONL
protocol reaches whatever spawned *this* untouched and there is nothing to
relay -- `lc0-directml-tui --supervise` works for exactly that reason. This
process must therefore keep stdout clean and log to stderr, same as the
daemon.
"""

import argparse
import contextlib
import ctypes
import logging
import subprocess
import sys
import time

logger = logging.getLogger("lc0-directml-supervisor")

DAEMON_MODULE = "lczero_training.commands.directml_daemon"

# Flags this command consumes itself, rather than passing to the daemon.
SUPERVISOR_FLAGS = (
    "--config",
    "--checkpoint",
    "--target-step",
    "--restart-every",
    "--max-stalls",
    "--max-launches",
    "--backoff",
)

# The subset the daemon also understands. Repeating one of these after the
# `--` would leave the daemon with two, and argparse takes the last -- which
# for --target-step silently disables the proactive restart.
_OWNED_FLAGS = ("--config", "--checkpoint", "--target-step")


def partition_flags(flags) -> tuple[list[str], list[str]]:
    """Split one flat flag list into (supervisor flags, daemon flags).

    ``lc0-directml-tui`` collects everything after its own ``--`` into a
    single list, but the supervisor wants its own flags before the ``--``
    and the daemon's after it. Deciding the boundary here, next to the
    parser, is what keeps the two commands from drifting apart.
    """
    mine: list[str] = []
    theirs: list[str] = []
    remaining = list(flags)
    while remaining:
        item = remaining.pop(0)
        if item.split("=", 1)[0] not in SUPERVISOR_FLAGS:
            theirs.append(item)
            continue
        mine.append(item)
        # `--flag=value` carries its own value; `--flag value` leaves it as
        # the next item, which has to travel with it.
        if "=" not in item and remaining and not remaining[0].startswith("-"):
            mine.append(remaining.pop(0))
    return mine, theirs


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config", required=True, help="Root config textproto."
    )
    parser.add_argument(
        "--checkpoint",
        help="Checkpoint directory override. Defaults to the config's.",
    )
    parser.add_argument(
        "--target-step",
        type=int,
        required=True,
        help="Absolute step to finish at. Restarts until it is reached.",
    )
    parser.add_argument(
        "--restart-every",
        type=int,
        default=15000,
        help=(
            "Restart the trainer this many steps into each launch, before "
            "the DirectML allocator runs out. 0 restarts only after a "
            "failure, which costs the steps a crash takes with it."
        ),
    )
    parser.add_argument(
        "--max-stalls",
        type=int,
        default=3,
        help=(
            "Give up after this many consecutive launches that advance the "
            "checkpoint by nothing. Stops a misconfigured run from "
            "relaunching forever."
        ),
    )
    parser.add_argument(
        "--max-launches",
        type=int,
        default=0,
        help=(
            "Stop after this many launches even if they are all making "
            "progress, leaving the checkpoint where it reached. A hard cap on "
            "how long the run may go unattended; 0 means only --target-step "
            "ends it."
        ),
    )
    parser.add_argument(
        "--backoff",
        type=float,
        default=30.0,
        help="Seconds to wait after a launch that made no progress.",
    )
    return parser


def _checkpoint_directory(config_filepath: str, override: str | None) -> str:
    if override:
        return override
    from google.protobuf import text_format

    from proto.root_config_pb2 import RootConfig

    config = RootConfig()
    with open(config_filepath, "r") as handle:
        text_format.Parse(handle.read(), config)
    return config.training.checkpoint.path


# --- keeping the trainer from outliving us ---------------------------------

# Win32 aliases spelled out rather than taken from ctypes.wintypes, which
# does not import at all off Windows. Plain ctypes does, so the structures
# below can live at module scope and be checked by the tests anywhere.
_DWORD = ctypes.c_uint32
_HANDLE = ctypes.c_void_p
_JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE = 0x2000
_JobObjectExtendedLimitInformation = 9
_PROCESS_TERMINATE = 0x0001
_PROCESS_SET_QUOTA = 0x0100


class _IoCounters(ctypes.Structure):
    _fields_ = [
        (name, ctypes.c_uint64)
        for name in (
            "ReadOperationCount",
            "WriteOperationCount",
            "OtherOperationCount",
            "ReadTransferCount",
            "WriteTransferCount",
            "OtherTransferCount",
        )
    ]


class _BasicLimits(ctypes.Structure):
    _fields_ = [
        ("PerProcessUserTimeLimit", ctypes.c_int64),
        ("PerJobUserTimeLimit", ctypes.c_int64),
        ("LimitFlags", _DWORD),
        # ULONG_PTR-sized, so c_size_t rather than _DWORD -- getting one of
        # these wrong shifts every field after it, and LimitFlags would then
        # be read from the wrong offset.
        ("MinimumWorkingSetSize", ctypes.c_size_t),
        ("MaximumWorkingSetSize", ctypes.c_size_t),
        ("ActiveProcessLimit", _DWORD),
        ("Affinity", ctypes.c_size_t),
        ("PriorityClass", _DWORD),
        ("SchedulingClass", _DWORD),
    ]


class _ExtendedLimits(ctypes.Structure):
    _fields_ = [
        ("BasicLimitInformation", _BasicLimits),
        ("IoInfo", _IoCounters),
        ("ProcessMemoryLimit", ctypes.c_size_t),
        ("JobMemoryLimit", ctypes.c_size_t),
        ("PeakProcessMemoryUsed", ctypes.c_size_t),
        ("PeakJobMemoryUsed", ctypes.c_size_t),
    ]


@contextlib.contextmanager
def _kill_children_with_me():
    """Yield a job object handle that kills the trainer if we die.

    On Windows, terminating a process does not touch its children. The TUI
    calls ``terminate()`` on quit and Task Manager does the same, either of
    which would leave a trainer running unattended -- still holding the
    several gigabytes of DX12 memory that only process exit returns, and
    now competing with the replacement this supervisor is about to launch.
    That is the precise failure this command exists to prevent, so it must
    not be able to cause it.

    A job object with ``JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE`` makes the
    kernel handle it: the handle closes when this process ends, however it
    ends, and every process in the job goes with it.

    Yields ``None`` where that is unavailable -- a non-Windows host, or a
    policy that refuses -- in which case children run unmanaged and a
    ``terminate()`` of this process can orphan one.
    """
    if sys.platform != "win32":
        yield None
        return

    job = None
    kernel32 = _kernel32()
    try:
        job = kernel32.CreateJobObjectW(None, None)
        if not job:
            raise ctypes.WinError(ctypes.get_last_error())

        limits = _ExtendedLimits()
        limits.BasicLimitInformation.LimitFlags = (
            _JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE
        )
        if not kernel32.SetInformationJobObject(
            job,
            _JobObjectExtendedLimitInformation,
            ctypes.byref(limits),
            ctypes.sizeof(limits),
        ):
            raise ctypes.WinError(ctypes.get_last_error())
    except Exception:  # noqa: BLE001
        logger.warning(
            "Could not create a job object; a terminated supervisor may "
            "leave the trainer running and holding device memory",
            exc_info=True,
        )
        if job:
            kernel32.CloseHandle(job)
        yield None
        return

    try:
        yield job
    finally:
        kernel32.CloseHandle(job)


def _kernel32():
    """kernel32 with the signatures this module uses declared.

    The declarations are not optional: ctypes defaults every return value to
    ``int``, which truncates the 64-bit handles both creators here return.
    """
    kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
    kernel32.CreateJobObjectW.restype = _HANDLE
    kernel32.CreateJobObjectW.argtypes = [ctypes.c_void_p, ctypes.c_wchar_p]
    kernel32.SetInformationJobObject.argtypes = [
        _HANDLE,
        ctypes.c_int,
        ctypes.c_void_p,
        _DWORD,
    ]
    kernel32.AssignProcessToJobObject.argtypes = [_HANDLE, _HANDLE]
    kernel32.OpenProcess.restype = _HANDLE
    kernel32.OpenProcess.argtypes = [_DWORD, ctypes.c_int, _DWORD]
    kernel32.CloseHandle.argtypes = [_HANDLE]
    return kernel32


def _adopt(job, pid: int) -> None:
    """Put a started child into the job. Best effort by design."""
    if job is None:
        return
    kernel32 = _kernel32()
    handle = kernel32.OpenProcess(
        _PROCESS_SET_QUOTA | _PROCESS_TERMINATE, False, pid
    )
    if not handle:
        logger.warning("Could not open the trainer process to adopt it")
        return
    try:
        if not kernel32.AssignProcessToJobObject(job, handle):
            # Racing a child that has already exited is the ordinary reason,
            # and it needs no adopting.
            logger.warning(
                "Could not assign the trainer to the job object (error %d)",
                ctypes.get_last_error(),
            )
    finally:
        kernel32.CloseHandle(handle)


def _run_daemon(command: list[str], job) -> int:
    """Run one daemon to completion, inheriting our streams. Returns its code."""
    process = subprocess.Popen(command)
    # After the fact, so there is a window in which a child could spawn a
    # grandchild outside the job. The daemon spawns none, and the
    # alternative -- CREATE_SUSPENDED plus ResumeThread -- needs a thread
    # handle subprocess does not expose.
    _adopt(job, process.pid)
    try:
        return process.wait()
    except BaseException:
        # Ctrl-C, or anything else unwinding through here: do not leave the
        # trainer holding the device while we exit.
        with contextlib.suppress(Exception):
            process.terminate()
            process.wait(timeout=30)
        raise


# --- the loop --------------------------------------------------------------


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(
        level=logging.INFO,
        format=(
            "%(levelname).1s%(asctime)s.%(msecs)03d %(name)s "
            "%(filename)s:%(lineno)d] %(message)s"
        ),
        datefmt="%m%d %H:%M:%S",
        stream=sys.stderr,  # stdout carries the daemon's protocol.
    )

    # Everything after a bare `--` goes to the daemon, the same idiom
    # lc0-directml-tui uses.
    argv = list(sys.argv[1:] if argv is None else argv)
    passthrough: list[str] = []
    if "--" in argv:
        index = argv.index("--")
        argv, passthrough = argv[:index], argv[index + 1 :]
    args = _build_parser().parse_args(argv)

    clashes = [
        flag
        for flag in _OWNED_FLAGS
        if any(
            item == flag or item.startswith(flag + "=") for item in passthrough
        )
    ]
    if clashes:
        logger.error(
            "%s belong to the supervisor, not the daemon; pass them before "
            "the `--`",
            ", ".join(clashes),
        )
        return 2

    from lczero_training.directml import checkpoint as checkpoint_io

    directory = _checkpoint_directory(args.config, args.checkpoint)
    step = checkpoint_io.latest_step(directory)
    if step is None:
        logger.error(
            "No checkpoint in %s; run lc0-directml-init first.", directory
        )
        return 1

    stalls = 0
    launches = 0
    with _kill_children_with_me() as job:
        while step < args.target_step:
            # Each launch aims at its own nearer target, so the daemon
            # exits cleanly -- checkpointing and exporting on the way out --
            # instead of being killed at an arbitrary step.
            child_target = args.target_step
            if args.restart_every > 0:
                child_target = min(child_target, step + args.restart_every)

            command = [
                sys.executable,
                "-m",
                DAEMON_MODULE,
                "--config",
                args.config,
                f"--target-step={child_target}",
                *passthrough,
            ]
            if args.checkpoint:
                command += ["--checkpoint", args.checkpoint]

            if args.max_launches and launches >= args.max_launches:
                logger.info(
                    "Stopping at the --max-launches cap of %d with the "
                    "checkpoint at step %d of %d; rerun to carry on",
                    args.max_launches,
                    step,
                    args.target_step,
                )
                return 0

            launches += 1
            logger.info(
                "Launch %d: resuming at step %d, running to %d of %d",
                launches,
                step,
                child_target,
                args.target_step,
            )
            try:
                code = _run_daemon(command, job)
            except KeyboardInterrupt:
                logger.info("Interrupted; the trainer has been stopped")
                return 130

            reached = checkpoint_io.latest_step(directory)
            reached = step if reached is None else max(reached, step)
            logger.info(
                "Launch %d exited with code %d; the checkpoint advanced "
                "%d step(s) to %d",
                launches,
                code,
                reached - step,
                reached,
            )

            if reached > step:
                stalls = 0
                step = reached
                continue

            # No progress. Either the run cannot start at all or it is
            # dying before its first checkpoint, and relaunching into the
            # same wall forever helps nobody.
            stalls += 1
            if stalls >= args.max_stalls:
                logger.error(
                    "%d consecutive launches made no progress; stopping at "
                    "step %d. The last daemon exited with code %d -- see the "
                    "log above for why.",
                    stalls,
                    step,
                    code,
                )
                return 1
            logger.warning(
                "No progress on launch %d; retrying in %.0fs (%d of %d)",
                launches,
                args.backoff,
                stalls,
                args.max_stalls,
            )
            try:
                time.sleep(args.backoff)
            except KeyboardInterrupt:
                return 130

    logger.info(
        "Target step %d reached after %d launch(es); the checkpoint is at %d",
        args.target_step,
        launches,
        step,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
