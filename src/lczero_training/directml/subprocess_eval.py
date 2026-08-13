"""Held-out evaluation in a subprocess, so it cannot leak into training.

Evaluation used to run inside the training process: 50 forward passes on the
training device every ``--eval-every`` steps. Under DirectML that is 50
passes' worth of blocks the trainer can never get back -- nothing in the
process can hand them to the allocator, and the OS reclaims a DX12 context
only on process exit. Eval was therefore buying a slow, permanent tax on the
run for a number that is only read on a chart.

So the forward passes move to a process that exits. What crosses the
boundary is deliberately small and host-side only:

* the config the trainer is actually using, written out rather than re-read
  from disk, so an edit mid-run cannot change what is evaluated;
* the weights, as host copies -- the same transfer the scheduled checkpoint
  already makes every ``steps_per_network`` steps;
* the batches, pulled from the loader's ``test`` output as **numpy**. This is
  the other half of the win: the trainer no longer moves a single test batch
  to the device, where the old ``batches_from_loader(loader, device, "test")``
  did it 50 times per eval.

The worker writes the TensorBoard ``-test`` run itself and leaves the scalars
in a JSON file for the trainer to relay to its own log or TUI.

What this does *not* remove is the test data pipeline. It stays resident in
the trainer, because it is the thing producing the batches, and its cost is
host memory in the C++ loader rather than device blocks that strand a heap.

The worker runs on the **CPU** by default. That is not a fallback: a worker
building a second DX12 context beside a trainer holding 3.8 GB of the same
shared memory failed to finish importing inside 900 seconds, while the same
50 batches take 15 seconds on the CPU -- 5 of them being ``import torch``.
Fifty forward passes at batch 32 do not need a GPU, and wanting one here
means contending for the exact resource the run is short of.

A failed or hung worker is logged and skipped. Evaluation is observational,
and no measurement is worth ending a training run over.

The timeout has genuinely fired in production -- 16 for 16 automatic evals
in one real run, always at exactly ``timeout`` seconds, the worker's own log
never touched. What did NOT reproduce it, each tried as a controlled,
faithful concurrent-with-live-training test: --device cpu with the real
work-dir files (standalone); the same, spawned while the real trainer was
actively stepping; spawned through the actual supervisor -> daemon process
chain (matching production's exact subprocess.Popen pattern, not a bare
console); and spawned with --gc-every and --eval-every forced to the same
step so a full gc.collect() lands immediately before the spawn, since
production's defaults (500 and any multiple of it, e.g. 5000) make that
coincidence happen on literally every automatic eval. All four succeeded in
10-15 seconds. So this is real, but its trigger needs something none of
those four hours of testing reproduced -- plausibly many hours of live
uptime, which a short test cannot compress. Rather than ship an unproven
theory, this module now does two honest things: makes the next real
occurrence diagnosable (the worker's PID and a live forensic snapshot are
logged before anything is killed), and adds one bounded retry, cheap
insurance against whatever fraction of the cause turns out to be transient.
"""

from __future__ import annotations

import hashlib
import json
import logging
import pathlib
import subprocess
import sys
import tempfile
from collections.abc import Callable, Iterator

import numpy as np

logger = logging.getLogger(__name__)

WORKER_MODULE = "lczero_training.commands.directml_eval_worker"

CONFIG_FILE = "config.textproto"
WEIGHTS_FILE = "weights.pt"
BATCHES_FILE = "batches.npz"
RESULT_FILE = "scalars.json"
# The worker's own log, which it writes before it can hang. Read back here on
# failure, because a killed worker's stderr may never have reached us.
WORKER_LOG_FILE = "worker.log"

# The trainer is paused for every second of this, so it is a ceiling on the
# damage one stuck eval can do rather than a generous allowance. Measured: 50
# batches on the CPU take 15 s end to end, imports included.
DEFAULT_TIMEOUT_SECONDS = 300.0

# One retry beyond the first attempt. Cheap insurance, not a fix: if the real
# trigger is a transient condition (a momentary I/O or driver stall), this
# rides through it for the cost of one more `timeout` on the rare step where
# it happens; if the trigger is a persistent per-process condition, this
# retry fails the same way and costs one extra `timeout` for nothing. Either
# way the trainer is never blocked longer than (1 + DEFAULT_RETRIES) *
# timeout on any one eval, and it was never blocking training itself either
# way -- a skipped eval just means one fewer point on the chart.
DEFAULT_RETRIES = 1

# Below this much free physical memory the subprocess can no longer start its
# python interpreter next to a live trainer. Every hung eval in the 225k-300k
# step window of one real run was logged against ``available 2.11 GB of
# 11.65``, while the subprocess never reached its first log line -- the
# interpreter was parked at startup (cpu ~0), not slowly working. A second
# ``import torch`` of its own needs commit headroom that is simply not there
# once the trainer has booked its 5+ GB. There is no good second chance
# through that wall; a spawn would just timeout-and-kill, which is what
# happened twenty-two times in a row before this guard existed. Skipping
# up-front with a clear log beats a 300 s park followed by a kill, and the
# eval retry lands at the next ``--eval-every`` step by which point the
# machine may have recovered (e.g. after a supervisor restart freed the
# trainer's committed memory).
MIN_AVAILABLE_GB = 2.5


def work_directory(config_filepath: str) -> pathlib.Path:
    """A stable scratch directory, reused and overwritten on every eval.

    Stable rather than ``mkdtemp``: each request holds ~50 MB of batches, and
    a fresh directory per eval would accumulate one every ``--eval-every``
    steps for the life of the machine. Keyed by config path so two runs do
    not overwrite each other's requests.
    """
    digest = hashlib.sha256(config_filepath.encode("utf-8")).hexdigest()[:12]
    return pathlib.Path(tempfile.gettempdir()) / f"lc0-directml-eval-{digest}"


def write_batches(loader, alias: str, count: int, path: pathlib.Path) -> int:
    """Pull ``count`` batches off the loader and save them. Returns the count.

    Host-side throughout -- the arrays come off the loader as numpy and are
    never touched by torch here, which is the point.
    """
    inputs: list[np.ndarray] = []
    probabilities: list[np.ndarray] = []
    values: list[np.ndarray] = []
    for _ in range(count):
        arrays = loader.get_next(alias)
        if len(arrays) != 3:
            raise ValueError(
                f"expected 3 arrays from the loader, got {len(arrays)}"
            )
        inputs.append(np.ascontiguousarray(arrays[0]))
        probabilities.append(np.ascontiguousarray(arrays[1]))
        values.append(np.ascontiguousarray(arrays[2]))

    # Uncompressed: this is ~50 MB written and read back immediately, and
    # deflating it costs more time than the write saves.
    np.savez(
        path,
        inputs=np.stack(inputs),
        probabilities=np.stack(probabilities),
        values=np.stack(values),
    )
    return len(inputs)


def read_batches(path: pathlib.Path) -> tuple[Iterator[tuple], int]:
    """The worker's side of ``write_batches``: (batch iterator, count)."""
    stored = np.load(path)
    inputs = stored["inputs"]
    probabilities = stored["probabilities"]
    values = stored["values"]
    count = len(inputs)

    def batches() -> Iterator[tuple]:
        for index in range(count):
            yield inputs[index], probabilities[index], values[index]

    return batches(), count


def _diagnose_hang(process: subprocess.Popen) -> str:
    """A live forensic snapshot of a worker that has not finished in time.

    Called before the process is killed, so this is the only chance to see
    what it was actually doing. ``cpu_times()`` near zero is the most
    useful bit: a worker parked waiting on something (a lock, a driver
    call, a handle) burns no CPU while it waits, where one that is merely
    slow keeps accumulating it. Best-effort: psutil may be unavailable, and
    the process may finish or vanish between the timeout firing and this
    running.
    """
    try:
        import psutil

        info = psutil.Process(process.pid)
        cpu = info.cpu_times()
        busy = cpu.user + cpu.system
        return (
            f"PID {process.pid}: status={info.status()} "
            f"threads={info.num_threads()} cpu={busy:.2f}s -- "
            + (
                "parked (no CPU consumed; waiting on a lock, a handle, or "
                "the OS, not merely slow)"
                if busy < 0.05
                else "was consuming CPU, not stuck idle"
            )
        )
    except Exception:  # noqa: BLE001 - forensics must not raise
        return (
            f"PID {process.pid}: could not inspect (psutil unavailable, or "
            "the process has already exited)"
        )


def make_eval_hook(
    *,
    config,
    model,
    loader,
    config_filepath: str,
    batch_count: int,
    device_spec: str | None = None,
    kda_chunk_size: int | None = None,
    timeout: float = DEFAULT_TIMEOUT_SECONDS,
    retries: int = DEFAULT_RETRIES,
    on_scalars: Callable[[int, dict[str, float]], None] | None = None,
) -> Callable[[int], None]:
    """An eval callback that runs the measurement in a child process.

    ``on_scalars`` receives the results for logging or the TUI; the
    TensorBoard ``-test`` run is written by the worker, not from here, so no
    writer for it stays open in the training process. ``retries`` is extra
    attempts after the first failure -- see ``DEFAULT_RETRIES`` for why one
    is worth the cost and why it is not a fix.
    """
    from google.protobuf import text_format

    directory = work_directory(config_filepath)
    directory.mkdir(parents=True, exist_ok=True)
    config_path = directory / CONFIG_FILE
    weights_path = directory / WEIGHTS_FILE
    batches_path = directory / BATCHES_FILE
    result_path = directory / RESULT_FILE

    # Written once. The sections the worker reads -- model, losses, metrics --
    # do not change during a run; the data loader's does (the phase farm
    # rewrites it), and the worker never looks at it.
    config_path.write_text(text_format.MessageToString(config))
    logger.info("Evaluation runs in a subprocess; requests go to %s", directory)

    command = [
        sys.executable,
        "-m",
        WORKER_MODULE,
        "--work-dir",
        str(directory),
        "--log-file",
        str(directory / WORKER_LOG_FILE),
    ]
    if device_spec:
        command += ["--device", device_spec]
    if kda_chunk_size:
        command += [f"--kda-chunk-size={kda_chunk_size}"]

    def spawn_and_wait(step: int) -> dict[str, float]:
        """One attempt: spawn, wait up to ``timeout``, return the scalars.

        Popen rather than ``subprocess.run(timeout=...)``: run() gives no
        access to the child until it is already finished or being killed,
        so a hung worker left no PID, no status, nothing to inspect --
        which is exactly the gap that made the earlier stalls unexplainable.
        Logging the PID the moment it exists is cheap and was missing.
        """
        process = subprocess.Popen(
            command + [f"--step={step}"],
            # The daemon's stdout carries the TUI protocol and this process
            # inherits it. The worker logs to stderr; anything it writes to
            # stdout would corrupt that stream, so stdout goes nowhere.
            stdout=subprocess.DEVNULL,
        )
        logger.info(
            "Eval worker for step %d spawned as PID %d", step, process.pid
        )
        try:
            returncode = process.wait(timeout=timeout)
        except subprocess.TimeoutExpired:
            # Before killing it: this is the only moment its state can still
            # be read. Once terminated there is nothing left to inspect.
            logger.error(
                "Eval worker PID %d (step %d) exceeded %.0fs: %s",
                process.pid,
                step,
                timeout,
                _diagnose_hang(process),
            )
            process.kill()
            process.wait()
            raise
        if returncode != 0:
            raise RuntimeError(
                f"the eval worker (PID {process.pid}) exited with code "
                f"{returncode}"
            )
        return json.loads(result_path.read_text())

    def prepare(step: int) -> None:
        """Write the weights and batches for ``step``. Run once per hook
        call, not once per attempt: a retry reuses this request rather than
        re-paying the weight copy and the batch pull for what is, if a
        retry helps at all, presumably unrelated to the data.
        """
        import torch

        # Never let a failed worker hand back the previous eval's numbers.
        result_path.unlink(missing_ok=True)
        # The worker's own log file, the same: if its python interpreter
        # fails to start -- the real failure mode when system memory is
        # gone -- it never reaches its own ``FileHandler(mode="w")`` and
        # the file is left holding the previous successful eval's lines.
        # ``worker_tail`` would then report those as "how far this worker
        # got", which is exactly the lie every post-mortem here has been
        # told. Truncated here, mirroring what the worker's own
        # ``FileHandler(mode="w")`` would have done, so a hang in
        # interpreter startup yields an honest "the worker's log is empty;
        # it died before its first stage" rather than stale content two
        # days old.
        (directory / WORKER_LOG_FILE).write_text("", encoding="utf-8")

        torch.save(
            {
                "step": step,
                "model_state": {
                    name: tensor.detach().cpu()
                    for name, tensor in model.state_dict().items()
                },
            },
            weights_path,
        )
        stored = write_batches(loader, "test", batch_count, batches_path)
        logger.info(
            "Evaluating step %d in a subprocess on %d batch(es)", step, stored
        )

    def worker_tail(lines: int = 8) -> str:
        """The last of the worker's own log, for a failure message.

        The worker may die without reaching our stderr at all -- that is how
        the first stall left no evidence anywhere -- so its log on disk is
        what says which stage it was in.
        """
        try:
            recorded = (
                (directory / WORKER_LOG_FILE)
                .read_text(encoding="utf-8", errors="replace")
                .splitlines()
            )
        except OSError:
            return "the worker left no log"
        if not recorded:
            return "the worker's log is empty; it died before its first stage"
        return "\n".join(recorded[-lines:])

    def hook(step: int) -> None:
        try:
            prepare(step)
        except Exception:  # noqa: BLE001 - a measurement cannot end a run
            logger.exception(
                "Could not prepare the eval request at step %d; training "
                "continues",
                step,
            )
            return

        # Check system memory *before* spawning. The hung-eval failure mode
        # here is the worker's python interpreter never reaching its first
        # log line because ``import torch`` next to a live trainer cannot
        # get commit headroom once most of the machine is already booked. A
        # spawn in that state parks for the full ``timeout`` and is killed
        # by it; this guard turns that into an explicit skip with a clear
        # line in the log. Per-call, not per-attempt: retries happen seconds
        # apart, and memory in that window is whatever it is -- there is no
        # second chance here for the spawn loop to find.
        from . import host_memory

        available = host_memory.available_gb()
        if (
            available is not None
            and available < MIN_AVAILABLE_GB
        ):
            logger.warning(
                "Skipping evaluation at step %d: only %.2f GB of physical "
                "memory is free (need at least %.2f GB to start a second "
                "python interpreter that imports torch alongside the "
                "trainer). Eval will retry at the next --eval-every step. "
                "Memory: %s.",
                step,
                available,
                MIN_AVAILABLE_GB,
                host_memory.snapshot(),
            )
            return

        attempts = 1 + max(retries, 0)
        scalars: dict[str, float] | None = None
        for attempt in range(1, attempts + 1):
            label = (
                f"attempt {attempt}/{attempts}" if attempts > 1 else "attempt"
            )
            try:
                scalars = spawn_and_wait(step)
                break
            except subprocess.TimeoutExpired:
                logger.error(
                    "Evaluation at step %d timed out after %.0fs (%s). The "
                    "worker got this far:\n%s",
                    step,
                    timeout,
                    label,
                    worker_tail(),
                )
            except Exception:  # noqa: BLE001 - a measurement cannot end a run
                logger.exception(
                    "Evaluation at step %d failed (%s). The worker got this "
                    "far:\n%s",
                    step,
                    label,
                    worker_tail(),
                )

        if scalars is None:
            if attempts > 1:
                logger.error(
                    "Evaluation at step %d did not succeed in %d attempt(s); "
                    "training continues",
                    step,
                    attempts,
                )
            return
        if scalars and on_scalars is not None:
            on_scalars(step, scalars)

    return hook
