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

A failed or hung worker is logged and skipped. Evaluation is observational,
and no measurement is worth ending a training run over.
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

# Generous: a worker pushed onto the CPU with --eval-device cpu is minutes
# rather than seconds, and killing a slow eval costs the measurement.
DEFAULT_TIMEOUT_SECONDS = 900.0


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
    on_scalars: Callable[[int, dict[str, float]], None] | None = None,
) -> Callable[[int], None]:
    """An eval callback that runs the measurement in a child process.

    ``on_scalars`` receives the results for logging or the TUI; the
    TensorBoard ``-test`` run is written by the worker, not from here, so no
    writer for it stays open in the training process.
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
    ]
    if device_spec:
        command += ["--device", device_spec]
    if kda_chunk_size:
        command += [f"--kda-chunk-size={kda_chunk_size}"]

    def run(step: int) -> dict[str, float]:
        import torch

        # Never let a failed worker hand back the previous eval's numbers.
        result_path.unlink(missing_ok=True)

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

        completed = subprocess.run(
            command + [f"--step={step}"],
            # The daemon's stdout carries the TUI protocol and this process
            # inherits it. The worker logs to stderr; anything it writes to
            # stdout would corrupt that stream, so stdout goes nowhere.
            stdout=subprocess.DEVNULL,
            timeout=timeout,
            check=False,
        )
        if completed.returncode != 0:
            raise RuntimeError(
                f"the eval worker exited with code {completed.returncode}"
            )
        return json.loads(result_path.read_text())

    def hook(step: int) -> None:
        try:
            scalars = run(step)
        except subprocess.TimeoutExpired:
            logger.error(
                "Evaluation at step %d timed out after %.0fs and was killed; "
                "training continues",
                step,
                timeout,
            )
            return
        except Exception:  # noqa: BLE001 - a measurement cannot end a run
            logger.exception(
                "Evaluation at step %d failed; training continues", step
            )
            return
        if scalars and on_scalars is not None:
            on_scalars(step, scalars)

    return hook
