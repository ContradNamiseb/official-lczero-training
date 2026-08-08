"""Native PyTorch checkpoints for the DirectML port.

Phase 8 of docs/directml_training_port.md. Deliberately does not use Orbax:
the Windows environment has no need for it, and a plain ``torch.save`` of a
dict is enough for a single-process trainer.

Checkpoints are written to a temporary file and renamed, so an interrupted
save cannot leave a half-written checkpoint that the next run would try to
resume from.

``torch`` is imported inside ``save`` and ``load_latest`` rather than at the
top. The rest of this module -- the filename convention, ``latest_step``,
``config_digest`` -- is pure filesystem and protobuf work, and the restart
supervisor polls ``latest_step`` while a trainer is running on a machine
with 4-5 GB free. Importing torch there would cost a few hundred megabytes
of the headroom the trainer needs.
"""

from __future__ import annotations

import dataclasses
import hashlib
import logging
import os
import pathlib
import re
from typing import Any

from proto import model_config_pb2
from proto.root_config_pb2 import RootConfig

logger = logging.getLogger(__name__)

CHECKPOINT_VERSION = 1
_FILENAME = "checkpoint-{step:09d}.pt"
_FILENAME_RE = re.compile(r"^checkpoint-(\d+)\.pt$")


def config_digest(config: RootConfig) -> str:
    """Stable hash of the parts of the config a checkpoint depends on.

    Only the model and optimizer settings are hashed. Data loader paths,
    step counts, and tensorboard directories are expected to change between
    runs and must not invalidate a checkpoint.

    ``KdaConfig.chunk_size`` is deliberately excluded. It lives inside the
    model config, but the chunked recurrence is exact for any chunk size --
    it is a per-backend speed tunable, and a checkpoint has to stay loadable
    when it changes (DirectML wants 8, the default is 16).
    """
    model = model_config_pb2.ModelConfig()
    model.CopyFrom(config.model)
    model.encoder.kda.ClearField("chunk_size")

    digest = hashlib.sha256()
    digest.update(model.SerializeToString(deterministic=True))
    digest.update(
        config.training.optimizer.SerializeToString(deterministic=True)
    )
    return digest.hexdigest()


@dataclasses.dataclass
class Checkpoint:
    step: int
    model_state: dict[str, Any]
    optimizer_state: dict[str, Any] | None
    config_digest: str
    rng_state: Any
    version: int = CHECKPOINT_VERSION


def _checkpoint_files(
    directory: pathlib.Path,
) -> list[tuple[int, pathlib.Path]]:
    if not directory.is_dir():
        return []
    found = []
    for entry in directory.iterdir():
        match = _FILENAME_RE.match(entry.name)
        if match:
            found.append((int(match.group(1)), entry))
    return sorted(found)


def save(
    directory: str | os.PathLike,
    checkpoint: Checkpoint,
    *,
    max_to_keep: int = 0,
) -> pathlib.Path:
    """Write a checkpoint atomically. Returns the path written."""
    import torch

    path = pathlib.Path(directory)
    path.mkdir(parents=True, exist_ok=True)
    destination = path / _FILENAME.format(step=checkpoint.step)
    temporary = destination.with_suffix(".pt.tmp")

    payload = {
        "version": checkpoint.version,
        "step": checkpoint.step,
        "model_state": checkpoint.model_state,
        "optimizer_state": checkpoint.optimizer_state,
        "config_digest": checkpoint.config_digest,
        "rng_state": checkpoint.rng_state,
    }
    torch.save(payload, temporary)
    # os.replace is atomic on Windows and POSIX alike.
    os.replace(temporary, destination)
    logger.info("Wrote checkpoint %s", destination)

    if max_to_keep and max_to_keep > 0:
        existing = _checkpoint_files(path)
        for _, stale in existing[:-max_to_keep]:
            logger.info("Removing old checkpoint %s", stale)
            stale.unlink(missing_ok=True)
    return destination


def latest_step(directory: str | os.PathLike) -> int | None:
    files = _checkpoint_files(pathlib.Path(directory))
    return files[-1][0] if files else None


def load_latest(
    directory: str | os.PathLike,
    *,
    expected_digest: str | None = None,
    ignore_config_mismatch: bool = False,
) -> Checkpoint | None:
    """Load the highest-step checkpoint, or None if there are none."""
    import torch

    files = _checkpoint_files(pathlib.Path(directory))
    if not files:
        return None
    step, path = files[-1]
    logger.info("Restoring checkpoint %s", path)
    payload = torch.load(path, map_location="cpu", weights_only=False)

    version = payload.get("version")
    if version != CHECKPOINT_VERSION:
        raise ValueError(
            f"{path} is checkpoint version {version}; this build writes "
            f"version {CHECKPOINT_VERSION}"
        )

    digest = payload.get("config_digest")
    if expected_digest is not None and digest != expected_digest:
        message = (
            f"{path} was written from a different model/optimizer "
            f"configuration (checkpoint {digest[:16]}, config "
            f"{expected_digest[:16]})"
        )
        if not ignore_config_mismatch:
            raise ValueError(message)
        logger.warning("%s (ignored)", message)

    return Checkpoint(
        step=payload["step"],
        model_state=payload["model_state"],
        optimizer_state=payload.get("optimizer_state"),
        config_digest=digest,
        rng_state=payload.get("rng_state"),
        version=version,
    )
