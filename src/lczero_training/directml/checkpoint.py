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
from typing import Any, TYPE_CHECKING

from proto import model_config_pb2
from proto.root_config_pb2 import RootConfig
from proto.training_config_pb2 import WeightsSelector

if TYPE_CHECKING:
    import torch

logger = logging.getLogger(__name__)

CHECKPOINT_VERSION = 1
_FILENAME = "checkpoint-{step:09d}.pt"
_FILENAME_RE = re.compile(r"^checkpoint-(\d+)\.pt$")


class OptimizerStateMismatchError(RuntimeError):
    """The saved optimizer state cannot be paired back to the model by name.

    A distinct type so a caller can specifically catch and recover from
    this one -- by dropping the saved moments and resuming fresh, the
    error message's own suggested recovery -- without also swallowing the
    other, more concerning failures load_optimizer_state_dict_into can
    raise (a genuinely unnamed optimizer parameter, which is an integrity
    bug worth stopping on, not a checkpoint predating a fix).

    The known, expected cause: a checkpoint saved with
    training.make_checkpoint's pre-fix ``model_state`` builder, which
    called ``.cpu()`` on each state_dict tensor independently and so lost
    the storage identity that ties a shared parameter's two names
    together -- see ``_state_dict_to_host``. Every checkpoint saved before
    that fix looks, on reload, like it has a few more independent
    parameters than the saved optimizer state does, and this is what that
    looks like from here.
    """


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


class NonFiniteWeightsError(RuntimeError):
    """Raised rather than write a checkpoint whose weights are not finite."""


def first_non_finite(state: dict[str, Any]) -> str | None:
    """Name of the first tensor holding a NaN or an infinity, or None."""
    import torch

    for name, tensor in state.items():
        if not torch.is_tensor(tensor) or not tensor.is_floating_point():
            continue
        if not torch.isfinite(tensor).all():
            return name
    return None


def save(
    directory: str | os.PathLike,
    checkpoint: Checkpoint,
    *,
    max_to_keep: int = 0,
    require_finite: bool = True,
) -> pathlib.Path:
    """Write a checkpoint atomically. Returns the path written.

    Refuses a checkpoint whose weights are not finite. A real run diverged
    and then wrote six NaN checkpoints over the following three hours; with
    ``max_to_keep`` rotating the directory, each one deleted an older good
    checkpoint, and the last clean weights came within four writes of being
    destroyed. Nothing downstream can recover from a NaN checkpoint, so
    writing one is never the right outcome -- refusing costs a stopped run
    and saves the only weights worth having.

    Checked here rather than at the call sites because every path that
    persists weights goes through this function, including the emergency
    save on a crash, which is exactly when the weights are least trustworthy.
    """
    import torch

    if require_finite:
        poisoned = first_non_finite(checkpoint.model_state)
        if poisoned is not None:
            raise NonFiniteWeightsError(
                f"refusing to write a checkpoint at step {checkpoint.step}: "
                f"{poisoned} is not finite. The last good checkpoint in "
                f"{directory} is left untouched."
            )

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


def load_state_dict_into(model: "torch.nn.Module", state_dict: dict) -> None:
    """Load ``state_dict`` into ``model`` in place.

    Tolerates tensors the checkpoint has but this model no longer defines --
    e.g. a ``policy_head`` removed from the config -- but never silently
    accepts a checkpoint that is missing a tensor the model needs. The default
    ``strict=True`` does neither; ``strict=False`` does both, so this splits
    the difference and checks ``missing_keys`` explicitly. Failing to load a
    genuinely missing key is a hard error; warning away an unused tensor is
    safe.

    ``torch`` is not imported here: every caller already has it (loading a
    checkpoint means having a model, which is built from the same import).
    """
    incompatible = model.load_state_dict(state_dict, strict=False)
    if incompatible.missing_keys:
        raise RuntimeError(
            "Checkpoint is missing tensors the model needs: "
            f"{incompatible.missing_keys}"
        )
    if incompatible.unexpected_keys:
        logger.warning(
            "Ignoring %d checkpoint tensor(s) this model no longer "
            "defines (e.g. a policy_head removed from the config); "
            "first few: %s",
            len(incompatible.unexpected_keys),
            incompatible.unexpected_keys[:5],
        )


def load_optimizer_state_dict_into(
    optimizer: "torch.optim.Optimizer",
    saved_state: dict,
    model: "torch.nn.Module",
    saved_model_state: dict,
    decay_selector: WeightsSelector,
) -> None:
    """Load ``saved_state`` into ``optimizer``, filtered to parameters that
    still exist in ``model``.

    Pairs the saved state to the model's current parameters by **name**,
    because :meth:`torch.optim.Optimizer.load_state_dict` is strict -- it
    raises if the parameter count changes -- where :meth:`Module.load_state_dict`
    has the ``strict`` knob. A head removed from the config (e.g.
    ``optimistic_st``) leaves stale entries in the saved optimizer state;
    removing it also shifts the integer indices of every parameter defined
    after it, so a naive index-truncation would write the value-head's
    moment into the moves-left head's slot.

    Pair-by-name needs the saved model_state plus the **decay selector**
    that built the saved optimizer, so the (flat-index -> name) map can
    be reconstructed: ``Optimizer.state_dict`` keys the per-parameter
    moments by an opaque integer position that shifts whenever a
    parameter is added or removed between saves, so identity across
    saves survives only through the name, not the position. The flat-
    index enumeration follows ``build_optimizer``'s order: iterate
    ``named_parameters`` (deduplicating shared parameters), split into
    decayed-then-plain by the selector, and number the result 0..N-1.

    Shared parameters (the ``policy_embedding_shared`` Linear, which
    every policy head's ``tokens`` attribute aliases) are deduplicated
    by storage identity (``tensor.data_ptr()``), not by ``id(tensor)``:
    ``id`` is process-local and does not survive ``torch.save``/load, but
    storage sharing does, so two names pointing to the same underlying
    tensor at save time still agree on ``data_ptr`` after the round
    trip. ``id``-based dedup would over-count by the number of policy
    heads -- which is exactly the failure the first cut of this helper
    crashed with ("the saved model has 172 float parameter(s), but the
    saved optimizer state lists 169").

    The state entries themselves are a *sparse* subset of the full
    cross-group parameter index range: PyTorch's ``Optimizer.state_dict``
    emits a per-parameter state entry only when that parameter has
    actually been stepped (params with ``grad=None`` on every step so far
    have no entry).

    A parameter present in the saved state but not in the new model is
    dropped (with its moment buffers freed); a parameter present in the new
    model but not in the saved state is left at zero moments, exactly like
    the no-moments emergency-save path; a parameter whose shape has changed
    between saves is also dropped for that slot.
    """
    if not saved_state:
        return

    # Reconstruct the saved (flat-index -> name) map by replaying
    # ``build_optimizer``'s grouping on the saved model_state. iterate in
    # ``state_dict`` insertion order (= module-traversal order, what
    # ``named_parameters`` returns), keep float tensors only (the only
    # integer-typed tensors in this codebase are the policy_map /
    # policy_map_inverse buffers, which the optimizer never sees), dedup
    # shared parameters by ``data_ptr`` (see the docstring), and bucket
    # into decayed vs plain by the selector. The flat-index enumeration
    # build_optimizer produced was ``decayed_then_plain`` in that order,
    # which is what the saved optimizer's state dict's integer keys
    # reference.
    from .optimizer import selector_includes

    seen_ptrs: set[int] = set()
    saved_decayed: list[str] = []
    saved_plain: list[str] = []
    for name, tensor in saved_model_state.items():
        if not tensor.dtype.is_floating_point:
            continue
        # Empty tensors have data_ptr() == 0; fall back to id() for those
        # since dedup by storage is meaningless without storage. They are
        # rare (only uninitialized shape placeholders).
        ptr = tensor.data_ptr()
        key = ptr if ptr != 0 else id(tensor)
        if key in seen_ptrs:
            continue
        seen_ptrs.add(key)
        target = (
            saved_decayed
            if selector_includes(decay_selector, name)
            else saved_plain
        )
        target.append(name)
    saved_index_to_name = dict(enumerate(saved_decayed + saved_plain))

    # Sanity: the flat-index count the saved optimizer declares must
    # match the dedup'd-bucketed count we just rebuilt. If it does not,
    # the selector disagrees with the one that built the saved
    # optimizer, or the saved checkpoint is internally inconsistent;
    # either way by-name pairing would silently mis-pair, so stop.
    saved_flat_indices: list[int] = []
    for group in saved_state["param_groups"]:
        saved_flat_indices.extend(group["params"])
    if len(saved_flat_indices) != len(saved_index_to_name):
        raise OptimizerStateMismatchError(
            "Cannot pair saved optimizer state by name: the rebuilt "
            f"(flat-index -> name) map has {len(saved_index_to_name)} "
            f"entr(y/ies) but the saved optimizer declares "
            f"{len(saved_flat_indices)} parameter index(ices). The decay "
            "selector passed to load_optimizer_state_dict_into probably "
            "differs from the one that built the saved checkpoint, or "
            "this checkpoint predates training._state_dict_to_host's "
            "shared-parameter fix. "
            "Re-train from scratch, or drop the optimizer state and "
            "resume fresh moments -- the no-moments path fast-forwards "
            "via set_step on the optimizer side."
        )

    # Reconstruct the new optimizer's per-group parameter names in its own
    # iteration order. optimizer.param_groups[i]['params'] is a list of
    # tensors with no names; pair each tensor back to a name via
    # model.named_parameters(), which the optimizer construction itself
    # used (deduplicating by id).
    name_to_id: dict[str, int] = {}
    seen_ids: set[int] = set()
    for name, param in model.named_parameters():
        if id(param) in seen_ids:
            continue
        seen_ids.add(id(param))
        name_to_id[name] = id(param)

    new_groups: list[list[str]] = []
    for group in optimizer.param_groups:
        group_names: list[str] = []
        for param in group["params"]:
            for n, pid in name_to_id.items():
                if pid == id(param):
                    group_names.append(n)
                    break
            else:
                raise RuntimeError(
                    "An optimizer parameter has no name in "
                    "model.named_parameters(); cannot rebuild the new "
                    "optimizer's param order needed to filter the saved "
                    "state."
                )
        new_groups.append(group_names)

    # Build the filtered state: state keyed by new optimizer indices,
    # param_groups with the new sizes. Carries over per-tensor moments only
    # where the name survives AND the shape matches; otherwise leaves the
    # optimizer slot at its construction-time zero moments.
    new_state: dict = {"state": {}, "param_groups": []}
    dropped_shape_mismatch = 0
    started_fresh_count = 0
    no_saved_moments_count = 0
    for g_idx, group in enumerate(saved_state["param_groups"]):
        new_group = {k: v for k, v in group.items() if k != "params"}
        new_params_indices: list[int] = []
        for new_idx, name in enumerate(new_groups[g_idx]):
            saved_idx = next(
                (k for k, n in saved_index_to_name.items() if n == name),
                None,
            )
            if saved_idx is None:
                # Present in the new model but not the saved one: a
                # freshly-added parameter starts with zero moments. Still
                # happens in the "added a head" case, just not the case
                # we are currently diagnosing.
                started_fresh_count += 1
                new_params_indices.append(new_idx)
                continue
            # The saved state is a *sparse* subset of the flat-index range
            # (see the docstring): a parameter never stepped -- grad=None
            # on every step so far, e.g. a head the loss never touched --
            # has a valid name and index but no entry in ``state`` at all.
            # That is not the mismatch this function is guarding against;
            # it starts fresh exactly like a parameter absent from the
            # saved model, so treat it the same way instead of a raw
            # KeyError.
            saved_slot = saved_state["state"].get(saved_idx)
            if saved_slot is None:
                no_saved_moments_count += 1
                new_params_indices.append(new_idx)
                continue
            # Shape check: a changed-but-present parameter shape (e.g. a
            # resized head) is the same world as a removed one -- the saved
            # moment is the wrong shape, so drop it and let the new param
            # start fresh rather than slice or transpose silently.
            new_param = optimizer.param_groups[g_idx]["params"][new_idx]
            if "mu" in saved_slot and saved_slot["mu"].shape != new_param.shape:
                dropped_shape_mismatch += 1
                new_params_indices.append(new_idx)
                continue
            new_state["state"][new_idx] = saved_slot
            new_params_indices.append(new_idx)
        new_group["params"] = new_params_indices
        new_state["param_groups"].append(new_group)

    # Saved entries that did not land on a new parameter: removed heads or
    # any structural removal. Counted by position, since saved_index_to_name
    # carries the entire saved population and we do not loop over its
    # contents above.
    kept_count = len(new_state["state"])
    removed_count = len(saved_state["state"]) - kept_count

    optimizer.load_state_dict(new_state)
    if (
        removed_count
        or started_fresh_count
        or dropped_shape_mismatch
        or no_saved_moments_count
    ):
        logger.warning(
            "Loaded optimizer state filtered to surviving parameters: "
            "removed %d saved moment(s) whose parameter is gone "
            "(e.g. optimistic_st), %d new parameter(s) starting fresh, "
            "%d shape-changed parameter(s), %d parameter(s) present in "
            "both but never stepped in the saved run (grad was always "
            "None there). Their moment buffers start at zero; the rest "
            "of the schedule is preserved. Run with "
            "--ignore-config-mismatch for this launch only; the next "
            "saved checkpoint's optimizer state will be self-consistent "
            "again.",
            removed_count,
            started_fresh_count,
            dropped_shape_mismatch,
            no_saved_moments_count,
        )
