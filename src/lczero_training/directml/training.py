"""Native Windows training loop for the DirectML port.

Phase 8 of docs/directml_training_port.md. Pulls batches straight from the
natively-built C++ data loader -- no WSL, no localhost protocol.
"""

from __future__ import annotations

import dataclasses
import gc
import logging
import time
from collections.abc import Callable, Iterator, Sequence

import numpy as np
import torch

from proto.root_config_pb2 import RootConfig

from . import checkpoint as checkpoint_io
from . import derived_metrics
from .losses import LczeroLoss, materialize_metrics
from .lr_schedule import make_lr_schedule
from .metrics import Reporter
from .model import LczeroModel
from .optimizer import NAdamW, build_optimizer

logger = logging.getLogger(__name__)


@dataclasses.dataclass
class TrainingBatch:
    """One batch on the training device."""

    inputs: torch.Tensor  # (batch, 112, 8, 8)
    probabilities: torch.Tensor  # (batch, 1858)
    values: torch.Tensor  # (batch, 6, 3)

    @classmethod
    def from_arrays(
        cls, arrays: tuple[np.ndarray, ...], device: torch.device
    ) -> TrainingBatch:
        if len(arrays) != 3:
            raise ValueError(
                f"expected 3 arrays from the loader, got {len(arrays)}"
            )
        inputs, probabilities, values = (
            torch.from_numpy(np.ascontiguousarray(array)).to(device)
            for array in arrays
        )
        return cls(inputs=inputs, probabilities=probabilities, values=values)


def constant_learning_rate(schedules) -> float:
    """The learning rate at step 0.

    Kept for callers that only need a starting value; the training loop
    re-evaluates the full schedule on every step.
    """
    return make_lr_schedule(list(schedules))(0)


@torch.no_grad()
def evaluate(
    *,
    model: LczeroModel,
    loss_fn: LczeroLoss,
    batches: Iterator[TrainingBatch],
    batch_count: int,
) -> dict[str, float]:
    """Average the losses and diagnostics over `batch_count` held-out batches.

    No backward pass and no optimizer step -- this only measures. The model
    has no dropout or batchnorm, so eval() changes nothing about the
    computation; it is set anyway so that stays true if either is added.
    """
    was_training = model.training
    model.eval()
    totals: dict[str, float] = {}
    seen = 0
    try:
        for _ in range(batch_count):
            batch = next(batches)
            predictions = model(batch.inputs)
            _, metrics = loss_fn(predictions, batch, model)
            metrics.update(
                _diagnostic_metrics(predictions, batch, model, loss_fn)
            )
            for name, value in materialize_metrics(metrics).items():
                totals[name] = totals.get(name, 0.0) + value
            seen += 1
    finally:
        if was_training:
            model.train()

    if not seen:
        return {}
    return {name: value / seen for name, value in totals.items()}


@torch.no_grad()
def _diagnostic_metrics(
    predictions, batch: TrainingBatch, model: LczeroModel, loss_fn: LczeroLoss
) -> dict[str, torch.Tensor]:
    """The TF pipeline's observational metrics, for the configured heads.

    Computed against the same heads the losses use, so the numbers line up
    with what is actually being optimized.
    """
    metrics: dict[str, torch.Tensor] = {}

    for policy_loss in loss_fn.policy_losses:
        logits = predictions.policy.get(policy_loss.head_name)
        if logits is None:
            continue
        metrics.update(
            derived_metrics.policy_metrics(batch.probabilities, logits)
        )
        # The TF tags are unprefixed, so only the first policy head can own
        # them; further heads would collide.
        break

    for value_loss in loss_fn.value_losses:
        prediction = predictions.value.get(value_loss.head_name)
        if prediction is None:
            continue
        metrics.update(
            derived_metrics.value_metrics(
                prediction[0],
                batch.values[:, value_loss.value_type, 0],
                batch.values[:, value_loss.value_type, 1],
            )
        )
        break

    metrics.update(derived_metrics.kda_metrics(model))
    metrics.update(derived_metrics.parameter_metrics(model))
    return metrics


def batches_from_loader(
    loader, device: torch.device, alias: str = ""
) -> Iterator[TrainingBatch]:
    """Endless batches from one of the loader's named outputs.

    ``alias`` selects between the pipeline's exposed outputs -- "" for a
    single-output config, or e.g. "train"/"test" when the config splits.
    """
    while True:
        yield TrainingBatch.from_arrays(loader.get_next(alias), device)


def training_segments(
    start_step: int, target_step: int, checkpoint_interval: int
) -> Iterator[int]:
    """Yield run lengths that checkpoint up to an absolute target step."""
    current_step = start_step
    while current_step < target_step:
        segment_steps = min(checkpoint_interval, target_step - current_step)
        yield segment_steps
        current_step += segment_steps


def train(
    *,
    config: RootConfig,
    model: LczeroModel,
    optimizer: NAdamW,
    batches: Iterator[TrainingBatch],
    device: torch.device,
    start_step: int,
    steps: int,
    log_every: int = 1,
    progress: dict[str, int] | None = None,
    reporters: Sequence[Reporter] = (),
    report_every: int = 10,
    diagnostics: bool = True,
    eval_hook: "Callable[[int], None] | None" = None,
    eval_every: int = 0,
    gc_every: int = 0,
) -> int:
    """Run `steps` optimizer steps. Returns the resulting global step.

    ``reporters`` receive ``(step, scalars)`` every ``report_every`` steps --
    TensorBoard and the TUI daemon both hook in this way. Metrics are only
    pulled off the device on steps that actually report.

    ``eval_hook`` fires on global steps divisible by ``eval_every``, so the
    cadence is the same whether this is called once for 20,000 steps or
    twenty times for 1,000. A run shorter than ``eval_every`` that straddles
    no multiple of it therefore evaluates not at all, which is what "every N
    steps" means.
    """
    loss_fn = LczeroLoss(config.training.losses)
    max_grad_norm = config.training.max_grad_norm
    accumulation = max(config.training.gradient_accumulation_steps, 1)
    schedule = make_lr_schedule(list(config.training.lr_schedule))
    model.train()
    entered = time.perf_counter()
    if accumulation > 1:
        logger.info(
            "Accumulating %d micro-batches per optimizer step", accumulation
        )

    step = start_step
    started: float | None = None
    warmup_seconds = 0.0
    for index in range(steps):
        # Decided before the forward pass, since the KDA stats are captured
        # during it. Anything that reports needs them; the other steps run
        # the mixer untouched.
        last = index == steps - 1
        should_log = bool(log_every) and (index % log_every == 0 or last)
        should_report = bool(reporters) and (
            index % max(report_every, 1) == 0 or last
        )
        # Re-evaluated every step: the schedule may ramp or cycle, and the
        # rate for a resumed run depends on the restored global step.
        learning_rate = schedule(step)
        for group in optimizer.param_groups:
            group["lr"] = learning_rate
        optimizer.zero_grad(set_to_none=True)

        metrics: dict[str, torch.Tensor] = {}
        for micro in range(accumulation):
            final_micro = micro == accumulation - 1
            # Only the last micro-batch carries the diagnostics, so the KDA
            # mixers capture stats on that pass alone. Enabling it for all
            # of them would run the extra reductions `accumulation` times
            # to produce a number that gets thrown away.
            if diagnostics:
                derived_metrics.set_kda_stats_collection(
                    model, final_micro and (should_log or should_report)
                )

            batch = next(batches)
            if started is None:
                # Start timing only once the first batch is in hand. The
                # loader spends minutes indexing chunks and filling its
                # shuffle pool before it yields anything, and folding that
                # into ms/step makes training look several times slower
                # than it is.
                warmup_seconds = time.perf_counter() - entered
                started = time.perf_counter()
                logger.info(
                    "First batch after %.1fs of data loader startup",
                    warmup_seconds,
                )
            predictions = model(batch.inputs)
            loss, micro_metrics = loss_fn(predictions, batch, model)
            # Divided so the accumulated gradient is the mean over the
            # effective batch, not its sum -- otherwise the update would
            # scale with `accumulation` and the configured learning rate
            # would mean something different at every setting.
            (loss / accumulation).backward()

            if should_log or should_report:
                # Losses are summed across every micro-batch and divided
                # below, so the reported value describes the whole
                # effective batch rather than whichever one was last.
                for name, value in micro_metrics.items():
                    prior = metrics.get(name)
                    metrics[name] = value if prior is None else prior + value

            # Drop the activations before the next forward allocates. The
            # graph itself is freed by backward, but the output tensors are
            # not, and under DirectML the allocator does not reliably reuse
            # a block that is still referenced. Only the last micro-batch's
            # outputs are kept, for the diagnostics below.
            del loss, micro_metrics
            if not final_micro:
                del predictions

        if metrics and accumulation > 1:
            for name in metrics:
                metrics[name] = metrics[name] / accumulation

        if (should_log or should_report) and diagnostics:
            # Once per step, on the last micro-batch's outputs -- not once
            # per micro-batch. These are observational: parameter norms do
            # not vary within a step at all, and an accuracy measured on
            # one micro-batch is the same estimate the run reported before
            # accumulation existed. Computing them 8x to average them cost
            # 8x the kernels and 8x the allocation churn.
            metrics.update(
                _diagnostic_metrics(predictions, batch, model, loss_fn)
            )
        del predictions

        if max_grad_norm and max_grad_norm > 0:
            grad_norm = torch.nn.utils.clip_grad_norm_(
                model.parameters(), max_grad_norm
            )
        else:
            grad_norm = torch.sqrt(
                sum(
                    p.grad.square().sum()
                    for p in model.parameters()
                    if p.grad is not None
                )
            )

        optimizer.step()
        step += 1
        if progress is not None:
            # Published every step so a caller that catches an exception
            # knows exactly how many steps actually completed.
            progress["step"] = step

        if should_log or should_report:
            # The single GPU-to-CPU sync per reporting step. `metrics` was
            # already averaged over the micro-batches above.
            scalars = materialize_metrics(metrics)
            scalars["grad_norm"] = float(grad_norm)
            scalars["lr"] = learning_rate
            elapsed_so_far = time.perf_counter() - started
            scalars["ms_per_step"] = elapsed_so_far / (index + 1) * 1000.0

            for reporter in reporters if should_report else ():
                try:
                    reporter(step, scalars)
                except Exception:  # noqa: BLE001 - never kill a run for a metric
                    logger.exception("Reporter failed at step %d", step)

        if should_log:
            # From `scalars`, not `metrics`: those are still device tensors,
            # and formatting them would sync a second time.
            detail = "  ".join(
                f"{name}={value:.4f}"
                for name, value in sorted(scalars.items())
                if name not in ("grad_norm", "lr", "ms_per_step")
            )
            logger.info(
                "step %d  %s  grad_norm=%.4f  lr=%.3g",
                step,
                detail,
                float(grad_norm),
                learning_rate,
            )

        # Everything the step still owns, dropped before the next forward
        # allocates. `predictions` went above; these are the rest. Without
        # this they stay referenced through the following step's forward
        # and backward, so the run carries one extra step's tensors at all
        # times -- and on DirectML a referenced block is one the allocator
        # will not reuse.
        del metrics, grad_norm, batch
        if should_log or should_report:
            del scalars

        if gc_every and step % gc_every == 0:
            # Reference counting frees a tensor the moment its last Python
            # reference dies, so this is only for what cycles hold: autograd
            # nodes and any exception context that outlived its frame.
            # Python's own collector runs on allocation-count heuristics
            # that a training loop can starve for a long time.
            #
            # This does NOT flush a device cache. torch_directml exposes no
            # empty_cache, no memory_allocated, and no synchronize (checked
            # against 0.2.5.dev240914), and torch._C._host_emptyCache does
            # not exist -- so nothing in this process can hand blocks back
            # to the DirectML allocator. All this can do is make sure Python
            # is not the one holding them.
            collected = gc.collect()
            if collected:
                logger.debug(
                    "gc at step %d freed %d object(s)", step, collected
                )

        # Evaluation runs after the step, so the reported test metrics
        # correspond to the weights the matching train metrics describe.
        #
        # On the global step, not `index`. This used to read
        # `index % eval_every == 0 or last`, and `index` restarts at 0 in
        # every call -- which is once per checkpoint interval. With
        # steps_per_network at 1,000, any --eval-every above that evaluated
        # at the first and last step of every segment: `--eval-every 5000`
        # meant an eval every ~500 steps, ten times more often than asked,
        # each one a permanent slice of the trainer's device memory.
        if eval_hook is not None and eval_every > 0 and step % eval_every == 0:
            eval_hook(step)

    elapsed = time.perf_counter() - started if started is not None else 0.0
    if steps:
        logger.info(
            "%d steps in %.1fs (%.0f ms/step, %d micro-batches/step), "
            "plus %.1fs loader startup",
            steps,
            elapsed,
            elapsed / steps * 1000.0,
            accumulation,
            warmup_seconds,
        )
    return step


def build_model_and_optimizer(
    config: RootConfig, device: torch.device
) -> tuple[LczeroModel, NAdamW]:
    model = LczeroModel(config.model).to(device)
    optimizer = build_optimizer(
        model.named_parameters(),
        config.training.optimizer,
        learning_rate=constant_learning_rate(config.training.lr_schedule),
    )
    return model, optimizer


def release_to_host(model: LczeroModel, optimizer: NAdamW | None = None) -> int:
    """Move every device tensor to the host one at a time. Returns the count.

    For the emergency checkpoint path: after this the model's parameters
    live on the host and it can no longer run a step on the device, so only
    call it on a run that is ending.

    ``make_checkpoint`` builds the whole host-side state dict in one
    comprehension, which needs room for every parameter at once -- about
    72 MB with the moments -- on a device that has just refused an
    allocation. That is why the recovery checkpoint failed every time.
    Copying one tensor at a time needs room for one tensor, and because
    each copy drops its device original as it goes, the pressure only ever
    falls: the transfer pays for itself after the first parameter.

    Gradients go first, and the order is load-bearing twice over. They are
    the same size as the parameters, no checkpoint wants them, and
    discarding them needs no allocation at all, so they are the cheapest
    relief available before the copying starts -- and a parameter left
    holding a device gradient after its own data moves to the host is a
    device mismatch waiting for the next thing that touches the pair.
    """
    model.zero_grad(set_to_none=True)

    moved = 0
    # keep_vars=True yields the Parameters and buffers themselves rather
    # than detached copies, so assigning .data rebinds the storage the
    # module actually holds. Setting .data (not the whole entry) is also
    # what keeps each Parameter's identity, which the optimizer's state is
    # keyed on -- rebuilding them would orphan the moments below.
    for tensor in model.state_dict(keep_vars=True).values():
        # A tied parameter appears under several names; the second visit
        # already sees a host tensor.
        if tensor.device.type == "cpu":
            continue
        tensor.data = tensor.data.detach().to("cpu")
        moved += 1

    if optimizer is not None:
        for state in optimizer.state.values():
            for key, value in list(state.items()):
                if not isinstance(value, torch.Tensor):
                    continue
                if value.device.type == "cpu":
                    continue
                state[key] = value.detach().to("cpu")
                moved += 1

    logger.info("Released %d tensor(s) from the device to the host", moved)
    return moved


def make_checkpoint(
    config: RootConfig,
    model: LczeroModel,
    optimizer: NAdamW | None,
    step: int,
) -> checkpoint_io.Checkpoint:
    return checkpoint_io.Checkpoint(
        step=step,
        # Always store on the host: a checkpoint written from DirectML
        # tensors would not reload on a machine without that adapter.
        model_state={
            name: tensor.detach().cpu()
            for name, tensor in model.state_dict().items()
        },
        optimizer_state=optimizer.state_dict() if optimizer else None,
        config_digest=checkpoint_io.config_digest(config),
        rng_state=torch.get_rng_state(),
    )
