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
from . import gpu_memory, host_memory
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


def _clip_grad_norm_preserving_origin(
    model: LczeroModel, max_norm: float
) -> torch.Tensor:
    """clip_grad_norm_, but only mutates gradients when the norm is finite.

    torch.nn.utils.clip_grad_norm_ always applies its clip_coef, even when
    total_norm came back non-finite: clip_coef = max_norm / (nan + 1e-6) is
    itself nan, and multiplying every gradient by nan overwrites every one
    of them -- including whichever ones were still fine. describe_non_finite
    was called after this ran, so it only ever found nan grad_norm
    everywhere and nothing else: the actual parameter whose backward first
    produced a non-finite value was already gone by the time anything
    looked. This computes the same total_norm the same way, but leaves the
    gradients untouched when it is not finite, so the one that was actually
    bad is still there to name.

    Harmless on the step this changes anything about: a non-finite
    total_norm already means the update gets discarded by the caller,
    either raised on ("report"/"step") or zeroed by zero_grad()
    ("skip") -- clipping grads that are about to be thrown away was always
    a no-op on the training dynamics, just not on what could be read back
    out of them afterward.
    """
    grads = [p.grad for p in model.parameters() if p.grad is not None]
    if not grads:
        return torch.zeros((), device=next(model.parameters()).device)
    device = grads[0].device
    total_norm = torch.norm(
        torch.stack([torch.norm(g.detach(), 2).to(device) for g in grads]), 2
    )
    if bool(torch.isfinite(total_norm)):
        clip_coef = torch.clamp(max_norm / (total_norm + 1e-6), max=1.0)
        for g in grads:
            g.detach().mul_(clip_coef)
    return total_norm


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
    nan_check: str = "report",
    max_skips: int = 20,
) -> int:
    """Run `steps` optimizer steps. Returns the resulting global step.

    ``reporters`` receive ``(step, scalars)`` every ``report_every`` steps --
    TensorBoard and the TUI daemon both hook in this way. Metrics are only
    pulled off the device on steps that actually report.

    ``nan_check`` governs what happens on a non-finite gradient, which the
    optimizer would otherwise turn into non-finite weights. ``"report"``
    (default) checks on the reporting cadence and adds no sync of its own,
    stopping the run; ``"step"`` checks every step for +7% and stops on the
    exact one; ``"skip"`` also checks every step but drops the bad gradient
    and continues, giving up only after ``max_skips`` skips; ``"off"``
    disables it. ``"skip"`` is for an unattended run through a rough patch;
    ``"report"`` is right when a diverged run should stop and be looked at.

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
    skipped_steps = 0
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

        metrics: dict[str, float] = {}
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
                # Losses are converted to Python floats immediately to avoid
                # device tensor allocation churn in the micro-batch loop.
                for name, value in micro_metrics.items():
                    val_float = float(value)
                    metrics[name] = metrics.get(name, 0.0) + val_float

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
                metrics[name] /= accumulation

        if (should_log or should_report) and diagnostics:
            # Once per step, on the last micro-batch's outputs -- not once
            # per micro-batch. These are observational: parameter norms do
            # not vary within a step at all, and an accuracy measured on
            # one micro-batch is the same estimate the run reported before
            # accumulation existed. Computing them 8x to average them cost
            # 8x the kernels and 8x the allocation churn.
            diag_metrics = _diagnostic_metrics(
                predictions, batch, model, loss_fn
            )
            for name, value in diag_metrics.items():
                metrics[name] = float(value)
            del diag_metrics
        del predictions

        if max_grad_norm and max_grad_norm > 0:
            grad_norm = _clip_grad_norm_preserving_origin(model, max_grad_norm)
        else:
            grad_norm = torch.sqrt(
                sum(
                    p.grad.square().sum()
                    for p in model.parameters()
                    if p.grad is not None
                )
            )

        # The last point at which the weights are still good. A non-finite
        # gradient here becomes a non-finite parameter the instant
        # optimizer.step() runs, and there is no recovering from that: a real
        # run went from a healthy step to every one of 153 tensors being NaN
        # inside a single 250-step window, then trained on for three hours
        # producing nothing.
        #
        # Clipping does not help. clip_grad_norm_ scales by
        # max_norm/total_norm, and with a NaN total_norm every gradient comes
        # out NaN -- max_grad_norm was set to 16.0 throughout that run.
        #
        # Checking costs a device-to-host sync, which the rest of this loop
        # is careful to avoid: measured at +37 ms on a 518 ms step, +7.2%.
        # So "report" is the default and piggybacks the sync the reporting
        # path already performs -- the weights can then be poisoned for up
        # to report_every steps, which costs nothing that matters, because
        # checkpoint_io.save refuses to persist them and a diverged run is
        # rolled back to the last checkpoint either way. "step" pays the 7%
        # to catch the exact step, which is what to use when hunting one.
        # "skip" also checks every step, but rather than stopping it drops
        # the bad gradient and carries on: a rare exploding batch then costs
        # one skipped update instead of a crash-and-restart loop. This model
        # hit reliable data-triggered spikes in one step region that a plain
        # restart could not get past, because every relaunch walked back into
        # the same batches; skipping rides through them.
        apply_update = True
        if nan_check in ("step", "skip") or (
            nan_check == "report" and (should_log or should_report)
        ):
            if not bool(torch.isfinite(grad_norm)):
                if nan_check == "skip":
                    apply_update = False
                    skipped_steps += 1
                    # Describe *before* clearing: zero_grad(set_to_none=True)
                    # below drops every param.grad to None, and
                    # describe_non_finite read through those same
                    # attributes. Called after, it could only ever report
                    # "no non-finite gradients" -- there were no gradients
                    # left to inspect, non-finite or otherwise. That is
                    # exactly what every skip in production logged, and it
                    # is why the actual gradient this was trying to name was
                    # never once identified.
                    detail = describe_non_finite(model)
                    optimizer.zero_grad(set_to_none=True)
                    logger.warning(
                        "Non-finite gradient at step %d (grad_norm=%s); "
                        "skipping this update and continuing (%d skipped "
                        "so far). %s",
                        step,
                        float(grad_norm),
                        skipped_steps,
                        detail,
                    )
                    if skipped_steps > max_skips > 0:
                        # A handful of skips is a bad batch; a flood is a
                        # diverged run that skipping cannot rescue, and
                        # looping forever hides that. Stop and let the guard
                        # roll back to the last good checkpoint.
                        raise NonFiniteGradientError(
                            step,
                            f"{skipped_steps} skipped updates exceeds "
                            f"--max-skips={max_skips}; this is divergence, "
                            f"not a bad batch. {detail}",
                        )
                else:
                    raise NonFiniteGradientError(
                        step, describe_non_finite(model)
                    )

        if apply_update:
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
            # Charted alongside the losses on purpose. Whether a run died
            # because a pool filled up or because the allocator stranded its
            # heaps is the question every OOM here has turned on, and a trace
            # that climbs steadily answers it at a glance.
            #
            # Both pools, because they are not the same and confusing them
            # cost a night: system RAM is what psutil sees, and gpu_committed
            # is what DirectML actually allocates from. A run has died with
            # 6 GB of the first free and 98% of the second gone.
            available = host_memory.available_gb()
            if available is not None:
                scalars["mem_available_gb"] = available
            committed = gpu_memory.adapter_committed_mb()
            if committed is not None:
                scalars["gpu_committed_mb"] = committed

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
                if name
                not in (
                    "grad_norm",
                    "lr",
                    "ms_per_step",
                    "mem_available_gb",
                    "gpu_committed_mb",
                )
            )
            logger.info(
                "step %d  %s  grad_norm=%.4f  lr=%.3g  mem[%s]  [%s]",
                step,
                detail,
                float(grad_norm),
                learning_rate,
                # On the log line rather than in the metrics: this describes
                # the machine, not the model, and it is the number every
                # out-of-memory post-mortem here has had to guess at.
                host_memory.snapshot(),
                gpu_memory.snapshot(),
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

        # Every step, not on the logging cadence. log_every is a quarter of a
        # checkpoint interval -- 250 steps -- and the launches this exists to
        # explain died at steps 7, 9 and 27, so a warning gated on the log
        # line would never have fired once. One psutil counter and one warm
        # PDH query, ~0.05 ms together, against a ~900 ms step.
        host_memory.warn_if_low(f"step {step}")
        gpu_memory.warn_if_low(f"step {step}")

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
    if skipped_steps:
        logger.warning(
            "Skipped %d non-finite update(s) this segment; the run rode "
            "through them rather than diverging",
            skipped_steps,
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


class NonFiniteGradientError(RuntimeError):
    """A gradient went NaN or infinite before it could reach the weights."""

    def __init__(self, step: int, detail: str):
        self.step = step
        self.detail = detail
        super().__init__(
            f"non-finite gradient at step {step}, stopping before the "
            f"optimizer applies it. {detail}"
        )


def describe_non_finite(model: LczeroModel, limit: int = 6) -> str:
    """Which parameters and gradients are bad, for the failure message.

    Only ever called once, on the way out, so the cost does not matter --
    and this is the evidence that was missing when a run diverged twice
    with nothing in the log but `total=nan` a quarter of an hour later.
    Distinguishing bad *gradients* over good *weights* from both being bad
    says whether this step caused it or an earlier one did.

    Also ranks parameters by grad L2 norm, computed in float64 on the CPU.
    This is the check that actually explains a non-finite ``grad_norm``
    when every per-element ``isfinite`` check below comes back clean:
    clip_grad_norm_ reduces each parameter's norm and then combines those
    norms in float32, and a gradient with large-but-finite elements
    (nowhere near float32's ~3.4e38 ceiling individually) can still
    square-and-sum past it during that reduction. float64 has ~1.8e308 of
    headroom, so the same computation done here does not re-overflow the
    very thing it is trying to measure, and the ranking names which
    parameter's gradient is actually huge instead of leaving every check
    reporting "clean". On the CPU because DirectML has no float64 kernels
    at all -- doing this on-device raises "The GPU device does not support
    Double (Float64) operations!" instead of returning an answer.
    """
    bad_params: list[str] = []
    bad_grads: list[str] = []
    grad_norms: list[tuple[float, str]] = []
    with torch.no_grad():
        for name, param in model.named_parameters():
            if not torch.isfinite(param).all():
                bad_params.append(name)
            if param.grad is None:
                continue
            if not torch.isfinite(param.grad).all():
                bad_grads.append(name)
            grad_norms.append((param.grad.cpu().double().norm().item(), name))

    def summarise(names: list[str], what: str) -> str:
        if not names:
            return f"no non-finite {what}"
        shown = ", ".join(names[:limit])
        more = f" (+{len(names) - limit} more)" if len(names) > limit else ""
        return f"{len(names)} non-finite {what}: {shown}{more}"

    grad_norms.sort(reverse=True)
    largest = ", ".join(
        f"{name}={norm:.3g}" for norm, name in grad_norms[:limit]
    )

    weights_ok = not bad_params
    return (
        f"{summarise(bad_grads, 'gradients')}; "
        f"{summarise(bad_params, 'weights')}."
        + (
            " The weights are still good, so this step is the origin."
            if weights_ok
            else " The weights are already bad, so the origin is earlier."
        )
        + f" Largest grad norms (float64): {largest}."
    )


def describe_error(error: BaseException) -> str:
    """A description that names the failure even when it carries no message.

    ``KeyboardInterrupt`` stringifies to the empty string, so a deliberate
    Ctrl-C logged ``Training stopped at step 197835:`` and stopped there --
    indistinguishable from a crash that lost its message, in a log where
    telling those apart is the whole job. The exception type is never empty.
    """
    text = str(error).strip()
    name = type(error).__name__
    return f"{name}: {text}" if text else name


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
    # Same argument as the gradients: the optimizer's scratch buffers are
    # ~48 MB of device memory that no checkpoint wants, and dropping them
    # costs no allocation. Rebuilt on the next step, which for this caller
    # never comes.
    if optimizer is not None and hasattr(optimizer, "free_scratch"):
        optimizer.free_scratch()
    gc.collect()

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

    gc.collect()
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
