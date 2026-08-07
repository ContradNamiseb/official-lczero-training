"""Training losses for the DirectML port.

Phase 7 of docs/directml_training_port.md. Ports the losses the real-import
configuration actually uses -- `vanilla` policy cross-entropy with
illegal-move masking, `winner` WDL value cross-entropy, and `main`
moves-left -- from model/loss_function.py.

Note on the moves-left loss: the plan calls it "mean-squared error", but
the JAX reference uses a *Huber* loss (delta 10/20 on values scaled by 20).
This follows the reference, since matching the existing trainer matters more
than matching the prose.

Everything here is computed per-sample and then averaged over the batch,
which is what the JAX side gets by vmapping a per-sample loss.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from proto.training_config_pb2 import (
    LossConfig,
    MovesLeftLossConfig,
    PolicyLossConfig,
    RegularizationLossConfig,
    ValueCategoricalLossConfig,
    ValueErrorLossConfig,
    ValueLossConfig,
)

from .model import ModelPrediction

if TYPE_CHECKING:
    # Annotation-only: training.py imports this module, so importing it
    # back at runtime would be circular.
    from .training import TrainingBatch

# The moves-left target and prediction are divided by this before the Huber
# loss, putting the term in the same range as the others.
MOVESLEFT_SCALE = 20.0
MOVESLEFT_HUBER_DELTA = 10.0 / MOVESLEFT_SCALE

# Floor for log() inside the KL term; the surrounding torch.where already
# excludes zero targets, but the clamp keeps the *gradient* finite too.
_TINY = 1e-30


def _q_from_wdl(wdl_logits: torch.Tensor) -> torch.Tensor:
    """Expected score from WDL logits: P(win) - P(loss)."""
    probabilities = torch.softmax(wdl_logits, dim=-1)
    return probabilities[..., 0] - probabilities[..., 2]


def _huber(
    prediction: torch.Tensor, target: torch.Tensor, delta: float
) -> torch.Tensor:
    """Huber loss, composed from primitives.

    Deliberately not ``F.huber_loss``: DirectML has no ``aten::huber_loss``
    kernel, so that call silently copies the tensors to the CPU and back on
    every single training step.

    Matches optax.huber_loss: quadratic within delta, linear beyond it.
    """
    error = prediction - target
    absolute = error.abs()
    return torch.where(
        absolute <= delta,
        0.5 * error.square(),
        delta * (absolute - 0.5 * delta),
    )


def _softmax_cross_entropy(
    logits: torch.Tensor, labels: torch.Tensor
) -> torch.Tensor:
    """Cross-entropy against soft labels, summed over the last axis."""
    return -(labels * torch.log_softmax(logits, dim=-1)).sum(dim=-1)


def _safe_softmax_cross_entropy(
    logits: torch.Tensor, labels: torch.Tensor
) -> torch.Tensor:
    """As above, but tolerating -inf logits.

    Masked-out moves carry -inf, whose log_softmax contribution is
    ``0 * -inf = nan``. Every masked position has a zero label, so the term
    is mathematically zero; substitute it explicitly rather than letting the
    nan propagate (this is what optax.safe_softmax_cross_entropy does).
    """
    log_probs = torch.log_softmax(logits, dim=-1)
    contributions = labels * log_probs
    contributions = torch.where(
        labels > 0, contributions, torch.zeros_like(contributions)
    )
    return -contributions.sum(dim=-1)


class PolicyLoss:
    """Cross-entropy between predicted and searched move distributions."""

    def __init__(self, config: PolicyLossConfig):
        if config.type == PolicyLossConfig.LOSS_TYPE_UNSPECIFIED:
            raise ValueError(
                f"Policy loss type must be set for head '{config.head_name}'."
            )
        if config.type not in (
            PolicyLossConfig.CROSS_ENTROPY,
            PolicyLossConfig.KL,
        ):
            raise NotImplementedError(
                f"Unknown policy loss type {config.type}."
            )
        self.loss_type = config.type

        if config.HasField("optimistic"):
            optimistic = config.optimistic
            self.opt_value_head: str | None = optimistic.value_head_name
            self.opt_value_type = optimistic.value_type
            self.opt_strength = optimistic.strength
            self.opt_eps = optimistic.eps
            self.opt_alpha = optimistic.alpha
            self.opt_propagate = optimistic.propagate_value_gradients
        else:
            self.opt_value_head = None

        self.head_name = config.head_name
        self.metric_name = config.metric_name or config.head_name
        self.weight = config.weight
        self.mask_illegal = config.illegal_moves == PolicyLossConfig.MASK
        temperature = config.temperature
        self.temperature = temperature if temperature > 0 else 1.0

    def __call__(
        self, predictions: ModelPrediction, batch: TrainingBatch
    ) -> torch.Tensor:
        logits = predictions.policy[self.head_name]
        targets = batch.probabilities.to(logits.dtype)

        if self.mask_illegal:
            # Illegal moves are flagged with a negative target; -inf drives
            # their softmax probability to zero so no gradient reaches them.
            logits = torch.where(
                targets >= 0, logits, torch.full_like(logits, float("-inf"))
            )
        targets = torch.relu(targets)

        if self.temperature != 1.0:
            targets = targets.pow(1.0 / self.temperature)
            total = targets.sum(dim=-1, keepdim=True)
            targets = targets / torch.where(
                total > 0, total, torch.ones_like(total)
            )

        cross_entropy = _safe_softmax_cross_entropy(logits, targets)
        if self.loss_type == PolicyLossConfig.KL:
            # KL = CE - H(target). xlogy(0, 0) is 0, not nan.
            entropy_terms = torch.where(
                targets > 0,
                targets * torch.log(targets.clamp_min(_TINY)),
                torch.zeros_like(targets),
            )
            cross_entropy = cross_entropy + entropy_terms.sum(dim=-1)

        if self.opt_value_head is not None:
            weight = self._optimistic_weight(predictions, batch)
            cross_entropy = cross_entropy * weight
        return cross_entropy.mean()

    def _optimistic_weight(
        self, predictions: ModelPrediction, batch: TrainingBatch
    ) -> torch.Tensor:
        """Up-weight positions the value head is pessimistic about."""
        wdl_logits, error_pred, _ = predictions.value[self.opt_value_head]
        assert error_pred is not None, (
            "optimistic policy weighting needs a value head with an error "
            f"output; '{self.opt_value_head}' has none"
        )
        if not self.opt_propagate:
            wdl_logits = wdl_logits.detach()
            error_pred = error_pred.detach()

        predicted_q = _q_from_wdl(wdl_logits)
        target_q = batch.values[:, self.opt_value_type, 0]
        sigma = torch.sqrt(error_pred.squeeze(-1).clamp_min(0.0))
        z = (target_q - predicted_q) / (sigma + self.opt_eps)
        return torch.sigmoid((z - self.opt_strength) * self.opt_alpha)


class ValueLoss:
    """Cross-entropy between predicted and observed win/draw/loss."""

    def __init__(self, config: ValueLossConfig):
        self.head_name = config.head_name
        self.metric_name = config.metric_name or config.head_name
        self.weight = config.weight
        self.value_type = config.value_type

    def __call__(
        self, predictions: ModelPrediction, batch: TrainingBatch
    ) -> torch.Tensor:
        logits = predictions.value[self.head_name][0]
        q = batch.values[:, self.value_type, 0]
        d = batch.values[:, self.value_type, 1]
        # w = (1 + q - d) / 2, l = (1 - q - d) / 2.
        #
        # `one` is a tensor rather than the literal 1.0: on DirectML a
        # reverse subtraction against a Python scalar (`1.0 - q`) promotes
        # the scalar to a float64 tensor and dies with "The GPU device does
        # not support Double (Float64) operations!". Forward ops (`q + 1.0`)
        # are fine; only the reflected ones promote.
        one = torch.ones_like(q)
        wdl = torch.stack([(one + q - d) / 2.0, d, (one - q - d) / 2.0], dim=-1)
        return _softmax_cross_entropy(logits, wdl.detach()).mean()


class MovesLeftLoss:
    """Huber loss on the remaining-plies prediction."""

    def __init__(self, config: MovesLeftLossConfig):
        self.head_name = config.head_name
        self.metric_name = config.metric_name or config.head_name
        self.weight = config.weight
        self.value_type = config.value_type

    def __call__(
        self, predictions: ModelPrediction, batch: TrainingBatch
    ) -> torch.Tensor:
        prediction = predictions.movesleft[self.head_name].squeeze(-1)
        # values is [batch, 6, 3]; component 2 is the moves-left target.
        target = batch.values[:, self.value_type, 2] / MOVESLEFT_SCALE
        scaled = prediction / MOVESLEFT_SCALE
        return _huber(scaled, target, MOVESLEFT_HUBER_DELTA).mean()


class ValueErrorLoss:
    """Trains the error head to predict its own WDL head's squared error."""

    def __init__(self, config: ValueErrorLossConfig):
        self.head_name = config.head_name
        self.metric_name = config.metric_name or config.head_name
        self.weight = config.weight
        self.value_type = config.value_type
        self.propagate_value_gradients = config.propagate_value_gradients

    def __call__(
        self, predictions: ModelPrediction, batch: "TrainingBatch"
    ) -> torch.Tensor:
        wdl_logits, error_pred, _ = predictions.value[self.head_name]
        assert error_pred is not None, (
            f"value head '{self.head_name}' has no error output"
        )
        predicted_q = _q_from_wdl(wdl_logits)
        target_q = batch.values[:, self.value_type, 0]
        squared_error = (predicted_q - target_q).square()
        if not self.propagate_value_gradients:
            # The error head learns to track the WDL head, not to reshape it.
            squared_error = squared_error.detach()
        return (error_pred.squeeze(-1) - squared_error).square().mean()


class ValueCategoricalLoss:
    """Cross-entropy over Q discretized into buckets."""

    def __init__(self, config: ValueCategoricalLossConfig):
        self.head_name = config.head_name
        self.metric_name = config.metric_name or config.head_name
        self.weight = config.weight
        self.value_type = config.value_type

    def __call__(
        self, predictions: ModelPrediction, batch: "TrainingBatch"
    ) -> torch.Tensor:
        logits = predictions.value[self.head_name][2]
        assert logits is not None, (
            f"value head '{self.head_name}' has no categorical output"
        )
        target_q = batch.values[:, self.value_type, 0]
        buckets = logits.shape[-1]
        # Map [-1, 1) onto [0, buckets).
        index = torch.floor((target_q + 1.0) / 2.0 * buckets).long()
        index = index.clamp(0, buckets - 1)
        # cross_entropy with hard indices is the one-hot softmax CE the JAX
        # side builds explicitly.
        return torch.nn.functional.cross_entropy(
            logits, index.detach(), reduction="mean"
        )


class RegularizationLoss:
    """L2 over the parameters the selector includes."""

    def __init__(self, config: RegularizationLossConfig):
        self.metric_name = config.metric_name or "l2"
        self.weight = config.weight
        self.selector = config.selector

    def __call__(self, model: torch.nn.Module) -> torch.Tensor:
        from .optimizer import selector_includes

        total = None
        seen: set[int] = set()
        for name, parameter in model.named_parameters():
            if id(parameter) in seen:
                continue
            seen.add(id(parameter))
            if not selector_includes(self.selector, name):
                continue
            term = parameter.square().sum()
            total = term if total is None else total + term
        if total is None:
            # No parameter matched; a zero that still tracks the graph.
            return torch.zeros((), device=next(model.parameters()).device)
        return total


class LczeroLoss:
    """The configured losses, combined by weight."""

    def __init__(self, config: LossConfig):
        self.policy_losses = [PolicyLoss(c) for c in config.policy]
        self.value_losses = [ValueLoss(c) for c in config.value]
        self.movesleft_losses = [MovesLeftLoss(c) for c in config.movesleft]
        self.value_error_losses = [
            ValueErrorLoss(c) for c in config.value_error
        ]
        self.value_categorical_losses = [
            ValueCategoricalLoss(c) for c in config.value_categorical
        ]
        self.regularization_losses = [
            RegularizationLoss(c) for c in config.regularization
        ]

    def __call__(
        self,
        predictions: ModelPrediction,
        batch: TrainingBatch,
        model: torch.nn.Module | None = None,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        """Returns the weighted total and per-loss metrics.

        The metrics are **detached tensors, not floats**. Converting them
        here would force a GPU-to-CPU sync on every single step, even the
        ones nothing reports; the caller materializes them with
        :func:`materialize_metrics` only on steps it actually logs.

        ``model`` is needed only when a regularization loss is configured,
        since that one reads parameters rather than predictions.
        """
        total: torch.Tensor | None = None
        metrics: dict[str, torch.Tensor] = {}

        for group, prefix in (
            (self.policy_losses, "policy"),
            (self.value_losses, "value"),
            (self.movesleft_losses, "movesleft"),
            (self.value_error_losses, "value_error"),
            (self.value_categorical_losses, "value_categorical"),
        ):
            for loss_fn in group:
                value = loss_fn(predictions, batch)
                metrics[f"{prefix}/{loss_fn.metric_name}"] = value.detach()
                weighted = value * loss_fn.weight
                total = weighted if total is None else total + weighted

        for regularization in self.regularization_losses:
            if model is None:
                raise ValueError(
                    "a regularization loss is configured but no model was "
                    "passed to LczeroLoss.__call__"
                )
            value = regularization(model)
            metrics[f"regularization/{regularization.metric_name}"] = (
                value.detach()
            )
            weighted = value * regularization.weight
            total = weighted if total is None else total + weighted

        assert total is not None, "no losses configured"
        metrics["total"] = total.detach()
        return total, metrics


def materialize_metrics(
    metrics: dict[str, torch.Tensor],
) -> dict[str, float]:
    """Pull metric tensors to the host. One sync, on reporting steps only."""
    return {name: float(value) for name, value in metrics.items()}
