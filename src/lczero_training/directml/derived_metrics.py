"""Diagnostic metrics matching the TF pipeline's TensorBoard tags.

These are *observations*, not losses -- nothing here contributes to the
gradient. The names are the TF trainer's `Metric` long names verbatim
(stable-branch/tf/tfprocess.py, ~L1070-1140), so a DirectML run can be
overlaid on an old `leelalogs` run in TensorBoard and read directly.

Every formula mirrors the corresponding *_fn in tfprocess.py ~L790-950.
Percentages are scaled by 100 there too, so they match without conversion.
"""

from __future__ import annotations

import torch

# tfprocess.py's default accuracy_thresholds, in percent.
ACCURACY_THRESHOLDS = (1, 2, 5, 10)
# policy_search_loss's epsilon.
_SEARCH_EPSILON = 0.003
# Stands in for tfprocess's -1e10 illegal-move filler. Large enough to zero
# the softmax, small enough not to produce inf on the way there.
_ILLEGAL_LOGIT = -1.0e10


def _correct_policy(
    target: torch.Tensor, output: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    """Mask illegal moves and renormalize the target.

    Mirrors correct_policy() in tfprocess.py: illegal moves are flagged by a
    negative target, their logits are pushed to -1e10, and the target is
    relu'd then renormalized to sum to one.
    """
    legal = target >= 0
    output = torch.where(
        legal, output, torch.full_like(output, _ILLEGAL_LOGIT)
    )
    target = torch.relu(target)
    total = target.sum(dim=1, keepdim=True)
    target = target / torch.where(total > 0, total, torch.ones_like(total))
    return target, output


def _probability_at_best_move(
    target: torch.Tensor, softmaxed: torch.Tensor
) -> torch.Tensor:
    """Predicted probability of the move the search liked most."""
    best = target.argmax(dim=1, keepdim=True)
    return softmaxed.gather(1, best).squeeze(1)


def policy_metrics(
    target: torch.Tensor, output: torch.Tensor
) -> dict[str, torch.Tensor]:
    """Accuracy, entropy, uniform loss, search loss, thresholded accuracy."""
    target, output = _correct_policy(target, output)
    softmaxed = torch.softmax(output, dim=1)

    accuracy = (
        (target.argmax(dim=1) == output.argmax(dim=1)).float().mean() * 100.0
    )

    # xlogy: 0 * log(0) is 0, not nan.
    entropy_terms = torch.where(
        softmaxed > 0,
        softmaxed * torch.log(softmaxed.clamp_min(1e-30)),
        torch.zeros_like(softmaxed),
    )
    entropy = -entropy_terms.sum(dim=1).mean()

    # Uniform loss: cross-entropy against a flat distribution over the legal
    # moves, i.e. how far the policy is from knowing nothing.
    legal = (target > 0).float()
    uniform = legal / legal.sum(dim=1, keepdim=True).clamp_min(1.0)
    log_probs = torch.log_softmax(output, dim=1)
    uniform_terms = torch.where(
        uniform > 0, uniform * log_probs, torch.zeros_like(uniform)
    )
    uniform_loss = -uniform_terms.sum(dim=1).mean()

    # Search loss: time to find the best move is roughly 1/P(best move).
    at_best = _probability_at_best_move(target, softmaxed)
    search_loss = (1.0 / (at_best + _SEARCH_EPSILON)).mean()

    metrics = {
        "Policy Accuracy": accuracy,
        "Policy Entropy": entropy,
        "Policy UL": uniform_loss,
        "Policy SL": search_loss,
    }
    for threshold in ACCURACY_THRESHOLDS:
        metrics[f"Thresholded Policy Accuracy @ {threshold}"] = (
            (at_best > threshold / 100.0).float().mean() * 100.0
        )
    return metrics


def value_metrics(
    wdl_logits: torch.Tensor, q: torch.Tensor, d: torch.Tensor
) -> dict[str, torch.Tensor]:
    """Value accuracy and MSE against the target WDL."""
    target = torch.stack(
        [
            (torch.ones_like(q) + q - d) / 2.0,
            d,
            (torch.ones_like(q) - q - d) / 2.0,
        ],
        dim=-1,
    )
    predicted = torch.softmax(wdl_logits, dim=-1)
    accuracy = (
        (predicted.argmax(dim=-1) == target.argmax(dim=-1)).float().mean()
        * 100.0
    )
    mse = (predicted - target).square().sum(dim=-1).mean()
    return {"Value Accuracy": accuracy, "MSE Loss": mse}


def set_kda_stats_collection(model: torch.nn.Module, enabled: bool) -> None:
    """Turn the KDA mixers' internal stat capture on or off."""
    from .kda import KdaMixer

    for module in model.modules():
        if isinstance(module, KdaMixer):
            module.collect_stats = enabled


def kda_metrics(model: torch.nn.Module) -> dict[str, torch.Tensor]:
    """Per-block gate diagnostics from whatever the last forward captured.

    KDA has failure modes the losses do not expose: the forget gate pinning
    at its floor (state wiped every token), beta collapsing to zero (nothing
    written), or the output gate closing (mixer silenced). Each leaves total
    loss looking unremarkable while the mixer stops doing its job.
    """
    from .kda import KdaMixer

    metrics: dict[str, torch.Tensor] = {}
    index = 0
    for module in model.modules():
        if not isinstance(module, KdaMixer):
            continue
        for name, value in module.last_stats.items():
            metrics[f"KDA/block{index} {name}"] = value
        index += 1
    return metrics


def parameter_metrics(model: torch.nn.Module) -> dict[str, torch.Tensor]:
    """Weight-norm summaries. The TF pipeline logs these on the test writer.

    Reported as the L2 norm over each group rather than a histogram, since
    the point here is a trend line, not a distribution.
    """
    groups: dict[str, list[torch.Tensor]] = {
        "Params": [],
        "Embedding params": [],
        "Smolgen params": [],
    }
    seen: set[int] = set()
    for name, parameter in model.named_parameters():
        if id(parameter) in seen:
            continue
        seen.add(id(parameter))
        squared = parameter.detach().square().sum()
        groups["Params"].append(squared)
        if name.startswith("embedding."):
            groups["Embedding params"].append(squared)
        if "smolgen" in name:
            groups["Smolgen params"].append(squared)

    return {
        name: torch.sqrt(torch.stack(values).sum())
        for name, values in groups.items()
        if values
    }


# The trainer's own metric names mapped onto the TF pipeline's, so both show
# up on the same TensorBoard axes. Anything not listed keeps its own name.
TF_TAG_ALIASES = {
    "total": "Total Loss",
    "grad_norm": "Gradient norm",
    "lr": "LR",
    "policy/main_ce": "Policy Loss",
    "value/winner": "Value Winner Loss",
    "movesleft/main": "Moves Left Loss",
    "value/q": "Value Q Loss",
    "value/st": "Value ST Loss",
    "value_error/q": "Value Err L",
    "value_error/st": "Value ST Err Loss",
    "value_categorical/q": "Value Cat L",
    "value_categorical/st": "Value ST Cat Loss",
    "policy/optimistic_st": "Policy Optimistic ST Loss",
    "regularization/l2": "Reg term",
}


def apply_tf_aliases(scalars: dict[str, float]) -> dict[str, float]:
    """Rename metrics to the TF pipeline's tags where an equivalent exists."""
    return {TF_TAG_ALIASES.get(name, name): value for name, value in scalars.items()}
