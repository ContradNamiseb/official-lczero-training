"""NAdamW matching Optax, plus the decay selector, for the DirectML port.

Phase 7 of docs/directml_training_port.md, which anticipated this: "If
PyTorch's NAdam equations or momentum-product handling differ from Optax,
implement the Optax update equations directly." They do differ.

``torch.optim.NAdam`` implements the original Nadam paper's *scheduled*
momentum, accumulating a running product of
``beta1 * (1 - 0.5 * 0.96 ** (t * momentum_decay))``. Optax's
``scale_by_adam(nesterov=True)`` uses plain ``beta1`` with two different
bias corrections and no schedule. Restoring an optimizer state at step
97000 under the wrong equations would quietly change the trajectory, so
this module reproduces Optax exactly.
"""

from __future__ import annotations

import fnmatch
from collections.abc import Iterable, Iterator

import torch

from proto.training_config_pb2 import OptimizerConfig, WeightsSelector


def _match_parts(parts: tuple[str, ...], patterns: list[str]) -> bool:
    """Glob match with ``**`` spanning any number of path segments.

    Reimplements ``pathlib.PurePath.full_match``, which the JAX side uses
    but which only exists in Python 3.13; the DirectML environment is
    pinned to 3.12 because torch-directml has no 3.13 wheels.
    """
    if not patterns:
        return not parts
    head, rest = patterns[0], patterns[1:]
    if head == "**":
        return any(
            _match_parts(parts[index:], rest) for index in range(len(parts) + 1)
        )
    if not parts:
        return False
    if fnmatch.fnmatchcase(parts[0], head):
        return _match_parts(parts[1:], rest)
    return False


def selector_includes(selector: WeightsSelector, name: str) -> bool:
    """Apply a WeightsSelector to a PyTorch parameter name.

    ``name`` is dotted (``encoders.0.ln1.scale``); the rules are written as
    posix-style globs against the JAX state tree. The two agree for every
    rule the shipped configs use -- none of them depend on PyTorch calling
    a linear's matrix ``weight`` where Flax calls it ``kernel``.
    """
    parts = tuple(name.split("."))
    for rule in selector.rule:
        if _match_parts(parts, rule.match.split("/")):
            return rule.include
    return selector.otherwise_include


class NAdamW(torch.optim.Optimizer):
    """Optax's ``nadamw``: Nesterov Adam with decoupled weight decay."""

    def __init__(
        self,
        params,
        lr: float,
        betas: tuple[float, float] = (0.9, 0.98),
        eps: float = 1e-7,
        weight_decay: float = 0.0,
        eps_root: float = 0.0,
    ):
        defaults = dict(
            lr=lr,
            betas=betas,
            eps=eps,
            weight_decay=weight_decay,
            eps_root=eps_root,
        )
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure=None):  # type: ignore[override]
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            beta1, beta2 = group["betas"]
            lr = group["lr"]
            eps = group["eps"]
            eps_root = group["eps_root"]
            weight_decay = group["weight_decay"]

            for param in group["params"]:
                if param.grad is None:
                    continue
                grad = param.grad
                state = self.state[param]
                if not state:
                    state["step"] = 0
                    state["mu"] = torch.zeros_like(param)
                    state["nu"] = torch.zeros_like(param)

                mu, nu = state["mu"], state["nu"]
                mu.mul_(beta1).add_(grad, alpha=1.0 - beta1)
                nu.mul_(beta2).addcmul_(grad, grad, value=1.0 - beta2)
                count = state["step"] + 1
                state["step"] = count

                # Optax's nesterov branch: the first moment is bias-corrected
                # one step ahead, and blended with the bias-corrected raw
                # gradient. This is the part torch.optim.NAdam does not do.
                mu_hat = mu / (1.0 - beta1 ** (count + 1))
                grad_hat = grad / (1.0 - beta1**count)
                blended = beta1 * mu_hat + (1.0 - beta1) * grad_hat
                nu_hat = nu / (1.0 - beta2**count)

                update = blended / (torch.sqrt(nu_hat + eps_root) + eps)
                if weight_decay != 0.0:
                    # Decoupled: added to the update, not to the gradient, so
                    # it is not scaled by the adaptive denominator.
                    update = update + weight_decay * param
                param.add_(update, alpha=-lr)

        return loss

    def set_step(self, step: int) -> None:
        """Force every moment's step counter, for a mid-run import.

        Importing a network at step 97000 with counters at zero would apply
        the early-training bias correction to a fully warmed-up model.
        """
        for group in self.param_groups:
            for param in group["params"]:
                state = self.state[param]
                if not state:
                    state["step"] = step
                    state["mu"] = torch.zeros_like(param)
                    state["nu"] = torch.zeros_like(param)
                else:
                    state["step"] = step


def build_optimizer(
    named_parameters: Iterable[tuple[str, torch.nn.Parameter]],
    config: OptimizerConfig,
    *,
    learning_rate: float,
) -> NAdamW:
    """Build NAdamW with decay/no-decay parameter groups from the selector."""
    if not config.HasField("nadamw"):
        raise NotImplementedError(
            "Only the nadamw optimizer is ported; the real-import config "
            f"uses {config.WhichOneof('optimizer_type')}."
        )
    if config.HasField("freeze_selector"):
        raise NotImplementedError("freeze_selector is not ported yet.")

    nadamw = config.nadamw
    decayed: list[torch.nn.Parameter] = []
    plain: list[torch.nn.Parameter] = []
    seen: set[int] = set()
    for name, param in named_parameters:
        # Shared modules appear under several names; decay them once.
        if id(param) in seen:
            continue
        seen.add(id(param))
        target = (
            decayed if selector_includes(nadamw.decay_selector, name) else plain
        )
        target.append(param)

    return NAdamW(
        [
            {"params": decayed, "weight_decay": nadamw.weight_decay},
            {"params": plain, "weight_decay": 0.0},
        ],
        lr=learning_rate,
        betas=(nadamw.beta_1, nadamw.beta_2),
        eps=nadamw.epsilon,
    )


def iter_decay_decisions(
    named_parameters: Iterable[tuple[str, torch.nn.Parameter]],
    selector: WeightsSelector,
) -> Iterator[tuple[str, bool]]:
    """(name, decayed) for each parameter. Used by the tests and diagnostics."""
    for name, _ in named_parameters:
        yield name, selector_includes(selector, name)
