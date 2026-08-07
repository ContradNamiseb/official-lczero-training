"""DirectML device discovery and smoke tests.

Importing this module requires the optional `directml` extra (torch and
torch-directml). Nothing in the base package imports it, so a Linux
environment without PyTorch can still import `lczero_training`.
"""

from __future__ import annotations

import dataclasses

import torch

_IMPORT_HINT = (
    "torch-directml is not installed. Create the Python 3.12 DirectML "
    "environment first: uv sync --extra directml"
)

try:
    import torch_directml
except ImportError as error:  # pragma: no cover - environment dependent
    raise ImportError(_IMPORT_HINT) from error


def ensure_initialized() -> None:
    """Guarantee ``torch_directml`` was imported before autograd starts.

    Importing this module is the whole implementation -- the function exists
    so callers have something explicit to call, and somewhere to read why.

    PyTorch's autograd engine sizes its per-device ready queues the first
    time a backward pass runs. If ``torch_directml`` has not been imported
    by then, the PrivateUse1 backend is absent from that count and *every*
    later DirectML backward dies with::

        RuntimeError: 0 <= device.index() && device.index() <
        device_ready_queues_.size() INTERNAL ASSERT FAILED at
        torch/csrc/autograd/engine.cpp:1451

    Note the trigger is the import, not touching a device: a CPU-only
    backward run before ``import torch_directml`` is enough to poison the
    process. Any entry point that may run a CPU backward before reaching
    the DirectML path must import this module first.
    """


def device_count() -> int:
    """Number of DirectML adapters visible to this process."""
    return int(torch_directml.device_count())


def adapter_name(index: int = 0) -> str:
    """Human-readable adapter name, e.g. ``Intel(R) Iris(R) Xe Graphics``."""
    return str(torch_directml.device_name(index))


def get_device(index: int = 0) -> torch.device:
    """Return the ``privateuseone`` device for the given adapter index."""
    if device_count() == 0:
        raise RuntimeError("No DirectML adapter was found.")
    return torch_directml.device(index)


def resolve_device(spec: str | None = None) -> torch.device:
    """Turn a CLI device string into a torch device.

    ``None`` and ``"directml"`` select adapter 0; ``"directml:N"`` selects
    adapter N; anything else is passed to ``torch.device`` unchanged, which
    keeps ``--device cpu`` working for the JAX-comparison paths.
    """
    if spec is None or spec == "directml":
        return get_device(0)
    if spec.startswith("directml:"):
        return get_device(int(spec.split(":", 1)[1]))
    return torch.device(spec)


@dataclasses.dataclass
class SmokeResult:
    name: str
    passed: bool
    detail: str


def _check(name: str, fn) -> SmokeResult:
    try:
        detail = fn()
    except Exception as error:  # noqa: BLE001 - diagnostics report everything
        return SmokeResult(name, False, f"{type(error).__name__}: {error}")
    return SmokeResult(name, True, detail)


def run_smoke_tests(device: torch.device) -> list[SmokeResult]:
    """Allocation and backward smoke tests against a DirectML device.

    These cover exactly the operations the port depends on and that were
    verified working in docs/directml_training_port.md: allocation,
    elementwise backward, matmul backward, softplus/exp/cumsum, batched
    matmul, and depthwise-convolution backward.
    """

    def allocation() -> str:
        tensor = torch.ones(64, 128, device=device)
        return f"sum={float(tensor.sum().cpu()):.1f}"

    def elementwise_backward() -> str:
        x = torch.randn(64, 128, device=device, requires_grad=True)
        (x * x).sum().backward()
        assert x.grad is not None
        return f"finite={bool(torch.isfinite(x.grad).all().cpu())}"

    def matmul_backward() -> str:
        a = torch.randn(64, 128, device=device, requires_grad=True)
        b = torch.randn(128, 32, device=device, requires_grad=True)
        (a @ b).sum().backward()
        assert a.grad is not None and b.grad is not None
        finite = bool(
            torch.isfinite(a.grad).all().cpu()
            and torch.isfinite(b.grad).all().cpu()
        )
        return f"finite={finite}"

    def unary_ops() -> str:
        x = torch.randn(8, 16, device=device, requires_grad=True)
        # dim=1, not dim=-1: cumsum's backward flips along the dim recorded
        # here, and DirectML's flip faults on a negative axis. See
        # layers.cumsum.
        y = torch.nn.functional.softplus(x).exp().cumsum(dim=1)
        y.sum().backward()
        assert x.grad is not None
        return f"finite={bool(torch.isfinite(x.grad).all().cpu())}"

    def flip_positive_dim() -> str:
        x = torch.randn(8, 16, device=device)
        return f"sum={float(torch.flip(x, [1]).sum().cpu()):.4f}"

    def batched_matmul() -> str:
        a = torch.randn(4, 8, 16, 32, device=device, requires_grad=True)
        b = torch.randn(4, 8, 32, 16, device=device, requires_grad=True)
        (a @ b).sum().backward()
        assert a.grad is not None
        return f"finite={bool(torch.isfinite(a.grad).all().cpu())}"

    def depthwise_conv_backward() -> str:
        x = torch.randn(2, 16, 8, 8, device=device, requires_grad=True)
        weight = torch.randn(16, 1, 3, 3, device=device, requires_grad=True)
        out = torch.nn.functional.conv2d(x, weight, padding=1, groups=16)
        out.sum().backward()
        assert x.grad is not None and weight.grad is not None
        finite = bool(
            torch.isfinite(x.grad).all().cpu()
            and torch.isfinite(weight.grad).all().cpu()
        )
        return f"finite={finite}"

    return [
        _check("allocation", allocation),
        _check("elementwise backward", elementwise_backward),
        _check("matmul backward", matmul_backward),
        _check("softplus/exp/cumsum backward", unary_ops),
        _check("flip (positive dim)", flip_positive_dim),
        _check("batched matmul backward", batched_matmul),
        _check("depthwise conv backward", depthwise_conv_backward),
    ]
