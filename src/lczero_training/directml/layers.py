"""DirectML-safe primitives for the native Windows training port.

Phase 3 of docs/directml_training_port.md. Every helper in the "DirectML
workarounds" section below exists because the obvious PyTorch spelling
either crashes the process or silently returns wrong values on a DirectML
device. Prefer these over the torch built-ins throughout the port; they are
ordinary PyTorch on CPU, so nothing is lost by using them everywhere.

Shapes here all carry a native leading batch dimension. The JAX model is
unbatched and vmapped externally (see model/model.py); this port does not
reproduce that, so every reference below to a JAX shape has one extra
leading axis here.
"""

from __future__ import annotations

import math
from collections.abc import Callable

import torch
from torch import nn

from proto import net_pb2

# nnx.LayerNorm(epsilon=1e-3) throughout the JAX model (encoder.py,
# embedding.py, Smolgen). Not PyTorch's 1e-5 default.
LAYER_NORM_EPS = 1e-3
# The output_rms_norm branch of KdaMixer uses 1e-6.
RMS_NORM_EPS = 1e-6

# flax's truncated_normal initializer divides by the standard deviation of a
# standard normal truncated to [-2, 2] so the result has the requested
# variance. Reproduced here so ported layers initialize identically.
_TRUNCATED_NORMAL_STDDEV = 0.87962566103423978


# --------------------------------------------------------------------------
# DirectML workarounds
# --------------------------------------------------------------------------


def normalize_dim(dim: int, ndim: int) -> int:
    """Map a possibly-negative axis index to its non-negative equivalent."""
    return dim + ndim if dim < 0 else dim


def flip(tensor: torch.Tensor, dim: int) -> torch.Tensor:
    """``torch.flip`` along one axis, with the axis forced non-negative.

    DirectML's flip kernel hard-crashes the process (access violation,
    0xC0000005) when handed a negative dim; the same call with the
    equivalent positive dim is correct. Verified on
    `Intel(R) Iris(R) Xe Graphics` with torch-directml 0.2.5.dev240914.
    """
    return torch.flip(tensor, [normalize_dim(dim, tensor.dim())])


def cumsum(tensor: torch.Tensor, dim: int) -> torch.Tensor:
    """``torch.cumsum`` whose *backward* is DirectML-safe.

    cumsum's gradient is a reverse cumulative sum, which autograd implements
    by flipping along the dim recorded during the forward pass. A negative
    dim therefore survives the forward pass and crashes in backward -- the
    failure surfaces far from its cause, so always route cumsum through
    here rather than calling torch.cumsum directly. See flip() above.
    """
    return torch.cumsum(tensor, dim=normalize_dim(dim, tensor.dim()))


def identity_matrix(
    size: int, *, dtype: torch.dtype, device: torch.device
) -> torch.Tensor:
    """A ``size x size`` identity, built without ``torch.eye``.

    ``torch.eye(..., device=directml_device)`` follows a broken fallback
    path. Constructing from ones and diag stays on the device.
    """
    return torch.diag(torch.ones(size, dtype=dtype, device=device))


def pad_last(tensor: torch.Tensor, width: int) -> torch.Tensor:
    """Right-pad the last axis with ``width`` zeros, skipping width 0.

    A zero-width ``torch.nn.functional.pad`` produces incorrect KDA results
    at chunk boundaries, so the call has to be elided rather than issued
    with a no-op width. Eliding returns the tensor as-is, and the KDA row
    loops feed it a ragged ``sum`` output that is still a view over the
    preceding elementwise product. Chaining further ops off an alias like
    that leaves the DirectML allocator unable to reclaim the underlying
    block until every view of it is dead, which shows up as a training OOM
    long after the row loop itself. ``contiguous()`` is a no-op on an
    already-contiguous tensor, so this costs nothing outside the ragged
    last row.
    """
    if width == 0:
        return tensor.contiguous()
    return torch.nn.functional.pad(tensor, (0, width))


def gather_along(
    tensor: torch.Tensor, dim: int, index: torch.Tensor
) -> torch.Tensor:
    """Static index selection along one axis.

    ``index`` is a precomputed int64 buffer (a board traversal order or a
    policy map), never a traced or data-dependent tensor -- keeping the
    selection static is what lets the whole step run with fixed shapes.

    Prefer :func:`permute_along` when the index is a permutation: this
    function's backward is ``index_add``, which DirectML does not implement
    and silently runs on the CPU.
    """
    return torch.index_select(tensor, normalize_dim(dim, tensor.dim()), index)


class _PermuteAlong(torch.autograd.Function):
    """index_select whose backward is another index_select.

    The generic gradient of ``index_select`` is a scatter-add, and DirectML
    has no ``aten::index_add.out`` kernel -- it falls back to the CPU,
    copying the tensor off and back on every backward pass. For a
    *permutation* the scatter-add degenerates to a plain gather by the
    inverse permutation, which stays on the device.
    """

    @staticmethod
    def forward(ctx, tensor, dim, order, inverse):  # type: ignore[override]
        ctx.dim = dim
        ctx.save_for_backward(inverse)
        return torch.index_select(tensor, dim, order)

    @staticmethod
    def backward(ctx, grad_output):  # type: ignore[override]
        (inverse,) = ctx.saved_tensors
        grad = torch.index_select(grad_output, ctx.dim, inverse)
        return grad, None, None, None


def permute_along(
    tensor: torch.Tensor,
    dim: int,
    order: torch.Tensor,
    inverse: torch.Tensor,
) -> torch.Tensor:
    """Reorder ``tensor`` along ``dim`` by the permutation ``order``.

    ``inverse`` must satisfy ``inverse[order] == arange(n)``; it is used
    only by the backward pass. Both are static int64 buffers.
    """
    return _PermuteAlong.apply(
        tensor, normalize_dim(dim, tensor.dim()), order, inverse
    )


class _InjectiveGather(torch.autograd.Function):
    """index_select by an injective (no repeats) index, without index_add.

    Selecting 1858 of 4288 policy logits is not a permutation, so
    permute_along does not apply -- but the index has no repeats, so the
    gradient still never *accumulates* into a slot. That makes the
    scatter-add expressible as a plain gather: append one zero slot to the
    incoming gradient and have every unselected position read from it.
    Keeps the policy head's backward on the device.
    """

    @staticmethod
    def forward(ctx, tensor, dim, index, inverse):  # type: ignore[override]
        ctx.dim = dim
        ctx.save_for_backward(inverse)
        return torch.index_select(tensor, dim, index)

    @staticmethod
    def backward(ctx, grad_output):  # type: ignore[override]
        (inverse,) = ctx.saved_tensors
        zero_shape = list(grad_output.shape)
        zero_shape[ctx.dim] = 1
        padded = torch.cat(
            [grad_output, grad_output.new_zeros(zero_shape)], dim=ctx.dim
        )
        return torch.index_select(padded, ctx.dim, inverse), None, None, None


def injective_inverse(index: torch.Tensor, input_size: int) -> torch.Tensor:
    """Backward index for :func:`injective_gather`.

    ``inverse[j]`` is the output position that read input ``j``, or the
    zero slot (``index.numel()``) for inputs nothing selected.
    """
    flat = index.reshape(-1)
    assert flat.unique().numel() == flat.numel(), (
        "injective_gather requires an index with no repeats"
    )
    inverse = torch.full((input_size,), flat.numel(), dtype=torch.int64)
    inverse[flat] = torch.arange(flat.numel(), dtype=torch.int64)
    return inverse


def injective_gather(
    tensor: torch.Tensor,
    dim: int,
    index: torch.Tensor,
    inverse: torch.Tensor,
) -> torch.Tensor:
    """``index_select`` whose backward stays on the device.

    ``inverse`` comes from :func:`injective_inverse`. Both are static
    int64 buffers.
    """
    return _InjectiveGather.apply(
        tensor, normalize_dim(dim, tensor.dim()), index, inverse
    )


# --------------------------------------------------------------------------
# Activations
# --------------------------------------------------------------------------

# DirectML's softplus backward is not the standard branch-on-sign-stable
# sigmoid(x): verified against the CPU backend, it returns 0 (not ~1) past
# x~88 and NaN past x~89, instead of the correct ~1 either side -- consistent
# with a kernel that computes 1/(1+exp(x)) unconditionally, which overflows
# exp(x) to inf right where float32 does (exp(88.7) is float32's ceiling).
# softplus(x) and its true gradient are already indistinguishable from x and
# 1 by x=20 (checked to 1e-7), so linearizing past that threshold changes
# nothing about the answer and only keeps it out of the region where this
# backend's kernel is wrong. This is what silently corrupted the moves-left
# head's gradient in production: dense1's output only needs to clear 88 in
# one unit on one batch, the rest of the network reads a perfectly normal
# loss, and the *only* symptom is a non-finite gradient with nothing
# non-finite anywhere describe_non_finite could see -- see training.py's
# describe_non_finite and _clip_grad_norm_preserving_origin docstrings for
# the rest of that story.
#
# Rather than clamp at every call site, ``torch.nn.functional.softplus`` is
# monkey-patched globally below to a threshold-linearized implementation,
# so every caller (mish, KDA log-decay, the device smoke tests) is protected
# without per-call clamping.
SOFTPLUS_SAFE_MAX = 20.0


def safe_directml_softplus(
    input: torch.Tensor, beta: float = 1, threshold: float = SOFTPLUS_SAFE_MAX
) -> torch.Tensor:
    """Stable softplus for Intel DirectML.

    Mirrors the threshold-linearized form of ``torch.nn.functional.softplus``
    that DirectML's translation omits: for ``input * beta`` above
    ``threshold`` it returns ``input`` exactly, and below it the usual
    ``log1p(exp(...)) / beta``. The argument fed to ``exp`` is clamped to
    ``threshold`` first, so the non-selected branch never produces an
    overflowing local gradient in backward -- a bare ``torch.where`` would
    still compute ``exp`` of the large values and yield ``0 * inf = NaN``
    gradients on this backend.
    """
    scaled = input * beta
    return torch.where(
        scaled > threshold,
        input,
        torch.log1p(torch.exp(scaled.clamp(max=threshold))) / beta,
    )


# Install the override globally for the DirectML backend. ``setattr`` is used
# so static type checkers do not flag the signature mismatch with PyTorch's
# overloaded ``softplus``.
setattr(torch.nn.functional, "softplus", safe_directml_softplus)
setattr(
    torch.nn.Softplus,
    "forward",
    lambda self, input: safe_directml_softplus(
        input, self.beta, self.threshold
    ),
)


def mish(x: torch.Tensor) -> torch.Tensor:
    """Mish, composed from primitives rather than calling F.mish."""
    return x * torch.tanh(torch.nn.functional.softplus(x))


def swish(x: torch.Tensor) -> torch.Tensor:
    """Swish/SiLU, composed from primitives rather than calling F.silu."""
    return x * torch.sigmoid(x)


def _identity(x: torch.Tensor) -> torch.Tensor:
    return x


_ACTIVATIONS: dict[int, Callable[[torch.Tensor], torch.Tensor]] = {
    net_pb2.NetworkFormat.ACTIVATION_MISH: mish,
    net_pb2.NetworkFormat.ACTIVATION_RELU: torch.relu,
    net_pb2.NetworkFormat.ACTIVATION_NONE: _identity,
    net_pb2.NetworkFormat.ACTIVATION_TANH: torch.tanh,
    net_pb2.NetworkFormat.ACTIVATION_SIGMOID: torch.sigmoid,
    net_pb2.NetworkFormat.ACTIVATION_SELU: torch.selu,
    net_pb2.NetworkFormat.ACTIVATION_SWISH: swish,
    net_pb2.NetworkFormat.ACTIVATION_SOFTMAX: lambda x: torch.softmax(
        x, dim=-1
    ),
}


def get_activation(
    activation: int,
) -> Callable[[torch.Tensor], torch.Tensor]:
    """Mirror of model/utils.py get_activation for the PyTorch port."""
    return _ACTIVATIONS[activation]


# --------------------------------------------------------------------------
# Normalization
# --------------------------------------------------------------------------


class LayerNorm(nn.Module):
    """Layer normalization built from mean, variance, rsqrt, scale, bias.

    Deliberately does not call ``torch.nn.LayerNorm`` or
    ``F.layer_norm``: their backward fails on DirectML with "tensor does
    not have a device".

    Parameter names match flax's (``scale``/``bias``) so the Phase 6 weight
    importer maps one-to-one.
    """

    def __init__(self, num_features: int, eps: float = LAYER_NORM_EPS):
        super().__init__()
        self.eps = eps
        self.scale = nn.Parameter(torch.ones(num_features))
        self.bias = nn.Parameter(torch.zeros(num_features))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        mean = x.mean(dim=-1, keepdim=True)
        centered = x - mean
        variance = centered.square().mean(dim=-1, keepdim=True)
        normalized = centered * torch.rsqrt(variance + self.eps)
        return normalized * self.scale + self.bias


class RmsNorm(nn.Module):
    """Root-mean-square normalization with a learned per-channel scale.

    Mirrors the ``output_rms_norm`` branch of the JAX KdaMixer.
    """

    def __init__(self, num_features: int, eps: float = RMS_NORM_EPS):
        super().__init__()
        self.eps = eps
        self.scale = nn.Parameter(torch.ones(num_features))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        variance = x.square().mean(dim=-1, keepdim=True)
        return x * torch.rsqrt(variance + self.eps) * self.scale


# --------------------------------------------------------------------------
# DeepNorm
# --------------------------------------------------------------------------


def deepnorm_alpha(num_blocks: int) -> float:
    """Residual scale, ``(2N)^-0.25``. Matches model/model.py."""
    return math.pow(2.0 * num_blocks, -0.25)


def deepnorm_beta(num_blocks: int) -> float:
    """Initializer scale, ``(8N)^-0.25``. Matches model/model.py."""
    return math.pow(8.0 * num_blocks, -0.25)


def deepnorm_residual(
    x: torch.Tensor, sublayer_out: torch.Tensor, alpha: float
) -> torch.Tensor:
    """DeepNorm residual: the skip path is unscaled, the sublayer is."""
    return x + sublayer_out * alpha


def init_variance_scaling_(
    linear: nn.Linear, scale: float, *, bias_zero: bool = True
) -> None:
    """flax ``variance_scaling(scale, "fan_avg", "truncated_normal")``.

    ``nn.Linear.weight`` is (out_features, in_features), the transpose of
    flax's kernel, but fan_avg is symmetric in the two so the resulting
    standard deviation is identical.
    """
    fan_in = linear.in_features
    fan_out = linear.out_features
    stddev = math.sqrt(scale / ((fan_in + fan_out) / 2.0))
    stddev /= _TRUNCATED_NORMAL_STDDEV
    nn.init.trunc_normal_(
        linear.weight, mean=0.0, std=stddev, a=-2.0 * stddev, b=2.0 * stddev
    )
    if bias_zero and linear.bias is not None:
        nn.init.zeros_(linear.bias)


def init_lecun_normal_(linear: nn.Linear, *, bias_zero: bool = True) -> None:
    """flax's default ``nnx.Linear`` kernel init (lecun_normal) plus zero bias."""
    stddev = math.sqrt(1.0 / linear.in_features) / _TRUNCATED_NORMAL_STDDEV
    nn.init.trunc_normal_(
        linear.weight, mean=0.0, std=stddev, a=-2.0 * stddev, b=2.0 * stddev
    )
    if bias_zero and linear.bias is not None:
        nn.init.zeros_(linear.bias)


def init_lecun_normal_conv_(conv: nn.Conv2d, *, bias_zero: bool = True) -> None:
    """flax's default ``nnx.Conv`` kernel init (lecun_normal) plus zero bias.

    flax initializes conv kernels with ``variance_scaling(1.0, "fan_in",
    "truncated_normal")`` and biases at zero; PyTorch's ``nn.Conv2d``
    defaults (Kaiming-uniform kernel, uniform bias) differ, so a torch
    model built from scratch started from a different distribution than
    its JAX counterpart. Every other ported layer matched flax's inits
    already -- this was the one that slipped through.

    Fan-in for a (possibly grouped) conv kernel of shape
    ``(out, in // groups, kh, kw)`` is ``in // groups * kh * kw`` --
    flax's receptive-field size, not ``in_channels``.
    """
    kernel = conv.weight
    receptive = kernel.shape[1] * kernel.shape[2] * kernel.shape[3]
    stddev = math.sqrt(1.0 / receptive) / _TRUNCATED_NORMAL_STDDEV
    nn.init.trunc_normal_(
        kernel, mean=0.0, std=stddev, a=-2.0 * stddev, b=2.0 * stddev
    )
    if bias_zero and conv.bias is not None:
        nn.init.zeros_(conv.bias)


# --------------------------------------------------------------------------
# Feed-forward
# --------------------------------------------------------------------------


class Ffn(nn.Module):
    """Two-layer feed-forward block. Mirrors model/shared.py Ffn."""

    def __init__(
        self,
        in_features: int,
        hidden_features: int,
        hidden_activation: int,
        deepnorm_beta_value: float,
    ):
        super().__init__()
        self.linear1 = nn.Linear(in_features, hidden_features)
        self.linear2 = nn.Linear(hidden_features, in_features)
        init_variance_scaling_(self.linear1, deepnorm_beta_value)
        init_variance_scaling_(self.linear2, deepnorm_beta_value)
        self.activation = get_activation(hidden_activation)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear2(self.activation(self.linear1(x)))
