"""Three-way tests for the DirectML-safe primitives.

Phase 3 of docs/directml_training_port.md requires, for every primitive:

1. PyTorch CPU output against the JAX implementation.
2. PyTorch CPU gradient against JAX where practical.
3. DirectML execution with finite output and gradients.

The DirectML leg skips cleanly when no adapter is present, so this file is
importable and mostly runnable in Linux CI.
"""

import numpy as np
import pytest

torch = pytest.importorskip("torch")
jax = pytest.importorskip("jax")

import jax.numpy as jnp
from flax import nnx

from lczero_training.directml import layers
from lczero_training.model.shared import Ffn as JaxFfn
from proto import net_pb2

RTOL = 1e-5
ATOL = 1e-5


# The `dml_device` fixture lives in conftest.py, which also performs the
# eager torch_directml import the autograd engine needs. Do not move it
# here: conftest is imported before any test module, so it is the only
# place that can guarantee the import precedes the first backward pass.


def _assert_close(actual, expected, rtol=RTOL, atol=ATOL):
    np.testing.assert_allclose(
        np.asarray(actual, dtype=np.float64),
        np.asarray(expected, dtype=np.float64),
        rtol=rtol,
        atol=atol,
    )


# --------------------------------------------------------------------------
# Activations
# --------------------------------------------------------------------------

ACTIVATION_CASES = [
    ("mish", layers.mish, jax.nn.mish),
    ("swish", layers.swish, jax.nn.swish),
]


@pytest.mark.parametrize("name,torch_fn,jax_fn", ACTIVATION_CASES)
def test_activation_matches_jax(name, torch_fn, jax_fn):
    rng = np.random.default_rng(0)
    x = rng.normal(size=(7, 13)).astype(np.float32)
    _assert_close(torch_fn(torch.from_numpy(x)).numpy(), jax_fn(jnp.asarray(x)))


@pytest.mark.parametrize("name,torch_fn,jax_fn", ACTIVATION_CASES)
def test_activation_gradient_matches_jax(name, torch_fn, jax_fn):
    rng = np.random.default_rng(1)
    x = rng.normal(size=(7, 13)).astype(np.float32)

    tensor = torch.from_numpy(x).requires_grad_(True)
    torch_fn(tensor).square().sum().backward()

    def loss(value):
        return jnp.sum(jnp.square(jax_fn(value)))

    expected = jax.grad(loss)(jnp.asarray(x))
    _assert_close(tensor.grad.numpy(), expected, rtol=1e-4, atol=1e-5)


def test_get_activation_covers_model_defaults():
    """The activations the target config actually names must resolve."""
    for activation in (
        net_pb2.NetworkFormat.ACTIVATION_MISH,
        net_pb2.NetworkFormat.ACTIVATION_SWISH,
        net_pb2.NetworkFormat.ACTIVATION_RELU,
        net_pb2.NetworkFormat.ACTIVATION_NONE,
    ):
        assert callable(layers.get_activation(activation))


# --------------------------------------------------------------------------
# Normalization
# --------------------------------------------------------------------------


def test_layer_norm_matches_jax():
    rng = np.random.default_rng(2)
    features = 16
    x = rng.normal(size=(5, features)).astype(np.float32)
    scale = rng.normal(size=(features,)).astype(np.float32)
    bias = rng.normal(size=(features,)).astype(np.float32)

    torch_norm = layers.LayerNorm(features)
    with torch.no_grad():
        torch_norm.scale.copy_(torch.from_numpy(scale))
        torch_norm.bias.copy_(torch.from_numpy(bias))
    actual = torch_norm(torch.from_numpy(x)).detach().numpy()

    jax_norm = nnx.LayerNorm(
        features, epsilon=layers.LAYER_NORM_EPS, rngs=nnx.Rngs(0)
    )
    jax_norm.scale.value = jnp.asarray(scale)
    jax_norm.bias.value = jnp.asarray(bias)
    _assert_close(actual, jax_norm(jnp.asarray(x)))


def test_layer_norm_gradient_matches_jax():
    rng = np.random.default_rng(3)
    features = 16
    x = rng.normal(size=(5, features)).astype(np.float32)

    torch_norm = layers.LayerNorm(features)
    tensor = torch.from_numpy(x).requires_grad_(True)
    torch_norm(tensor).square().sum().backward()

    jax_norm = nnx.LayerNorm(
        features, epsilon=layers.LAYER_NORM_EPS, rngs=nnx.Rngs(0)
    )

    def loss(value):
        return jnp.sum(jnp.square(jax_norm(value)))

    _assert_close(
        tensor.grad.numpy(),
        jax.grad(loss)(jnp.asarray(x)),
        rtol=1e-4,
        atol=1e-5,
    )


def test_rms_norm_matches_jax_formula():
    """Mirrors the output_rms_norm branch of the JAX KdaMixer."""
    rng = np.random.default_rng(4)
    features = 24
    x = rng.normal(size=(6, features)).astype(np.float32)
    gammas = rng.normal(size=(features,)).astype(np.float32)

    torch_norm = layers.RmsNorm(features)
    with torch.no_grad():
        torch_norm.scale.copy_(torch.from_numpy(gammas))
    actual = torch_norm(torch.from_numpy(x)).detach().numpy()

    value = jnp.asarray(x)
    variance = jnp.mean(jnp.square(value), axis=-1, keepdims=True)
    expected = value * jax.lax.rsqrt(variance + layers.RMS_NORM_EPS)
    expected = expected * jnp.asarray(gammas)
    _assert_close(actual, expected)


# --------------------------------------------------------------------------
# DeepNorm
# --------------------------------------------------------------------------


def test_deepnorm_constants_match_model():
    for num_blocks in (1, 4, 15):
        assert layers.deepnorm_alpha(num_blocks) == pytest.approx(
            pow(2.0 * num_blocks, -0.25)
        )
        assert layers.deepnorm_beta(num_blocks) == pytest.approx(
            pow(8.0 * num_blocks, -0.25)
        )


def test_deepnorm_residual_scales_only_the_sublayer():
    x = torch.ones(3, 4)
    sublayer = torch.full((3, 4), 2.0)
    out = layers.deepnorm_residual(x, sublayer, 0.5)
    _assert_close(out.numpy(), np.full((3, 4), 2.0, dtype=np.float32))


def test_variance_scaling_init_reproduces_flax_stddev():
    """Sample stddev of the init must match flax's fan_avg formula."""
    linear = torch.nn.Linear(256, 512)
    beta = layers.deepnorm_beta(4)
    layers.init_variance_scaling_(linear, beta)
    expected = np.sqrt(beta / ((256 + 512) / 2.0))
    assert float(linear.weight.std()) == pytest.approx(expected, rel=0.05)
    assert float(linear.bias.abs().sum()) == 0.0


# --------------------------------------------------------------------------
# Feed-forward
# --------------------------------------------------------------------------


def _make_matched_ffn(in_features, hidden_features, seed):
    """A torch Ffn and a JAX Ffn holding identical weights."""
    activation = net_pb2.NetworkFormat.ACTIVATION_MISH
    beta = layers.deepnorm_beta(4)
    torch_ffn = layers.Ffn(in_features, hidden_features, activation, beta)
    jax_ffn = JaxFfn(
        in_features=in_features,
        hidden_features=hidden_features,
        hidden_activation=activation,
        deepnorm_beta=beta,
        rngs=nnx.Rngs(seed),
    )
    # nn.Linear.weight is (out, in); the flax kernel is (in, out).
    jax_ffn.linear1.kernel.value = jnp.asarray(
        torch_ffn.linear1.weight.detach().numpy().T
    )
    jax_ffn.linear1.bias.value = jnp.asarray(
        torch_ffn.linear1.bias.detach().numpy()
    )
    jax_ffn.linear2.kernel.value = jnp.asarray(
        torch_ffn.linear2.weight.detach().numpy().T
    )
    jax_ffn.linear2.bias.value = jnp.asarray(
        torch_ffn.linear2.bias.detach().numpy()
    )
    return torch_ffn, jax_ffn


def test_ffn_matches_jax():
    rng = np.random.default_rng(5)
    x = rng.normal(size=(64, 32)).astype(np.float32)
    torch_ffn, jax_ffn = _make_matched_ffn(32, 48, seed=5)
    actual = torch_ffn(torch.from_numpy(x)).detach().numpy()
    _assert_close(actual, jax_ffn(jnp.asarray(x)), rtol=1e-4, atol=1e-5)


def test_ffn_gradient_matches_jax():
    rng = np.random.default_rng(6)
    x = rng.normal(size=(64, 32)).astype(np.float32)
    torch_ffn, jax_ffn = _make_matched_ffn(32, 48, seed=6)

    tensor = torch.from_numpy(x).requires_grad_(True)
    torch_ffn(tensor).square().sum().backward()

    def loss(value):
        return jnp.sum(jnp.square(jax_ffn(value)))

    _assert_close(
        tensor.grad.numpy(),
        jax.grad(loss)(jnp.asarray(x)),
        rtol=1e-4,
        atol=1e-5,
    )


# --------------------------------------------------------------------------
# Index selection
# --------------------------------------------------------------------------


def test_gather_along_matches_jax_take():
    rng = np.random.default_rng(7)
    x = rng.normal(size=(2, 64, 8)).astype(np.float32)
    order = rng.permutation(64).astype(np.int64)

    actual = layers.gather_along(
        torch.from_numpy(x), 1, torch.from_numpy(order)
    ).numpy()
    _assert_close(actual, jnp.take(jnp.asarray(x), jnp.asarray(order), axis=1))


def test_permute_along_matches_gather_along():
    rng = np.random.default_rng(12)
    x = rng.normal(size=(2, 64, 8)).astype(np.float32)
    order = torch.from_numpy(rng.permutation(64).astype(np.int64))
    inverse = torch.argsort(order)

    tensor = torch.from_numpy(x)
    _assert_close(
        layers.permute_along(tensor, 1, order, inverse).numpy(),
        layers.gather_along(tensor, 1, order).numpy(),
    )


def test_permute_along_gradient_matches_gather_along():
    """The custom backward must agree with autograd's scatter-add."""
    rng = np.random.default_rng(13)
    x = rng.normal(size=(2, 64, 8)).astype(np.float32)
    weights = torch.from_numpy(rng.normal(size=(2, 64, 8)).astype(np.float32))
    order = torch.from_numpy(rng.permutation(64).astype(np.int64))
    inverse = torch.argsort(order)

    reference = torch.from_numpy(x).requires_grad_(True)
    (layers.gather_along(reference, 1, order) * weights).sum().backward()

    actual = torch.from_numpy(x).requires_grad_(True)
    (layers.permute_along(actual, 1, order, inverse) * weights).sum().backward()

    _assert_close(actual.grad.numpy(), reference.grad.numpy())


def test_gather_along_roundtrips_with_inverse_order():
    rng = np.random.default_rng(8)
    x = torch.from_numpy(rng.normal(size=(2, 64, 8)).astype(np.float32))
    order = torch.from_numpy(rng.permutation(64).astype(np.int64))
    inverse = torch.argsort(order)
    gathered = layers.gather_along(x, 1, order)
    _assert_close(layers.gather_along(gathered, 1, inverse).numpy(), x.numpy())


# --------------------------------------------------------------------------
# Workaround helpers
# --------------------------------------------------------------------------


def test_identity_matrix_matches_eye_on_cpu():
    actual = layers.identity_matrix(
        8, dtype=torch.float32, device=torch.device("cpu")
    )
    _assert_close(actual.numpy(), np.eye(8, dtype=np.float32))


def test_pad_last_skips_zero_width():
    x = torch.arange(6, dtype=torch.float32).reshape(2, 3)
    assert layers.pad_last(x, 0) is x
    _assert_close(
        layers.pad_last(x, 2).numpy(),
        np.pad(x.numpy(), ((0, 0), (0, 2))),
    )


def test_normalize_dim():
    assert layers.normalize_dim(-1, 4) == 3
    assert layers.normalize_dim(-4, 4) == 0
    assert layers.normalize_dim(2, 4) == 2


def test_safe_cumsum_matches_torch_on_cpu():
    rng = np.random.default_rng(9)
    x = torch.from_numpy(rng.normal(size=(2, 3, 8, 5)).astype(np.float32))
    for dim in (-1, -2, 2, 3):
        _assert_close(
            layers.cumsum(x, dim).numpy(), torch.cumsum(x, dim=dim).numpy()
        )


def test_safe_flip_matches_torch_on_cpu():
    rng = np.random.default_rng(10)
    x = torch.from_numpy(rng.normal(size=(2, 3, 8)).astype(np.float32))
    for dim in (-1, -2, 1, 2):
        _assert_close(layers.flip(x, dim).numpy(), torch.flip(x, [dim]).numpy())


# --------------------------------------------------------------------------
# DirectML leg: finite output and gradients
# --------------------------------------------------------------------------


def _finite(tensor):
    return bool(torch.isfinite(tensor).all().cpu())


DEVICE_UNARY_CASES = [
    ("mish", layers.mish),
    ("swish", layers.swish),
]


@pytest.mark.parametrize("name,fn", DEVICE_UNARY_CASES)
def test_activation_runs_on_directml(name, fn, dml_device):
    x = torch.randn(7, 13, device=dml_device, requires_grad=True)
    out = fn(x)
    out.square().sum().backward()
    assert _finite(out.detach())
    assert x.grad is not None and _finite(x.grad)


def test_mish_gradient_stays_finite_at_extreme_input_on_directml(dml_device):
    """Regression guard for a real production bug, not a hypothetical one.

    torch.randn inputs (the case above) never exceed ~4 in magnitude, so
    they cannot exercise this: DirectML's softplus backward returns 0 (not
    ~1) past x~88 and NaN past x~89 -- consistent with a kernel computing
    1/(1+exp(x)) unconditionally, overflowing exp(x) right where float32
    does. mish() is protected by the global
    layers.safe_directml_softplus monkey-patch, which linearizes softplus
    past layers.SOFTPLUS_SAFE_MAX to stay out of that region. This silently
    corrupted a real training run's moves-left head gradient with nothing
    else in the step non-finite anywhere describe_non_finite could see.
    """
    x = torch.tensor([200.0, -5.0, 0.0], device=dml_device, requires_grad=True)
    out = layers.mish(x)
    out.sum().backward()
    assert _finite(out.detach())
    assert x.grad is not None and _finite(x.grad)
    # The linearization must not change the answer for the large positive
    # case: mish(200) is indistinguishable from 200 itself, and its gradient
    # from 1.0, well before the threshold point.
    assert abs(float(x.grad[0].cpu()) - 1.0) < 1e-3


@pytest.mark.parametrize("module_factory", [layers.LayerNorm, layers.RmsNorm])
def test_norm_runs_on_directml(module_factory, dml_device):
    features = 16
    module = module_factory(features).to(dml_device)
    x = torch.randn(5, features, device=dml_device, requires_grad=True)
    out = module(x)
    out.square().sum().backward()
    assert _finite(out.detach())
    assert x.grad is not None and _finite(x.grad)


def test_ffn_runs_on_directml(dml_device):
    ffn = layers.Ffn(
        32, 48, net_pb2.NetworkFormat.ACTIVATION_MISH, layers.deepnorm_beta(4)
    ).to(dml_device)
    x = torch.randn(64, 32, device=dml_device, requires_grad=True)
    out = ffn(x)
    out.square().sum().backward()
    assert _finite(out.detach())
    assert x.grad is not None and _finite(x.grad)


def test_layer_norm_matches_cpu_on_directml(dml_device):
    """The manual layer norm must agree with its own CPU result.

    This is the regression guard for the built-in F.layer_norm backward
    failing with "tensor does not have a device" on DirectML.
    """
    features = 16
    module = layers.LayerNorm(features)
    x = torch.randn(5, features)
    expected = module(x).detach().numpy()
    actual = module.to(dml_device)(x.to(dml_device)).detach().cpu().numpy()
    _assert_close(actual, expected, rtol=1e-4, atol=1e-5)


def test_identity_matrix_runs_on_directml(dml_device):
    """Regression guard for torch.eye's broken DirectML fallback path."""
    actual = layers.identity_matrix(8, dtype=torch.float32, device=dml_device)
    _assert_close(actual.cpu().numpy(), np.eye(8, dtype=np.float32))


def test_gather_along_runs_on_directml(dml_device):
    rng = np.random.default_rng(11)
    x = torch.randn(2, 64, 8, device=dml_device, requires_grad=True)
    order = torch.from_numpy(rng.permutation(64).astype(np.int64))
    out = layers.gather_along(x, 1, order.to(dml_device))
    out.square().sum().backward()
    assert _finite(out.detach())
    assert x.grad is not None and _finite(x.grad)


def test_permute_along_runs_on_directml(dml_device):
    """Must produce the same gradient as the CPU scatter-add path."""
    rng = np.random.default_rng(14)
    x = rng.normal(size=(2, 64, 8)).astype(np.float32)
    weights = torch.from_numpy(rng.normal(size=(2, 64, 8)).astype(np.float32))
    order = torch.from_numpy(rng.permutation(64).astype(np.int64))
    inverse = torch.argsort(order)

    reference = torch.from_numpy(x).requires_grad_(True)
    (layers.gather_along(reference, 1, order) * weights).sum().backward()

    actual = torch.from_numpy(x).to(dml_device).requires_grad_(True)
    out = layers.permute_along(
        actual, 1, order.to(dml_device), inverse.to(dml_device)
    )
    (out * weights.to(dml_device)).sum().backward()

    assert _finite(out.detach())
    _assert_close(
        actual.grad.cpu().numpy(), reference.grad.numpy(), rtol=1e-5, atol=1e-6
    )


@pytest.mark.parametrize("dim", [-1, -2, 2, 3])
def test_safe_cumsum_backward_runs_on_directml(dim, dml_device):
    """Negative-dim regression guard.

    torch.cumsum(dim=-1).backward() hard-crashes the process on DirectML,
    because cumsum's gradient flips along the recorded dim and DirectML's
    flip kernel faults on a negative axis. layers.cumsum normalizes first.
    """
    x = torch.randn(2, 3, 8, 5, device=dml_device, requires_grad=True)
    out = layers.cumsum(x, dim)
    out.sum().backward()
    assert _finite(out.detach())
    assert x.grad is not None and _finite(x.grad)


@pytest.mark.parametrize("dim", [-1, -2, 1, 2])
def test_safe_flip_runs_on_directml(dim, dml_device):
    """Negative-dim regression guard for torch.flip itself."""
    x = torch.randn(2, 3, 8, device=dml_device)
    out = layers.flip(x, dim)
    expected = torch.flip(x.cpu(), [dim])
    _assert_close(out.cpu().numpy(), expected.numpy())
