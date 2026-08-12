"""Tests for the PyTorch/DirectML KDA mixer.

Covers the same three legs as test_layers.py: PyTorch CPU against the JAX
reference, gradients where practical, and DirectML execution with finite
output and gradients.
"""

import numpy as np
import pytest

torch = pytest.importorskip("torch")
jax = pytest.importorskip("jax")

import jax.numpy as jnp
from flax import nnx

from lczero_training.directml import layers
from lczero_training.directml.kda import (
    KDA_LOG_DECAY_FLOOR,
    KDA_TRAVERSALS,
    KdaLocalConv,
    KDALogDecay,
    KdaMixer,
    kda_recurrence,
)
from lczero_training.model import kda as jax_kda
from proto import model_config_pb2

ALL_DIRECTIONS = [
    "rank_forward",
    "rank_reverse",
    "file_forward",
    "file_reverse",
    "diag_forward",
    "diag_reverse",
    "anti_diag_forward",
    "anti_diag_reverse",
]


def _assert_close(actual, expected, rtol=1e-4, atol=1e-5):
    np.testing.assert_allclose(
        np.asarray(actual, dtype=np.float64),
        np.asarray(expected, dtype=np.float64),
        rtol=rtol,
        atol=atol,
    )


# --------------------------------------------------------------------------
# Recurrence (the Phase 0 prototype)
# --------------------------------------------------------------------------


def _reference_recurrence(
    query: np.ndarray,
    key: np.ndarray,
    value: np.ndarray,
    log_decay: np.ndarray,
    beta: np.ndarray,
) -> np.ndarray:
    """Independent sequential NumPy reference, one token at a time."""
    query = query / np.maximum(
        np.linalg.norm(query, axis=-1, keepdims=True), 1e-12
    )
    key = key / np.maximum(np.linalg.norm(key, axis=-1, keepdims=True), 1e-12)
    batch, tokens, heads, key_dim = query.shape
    value_dim = value.shape[-1]
    state = np.zeros((batch, heads, key_dim, value_dim), dtype=np.float32)
    output = np.zeros((batch, tokens, heads, value_dim), dtype=np.float32)
    scale = np.float32(1.0 / np.sqrt(key_dim))

    for token in range(tokens):
        state *= np.exp(log_decay[:, token])[..., None]
        prediction = np.einsum("bhk,bhkv->bhv", key[:, token], state)
        delta = beta[:, token, :, None] * (value[:, token] - prediction)
        state += np.einsum("bhk,bhv->bhkv", key[:, token], delta)
        output[:, token] = np.einsum(
            "bhk,bhkv->bhv", query[:, token] * scale, state
        )
    return output


@pytest.mark.parametrize("chunk_size", [4, 8, 16])
def test_recurrence_matches_reference(chunk_size: int) -> None:
    rng = np.random.default_rng(7)
    query = rng.normal(size=(2, 17, 2, 3)).astype(np.float32)
    key = rng.normal(size=(2, 17, 2, 3)).astype(np.float32)
    value = rng.normal(size=(2, 17, 2, 4)).astype(np.float32)
    log_decay = -rng.uniform(0.001, 2.0, size=query.shape).astype(np.float32)
    beta = rng.uniform(0.0, 1.0, size=query.shape[:-1]).astype(np.float32)

    actual = kda_recurrence(
        torch.from_numpy(query),
        torch.from_numpy(key),
        torch.from_numpy(value),
        torch.from_numpy(log_decay),
        torch.from_numpy(beta),
        chunk_size,
    )
    expected = _reference_recurrence(query, key, value, log_decay, beta)
    np.testing.assert_allclose(
        actual.detach().numpy(), expected, rtol=1e-4, atol=1e-5
    )


def test_recurrence_matches_jax() -> None:
    """Batched PyTorch against the unbatched JAX reference, per element."""
    rng = np.random.default_rng(21)
    query = rng.normal(size=(3, 64, 4, 8)).astype(np.float32)
    key = rng.normal(size=(3, 64, 4, 8)).astype(np.float32)
    value = rng.normal(size=(3, 64, 4, 8)).astype(np.float32)
    log_decay = -rng.uniform(0.001, 2.0, size=query.shape).astype(np.float32)
    beta = rng.uniform(0.0, 1.0, size=query.shape[:-1]).astype(np.float32)

    actual = kda_recurrence(
        torch.from_numpy(query),
        torch.from_numpy(key),
        torch.from_numpy(value),
        torch.from_numpy(log_decay),
        torch.from_numpy(beta),
        chunk_size=16,
    ).numpy()

    expected = np.stack(
        [
            np.asarray(
                jax_kda.kda_recurrence(
                    jnp.asarray(query[i]),
                    jnp.asarray(key[i]),
                    jnp.asarray(value[i]),
                    jnp.asarray(log_decay[i]),
                    jnp.asarray(beta[i]),
                    chunk_size=16,
                )
            )
            for i in range(query.shape[0])
        ]
    )
    _assert_close(actual, expected)


def test_recurrence_gradients_are_finite() -> None:
    query = torch.randn(2, 17, 2, 3, requires_grad=True)
    key = torch.randn(2, 17, 2, 3, requires_grad=True)
    value = torch.randn(2, 17, 2, 4, requires_grad=True)
    log_decay = -torch.rand(2, 17, 2, 3)
    beta = torch.sigmoid(torch.randn(2, 17, 2))

    output = kda_recurrence(query, key, value, log_decay, beta, chunk_size=8)
    output.square().mean().backward()

    for tensor in (query, key, value):
        assert tensor.grad is not None
        assert bool(torch.isfinite(tensor.grad).all())


@pytest.mark.parametrize("chunk_size", [4, 8, 16])
@pytest.mark.parametrize("saturated_fraction", [0.25, 0.5, 1.0])
def test_recurrence_is_finite_with_a_saturated_gate(
    saturated_fraction: float, chunk_size: int
) -> None:
    """The forget gate sitting on its floor is a routine input, not a corner.

    Live runs report 6-24% of gate elements pinned at KDA_LOG_DECAY_FLOOR,
    and a channel saturated across a whole chunk drives the within-chunk
    cumulative sum to chunk_size * -10.

    Regression test for two separate breaks in the factored-decay form,
    which replaces exp(cum[i] - cum[j]) with exp(cum[i]) * exp(-cum[j]):

    * Written as `key / exp(cumulative)` the forward pass is right, but
      division backpropagates as -grad * key / exp(cumulative)**2 and
      exp(-80)**2 underflows float32 to zero, so the log_decay gradient
      became -inf. The multiply has the same value and a finite derivative.
    * The factored form itself only holds while exp(chunk_size * 10) fits
      in the dtype. float32 tops out at exp(88.7), so chunk_size 16 --
      the default -- overflowed to inf, and the following matmul turned
      that into 0 * inf = NaN inside entries the causal mask keeps. The
      whole network went to NaN. kda_recurrence now falls back to pairwise
      differences when the chunk size makes the split unsafe.

    Both needed conditions the shipped suites never combined: log_decay is
    drawn from [-2, -0.001] elsewhere, so `cumulative` never approached the
    floor, and test_recurrence_gradients_are_finite does not set
    requires_grad on log_decay at all.
    """
    generator = torch.Generator().manual_seed(11)
    shape = (2, 16, 2, 8)
    query = torch.randn(shape, generator=generator, requires_grad=True)
    key = torch.randn(shape, generator=generator, requires_grad=True)
    value = torch.randn((2, 16, 2, 8), generator=generator, requires_grad=True)
    beta = torch.sigmoid(torch.randn((2, 16, 2), generator=generator))

    decay = -torch.rand(shape, generator=generator) * 2.0 - 0.001
    saturated = torch.rand(shape, generator=generator) < saturated_fraction
    decay = torch.where(
        saturated, torch.full_like(decay, KDA_LOG_DECAY_FLOOR), decay
    )
    log_decay = decay.clone().requires_grad_(True)

    output = kda_recurrence(
        query, key, value, log_decay, beta, chunk_size=chunk_size
    )
    assert bool(torch.isfinite(output).all()), (
        f"forward produced non-finite at chunk_size {chunk_size}"
    )

    output.square().mean().backward()

    for name, tensor in (
        ("query", query),
        ("key", key),
        ("value", value),
        ("log_decay", log_decay),
    ):
        assert tensor.grad is not None, f"{name} got no gradient"
        assert bool(torch.isfinite(tensor.grad).all()), (
            f"{name} gradient is not finite at "
            f"{saturated_fraction:.0%} gate saturation"
        )


# --------------------------------------------------------------------------
# Traversal tables
# --------------------------------------------------------------------------


def test_traversal_tables_match_jax_reference():
    """The duplicated tables must not drift from model/kda.py."""
    assert KDA_TRAVERSALS == jax_kda.KDA_TRAVERSALS


def test_every_traversal_is_a_permutation_of_the_board():
    for name, order in KDA_TRAVERSALS.items():
        assert sorted(order) == list(range(64)), name


def test_log_decay_floor_matches_jax_reference():
    from lczero_training.directml.kda import KDA_LOG_DECAY_FLOOR

    assert KDA_LOG_DECAY_FLOOR == jax_kda.KDA_LOG_DECAY_FLOOR


# --------------------------------------------------------------------------
# KDALogDecay
# --------------------------------------------------------------------------


def test_log_decay_matches_jax():
    heads, key_dim = 8, 32
    rng = np.random.default_rng(22)
    raw = rng.normal(size=(3, 64, heads, key_dim)).astype(np.float32)

    torch_decay = KDALogDecay(heads, key_dim)
    actual = torch_decay(torch.from_numpy(raw)).detach().numpy()

    jax_decay = jax_kda.KDALogDecay(heads, key_dim, rngs=nnx.Rngs(0))
    expected = np.stack(
        [
            np.asarray(jax_decay(jnp.asarray(raw[i])))
            for i in range(raw.shape[0])
        ]
    )
    _assert_close(actual, expected)


def test_log_decay_initializers_match_jax():
    """The parameters are deterministic, so they must match exactly."""
    heads, key_dim = 8, 32
    torch_decay = KDALogDecay(heads, key_dim)
    jax_decay = jax_kda.KDALogDecay(heads, key_dim, rngs=nnx.Rngs(0))
    _assert_close(
        torch_decay.a_log.detach().numpy(),
        np.asarray(jax_decay.a_log.value),
        rtol=1e-6,
        atol=1e-7,
    )
    _assert_close(
        torch_decay.dt_bias.detach().numpy(),
        np.asarray(jax_decay.dt_bias.value),
        rtol=1e-6,
        atol=1e-7,
    )


def test_log_decay_gradient_matches_jax():
    heads, key_dim = 4, 8
    rng = np.random.default_rng(23)
    raw = rng.normal(size=(1, 64, heads, key_dim)).astype(np.float32)

    torch_decay = KDALogDecay(heads, key_dim)
    tensor = torch.from_numpy(raw).requires_grad_(True)
    torch_decay(tensor).square().sum().backward()

    jax_decay = jax_kda.KDALogDecay(heads, key_dim, rngs=nnx.Rngs(0))

    def loss(value):
        return jnp.sum(jnp.square(jax_decay(value)))

    expected = jax.grad(loss)(jnp.asarray(raw[0]))
    _assert_close(tensor.grad.numpy()[0], expected)


# --------------------------------------------------------------------------
# KdaLocalConv
# --------------------------------------------------------------------------


def _copy_conv_to_jax(torch_conv: KdaLocalConv, jax_conv) -> None:
    # torch weight is (out, in/groups, kh, kw); the flax kernel is
    # (kh, kw, in/groups, out).
    weight = torch_conv.conv.weight.detach().numpy()
    jax_conv.conv.kernel.value = jnp.asarray(weight.transpose(2, 3, 1, 0))
    jax_conv.conv.bias.value = jnp.asarray(
        torch_conv.conv.bias.detach().numpy()
    )


def test_local_conv_matches_jax():
    emb = 16
    rng = np.random.default_rng(24)
    x = rng.normal(size=(3, 64, emb)).astype(np.float32)

    torch_conv = KdaLocalConv(emb)
    jax_conv = jax_kda.KdaLocalConv(emb, rngs=nnx.Rngs(0))
    _copy_conv_to_jax(torch_conv, jax_conv)

    actual = torch_conv(torch.from_numpy(x)).detach().numpy()
    expected = np.stack(
        [np.asarray(jax_conv(jnp.asarray(x[i]))) for i in range(x.shape[0])]
    )
    _assert_close(actual, expected)


def test_local_conv_keeps_the_residual_skip():
    """The conv adds to its input; it must never replace it."""
    emb = 8
    torch_conv = KdaLocalConv(emb)
    with torch.no_grad():
        torch_conv.conv.weight.zero_()
        torch_conv.conv.bias.zero_()
    x = torch.randn(2, 64, emb)
    _assert_close(torch_conv(x).detach().numpy(), x.numpy())


# --------------------------------------------------------------------------
# KdaMixer
# --------------------------------------------------------------------------


def _make_config(**overrides) -> model_config_pb2.KdaConfig:
    kwargs = {
        "key_dim": 8,
        "value_dim": 8,
        "gate_rank": 8,
        "directions": ALL_DIRECTIONS,
        "output_gate": True,
        "output_rms_norm": False,
        "local_conv": True,
        "chunk_size": 16,
    }
    kwargs.update(overrides)
    return model_config_pb2.KdaConfig(**kwargs)


def _copy_linear_to_jax(torch_linear, jax_linear) -> None:
    jax_linear.kernel.value = jnp.asarray(
        torch_linear.weight.detach().numpy().T
    )
    jax_linear.bias.value = jnp.asarray(torch_linear.bias.detach().numpy())


def _make_matched_mixers(config, in_features, heads, seed):
    """A torch KdaMixer and a JAX KdaMixer holding identical weights."""
    beta = layers.deepnorm_beta(4)
    torch_mixer = KdaMixer(
        in_features=in_features,
        config=config,
        heads=heads,
        deepnorm_beta=beta,
    )
    jax_mixer = jax_kda.KdaMixer(
        in_features=in_features,
        config=config,
        heads=heads,
        deepnorm_beta=beta,
        rngs=nnx.Rngs(seed),
    )
    for name in ("q", "k", "v", "decay_a", "decay_b", "beta", "output_dense"):
        _copy_linear_to_jax(
            getattr(torch_mixer, name), getattr(jax_mixer, name)
        )
    if config.output_gate:
        _copy_linear_to_jax(torch_mixer.gate_a, jax_mixer.gate_a)
        _copy_linear_to_jax(torch_mixer.gate_b, jax_mixer.gate_b)
    if config.local_conv:
        _copy_conv_to_jax(torch_mixer.local_conv, jax_mixer.local_conv)
    if config.output_rms_norm:
        jax_mixer.rms_norm_gammas.value = jnp.asarray(
            torch_mixer.rms_norm.scale.detach().numpy()
        )
    jax_mixer.log_decay.a_log.value = jnp.asarray(
        torch_mixer.log_decay.a_log.detach().numpy()
    )
    jax_mixer.log_decay.dt_bias.value = jnp.asarray(
        torch_mixer.log_decay.dt_bias.detach().numpy()
    )
    return torch_mixer, jax_mixer


@pytest.mark.parametrize("local_conv", [True, False])
@pytest.mark.parametrize("output_gate", [True, False])
@pytest.mark.parametrize("output_rms_norm", [True, False])
def test_mixer_matches_jax(local_conv, output_gate, output_rms_norm):
    in_features, heads = 16, 8
    config = _make_config(
        local_conv=local_conv,
        output_gate=output_gate,
        output_rms_norm=output_rms_norm,
    )
    torch_mixer, jax_mixer = _make_matched_mixers(
        config, in_features, heads, seed=31
    )

    rng = np.random.default_rng(31)
    x = rng.normal(size=(3, 64, in_features)).astype(np.float32)

    actual = torch_mixer(torch.from_numpy(x)).detach().numpy()
    expected = np.stack(
        [np.asarray(jax_mixer(jnp.asarray(x[i]))) for i in range(x.shape[0])]
    )
    _assert_close(actual, expected, rtol=1e-4, atol=1e-5)


@pytest.mark.parametrize("directions", [1, 2, 4, 8])
def test_mixer_matches_jax_for_each_direction_count(directions):
    """Head-group slicing must line up for every legal direction count."""
    in_features, heads = 16, 8
    config = _make_config(directions=ALL_DIRECTIONS[:directions])
    torch_mixer, jax_mixer = _make_matched_mixers(
        config, in_features, heads, seed=32
    )

    rng = np.random.default_rng(32)
    x = rng.normal(size=(2, 64, in_features)).astype(np.float32)

    actual = torch_mixer(torch.from_numpy(x)).detach().numpy()
    expected = np.stack(
        [np.asarray(jax_mixer(jnp.asarray(x[i]))) for i in range(x.shape[0])]
    )
    _assert_close(actual, expected, rtol=1e-4, atol=1e-5)


def test_mixer_gradient_matches_jax():
    in_features, heads = 16, 8
    config = _make_config()
    torch_mixer, jax_mixer = _make_matched_mixers(
        config, in_features, heads, seed=33
    )

    rng = np.random.default_rng(33)
    x = rng.normal(size=(1, 64, in_features)).astype(np.float32)

    tensor = torch.from_numpy(x).requires_grad_(True)
    torch_mixer(tensor).square().sum().backward()

    def loss(value):
        return jnp.sum(jnp.square(jax_mixer(value)))

    expected = jax.grad(loss)(jnp.asarray(x[0]))
    _assert_close(tensor.grad.numpy()[0], expected, rtol=1e-3, atol=1e-5)


def test_mixer_scan_permutation_is_a_permutation():
    config = _make_config()
    mixer = KdaMixer(in_features=16, config=config, heads=8, deepnorm_beta=0.5)
    order = mixer.scan_order.numpy()
    inverse = mixer.scan_inverse.numpy()
    assert sorted(order) == list(range(64 * 8))
    np.testing.assert_array_equal(order[inverse], np.arange(64 * 8))


def test_mixer_rank_forward_only_is_the_identity_permutation():
    """A single rank_forward direction must not reorder anything."""
    config = _make_config(directions=["rank_forward"])
    mixer = KdaMixer(in_features=16, config=config, heads=8, deepnorm_beta=0.5)
    np.testing.assert_array_equal(mixer.scan_order.numpy(), np.arange(64 * 8))


# --------------------------------------------------------------------------
# DirectML leg
# --------------------------------------------------------------------------


def test_log_decay_gradient_stays_finite_at_extreme_input_on_directml(
    dml_device,
):
    """Regression guard for a real production bug, not a hypothetical one.

    KDALogDecay feeds an unbounded learned projection into softplus.
    DirectML's softplus backward returns 0 (not ~1) past x~88 and NaN past
    x~89 instead of the correct ~1 either side -- consistent with a kernel
    computing 1/(1+exp(x)) unconditionally, overflowing exp(x) right where
    float32 does. forward() now clamps the softplus argument to stay out of
    that region (see layers.SOFTPLUS_SAFE_MAX). A batch that pushes
    raw_decay this far is exactly the kind of rare-but-real input that
    silently corrupted a real training run's gradients with nothing else in
    the step non-finite anywhere describe_non_finite could see.
    """
    heads, key_dim = 4, 8
    decay = KDALogDecay(heads, key_dim).to(dml_device)
    raw_decay = torch.full(
        (1, 1, heads, key_dim), 300.0, device=dml_device, dtype=torch.float32
    )
    raw_decay.requires_grad_(True)
    out = decay(raw_decay)
    out.sum().backward()

    assert bool(torch.isfinite(out.detach()).all().cpu())
    assert raw_decay.grad is not None
    assert bool(torch.isfinite(raw_decay.grad).all().cpu())
    assert bool(torch.isfinite(decay.a_log.grad).all().cpu())
    assert bool(torch.isfinite(decay.dt_bias.grad).all().cpu())


def test_mixer_runs_on_directml(dml_device):
    in_features, heads = 16, 8
    config = _make_config()
    mixer = KdaMixer(
        in_features=in_features,
        config=config,
        heads=heads,
        deepnorm_beta=layers.deepnorm_beta(4),
    ).to(dml_device)

    x = torch.randn(2, 64, in_features, device=dml_device, requires_grad=True)
    out = mixer(x)
    out.square().sum().backward()

    assert bool(torch.isfinite(out.detach()).all().cpu())
    assert x.grad is not None and bool(torch.isfinite(x.grad).all().cpu())
    for name, parameter in mixer.named_parameters():
        assert parameter.grad is not None, name
        assert bool(torch.isfinite(parameter.grad).all().cpu()), name


def test_mixer_directml_matches_cpu(dml_device):
    in_features, heads = 16, 8
    config = _make_config()
    mixer = KdaMixer(
        in_features=in_features,
        config=config,
        heads=heads,
        deepnorm_beta=layers.deepnorm_beta(4),
    )

    rng = np.random.default_rng(41)
    x = torch.from_numpy(
        rng.normal(size=(2, 64, in_features)).astype(np.float32)
    )
    expected = mixer(x).detach().numpy()
    actual = mixer.to(dml_device)(x.to(dml_device)).detach().cpu().numpy()
    _assert_close(actual, expected, rtol=1e-3, atol=1e-4)
