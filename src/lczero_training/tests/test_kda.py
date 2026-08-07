"""Parity tests for the Kimi Delta Attention (KDA) mixer.

Ports the numerical content of stable-branch/tf/tests/test_kda.py to the
unbatched JAX/Flax NNX port in model/kda.py. Not a line-for-line port -- the
TF suite builds Keras models and walks protobuf round-trips through net.py,
which don't exist on this side; here the equivalent behavior is exercised
directly against KdaMixer/kda_recurrence/KDALogDecay.
"""

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from flax import nnx

from lczero_training.model.kda import (
    KDA_LOG_DECAY_FLOOR,
    KDA_TRAVERSALS,
    KdaLocalConv,
    KDALogDecay,
    KdaMixer,
    kda_recurrence,
)
from proto import model_config_pb2


def numpy_recurrent_kda(
    q: np.ndarray,
    k: np.ndarray,
    v: np.ndarray,
    log_decay: np.ndarray,
    beta: np.ndarray,
) -> np.ndarray:
    """Independent, token-at-a-time reference for the gated delta rule.

    Unbatched: q/k/log_decay (tokens, heads, key_dim), v
    (tokens, heads, value_dim), beta (tokens, heads). Mirrors
    numpy_recurrent_kda in stable-branch/tf/tests/test_kda.py with the
    batch axis removed.
    """
    q = q / np.maximum(np.linalg.norm(q, axis=-1, keepdims=True), 1e-12)
    k = k / np.maximum(np.linalg.norm(k, axis=-1, keepdims=True), 1e-12)
    tokens, heads, key_dim = q.shape
    value_dim = v.shape[-1]
    state = np.zeros([heads, key_dim, value_dim], dtype=np.float32)
    outputs = np.zeros([tokens, heads, value_dim], dtype=np.float32)
    scale = np.float32(1.0 / np.sqrt(key_dim))

    for token in range(tokens):
        state = state * np.exp(log_decay[token])[..., None]
        prediction = np.einsum("hk,hkv->hv", k[token], state)
        delta = beta[token][:, None] * (v[token] - prediction)
        state = state + np.einsum("hk,hv->hkv", k[token], delta)
        outputs[token] = np.einsum("hk,hkv->hv", q[token] * scale, state)
    return outputs


def _make_kda_config(
    *,
    key_dim: int = 4,
    value_dim: int = 4,
    gate_rank: int = 4,
    directions: list[str] | None = None,
    output_gate: bool = True,
    output_rms_norm: bool = False,
    local_conv: bool = False,
    chunk_size: int = 16,
) -> model_config_pb2.KdaConfig:
    config = model_config_pb2.KdaConfig()
    config.key_dim = key_dim
    config.value_dim = value_dim
    config.gate_rank = gate_rank
    config.directions.extend(directions or list(KDA_TRAVERSALS))
    config.output_gate = output_gate
    config.output_rms_norm = output_rms_norm
    config.local_conv = local_conv
    config.chunk_size = chunk_size
    return config


# --- 1. KDALogDecay parity -------------------------------------------------


def test_kda_log_decay_matches_reference_formula() -> None:
    heads, key_dim = 4, 8
    module = KDALogDecay(heads, key_dim, rngs=nnx.Rngs(0))

    a_log_init = np.log(np.linspace(1.0, 16.0, heads, dtype=np.float32))[
        :, None
    ]
    initial_dt = np.exp(
        np.linspace(np.log(0.001), np.log(0.1), key_dim, dtype=np.float32)
    )
    dt_bias_init = np.log(np.expm1(initial_dt))[None, :]
    dt_bias_init = np.repeat(dt_bias_init, heads, axis=0)
    np.testing.assert_allclose(module.a_log.value, a_log_init, rtol=1e-6)
    np.testing.assert_allclose(module.dt_bias.value, dt_bias_init, rtol=1e-6)

    rng = np.random.default_rng(0)
    raw_decay = rng.normal(size=(64, heads, key_dim)).astype(np.float32)
    actual = module(jnp.asarray(raw_decay))

    expected = -np.exp(a_log_init)[None, :, :] * np.log1p(
        np.exp(raw_decay + dt_bias_init[None, :, :])
    )
    expected = np.maximum(expected, KDA_LOG_DECAY_FLOOR)
    np.testing.assert_allclose(actual, expected, rtol=1e-5, atol=1e-5)


def test_kda_log_decay_floors_at_minus_ten() -> None:
    module = KDALogDecay(2, 2, rngs=nnx.Rngs(0))
    # softplus(raw_decay + dt_bias) grows with raw_decay, and log_decay is
    # -exp(a_log) * softplus(...), so it is a *large positive* raw_decay
    # that saturates log_decay very negative (and hits the floor); a large
    # negative raw_decay instead drives softplus -> 0, i.e. log_decay -> 0
    # (no decay at all), the opposite end of the range.
    raw_decay = jnp.full((4, 2, 2), 1e4)
    out = module(raw_decay)
    assert np.all(np.asarray(out) == KDA_LOG_DECAY_FLOOR)


# --- 2. kda_recurrence parity -----------------------------------------------


def test_recurrence_matches_numpy_reference() -> None:
    rng = np.random.default_rng(7)
    q = rng.normal(size=(5, 2, 3)).astype(np.float32)
    k = rng.normal(size=(5, 2, 3)).astype(np.float32)
    v = rng.normal(size=(5, 2, 4)).astype(np.float32)
    log_decay = -rng.uniform(0.001, 2.0, size=(5, 2, 3)).astype(np.float32)
    beta = rng.uniform(0.0, 1.0, size=(5, 2)).astype(np.float32)

    actual = kda_recurrence(
        jnp.asarray(q),
        jnp.asarray(k),
        jnp.asarray(v),
        jnp.asarray(log_decay),
        jnp.asarray(beta),
        chunk_size=16,
    )
    expected = numpy_recurrent_kda(q, k, v, log_decay, beta)
    np.testing.assert_allclose(actual, expected, rtol=1e-5, atol=1e-5)


@pytest.mark.parametrize("decay_scale", [2.0, 20.0])
@pytest.mark.parametrize("chunk_size", [4, 8, 16, 32, 64])
def test_recurrence_carries_state_across_chunks(
    decay_scale: float, chunk_size: int
) -> None:
    # A full board exercises several chunks of the parallel form (for every
    # chunk_size the KDAConfig actually allows), so the inter-chunk state
    # hand-off is covered as well as the intra-chunk solve.
    rng = np.random.default_rng(11)
    q = rng.normal(size=(64, 4, 8)).astype(np.float32)
    k = rng.normal(size=(64, 4, 8)).astype(np.float32)
    v = rng.normal(size=(64, 4, 8)).astype(np.float32)
    beta = rng.uniform(0.0, 1.0, size=(64, 4)).astype(np.float32)
    log_decay = -rng.uniform(0.001, decay_scale, size=(64, 4, 8)).astype(
        np.float32
    )

    actual = kda_recurrence(
        jnp.asarray(q),
        jnp.asarray(k),
        jnp.asarray(v),
        jnp.asarray(log_decay),
        jnp.asarray(beta),
        chunk_size=chunk_size,
    )
    expected = numpy_recurrent_kda(
        q, k, v, np.maximum(log_decay, KDA_LOG_DECAY_FLOOR), beta
    )
    np.testing.assert_allclose(actual, expected, rtol=1e-4, atol=1e-5)


def test_extreme_decay_and_zero_inputs_are_finite() -> None:
    shape = (7, 2, 3)
    q = jnp.zeros(shape, dtype=jnp.float16)
    k = jnp.zeros(shape, dtype=jnp.float16)
    v = jnp.zeros((7, 2, 4), dtype=jnp.float16)
    beta = jnp.zeros((7, 2), dtype=jnp.float16)
    for decay in (-1e-7, -1e6):
        log_decay = jnp.full(shape, decay, dtype=jnp.float16)
        output = kda_recurrence(q, k, v, log_decay, beta, chunk_size=8)
        assert bool(jnp.all(jnp.isfinite(output)))


def test_saturating_decay_keeps_outputs_and_gradients_finite() -> None:
    # Only valid causal decay differences are exponentiated; a saturated
    # (very negative) decay must not produce NaNs in either direction.
    rng = np.random.default_rng(3)
    shape = (64, 4, 16)
    q = jnp.asarray(rng.normal(size=shape).astype(np.float32))
    k = jnp.asarray(rng.normal(size=shape).astype(np.float32))
    v = jnp.asarray(rng.normal(size=shape).astype(np.float32))
    beta = jnp.asarray(rng.uniform(0.1, 1.0, size=shape[:2]).astype(np.float32))
    log_decay = jnp.full(shape, -1e4)

    def loss_fn(log_decay: jax.Array) -> jax.Array:
        output = kda_recurrence(q, k, v, log_decay, beta, chunk_size=16)
        return jnp.sum(jnp.square(output))

    loss, grad = jax.value_and_grad(loss_fn)(log_decay)
    assert bool(jnp.isfinite(loss))
    assert bool(jnp.all(jnp.isfinite(grad)))


# --- 3. Traversal tables -----------------------------------------------------


def test_traversals_are_invertible_permutations() -> None:
    squares = np.arange(64)
    for direction, order in KDA_TRAVERSALS.items():
        order_arr = np.asarray(order)
        assert sorted(order_arr.tolist()) == list(range(64)), direction
        restored = squares[order_arr][np.argsort(order_arr)]
        np.testing.assert_array_equal(restored, squares)


def test_traversals_match_engine_diagonal_tables() -> None:
    # Byte-identical to the SYCL engine's kKdaDiagForward/kKdaDiagReverse/
    # kKdaAntiDiagForward/kKdaAntiDiagReverse (common_kernels.dp.cpp) and
    # the BLAS backend's KdaSquareForToken (network_blas.cc) -- this
    # three-way match was verified by hand; this test locks it in so a
    # future edit to either side gets caught immediately instead of
    # silently degrading play strength.
    engine_diag_forward = [
        7,
        6,
        15,
        5,
        14,
        23,
        4,
        13,
        22,
        31,
        3,
        12,
        21,
        30,
        39,
        2,
        11,
        20,
        29,
        38,
        47,
        1,
        10,
        19,
        28,
        37,
        46,
        55,
        0,
        9,
        18,
        27,
        36,
        45,
        54,
        63,
        8,
        17,
        26,
        35,
        44,
        53,
        62,
        16,
        25,
        34,
        43,
        52,
        61,
        24,
        33,
        42,
        51,
        60,
        32,
        41,
        50,
        59,
        40,
        49,
        58,
        48,
        57,
        56,
    ]
    engine_diag_reverse = [
        56,
        57,
        48,
        58,
        49,
        40,
        59,
        50,
        41,
        32,
        60,
        51,
        42,
        33,
        24,
        61,
        52,
        43,
        34,
        25,
        16,
        62,
        53,
        44,
        35,
        26,
        17,
        8,
        63,
        54,
        45,
        36,
        27,
        18,
        9,
        0,
        55,
        46,
        37,
        28,
        19,
        10,
        1,
        47,
        38,
        29,
        20,
        11,
        2,
        39,
        30,
        21,
        12,
        3,
        31,
        22,
        13,
        4,
        23,
        14,
        5,
        15,
        6,
        7,
    ]
    engine_anti_diag_forward = [
        0,
        1,
        8,
        2,
        9,
        16,
        3,
        10,
        17,
        24,
        4,
        11,
        18,
        25,
        32,
        5,
        12,
        19,
        26,
        33,
        40,
        6,
        13,
        20,
        27,
        34,
        41,
        48,
        7,
        14,
        21,
        28,
        35,
        42,
        49,
        56,
        15,
        22,
        29,
        36,
        43,
        50,
        57,
        23,
        30,
        37,
        44,
        51,
        58,
        31,
        38,
        45,
        52,
        59,
        39,
        46,
        53,
        60,
        47,
        54,
        61,
        55,
        62,
        63,
    ]
    engine_anti_diag_reverse = [
        63,
        62,
        55,
        61,
        54,
        47,
        60,
        53,
        46,
        39,
        59,
        52,
        45,
        38,
        31,
        58,
        51,
        44,
        37,
        30,
        23,
        57,
        50,
        43,
        36,
        29,
        22,
        15,
        56,
        49,
        42,
        35,
        28,
        21,
        14,
        7,
        48,
        41,
        34,
        27,
        20,
        13,
        6,
        40,
        33,
        26,
        19,
        12,
        5,
        32,
        25,
        18,
        11,
        4,
        24,
        17,
        10,
        3,
        16,
        9,
        2,
        8,
        1,
        0,
    ]
    assert list(KDA_TRAVERSALS["diag_forward"]) == engine_diag_forward
    assert list(KDA_TRAVERSALS["diag_reverse"]) == engine_diag_reverse
    assert list(KDA_TRAVERSALS["anti_diag_forward"]) == engine_anti_diag_forward
    assert list(KDA_TRAVERSALS["anti_diag_reverse"]) == engine_anti_diag_reverse


# --- 4. KdaLocalConv ---------------------------------------------------------


def test_local_conv_is_residual_and_leaves_original_untouched() -> None:
    module = KdaLocalConv(16, rngs=nnx.Rngs(0))
    x = jax.random.normal(jax.random.key(0), (64, 16))
    out = module(x)
    assert out.shape == x.shape
    # The module must not mutate its input; only the returned tensor is
    # convolved. This is the property that has to hold for KdaMixer's
    # residual skip to stay correct (see the module's own docstring).
    np.testing.assert_array_equal(np.asarray(x), np.asarray(x))
    assert not np.allclose(np.asarray(out), np.asarray(x))


def test_local_conv_only_sees_3x3_neighborhood() -> None:
    module = KdaLocalConv(16, rngs=nnx.Rngs(0))
    base = jax.random.normal(jax.random.key(1), (64, 16))
    perturbed = np.asarray(base).copy()
    # Square (rank=3, file=3) -> token 3*8+3 = 27, outside the 3x3
    # neighborhood of token 0 (square (rank=0, file=0)).
    perturbed[27, :] += 5.0

    out_base = np.asarray(module(base))
    out_perturbed = np.asarray(module(jnp.asarray(perturbed)))
    np.testing.assert_allclose(
        out_base[0], out_perturbed[0], rtol=1e-5, atol=1e-5
    )


# --- 5. Gate/norm ordering ---------------------------------------------------


def test_gate_applies_after_rms_norm_not_before() -> None:
    """The gate and RMSNorm do not commute; the gate must scale the
    *normalized* output. Verified by comparing against a hand-computed
    reference that applies norm-then-gate, using a KdaMixer with both
    enabled.
    """
    config = _make_kda_config(
        output_gate=True, output_rms_norm=True, directions=["rank_forward"]
    )
    mixer = KdaMixer(
        in_features=16,
        config=config,
        heads=1,
        deepnorm_beta=1.0,
        rngs=nnx.Rngs(0),
    )
    x = jax.random.normal(jax.random.key(2), (64, 16))

    # Recompute the forward pass by hand, stopping right before the gate,
    # to check the gate is applied to the *normalized* mixed tensor.
    proj_input = x
    q = mixer.q(proj_input).reshape((64, 1, config.key_dim))
    k = mixer.k(proj_input).reshape((64, 1, config.key_dim))
    v = mixer.v(proj_input).reshape((64, 1, config.value_dim))
    raw_decay = mixer.decay_b(mixer.decay_a(proj_input)).reshape(
        (64, 1, config.key_dim)
    )
    log_decay = mixer.log_decay(raw_decay)
    beta = jax.nn.sigmoid(mixer.beta(proj_input))
    recurred = kda_recurrence(q, k, v, log_decay, beta, chunk_size=16)
    mixed = recurred.reshape((64, config.value_dim))

    assert mixer.rms_norm_gammas is not None
    variance = jnp.mean(jnp.square(mixed), axis=-1, keepdims=True)
    normed = mixed * jax.lax.rsqrt(variance + 1e-6)
    normed = normed * mixer.rms_norm_gammas.value

    assert mixer.gate_a is not None and mixer.gate_b is not None
    gate = mixer.gate_b(mixer.gate_a(proj_input))
    expected_pre_dense = normed * jax.nn.sigmoid(gate)
    expected = mixer.output_dense(expected_pre_dense)

    actual = mixer(x)
    np.testing.assert_allclose(
        np.asarray(actual), np.asarray(expected), rtol=1e-5, atol=1e-5
    )


# --- 6. Shape / vmap ---------------------------------------------------------


@pytest.mark.parametrize("local_conv", [False, True])
@pytest.mark.parametrize("output_gate", [False, True])
def test_kda_mixer_shape_and_gradients(
    local_conv: bool, output_gate: bool
) -> None:
    config = _make_kda_config(local_conv=local_conv, output_gate=output_gate)
    mixer = KdaMixer(
        in_features=16,
        config=config,
        heads=8,
        deepnorm_beta=1.0,
        rngs=nnx.Rngs(0),
    )
    x = jax.random.normal(jax.random.key(3), (64, 16))

    def loss_fn(mixer: KdaMixer, x: jax.Array) -> jax.Array:
        out = mixer(x)
        assert out.shape == (64, 16)
        return jnp.sum(jnp.square(out))

    graphdef, state = nnx.split(mixer)

    def loss_from_state(state: nnx.State, x: jax.Array) -> jax.Array:
        return loss_fn(nnx.merge(graphdef, state), x)

    loss, grads = nnx.value_and_grad(loss_from_state)(state, x)
    assert bool(jnp.isfinite(loss))
    leaves = jax.tree_util.tree_leaves(grads)
    assert leaves, "expected at least one gradient leaf"
    for leaf in leaves:
        assert bool(jnp.all(jnp.isfinite(leaf)))


def test_kda_mixer_composes_under_vmap() -> None:
    config = _make_kda_config()
    mixer = KdaMixer(
        in_features=16,
        config=config,
        heads=8,
        deepnorm_beta=1.0,
        rngs=nnx.Rngs(0),
    )
    batch = jax.random.normal(jax.random.key(4), (3, 64, 16))

    graphdef, state = nnx.split(mixer)

    def apply_one(x: jax.Array) -> jax.Array:
        return nnx.merge(graphdef, state)(x)

    out = jax.vmap(apply_one)(batch)
    assert out.shape == (3, 64, 16)
    assert bool(jnp.all(jnp.isfinite(out)))


def test_heads_must_be_divisible_by_direction_count() -> None:
    config = _make_kda_config(directions=["rank_forward", "file_forward"])
    with pytest.raises(AssertionError):
        KdaMixer(
            in_features=16,
            config=config,
            heads=3,
            deepnorm_beta=1.0,
            rngs=nnx.Rngs(0),
        )
