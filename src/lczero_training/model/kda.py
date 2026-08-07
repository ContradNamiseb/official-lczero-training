"""Kimi Delta Attention (KDA) mixer.

Ported from the Keras reference in
stable-branch/tf/tfprocess.py:107-300 (KDA_TRAVERSALS, KDALogDecay,
KDARecurrence) and :2108-2204 (the kda() method that wires them together).

Unlike the rest of this model, none of this module's tensors carry a batch
dimension -- LczeroModel operates on a single (64, features) position and is
batched externally via vmap (see model/model.py). Every reshape here is the
TF reference's reshape with the leading batch axis deleted; see
docs/implementation_plan (Blocker: "The model is unbatched") for why
transcribing the TF shapes directly would be wrong.
"""

from typing import Optional

import jax
import jax.numpy as jnp
import numpy as np
from flax import nnx
from flax.linen import initializers as flax_initializers

from proto import model_config_pb2

# Board traversal orders for the recurrence. A chess position is not a
# causal sequence, so heads are split evenly across these and each group
# scans the 64 squares in its own order. Byte-identical to KDA_TRAVERSALS in
# stable-branch/tf/tfprocess.py:107-140, and to the SYCL engine's
# kKdaDiag*/kKdaAntiDiag* tables and the BLAS backend's KdaSquareForToken --
# do not regenerate.
KDA_TRAVERSALS = {
    "rank_forward": tuple(range(64)),
    "rank_reverse": tuple(reversed(range(64))),
    "file_forward": tuple(
        rank * 8 + file for file in range(8) for rank in range(8)
    ),
    "file_reverse": tuple(
        reversed(
            tuple(rank * 8 + file for file in range(8) for rank in range(8))
        )
    ),
    # Main diagonals (a1-h8 style, rank - file constant), grouped from the
    # a8 corner to the h1 corner and walked bottom-to-top within each one.
    "diag_forward": tuple(
        rank * 8 + file
        for diag in range(-7, 8)
        for rank in range(8)
        for file in [rank - diag]
        if 0 <= file < 8
    ),
    "diag_reverse": tuple(
        reversed(
            tuple(
                rank * 8 + file
                for diag in range(-7, 8)
                for rank in range(8)
                for file in [rank - diag]
                if 0 <= file < 8
            )
        )
    ),
    # Anti-diagonals (a8-h1 style, rank + file constant).
    "anti_diag_forward": tuple(
        rank * 8 + file
        for diag in range(15)
        for rank in range(8)
        for file in [diag - rank]
        if 0 <= file < 8
    ),
    "anti_diag_reverse": tuple(
        reversed(
            tuple(
                rank * 8 + file
                for diag in range(15)
                for rank in range(8)
                for file in [diag - rank]
                if 0 <= file < 8
            )
        )
    ),
}

# Per-token log decay floor. Retaining exp(-10) of the state per token is
# already indistinguishable from total forgetting, so lc0's unclamped
# sequential recurrence still matches.
KDA_LOG_DECAY_FLOOR = -10.0


class KDALogDecay(nnx.Module):
    """Owns the per-head/per-channel forget-gate parameters.

    Mirrors KDALogDecay in stable-branch/tf/tfprocess.py:152-179.
    """

    def __init__(self, heads: int, key_dim: int, *, rngs: nnx.Rngs):
        del rngs  # Deterministic initializers; nothing here is random.
        a_log_init = np.log(np.linspace(1.0, 16.0, heads, dtype=np.float32))[
            :, None
        ]
        initial_dt = np.exp(
            np.linspace(np.log(0.001), np.log(0.1), key_dim, dtype=np.float32)
        )
        dt_bias_init = np.log(np.expm1(initial_dt))[None, :]
        dt_bias_init = np.repeat(dt_bias_init, heads, axis=0)
        self.a_log = nnx.Param(jnp.asarray(a_log_init))  # (heads, 1)
        self.dt_bias = nnx.Param(jnp.asarray(dt_bias_init))  # (heads, key)

    def __call__(self, raw_decay: jax.Array) -> jax.Array:
        # raw_decay: (64, heads, key_dim) -- no batch dim.
        raw_decay = raw_decay.astype(jnp.float32)
        a_log = self.a_log.value.astype(jnp.float32)
        dt_bias = self.dt_bias.value.astype(jnp.float32)
        log_decay = -jnp.exp(a_log)[None, :, :] * jax.nn.softplus(
            raw_decay + dt_bias[None, :, :]
        )
        return jnp.maximum(log_decay, KDA_LOG_DECAY_FLOOR)


def kda_recurrence(
    q: jax.Array,
    k: jax.Array,
    v: jax.Array,
    log_decay: jax.Array,
    beta: jax.Array,
    chunk_size: int,
) -> jax.Array:
    """Chunkwise-parallel form of the gated delta rule.

    Equivalent to scanning the recurrence one token at a time, but each
    chunk is solved in closed form with dense matmuls, so the sequential
    depth drops from `tokens` to `tokens / chunk_size`.

    This is a plain function, not an nnx.Module: KDARecurrence in the TF
    reference holds no trainable weights (it is a Keras Layer purely for
    graph tracing), so there is nothing here that should look like model
    state.

    Mirrors KDARecurrence.call() in stable-branch/tf/tfprocess.py:182-299.
    All arguments are unbatched: q/k/log_decay are (tokens, heads, key_dim),
    v is (tokens, heads, value_dim), beta is (tokens, heads). `chunk_size`
    is a concrete Python int (from config, never traced), so the two
    Python-level loops below unroll at trace time exactly as the TF
    reference's do -- do not replace the causal-masking loop with a
    vectorized form: a vectorized rewrite that computes the full
    chunk x chunk square (instead of only the valid triangular entries) was
    benchmarked ~2x slower on CPU (see docs/implementation_plan).
    """
    q = q.astype(jnp.float32)
    k = k.astype(jnp.float32)
    # tf.math.l2_normalize floors the *squared* norm before the sqrt, not
    # the norm itself -- match that exactly rather than jnp's usual
    # normalize-then-floor idiom, since this governs how saturated decay
    # interacts with near-zero query/key vectors.
    q = q / jnp.sqrt(
        jnp.maximum(jnp.sum(jnp.square(q), axis=-1, keepdims=True), 1e-12)
    )
    k = k / jnp.sqrt(
        jnp.maximum(jnp.sum(jnp.square(k), axis=-1, keepdims=True), 1e-12)
    )
    v = v.astype(jnp.float32)
    log_decay = jnp.maximum(log_decay.astype(jnp.float32), KDA_LOG_DECAY_FLOOR)
    beta = beta.astype(jnp.float32)[..., None]  # (tokens, heads, 1)

    tokens, heads, key_dim = q.shape
    value_dim = v.shape[-1]
    chunk = chunk_size
    padding = -tokens % chunk
    if padding:
        # Zero keys and betas make padded tokens leave the state untouched.
        pad3 = [(0, padding), (0, 0), (0, 0)]
        q = jnp.pad(q, pad3)
        k = jnp.pad(k, pad3)
        v = jnp.pad(v, pad3)
        log_decay = jnp.pad(log_decay, pad3)
        beta = jnp.pad(beta, pad3)
    chunks = (tokens + padding) // chunk

    def to_chunks(x: jax.Array, depth: int) -> jax.Array:
        x = x.reshape((chunks, chunk, heads, depth))
        return x.transpose((0, 2, 1, 3))  # (chunks, heads, chunk, depth)

    q = to_chunks(q, key_dim)
    k = to_chunks(k, key_dim)
    v = to_chunks(v, value_dim)
    log_decay = to_chunks(log_decay, key_dim)
    beta = to_chunks(beta, 1)

    cumulative = jnp.cumsum(log_decay, axis=2)
    final = cumulative[:, :, -1:, :]
    decayed_q = q * jnp.exp(cumulative)
    decayed_k = k * jnp.exp(cumulative)
    trailing_k = k * jnp.exp(final - cumulative)

    key_attn_rows = [jnp.zeros((chunks, heads, 1, chunk), jnp.float32)]
    query_attn_rows = []
    for row in range(chunk):
        cumulative_row = cumulative[:, :, row : row + 1, :]

        query_keys = k[:, :, : row + 1, :]
        query_decay = jnp.exp(cumulative_row - cumulative[:, :, : row + 1, :])
        query_row = jnp.sum(
            q[:, :, row : row + 1, :] * query_keys * query_decay, axis=-1
        )
        query_row = jnp.pad(query_row, [(0, 0), (0, 0), (0, chunk - row - 1)])
        query_attn_rows.append(query_row[:, :, None, :])

        if row:
            key_columns = k[:, :, :row, :]
            key_decay = jnp.exp(cumulative_row - cumulative[:, :, :row, :])
            key_row = jnp.sum(
                k[:, :, row : row + 1, :] * key_columns * key_decay, axis=-1
            )
            key_row = jnp.pad(key_row, [(0, 0), (0, 0), (0, chunk - row)])
            key_attn_rows.append(key_row[:, :, None, :])

    key_attn = jnp.concatenate(key_attn_rows, axis=2)
    query_attn = jnp.concatenate(query_attn_rows, axis=2)

    # Invert the unit lower triangular (I + diag(beta) @ key_attn) with a
    # Neumann series; key_attn is strictly lower triangular, so it is
    # nilpotent and the series is exact after `chunk` terms.
    nilpotent = -beta * key_attn
    identity = jnp.eye(chunk, dtype=jnp.float32)
    inverse = identity + nilpotent
    power = nilpotent
    span = 2
    while span < chunk:
        power = power @ power
        inverse = inverse @ (identity + power)
        span *= 2

    beta_v = beta * v
    beta_k = beta * decayed_k
    final_decay = jnp.transpose(jnp.exp(final), (0, 1, 3, 2))

    state = jnp.zeros((heads, key_dim, value_dim), jnp.float32)
    outputs = []
    for index in range(chunks):
        delta = inverse[index] @ (beta_v[index] - beta_k[index] @ state)
        outputs.append(decayed_q[index] @ state + query_attn[index] @ delta)
        state = (
            state * final_decay[index]
            + jnp.swapaxes(trailing_k[index], -1, -2) @ delta
        )

    outputs_arr = jnp.stack(outputs, axis=0) * jax.lax.rsqrt(
        jnp.asarray(key_dim, jnp.float32)
    )
    outputs_arr = outputs_arr.transpose((0, 2, 1, 3))
    outputs_arr = outputs_arr.reshape((chunks * chunk, heads, value_dim))
    return outputs_arr[:tokens]


class KdaLocalConv(nnx.Module):
    """Local 3x3 depthwise board convolution before the KDA projections.

    Mirrors the `if self.kda_local_conv:` block in
    stable-branch/tf/tfprocess.py:2116-2125.
    """

    def __init__(self, emb_size: int, *, rngs: nnx.Rngs):
        self.conv = nnx.Conv(
            in_features=emb_size,
            out_features=emb_size,
            kernel_size=(3, 3),
            padding="SAME",
            feature_group_count=emb_size,  # depthwise
            rngs=rngs,
        )

    def __call__(self, x: jax.Array) -> jax.Array:
        # x: (64, emb_size) -- no batch dim. Squares are token-major
        # (rank * 8 + file), matching rank_forward, so this reshape lines
        # up with the board without any gather.
        grid = x.reshape((1, 8, 8, x.shape[-1]))
        conv_out = self.conv(grid).reshape(x.shape)
        # The caller must keep the *original* x as the encoder block's
        # residual skip -- only the projection input is convolved. Do not
        # return conv_out on its own or let it escape KdaMixer; that is the
        # exact bug that had to be fixed in the SYCL engine (conv applied
        # in place on the tensor that was also the LN1 residual skip).
        return x + conv_out


class KdaMixer(nnx.Module):
    """Kimi Delta Attention mixer: an 8-direction board scan through a
    chunkwise-parallel gated delta rule, with an optional local 3x3
    depthwise conv and an optional sigmoid output gate.

    Mirrors the kda() method in stable-branch/tf/tfprocess.py:2108-2204.
    """

    def __init__(
        self,
        *,
        in_features: int,
        config: model_config_pb2.KdaConfig,
        heads: int,
        deepnorm_beta: float,
        rngs: nnx.Rngs,
    ):
        directions = list(config.directions)
        assert directions, "KdaConfig.directions must not be empty."
        assert heads % len(directions) == 0, (
            "encoder heads must be evenly divisible by len(directions)."
        )

        self.heads = heads
        self.key_dim = config.key_dim
        self.value_dim = config.value_dim
        self.gate_rank = config.gate_rank
        self.chunk_size = config.chunk_size
        self.output_gate = config.output_gate
        self.output_rms_norm = config.output_rms_norm

        key_depth = heads * self.key_dim
        value_depth = heads * self.value_dim

        deepnorm_init = flax_initializers.variance_scaling(
            scale=deepnorm_beta,
            mode="fan_avg",
            distribution="truncated_normal",
        )

        self.local_conv: Optional[KdaLocalConv]
        if config.local_conv:
            self.local_conv = KdaLocalConv(in_features, rngs=rngs)
        else:
            self.local_conv = None

        self.q = nnx.Linear(
            in_features=in_features, out_features=key_depth, rngs=rngs
        )
        self.k = nnx.Linear(
            in_features=in_features, out_features=key_depth, rngs=rngs
        )
        self.v = nnx.Linear(
            in_features=in_features,
            out_features=value_depth,
            kernel_init=deepnorm_init,
            rngs=rngs,
        )

        self.decay_a = nnx.Linear(
            in_features=in_features,
            out_features=self.gate_rank,
            kernel_init=deepnorm_init,
            rngs=rngs,
        )
        self.decay_b = nnx.Linear(
            in_features=self.gate_rank,
            out_features=key_depth,
            kernel_init=deepnorm_init,
            rngs=rngs,
        )
        self.log_decay = KDALogDecay(heads, self.key_dim, rngs=rngs)

        self.beta = nnx.Linear(
            in_features=in_features,
            out_features=heads,
            kernel_init=deepnorm_init,
            rngs=rngs,
        )

        self.gate_a: Optional[nnx.Linear]
        self.gate_b: Optional[nnx.Linear]
        if self.output_gate:
            self.gate_a = nnx.Linear(
                in_features=in_features,
                out_features=self.gate_rank,
                kernel_init=deepnorm_init,
                rngs=rngs,
            )
            self.gate_b = nnx.Linear(
                in_features=self.gate_rank,
                out_features=value_depth,
                kernel_init=deepnorm_init,
                rngs=rngs,
            )
        else:
            self.gate_a = None
            self.gate_b = None

        self.rms_norm_gammas: Optional[nnx.Param]
        if self.output_rms_norm:
            self.rms_norm_gammas = nnx.Param(jnp.ones((value_depth,)))
        else:
            self.rms_norm_gammas = None

        self.output_dense = nnx.Linear(
            in_features=value_depth,
            out_features=in_features,
            kernel_init=deepnorm_init,
            rngs=rngs,
        )

        heads_per_direction = heads // len(directions)
        self.head_slices = [
            slice(
                index * heads_per_direction, (index + 1) * heads_per_direction
            )
            for index in range(len(directions))
        ]
        # Plain tuples of Python ints, not numpy arrays: these are
        # traversal-order lookup tables baked in at construction time, not
        # trainable state. Flax NNX (>=0.12) rejects a bare numpy array
        # stored on a non-nnx.data attribute of an nnx.Module -- it wants
        # every array-shaped attribute to be an explicit pytree leaf (see
        # nnx.split/nnx.value_and_grad in the tests). A tuple of plain ints
        # isn't array-shaped, so it stays outside the pytree entirely,
        # which is what a static, never-updated permutation table should
        # do (and keeps it out of the state dict the export visitor walks).
        self.orders = [tuple(KDA_TRAVERSALS[d]) for d in directions]
        self.inv_orders = [
            tuple(np.argsort(np.asarray(order)).tolist())
            for order in self.orders
        ]

    def __call__(self, x: jax.Array) -> jax.Array:
        # x: (64, in_features) -- no batch dim.
        proj_input = x
        if self.local_conv is not None:
            proj_input = self.local_conv(x)

        q = self.q(proj_input).reshape((64, self.heads, self.key_dim))
        k = self.k(proj_input).reshape((64, self.heads, self.key_dim))
        v = self.v(proj_input).reshape((64, self.heads, self.value_dim))

        raw_decay = self.decay_b(self.decay_a(proj_input))
        raw_decay = raw_decay.reshape((64, self.heads, self.key_dim))
        log_decay = self.log_decay(raw_decay)

        beta = jax.nn.sigmoid(self.beta(proj_input))  # (64, heads)

        # All directions share one recurrence call: each head group is
        # permuted into its own traversal order first, so the scan runs
        # once, not once per direction.
        def gather_by_direction(t: jax.Array) -> jax.Array:
            # order/inv_order are plain int tuples (see __init__), and
            # indexing a JAX/numpy array with a bare tuple of ints means
            # "one index per axis" (arr[i0, i1, ...]), not fancy indexing
            # along axis 0 -- wrap in jnp.asarray to get the gather this
            # is actually meant to perform.
            return jnp.concatenate(
                [
                    t[:, group, ...][jnp.asarray(order)]
                    for group, order in zip(self.head_slices, self.orders)
                ],
                axis=1,
            )

        ordered_out = kda_recurrence(
            gather_by_direction(q),
            gather_by_direction(k),
            gather_by_direction(v),
            gather_by_direction(log_decay),
            gather_by_direction(beta),
            chunk_size=self.chunk_size,
        )

        mixed = jnp.concatenate(
            [
                ordered_out[:, group, :][jnp.asarray(inv_order)]
                for group, inv_order in zip(self.head_slices, self.inv_orders)
            ],
            axis=1,
        )
        mixed = mixed.reshape((64, self.heads * self.value_dim))

        if self.output_rms_norm:
            assert self.rms_norm_gammas is not None
            variance = jnp.mean(jnp.square(mixed), axis=-1, keepdims=True)
            mixed = mixed * jax.lax.rsqrt(variance + 1e-6)
            mixed = mixed * self.rms_norm_gammas.value

        # The gate must apply after RMSNorm, not before -- they do not
        # commute, since the gate rescales each element before the norm
        # would divide by the resulting RMS. This ordering bug was found
        # and fixed in the SYCL engine; match its corrected order here.
        if self.output_gate:
            assert self.gate_a is not None and self.gate_b is not None
            gate = self.gate_b(self.gate_a(proj_input))
            mixed = mixed * jax.nn.sigmoid(gate).astype(mixed.dtype)

        return self.output_dense(mixed)
