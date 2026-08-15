"""PyTorch/DirectML implementation of the KDA mixer.

Phase 4 of docs/directml_training_port.md. Mirrors the JAX reference in
model/kda.py, with one difference that runs through the whole file: every
tensor here carries a native leading batch dimension. The JAX model is
unbatched and vmapped externally, so each shape below is the JAX shape with
one extra axis in front.

This module deliberately does not import jax or flax -- the DirectML
trainer must run in an environment that has neither. The traversal tables
are duplicated from model/kda.py rather than imported for that reason;
test_kda.py asserts the two copies stay identical.
"""

from __future__ import annotations

import math

import numpy as np
import torch
from torch import nn

from proto import model_config_pb2

from . import layers

# Board traversal orders for the recurrence. A chess position is not a
# causal sequence, so heads are split evenly across these and each group
# scans the 64 squares in its own order. Must stay byte-identical to
# KDA_TRAVERSALS in model/kda.py, to KDA_TRAVERSALS in the TF reference,
# and to the SYCL engine's kKdaDiag*/kKdaAntiDiag* tables -- do not
# regenerate.
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

BOARD_SQUARES = 64


def kda_recurrence(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    log_decay: torch.Tensor,
    beta: torch.Tensor,
    chunk_size: int,
) -> torch.Tensor:
    """Apply the batched chunkwise KDA recurrence.

    Query, key, and log decay have shape ``(batch, tokens, heads, key_dim)``.
    Value has shape ``(batch, tokens, heads, value_dim)`` and beta has shape
    ``(batch, tokens, heads)``.

    Equivalent to scanning the recurrence one token at a time, but each
    chunk is solved in closed form with dense matmuls, so the sequential
    depth drops from ``tokens`` to ``tokens / chunk_size``. Mirrors
    kda_recurrence() in model/kda.py; ``chunk_size`` is a concrete Python
    int so both loops below unroll exactly as the JAX reference's do.
    """
    query = query.float()
    key = key.float()
    # tf.math.l2_normalize floors the *squared* norm before the sqrt, not
    # the norm itself -- match that exactly, since this governs how
    # saturated decay interacts with near-zero query/key vectors.
    query = query / torch.sqrt(
        torch.clamp(torch.square(query).sum(dim=-1, keepdim=True), min=1e-12)
    )
    key = key / torch.sqrt(
        torch.clamp(torch.square(key).sum(dim=-1, keepdim=True), min=1e-12)
    )
    value = value.float()
    log_decay = torch.clamp(log_decay.float(), min=KDA_LOG_DECAY_FLOOR)
    beta = beta.float().unsqueeze(-1)

    batch, tokens, heads, key_dim = query.shape
    value_dim = value.shape[-1]
    padding = -tokens % chunk_size
    if padding:
        # Zero keys and betas make padded tokens leave the state untouched.
        pad_shape = (batch, padding, heads)
        query = torch.cat(
            [query, query.new_zeros((*pad_shape, key_dim))], dim=1
        )
        key = torch.cat([key, key.new_zeros((*pad_shape, key_dim))], dim=1)
        value = torch.cat(
            [value, value.new_zeros((*pad_shape, value_dim))], dim=1
        )
        log_decay = torch.cat(
            [log_decay, log_decay.new_zeros((*pad_shape, key_dim))], dim=1
        )
        beta = torch.cat([beta, beta.new_zeros((*pad_shape, 1))], dim=1)
    chunks = (tokens + padding) // chunk_size

    def to_chunks(tensor: torch.Tensor, depth: int) -> torch.Tensor:
        tensor = tensor.reshape(batch, chunks, chunk_size, heads, depth)
        return tensor.permute(0, 1, 3, 2, 4)

    query = to_chunks(query, key_dim)
    key = to_chunks(key, key_dim)
    value = to_chunks(value, value_dim)
    log_decay = to_chunks(log_decay, key_dim)
    beta = to_chunks(beta, 1)

    # dim=3 is the within-chunk token axis. Routed through layers.cumsum so
    # the axis is normalized: cumsum's backward flips along the recorded
    # dim, and DirectML's flip faults on a negative axis.
    cumulative = layers.cumsum(log_decay, 3)
    final = cumulative[:, :, :, -1:, :]
    decayed_query = query * torch.exp(cumulative)
    decayed_key = key * torch.exp(cumulative)
    trailing_key = key * torch.exp(final - cumulative)

    key_attention_rows = [
        query.new_zeros((batch, chunks, heads, 1, chunk_size))
    ]
    query_attention_rows = []
    for row in range(chunk_size):
        cumulative_row = cumulative[:, :, :, row : row + 1, :]
        query_keys = key[:, :, :, : row + 1, :]
        query_decay = torch.exp(
            cumulative_row - cumulative[:, :, :, : row + 1, :]
        )
        query_row = torch.sum(
            query[:, :, :, row : row + 1, :] * query_keys * query_decay,
            dim=-1,
        )
        query_row = layers.pad_last(query_row, chunk_size - row - 1)
        query_attention_rows.append(query_row.unsqueeze(-2))

        if row:
            key_columns = key[:, :, :, :row, :]
            key_decay = torch.exp(cumulative_row - cumulative[:, :, :, :row, :])
            key_row = torch.sum(
                key[:, :, :, row : row + 1, :] * key_columns * key_decay,
                dim=-1,
            )
            key_row = layers.pad_last(key_row, chunk_size - row)
            key_attention_rows.append(key_row.unsqueeze(-2))

    key_attention = torch.cat(key_attention_rows, dim=3)
    query_attention = torch.cat(query_attention_rows, dim=3)

    # Invert the unit lower triangular (I + diag(beta) @ key_attention) with
    # a Neumann series; key_attention is strictly lower triangular, so it is
    # nilpotent and the series is exact after `chunk_size` terms.
    nilpotent = -beta * key_attention
    identity = layers.identity_matrix(
        chunk_size, dtype=query.dtype, device=query.device
    )
    inverse = identity + nilpotent
    power = nilpotent
    span = 2
    while span < chunk_size:
        power = power @ power
        inverse = inverse @ (identity + power)
        span *= 2

    beta_value = beta * value
    beta_key = beta * decayed_key
    final_decay = torch.exp(final).transpose(-1, -2)

    state = query.new_zeros((batch, heads, key_dim, value_dim))
    outputs = []
    for index in range(chunks):
        delta = inverse[:, index] @ (
            beta_value[:, index] - beta_key[:, index] @ state
        )
        outputs.append(
            decayed_query[:, index] @ state + query_attention[:, index] @ delta
        )
        state = (
            state * final_decay[:, index]
            + trailing_key[:, index].transpose(-1, -2) @ delta
        )

    output = torch.stack(outputs, dim=1) / math.sqrt(key_dim)
    output = output.permute(0, 1, 3, 2, 4)
    output = output.reshape(batch, chunks * chunk_size, heads, value_dim)
    return output[:, :tokens]


class KDALogDecay(nn.Module):
    """Owns the per-head/per-channel forget-gate parameters.

    Mirrors KDALogDecay in model/kda.py. Both initializers are
    deterministic, so nothing here consumes an RNG.
    """

    def __init__(self, heads: int, key_dim: int):
        super().__init__()
        a_log_init = np.log(np.linspace(1.0, 16.0, heads, dtype=np.float32))[
            :, None
        ]
        initial_dt = np.exp(
            np.linspace(np.log(0.001), np.log(0.1), key_dim, dtype=np.float32)
        )
        dt_bias_init = np.log(np.expm1(initial_dt))[None, :]
        dt_bias_init = np.repeat(dt_bias_init, heads, axis=0)
        self.a_log = nn.Parameter(torch.from_numpy(a_log_init))  # (heads, 1)
        self.dt_bias = nn.Parameter(
            torch.from_numpy(dt_bias_init)
        )  # (heads, key_dim)

    def forward(self, raw_decay: torch.Tensor) -> torch.Tensor:
        # raw_decay: (batch, tokens, heads, key_dim). a_log and dt_bias
        # broadcast over batch and tokens.
        raw_decay = raw_decay.float()
        # No per-call clamp: torch.nn.functional.softplus is monkey-patched
        # globally in layers.safe_directml_softplus to linearize past
        # layers.SOFTPLUS_SAFE_MAX, which keeps DirectML's exp backward out
        # of the overflow region for this unbounded pre-activation too.
        log_decay = -torch.exp(
            self.a_log.float()
        ) * torch.nn.functional.softplus(raw_decay + self.dt_bias.float())
        return torch.clamp(log_decay, min=KDA_LOG_DECAY_FLOOR)


class KdaLocalConv(nn.Module):
    """Local 3x3 depthwise board convolution before the KDA projections.

    Mirrors KdaLocalConv in model/kda.py. flax stores the board as NHWC and
    PyTorch's Conv2d wants NCHW, so this permutes in and back out; the
    kernel layout conversion itself is the weight importer's job.
    """

    def __init__(self, emb_size: int):
        super().__init__()
        self.conv = nn.Conv2d(
            in_channels=emb_size,
            out_channels=emb_size,
            kernel_size=3,
            padding=1,
            groups=emb_size,  # depthwise
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (batch, 64, emb_size). Squares are token-major (rank * 8 +
        # file), matching rank_forward, so this reshape lines up with the
        # board without any gather.
        batch, tokens, emb = x.shape
        grid = x.reshape(batch, 8, 8, emb).permute(0, 3, 1, 2)
        conv_out = (
            self.conv(grid).permute(0, 2, 3, 1).reshape(batch, tokens, emb)
        )
        # The caller must keep the *original* x as the encoder block's
        # residual skip -- only the projection input is convolved. Do not
        # return conv_out on its own or let it escape KdaMixer; that is the
        # exact bug that had to be fixed in the SYCL engine.
        return x + conv_out


class KdaMixer(nn.Module):
    """Kimi Delta Attention mixer: an 8-direction board scan through a
    chunkwise-parallel gated delta rule, with an optional local 3x3
    depthwise conv and an optional sigmoid output gate.

    Mirrors KdaMixer in model/kda.py, batched.
    """

    def __init__(
        self,
        *,
        in_features: int,
        config: model_config_pb2.KdaConfig,
        heads: int,
        deepnorm_beta: float,
    ):
        super().__init__()
        directions = list(config.directions)
        assert directions, "KdaConfig.directions must not be empty."
        assert heads % len(directions) == 0, (
            "encoder heads must be evenly divisible by len(directions)."
        )

        self.heads = heads
        self.directions = directions
        self.heads_per_direction = heads // len(directions)
        self.key_dim = config.key_dim
        self.value_dim = config.value_dim
        self.gate_rank = config.gate_rank
        self.chunk_size = config.chunk_size
        self.output_gate = config.output_gate
        self.output_rms_norm = config.output_rms_norm

        key_depth = heads * self.key_dim
        value_depth = heads * self.value_dim

        self.local_conv: KdaLocalConv | None
        if config.local_conv:
            self.local_conv = KdaLocalConv(in_features)
        else:
            self.local_conv = None

        self.q = nn.Linear(in_features, key_depth)
        self.k = nn.Linear(in_features, key_depth)
        self.v = nn.Linear(in_features, value_depth)
        layers.init_lecun_normal_(self.q)
        layers.init_lecun_normal_(self.k)
        layers.init_variance_scaling_(self.v, deepnorm_beta)

        self.decay_a = nn.Linear(in_features, self.gate_rank)
        self.decay_b = nn.Linear(self.gate_rank, key_depth)
        layers.init_variance_scaling_(self.decay_a, deepnorm_beta)
        layers.init_variance_scaling_(self.decay_b, deepnorm_beta)
        self.log_decay = KDALogDecay(heads, self.key_dim)

        self.beta = nn.Linear(in_features, heads)
        layers.init_variance_scaling_(self.beta, deepnorm_beta)

        self.gate_a: nn.Linear | None
        self.gate_b: nn.Linear | None
        if self.output_gate:
            self.gate_a = nn.Linear(in_features, self.gate_rank)
            self.gate_b = nn.Linear(self.gate_rank, value_depth)
            layers.init_variance_scaling_(self.gate_a, deepnorm_beta)
            layers.init_variance_scaling_(self.gate_b, deepnorm_beta)
        else:
            self.gate_a = None
            self.gate_b = None

        self.rms_norm: layers.RmsNorm | None
        if self.output_rms_norm:
            self.rms_norm = layers.RmsNorm(value_depth)
        else:
            self.rms_norm = None

        self.output_dense = nn.Linear(value_depth, in_features)
        layers.init_variance_scaling_(self.output_dense, deepnorm_beta)

        # Off by default: when enabled, forward() stashes detached summaries
        # of the recurrence's internal gates for diagnostics. The training
        # loop turns it on only for reporting steps.
        self.collect_stats = False
        self.last_stats: dict[str, torch.Tensor] = {}

        # A SEPARATE flag from collect_stats, and deliberately never set by
        # the training loop: this one stashes the *undetached* value
        # projection and mixer output (board-square order, pre rms_norm/
        # gate/output_dense), so an interpretability tool can backprop
        # through them afterwards (see chessformer_lens's LczeroEngine.
        # kda_attention -- the recurrence is linear in value for fixed
        # query/key/decay/beta, so d(mixed)/d(value) is an exact,
        # non-approximate "effective attention matrix", the same identity
        # real softmax attention weights satisfy). Keeping this off the
        # collect_stats path matters: collect_stats already runs on every
        # real training report step, and holding an undetached tensor (and
        # the graph behind it) alive there would be a production memory
        # regression for a feature training never uses.
        self.collect_attention_target = False
        self.last_value: torch.Tensor | None = None
        self.last_mixed: torch.Tensor | None = None

        scan_order, scan_inverse = _build_scan_permutation(directions, heads)
        # Buffers, not parameters: static lookup tables that .to(device)
        # must follow the module but that no optimizer may ever touch.
        self.register_buffer("scan_order", scan_order, persistent=False)
        self.register_buffer("scan_inverse", scan_inverse, persistent=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (batch, 64, in_features).
        batch = x.shape[0]
        proj_input = x if self.local_conv is None else self.local_conv(x)

        shape = (batch, BOARD_SQUARES, self.heads, -1)
        query = self.q(proj_input).reshape(shape)
        key = self.k(proj_input).reshape(shape)
        value = self.v(proj_input).reshape(shape)
        if self.collect_attention_target:
            # Undetached, deliberately: kda_attention() backprops through
            # this into `last_mixed` below to get an exact Jacobian.
            self.last_value = value

        raw_decay = self.decay_b(self.decay_a(proj_input)).reshape(shape)
        log_decay = self.log_decay(raw_decay)

        beta = torch.sigmoid(self.beta(proj_input))  # (batch, 64, heads)

        if self.collect_stats:
            with torch.no_grad():
                clamped = log_decay <= KDA_LOG_DECAY_FLOOR + 1e-6
                self.last_stats = {
                    # Per-token log retention. More negative forgets faster;
                    # pinned at the floor means the state is wiped each token
                    # and the recurrence has stopped carrying information.
                    "log decay mean": log_decay.mean(),
                    "decay saturated %": clamped.float().mean() * 100.0,
                    # Delta-rule write strength. Near 0 means the mixer has
                    # stopped writing to its state at all.
                    "beta mean": beta.mean(),
                }

        # All directions share one recurrence call: each head group is
        # permuted into its own traversal order first, so the scan runs
        # once, not once per direction. Packing the five tensors along the
        # last axis turns 5 x len(directions) gathers into a single one --
        # this matters on DirectML, where every eager kernel launch costs.
        packed = torch.cat(
            [query, key, value, log_decay, beta.unsqueeze(-1)], dim=-1
        )
        ordered = self._scan_permute(packed, forward=True)
        query_o, key_o, value_o, decay_o, beta_o = torch.split(
            ordered,
            [self.key_dim, self.key_dim, self.value_dim, self.key_dim, 1],
            dim=-1,
        )

        ordered_out = kda_recurrence(
            query_o,
            key_o,
            value_o,
            decay_o,
            beta_o.squeeze(-1),
            chunk_size=self.chunk_size,
        )

        mixed = self._scan_permute(ordered_out, forward=False)
        if self.collect_attention_target:
            # (batch, 64, heads, value_dim), board-square order, still
            # linear in last_value at this point -- captured before
            # rms_norm/gate/output_dense so it lines up with what MHA's own
            # `attention @ value` represents (mixing only, no output
            # projection folded in).
            self.last_mixed = mixed
        mixed = mixed.reshape(batch, BOARD_SQUARES, self.heads * self.value_dim)

        if self.rms_norm is not None:
            mixed = self.rms_norm(mixed)

        # The gate must apply after RMSNorm, not before -- they do not
        # commute, since the gate rescales each element before the norm
        # would divide by the resulting RMS. This ordering bug was found
        # and fixed in the SYCL engine; match its corrected order here.
        if self.output_gate:
            assert self.gate_a is not None and self.gate_b is not None
            gate = self.gate_b(self.gate_a(proj_input))
            gate = torch.sigmoid(gate)
            if self.collect_stats:
                with torch.no_grad():
                    # A gate collapsing toward 0 silences the mixer, leaving
                    # the block a pass-through however well the recurrence
                    # itself is behaving.
                    self.last_stats["output gate mean"] = gate.mean()
            mixed = mixed * gate

        out = self.output_dense(mixed)
        if self.collect_stats:
            with torch.no_grad():
                # Magnitude of what the mixer contributes to the residual
                # stream, before the encoder block's DeepNorm alpha.
                self.last_stats["output rms"] = out.square().mean().sqrt()
                self.last_stats["recurrence rms"] = (
                    ordered_out.square().mean().sqrt()
                )
        return out

    def _scan_permute(
        self, tensor: torch.Tensor, *, forward: bool
    ) -> torch.Tensor:
        """Reorder the (token, head) axis pair into or out of scan order.

        ``tensor`` is (batch, 64, heads, depth). The two axes are flattened
        into one so a single index_select covers every direction at once;
        the un-permute is the exact inverse permutation, so the same pair
        of index buffers serves both directions and both backward passes.
        """
        batch, tokens, heads, depth = tensor.shape
        flat = tensor.reshape(batch, tokens * heads, depth)
        order, inverse = self.scan_order, self.scan_inverse
        if not forward:
            order, inverse = inverse, order
        permuted = layers.permute_along(flat, 1, order, inverse)
        return permuted.reshape(batch, tokens, heads, depth)


def _build_scan_permutation(
    directions: list[str], heads: int
) -> tuple[torch.Tensor, torch.Tensor]:
    """Flat (token, head) permutation implementing all traversal orders.

    Head groups are contiguous blocks, one per direction. Indexing a
    ``(64 * heads,)`` axis as ``token * heads + head`` lets a single gather
    apply a different board traversal to each group.
    """
    heads_per_direction = heads // len(directions)
    order = np.empty(BOARD_SQUARES * heads, dtype=np.int64)
    for group, direction in enumerate(directions):
        traversal = np.asarray(KDA_TRAVERSALS[direction], dtype=np.int64)
        head_start = group * heads_per_direction
        for head in range(head_start, head_start + heads_per_direction):
            destination = np.arange(BOARD_SQUARES) * heads + head
            order[destination] = traversal * heads + head
    inverse = np.argsort(order)
    return torch.from_numpy(order), torch.from_numpy(inverse)
