"""Input embedding for the DirectML port.

Mirrors model/embedding.py. Every tensor here carries a native leading batch
dimension; the JAX original is unbatched and vmapped externally.
"""

from __future__ import annotations

import torch
from torch import nn

from proto import model_config_pb2

from . import layers

BOARD_SQUARES = 64
# The first 12 input planes are the piece placements; only those feed the
# positional preprocessing.
POSITIONAL_PLANES = 12


class Gating(nn.Module):
    """Per-square, per-channel multiplicative or additive gate."""

    def __init__(self, feature_shape: tuple[int, ...], additive: bool = True):
        super().__init__()
        self.additive = additive
        initial = 0.0 if additive else 1.0
        self.gate = nn.Parameter(torch.full(feature_shape, initial))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.additive:
            return x + self.gate
        # The multiplicative gate is rectified, so a gate driven negative
        # closes rather than inverting the sign of its channel.
        return x * torch.relu(self.gate)


class MaGating(nn.Module):
    """Multiplicative gate followed by an additive one."""

    def __init__(self, feature_shape: tuple[int, ...]):
        super().__init__()
        self.mult_gate = Gating(feature_shape, additive=False)
        self.add_gate = Gating(feature_shape, additive=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.add_gate(self.mult_gate(x))


class Embedding(nn.Module):
    """Positional preprocessing, square embedding, MaGating, residual FFN."""

    def __init__(
        self,
        *,
        input_channels: int,
        config: model_config_pb2.EmbeddingConfig,
        defaults: model_config_pb2.DefaultsConfig,
        deepnorm_alpha: float,
        deepnorm_beta: float,
    ):
        super().__init__()
        dense_size = config.dense_size
        embedding_size = config.embedding_size
        assert dense_size > 0
        assert embedding_size > 0

        self.activation = layers.get_activation(defaults.activation)
        self.deepnorm_alpha = deepnorm_alpha

        # Sees all 64 squares at once and produces a per-square vector, so
        # the embedding below has global context before any mixer runs.
        self.preprocess = nn.Linear(
            BOARD_SQUARES * POSITIONAL_PLANES, BOARD_SQUARES * dense_size
        )
        self.embedding = nn.Linear(input_channels + dense_size, embedding_size)
        layers.init_lecun_normal_(self.preprocess)
        layers.init_lecun_normal_(self.embedding)

        self.norm = layers.LayerNorm(embedding_size)
        self.ma_gating = MaGating((BOARD_SQUARES, embedding_size))
        self.ffn = layers.Ffn(
            embedding_size, config.dff, defaults.ffn_activation, deepnorm_beta
        )
        self.out_norm = layers.LayerNorm(embedding_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (batch, 64, input_channels).
        batch = x.shape[0]
        positional = x[..., :POSITIONAL_PLANES].reshape(batch, -1)
        positional = self.preprocess(positional).reshape(
            batch, BOARD_SQUARES, -1
        )
        x = torch.cat([x, positional], dim=-1)

        x = self.embedding(x)
        x = self.activation(x)
        x = self.norm(x)
        x = self.ma_gating(x)
        x = layers.deepnorm_residual(x, self.ffn(x), self.deepnorm_alpha)
        return self.out_norm(x)
