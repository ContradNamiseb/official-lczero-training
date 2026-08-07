"""Multi-head attention and Smolgen for the DirectML port.

Mirrors the MultiHeadAttention and Smolgen classes in model/encoder.py,
batched.
"""

from __future__ import annotations

import math

import torch
from torch import nn

from proto import model_config_pb2

from . import layers

BOARD_SQUARES = 64


class Smolgen(nn.Module):
    """Generates a per-head 64x64 additive bias for the attention logits.

    The final ``weight_gen_dense`` is shared across every MHA block in the
    tower, so it is passed in rather than constructed here.
    """

    def __init__(
        self,
        *,
        in_features: int,
        config: model_config_pb2.SmolgenConfig,
        defaults: model_config_pb2.DefaultsConfig,
        heads: int,
        weight_gen_dense: nn.Linear,
    ):
        super().__init__()
        self.heads = heads
        self.compress = nn.Linear(
            in_features, config.hidden_channels, bias=False
        )
        self.dense1 = nn.Linear(
            config.hidden_channels * BOARD_SQUARES, config.hidden_size
        )
        self.ln1 = layers.LayerNorm(config.hidden_size)
        self.dense2 = nn.Linear(config.hidden_size, config.gen_size * heads)
        self.ln2 = layers.LayerNorm(config.gen_size * heads)
        for linear in (self.compress, self.dense1, self.dense2):
            layers.init_lecun_normal_(linear)
        self.weight_gen_dense = weight_gen_dense
        self.activation = layers.get_activation(
            config.activation or defaults.activation
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (batch, 64, in_features) -> (batch, heads, 64, 64).
        batch = x.shape[0]
        compressed = self.compress(x).reshape(batch, -1)
        hidden = self.ln1(self.activation(self.dense1(compressed)))
        generated = self.ln2(self.activation(self.dense2(hidden)))
        generated = generated.reshape(batch, self.heads, -1)
        out = self.weight_gen_dense(generated)
        return out.reshape(batch, self.heads, BOARD_SQUARES, BOARD_SQUARES)


class MultiHeadAttention(nn.Module):
    """Scaled dot-product attention over the 64 squares."""

    def __init__(
        self,
        *,
        in_features: int,
        config: model_config_pb2.EncoderConfig,
        defaults: model_config_pb2.DefaultsConfig,
        smol_gen_dense: nn.Linear | None,
        deepnorm_beta: float,
    ):
        super().__init__()
        depth = config.d_model
        assert depth % config.heads == 0, (
            "Model depth must be divisible by the number of heads."
        )
        self.depth = depth
        self.num_heads = config.heads
        self.head_depth = depth // config.heads

        self.q = nn.Linear(in_features, depth)
        self.k = nn.Linear(in_features, depth)
        self.v = nn.Linear(in_features, depth)
        self.output_dense = nn.Linear(depth, in_features)
        layers.init_lecun_normal_(self.q)
        layers.init_lecun_normal_(self.k)
        layers.init_variance_scaling_(self.v, deepnorm_beta)
        layers.init_variance_scaling_(self.output_dense, deepnorm_beta)

        assert (smol_gen_dense is not None) == config.HasField("smolgen")
        self.smolgen: Smolgen | None
        if smol_gen_dense is not None:
            self.smolgen = Smolgen(
                in_features=in_features,
                config=config.smolgen,
                defaults=defaults,
                heads=config.heads,
                weight_gen_dense=smol_gen_dense,
            )
        else:
            self.smolgen = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch = x.shape[0]

        def to_heads(tensor: torch.Tensor) -> torch.Tensor:
            return tensor.reshape(
                batch, BOARD_SQUARES, self.num_heads, self.head_depth
            ).permute(0, 2, 1, 3)

        query = to_heads(self.q(x))
        key = to_heads(self.k(x))
        value = to_heads(self.v(x))

        logits = query @ key.transpose(-1, -2)
        logits = logits / math.sqrt(self.head_depth)
        if self.smolgen is not None:
            logits = logits + self.smolgen(x)

        attention = torch.softmax(logits, dim=-1)
        out = attention @ value
        out = out.permute(0, 2, 1, 3).reshape(batch, BOARD_SQUARES, self.depth)
        return self.output_dense(out)
