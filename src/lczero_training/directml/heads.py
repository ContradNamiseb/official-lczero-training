"""Policy, value, and moves-left heads for the DirectML port.

Mirrors model/policy_head.py, model/value_head.py, and
model/movesleft_head.py, batched.
"""

from __future__ import annotations

import math

import torch
from torch import nn

from lczero_training.model.policy_map import POLICY_MAP
from proto import model_config_pb2

from . import layers

BOARD_SQUARES = 64
# Flat attention logits (64*64) followed by promotion logits (8*24).
POLICY_LOGIT_COUNT = BOARD_SQUARES * BOARD_SQUARES + 8 * 24


class PolicyHead(nn.Module):
    """Attention-style policy: a 64x64 logit grid plus promotion offsets."""

    def __init__(
        self,
        *,
        in_features: int,
        config: model_config_pb2.PolicyHeadConfig,
        defaults: model_config_pb2.DefaultsConfig,
        shared_embedding: nn.Linear | None = None,
    ):
        super().__init__()
        assert (shared_embedding is not None) != config.HasField(
            "embedding_size"
        )
        self.activation = layers.get_activation(defaults.activation)

        if shared_embedding is not None:
            self.tokens = shared_embedding
            embedding_size = shared_embedding.out_features
        else:
            embedding_size = config.embedding_size
            self.tokens = nn.Linear(in_features, embedding_size)
            layers.init_lecun_normal_(self.tokens)

        self.q = nn.Linear(
            embedding_size, config.d_model, bias=config.use_bias
        )
        self.k = nn.Linear(
            embedding_size, config.d_model, bias=config.use_bias
        )
        layers.init_lecun_normal_(self.q)
        layers.init_lecun_normal_(self.k)

        self.dk = math.sqrt(config.d_model)
        self.promotion_dense = nn.Linear(config.d_model, 4, bias=False)
        layers.init_lecun_normal_(self.promotion_dense)

        # The map selects 1858 of the 4288 flat logits with no repeats, so
        # its gradient never accumulates -- injective_gather exploits that
        # to avoid index_add, which DirectML runs on the CPU.
        policy_map = torch.tensor(POLICY_MAP, dtype=torch.int64)
        self.register_buffer("policy_map", policy_map, persistent=False)
        self.register_buffer(
            "policy_map_inverse",
            layers.injective_inverse(policy_map, POLICY_LOGIT_COUNT),
            persistent=False,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch = x.shape[0]
        x = self.activation(self.tokens(x))
        query = self.q(x)
        key = self.k(x)
        qk = query @ key.transpose(-1, -2)  # (batch, 64, 64)

        # The last rank's keys carry the promotion offsets.
        promotion_keys = key[:, -8:, :]
        offsets = self.promotion_dense(promotion_keys)  # (batch, 8, 4)
        offsets = offsets.transpose(-1, -2) * self.dk  # (batch, 4, 8)
        # The knight offset is the baseline, added to the other three.
        offsets = offsets[:, :3, :] + offsets[:, 3:4, :]

        knight_promo = qk[:, -16:-8, -8:]  # (batch, 8, 8)
        promotion_logits = torch.stack(
            [
                knight_promo + offsets[:, index : index + 1, :]
                for index in range(3)
            ],
            dim=-1,
        )  # (batch, 8, 8, 3)

        attention_logits = qk / self.dk
        promotion_logits = promotion_logits.reshape(batch, 8, 24) / self.dk

        logits = torch.cat(
            [
                attention_logits.reshape(batch, -1),
                promotion_logits.reshape(batch, -1),
            ],
            dim=-1,
        )
        return layers.injective_gather(
            logits, 1, self.policy_map, self.policy_map_inverse
        )


class ValueHead(nn.Module):
    """WDL value head, optionally with error and categorical outputs."""

    def __init__(
        self,
        *,
        in_features: int,
        config: model_config_pb2.ValueHeadConfig,
        defaults: model_config_pb2.DefaultsConfig,
    ):
        super().__init__()
        self.activation = layers.get_activation(defaults.activation)
        self.has_error_output = config.has_error_output
        self.num_categorical_buckets = config.num_categorical_buckets

        self.embed = nn.Linear(in_features, config.num_channels)
        self.dense1 = nn.Linear(config.num_channels * BOARD_SQUARES, 128)
        self.wdl = nn.Linear(128, 3)
        layers.init_lecun_normal_(self.embed)
        layers.init_lecun_normal_(self.dense1)
        layers.init_lecun_normal_(self.wdl)

        self.error: nn.Linear | None = None
        if self.has_error_output:
            self.error = nn.Linear(128, 1)
            layers.init_lecun_normal_(self.error)

        self.categorical: nn.Linear | None = None
        if self.num_categorical_buckets > 0:
            self.categorical = nn.Linear(128, self.num_categorical_buckets)
            layers.init_lecun_normal_(self.categorical)

    def forward(
        self, x: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor | None, torch.Tensor | None]:
        batch = x.shape[0]
        x = self.embed(x).reshape(batch, -1)
        x = self.activation(x)
        x = self.activation(self.dense1(x))

        wdl = self.wdl(x)
        error = torch.sigmoid(self.error(x)) if self.error is not None else None
        categorical = (
            self.categorical(x) if self.categorical is not None else None
        )
        return wdl, error, categorical


class MovesLeftHead(nn.Module):
    """Scalar plies-left prediction, rectified to stay non-negative."""

    def __init__(
        self,
        *,
        in_features: int,
        config: model_config_pb2.MovesLeftHeadConfig,
        defaults: model_config_pb2.DefaultsConfig,
    ):
        super().__init__()
        self.activation = layers.get_activation(defaults.activation)
        self.embed = nn.Linear(in_features, config.num_channels)
        self.dense1 = nn.Linear(config.num_channels * BOARD_SQUARES, 128)
        self.out = nn.Linear(128, 1)
        for linear in (self.embed, self.dense1, self.out):
            layers.init_lecun_normal_(linear)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch = x.shape[0]
        x = self.embed(x).reshape(batch, -1)
        x = self.activation(x)
        x = self.activation(self.dense1(x))
        return torch.relu(self.out(x))
