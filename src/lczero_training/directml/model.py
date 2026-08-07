"""Batched LczeroModel for the DirectML port.

Mirrors model/model.py. The JAX model operates on a single (64, features)
position and is batched externally with vmap; this one takes the batch
natively, as Phase 5 of docs/directml_training_port.md requires.
"""

from __future__ import annotations

import dataclasses
import math

import torch
from torch import nn

from proto import hlo_pb2, model_config_pb2

from . import layers
from .embedding import Embedding
from .encoder import EncoderTower
from .heads import MovesLeftHead, PolicyHead, ValueHead

INPUT_CHANNELS = 112
BOARD_SQUARES = 64


@dataclasses.dataclass
class ModelPrediction:
    """Named outputs, keyed by head name.

    ``value`` maps to (wdl_logits, error, categorical); the last two are
    None for heads that do not configure them.
    """

    value: dict[
        str, tuple[torch.Tensor, torch.Tensor | None, torch.Tensor | None]
    ]
    policy: dict[str, torch.Tensor]
    movesleft: dict[str, torch.Tensor]


def _check_compute_dtype(config: model_config_pb2.ModelConfig) -> None:
    """Refuse a compute_dtype this port does not actually implement.

    The JAX model casts its input to ``defaults.compute_dtype``; this one
    runs in float32 throughout, and the KDA recurrence deliberately forces
    float32 internally for the exp/cumsum of the log decay. Accepting F16
    here would silently train in float32 anyway while changing the model
    config hash -- invalidating every checkpoint for no benefit. Fail loudly
    instead.
    """
    dtype = config.defaults.compute_dtype
    if dtype not in (
        hlo_pb2.XlaShapeProto.F32,
        hlo_pb2.XlaShapeProto.PRIMITIVE_TYPE_INVALID,
    ):
        name = hlo_pb2.XlaShapeProto.Type.Name(dtype)
        raise NotImplementedError(
            f"defaults.compute_dtype is {name}, but the DirectML port only "
            "implements F32. Setting it to anything else changes the model "
            "config hash (invalidating checkpoints) without changing what is "
            "computed. Set compute_dtype: F32."
        )


class LczeroModel(nn.Module):
    def __init__(self, config: model_config_pb2.ModelConfig):
        super().__init__()
        _check_compute_dtype(config)
        self.config = config
        num_blocks = config.encoder.num_blocks
        assert num_blocks > 0
        deepnorm_beta = layers.deepnorm_beta(num_blocks)

        self.embedding = Embedding(
            input_channels=INPUT_CHANNELS,
            config=config.embedding,
            defaults=config.defaults,
            deepnorm_alpha=math.pow(2.0 * num_blocks, -0.25),
            deepnorm_beta=deepnorm_beta,
        )
        self.encoders = EncoderTower(
            in_features=config.embedding.embedding_size,
            config=config.encoder,
            defaults=config.defaults,
            deepnorm_beta=deepnorm_beta,
        )

        embedding_size = config.embedding.embedding_size

        self.value_heads = nn.ModuleDict(
            {
                head.name: ValueHead(
                    in_features=embedding_size,
                    config=head,
                    defaults=config.defaults,
                )
                for head in config.value_head
            }
        )

        # Shared across every policy head when configured, so it must be
        # built once and handed to each of them.
        self.policy_embedding_shared: nn.Linear | None = None
        if config.HasField("shared_policy_embedding_size"):
            self.policy_embedding_shared = nn.Linear(
                embedding_size, config.shared_policy_embedding_size
            )
            layers.init_lecun_normal_(self.policy_embedding_shared)

        self.policy_heads = nn.ModuleDict(
            {
                head.name: PolicyHead(
                    in_features=embedding_size,
                    config=head,
                    defaults=config.defaults,
                    shared_embedding=self.policy_embedding_shared,
                )
                for head in config.policy_head
            }
        )
        self.movesleft_heads = nn.ModuleDict(
            {
                head.name: MovesLeftHead(
                    in_features=embedding_size,
                    config=head,
                    defaults=config.defaults,
                )
                for head in config.movesleft_head
            }
        )

    def forward(self, x: torch.Tensor) -> ModelPrediction:
        # x: (batch, 112, 8, 8) as the data loader produces it. Squares end
        # up token-major (rank * 8 + file), matching rank_forward and the
        # KDA local conv's reshape.
        batch = x.shape[0]
        x = x.permute(0, 2, 3, 1).reshape(batch, BOARD_SQUARES, INPUT_CHANNELS)
        x = self.embedding(x)
        x = self.encoders(x)

        return ModelPrediction(
            value={name: head(x) for name, head in self.value_heads.items()},
            policy={name: head(x) for name, head in self.policy_heads.items()},
            movesleft={
                name: head(x) for name, head in self.movesleft_heads.items()
            },
        )
