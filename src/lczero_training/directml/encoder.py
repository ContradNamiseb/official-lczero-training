"""Encoder blocks and the tower for the DirectML port.

Mirrors EncoderTower and EncoderBlock in model/encoder.py, batched.
"""

from __future__ import annotations

import math

import torch
from torch import nn

from proto import model_config_pb2

from . import layers
from .attention import MultiHeadAttention
from .kda import KdaMixer

BOARD_SQUARES = 64


def encoder_mixer_pattern(
    config: model_config_pb2.EncoderConfig,
) -> list[int]:
    """Per-block mixer type for each of config.num_blocks blocks.

    Mirrors model/utils.py: mixer_pattern is tiled with index % len(pattern),
    so a 1-element pattern (or the unset case, falling back to mixer_type)
    still means "every block", and [kda, kda, kda, mha] is expressed exactly.
    """
    pattern = list(config.mixer_pattern) or [config.mixer_type]
    return [pattern[i % len(pattern)] for i in range(config.num_blocks)]


class EncoderBlock(nn.Module):
    """One encoder block: a mixer, then an FFN, both DeepNorm-residual."""

    def __init__(
        self,
        *,
        in_features: int,
        config: model_config_pb2.EncoderConfig,
        mixer_type: int,
        defaults: model_config_pb2.DefaultsConfig,
        smol_gen_dense: nn.Linear | None,
        deepnorm_beta: float,
    ):
        super().__init__()
        self.is_kda = mixer_type == model_config_pb2.MIXER_KDA
        # A KDA block never uses the shared smolgen dense, whatever the rest
        # of the tower contains.
        assert smol_gen_dense is None or not self.is_kda

        # Attribute is named `mha` for an MHA block and `mixer` for a KDA
        # block -- never both. The weight importer dispatches on which one is
        # present, matching the JAX state tree.
        if self.is_kda:
            self.mixer = KdaMixer(
                in_features=in_features,
                config=config.kda,
                heads=config.heads,
                deepnorm_beta=deepnorm_beta,
            )
        else:
            self.mha = MultiHeadAttention(
                in_features=in_features,
                config=config,
                defaults=defaults,
                smol_gen_dense=smol_gen_dense,
                deepnorm_beta=deepnorm_beta,
            )

        self.alpha = math.pow(2.0 * config.num_blocks, -0.25)
        self.ln1 = layers.LayerNorm(in_features)
        self.ffn = layers.Ffn(
            in_features, config.dff, defaults.ffn_activation, deepnorm_beta
        )
        self.ln2 = layers.LayerNorm(in_features)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        mixer_out = self.mixer(x) if self.is_kda else self.mha(x)
        x = layers.deepnorm_residual(x, mixer_out, self.alpha)
        out1 = self.ln1(x)
        return self.ln2(
            layers.deepnorm_residual(out1, self.ffn(out1), self.alpha)
        )


class EncoderTower(nn.Module):
    """The stack of encoder blocks, mixed per `mixer_pattern`."""

    def __init__(
        self,
        *,
        in_features: int,
        config: model_config_pb2.EncoderConfig,
        defaults: model_config_pb2.DefaultsConfig,
        deepnorm_beta: float,
    ):
        super().__init__()
        mixer_pattern = encoder_mixer_pattern(config)

        # Smolgen is only meaningful for MHA blocks and is optional even for
        # those. Build the shared dense only when the tower actually contains
        # an MHA block and opted into smolgen at all.
        self.smolgen_shared_gen_dense: nn.Linear | None = None
        if model_config_pb2.MIXER_MHA in mixer_pattern and config.HasField(
            "smolgen"
        ):
            self.smolgen_shared_gen_dense = nn.Linear(
                config.smolgen.gen_size,
                BOARD_SQUARES * BOARD_SQUARES,
                bias=False,
            )
            layers.init_lecun_normal_(self.smolgen_shared_gen_dense)

        self.encoders = nn.ModuleList(
            [
                EncoderBlock(
                    in_features=in_features,
                    config=config,
                    mixer_type=mixer_pattern[index],
                    defaults=defaults,
                    smol_gen_dense=(
                        self.smolgen_shared_gen_dense
                        if mixer_pattern[index] == model_config_pb2.MIXER_MHA
                        else None
                    ),
                    deepnorm_beta=deepnorm_beta,
                )
                for index in range(config.num_blocks)
            ]
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for block in self.encoders:
            x = block(x)
        return x
