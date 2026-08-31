import dataclasses
import logging
from typing import Optional, cast

import numpy as np
from flax import nnx

from lczero_training.convert.leela_pytree_visitor import (
    LeelaPytreeWeightsVisitor,
)
from lczero_training.model.utils import encoder_mixer_pattern
from proto import model_config_pb2, net_pb2

logger = logging.getLogger(__name__)

_EMBEDDING_PLANE_TO_SCALE = 109
_EMBEDDING_SCALE = 99.0

# Mirrors SERPENTINE_SUBSTITUTIONS in model/kda.py. Duplicated rather than
# imported so the converter does not pull in the model package.
_SERPENTINE_SUBSTITUTIONS = {
    "rank_forward": "rank_serpentine",
    "rank_reverse": "rank_serpentine_reverse",
    "file_forward": "file_serpentine",
    "file_reverse": "file_serpentine_reverse",
    "diag_forward": "diag_serpentine",
    "diag_reverse": "diag_serpentine_reverse",
    "anti_diag_forward": "anti_diag_serpentine",
    "anti_diag_reverse": "anti_diag_serpentine_reverse",
}

# Maps a KdaConfig.directions string to the engine's KdaDirection enum.
# Mirrors set_kda_directions() in stable-branch/tf/net.py.
_KDA_DIRECTION_TO_ENUM = {
    "rank_forward": net_pb2.NetworkFormat.KDA_DIRECTION_RANK_FORWARD,
    "rank_reverse": net_pb2.NetworkFormat.KDA_DIRECTION_RANK_REVERSE,
    "file_forward": net_pb2.NetworkFormat.KDA_DIRECTION_FILE_FORWARD,
    "file_reverse": net_pb2.NetworkFormat.KDA_DIRECTION_FILE_REVERSE,
    "diag_forward": net_pb2.NetworkFormat.KDA_DIRECTION_DIAG_FORWARD,
    "diag_reverse": net_pb2.NetworkFormat.KDA_DIRECTION_DIAG_REVERSE,
    "anti_diag_forward": net_pb2.NetworkFormat.KDA_DIRECTION_ANTI_DIAG_FORWARD,
    "anti_diag_reverse": net_pb2.NetworkFormat.KDA_DIRECTION_ANTI_DIAG_REVERSE,
    "rank_serpentine": net_pb2.NetworkFormat.KDA_DIRECTION_RANK_SERPENTINE,
    "rank_serpentine_reverse": (
        net_pb2.NetworkFormat.KDA_DIRECTION_RANK_SERPENTINE_REVERSE
    ),
    "file_serpentine": net_pb2.NetworkFormat.KDA_DIRECTION_FILE_SERPENTINE,
    "file_serpentine_reverse": (
        net_pb2.NetworkFormat.KDA_DIRECTION_FILE_SERPENTINE_REVERSE
    ),
    "diag_serpentine": net_pb2.NetworkFormat.KDA_DIRECTION_DIAG_SERPENTINE,
    "diag_serpentine_reverse": (
        net_pb2.NetworkFormat.KDA_DIRECTION_DIAG_SERPENTINE_REVERSE
    ),
    "anti_diag_serpentine": (
        net_pb2.NetworkFormat.KDA_DIRECTION_ANTI_DIAG_SERPENTINE
    ),
    "anti_diag_serpentine_reverse": (
        net_pb2.NetworkFormat.KDA_DIRECTION_ANTI_DIAG_SERPENTINE_REVERSE
    ),
}


class JaxToLeela(LeelaPytreeWeightsVisitor):
    def __init__(
        self,
        nnx_state: nnx.State,
        leela_net: net_pb2.Net,
        encoder_config: model_config_pb2.EncoderConfig,
    ) -> None:
        super().__init__(nnx_state, leela_net)
        # Scalar KDA config (key_dim, output_gate, ...) is not part of the
        # NNX pytree, so it has to come from the config that built the
        # model rather than from the state tree being walked.
        self._encoder_config = encoder_config

    def embedding_block(
        self, nnx_dict: nnx.State, weights: net_pb2.Weights
    ) -> None:
        embedding_kernel = cast(nnx.Param, nnx_dict["embedding"]["kernel"])
        original_values = embedding_kernel.value
        arr = np.asarray(original_values).copy()
        arr[_EMBEDDING_PLANE_TO_SCALE] /= _EMBEDDING_SCALE
        embedding_kernel.value = arr
        try:
            super().embedding_block(nnx_dict=nnx_dict, weights=weights)
        finally:
            embedding_kernel.value = original_values

    def tensor(
        self,
        param: nnx.Param,
        leela: net_pb2.Weights.Layer,
    ) -> None:
        weights = np.asarray(param.value, dtype=np.float32).T.flatten()
        self._write_quantized(weights, leela)

    def _write_quantized(
        self, weights: np.ndarray, leela: net_pb2.Weights.Layer
    ) -> None:
        min_val, max_val = np.min(weights), np.max(weights)
        range_val = max_val - min_val

        # Normalize to [0, 1], handling the case where all weights are equal.
        normalized = np.where(
            range_val > 1e-8, (weights - min_val) / range_val, 0.5
        )

        # Scale to uint16 and convert to bytes.
        quantized = np.round(normalized * 65535.0).astype(np.uint16)
        leela.params = quantized.tobytes()
        leela.min_val = float(min_val)
        leela.max_val = float(max_val)

        assert len(leela.params) // 2 == weights.size

    def zero_bias(
        self,
        kernel_param: nnx.Param,
        biases: net_pb2.Weights.Layer,
    ) -> None:
        out_features = np.asarray(kernel_param.value).shape[-1]
        self._write_quantized(
            np.zeros(out_features, dtype=np.float32), biases
        )

    def kda_local_conv(
        self,
        nnx_dict: nnx.State,
        kernel_weights: net_pb2.Weights.Layer,
        bias_weights: net_pb2.Weights.Layer,
    ) -> None:
        # Flax's grouped/depthwise Conv kernel is (kh, kw, 1, channels).
        # tensor()'s generic path does .T.flatten(), which reverses *all*
        # axes and would flatten this as c*9 + kw*3 + kh -- the engine
        # expects w[c*9 + kr*3 + kf] (kr=height offset, kf=width offset),
        # i.e. c*9 + kh*3 + kw. Swapping the two spatial axes before the
        # generic .T.flatten() produces exactly that layout: swapped shape
        # is (kw, kh, 1, c); after .T it is (c, 1, kh, kw), whose C-order
        # flatten is c*9 + kh*3 + kw.
        kernel = cast(nnx.Param, nnx_dict["kernel"])
        kernel_arr = np.asarray(kernel.value, dtype=np.float32)
        assert kernel_arr.ndim == 4 and kernel_arr.shape[:2] == (3, 3), (
            f"expected a (3, 3, 1, channels) depthwise kernel, got "
            f"{kernel_arr.shape}"
        )
        swapped = np.swapaxes(kernel_arr, 0, 1)
        self._write_quantized(swapped.T.flatten(), kernel_weights)

        if "bias" in nnx_dict:
            self.tensor(cast(nnx.Param, nnx_dict["bias"]), bias_weights)

    def encoder_block(
        self, nnx_dict: nnx.State, weights: net_pb2.Weights.EncoderLayer
    ) -> None:
        if "mixer" in nnx_dict:
            kda_config = self._encoder_config.kda
            weights.mixer = net_pb2.Weights.EncoderLayer.MIXER_KDA
            weights.kda.key_dim = kda_config.key_dim
            weights.kda.value_dim = kda_config.value_dim
            weights.kda.gate_rank = kda_config.gate_rank
            weights.kda.output_gate = kda_config.output_gate
            # The protobuf default is true, so this must always be written
            # explicitly, matching stable-branch/tf/net.py:120.
            weights.kda.output_rms_norm = kda_config.output_rms_norm
            weights.kda.local_conv = kda_config.local_conv
            weights.kda.qkv_silu = kda_config.qkv_silu
        else:
            weights.mixer = net_pb2.Weights.EncoderLayer.MIXER_MHA
        super().encoder_block(nnx_dict=nnx_dict, weights=weights)

    def encoder_tower(
        self, nnx_dict: nnx.State, weights: net_pb2.Weights
    ) -> None:
        for i in range(len(nnx_dict["encoders"]["layers"])):
            weights.encoder.append(weights.EncoderLayer())
        return super().encoder_tower(nnx_dict=nnx_dict, weights=weights)


@dataclasses.dataclass
class LeelaExportOptions:
    min_version: str
    num_heads: int
    license: Optional[str]
    training_steps: Optional[int] = None


def jax_to_leela(
    jax_weights: nnx.State,
    export_options: LeelaExportOptions,
    encoder_config: model_config_pb2.EncoderConfig,
) -> net_pb2.Net:
    lc0_weights = net_pb2.Net()
    lc0_weights.magic = 0x1C0
    if export_options.license:
        lc0_weights.license = export_options.license
    (
        lc0_weights.min_version.major,
        lc0_weights.min_version.minor,
        lc0_weights.min_version.patch,
    ) = _split_version(export_options.min_version)
    lc0_weights.format.CopyFrom(_make_format(encoder_config))
    if export_options.training_steps is not None:
        lc0_weights.training_params.training_steps = (
            export_options.training_steps
        )

    visitor = JaxToLeela(jax_weights, lc0_weights, encoder_config)
    lc0_weights.weights.headcount = export_options.num_heads
    visitor.run()

    return lc0_weights


def _split_version(version_str: str) -> tuple[int, int, int]:
    """Splits a version string like "v12.34.56" into (12, 34, 56)."""
    parts = (version_str.lstrip("v").split(".") + ["0", "0"])[:3]
    return cast(tuple[int, int, int], tuple(map(int, parts)))


def _make_format(
    encoder_config: model_config_pb2.EncoderConfig,
) -> net_pb2.Format:
    mixer_pattern = encoder_mixer_pattern(encoder_config)
    has_kda = model_config_pb2.MIXER_KDA in mixer_pattern
    has_mha = model_config_pb2.MIXER_MHA in mixer_pattern

    fmt = net_pb2.Format()
    fmt.weights_encoding = fmt.LINEAR16
    netfmt = fmt.network_format
    netfmt.input = netfmt.INPUT_CLASSICAL_112_PLANE
    netfmt.output = netfmt.OUTPUT_WDL
    # NETWORK_KDA_HYBRID_WITH_MULTIHEADFORMAT covers both a pure-KDA tower
    # and a genuinely mixed one (e.g. [kda, kda, kda, mha], the pattern the
    # real kda-hybrid-* checkpoints use) -- any KDA block at all routes
    # through it.
    netfmt.network = (
        netfmt.NETWORK_KDA_HYBRID_WITH_MULTIHEADFORMAT
        if has_kda
        else netfmt.NETWORK_ATTENTIONBODY_WITH_MULTIHEADFORMAT
    )
    netfmt.policy = netfmt.POLICY_ATTENTION
    netfmt.value = netfmt.VALUE_WDL
    netfmt.moves_left = netfmt.MOVES_LEFT_V1
    netfmt.default_activation = netfmt.DEFAULT_ACTIVATION_MISH
    netfmt.ffn_activation = netfmt.ACTIVATION_DEFAULT
    netfmt.input_embedding = netfmt.INPUT_EMBEDDING_PE_DENSE
    # Smolgen only applies to MHA blocks, but a hybrid tower still needs it
    # for the MHA blocks it does have.
    if has_mha:
        netfmt.smolgen_activation = netfmt.ACTIVATION_SWISH

    if has_kda:
        directions = list(encoder_config.kda.directions)
        if encoder_config.kda.serpentine:
            # KdaMixer substitutes these internally, so the config still
            # names the orthogonal walks. Resolve them here or the exported
            # net would declare a scan order the weights were never trained
            # on -- silently wrong at inference, which is exactly what a
            # graph-based backend like OpenVINO would faithfully reproduce.
            directions = [
                _SERPENTINE_SUBSTITUTIONS.get(name, name) for name in directions
            ]
        if not directions:
            raise ValueError(
                "KdaConfig.directions must not be empty for a KDA network."
            )
        if encoder_config.heads % len(directions) != 0:
            raise ValueError(
                f"encoder heads ({encoder_config.heads}) must be evenly "
                f"divisible by len(directions) ({len(directions)})."
            )
        del netfmt.kda_directions[:]
        for direction in directions:
            if direction not in _KDA_DIRECTION_TO_ENUM:
                raise ValueError(f"Unknown KDA direction: {direction!r}")
            netfmt.kda_directions.append(_KDA_DIRECTION_TO_ENUM[direction])

    return fmt
