import dataclasses
import gzip
import logging
import math
from typing import Optional, cast

import jax.numpy as jnp
from flax import nnx, serialization

from lczero_training.model.model import LczeroModel
from proto import hlo_pb2, net_pb2

from .jax_to_leela import LeelaExportOptions, jax_to_leela
from .leela_pytree_visitor import LeelaPytreeWeightsVisitor
from .leela_to_modelconfig import leela_to_modelconfig

logger = logging.getLogger(__name__)


_EMBEDDING_PLANE_TO_SCALE = 109
_EMBEDDING_SCALE = 99.0


@dataclasses.dataclass
class LeelaImportOptions:
    weights_dtype: hlo_pb2.XlaShapeProto.Type
    compute_dtype: hlo_pb2.XlaShapeProto.Type


def fix_older_weights_file(file: net_pb2.Net) -> None:
    nf = net_pb2.NetworkFormat
    has_network_format = file.format.HasField("network_format")
    network_format = (
        file.format.network_format.network if has_network_format else None
    )

    net = file.format.network_format

    if not has_network_format:
        # Older protobufs don't have format definition.
        net.input = nf.INPUT_CLASSICAL_112_PLANE
        net.output = nf.OUTPUT_CLASSICAL
        net.network = nf.NETWORK_CLASSICAL_WITH_HEADFORMAT
        net.value = nf.VALUE_CLASSICAL
        net.policy = nf.POLICY_CLASSICAL
    elif network_format == nf.NETWORK_CLASSICAL:
        # Populate policyFormat and valueFormat fields in old protobufs
        # without these fields.
        net.network = nf.NETWORK_CLASSICAL_WITH_HEADFORMAT
        net.value = nf.VALUE_CLASSICAL
        net.policy = nf.POLICY_CLASSICAL
    elif network_format == nf.NETWORK_SE:
        net.network = nf.NETWORK_SE_WITH_HEADFORMAT
        net.value = nf.VALUE_CLASSICAL
        net.policy = nf.POLICY_CLASSICAL
    elif (
        network_format == nf.NETWORK_SE_WITH_HEADFORMAT
        and len(file.weights.encoder) > 0
    ):
        # Attention body network made with old protobuf.
        net.network = nf.NETWORK_ATTENTIONBODY_WITH_HEADFORMAT
        if file.weights.HasField("smolgen_w"):
            # Need to override activation defaults for smolgen.
            net.ffn_activation = nf.ACTIVATION_RELU_2
            net.smolgen_activation = nf.ACTIVATION_SWISH
    elif network_format == nf.NETWORK_AB_LEGACY_WITH_MULTIHEADFORMAT:
        net.network = nf.NETWORK_ATTENTIONBODY_WITH_MULTIHEADFORMAT

    if (
        file.format.network_format.network
        == nf.NETWORK_ATTENTIONBODY_WITH_HEADFORMAT
    ):
        weights = file.weights
        if weights.HasField("policy_heads") and weights.HasField("value_heads"):
            logger.info(
                "Weights file has multihead format, updating format flag"
            )
            net.network = nf.NETWORK_ATTENTIONBODY_WITH_MULTIHEADFORMAT
            net.input_embedding = nf.INPUT_EMBEDDING_PE_DENSE
        if not file.format.network_format.HasField("input_embedding"):
            net.input_embedding = nf.INPUT_EMBEDDING_PE_MAP


class LeelaToJax(LeelaPytreeWeightsVisitor):
    def embedding_block(
        self, nnx_dict: nnx.State, weights: net_pb2.Weights
    ) -> None:
        super().embedding_block(nnx_dict=nnx_dict, weights=weights)
        embedding_kernel = cast(nnx.Param, nnx_dict["embedding"]["kernel"])
        values = embedding_kernel.value
        scaled_values = values.at[_EMBEDDING_PLANE_TO_SCALE].set(
            values[_EMBEDDING_PLANE_TO_SCALE] * _EMBEDDING_SCALE
        )
        embedding_kernel.value = scaled_values

    def tensor(
        self,
        param: nnx.Param,
        leela: net_pb2.Weights.Layer,
    ) -> None:
        assert len(leela.params) // 2 == math.prod(param.shape)
        assert len(leela.params) != 0

        values = jnp.frombuffer(leela.params, dtype=jnp.uint16)
        values = values.astype(jnp.float32)
        alpha = values / 65535.0
        values = alpha * leela.max_val + (1.0 - alpha) * leela.min_val
        values = values.astype(param.dtype)
        values = values.reshape(param.shape[::-1]).transpose()
        param.value = values

    def kda_local_conv(
        self,
        nnx_dict: nnx.State,
        kernel_weights: net_pb2.Weights.Layer,
        bias_weights: net_pb2.Weights.Layer,
    ) -> None:
        # Inverse of JaxToLeela.kda_local_conv (jax_to_leela.py): that
        # direction proved leela_flat[c*9 + kr*3 + kf] == kernel[kr, kf,
        # 0, c] for the (kh, kw, 1, channels) Flax depthwise kernel, i.e.
        # flattening (channels, 1, kr, kf) in C order. Undo that by
        # dequantizing (same as tensor()'s generic path, which can't be
        # reused directly because the *shape* this reshapes into isn't
        # simply param.shape[::-1]), reshaping back to (channels, 1, kr,
        # kf), then transposing to Flax's (kr, kf, 1, channels).
        kernel_param = cast(nnx.Param, nnx_dict["kernel"])
        channels = kernel_param.value.shape[-1]
        assert len(kernel_weights.params) // 2 == channels * 9, (
            f"expected a channels*9 flat depthwise kernel for "
            f"channels={channels}, got "
            f"{len(kernel_weights.params) // 2} values"
        )
        values = jnp.frombuffer(kernel_weights.params, dtype=jnp.uint16)
        values = values.astype(jnp.float32)
        alpha = values / 65535.0
        values = (
            alpha * kernel_weights.max_val
            + (1.0 - alpha) * kernel_weights.min_val
        )
        values = values.astype(kernel_param.dtype)
        channel_major = values.reshape((channels, 1, 3, 3))
        kernel_param.value = jnp.transpose(channel_major, (2, 3, 1, 0))

        if "bias" in nnx_dict:
            self.tensor(cast(nnx.Param, nnx_dict["bias"]), bias_weights)


def leela_to_jax(
    leela_net: net_pb2.Net, import_options: LeelaImportOptions
) -> nnx.State:
    config = leela_to_modelconfig(
        leela_net,
        import_options.weights_dtype,
        import_options.compute_dtype,
    )

    model = LczeroModel(config=config, rngs=nnx.Rngs(params=42))
    state = nnx.state(model)
    visitor = LeelaToJax(state, leela_net)
    visitor.run()

    return state


def leela_to_jax_files(
    input_path: str,
    weights_dtype: str,
    compute_dtype: str,
    output_modelconfig: Optional[str],
    output_serialized_jax: Optional[str],
    output_leela_verification: Optional[str],
    print_modelconfig: bool = False,
) -> None:
    lc0_weights = net_pb2.Net()
    with gzip.open(input_path, "rb") as f:
        contents = f.read()
        assert isinstance(contents, bytes)
        lc0_weights.ParseFromString(contents)

    fix_older_weights_file(lc0_weights)

    import_options = LeelaImportOptions(
        weights_dtype=getattr(hlo_pb2.XlaShapeProto, weights_dtype),
        compute_dtype=getattr(hlo_pb2.XlaShapeProto, compute_dtype),
    )

    config = leela_to_modelconfig(
        lc0_weights,
        import_options.weights_dtype,
        import_options.compute_dtype,
    )

    if print_modelconfig:
        print(config)

    if output_modelconfig:
        with open(output_modelconfig, "w") as f:
            f.write(str(config))

    if output_serialized_jax is None and output_leela_verification is None:
        return

    state = leela_to_jax(lc0_weights, import_options)

    if output_serialized_jax:
        # flax.serialization.to_bytes/msgpack only understands plain
        # pytrees of arrays, not nnx.State's nnx.Variable-wrapped leaves
        # (this predates the KDA work -- any leela2jax run with this flag
        # hit it). to_pure_dict() strips the wrappers.
        with open(output_serialized_jax, "wb") as f:
            f.write(serialization.to_bytes(state.to_pure_dict()))

    if output_leela_verification:
        min_version = (
            f"v{lc0_weights.min_version.major}."
            f"{lc0_weights.min_version.minor}."
            f"{lc0_weights.min_version.patch}"
        )
        license_str = (
            lc0_weights.license if lc0_weights.HasField("license") else None
        )
        export_options = LeelaExportOptions(
            min_version=min_version,
            num_heads=lc0_weights.weights.headcount,
            license=license_str,
            training_steps=lc0_weights.training_params.training_steps,
        )
        verification_net = jax_to_leela(
            jax_weights=state,
            export_options=export_options,
            encoder_config=config.encoder,
        )
        with gzip.open(output_leela_verification, "wb") as f:
            f.write(verification_net.SerializeToString())
