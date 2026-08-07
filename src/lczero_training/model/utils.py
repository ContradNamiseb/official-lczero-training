from typing import Any

import jax.numpy as jnp
from flax import nnx
from jax.nn import mish

from proto import model_config_pb2, net_pb2
from proto.hlo_pb2 import XlaShapeProto


def encoder_mixer_pattern(
    config: model_config_pb2.EncoderConfig,
) -> list[model_config_pb2.MixerType]:
    """Per-block mixer type for each of config.num_blocks encoder blocks.

    Mirrors the TF reference's encoder_mixer_pattern tiling
    (stable-branch/tf/tfprocess.py, ~L379-409): mixer_pattern is repeated
    (via index % len(pattern)) to cover num_blocks, so e.g. a 1-element
    pattern (or the unset case, which falls back to mixer_type) still
    means "every block", and a genuinely mixed tower like [kda, kda, kda,
    mha] is expressed exactly, not approximated by a single mixer_type.
    """
    pattern = list(config.mixer_pattern) or [config.mixer_type]
    return [pattern[i % len(pattern)] for i in range(config.num_blocks)]


def get_activation(
    activation: net_pb2.NetworkFormat.ActivationFunction,
) -> Any:
    return {
        net_pb2.NetworkFormat.ACTIVATION_MISH: mish,
        net_pb2.NetworkFormat.ACTIVATION_RELU: nnx.relu,
        net_pb2.NetworkFormat.ACTIVATION_NONE: lambda x: x,
        net_pb2.NetworkFormat.ACTIVATION_TANH: nnx.tanh,
        net_pb2.NetworkFormat.ACTIVATION_SIGMOID: nnx.sigmoid,
        net_pb2.NetworkFormat.ACTIVATION_SELU: nnx.selu,
        net_pb2.NetworkFormat.ACTIVATION_SWISH: nnx.swish,
        net_pb2.NetworkFormat.ACTIVATION_SOFTMAX: nnx.softmax,
    }[activation]


def get_dtype(dtype: XlaShapeProto.Type) -> jnp.dtype:
    return {
        XlaShapeProto.PRED: jnp.bool_,
        XlaShapeProto.S4: jnp.int4,
        XlaShapeProto.S8: jnp.int8,
        XlaShapeProto.S16: jnp.int16,
        XlaShapeProto.S32: jnp.int32,
        XlaShapeProto.S64: jnp.int64,
        XlaShapeProto.U4: jnp.uint4,
        XlaShapeProto.U8: jnp.uint8,
        XlaShapeProto.U16: jnp.uint16,
        XlaShapeProto.U32: jnp.uint32,
        XlaShapeProto.U64: jnp.uint64,
        XlaShapeProto.F16: jnp.float16,
        XlaShapeProto.F32: jnp.float32,
        XlaShapeProto.BF16: jnp.bfloat16,
        XlaShapeProto.F64: jnp.float64,
        XlaShapeProto.F8E5M2: jnp.float8_e5m2,
        XlaShapeProto.F8E4M3FN: jnp.float8_e4m3fn,
        XlaShapeProto.F8E4M3B11FNUZ: jnp.float8_e4m3b11fnuz,
        XlaShapeProto.F8E5M2FNUZ: jnp.float8_e5m2fnuz,
        XlaShapeProto.F8E4M3FNUZ: jnp.float8_e4m3fnuz,
        XlaShapeProto.C64: jnp.complex64,
        XlaShapeProto.C128: jnp.complex128,
    }[dtype]
