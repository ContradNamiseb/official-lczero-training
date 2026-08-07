"""Export a PyTorch model back to a Leela ``.pb.gz`` network.

Phase 9 of docs/directml_training_port.md. The mirror image of
leela_to_torch: rather than re-deriving the protobuf mapping and the
per-layer uint16 quantization, this converts the PyTorch parameters into an
NNX state through `weight_map` and hands that to the existing, tested
`jax_to_leela`. Layout transposes, the KDA depthwise kernel's special case,
the MHA/KDA submessage dispatch, and the head/step metadata all come along
for free.
"""

from __future__ import annotations

import gzip
import logging
import os
from typing import Optional

from flax import nnx

from lczero_training.convert.jax_to_leela import (
    LeelaExportOptions,
    jax_to_leela,
)
from lczero_training.model.model import LczeroModel as JaxModel
from proto import model_config_pb2, net_pb2

from .model import LczeroModel
from .weight_map import copy_torch_to_jax

logger = logging.getLogger(__name__)

# Matches the minimum engine version the JAX exporter targets.
DEFAULT_MIN_VERSION = "0.28"


def torch_to_leela(
    model: LczeroModel,
    config: model_config_pb2.ModelConfig,
    *,
    training_steps: Optional[int] = None,
    num_heads: Optional[int] = None,
    license_text: Optional[str] = None,
    min_version: str = DEFAULT_MIN_VERSION,
) -> net_pb2.Net:
    """Build a Leela network protobuf from a trained PyTorch model."""
    # Scaffolding, exactly as in leela_to_torch: somewhere for the weights
    # to live in the layout jax_to_leela expects to read.
    jax_model = JaxModel(config=config, rngs=nnx.Rngs(params=0))
    exported = copy_torch_to_jax(model, jax_model)
    state = nnx.state(jax_model)

    options = LeelaExportOptions(
        min_version=min_version,
        num_heads=(
            num_heads if num_heads is not None else config.encoder.heads
        ),
        license=license_text,
        training_steps=training_steps,
    )
    net = jax_to_leela(
        jax_weights=state,
        export_options=options,
        encoder_config=config.encoder,
    )
    logger.info(
        "Exported %d weight arrays at step %s", exported, training_steps
    )
    return net


def write_leela_file(
    path: str | os.PathLike,
    net: net_pb2.Net,
) -> None:
    """Write the network gzipped, atomically."""
    destination = os.fspath(path)
    temporary = destination + ".tmp"
    parent = os.path.dirname(destination)
    if parent:
        os.makedirs(parent, exist_ok=True)
    with gzip.open(temporary, "wb") as handle:
        handle.write(net.SerializeToString())
    os.replace(temporary, destination)
    logger.info("Wrote %s", destination)
