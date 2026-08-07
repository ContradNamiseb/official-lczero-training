"""Import a Leela ``.pb.gz`` network into a PyTorch model.

Phase 6 of docs/directml_training_port.md.

Rather than re-deriving the block-by-block weight traversal, this reuses
the existing, tested `leela_to_jax` importer and converts the resulting
NNX state into PyTorch through `weight_map`. That inherits the
dequantization, the per-block MHA/KDA dispatch, the once-only shared
Smolgen restore, and the embedding-plane rescale, instead of duplicating
them -- and `weight_map` is exercised in both directions by the tests, so
a mistake in it cannot pass unnoticed.

JAX is needed to *import* a network, not to train one; this module is
touched once by `lc0-directml-init` and never by the training loop.
Orbax is not involved at all.
"""

from __future__ import annotations

import dataclasses
import gzip
import logging

from flax import nnx

from lczero_training.convert.leela_to_jax import (
    LeelaImportOptions,
    fix_older_weights_file,
    leela_to_jax,
)
from lczero_training.convert.leela_to_modelconfig import leela_to_modelconfig
from lczero_training.model.model import LczeroModel as JaxModel
from proto import hlo_pb2, model_config_pb2, net_pb2

from .model import LczeroModel
from .weight_map import copy_jax_to_torch

logger = logging.getLogger(__name__)


@dataclasses.dataclass
class ImportedNetwork:
    model: LczeroModel
    config: model_config_pb2.ModelConfig
    training_steps: int
    weights_imported: int


def read_leela_net(path: str) -> net_pb2.Net:
    """Read a compressed Leela network straight from disk."""
    net = net_pb2.Net()
    with gzip.open(path, "rb") as handle:
        contents = handle.read()
        assert isinstance(contents, bytes)
        net.ParseFromString(contents)
    fix_older_weights_file(net)
    return net


def leela_config(
    net: net_pb2.Net,
    compute_dtype: int = hlo_pb2.XlaShapeProto.F32,
) -> model_config_pb2.ModelConfig:
    """The exact model configuration this network encodes."""
    return leela_to_modelconfig(net, hlo_pb2.XlaShapeProto.F32, compute_dtype)


def leela_to_torch(
    net: net_pb2.Net,
    expected_config: model_config_pb2.ModelConfig | None = None,
    *,
    ignore_config_mismatch: bool = False,
    compute_dtype: int = hlo_pb2.XlaShapeProto.F32,
) -> ImportedNetwork:
    """Build a PyTorch model holding this network's weights.

    When ``expected_config`` is given it must match the configuration the
    network itself encodes, or the import is refused -- training on
    silently mismatched weights is far worse than failing here.
    """
    config = leela_config(net, compute_dtype)

    if expected_config is not None and config != expected_config:
        if not ignore_config_mismatch:
            raise ValueError(
                "The lczero model configuration differs from the one in the "
                "config file.\nConfig file model config:\n"
                f"{expected_config}\nLeela model config:\n{config}"
            )
        logger.warning(
            "lczero model configuration differs from the config file (ignored)"
        )

    # The JAX model is scaffolding: leela_to_jax needs somewhere to put the
    # dequantized weights, and weight_map reads them straight back out.
    jax_state = leela_to_jax(
        net,
        LeelaImportOptions(
            weights_dtype=hlo_pb2.XlaShapeProto.F32,
            compute_dtype=compute_dtype,
        ),
    )
    jax_model = JaxModel(config=config, rngs=nnx.Rngs(params=0))
    nnx.update(jax_model, jax_state)

    model = LczeroModel(config)
    imported = copy_jax_to_torch(model, jax_model)

    training_steps = net.training_params.training_steps
    logger.info(
        "Imported %d weight arrays from a network at step %d",
        imported,
        training_steps,
    )
    return ImportedNetwork(
        model=model,
        config=config,
        training_steps=training_steps,
        weights_imported=imported,
    )


def load_leela_file(
    path: str,
    expected_config: model_config_pb2.ModelConfig | None = None,
    *,
    ignore_config_mismatch: bool = False,
) -> ImportedNetwork:
    """Read and import a ``.pb.gz`` in one step."""
    logger.info("Reading Leela network from %s", path)
    return leela_to_torch(
        read_leela_net(path),
        expected_config,
        ignore_config_mismatch=ignore_config_mismatch,
    )
