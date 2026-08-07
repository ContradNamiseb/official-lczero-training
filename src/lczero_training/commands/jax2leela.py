import argparse
import gzip
import logging
import os
import sys

import orbax.checkpoint as ocp
from google.protobuf import text_format

from lczero_training.commands import configure_root_logging
from lczero_training.convert.jax_to_leela import (
    LeelaExportOptions,
    jax_to_leela,
)
from lczero_training.training.state import TrainingState
from proto import model_config_pb2
from proto.root_config_pb2 import RootConfig

# Below this, a KDA network's 8-direction scan / no-RMSNorm / local-conv
# features are not understood by the engine. Mirrors LC0_MINOR_WITH_KDA_*
# in stable-branch/tf/net.py -- these constants mirror the engine's own
# versioning and are not ours to invent; upstream lc0 maintainers own
# version assignment, so this only ever matches the TF reference, never
# introduces a new minor version.
_KDA_MIN_VERSION = (0, 33, 0)


def _parse_version(version_str: str) -> tuple[int, int, int]:
    parts = (version_str.lstrip("v").split(".") + ["0", "0"])[:3]
    major, minor, patch = (int(p) for p in parts)
    return (major, minor, patch)


def _format_version(version: tuple[int, int, int]) -> str:
    return ".".join(str(p) for p in version)


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Export JAX checkpoint to Leela format."
    )
    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Path to the training config file.",
    )
    parser.add_argument(
        "--output",
        type=str,
        required=True,
        help="Output path for the Leela network (.pb.gz).",
    )
    parser.add_argument(
        "--export-swa",
        action="store_true",
        help="Export SWA model instead of regular model_state.",
    )
    parser.add_argument(
        "--min-version",
        type=str,
        default="0.31",
        help="Minimum lc0 version for exported network (default: 0.31).",
    )
    return parser


def jax2leela(
    config_filename: str,
    output_path: str,
    export_swa: bool,
    min_version: str,
) -> None:
    config = RootConfig()
    logging.info("Reading configuration from %s", config_filename)
    with open(config_filename, "r") as f:
        text_format.Parse(f.read(), config)

    if not config.training.checkpoint.path:
        logging.error("Checkpoint path must be set in the configuration.")
        sys.exit(1)

    logging.info("Loading checkpoint from %s", config.training.checkpoint.path)
    checkpoint_mgr = ocp.CheckpointManager(
        config.training.checkpoint.path,
        options=ocp.CheckpointManagerOptions(create=True),
    )

    empty_state = TrainingState.new_from_config(
        model_config=config.model,
        training_config=config.training,
    )

    restored_state = checkpoint_mgr.restore(
        checkpoint_mgr.latest_step(), args=ocp.args.PyTreeRestore(empty_state)
    )
    assert isinstance(restored_state, TrainingState)
    logging.info(
        "Restored checkpoint at step %d", restored_state.jit_state.step
    )

    if export_swa:
        if restored_state.jit_state.swa_state is None:
            logging.error(
                "SWA export requested but SWA state is None in checkpoint."
            )
            sys.exit(1)
        export_state = restored_state.jit_state.swa_state
        logging.info("Exporting SWA model")
    else:
        export_state = restored_state.jit_state.model_state
        logging.info("Exporting regular model")

    if (
        config.model.encoder.mixer_type == model_config_pb2.MIXER_KDA
        and _parse_version(min_version) < _KDA_MIN_VERSION
    ):
        logging.warning(
            "KDA network detected; overriding min-version %s to %s.",
            min_version,
            _format_version(_KDA_MIN_VERSION),
        )
        min_version = _format_version(_KDA_MIN_VERSION)

    options = LeelaExportOptions(
        min_version=min_version,
        num_heads=restored_state.num_heads,
        license=None,
        training_steps=restored_state.jit_state.step,
    )

    logging.info("Converting to Leela format")
    net = jax_to_leela(
        jax_weights=export_state,
        export_options=options,
        encoder_config=config.model.encoder,
    )

    logging.info("Serializing network")
    network_bytes = gzip.compress(net.SerializeToString())

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    logging.info("Writing network to %s", output_path)
    with open(output_path, "wb") as f:
        f.write(network_bytes)

    logging.info("Export complete")


def main(argv: list[str] | None = None) -> int:
    configure_root_logging(logging.INFO)

    parser = _build_parser()
    args = parser.parse_args(argv)

    jax2leela(
        config_filename=args.config,
        output_path=args.output,
        export_swa=args.export_swa,
        min_version=args.min_version,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
