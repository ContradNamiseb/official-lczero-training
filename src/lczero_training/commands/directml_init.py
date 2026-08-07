"""Seed a native PyTorch checkpoint from a Leela .pb.gz network.

    uv run lc0-directml-init --config CONFIG --lczero-model NET.pb.gz

Phase 8 of docs/directml_training_port.md. Imports the network directly;
Orbax is not involved.
"""

import argparse
import logging
import sys

# Before anything can trigger a backward pass: the autograd engine only
# counts the DirectML device if torch_directml is already imported.
from lczero_training.directml import device as dml_device  # noqa: F401


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config", required=True, help="Root config textproto."
    )
    parser.add_argument(
        "--lczero-model",
        "--lczero_model",
        dest="lczero_model",
        help="Leela .pb.gz to import. Omit to start from a fresh init.",
    )
    parser.add_argument(
        "--checkpoint",
        help="Checkpoint directory. Defaults to training.checkpoint.path.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Allow replacing an existing checkpoint.",
    )
    parser.add_argument(
        "--ignore-config-mismatch",
        action="store_true",
        help="Import even if the network's model config differs.",
    )
    parser.add_argument(
        "--override-training-steps",
        type=int,
        help="Start from this step instead of the network's own.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    from lczero_training.commands import configure_root_logging

    configure_root_logging(logging.INFO)
    args = _build_parser().parse_args(argv)

    import torch
    from google.protobuf import text_format

    from lczero_training.directml import checkpoint as checkpoint_io
    from lczero_training.directml.leela_to_torch import load_leela_file
    from lczero_training.directml.model import LczeroModel
    from lczero_training.directml.training import (
        build_model_and_optimizer,
        make_checkpoint,
    )
    from proto.root_config_pb2 import RootConfig

    config = RootConfig()
    with open(args.config, "r") as handle:
        text_format.Parse(handle.read(), config)

    directory = args.checkpoint or config.training.checkpoint.path
    if not directory:
        logging.error("No checkpoint path: set training.checkpoint.path.")
        return 1

    if checkpoint_io.latest_step(directory) is not None and not args.overwrite:
        logging.error(
            "Checkpoint already exists in %s; pass --overwrite to replace it.",
            directory,
        )
        return 1

    cpu = torch.device("cpu")
    if args.lczero_model:
        imported = load_leela_file(
            args.lczero_model,
            config.model,
            ignore_config_mismatch=args.ignore_config_mismatch,
        )
        model = imported.model
        step = imported.training_steps
        logging.info(
            "Imported %d weights from a network at step %d",
            imported.weights_imported,
            step,
        )
    else:
        logging.info("No --lczero-model given; initializing from scratch.")
        model = LczeroModel(config.model)
        step = 0

    if args.override_training_steps is not None:
        step = args.override_training_steps
        logging.info("Overriding training step to %d", step)

    # Built only for its optimizer, whose per-parameter state must exist and
    # carry the right step before the first resumed update.
    _, optimizer = build_model_and_optimizer(config, cpu)
    optimizer.set_step(step)

    written = checkpoint_io.save(
        directory,
        make_checkpoint(config, model, optimizer, step),
        max_to_keep=config.training.checkpoint.max_to_keep,
    )
    logging.info("Initialized checkpoint at step %d: %s", step, written)
    return 0


if __name__ == "__main__":
    sys.exit(main())
