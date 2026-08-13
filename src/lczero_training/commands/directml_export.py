"""Export a native PyTorch checkpoint to a Leela .pb.gz network.

    uv run lc0-directml-export --config CONFIG --output network.pb.gz

Phase 9 of docs/directml_training_port.md.
"""

import argparse
import logging
import sys


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config", required=True, help="Root config textproto."
    )
    parser.add_argument(
        "--checkpoint",
        help="Checkpoint directory. Defaults to training.checkpoint.path.",
    )
    parser.add_argument(
        "--output",
        help=(
            "Destination .pb.gz. Defaults to export.destination_filename "
            "from the config, which may use {step} and {datetime}."
        ),
    )
    parser.add_argument("--license", dest="license_text", help="License text.")
    parser.add_argument(
        "--min-version",
        default="0.28",
        help="Minimum lc0 version to record in the network.",
    )
    parser.add_argument(
        "--ignore-config-mismatch",
        action="store_true",
        help="Export even if the checkpoint's config digest differs.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    from lczero_training.commands import configure_root_logging

    configure_root_logging(logging.INFO)
    args = _build_parser().parse_args(argv)

    import datetime

    from google.protobuf import text_format

    from lczero_training.directml import checkpoint as checkpoint_io
    from lczero_training.directml.model import LczeroModel
    from lczero_training.directml.torch_to_leela import (
        torch_to_leela,
        write_leela_file,
    )
    from proto.root_config_pb2 import RootConfig

    config = RootConfig()
    with open(args.config, "r") as handle:
        text_format.Parse(handle.read(), config)

    directory = args.checkpoint or config.training.checkpoint.path
    if not directory:
        logging.error("No checkpoint path: set training.checkpoint.path.")
        return 1

    restored = checkpoint_io.load_latest(
        directory,
        expected_digest=checkpoint_io.config_digest(config),
        ignore_config_mismatch=args.ignore_config_mismatch,
    )
    if restored is None:
        logging.error("No checkpoint found in %s.", directory)
        return 1
    logging.info("Exporting checkpoint at step %d", restored.step)

    model = LczeroModel(config.model)
    checkpoint_io.load_state_dict_into(model, restored.model_state)
    model.eval()

    net = torch_to_leela(
        model,
        config.model,
        training_steps=restored.step,
        license_text=args.license_text,
        min_version=args.min_version,
    )

    destinations = []
    if args.output:
        destinations.append(args.output)
    else:
        destinations.extend(config.export.destination_filename)
    if not destinations:
        logging.error(
            "No output: pass --output or set export.destination_filename."
        )
        return 1

    stamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    for template in destinations:
        write_leela_file(
            template.format(datetime=stamp, step=restored.step), net
        )
    return 0


if __name__ == "__main__":
    sys.exit(main())
