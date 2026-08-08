"""One held-out evaluation, then exit. Not usually run by hand.

The trainer spawns this every ``--eval-every`` steps and waits for it. Its
whole reason to exist is the exit: every device block it allocates goes back
to the OS when the process ends, which is the only way DirectML ever returns
one. See ``directml/subprocess_eval.py`` for the trainer's half and for why
the batches arrive as a file rather than from a data loader here.

Reads ``--work-dir`` for the config, the weights and the batches the trainer
left; writes the TensorBoard ``-test`` run and ``scalars.json`` back into it.
Logs to stderr, because the parent's stdout may be carrying the TUI protocol.
"""

import argparse
import json
import logging
import pathlib
import sys

# Before anything can touch the device.
from lczero_training.directml import device as dml_device  # noqa: F401

logger = logging.getLogger("lc0-directml-eval-worker")


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--work-dir",
        required=True,
        help="Directory holding the request, and where results are written.",
    )
    parser.add_argument(
        "--step",
        type=int,
        required=True,
        help="Global step to tag the scalars with.",
    )
    parser.add_argument(
        "--device",
        help=(
            "Defaults to directml:0. Use cpu when the trainer leaves too "
            "little device memory for a second process to allocate in."
        ),
    )
    parser.add_argument("--kda-chunk-size", type=int)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    logging.basicConfig(
        level=logging.INFO,
        format=(
            "%(levelname).1s%(asctime)s.%(msecs)03d %(name)s "
            "%(filename)s:%(lineno)d] %(message)s"
        ),
        datefmt="%m%d %H:%M:%S",
        stream=sys.stderr,
    )

    import torch
    from google.protobuf import text_format

    from lczero_training.directml import derived_metrics
    from lczero_training.directml import metrics as metrics_sinks
    from lczero_training.directml import subprocess_eval
    from lczero_training.directml.losses import LczeroLoss
    from lczero_training.directml.model import LczeroModel
    from lczero_training.directml.training import TrainingBatch, evaluate
    from proto.root_config_pb2 import RootConfig

    work_dir = pathlib.Path(args.work_dir)
    config = RootConfig()
    text_format.Parse(
        (work_dir / subprocess_eval.CONFIG_FILE).read_text(), config
    )
    if args.kda_chunk_size:
        config.model.encoder.kda.chunk_size = args.kda_chunk_size

    device = dml_device.resolve_device(args.device)
    model = LczeroModel(config.model)
    payload = torch.load(
        work_dir / subprocess_eval.WEIGHTS_FILE,
        map_location="cpu",
        weights_only=False,
    )
    model.load_state_dict(payload["model_state"])
    model.to(device)
    # The test run carries the KDA gate diagnostics, and the mixers only
    # capture them when asked. In-process eval used to get this by accident,
    # from whatever the training loop had left the flag set to; here it has
    # to be deliberate. The extra reductions cost nothing worth counting
    # across --eval-batches forward passes in a process that then exits.
    derived_metrics.set_kda_stats_collection(model, True)

    arrays, count = subprocess_eval.read_batches(
        work_dir / subprocess_eval.BATCHES_FILE
    )
    # Moved to the device one batch at a time, as the evaluation consumes
    # them, rather than all 50 up front.
    batches = (TrainingBatch.from_arrays(item, device) for item in arrays)

    scalars = evaluate(
        model=model,
        loss_fn=LczeroLoss(config.training.losses),
        batches=batches,
        batch_count=count,
    )
    if not scalars:
        logger.error("No batches to evaluate in %s", work_dir)
        return 1

    # Written here rather than by the trainer, so no writer for the test run
    # stays open in the training process.
    if config.metrics.tensorboard_path:
        writer = metrics_sinks.TensorboardReporter(
            metrics_sinks.run_logdir(
                config.metrics.tensorboard_path,
                config.name or "directml",
                "test",
            )
        )
        try:
            writer(args.step, scalars)
        finally:
            writer.close()

    (work_dir / subprocess_eval.RESULT_FILE).write_text(json.dumps(scalars))
    logger.info(
        "Evaluated step %d over %d batch(es) on %s: total=%.4f",
        args.step,
        count,
        device,
        scalars.get("total", float("nan")),
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
