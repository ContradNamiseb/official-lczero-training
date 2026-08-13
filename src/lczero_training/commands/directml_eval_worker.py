"""One held-out evaluation, then exit. Not usually run by hand.

The trainer spawns this every ``--eval-every`` steps and waits for it. Its
whole reason to exist is the exit: every device block it allocates goes back
to the OS when the process ends, which is the only way DirectML ever returns
one. See ``directml/subprocess_eval.py`` for the trainer's half and for why
the batches arrive as a file rather than from a data loader here.

Reads ``--work-dir`` for the config, the weights and the batches the trainer
left; writes the TensorBoard ``-test`` run and ``scalars.json`` back into it.

**Nothing heavy may be imported at module scope here, and logging has to be
configured before the first such import.** The first version of this file
imported ``directml.device`` -- and so torch and torch_directml -- on line
one, before ``basicConfig``. At step 195000 of a real run that import did not
return inside the trainer's 900-second timeout, and because logging did not
yet exist the worker was killed without emitting one byte: a fifteen-minute
stall with no evidence in it at all. Every stage below therefore logs before
it starts, to stderr *and* to ``worker.log`` in the work directory, so a
worker that dies invisibly to its parent still leaves a record on disk.
"""

import argparse
import json
import logging
import pathlib
import sys
import time

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
    parser.add_argument(
        "--log-file",
        help=(
            "Mirror this worker's log here as well as to stderr. The trainer "
            "passes it and reads it back if the eval fails."
        ),
    )
    return parser


def _configure_logging(log_file: str | None) -> None:
    """stderr, plus a file if the parent asked for one, before anything heavy.

    The file handler is the point. The parent redirects our stdout and may
    itself be killed, or lose the stderr pipe with it; a log on disk survives
    all of that and is where to look when an eval produces no scalars.
    """
    formatter = logging.Formatter(
        "%(levelname).1s%(asctime)s.%(msecs)03d %(name)s "
        "%(filename)s:%(lineno)d] %(message)s",
        datefmt="%m%d %H:%M:%S",
    )
    handlers: list[logging.Handler] = [logging.StreamHandler(sys.stderr)]
    if log_file:
        # Truncated per run: this is the record of the eval in progress, not
        # a history, and the trainer reads it back on failure.
        handlers.append(
            logging.FileHandler(log_file, mode="w", encoding="utf-8")
        )
    root = logging.getLogger()
    root.setLevel(logging.INFO)
    for handler in handlers:
        handler.setFormatter(formatter)
        root.addHandler(handler)


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    work_dir = pathlib.Path(args.work_dir)
    _configure_logging(args.log_file)
    logger.info(
        "Eval worker for step %d starting in %s on device %s",
        args.step,
        work_dir,
        args.device or "the default",
    )

    # Imported here, not at module scope, and only once logging exists. On a
    # memory-starved machine `import torch` alone can take minutes, and
    # importing torch_directml builds a second DX12 context alongside the
    # trainer's; both used to happen before there was any way to say so.
    started = time.perf_counter()
    logger.info("Importing torch")
    import torch

    from google.protobuf import text_format

    logger.info("Importing the DirectML backend")
    from lczero_training.directml import checkpoint as checkpoint_io
    from lczero_training.directml import device as dml_device
    from lczero_training.directml import derived_metrics
    from lczero_training.directml import metrics as metrics_sinks
    from lczero_training.directml import subprocess_eval
    from lczero_training.directml.losses import LczeroLoss
    from lczero_training.directml.model import LczeroModel
    from lczero_training.directml.training import TrainingBatch, evaluate
    from proto.root_config_pb2 import RootConfig

    logger.info("Imports done after %.1fs", time.perf_counter() - started)

    config = RootConfig()
    text_format.Parse(
        (work_dir / subprocess_eval.CONFIG_FILE).read_text(), config
    )
    if args.kda_chunk_size:
        config.model.encoder.kda.chunk_size = args.kda_chunk_size

    logger.info("Resolving the device")
    device = dml_device.resolve_device(args.device)
    logger.info("Building the model on %s", device)
    model = LczeroModel(config.model)
    payload = torch.load(
        work_dir / subprocess_eval.WEIGHTS_FILE,
        map_location="cpu",
        weights_only=False,
    )
    # tolerate extra tensors in the saved state dict if a policy_head was
    # dropped from the config mid-run; refuse a missing tensor.
    checkpoint_io.load_state_dict_into(model, payload["model_state"])
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

    logger.info("Evaluating %d batch(es)", count)
    forward_started = time.perf_counter()
    scalars = evaluate(
        model=model,
        loss_fn=LczeroLoss(config.training.losses),
        batches=batches,
        batch_count=count,
    )
    logger.info(
        "%d forward pass(es) in %.1fs",
        count,
        time.perf_counter() - forward_started,
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
    # Total alone is not enough to diagnose a stalled head: the per-step
    # training log is too noisy to read at effective batch 64, so the held-out
    # eval line is the place a flat policy head will show up. The headline
    # per-loss scalars follow total on the same line so a plain
    # `grep "Evaluated step"` covers it without opening TensorBoard.
    headline = (
        "policy/main_ce=%(policy/main_ce).4f"
        "  value/winner=%(value/winner).4f"
        "  movesleft/main=%(movesleft/main).4f"
    )
    # Missing heads (e.g. an omitted moves-left config) should not raise here:
    # use the sentinel and let the formatted value print nan.
    fmt = {
        key: scalars.get(key, float("nan"))
        for key in (
            "policy/main_ce",
            "value/winner",
            "movesleft/main",
        )
    }
    logger.info(
        "Evaluated step %d over %d batch(es) on %s: total=%.4f  "
        + headline % fmt,
        args.step,
        count,
        device,
        scalars.get("total", float("nan")),
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
