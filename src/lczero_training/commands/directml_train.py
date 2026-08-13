"""Train natively on Windows with DirectML.

    uv run lc0-directml-train --config CONFIG

Restores the latest checkpoint, runs training.schedule.steps_per_network
steps against the native C++ data loader, and saves. Re-running continues
from the saved step. Pass --target-step to keep training in one process and
save after every steps_per_network steps until reaching that global step.
"""

import argparse
import contextlib
import logging
import sys

# Must precede any backward pass in the process.
from lczero_training.directml import device as dml_device


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
        "--device",
        default=None,
        help="Defaults to directml:0. Use cpu to compare.",
    )
    parser.add_argument(
        "--steps",
        type=int,
        help=(
            "Override training.schedule.steps_per_network (the checkpoint "
            "interval when --target-step is set)."
        ),
    )
    parser.add_argument(
        "--target-step",
        type=int,
        help=(
            "Train continuously until this absolute global step, saving a "
            "checkpoint every steps_per_network steps."
        ),
    )
    parser.add_argument(
        "--log-every",
        type=int,
        default=100,
        help=(
            "Log a training line every N steps. The default keeps a long run's "
            "console output small; pass 1 to watch every step."
        ),
    )
    parser.add_argument(
        "--log-file",
        help=(
            "Append all log output to this file instead of the console. The "
            "C++ data loader writes to stderr directly, so redirect that too "
            "if you want a completely quiet console: 2>>loader.log"
        ),
    )
    parser.add_argument(
        "--ignore-config-mismatch",
        action="store_true",
        help="Resume even if the checkpoint's config digest differs.",
    )
    parser.add_argument(
        "--tensorboard",
        help=(
            "TensorBoard log directory. Defaults to metrics.tensorboard_path "
            "from the config; pass 'off' to disable."
        ),
    )
    parser.add_argument(
        "--report-every",
        type=int,
        default=10,
        help="Write metrics to TensorBoard every N steps.",
    )
    parser.add_argument(
        "--eval-every",
        type=int,
        default=0,
        help=(
            "Evaluate on the held-out split every N global steps. Requires a "
            "config with a chunk_source_splitter (see "
            "scripts/add_test_split.py). The measurement runs in a child "
            "process, so it costs this one no device memory. 0 disables."
        ),
    )
    parser.add_argument(
        "--eval-batches",
        type=int,
        default=20,
        help="Batches per evaluation pass.",
    )
    parser.add_argument(
        "--eval-device",
        help=(
            "Device for the eval worker. Defaults to --device. Use cpu if "
            "this process leaves too little for a second one to allocate in."
        ),
    )
    parser.add_argument(
        "--eval-timeout",
        type=float,
        default=900.0,
        help=(
            "Seconds before a stuck eval worker is killed and the eval "
            "skipped. Raise it for --eval-device cpu."
        ),
    )
    parser.add_argument(
        "--kda-chunk-size",
        type=int,
        help=(
            "Override KdaConfig.chunk_size for this run. 8 is roughly 2.4x "
            "faster than the default 16 on DirectML. Safe to change freely: "
            "the chunked recurrence is mathematically identical for any "
            "chunk size, so it is a backend tunable, not part of the model."
        ),
    )
    parser.add_argument(
        "--grad-accum",
        type=int,
        help=(
            "Override TrainingConfig.gradient_accumulation_steps: how many "
            "micro-batches to accumulate into one optimizer step. The "
            "effective batch is this times the loader's batch_size, at the "
            "peak memory of one micro-batch. Not part of the checkpoint "
            "digest, so it can be changed between runs."
        ),
    )
    parser.add_argument(
        "--output",
        help=(
            "Also export a Leela .pb.gz after every checkpoint save. "
            "Defaults to export.destination_filename from the config, "
            "which may use {step} and {datetime}."
        ),
    )
    return parser


def _export_network(config, model, step: int, output: str | None) -> None:
    """Export the current model to the configured/requested .pb.gz path(s)."""
    import datetime

    from lczero_training.directml.torch_to_leela import (
        torch_to_leela,
        write_leela_file,
    )

    destinations = (
        [output] if output else list(config.export.destination_filename)
    )
    if not destinations:
        return
    model.eval()
    try:
        net = torch_to_leela(model, config.model, training_steps=step)
    finally:
        model.train()
    stamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    for template in destinations:
        write_leela_file(template.format(datetime=stamp, step=step), net)


def main(argv: list[str] | None = None) -> int:
    from lczero_training.commands import configure_root_logging

    args = _build_parser().parse_args(argv)
    configure_root_logging(logging.INFO)
    if args.log_file:
        # Replace the stderr handler rather than adding to it, so a long run
        # does not fill the console scrollback buffer.
        root = logging.getLogger()
        for handler in list(root.handlers):
            root.removeHandler(handler)
        # utf-8-sig, not utf-8: without a BOM, Notepad's encoding heuristic
        # misreads an all-ASCII log as UTF-16 and shows mojibake.
        file_handler = logging.FileHandler(args.log_file, encoding="utf-8-sig")
        file_handler.setFormatter(
            logging.Formatter(
                "%(levelname).1s%(asctime)s.%(msecs)03d %(name)s "
                "%(filename)s:%(lineno)d] %(message)s",
                datefmt="%m%d %H:%M:%S",
            )
        )
        root.addHandler(file_handler)
        print(f"Logging to {args.log_file}")

    from google.protobuf import text_format

    import gc
    from lczero_training.dataloader import make_dataloader
    from lczero_training.directml import checkpoint as checkpoint_io
    from lczero_training.directml.training import (
        batches_from_loader,
        build_model_and_optimizer,
        make_checkpoint,
        release_to_host,
        train,
        training_segments,
    )
    from proto.root_config_pb2 import RootConfig

    config = RootConfig()
    with open(args.config, "r") as handle:
        text_format.Parse(handle.read(), config)

    directory = args.checkpoint or config.training.checkpoint.path
    if not directory:
        logging.error("No checkpoint path: set training.checkpoint.path.")
        return 1

    device = dml_device.resolve_device(args.device)
    logging.info("Training on %s", device)
    if device.type == "privateuseone":
        logging.info("Adapter: %s", dml_device.adapter_name(device.index or 0))

    restored = checkpoint_io.load_latest(
        directory,
        expected_digest=checkpoint_io.config_digest(config),
        ignore_config_mismatch=args.ignore_config_mismatch,
    )
    if restored is None:
        logging.error(
            "No checkpoint in %s. Run lc0-directml-init first.", directory
        )
        return 1

    if args.kda_chunk_size:
        # Applied after the digest check above: chunk_size is inside
        # KdaConfig and so is part of the hashed model config, but it does
        # not change what the model computes.
        config.model.encoder.kda.chunk_size = args.kda_chunk_size
        logging.info("KDA chunk size overridden to %d", args.kda_chunk_size)

    if args.grad_accum:
        config.training.gradient_accumulation_steps = args.grad_accum
        logging.info(
            "Accumulating %d micro-batches per optimizer step",
            args.grad_accum,
        )

    model, optimizer = build_model_and_optimizer(config, device)
    checkpoint_io.load_state_dict_into(model, restored.model_state)
    model.to(device)
    if restored.optimizer_state is not None:
        checkpoint_io.load_optimizer_state_dict_into(
            optimizer,
            restored.optimizer_state,
            model,
            restored.model_state,
            config.training.optimizer.nadamw.decay_selector,
        )
    logging.info("Resumed from step %d", restored.step)

    checkpoint_interval = (
        args.steps or config.training.schedule.steps_per_network
    )
    if checkpoint_interval <= 0:
        logging.error("steps_per_network must be positive.")
        return 1
    target_step = args.target_step or restored.step + checkpoint_interval
    if target_step <= restored.step:
        logging.error(
            "Target step %d has already been reached (current step %d).",
            target_step,
            restored.step,
        )
        return 1

    from lczero_training.directml import metrics as metrics_sinks

    reporters: list[metrics_sinks.ClosableReporter] = []
    tensorboard_dir = args.tensorboard or config.metrics.tensorboard_path
    if tensorboard_dir and tensorboard_dir.lower() != "off":
        # One directory per split, named after the run, so TensorBoard lists
        # this beside the old leelalogs runs instead of merging them.
        reporters.append(
            metrics_sinks.TensorboardReporter(
                metrics_sinks.run_logdir(
                    tensorboard_dir, config.name or "directml", "train"
                )
            )
        )
    else:
        logging.info("TensorBoard disabled; no metrics will be written.")
        tensorboard_dir = ""
    # Folded back into the config, because the eval worker writes the test
    # run from the config it is handed and has no --tensorboard of its own.
    # Without this, --tensorboard would move the train run and leave the test
    # run behind in whatever the config said.
    config.metrics.tensorboard_path = tensorboard_dir

    # A split config exposes named outputs; a plain one exposes exactly one
    # with an empty alias.
    aliases = [
        spec.split(":", 1)[0] if ":" in spec else ""
        for spec in config.data_loader.output
    ]
    has_split = "test" in aliases
    train_alias = "train" if has_split else ""
    if args.eval_every and not has_split:
        logging.error(
            "--eval-every needs a config with a test split. Generate one "
            "with: python scripts/add_test_split.py %s out.textproto "
            "--test-weight 5",
            args.config,
        )
        return 1

    logging.info("Starting the data loader")
    loader = make_dataloader(config.data_loader)
    # Mutated in place by train() so a crash can still report how far it got.
    progress: dict[str, int] = {"step": restored.step}
    failed = False
    batches = batches_from_loader(loader, device, train_alias)
    final_step = restored.step

    eval_hook = None
    if args.eval_every:
        from lczero_training.directml import subprocess_eval

        def log_eval(step: int, scalars: dict[str, float]) -> None:
            logging.info(
                "eval @ %d  %s",
                step,
                "  ".join(
                    f"{name}={value:.4f}"
                    for name, value in sorted(scalars.items())
                    if name in ("total", "policy/main_ce", "value/winner")
                ),
            )

        # The forward passes run in a child process that exits, so they cost
        # this one no device memory -- and the TensorBoard test run is written
        # there too, which is why no writer for it is registered here.
        eval_hook = subprocess_eval.make_eval_hook(
            config=config,
            model=model,
            loader=loader,
            config_filepath=args.config,
            batch_count=args.eval_batches,
            device_spec=args.eval_device or args.device,
            kda_chunk_size=args.kda_chunk_size,
            timeout=args.eval_timeout,
            on_scalars=log_eval,
        )
    try:
        for segment_steps in training_segments(
            restored.step, target_step, checkpoint_interval
        ):
            segment_start = final_step
            final_step = train(
                config=config,
                model=model,
                optimizer=optimizer,
                batches=batches,
                device=device,
                start_step=segment_start,
                steps=segment_steps,
                log_every=args.log_every,
                progress=progress,
                reporters=reporters,
                report_every=args.report_every,
                eval_hook=eval_hook,
                eval_every=args.eval_every,
            )
            written = checkpoint_io.save(
                directory,
                make_checkpoint(config, model, optimizer, final_step),
                max_to_keep=config.training.checkpoint.max_to_keep,
            )
            logging.info("Saved step %d: %s", final_step, written)
            _export_network(config, model, final_step, args.output)
    except (RuntimeError, KeyboardInterrupt) as error:
        # Running out of shared GPU memory part-way through is a real
        # possibility on an integrated adapter. Losing every completed step
        # to it is not acceptable, so checkpoint what finished and let the
        # next invocation resume from there.
        failed = True
        final_step = progress["step"]
        from lczero_training.directml.training import describe_error

        logging.error(
            "Training stopped at step %d: %s", final_step, describe_error(error)
        )
    finally:
        logging.info("Stopping the data loader")
        try:
            loader.stop()
        except Exception:
            logging.exception("Data loader failed to stop cleanly")
        metrics_sinks.close_all(reporters)
        with contextlib.suppress(Exception):
            release_to_host(model, optimizer)
        gc.collect()

    if failed and final_step > restored.step:
        written = checkpoint_io.save(
            directory,
            make_checkpoint(config, model, optimizer, final_step),
            max_to_keep=config.training.checkpoint.max_to_keep,
        )
        logging.info("Saved step %d: %s", final_step, written)
        _export_network(config, model, final_step, args.output)

    if final_step == restored.step:
        logging.warning("No steps completed; leaving the checkpoint alone.")

    if failed:
        logging.error(
            "Re-run to continue from step %d. If this was a memory error, "
            "lower tensor_generator.batch_size or "
            "shuffling_frame_sampler.reservoir_size_per_thread.",
            final_step,
        )
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
