"""Training daemon for the DirectML TUI.

Not usually run by hand -- lc0-directml-tui spawns it. Speaks the JSONL
protocol on stdin/stdout and logs to stderr, so stdout must stay clean.
"""

import argparse
import logging
import sys

# Before anything can trigger a backward pass.
from lczero_training.directml import device as dml_device  # noqa: F401


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        help="Root config textproto. May also arrive over the protocol.",
    )
    parser.add_argument("--checkpoint", help="Checkpoint directory override.")
    parser.add_argument("--device", help="Defaults to directml:0.")
    parser.add_argument(
        "--steps",
        type=int,
        help=(
            "Steps between checkpoints. Defaults to steps_per_network; acts "
            "as the checkpoint interval when --target-step is set."
        ),
    )
    parser.add_argument(
        "--target-step",
        type=int,
        help=(
            "Absolute step to train up to, checkpointing every --steps along "
            "the way. Keeps the data loader warm across the whole run."
        ),
    )
    parser.add_argument(
        "--output",
        help=(
            "Export a .pb.gz at every checkpoint. {step} and {datetime} are "
            "substituted. Defaults to export.destination_filename."
        ),
    )
    parser.add_argument("--kda-chunk-size", type=int, default=8)
    parser.add_argument(
        "--grad-accum",
        type=int,
        help=(
            "Micro-batches per optimizer step, overriding "
            "training.gradient_accumulation_steps. Multiplies the effective "
            "batch without raising peak memory."
        ),
    )
    parser.add_argument("--report-every", type=int, default=5)
    parser.add_argument(
        "--data-file-count",
        type=int,
        help=(
            "Limit the visible corpus to this many tar files per phase. "
            "Keeps the chunk-pool metadata -- and the DirectML allocator's "
            "peak -- small on long runs. Off when unset."
        ),
    )
    parser.add_argument(
        "--data-phase-step-interval",
        type=int,
        help=(
            "Steps per data phase; the tar window advances every this many "
            "steps. Defaults to 250000 when --data-file-count is set."
        ),
    )
    parser.add_argument(
        "--gc-every",
        type=int,
        default=500,
        help=(
            "Force a Python garbage collection every N optimizer steps. "
            "Frees tensors that reference cycles still hold; it cannot "
            "flush the DirectML allocator, which exposes no such API. 0 "
            "disables."
        ),
    )
    parser.add_argument(
        "--nan-check",
        choices=("report", "step", "skip", "off"),
        default="report",
        help=(
            "What to do about a non-finite gradient, which the optimizer "
            "would turn into non-finite weights. 'report' checks on the "
            "reporting cadence, no extra sync, and stops the run; 'step' "
            "checks every step (+7%% on a 518 ms step) and stops on the "
            "exact one; 'skip' checks every step but drops the bad gradient "
            "and keeps training, for an unattended run through a rough patch; "
            "'off' disables it. Checkpoints are refused when the weights are "
            "non-finite regardless of this setting."
        ),
    )
    parser.add_argument(
        "--max-skips",
        type=int,
        default=20,
        help=(
            "With --nan-check skip, give up after this many skipped updates. "
            "A few is a bad batch; a flood is a diverged run that skipping "
            "cannot rescue, and stopping lets the checkpoint guard roll back."
        ),
    )
    parser.add_argument(
        "--eval-every",
        type=int,
        default=0,
        help=(
            "Evaluate on the held-out split every N global steps. Needs a "
            "split config. The measurement runs in a child process, so it "
            "costs the trainer no device memory. 0 disables."
        ),
    )
    parser.add_argument("--eval-batches", type=int, default=50)
    parser.add_argument(
        "--eval-device",
        default="cpu",
        help=(
            "Device for the eval worker. CPU by default, and not as a "
            "fallback: a worker that builds a second DX12 context while the "
            "trainer holds 3.8 GB of shared memory did not finish importing "
            "inside 900 s, where the same 50 batches take 15 s on the CPU "
            "(5 s of that being `import torch`). Forward-only work at batch "
            "32 is simply not worth a GPU here. directml:0 still works if "
            "there is ever headroom to spare."
        ),
    )
    parser.add_argument(
        "--eval-timeout",
        type=float,
        default=300.0,
        help=(
            "Seconds before a stuck eval worker is killed and the eval "
            "skipped. Every second of it is a second training is paused, so "
            "this is a ceiling on the damage one bad eval can do -- 20x the "
            "measured 15 s, which leaves room for a contended machine."
        ),
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)

    # stderr only: stdout carries the protocol.
    logging.basicConfig(
        level=logging.INFO,
        format=(
            "%(levelname).1s%(asctime)s.%(msecs)03d %(name)s "
            "%(filename)s:%(lineno)d] %(message)s"
        ),
        datefmt="%m%d %H:%M:%S",
        stream=sys.stderr,
    )

    from lczero_training.directml.daemon import DirectMlTrainingDaemon

    daemon = DirectMlTrainingDaemon(
        config_filepath=args.config,
        checkpoint_dir=args.checkpoint,
        device_spec=args.device,
        kda_chunk_size=args.kda_chunk_size,
        grad_accum=args.grad_accum,
        steps=args.steps,
        report_every=args.report_every,
        target_step=args.target_step,
        output=args.output,
        eval_every=args.eval_every,
        eval_batches=args.eval_batches,
        eval_device=args.eval_device,
        eval_timeout=args.eval_timeout,
        data_file_count=args.data_file_count,
        data_phase_step_interval=args.data_phase_step_interval,
        gc_every=args.gc_every,
        nan_check=args.nan_check,
        max_skips=args.max_skips,
    )
    return daemon.run()


if __name__ == "__main__":
    sys.exit(main())
