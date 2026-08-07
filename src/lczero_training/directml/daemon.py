"""Training daemon for the DirectML port, driving the TUI.

Deliberately *not* a subclass of daemon/daemon.py: that one calls
``signal.pthread_sigmask`` and ``signal.sigwait``, which do not exist on
Windows, so it cannot run here at all. It also drives the JAX pipeline with
its own wait-for-chunks state machine; this trainer just runs
``steps_per_network`` steps and exits, which is a much smaller contract.

Speaks the same JSONL protocol as the JAX daemon, emitting the
TrainingMetricsPayload the TUI renders.
"""

from __future__ import annotations

import logging
import sys
import threading
import time
from typing import Optional

from lczero_training.daemon.protocol.communicator import Communicator
from lczero_training.daemon.protocol.messages import (
    StartTrainingImmediatelyPayload,
    StartTrainingPayload,
    TrainingMetricsPayload,
    TrainingPhase,
)

logger = logging.getLogger(__name__)


class DirectMlTrainingDaemon:
    """Runs one DirectML training run and streams its progress."""

    def __init__(
        self,
        *,
        config_filepath: Optional[str] = None,
        checkpoint_dir: Optional[str] = None,
        device_spec: Optional[str] = None,
        kda_chunk_size: Optional[int] = None,
        grad_accum: Optional[int] = None,
        steps: Optional[int] = None,
        report_every: int = 5,
        target_step: Optional[int] = None,
        output: Optional[str] = None,
        eval_every: int = 0,
        eval_batches: int = 50,
    ):
        self._config_filepath = config_filepath
        self._checkpoint_dir = checkpoint_dir
        self._device_spec = device_spec
        self._kda_chunk_size = kda_chunk_size
        self._grad_accum = grad_accum
        self._steps = steps
        self._report_every = report_every
        self._target_step = target_step
        self._output = output
        self._eval_every = eval_every
        self._eval_batches = eval_batches

        self._extra_reporters: list = []
        self._stop = threading.Event()
        self._communicator = Communicator(self, sys.stdin, sys.stdout)
        self._communicator_thread = threading.Thread(
            target=self._communicator.run, daemon=True
        )
        self._communicator_thread.start()

    # --- protocol handlers -------------------------------------------------

    def on_start_training(self, payload: StartTrainingPayload) -> None:
        self._config_filepath = payload.config_filepath

    def on_start_training_immediately(
        self, payload: StartTrainingImmediatelyPayload
    ) -> None:
        # This trainer never waits for chunks, so there is nothing to skip.
        logger.info(
            "Immediate-start requested; training already runs on start."
        )

    # --- helpers -----------------------------------------------------------

    def _export(self, config, model, step: int) -> None:
        """Write a .pb.gz for this checkpoint, if a destination is set."""
        import datetime

        from .torch_to_leela import torch_to_leela, write_leela_file

        destinations = (
            [self._output]
            if self._output
            else list(config.export.destination_filename)
        )
        if not destinations:
            return
        was_training = model.training
        model.eval()
        try:
            net = torch_to_leela(model, config.model, training_steps=step)
        finally:
            if was_training:
                model.train()
        stamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
        for template in destinations:
            write_leela_file(template.format(datetime=stamp, step=step), net)

    def _make_eval_hook(
        self, config, model, device, loader, has_split: bool, device_name: str
    ):
        """An eval callback, or None when evaluation is off or impossible."""
        if not self._eval_every:
            return None
        if not has_split:
            logger.warning(
                "--eval-every ignored: this config has no test split. "
                "Generate one with scripts/add_test_split.py."
            )
            return None

        from . import metrics as metrics_sinks
        from .losses import LczeroLoss
        from .training import batches_from_loader, evaluate

        eval_batches = batches_from_loader(loader, device, "test")
        eval_loss_fn = LczeroLoss(config.training.losses)
        writer = metrics_sinks.TensorboardReporter(
            metrics_sinks.run_logdir(
                config.metrics.tensorboard_path,
                config.name or "directml",
                "test",
            )
        )
        # Driven only from here, never from the training loop's reporters,
        # so train metrics cannot leak into the test run.
        self._extra_reporters.append(writer)

        def hook(step: int) -> None:
            scalars = evaluate(
                model=model,
                loss_fn=eval_loss_fn,
                batches=eval_batches,
                batch_count=self._eval_batches,
            )
            if not scalars:
                return
            writer(step, scalars)
            self._emit(
                phase=TrainingPhase.TRAINING.value,
                step=step,
                device=device_name,
                message=f"eval total={scalars.get('total', float('nan')):.4f}",
            )

        return hook

    # --- reporting ---------------------------------------------------------

    def _emit(self, **fields) -> None:
        try:
            self._communicator.send(TrainingMetricsPayload(**fields))
        except Exception:  # noqa: BLE001 - a broken pipe must not kill training
            logger.exception("Failed to send a metrics message")

    def run(self) -> int:
        while self._config_filepath is None and not self._stop.is_set():
            logger.info("Waiting for training config...")
            time.sleep(1)
        if self._config_filepath is None:
            return 1

        from google.protobuf import text_format

        from lczero_training.dataloader import make_dataloader
        from lczero_training.directml import checkpoint as checkpoint_io
        from lczero_training.directml import device as dml_device
        from lczero_training.directml import metrics as metrics_sinks
        from lczero_training.directml.training import (
            batches_from_loader,
            build_model_and_optimizer,
            make_checkpoint,
            train,
            training_segments,
        )
        from proto.root_config_pb2 import RootConfig

        config = RootConfig()
        with open(self._config_filepath, "r") as handle:
            text_format.Parse(handle.read(), config)

        directory = self._checkpoint_dir or config.training.checkpoint.path
        device = dml_device.resolve_device(self._device_spec)
        device_name = (
            dml_device.adapter_name(device.index or 0)
            if device.type == "privateuseone"
            else str(device)
        )
        self._emit(
            phase=TrainingPhase.STARTING.value,
            device=device_name,
            message="Restoring checkpoint",
        )

        restored = checkpoint_io.load_latest(
            directory, expected_digest=checkpoint_io.config_digest(config)
        )
        if restored is None:
            self._emit(
                phase=TrainingPhase.FAILED.value,
                message=f"No checkpoint in {directory}; run lc0-directml-init.",
            )
            return 1

        if self._kda_chunk_size:
            config.model.encoder.kda.chunk_size = self._kda_chunk_size
        if self._grad_accum:
            # Safe to change between runs: the field lives on TrainingConfig
            # rather than inside the optimizer, so it is not part of the
            # checkpoint digest and an existing checkpoint still resumes.
            config.training.gradient_accumulation_steps = self._grad_accum

        model, optimizer = build_model_and_optimizer(config, device)
        model.load_state_dict(restored.model_state)
        model.to(device)
        if restored.optimizer_state is not None:
            optimizer.load_state_dict(restored.optimizer_state)

        # One "segment" is a checkpoint interval; --target-step keeps the
        # loader warm across many of them in this one process, which is what
        # makes the ~4 minute startup a one-time cost.
        checkpoint_interval = (
            self._steps or config.training.schedule.steps_per_network
        )
        start_step = restored.step
        target_step = self._target_step or (start_step + checkpoint_interval)
        if target_step <= start_step:
            self._emit(
                phase=TrainingPhase.FAILED.value,
                step=start_step,
                message=(
                    f"target step {target_step} is not beyond the checkpoint "
                    f"at {start_step}"
                ),
            )
            return 1

        self._emit(
            phase=TrainingPhase.LOADING_DATA.value,
            step=start_step,
            start_step=start_step,
            target_step=target_step,
            device=device_name,
            message="Indexing chunks and filling the shuffle pool",
        )

        reporters: list[metrics_sinks.ClosableReporter] = []
        tensorboard_dir = config.metrics.tensorboard_path
        if tensorboard_dir:
            reporters.append(metrics_sinks.TensorboardReporter(tensorboard_dir))

        def to_tui(step: int, scalars: dict[str, float]) -> None:
            losses = {
                name: value
                for name, value in scalars.items()
                if name not in ("grad_norm", "lr", "ms_per_step")
            }
            self._emit(
                phase=TrainingPhase.TRAINING.value,
                step=step,
                start_step=start_step,
                target_step=target_step,
                losses=losses,
                grad_norm=scalars.get("grad_norm"),
                learning_rate=scalars.get("lr"),
                ms_per_step=scalars.get("ms_per_step"),
                device=device_name,
            )

        reporters.append(metrics_sinks.CallbackReporter(to_tui))

        aliases = [
            spec.split(":", 1)[0] if ":" in spec else ""
            for spec in config.data_loader.output
        ]
        has_split = "test" in aliases
        loader = make_dataloader(config.data_loader)
        batches = batches_from_loader(
            loader, device, "train" if has_split else ""
        )

        eval_hook = self._make_eval_hook(
            config, model, device, loader, has_split, device_name
        )

        progress: dict[str, int] = {"step": start_step}
        failed_message: Optional[str] = None
        final_step = start_step
        try:
            for segment_steps in training_segments(
                start_step, target_step, checkpoint_interval
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
                    # The TUI is the live view; keep stderr sparse so the log
                    # pane stays readable.
                    log_every=max(segment_steps // 4, 1),
                    progress=progress,
                    reporters=reporters,
                    report_every=self._report_every,
                    eval_hook=eval_hook,
                    eval_every=self._eval_every,
                )
                self._emit(
                    phase=TrainingPhase.SAVING.value,
                    step=final_step,
                    start_step=start_step,
                    target_step=target_step,
                    device=device_name,
                    message=f"Saving checkpoint at step {final_step}",
                )
                checkpoint_io.save(
                    directory,
                    make_checkpoint(config, model, optimizer, final_step),
                    max_to_keep=config.training.checkpoint.max_to_keep,
                )
                self._export(config, model, final_step)
        except (RuntimeError, KeyboardInterrupt) as error:
            final_step = progress["step"]
            failed_message = str(error)
            logger.error("Training stopped at step %d: %s", final_step, error)
            # Save whatever completed rather than losing the segment.
            if final_step > start_step:
                checkpoint_io.save(
                    directory,
                    make_checkpoint(config, model, optimizer, final_step),
                    max_to_keep=config.training.checkpoint.max_to_keep,
                )
        finally:
            try:
                loader.stop()
            except Exception:  # noqa: BLE001
                logger.exception("Data loader failed to stop cleanly")
            metrics_sinks.close_all(reporters + self._extra_reporters)

        self._emit(
            phase=(
                TrainingPhase.FAILED.value
                if failed_message
                else TrainingPhase.FINISHED.value
            ),
            step=final_step,
            start_step=start_step,
            target_step=target_step,
            device=device_name,
            message=failed_message or f"Finished at step {final_step}",
        )
        return 1 if failed_message else 0
