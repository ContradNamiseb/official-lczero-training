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
        eval_device: Optional[str] = None,
        eval_timeout: float = 900.0,
        data_file_count: Optional[int] = None,
        data_phase_step_interval: Optional[int] = None,
        gc_every: int = 0,
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
        self._eval_device = eval_device
        self._eval_timeout = eval_timeout
        self._data_file_count = data_file_count
        self._data_phase_step_interval = data_phase_step_interval
        self._gc_every = gc_every

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

    @staticmethod
    def _emergency_save(
        config, model, optimizer, step, directory, error
    ) -> None:
        """Checkpoint after a failed step, on a device that just ran out.

        Static because it touches no daemon state: a test can call it
        without constructing a daemon, and constructing one starts a
        thread that reads sys.stdin, which does not belong in a unit test.

        This exists for out-of-memory failures, and naively it does not
        survive one: a checkpoint copies every parameter and both optimizer
        moments to the host, which needs memory, and the exception being
        handled still holds the whole failed step alive. Python keeps a
        traceback on the exception, every frame in it keeps its locals, and
        those locals are the activations that could not be allocated around.

        Clearing those frames was necessary but nowhere near sufficient --
        the save still failed on every real out-of-memory crash -- so there
        are three stages, cheapest first:

        1. Drop the exception's frames and collect, so nothing from the
           failed step is still referenced.
        2. Get the weights off the device one tensor at a time. See
           ``release_to_host``: this *reduces* device pressure as it runs,
           where building the whole state dict demanded ~72 MB at once.
        3. If a save still fails, retry without the optimizer state. Losing
           the moments costs a few perturbed steps on resume; losing the
           checkpoint costs every step since the last scheduled one.

        The model is left on the host, so the caller must not train it
        afterwards. Every caller is on its way out of ``run``.
        """
        import gc
        import traceback

        # Imported here, not taken from run(): run()'s imports are local to
        # that function, so referencing them from this method would raise
        # NameError at precisely the moment the checkpoint matters.
        from lczero_training.directml import checkpoint as checkpoint_io
        from lczero_training.directml.training import (
            make_checkpoint,
            release_to_host,
        )

        if error.__traceback__ is not None:
            traceback.clear_frames(error.__traceback__)
        gc.collect()

        try:
            release_to_host(model, optimizer)
        except Exception:  # noqa: BLE001
            # Worth trying the save anyway: whatever was transferred before
            # the failure is device memory the save no longer has to find.
            logger.exception(
                "Could not move every tensor to the host before the "
                "recovery checkpoint at step %d; saving anyway",
                step,
            )

        def attempt(with_optimizer: bool) -> bool:
            try:
                checkpoint_io.save(
                    directory,
                    make_checkpoint(
                        config,
                        model,
                        optimizer if with_optimizer else None,
                        step,
                    ),
                    max_to_keep=config.training.checkpoint.max_to_keep,
                )
            except Exception:  # noqa: BLE001
                # Never mask the original failure with this one, but never
                # fail silently either -- the whole point is that the user
                # knows whether the steps since the last checkpoint
                # survived.
                logger.exception(
                    "Recovery checkpoint at step %d failed %s the optimizer "
                    "state",
                    step,
                    "with" if with_optimizer else "without",
                )
                return False
            return True

        if attempt(with_optimizer=True):
            logger.info("Saved recovery checkpoint at step %d", step)
            return
        if attempt(with_optimizer=False):
            logger.warning(
                "Saved a recovery checkpoint at step %d without the "
                "optimizer state; the resumed run starts without momentum",
                step,
            )
            return
        logger.error(
            "Could not save a recovery checkpoint at step %d; progress "
            "since the last checkpoint is lost",
            step,
        )

    def _apply_data_phasing(self, config, step: int) -> None:
        """Restrict the corpus to the current phase's tars via a farm.

        Rewrites the file_path_provider's directory in ``config`` to a
        symlink farm holding only the tar window that owns ``step``. The
        farm lives in a temp dir so the real corpus is never touched. The
        provider's watcher then indexes only that window, which is what
        keeps the resident chunk-pool metadata -- and so the peak memory
        the DirectML allocator must carve from -- small on a long run.
        """
        import hashlib
        import tempfile
        from pathlib import Path

        from lczero_training.directml import data_phasing

        source_dir = config.data_loader.stage[0].file_path_provider.directory
        interval = (
            self._data_phase_step_interval
            or data_phasing.DEFAULT_PHASE_STEP_INTERVAL
        )
        window = data_phasing.phase_window(
            directory=Path(source_dir),
            file_count=self._data_file_count,
            phase_step_interval=interval,
            step=step,
        )
        # A stable path, not tempfile.mkdtemp(): the farm is rebuilt from
        # scratch on every start, so a fresh directory each time would
        # accumulate one per run forever -- and when the link fallbacks
        # degrade to copying, each of those is the size of the window.
        # Derived from the source directory so two corpora do not collide.
        digest = hashlib.sha256(source_dir.encode("utf-8")).hexdigest()[:12]
        farm_dir = Path(tempfile.gettempdir()) / f"lc0-data-phase-{digest}"
        stats = data_phasing.build_phase_farm(farm_dir=farm_dir, window=window)
        config.data_loader.stage[0].file_path_provider.directory = str(farm_dir)
        logger.info(
            "file_path_provider now reads the phase farm %s (%d tars: "
            "%d symlinked, %d hardlinked, %d copied)",
            farm_dir,
            len(window),
            stats.symlinked,
            stats.hardlinked,
            stats.copied,
        )

    def _make_eval_hook(
        self, config, model, loader, has_split: bool, device_name: str
    ):
        """An eval callback, or None when evaluation is off or impossible.

        The measurement itself runs in a child process that exits, so it
        costs the trainer no device memory at all -- see
        ``directml/subprocess_eval.py`` for why that matters more than it
        sounds. Nothing here touches the device, including the batches.
        """
        if not self._eval_every:
            return None
        if not has_split:
            logger.warning(
                "--eval-every ignored: this config has no test split. "
                "Generate one with scripts/add_test_split.py."
            )
            return None

        from . import subprocess_eval

        def report(step: int, scalars: dict[str, float]) -> None:
            self._emit(
                phase=TrainingPhase.TRAINING.value,
                step=step,
                device=device_name,
                message=f"eval total={scalars.get('total', float('nan')):.4f}",
            )

        return subprocess_eval.make_eval_hook(
            config=config,
            model=model,
            loader=loader,
            config_filepath=self._config_filepath,
            batch_count=self._eval_batches,
            device_spec=self._eval_device or self._device_spec,
            kda_chunk_size=self._kda_chunk_size,
            timeout=self._eval_timeout,
            on_scalars=report,
        )

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

        # Phase the visible corpus down to a window of tars before the
        # loader indexes anything. The resident pool metadata scales with
        # the number of tars it can see; on the Iris Xe's shared memory
        # that growth is what eventually OOMs a long run, so the loader is
        # pointed at a symlink farm exposing only the current phase's tars.
        if self._data_file_count is not None:
            self._apply_data_phasing(config, restored.step)

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
        else:
            # A checkpoint with no moments -- the emergency save's last
            # resort. Starting the moments at zero is unavoidable, but
            # leaving their step counters at zero as well is not: the bias
            # correction would treat a network at step 191000 as if it were
            # on its first, and the first updates would be full-rate sign
            # steps. Fast-forwarding the counters keeps them the size the
            # schedule intends.
            logger.warning(
                "Checkpoint at step %d has no optimizer state; resuming "
                "without momentum",
                restored.step,
            )
            optimizer.set_step(restored.step)

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
            config, model, loader, has_split, device_name
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
                    gc_every=self._gc_every,
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
                self._emergency_save(
                    config, model, optimizer, final_step, directory, error
                )
        finally:
            try:
                loader.stop()
            except Exception:  # noqa: BLE001
                logger.exception("Data loader failed to stop cleanly")
            metrics_sinks.close_all(reporters)

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
