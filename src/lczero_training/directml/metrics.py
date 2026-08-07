"""Metric sinks for the DirectML trainer.

A "reporter" is anything that accepts ``(step, scalars)``. The training loop
calls every registered reporter on its reporting cadence and nothing else,
so adding an output (TensorBoard, the TUI daemon, a CSV) never touches the
loop itself.
"""

from __future__ import annotations

import logging
import os
from typing import Callable, Protocol

from . import derived_metrics

logger = logging.getLogger(__name__)


def run_logdir(tensorboard_path: str, run_name: str, split: str = "train") -> str:
    """``<path>/<run_name>-<split>``, matching the TF pipeline's layout.

    The old trainer wrote one directory per split (``-train``, ``-test``,
    ``-validation``, plus ``-swa-*``), which is what makes TensorBoard show
    them as separate selectable runs rather than one merged trace.
    """
    return os.path.join(tensorboard_path, f"{run_name}-{split}")

# (global_step, {metric name: value})
Reporter = Callable[[int, dict[str, float]], None]


class ClosableReporter(Protocol):
    def __call__(self, step: int, scalars: dict[str, float]) -> None: ...

    def close(self) -> None: ...


class TensorboardReporter:
    """Writes scalars to a TensorBoard event file.

    Separate from training/tensorboard.py, which converts through
    ``jax.device_get``; by the time metrics reach here they are already
    plain floats.
    """

    def __init__(self, logdir: str, *, tf_tag_names: bool = True):
        from tensorboardX import SummaryWriter

        self._writer = SummaryWriter(logdir)
        self._tf_tag_names = tf_tag_names
        logger.info("Writing TensorBoard events to %s", logdir)

    def __call__(self, step: int, scalars: dict[str, float]) -> None:
        if self._tf_tag_names:
            # Emit the TF pipeline's tag names so a DirectML run overlays
            # cleanly on an old leelalogs run.
            scalars = derived_metrics.apply_tf_aliases(scalars)
        for tag, value in scalars.items():
            self._add_scalar(tag, float(value), step)
        # Flushed every report so `tensorboard --logdir` shows a running
        # job, not just a finished one.
        self._writer.flush()

    def _add_scalar(self, tag: str, value: float, step: int) -> None:
        """Write a scalar without tensorboardX's tag sanitizing.

        ``SummaryWriter.add_scalar`` runs the tag through ``clean_tag``,
        which rewrites every character outside ``[-/\\w.]`` to an
        underscore: "Policy Accuracy" becomes "Policy_Accuracy" and
        "Thresholded Policy Accuracy @ 1" becomes
        "Thresholded_Policy_Accuracy___1". The TF pipeline wrote the literal
        names, so sanitized tags would sit on separate axes instead of
        overlaying. Building the Summary directly keeps them intact.
        """
        from tensorboardX.proto.summary_pb2 import Summary

        summary = Summary(
            value=[Summary.Value(tag=tag, simple_value=value)]
        )
        self._writer._get_file_writer().add_summary(summary, step)

    def close(self) -> None:
        self._writer.close()


class CallbackReporter:
    """Adapts a plain function into a reporter with a no-op close."""

    def __init__(self, callback: Reporter):
        self._callback = callback

    def __call__(self, step: int, scalars: dict[str, float]) -> None:
        self._callback(step, scalars)

    def close(self) -> None:
        return


def close_all(reporters: list[ClosableReporter]) -> None:
    for reporter in reporters:
        try:
            reporter.close()
        except Exception:  # noqa: BLE001 - closing must not mask a failure
            logger.exception("Reporter failed to close cleanly")
