"""Live training pane fed by TrainingMetricsPayload.

Replaces the JAXTrainingPane placeholder, which only ever showed static
text -- TrainingStatusPayload carries no loss values, so there was nothing
for it to render.
"""

from __future__ import annotations

from collections import deque
from typing import Optional

from textual.app import ComposeResult
from textual.containers import Vertical
from textual.widgets import Label, ProgressBar, Static

from ..daemon.protocol.messages import TrainingMetricsPayload, TrainingPhase

# How many recent values the inline sparkline keeps.
_HISTORY = 40
_SPARK = "▁▂▃▄▅▆▇█"


def _sparkline(values: list[float]) -> str:
    """A tiny inline chart. TensorBoard is the real plot; this is a pulse."""
    if len(values) < 2:
        return ""
    low, high = min(values), max(values)
    if high - low < 1e-12:
        return _SPARK[0] * len(values)
    span = high - low
    return "".join(
        _SPARK[min(int((value - low) / span * len(_SPARK)), len(_SPARK) - 1)]
        for value in values
    )


class DirectMlTrainingPane(Static):
    """Step progress, losses, and throughput for a DirectML run."""

    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)
        self._history: dict[str, deque[float]] = {}
        self._last: Optional[TrainingMetricsPayload] = None

    def compose(self) -> ComposeResult:
        with Vertical():
            yield Label("DirectML Training", classes="pane-title")
            yield Label("waiting for the daemon...", id="dml-phase")
            yield ProgressBar(total=100, show_eta=False, id="dml-progress")
            yield Label("", id="dml-step")
            yield Label("", id="dml-losses")
            yield Label("", id="dml-rates")
            yield Label("", id="dml-message")

    def update_metrics(self, payload: TrainingMetricsPayload) -> None:
        self._last = payload

        phase = payload.phase or TrainingPhase.STARTING.value
        device = f"  [{payload.device}]" if payload.device else ""
        self.query_one("#dml-phase", Label).update(f"{phase}{device}")

        total = max(payload.target_step - payload.start_step, 0)
        done = max(payload.step - payload.start_step, 0)
        progress = self.query_one("#dml-progress", ProgressBar)
        if total:
            progress.update(total=total, progress=min(done, total))
            self.query_one("#dml-step", Label).update(
                f"step {payload.step:,} / {payload.target_step:,}"
                f"   ({done:,} of {total:,} this run)"
            )
        else:
            self.query_one("#dml-step", Label).update(f"step {payload.step:,}")

        if payload.losses:
            lines = []
            for name in sorted(payload.losses):
                value = payload.losses[name]
                series = self._history.setdefault(name, deque(maxlen=_HISTORY))
                series.append(value)
                lines.append(
                    f"{name:<26} {value:>9.4f}  {_sparkline(list(series))}"
                )
            self.query_one("#dml-losses", Label).update("\n".join(lines))

        rates = []
        if payload.ms_per_step is not None:
            rates.append(f"{payload.ms_per_step:,.0f} ms/step")
            if payload.ms_per_step > 0 and total:
                remaining = (total - done) * payload.ms_per_step / 1000.0
                rates.append(f"eta {remaining / 60.0:.1f} min")
        if payload.grad_norm is not None:
            rates.append(f"grad_norm {payload.grad_norm:.3f}")
        if payload.learning_rate is not None:
            rates.append(f"lr {payload.learning_rate:.3g}")
        if rates:
            self.query_one("#dml-rates", Label).update("   ".join(rates))

        if payload.message:
            self.query_one("#dml-message", Label).update(payload.message)
