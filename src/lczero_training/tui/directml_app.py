"""Dense dashboard for native DirectML training.

Purpose-built rather than reusing TrainingTuiApp: that one is laid out
around the JAX daemon's TrainingStatusPayload (data-pipeline meters, epoch
schedule), none of which the DirectML daemon sends. Reusing it left half
the screen as an empty pane.

Layout is the widget-dashboard shape (btop, bottom): fixed boxes, packed
data, titles in the border, no nested frames. Panels never move.
"""

from __future__ import annotations

import datetime
import logging
import math
import subprocess
import sys
from collections import deque
from typing import Optional

import anyio
from anyio.streams.text import TextReceiveStream, TextSendStream
from rich.table import Table
from rich.text import Text
from textual.app import App, ComposeResult
from textual.containers import Horizontal
from textual.widgets import Footer, Static

from ..daemon.protocol.communicator import AsyncCommunicator
from ..daemon.protocol.messages import (
    StartTrainingPayload,
    TrainingMetricsPayload,
    TrainingPhase,
)
from .log_pane import StreamingLogPane

logger = logging.getLogger(__name__)

_DAEMON_MODULE = "lczero_training.commands.directml_daemon"
_SUPERVISOR_MODULE = "lczero_training.commands.directml_supervisor"

_HISTORY = 60
_SPARK = "▁▂▃▄▅▆▇█"
# Drawn where a sample is not finite, so a NaN is visible as a gap rather
# than silently plotted as the series minimum.
_GAP = "·"

# Health thresholds from docs/metrics.md. Colour is paired with the value
# itself and a letter marker, never used alone.
_DECAY_SATURATED_WARN = 40.0
_DECAY_SATURATED_BAD = 50.0
_GATE_WARN = 0.45
_GATE_BAD = 0.30


def _spark(values) -> str:
    """A sparkline, with non-finite samples drawn as gaps.

    NaN passes straight through min() and max() -- every comparison
    against it is False -- so it survives unnoticed and only fails at
    int(), which raises ValueError. Textual treats an exception in render
    as fatal, and this app owns the training daemon, so one unplottable
    number ends the run. That is not hypothetical: a single NaN in `total`
    took down 4,250 steps of training.
    """
    series = list(values)
    if len(series) < 2:
        return ""
    finite = [v for v in series if math.isfinite(v)]
    if not finite:
        return _GAP * len(series)
    low, high = min(finite), max(finite)
    span = high - low
    scale = len(_SPARK) - 1

    def cell(value: float) -> str:
        if not math.isfinite(value):
            return _GAP
        if span < 1e-12:
            return _SPARK[0]
        return _SPARK[max(0, min(int((value - low) / span * scale), scale))]

    return "".join(cell(v) for v in series)


def _trend(values) -> Text:
    """Direction of the last few samples, as an arrow plus colour."""
    series = [v for v in values if math.isfinite(v)]
    if len(series) < 6:
        return Text("  ")
    older = sum(series[-6:-3]) / 3.0
    newer = sum(series[-3:]) / 3.0
    if abs(newer - older) < abs(older) * 0.01:
        return Text("→ ", style="dim")
    return (
        Text("↓ ", style="green") if newer < older else Text("↑ ", style="red")
    )


def _duration(seconds: float) -> str:
    if seconds <= 0 or seconds != seconds:  # NaN-safe
        return "--"
    minutes, _ = divmod(int(seconds), 60)
    hours, minutes = divmod(minutes, 60)
    days, hours = divmod(hours, 24)
    if days:
        return f"{days}d{hours:02d}h"
    if hours:
        return f"{hours}h{minutes:02d}m"
    return f"{minutes}m"


class DashboardPanel(Static):
    """A panel whose rendering cannot take the training run down with it.

    Textual treats an exception from render() as fatal, and this app owns
    the daemon as a child process, so a display failure kills training.
    One NaN in a sparkline did exactly that. A monitor is allowed to show
    nothing; it is not allowed to stop the thing it is monitoring.

    Subclasses implement render_panel(); a failure is logged and shown in
    place of the panel's contents.

    The hook is NOT called _render: Textual's Widget already defines that
    as internal API returning a Visual, and overriding it with something
    that returns a Rich renderable breaks layout with a confusing
    "'Text' object has no attribute 'get_height'".
    """

    def render(self):
        try:
            return self.render_panel()
        except Exception:  # noqa: BLE001 - never kill a run for a display
            logger.exception("Panel %s failed to render", type(self).__name__)
            return Text("render failed - see the log", style="red")

    def render_panel(self):  # pragma: no cover - subclasses override this
        raise NotImplementedError


class HeaderPanel(DashboardPanel):
    """Run identity, phase, and overall progress."""

    def render_panel(self):
        payload: Optional[TrainingMetricsPayload] = getattr(
            self, "payload", None
        )
        if payload is None:
            return Text("connecting to the daemon...", style="dim")

        phase = payload.phase or TrainingPhase.STARTING.value
        style = {
            TrainingPhase.TRAINING.value: "bold green",
            TrainingPhase.LOADING_DATA.value: "bold yellow",
            TrainingPhase.SAVING.value: "bold cyan",
            TrainingPhase.FINISHED.value: "bold blue",
            TrainingPhase.FAILED.value: "bold red",
        }.get(phase, "bold")

        total = max(payload.target_step - payload.start_step, 0)
        done = max(payload.step - payload.start_step, 0)
        fraction = (done / total) if total else 0.0
        width = max(self.size.width - 46, 10)
        filled = int(fraction * width)
        bar = "━" * filled + "─" * (width - filled)

        line = Text()
        line.append(f"{phase:<13}", style=style)
        line.append(f"{payload.step:>10,}", style="bold")
        line.append(f" / {payload.target_step:<10,}  ", style="dim")
        line.append(bar[:filled], style="green")
        line.append(bar[filled:], style="dim")
        line.append(f" {fraction * 100:5.1f}%  ", style="bold")

        eta = ""
        if payload.ms_per_step and total:
            eta = _duration((total - done) * payload.ms_per_step / 1000.0)
        line.append(f"eta {eta:>6}", style="cyan")
        return line


class LossPanel(DashboardPanel):
    """The optimized losses, with sparklines and trend arrows."""

    def render_panel(self):
        history = getattr(self, "history", {})
        if not history:
            return Text("waiting for the first step...", style="dim")

        table = Table.grid(padding=(0, 1), expand=True)
        table.add_column("metric", ratio=2)
        table.add_column("value", justify="right", width=9)
        table.add_column("trend", width=2)
        table.add_column("spark", ratio=3)

        order = ("total", "policy/main_ce", "value/winner", "movesleft/main")
        labels = {
            "total": "total",
            "policy/main_ce": "policy",
            "value/winner": "value",
            "movesleft/main": "mleft",
        }
        for key in order:
            series = history.get(key)
            if not series:
                continue
            table.add_row(
                Text(labels[key], style="bold" if key == "total" else ""),
                Text(f"{series[-1]:.4f}"),
                _trend(series),
                Text(_spark(series), style="cyan"),
            )
        return table


class KdaPanel(DashboardPanel):
    """Per-block KDA gate health. See docs/metrics.md."""

    def render_panel(self):
        blocks = getattr(self, "blocks", {})
        if not blocks:
            return Text("no KDA metrics yet", style="dim")

        # The rms column is the least diagnostic of the four, so it is the
        # first thing dropped when the panel cannot fit them all.
        wide = self.size.width >= 38
        table = Table.grid(padding=(0, 1), expand=True)
        table.add_column("blk", width=6)
        table.add_column("sat", justify="right", width=7)
        table.add_column("beta", justify="right", width=6)
        table.add_column("gate", justify="right", width=6)
        if wide:
            table.add_column("rms", justify="right", width=6)
        header = [
            Text("", style="dim"),
            Text("sat%", style="dim"),
            Text("beta", style="dim"),
            Text("gate", style="dim"),
        ]
        if wide:
            header.append(Text("rms", style="dim"))
        table.add_row(*header)

        for index in sorted(blocks):
            stats = blocks[index]
            saturated = stats.get("decay saturated %")
            beta = stats.get("beta mean")
            gate = stats.get("output gate mean")
            rms = stats.get("output rms")

            def mark(value, warn, bad, higher_is_worse=True):
                if value is None:
                    return Text("--", style="dim")
                over = value >= bad if higher_is_worse else value <= bad
                near = value >= warn if higher_is_worse else value <= warn
                text = f"{value:.2f}"
                if over:
                    return Text(f"{text}!", style="bold red")
                if near:
                    return Text(f"{text}?", style="yellow")
                return Text(f"{text} ", style="green")

            row = [
                Text(f"blk{index}", style="bold"),
                mark(saturated, _DECAY_SATURATED_WARN, _DECAY_SATURATED_BAD),
                mark(beta, _GATE_WARN, _GATE_BAD, higher_is_worse=False),
                mark(gate, _GATE_WARN, _GATE_BAD, higher_is_worse=False),
            ]
            if wide:
                row.append(Text(f"{rms:.2f}" if rms is not None else "--"))
            table.add_row(*row)
        return table


class PolicyPanel(DashboardPanel):
    """Move-ranking quality, which the policy loss alone does not show."""

    def render_panel(self):
        scalars = getattr(self, "scalars", {})
        if not scalars:
            return Text("--", style="dim")

        table = Table.grid(padding=(0, 1), expand=True)
        table.add_column("k", ratio=2)
        table.add_column("v", justify="right", ratio=1)

        def add(label, key, suffix="", fmt="{:.2f}"):
            if key in scalars:
                table.add_row(
                    Text(label, style="dim"),
                    Text(fmt.format(scalars[key]) + suffix),
                )

        add("accuracy", "Policy Accuracy", "%")
        thresholds = [
            scalars.get(f"Thresholded Policy Accuracy @ {t}")
            for t in (1, 2, 5, 10)
        ]
        if all(t is not None for t in thresholds):
            table.add_row(
                Text("@1/2/5/10", style="dim"),
                Text("/".join(f"{t:.0f}" for t in thresholds)),
            )
        add("entropy", "Policy Entropy")
        add("search loss", "Policy SL")
        add("value acc", "Value Accuracy", "%")
        return table


class OptimizerPanel(DashboardPanel):
    """Gradient norm against the clip, learning rate, throughput."""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.max_grad_norm = 10.0

    def render_panel(self):
        scalars = getattr(self, "scalars", {})
        if not scalars:
            return Text("--", style="dim")

        table = Table.grid(padding=(0, 1), expand=True)
        table.add_column("k", ratio=2)
        table.add_column("v", justify="right", ratio=1)

        norm = scalars.get("grad_norm")
        if norm is not None:
            # Clipping is not an error, but sustained clipping means the
            # effective LR is below the configured one -- worth seeing.
            clipped = norm >= self.max_grad_norm
            table.add_row(
                Text("grad", style="dim"),
                Text(
                    f"{norm:.2f}" + (" CLIP" if clipped else ""),
                    style="yellow" if clipped else "",
                ),
            )
        if "lr" in scalars:
            table.add_row(Text("lr", style="dim"), Text(f"{scalars['lr']:.2e}"))
        if "ms_per_step" in scalars:
            table.add_row(
                Text("ms/step", style="dim"),
                Text(f"{scalars['ms_per_step']:,.0f}"),
            )
        if "Params" in scalars:
            table.add_row(
                Text("params l2", style="dim"),
                Text(f"{scalars['Params']:.1f}"),
            )
        return table


class DirectMlTuiApp(App):
    """Live dashboard for a native DirectML training run."""

    CSS_PATH = "directml_app.tcss"
    BINDINGS = [
        ("q", "quit", "quit"),
        ("ctrl+c", "quit", ""),
    ]

    def __init__(self, args):
        super().__init__()
        self._config_file = args.config
        self._logfile = getattr(args, "logfile", None)
        self._daemon_flags = list(getattr(args, "daemon_flags", []))
        self._supervise = bool(getattr(args, "supervise", False))
        self._io_dump_file = getattr(args, "io_dump", None)
        self._io_dump = None
        self._history: dict[str, deque] = {}
        self._blocks: dict[int, dict[str, float]] = {}
        self._scalars: dict[str, float] = {}
        self._last: Optional[TrainingMetricsPayload] = None

    def _child_command(self) -> list[str]:
        """The process to spawn: the daemon, or a supervisor wrapped round it.

        Interchangeable from here. The supervisor hands the daemon its own
        stdin, stdout and stderr, so the protocol arrives on the same pipe
        either way and nothing in this app needs to know which it is talking
        to -- including across a restart, which looks like a run whose step
        count carries on from where the last checkpoint left it.
        """
        if not self._supervise:
            return [sys.executable, "-m", _DAEMON_MODULE, *self._daemon_flags]

        # Local so that importing this app does not drag the command modules
        # in; the rest of the file follows the same rule.
        from ..commands.directml_supervisor import partition_flags

        supervisor_flags, daemon_flags = partition_flags(self._daemon_flags)
        command = [sys.executable, "-m", _SUPERVISOR_MODULE, *supervisor_flags]
        if daemon_flags:
            command += ["--", *daemon_flags]
        return command

    async def on_load(self) -> None:
        self._process = await anyio.open_process(
            self._child_command(),
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
        self._log_stream = TextReceiveStream(self._process.stderr)
        if self._io_dump_file:
            # Line-buffered: a run that dies mid-step still leaves the
            # payloads that led up to it on disk.
            self._io_dump = open(self._io_dump_file, "a", buffering=1)
            stamp = datetime.datetime.now().isoformat(timespec="seconds")
            self._io_dump.write(f"======= {stamp} =======\n")
        self._communicator = AsyncCommunicator(
            handler=self,
            input_stream=TextReceiveStream(self._process.stdout),
            output_stream=TextSendStream(self._process.stdin),
            io_dump=self._io_dump,
        )

    def compose(self) -> ComposeResult:
        header = HeaderPanel(id="header")
        header.border_title = "directml training"
        yield header

        with Horizontal(id="row-top"):
            losses = LossPanel(id="losses")
            losses.border_title = "losses"
            yield losses
            kda = KdaPanel(id="kda")
            kda.border_title = "kda health"
            yield kda

        with Horizontal(id="row-mid"):
            policy = PolicyPanel(id="policy")
            policy.border_title = "policy"
            yield policy
            optimizer = OptimizerPanel(id="optimizer")
            optimizer.border_title = "optimizer"
            yield optimizer

        log = StreamingLogPane(
            stream=self._log_stream, logfile_path=self._logfile
        )
        log.border_title = "loader"
        yield log

        # Shown only below the size floor, where every panel above is
        # hidden. Without it the screen is a header, a void, and a footer,
        # which reads as a broken dashboard rather than a small window.
        yield Static(
            "terminal too small — needs at least 70x14",
            id="too-small-notice",
        )
        yield Footer()

    def on_mount(self) -> None:
        self._apply_breakpoints(self.size.width, self.size.height)
        self.run_worker(self._communicator.run(), exclusive=True)
        self.run_worker(self._send_start(), exclusive=False)
        self.run_worker(self._watch_daemon(), exclusive=False)

    async def _send_start(self) -> None:
        await self._communicator.send(
            StartTrainingPayload(config_filepath=self._config_file)
        )

    async def _watch_daemon(self) -> None:
        await self._process.wait()
        code = self._process.returncode
        self.notify(
            f"Daemon exited with code {code}",
            severity="warning" if code else "information",
            timeout=30,
        )

    def on_resize(self, event) -> None:
        self._apply_breakpoints(event.size.width, event.size.height)

    def _apply_breakpoints(self, width: int, height: int) -> None:
        # Breakpoint ladder. Width decides whether the two-up rows stack;
        # height decides whether the second row survives at all. Stacking
        # costs rows, so a short-and-narrow terminal needs both.
        #
        # Called from on_mount as well as on_resize: Textual delivers Resize
        # only when the terminal actually changes, so a window that is
        # already small when the app starts would otherwise never get any
        # of these classes and would lay itself out as if it were wide.
        # On the screen, not on self: App.set_class puts the class on the
        # App node, and every breakpoint rule in the stylesheet is written
        # `Screen.narrow`, `Screen.short`, ... Setting them on the App means
        # the selectors never match and the whole ladder silently does
        # nothing, which is exactly what it did until this was found.
        screen = self.screen
        screen.set_class(width < 90, "narrow")
        screen.set_class(height < 30, "short")
        # Stacking doubles the height the top row needs, so the point at
        # which the log pane stops fitting depends on width, not just
        # height. Budget: header 3 + rows + log 4 + footer 1.
        screen.set_class(height < (20 if width < 90 else 16), "no-log")
        screen.set_class(width < 70 or height < 14, "too-small")

    async def on_training_metrics(
        self, payload: TrainingMetricsPayload
    ) -> None:
        try:
            self._ingest_metrics(payload)
        except Exception:  # noqa: BLE001 - a bad payload must not stop a run
            logger.exception("Discarding an unusable metrics payload")

    def _ingest_metrics(self, payload: TrainingMetricsPayload) -> None:
        self._last = payload
        losses = payload.losses or {}

        for name, value in losses.items():
            if name.startswith("KDA/block"):
                head, stat = name.split(" ", 1)
                index = int(head.removeprefix("KDA/block"))
                self._blocks.setdefault(index, {})[stat] = value
                continue
            self._scalars[name] = value
            # Kept in the history so the sparkline shows it as a gap rather
            # than hiding it, but worth a log line: a NaN in a loss is a
            # training event, and the trainer's own log only records every
            # --log-every steps, so it can miss one entirely.
            if not math.isfinite(value):
                logger.warning(
                    "Non-finite metric from the daemon at step %d: %s=%r",
                    payload.step,
                    name,
                    value,
                )
            self._history.setdefault(name, deque(maxlen=_HISTORY)).append(value)
        for key in ("grad_norm", "lr", "ms_per_step"):
            value = getattr(
                payload,
                {"grad_norm": "grad_norm", "lr": "learning_rate"}.get(key, key),
                None,
            )
            if value is not None:
                self._scalars[key] = value

        header = self.query_one("#header", HeaderPanel)
        header.payload = payload
        header.border_title = (
            f"directml training  ·  {payload.device or 'unknown device'}"
        )
        header.refresh()

        for widget_id, widget_type, attribute, data in (
            ("#losses", LossPanel, "history", self._history),
            (
                "#kda",
                KdaPanel,
                "blocks",
                {k: dict(v) for k, v in self._blocks.items()},
            ),
            ("#policy", PolicyPanel, "scalars", dict(self._scalars)),
            ("#optimizer", OptimizerPanel, "scalars", dict(self._scalars)),
        ):
            widget = self.query_one(widget_id, widget_type)
            setattr(widget, attribute, data)
            widget.refresh()

    def on_unmount(self) -> None:
        process = getattr(self, "_process", None)
        if process is not None:
            with anyio.CancelScope(shield=True):
                process.terminate()
        if self._io_dump is not None:
            self._io_dump.close()
            self._io_dump = None
