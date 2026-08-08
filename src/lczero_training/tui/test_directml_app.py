"""Layout and metric-routing tests for the DirectML dashboard.

These assert on widget state and rendered renderables, never on
``export_screenshot``. That returns an SVG: its embedded @font-face CSS
survives naive tag-stripping, and text is split across <text> elements, so
a multi-word phrase never appears as a contiguous substring. Two real bugs
were masked by scraping it.
"""

import types

import pytest

pytest.importorskip("textual")

from rich.console import Console

from lczero_training.daemon.protocol.messages import (
    TrainingMetricsPayload,
    TrainingPhase,
)
from lczero_training.tui.directml_app import (
    DirectMlTuiApp,
    KdaPanel,
    LossPanel,
)

LOSSES = {
    "total": 3.66,
    "policy/main_ce": 2.21,
    "value/winner": 0.75,
    "movesleft/main": 0.69,
    "Policy Accuracy": 39.45,
    "Value Accuracy": 71.48,
    "KDA/block0 beta mean": 0.9594,
    "KDA/block0 decay saturated %": 24.2703,
    "KDA/block0 output gate mean": 0.9578,
    "KDA/block0 output rms": 0.5138,
    "KDA/block1 beta mean": 0.9162,
    "KDA/block1 decay saturated %": 12.8876,
    "KDA/block1 output gate mean": 0.9636,
    "KDA/block1 output rms": 0.5028,
    "KDA/block2 beta mean": 0.8666,
    "KDA/block2 decay saturated %": 6.4225,
    "KDA/block2 output gate mean": 0.9083,
    "KDA/block2 output rms": 0.5049,
}


@pytest.fixture
def anyio_backend():
    return "asyncio"


def _payload() -> TrainingMetricsPayload:
    return TrainingMetricsPayload(
        phase=TrainingPhase.TRAINING.value,
        step=136750,
        start_step=136708,
        target_step=1000000,
        losses=dict(LOSSES),
        grad_norm=3.9,
        learning_rate=1e-4,
        ms_per_step=3800.0,
        device="Intel(R) Iris(R) Xe Graphics",
    )


class _FakeCommunicator:
    """Accepts and drops everything the app sends."""

    def __init__(self):
        self.sent = []

    async def run(self):
        return None

    async def send(self, payload):
        self.sent.append(payload)


class _FakeProcess:
    returncode = 0

    def __init__(self):
        self.terminated = False

    async def wait(self):
        return 0

    def terminate(self):
        self.terminated = True


@pytest.fixture(autouse=True)
def _no_daemon(monkeypatch):
    """Stop the app spawning a real daemon subprocess.

    Patched on the class, not the instance and not via a subclass. Textual
    resolves event handlers by walking the MRO and invokes the handler it
    finds on *every* class in it, so an instance attribute is ignored and a
    subclass override runs *in addition to* the original. Only replacing
    the method on DirectMlTuiApp itself actually stops it. Until this was
    noticed every test in this file launched a real training daemon.
    """

    async def fake_on_load(self) -> None:
        self._log_stream = None
        self._io_dump = None
        self._communicator = _FakeCommunicator()
        self._process = _FakeProcess()

    monkeypatch.setattr(DirectMlTuiApp, "on_load", fake_on_load)


def _app() -> DirectMlTuiApp:
    args = types.SimpleNamespace(
        config="x.textproto", logfile=None, daemon_flags=[], io_dump=None
    )
    return DirectMlTuiApp(args)


def _text_of(widget, width: int = 60) -> str:
    """Render a widget's renderable to plain text."""
    console = Console(width=width, no_color=True)
    with console.capture() as capture:
        console.print(widget.render())
    return capture.get()


@pytest.mark.parametrize(
    "size", [(120, 34), (100, 30), (80, 24), (72, 16), (70, 14)]
)
@pytest.mark.anyio
async def test_kda_panel_visible_at_every_supported_size(size):
    """Every size at or above the floor must show all three KDA blocks.

    72x16 regressed once: the panels stack below 90 columns, which doubles
    the height the top row needs, and kda was pushed off the bottom.
    """
    app = _app()
    async with app.run_test(size=size) as pilot:
        await app.on_training_metrics(_payload())
        await pilot.pause()

        panel = app.query_one("#kda", KdaPanel)
        assert panel.display, f"kda panel hidden at {size}"
        assert panel.size.height > 0, f"kda panel has no height at {size}"

        text = _text_of(panel)
        for block in ("blk0", "blk1", "blk2"):
            assert block in text, f"{block} missing at {size}: {text!r}"
        assert "24.27" in text, f"block0 saturation missing at {size}"


@pytest.mark.anyio
async def test_breakpoint_classes_land_on_the_screen():
    """Regression: App.set_class puts classes on the App node.

    Every breakpoint rule in the stylesheet is written `Screen.narrow`,
    `Screen.short`, ... so classes on the App silently match nothing and
    the entire ladder does nothing at all.
    """
    app = _app()
    async with app.run_test(size=(80, 24)) as pilot:
        await pilot.pause()
        assert "narrow" in app.screen.classes
        assert "short" in app.screen.classes
        assert "narrow" not in app.classes


@pytest.mark.anyio
async def test_breakpoints_applied_on_mount_not_only_on_resize():
    """Textual delivers Resize only when the terminal actually changes, so
    a window that starts small would otherwise lay out as if it were wide.
    """
    app = _app()
    async with app.run_test(size=(72, 16)) as pilot:
        await pilot.pause()
        # Never resized; these can only have been set from the real
        # on_mount, which this test deliberately does not stub out.
        assert "narrow" in app.screen.classes
        assert "no-log" in app.screen.classes
        # And on_mount really did run its other work.
        assert app._communicator.sent, "on_mount did not send the start"


@pytest.mark.anyio
async def test_below_the_floor_explains_itself():
    """A header, a void and a footer reads as a broken dashboard."""
    app = _app()
    async with app.run_test(size=(60, 12)) as pilot:
        await pilot.pause()
        assert "too-small" in app.screen.classes
        notice = app.query_one("#too-small-notice")
        assert notice.display, "no explanation shown below the size floor"
        # The stylesheet hides the container, so the panel's own `display`
        # stays True while it is off-screen; its height is what tells you.
        assert app.query_one("#row-top").display is False
        assert app.query_one("#kda", KdaPanel).size.height == 0


@pytest.mark.anyio
async def test_kda_metrics_are_routed_out_of_the_losses_dict():
    """The daemon ships KDA stats inside `losses`; the app has to split the
    `KDA/blockN <stat>` keys back out into per-block rows."""
    app = _app()
    async with app.run_test(size=(120, 34)) as pilot:
        await app.on_training_metrics(_payload())
        await pilot.pause()

        assert sorted(app._blocks) == [0, 1, 2]
        assert app._blocks[0]["decay saturated %"] == pytest.approx(24.2703)
        assert app._blocks[2]["output gate mean"] == pytest.approx(0.9083)
        # KDA keys must not leak into the loss sparklines.
        assert not any(k.startswith("KDA/") for k in app._history)


def test_sparkline_survives_a_nan():
    """The exact crash: one NaN ended a 4,250-step run.

    NaN passes through min() and max() untouched -- every comparison with
    it is False -- so it reaches int() and raises ValueError there.
    Textual treats that as fatal and the app owns the daemon, so training
    died with the display.
    """
    from lczero_training.tui.directml_app import _spark

    series = [3.6, 3.2, float("nan"), 3.4, 3.5, 3.3]
    rendered = _spark(series)

    assert len(rendered) == len(series)
    assert "·" in rendered, "the NaN should show as a gap"
    assert any(char in rendered for char in _spark([1.0, 2.0]))


def test_sparkline_handles_an_all_nan_series():
    from lczero_training.tui.directml_app import _spark

    assert _spark([float("nan")] * 4) == "·" * 4


def test_sparkline_handles_infinities():
    from lczero_training.tui.directml_app import _spark

    rendered = _spark([1.0, float("inf"), 2.0, float("-inf")])
    assert len(rendered) == 4
    assert rendered.count("·") == 2


def test_sparkline_handles_a_flat_series_with_a_nan():
    from lczero_training.tui.directml_app import _spark

    rendered = _spark([2.0, 2.0, float("nan"), 2.0])
    assert len(rendered) == 4
    assert rendered.count("·") == 1


def test_trend_ignores_non_finite_samples():
    from lczero_training.tui.directml_app import _trend

    series = [5.0, 5.0, 5.0, float("nan"), 1.0, 1.0, 1.0]
    assert "↓" in _trend(series).plain


@pytest.mark.anyio
async def test_a_nan_metric_does_not_kill_the_app():
    """End to end: a NaN in the payload must not stop the run."""
    app = _app()
    async with app.run_test(size=(120, 34)) as pilot:
        payload = _payload()
        payload.losses["total"] = float("nan")
        await app.on_training_metrics(payload)
        await pilot.pause()

        text = _text_of(app.query_one("#losses", LossPanel))
        assert "total" in text
        assert app.is_running


@pytest.mark.anyio
async def test_a_panel_that_raises_does_not_kill_the_app():
    """Any render failure, not only NaN, has to stay contained."""
    app = _app()
    async with app.run_test(size=(120, 34)) as pilot:
        await app.on_training_metrics(_payload())
        await pilot.pause()

        panel = app.query_one("#losses", LossPanel)
        # render_panel, not _render: Textual owns _render as internal API.
        panel.render_panel = lambda: 1 / 0

        assert "render failed" in _text_of(panel)
        assert app.is_running


@pytest.mark.anyio
async def test_an_unusable_payload_is_discarded_not_fatal():
    app = _app()
    async with app.run_test(size=(120, 34)) as pilot:
        broken = _payload()
        # A KDA key the parser cannot split into a block index and a stat.
        broken.losses["KDA/blockX"] = 1.0
        await app.on_training_metrics(broken)
        await pilot.pause()
        assert app.is_running


@pytest.mark.anyio
async def test_loss_panel_shows_the_configured_losses():
    app = _app()
    async with app.run_test(size=(120, 34)) as pilot:
        await app.on_training_metrics(_payload())
        await pilot.pause()
        text = _text_of(app.query_one("#losses", LossPanel))
        for label in ("total", "policy", "value", "mleft"):
            assert label in text
