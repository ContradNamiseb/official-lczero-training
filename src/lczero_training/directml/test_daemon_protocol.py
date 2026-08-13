"""The DirectML daemon's metrics message must survive the JSONL protocol."""

import io
import json

import pytest

from lczero_training.daemon.protocol.communicator import Communicator
from lczero_training.daemon.protocol.messages import (
    TrainingMetricsPayload,
    TrainingPhase,
)


class _Handler:
    def __init__(self):
        self.received = []

    def on_training_metrics(self, payload: TrainingMetricsPayload) -> None:
        self.received.append(payload)


def _sent_line(payload) -> str:
    out = io.StringIO()
    Communicator(_Handler(), io.StringIO(), out).send(payload)
    return out.getvalue()


def test_metrics_payload_serializes():
    payload = TrainingMetricsPayload(
        phase=TrainingPhase.TRAINING.value,
        step=97205,
        start_step=97204,
        target_step=98204,
        losses={"policy/main_ce": 2.2313, "total": 3.4084},
        grad_norm=5.6455,
        learning_rate=0.0001,
        ms_per_step=1030.5,
        device="Intel(R) Iris(R) Xe Graphics",
    )
    line = _sent_line(payload)
    message = json.loads(line)
    assert message["type"] == "training_metrics"
    assert message["payload"]["step"] == 97205
    # The dict field is the one most likely to break a naive serializer.
    assert message["payload"]["losses"]["policy/main_ce"] == pytest.approx(
        2.2313
    )


def test_metrics_payload_roundtrips_to_handler():
    payload = TrainingMetricsPayload(
        phase=TrainingPhase.TRAINING.value,
        step=42,
        losses={"total": 1.5},
        grad_norm=0.25,
    )
    line = _sent_line(payload)

    handler = _Handler()
    Communicator(handler, io.StringIO(line), io.StringIO()).run()

    assert len(handler.received) == 1
    got = handler.received[0]
    assert got.step == 42
    assert got.losses == {"total": 1.5}
    assert got.grad_norm == pytest.approx(0.25)
    assert got.phase == TrainingPhase.TRAINING.value


def test_optional_fields_default_to_none():
    """A partial update must not require every field."""
    line = _sent_line(TrainingMetricsPayload(step=7))
    handler = _Handler()
    Communicator(handler, io.StringIO(line), io.StringIO()).run()

    got = handler.received[0]
    assert got.step == 7
    assert got.losses is None
    assert got.grad_norm is None
    assert got.phase == TrainingPhase.STARTING.value


def test_a_single_bad_line_does_not_kill_the_reader():
    """The reader must survive a malformed line and keep going. A single
    bad JSONL used to silently take down the AsyncCommunicator worker,
    after which every later payload was dropped -- including all KDA
    updates, which is what left the kda-health pane empty in the live TUI
    while the daemon kept producing KDA stats. Reproduce by feeding one
    unreadable line followed by a valid one and asserting both that the
    bad line was discarded and that the good one still arrived.
    """
    payload = TrainingMetricsPayload(step=42, losses={"total": 1.5})
    wire = (
        "this is not json\n"
        + _sent_line(payload)
        + "{\"type\": \"unknown_event\", \"payload\": {}}\n"
        + _sent_line(TrainingMetricsPayload(step=43, losses={"total": 1.6}))
    )

    handler = _Handler()
    Communicator(handler, io.StringIO(wire), io.StringIO()).run()

    # Both good lines delivered, in order; the two bad ones skipped.
    assert [p.step for p in handler.received] == [42, 43]


def test_kda_keys_survive_the_wire_round_trip():
    """The KDA panel is empty in the live TUI despite the daemon emitting
    KDA stats on every report step, so the path that has to actually
    work is the one that delivers ``KDA/blockN <stat>`` keys through the
    JSONL wire and into the handler. Dict keys with spaces survive
    ``json.dumps``/``json.loads`` (JSON keys are strings), but a future
    serializer change that turned them into something else would quietly
    break the kda panel -- pin them here."""
    losses = {
        "KDA/block0 beta mean": 0.9594,
        "KDA/block0 decay saturated %": 24.2703,
        "KDA/block0 output gate mean": 0.9578,
        "KDA/block0 output rms": 0.5138,
        "KDA/block2 output gate mean": 0.9083,
    }
    line = _sent_line(
        TrainingMetricsPayload(
            phase=TrainingPhase.TRAINING.value, step=1, losses=losses
        )
    )
    handler = _Handler()
    Communicator(handler, io.StringIO(line), io.StringIO()).run()

    got = handler.received[0]
    for key, value in losses.items():
        assert key in got.losses, f"{key!r} dropped on the wire"
        assert got.losses[key] == pytest.approx(value)


@pytest.mark.anyio
async def test_pane_renders_live_metrics():
    """Mount the pane headlessly and feed it a real payload."""
    from textual.app import App, ComposeResult
    from textual.widgets import Label

    from lczero_training.tui.directml_training_pane import (
        DirectMlTrainingPane,
    )

    class _App(App):
        def compose(self) -> ComposeResult:
            yield DirectMlTrainingPane(id="pane")

    app = _App()
    async with app.run_test() as pilot:
        pane = app.query_one("#pane", DirectMlTrainingPane)
        pane.update_metrics(
            TrainingMetricsPayload(
                phase=TrainingPhase.TRAINING.value,
                step=97250,
                start_step=97204,
                target_step=98204,
                losses={"policy/main_ce": 2.2313, "total": 3.4084},
                grad_norm=5.6455,
                learning_rate=0.0001,
                ms_per_step=1030.0,
                device="Intel(R) Iris(R) Xe Graphics",
            )
        )
        await pilot.pause()

        def text(widget_id: str) -> str:
            return str(app.query_one(widget_id, Label).content)

        assert "TRAINING" in text("#dml-phase")
        step_text = text("#dml-step")
        assert "97,250" in step_text and "98,204" in step_text
        losses_text = text("#dml-losses")
        assert "policy/main_ce" in losses_text and "2.2313" in losses_text
        rates_text = text("#dml-rates")
        assert "ms/step" in rates_text and "eta" in rates_text
        # 46 of 1000 steps done at ~1.03 s each -> roughly 16 minutes left.
        assert "16." in rates_text


@pytest.fixture
def anyio_backend():
    return "asyncio"


def test_pane_renders_a_payload_without_a_running_app():
    """The formatting helpers must not depend on Textual being mounted."""
    from lczero_training.tui.directml_training_pane import _sparkline

    assert _sparkline([]) == ""
    assert _sparkline([1.0]) == ""
    assert len(_sparkline([1.0, 2.0, 3.0])) == 3
    # A flat series must not divide by zero.
    assert len(_sparkline([2.0, 2.0, 2.0])) == 3
