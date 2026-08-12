"""Guards against a diverged run destroying its own recovery point.

A real run went from a healthy step to all 153 tensors NaN inside one
250-step window, trained on for three hours producing nothing, and wrote six
NaN checkpoints. With max_to_keep rotating the directory, each of those
deleted an older good checkpoint; the last clean weights came within four
writes of being gone. Nothing downstream can use a NaN checkpoint, so the
only useful behaviours are: stop, and refuse to persist the damage.

The other half of these tests is that none of it fires on a healthy run.
"""

import logging

import pytest

torch = pytest.importorskip("torch")

from lczero_training.directml import checkpoint as checkpoint_io
from lczero_training.directml.training import (
    NonFiniteGradientError,
    describe_non_finite,
)


def _checkpoint(step: int, state: dict) -> checkpoint_io.Checkpoint:
    return checkpoint_io.Checkpoint(
        step=step,
        model_state=state,
        optimizer_state=None,
        config_digest="digest",
        rng_state=torch.get_rng_state(),
    )


# --- refusing to persist the damage ---------------------------------------


def test_a_healthy_checkpoint_still_saves(tmp_path):
    """The guard must be invisible to every normal save."""
    path = checkpoint_io.save(
        tmp_path, _checkpoint(100, {"w": torch.ones(4, 4)})
    )

    assert path.exists()
    assert checkpoint_io.load_latest(tmp_path).step == 100


@pytest.mark.parametrize("poison", [float("nan"), float("inf"), float("-inf")])
def test_it_refuses_to_write_non_finite_weights(tmp_path, poison):
    state = {"good": torch.ones(4, 4), "bad": torch.full((2, 2), poison)}

    with pytest.raises(checkpoint_io.NonFiniteWeightsError, match="bad"):
        checkpoint_io.save(tmp_path, _checkpoint(200, state))

    assert not list(tmp_path.glob("checkpoint-*.pt")), (
        "a refused checkpoint must leave nothing behind, not even a partial"
    )


def test_a_refused_write_does_not_rotate_away_the_good_ones(tmp_path):
    """The actual damage in the incident. Six NaN checkpoints at
    max_to_keep=10 came within four writes of deleting the last good
    weights, which were the only thing worth having."""
    for step in (1000, 2000, 3000):
        checkpoint_io.save(
            tmp_path, _checkpoint(step, {"w": torch.ones(4, 4)}), max_to_keep=3
        )

    with pytest.raises(checkpoint_io.NonFiniteWeightsError):
        checkpoint_io.save(
            tmp_path,
            _checkpoint(4000, {"w": torch.full((4, 4), float("nan"))}),
            max_to_keep=3,
        )

    kept = sorted(p.name for p in tmp_path.glob("checkpoint-*.pt"))
    assert len(kept) == 3, kept
    assert checkpoint_io.latest_step(tmp_path) == 3000


def test_require_finite_can_be_turned_off(tmp_path):
    """An escape hatch for deliberately inspecting a poisoned state."""
    checkpoint_io.save(
        tmp_path,
        _checkpoint(1, {"w": torch.full((2, 2), float("nan"))}),
        require_finite=False,
    )

    assert checkpoint_io.latest_step(tmp_path) == 1


def test_first_non_finite_ignores_non_float_tensors():
    """Step counters and integer buffers are not weights and have no NaN."""
    state = {
        "step": torch.tensor([5], dtype=torch.int64),
        "w": torch.ones(3),
    }

    assert checkpoint_io.first_non_finite(state) is None


# --- naming the origin -----------------------------------------------------


class _Tiny(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.a = torch.nn.Linear(3, 3)
        self.b = torch.nn.Linear(3, 3)


def test_it_reports_bad_gradients_over_good_weights_as_this_step():
    """The distinction that says whether this step caused it: a bad gradient
    with clean weights means the damage has not landed yet."""
    model = _Tiny()
    model.a.weight.grad = torch.full_like(model.a.weight, float("nan"))

    detail = describe_non_finite(model)

    assert "1 non-finite gradients" in detail
    assert "a.weight" in detail
    assert "no non-finite weights" in detail
    assert "this step is the origin" in detail


def test_it_reports_already_bad_weights_as_an_earlier_origin():
    model = _Tiny()
    with torch.no_grad():
        model.b.bias.fill_(float("nan"))

    detail = describe_non_finite(model)

    assert "non-finite weights" in detail
    assert "b.bias" in detail
    assert "origin is earlier" in detail


def test_it_says_nothing_is_wrong_on_a_healthy_model():
    model = _Tiny()
    model.a.weight.grad = torch.ones_like(model.a.weight)

    detail = describe_non_finite(model)

    assert "no non-finite gradients" in detail
    assert "no non-finite weights" in detail


def test_the_error_carries_the_step_and_the_detail():
    error = NonFiniteGradientError(212336, "1 non-finite gradients: a.weight")

    assert error.step == 212336
    assert "212336" in str(error)
    assert "a.weight" in str(error)
    assert isinstance(error, RuntimeError), (
        "the daemon and the plain trainer both catch RuntimeError; a guard "
        "outside that hierarchy would escape as an unhandled traceback"
    )


# --- skip mode: ride through a bad batch rather than crash-loop ------------


def _skip_mode_setup(monkeypatch, bad_steps):
    """A tiny real train() run whose Nth clip returns a non-finite norm.

    Drives the guard branch through the actual training loop rather than a
    mock, but forces the non-finite gradient deterministically instead of
    hunting for a batch that explodes.
    """
    from lczero_training.directml.model import LczeroModel
    from lczero_training.directml.optimizer import NAdamW
    from lczero_training.directml.test_training import (
        _fixed_batches,
        _tiny_training_config,
    )

    config = _tiny_training_config()
    config.training.max_grad_norm = 1.0  # use the clip path
    torch.manual_seed(0)
    model = LczeroModel(config.model)
    optimizer = NAdamW(
        [{"params": list(model.parameters()), "weight_decay": 0.0}], lr=1e-4
    )

    from lczero_training.directml import training as training_module

    real_clip = training_module._clip_grad_norm_preserving_origin
    call = {"n": 0}

    def fake_clip(model, max_norm, *args, **kwargs):
        call["n"] += 1
        norm = real_clip(model, max_norm, *args, **kwargs)
        if call["n"] in bad_steps:
            return torch.tensor(float("inf"))
        return norm

    monkeypatch.setattr(
        training_module, "_clip_grad_norm_preserving_origin", fake_clip
    )

    applied = {"n": 0}
    real_step = optimizer.step

    def counting_step(*args, **kwargs):
        applied["n"] += 1
        return real_step(*args, **kwargs)

    monkeypatch.setattr(optimizer, "step", counting_step)
    return config, model, optimizer, applied


def test_skip_mode_rides_through_a_bad_gradient(monkeypatch):
    """The whole point: one exploding step costs a skipped update, not a
    crash. The run that motivated this hit reliable spikes in one region
    and a plain restart could not get past them."""
    from lczero_training.directml.test_training import _fixed_batches
    from lczero_training.directml.training import train

    config, model, optimizer, applied = _skip_mode_setup(
        monkeypatch, bad_steps={2}
    )

    final = train(
        config=config,
        model=model,
        optimizer=optimizer,
        batches=iter(_fixed_batches(4, 4, seed=1)),
        device=torch.device("cpu"),
        start_step=0,
        steps=4,
        log_every=0,
        diagnostics=False,
        nan_check="skip",
    )

    assert final == 4, "the run must reach the end, not stop at the bad step"
    assert applied["n"] == 3, (
        "the bad step's update must be skipped, the other three applied"
    )
    assert torch.isfinite(
        torch.cat([p.flatten() for p in model.parameters()])
    ).all(), "skipping the update must leave the weights finite"


def test_report_mode_still_stops_on_a_bad_gradient(monkeypatch):
    """The default is unchanged: a divergence should stop and be seen."""
    from lczero_training.directml.test_training import _fixed_batches
    from lczero_training.directml.training import train

    config, model, optimizer, _ = _skip_mode_setup(monkeypatch, bad_steps={2})

    # A reporter, because "report" mode checks on the reporting cadence and
    # the cadence only fires when something is listening -- which in a real
    # run is always true (the TUI callback and TensorBoard both attach).
    with pytest.raises(NonFiniteGradientError):
        train(
            config=config,
            model=model,
            optimizer=optimizer,
            batches=iter(_fixed_batches(4, 4, seed=1)),
            device=torch.device("cpu"),
            start_step=0,
            steps=4,
            log_every=0,
            reporters=[lambda step, scalars: None],
            report_every=1,
            diagnostics=False,
            nan_check="report",
        )


def test_skip_mode_gives_up_once_it_is_clearly_divergence(monkeypatch):
    """A flood of skips is not a bad batch. Stopping lets the checkpoint
    guard roll back rather than skipping forever on dead weights."""
    from lczero_training.directml.test_training import _fixed_batches
    from lczero_training.directml.training import train

    config, model, optimizer, _ = _skip_mode_setup(
        monkeypatch, bad_steps={1, 2, 3, 4, 5, 6}
    )

    with pytest.raises(NonFiniteGradientError, match="max-skips"):
        train(
            config=config,
            model=model,
            optimizer=optimizer,
            batches=iter(_fixed_batches(6, 4, seed=1)),
            device=torch.device("cpu"),
            start_step=0,
            steps=6,
            log_every=0,
            diagnostics=False,
            nan_check="skip",
            max_skips=3,
        )


def test_the_emergency_save_declines_a_poisoned_model(tmp_path, caplog):
    """After the guard fires, the recovery save must not try to write the
    dead weights, and must point at the checkpoint that is still good."""
    from lczero_training.directml.daemon import DirectMlTrainingDaemon
    from lczero_training.directml.test_daemon_recovery import _tiny_config
    from lczero_training.directml.model import LczeroModel

    config = _tiny_config()
    model = LczeroModel(config.model)
    with torch.no_grad():
        for param in model.parameters():
            param.fill_(float("nan"))
            break

    with caplog.at_level(logging.ERROR):
        DirectMlTrainingDaemon._emergency_save(
            config, model, None, 216835, str(tmp_path), RuntimeError("boom")
        )

    assert not list(tmp_path.glob("checkpoint-*.pt"))
    assert any("not finite" in record.message for record in caplog.records)
    assert any("resume from it" in record.message for record in caplog.records)
