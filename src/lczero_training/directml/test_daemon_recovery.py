"""The recovery checkpoint written when a training step fails.

This path failed silently twice in real runs: two out-of-memory crashes at
steps 136756 and 136821 left no checkpoint at all, so every step since the
last scheduled one was lost. Both causes are covered here.
"""

import pathlib

import pytest

torch = pytest.importorskip("torch")

from google.protobuf import text_format

from lczero_training.directml.daemon import DirectMlTrainingDaemon
from proto.root_config_pb2 import RootConfig

_CONFIG_PATH = (
    pathlib.Path(__file__).resolve().parents[3]
    / "docs"
    / "example_kda_real_import.textproto"
)


def _tiny_config() -> RootConfig:
    config = RootConfig()
    text_format.Parse(_CONFIG_PATH.read_text(), config)
    config.model.embedding.dense_size = 8
    config.model.embedding.embedding_size = 16
    config.model.embedding.dff = 16
    del config.model.encoder.mixer_pattern[1:]
    config.model.encoder.num_blocks = 1
    config.model.encoder.d_model = 16
    config.model.encoder.dff = 16
    config.model.encoder.heads = 8
    config.model.encoder.kda.key_dim = 8
    config.model.encoder.kda.value_dim = 8
    config.model.encoder.kda.gate_rank = 8
    config.model.shared_policy_embedding_size = 16
    for head in config.model.policy_head:
        head.d_model = 16
    for head in config.model.value_head:
        head.num_channels = 8
    for head in config.model.movesleft_head:
        head.num_channels = 8
    return config


# Deliberately not constructing a daemon: __init__ starts a thread that
# reads sys.stdin, and a stray one of those breaks any later test that
# touches the terminal -- the Textual layout tests are exactly that.
_emergency_save = DirectMlTrainingDaemon._emergency_save


def test_emergency_save_writes_a_checkpoint(tmp_path):
    """The whole point: a failed step must not cost the completed ones."""
    from lczero_training.directml import checkpoint as checkpoint_io
    from lczero_training.directml.model import LczeroModel
    from lczero_training.directml.optimizer import NAdamW

    config = _tiny_config()
    model = LczeroModel(config.model)
    optimizer = NAdamW(
        [{"params": list(model.parameters()), "weight_decay": 0.0}], lr=1e-4
    )

    error = RuntimeError("Not enough memory resources are available")
    _emergency_save(config, model, optimizer, 12345, str(tmp_path), error)

    restored = checkpoint_io.load_latest(str(tmp_path))
    assert restored is not None, "no recovery checkpoint was written"
    assert restored.step == 12345


def test_emergency_save_clears_the_traceback_frames():
    """Why the save used to fail: the exception being handled pins the whole
    failed step alive. Every frame in its traceback keeps its locals, and
    those locals are the tensors there was no room for. Clearing them is
    what makes the host copy possible."""
    holder = {}

    def failing_step():
        # Stands in for a training step's activations.
        activations = torch.ones(256, 256)
        holder["ref"] = activations
        raise RuntimeError("Not enough memory resources are available")

    try:
        failing_step()
    except RuntimeError as error:
        frames_before = error.__traceback__ is not None
        assert frames_before

        import traceback

        traceback.clear_frames(error.__traceback__)
        # The frame no longer holds `activations`; only our own dict does.
        import gc

        referrers = [
            r for r in gc.get_referrers(holder["ref"]) if isinstance(r, dict)
        ]
        assert referrers, "sanity: our own reference should still be found"


def test_emergency_save_reports_rather_than_raising(tmp_path, caplog):
    """A failure to save must not mask the original error, but must be
    visible -- silence is what made the lost steps hard to notice."""
    import logging

    from lczero_training.directml.model import LczeroModel

    config = _tiny_config()
    model = LczeroModel(config.model)

    # An unwritable destination: a file where the directory should be.
    blocker = tmp_path / "not-a-directory"
    blocker.write_text("")

    with caplog.at_level(logging.ERROR):
        # Must return normally rather than propagate.
        _emergency_save(
            config, model, None, 7, str(blocker), RuntimeError("boom")
        )

    assert any(
        "recovery checkpoint" in record.message for record in caplog.records
    ), "a failed recovery save must say so"
