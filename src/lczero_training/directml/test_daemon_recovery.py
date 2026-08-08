"""The recovery checkpoint written when a training step fails.

This path failed silently twice in real runs: two out-of-memory crashes at
steps 136756 and 136821 left no checkpoint at all, so every step since the
last scheduled one was lost. It then went on failing loudly -- clearing the
exception's frames was not enough on a device with nothing left to give --
which is what the host-transfer and drop-the-optimizer stages address. Every
cause is covered here.
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


def test_release_to_host_drops_gradients_and_hosts_every_tensor():
    """Stage two of the emergency save, and the reason it works.

    Every copy releases its device original, so the transfer only ever
    lowers pressure -- unlike ``make_checkpoint``, which needs room for the
    whole state dict at once. There is no DirectML device in CI, so this
    checks the invariants that hold on any device: the gradients are gone,
    nothing is left un-hosted, and the parameters keep their identity so
    the optimizer's state stays keyed on them.
    """
    from lczero_training.directml.model import LczeroModel
    from lczero_training.directml.optimizer import NAdamW
    from lczero_training.directml.training import release_to_host

    config = _tiny_config()
    model = LczeroModel(config.model)
    parameters = list(model.parameters())
    optimizer = NAdamW([{"params": parameters, "weight_decay": 0.0}], lr=1e-4)
    identities = [id(param) for param in parameters]

    # A step's worth of gradients and one optimizer step, so there are
    # moments to move.
    for param in parameters:
        param.grad = torch.ones_like(param)
    optimizer.step()
    assert optimizer.state, "sanity: the optimizer should hold moments"

    release_to_host(model, optimizer)

    assert all(param.grad is None for param in parameters), (
        "gradients are the cheapest memory to give back and no checkpoint "
        "wants them"
    )
    assert [id(param) for param in parameters] == identities, (
        "rebuilding the parameters would orphan the optimizer's state"
    )
    assert all(
        tensor.device.type == "cpu"
        for tensor in model.state_dict(keep_vars=True).values()
    )
    assert all(
        value.device.type == "cpu"
        for state in optimizer.state.values()
        for value in state.values()
        if isinstance(value, torch.Tensor)
    )


def test_emergency_save_falls_back_to_dropping_the_optimizer_state(
    tmp_path, caplog
):
    """The last resort: momentum is worth far less than the steps.

    An unsaveable optimizer state stands in for one the device cannot
    produce a host copy of. The checkpoint must still be written, must say
    that it is momentum-less, and must be loadable.
    """
    import logging

    from lczero_training.directml import checkpoint as checkpoint_io
    from lczero_training.directml.model import LczeroModel
    from lczero_training.directml.optimizer import NAdamW

    config = _tiny_config()
    model = LczeroModel(config.model)
    parameters = list(model.parameters())
    optimizer = NAdamW([{"params": parameters, "weight_decay": 0.0}], lr=1e-4)
    for param in parameters:
        param.grad = torch.ones_like(param)
    optimizer.step()
    # Unpicklable, so torch.save refuses any checkpoint carrying it. A
    # local function is the simplest thing pickle cannot handle.
    optimizer.state[parameters[0]]["unsaveable"] = lambda: None

    with caplog.at_level(logging.WARNING):
        _emergency_save(
            config, model, optimizer, 4242, str(tmp_path), RuntimeError("oom")
        )

    restored = checkpoint_io.load_latest(str(tmp_path))
    assert restored is not None, "the fallback save must still land"
    assert restored.step == 4242
    assert restored.optimizer_state is None
    assert any(
        "without momentum" in record.message for record in caplog.records
    ), "a momentum-less checkpoint must announce itself"


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
