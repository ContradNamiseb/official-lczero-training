"""Tests for checkpoint save/resume, especially the optimizer-state round trip.

No coverage existed for any of this before -- which is how a checkpoint
saved from a live DirectML model could silently lose the storage identity
between a shared parameter's two names on every single save, and only
surface as a crash-looping resume much later. See training._state_dict_to_host
and checkpoint.OptimizerStateMismatchError for the story.
"""

import pytest

torch = pytest.importorskip("torch")

from lczero_training.directml import checkpoint as checkpoint_io
from lczero_training.directml.model import LczeroModel
from lczero_training.directml.optimizer import NAdamW, build_optimizer
from lczero_training.directml.training import (
    _state_dict_to_host,
    build_model_and_optimizer,
    make_checkpoint,
)


def _shared_pair(model: LczeroModel) -> tuple[str, str]:
    """A (name, name) pair known to alias the same storage in tiny_config.

    policy_embedding_shared is what every policy head's ``tokens``
    attribute aliases -- tiny_config sets shared_policy_embedding_size, so
    this exists as long as at least one policy head is configured.
    """
    names = dict(model.named_parameters(remove_duplicate=False))
    assert "policy_embedding_shared.weight" in names
    head_name = next(
        name
        for name in names
        if name.endswith(".tokens.weight")
        and name != "policy_embedding_shared.weight"
    )
    return "policy_embedding_shared.weight", head_name


# --------------------------------------------------------------------------
# _state_dict_to_host: the actual bug
# --------------------------------------------------------------------------


def test_state_dict_to_host_preserves_shared_parameter_storage(tiny_config):
    """The regression this file exists for.

    Before the fix, model_state built each entry with an independent
    ``tensor.detach().cpu()``, so a shared parameter's two names ended up
    as two distinct CPU tensors with equal values but different
    data_ptr() -- which is exactly what
    checkpoint.load_optimizer_state_dict_into's by-name pairing uses to
    detect sharing after a save/load round trip.
    """
    model = LczeroModel(tiny_config.model)
    name_a, name_b = _shared_pair(model)

    host_state = _state_dict_to_host(model)
    assert host_state[name_a].data_ptr() == host_state[name_b].data_ptr()
    assert torch.equal(host_state[name_a], host_state[name_b])


def test_state_dict_to_host_runs_on_directml(tiny_config, dml_device):
    """DirectML tensors raise on .data_ptr() *before* a transfer to host

    ("Cannot access data pointer of Tensor that doesn't have storage") --
    an earlier version of this fix called .data_ptr() on the live
    on-device tensors to dedupe and crashed on every single parameter.
    _state_dict_to_host dedupes by object identity from
    state_dict(keep_vars=True) instead, which needs no device support.
    """
    model = LczeroModel(tiny_config.model).to(dml_device)
    name_a, name_b = _shared_pair(model)

    host_state = _state_dict_to_host(model)
    assert host_state[name_a].device.type == "cpu"
    assert host_state[name_a].data_ptr() == host_state[name_b].data_ptr()
    assert torch.equal(host_state[name_a], host_state[name_b])


def test_make_checkpoint_shared_parameters_survive_save_and_load(
    tiny_config, tmp_path
):
    """The property that actually matters: it survives torch.save/load."""
    model = LczeroModel(tiny_config.model)
    optimizer = build_optimizer(
        model.named_parameters(),
        tiny_config.training.optimizer,
        learning_rate=1e-4,
    )
    name_a, name_b = _shared_pair(model)

    ckpt = make_checkpoint(tiny_config, model, optimizer, step=1)
    checkpoint_io.save(str(tmp_path), ckpt, max_to_keep=2)
    restored = checkpoint_io.load_latest(
        str(tmp_path),
        expected_digest=checkpoint_io.config_digest(tiny_config),
    )
    assert restored is not None
    a = restored.model_state[name_a]
    b = restored.model_state[name_b]
    assert a.data_ptr() == b.data_ptr()


# --------------------------------------------------------------------------
# The full resume round trip
# --------------------------------------------------------------------------


def test_full_resume_round_trip_preserves_optimizer_moments(
    tiny_config, tmp_path
):
    """Save a real, stepped optimizer; resume it into a fresh model+optimizer
    without error -- the failure mode a real run hit was a crash-loop here.
    """
    model, optimizer = build_model_and_optimizer(
        tiny_config, torch.device("cpu")
    )
    x = torch.rand(2, 112, 8, 8)
    predictions = model(x)
    loss = sum(v.sum() for v in predictions.policy.values()) + sum(
        v[0].sum() for v in predictions.value.values()
    )
    loss.backward()
    optimizer.step()

    ckpt = make_checkpoint(tiny_config, model, optimizer, step=42)
    checkpoint_io.save(str(tmp_path), ckpt, max_to_keep=2)
    restored = checkpoint_io.load_latest(
        str(tmp_path),
        expected_digest=checkpoint_io.config_digest(tiny_config),
    )
    assert restored is not None
    assert restored.step == 42

    model2, optimizer2 = build_model_and_optimizer(
        tiny_config, torch.device("cpu")
    )
    checkpoint_io.load_state_dict_into(model2, restored.model_state)
    # Must not raise -- this is the exact call that crash-looped in
    # production against a checkpoint saved by the pre-fix code path.
    checkpoint_io.load_optimizer_state_dict_into(
        optimizer2,
        restored.optimizer_state,
        model2,
        restored.model_state,
        tiny_config.training.optimizer.nadamw.decay_selector,
    )


def test_optimizer_state_missing_sparse_entry_starts_that_param_fresh(
    tiny_config, tmp_path
):
    """A parameter with a valid name/index but no saved moments (never
    stepped -- grad was always None) must not crash the pairing.

    torch.optim.Optimizer.state_dict() only emits a ``state`` entry for a
    parameter that was actually stepped, so ``state`` is a sparse subset
    of the full index range by design -- not the internal-inconsistency
    case this function also has to detect.
    """
    model, optimizer = build_model_and_optimizer(
        tiny_config, torch.device("cpu")
    )
    x = torch.rand(2, 112, 8, 8)
    predictions = model(x)
    loss = sum(v.sum() for v in predictions.policy.values()) + sum(
        v[0].sum() for v in predictions.value.values()
    )
    loss.backward()
    optimizer.step()

    ckpt = make_checkpoint(tiny_config, model, optimizer, step=1)
    saved_state = ckpt.optimizer_state
    # Delete one real, present state entry to simulate "never stepped".
    some_index = next(iter(saved_state["state"]))
    del saved_state["state"][some_index]

    model2, optimizer2 = build_model_and_optimizer(
        tiny_config, torch.device("cpu")
    )
    # Must not raise KeyError.
    checkpoint_io.load_optimizer_state_dict_into(
        optimizer2,
        saved_state,
        model2,
        ckpt.model_state,
        tiny_config.training.optimizer.nadamw.decay_selector,
    )


# --------------------------------------------------------------------------
# The mismatch this whole file exists to catch, and its specific type
# --------------------------------------------------------------------------


def test_lost_sharing_raises_the_specific_mismatch_type(tiny_config, tmp_path):
    """Reproduces the pre-fix bug's *consequence* directly: a model_state
    where a normally-shared parameter's two names have drifted to distinct
    storage (what independently .cpu()'ing each key from a DirectML tensor
    does -- a device transfer allocates fresh host memory per call, unlike
    a same-device .detach() which is a view). Confirms loading it raises
    OptimizerStateMismatchError specifically -- the type daemon.py's resume
    path catches to fall back to fresh moments instead of crash-looping the
    whole supervisor.

    Forces the drift with .clone() rather than an actual DirectML transfer
    so this test runs on CPU-only CI: on CPU, .cpu() on an already-CPU
    tensor is a no-op view, so it would not reproduce the loss of sharing
    that only a real cross-device copy causes -- see
    test_state_dict_to_host_runs_on_directml for that half.
    """
    model, optimizer = build_model_and_optimizer(
        tiny_config, torch.device("cpu")
    )
    x = torch.rand(2, 112, 8, 8)
    predictions = model(x)
    loss = sum(v.sum() for v in predictions.policy.values()) + sum(
        v[0].sum() for v in predictions.value.values()
    )
    loss.backward()
    optimizer.step()

    name_a, name_b = _shared_pair(model)
    lossy_model_state = dict(model.state_dict())
    lossy_model_state[name_b] = lossy_model_state[name_b].clone()
    assert (
        lossy_model_state[name_a].data_ptr()
        != lossy_model_state[name_b].data_ptr()
    )

    model2, optimizer2 = build_model_and_optimizer(
        tiny_config, torch.device("cpu")
    )
    with pytest.raises(checkpoint_io.OptimizerStateMismatchError):
        checkpoint_io.load_optimizer_state_dict_into(
            optimizer2,
            optimizer.state_dict(),
            model2,
            lossy_model_state,
            tiny_config.training.optimizer.nadamw.decay_selector,
        )
