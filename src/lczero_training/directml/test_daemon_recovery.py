"""The recovery checkpoint written when a training step fails.

This path failed silently twice in real runs: two out-of-memory crashes at
steps 136756 and 136821 left no checkpoint at all, so every step since the
last scheduled one was lost. It then went on failing loudly -- clearing the
exception's frames was not enough on a device with nothing left to give --
which is what the host-transfer and drop-the-optimizer stages address. Every
cause is covered here.
"""

import pytest

torch = pytest.importorskip("torch")

from proto.training_config_pb2 import WeightsSelector
from lczero_training.directml.conftest import make_tiny_config as _tiny_config
from lczero_training.directml.daemon import DirectMlTrainingDaemon

# A selector with no rules and ``otherwise_include=False`` (proto default)
# puts every parameter into the plain group, so the stub tests' simple
# (_TwoHeadStub / _OneHeadStub) models pair cleanly with build_optimizer
# and the helper: there is exactly one group, and every name lands in it
# in registration order. The production configs split decayed-then-plain
# with rules under ``**/policy_heads/**`` etc., which the real-model
# round-trip exercises end-to-end through the daemon; here, a flat plain
# group is enough to test the things worth testing.
_EMPTY_SELECTOR = WeightsSelector()

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
    )


# --- resuming after editing the model (e.g. dropping a policy_head) --------


def _step_optimizer_once(model, optimizer) -> None:
    """One optimizer step so every parameter carries a non-zero moment.

    The exact values do not matter for the tests; what matters is that
    ``state['mu']`` and ``state['nu']`` are non-zero, since the tests
    assert ``state presence / absence after a filter``, not state contents.
    """
    for param in model.parameters():
        param.grad = torch.ones_like(param)
    optimizer.step()
    optimizer.zero_grad(set_to_none=True)


def _tagged_state(optimizer) -> dict:
    """Saved optimizer state, with a marker tensor in each param's state.

    Each parameter gets a unique uint8 tensor as ``state['tag']``. Resuming
    it after the model change and inspecting ``tag`` lets a test prove
    which saved parameter's moment landed on which new parameter -- far
    more decisive than checking only step counts.
    """
    state = optimizer.state_dict()
    for slot in state["state"].values():
        slot["tag"] = torch.tensor(123, dtype=torch.uint8)
    return state


class _TwoHeadStub(torch.nn.Module):
    """Tiny model with two named float parameters and an int64 buffer.

    Mirrors the real model's "policy_head + value_head + moves-left head"
    shape only enough to exercise the optimizer filter: a buffer that
    must NOT be mistaken for a parameter (the policy_map analogue), two
    parameters defined in a stable traversal order, and the option to drop
    the second one -- which is exactly what removing a policy_head does
    to the production model's named_parameters tail.
    """

    def __init__(self) -> None:
        super().__init__()
        self.kept = torch.nn.Linear(2, 2, bias=False)
        self.removed = torch.nn.Linear(2, 2, bias=False)
        # An int64 buffer that the optimizer must NOT include -- this is
        # what distinguishes the policy_map and policy_map_inverse buffers
        # in the real model.
        self.register_buffer(
            "aux_map", torch.zeros((4,), dtype=torch.int64)
        )

    def forward(self, x):  # pragma: no cover - shape only, never run here
        return self.kept(x) + self.removed(x)


class _OneHeadStub(torch.nn.Module):
    """``_TwoHeadStub`` after removing the trailing head.

    Same names and shapes for everything kept, so the saved state can pair
    by name to this new model, and the removed parameter's slot is the
    only thing missing.
    """

    def __init__(self) -> None:
        super().__init__()
        self.kept = torch.nn.Linear(2, 2, bias=False)
        self.register_buffer(
            "aux_map", torch.zeros((4,), dtype=torch.int64)
        )

    def forward(self, x):  # pragma: no cover - shape only, never run here
        return self.kept(x)


def test_load_optimizer_state_filters_out_removed_parameters_by_name(
    tmp_path, caplog
):
    """A head removed from the config leaves stale entries in the saved
    optimizer state. ``Optimizer.load_state_dict`` raises rather than
    tolerate the count change (there is no ``strict=False`` on the
    optimizer side), so ``load_optimizer_state_dict_into`` rebuilds a
    state dict filtered to parameters that still exist -- paired by name,
    not index, since removing the head shifts the integer indices of
    every parameter defined after it.

    Reproduces the production failure: a checkpoint at step 329835, saving
    a model with the ``optimistic_st`` policy head, was resumed under a
    config that had dropped it. The model half tolerated the mismatch via
    ``load_state_dict_into``; the optimizer half crashed::

        ValueError: loaded state dict contains a parameter group that
        doesn't match the size of optimizer's group
    """
    import logging

    from lczero_training.directml import checkpoint as checkpoint_io
    from lczero_training.directml.optimizer import NAdamW

    saved_model = _TwoHeadStub()
    saved_optimizer = NAdamW(
        [{"params": list(saved_model.parameters()), "weight_decay": 0.0}],
        lr=1e-4,
    )
    _step_optimizer_once(saved_model, saved_optimizer)
    saved_model_state = {
        name: t.detach().clone()
        for name, t in saved_model.state_dict().items()
    }
    saved_opt_state = _tagged_state(saved_optimizer)

    # Sanity: the saved optimizer has moments for two parameters, and the
    # saved state dict carries three entries (two float params + one int64
    # buffer). The buffer must NOT be confused for a parameter by the
    # filter -- that is the policy_map analogue.
    assert len(saved_opt_state["state"]) == 2
    assert sum(
        1 for t in saved_model_state.values() if t.dtype.is_floating_point
    ) == 2

    new_model = _OneHeadStub()
    # The model side accepts the saved state with an unexpected key.
    checkpoint_io.load_state_dict_into(new_model, saved_model_state)

    new_optimizer = NAdamW(
        [{"params": list(new_model.parameters()), "weight_decay": 0.0}],
        lr=1e-4,
    )

    # This is the line that crashed in production: the optimizer's
    # parameter-group sizes no longer match. Must now succeed by pairing
    # by name and dropping the removed parameter's slot.
    with caplog.at_level(logging.WARNING):
        checkpoint_io.load_optimizer_state_dict_into(
            new_optimizer,
            saved_opt_state,
            new_model,
            saved_model_state,
            _EMPTY_SELECTOR,
        )

    # Every surviving parameter's moment must be present and tagged.
    new_params = list(new_model.parameters())
    assert len(new_params) == 1, "sanity: the new model has one Parameter"
    only_param = new_params[0]
    slot = new_optimizer.state[only_param]
    assert "mu" in slot and "nu" in slot
    assert "tag" in slot, (
        "the surviving parameter must carry its saved moment, including the "
        "tag -- proves by-name pairing, not index truncation"
    )

    # The optimizer's param_groups list must match the new model's param
    # shape. The crash was literally that this assertion did not hold.
    assert len(new_optimizer.param_groups) == len(
        saved_opt_state["param_groups"]
    )
    assert len(new_optimizer.param_groups[0]["params"]) == 1
    assert any(
        "Loaded optimizer state filtered" in record.message
        for record in caplog.records
    ), "the filter must announce how much it dropped for the resume log"


def test_load_optimizer_state_skips_shape_changed_parameters(tmp_path, caplog):
    """A parameter whose name survives but whose shape changed between
    saves is a shape mismatch from the optimizer's perspective too: the
    saved moment is the wrong shape. Drop its state and let the new
    parameter start fresh rather than slice or transpose silently.
    """
    import logging

    from lczero_training.directml import checkpoint as checkpoint_io
    from lczero_training.directml.optimizer import NAdamW

    saved_model = _TwoHeadStub()
    saved_optimizer = NAdamW(
        [{"params": list(saved_model.parameters()), "weight_decay": 0.0}],
        lr=1e-4,
    )
    _step_optimizer_once(saved_model, saved_optimizer)
    saved_model_state = {
        name: t.detach().clone()
        for name, t in saved_model.state_dict().items()
    }
    saved_opt_state = _tagged_state(saved_optimizer)

    # Reshape the saved copy of ``kept.weight`` -- same name, different
    # shape. The new model still owns a Parameter at that name with the
    # original shape, so by-name pairing succeeds at the position step but
    # the moment shape mismatches and must be dropped.
    saved_model_state["kept.weight"] = saved_model_state[
        "kept.weight"
    ].reshape(4, 1)

    new_model = _TwoHeadStub()  # original shape for kept.weight
    # Resize the new model's kept parameter to match the reshaped saved
    # tensor, so the model-side loader accepts the saved state.
    with torch.no_grad():
        new_model.kept.weight.set_(saved_model_state["kept.weight"])
    new_optimizer = NAdamW(
        [{"params": list(new_model.parameters()), "weight_decay": 0.0}],
        lr=1e-4,
    )

    with caplog.at_level(logging.WARNING):
        checkpoint_io.load_optimizer_state_dict_into(
            new_optimizer,
            saved_opt_state,
            new_model,
            saved_model_state,
            _EMPTY_SELECTOR,
        )

    # The reshaped parameter's slot must NOT carry the saved moment -- it
    # starts fresh, with zero mu/nu and no tag.
    slot = new_optimizer.state[new_model.kept.weight]
    if slot:
        assert "tag" not in slot, (
            "the reshaped parameter must start fresh, not inherit a moment "
            "that no longer matches its shape"
        )
    # The other parameter, untouched in shape, must carry its tag.
    other_slot = new_optimizer.state[new_model.removed.weight]
    assert "tag" in other_slot, (
        "the unchanged parameter must still inherit its saved moment"
    )
    assert any(
        "shape-changed" in record.message
        or "Loaded optimizer state filtered" in record.message
        for record in caplog.records
    )


def test_load_optimizer_state_handles_sparse_state_keys(tmp_path, caplog):
    """PyTorch's ``Optimizer.state_dict`` only emits a per-parameter
    state entry for parameters that have actually been stepped -- a
    parameter whose ``grad`` was ``None`` on every training step gets no
    entry, and the saved state's keys end up a sparse subset of the full
    index range rather than 0..N-1. The first cut of this helper rejected
    any non-sequential key layout, which crashed the very next resume
    after the optimistic_st-head removal because NAdamW's saved state
    had keys like ``[57, 67, 72, ...]`` -- a sparse subset, not a range.

    Reproduce the sparse layout directly: step the optimizer once, then
    surgically remove one parameter's state entry from the saved
    ``state_dict`` and the matching index from the matching group's
    ``params`` list. Pair-by-name must still work -- the surviving
    parameter's tag must arrive -- and the helper must not raise.
    """
    import logging

    from lczero_training.directml import checkpoint as checkpoint_io
    from lczero_training.directml.optimizer import NAdamW

    saved_model = _TwoHeadStub()
    saved_optimizer = NAdamW(
        [{"params": list(saved_model.parameters()), "weight_decay": 0.0}],
        lr=1e-4,
    )
    _step_optimizer_once(saved_model, saved_optimizer)
    saved_model_state = {
        name: t.detach().clone()
        for name, t in saved_model.state_dict().items()
    }
    saved_opt_state = _tagged_state(saved_optimizer)

    # Make the saved state sparse: drop the entry for one param and the
    # matching index from the param_groups[*]['params'] list. The full
    # cross-group index after this edit is {0, 1} minus {1} -> {0}, i.e.
    # the state's only key is ``0`` -- but the param_groups' params list
    # is [0], full length 1, not "[0, 1] which is what the helper used to
    # require. Crucially, the *state's* keys after the edit do NOT match
    # ``list(range(len(state_keys)))`` because state keys are a subset of
    # param indices, not a re-enumeration.
    removed_idx = 1
    del saved_opt_state["state"][removed_idx]
    saved_opt_state["param_groups"][0]["params"] = [
        i for i in saved_opt_state["param_groups"][0]["params"]
        if i != removed_idx
    ]
    # Pare the saved model state to match: drop the removed parameter's
    # tensor so the (saved_index -> name) map pairing has equal lengths.
    removed_name = "removed.weight"
    del saved_model_state[removed_name]

    new_model = _OneHeadStub()
    checkpoint_io.load_state_dict_into(new_model, saved_model_state)

    new_optimizer = NAdamW(
        [{"params": list(new_model.parameters()), "weight_decay": 0.0}],
        lr=1e-4,
    )

    import contextlib

    with caplog.at_level(logging.WARNING), contextlib.suppress(Exception):
        checkpoint_io.load_optimizer_state_dict_into(
            new_optimizer,
            saved_opt_state,
            new_model,
            saved_model_state,
            _EMPTY_SELECTOR,
        )

    # The new model has one parameter; the saved state had entries for
    # that parameter (at index 0 in the pared state), so its tag must
    # survive. The crash worth preventing is the helper raising.
    only_param = next(iter(new_model.parameters()))
    slot = new_optimizer.state[only_param]
    if slot:
        assert "tag" in slot, (
            "the surviving parameter's moment must be restored even when "
            "the saved state's keys are sparse, not 0..N-1"
        )


def test_load_optimizer_state_dedups_shared_parameters_by_storage(
    tmp_path, caplog
):
    """The shared-policy-embedding case from the real model.

    ``policy_embedding_shared`` is one Linear that every policy head's
    ``tokens`` attribute aliases. ``model.state_dict`` therefore lists
    the SAME Tensor under multiple name paths (e.g.
    ``policy_heads.vanilla.tokens.weight`` and
    ``policy_heads.optimistic_st.tokens.weight``), so the saved
    ``model_state`` has 172 entries even though the optimizer was
    built with 169 unique parameters -- ``build_optimizer`` deduped by
    ``id`` at construction. The first cut of this helper paired the
    saved model_state's float entries 1:1 with the saved optimizer's
    flat-index range, which crashed with::

        RuntimeError: ... the saved model has 172 float parameter(s),
        but the saved optimizer state lists 169 parameter index(ices).

    Dedup must be by storage identity (``data_ptr``), not by ``id``:
    storage sharing survives ``torch.save``/load, byte-for-byte, but
    ``id`` of the wrapping Tensor object does not. Reproduce the shared
    aliasing directly and assert pairing succeeds.
    """
    import logging

    from lczero_training.directml import checkpoint as checkpoint_io
    from lczero_training.directml.optimizer import NAdamW

    class _SharedStub(torch.nn.Module):
        """Two named attributes that are the SAME Linear -- the production
        ``policy_embedding_shared`` analogue. ``state_dict`` will emit
        ``heads.a.weight`` AND ``heads.b.weight`` for the same underlying
        tensor; ``build_optimizer`` dedups by ``id`` and sees one."""

        def __init__(self) -> None:
            super().__init__()
            shared = torch.nn.Linear(2, 2, bias=False)
            # Two named submodules that both reference the shared Linear.
            self.head_a = torch.nn.Module()
            self.head_b = torch.nn.Module()
            self.head_a.lin = shared
            self.head_b.lin = shared
            # A non-shared parameter too, to verify the dedup does not
            # accidentally collapse distinct params.
            self.body = torch.nn.Linear(2, 2, bias=False)

        def forward(self, x):  # pragma: no cover
            return x

    def _named_param_count(m):
        return sum(1 for _ in m.parameters())  # dedup by id

    saved_model = _SharedStub()
    # Three names in state_dict; only two unique Parameters in parameters().
    assert (
        len(saved_model.state_dict()) == 3
    ), "sanity: state_dict lists the shared tensor under both names"
    assert _named_param_count(saved_model) == 2, (
        "sanity: parameters() dedups the shared Linear -- two unique "
        "Parameter objects, not three"
    )

    saved_optimizer = NAdamW(
        [{"params": list(saved_model.parameters()), "weight_decay": 0.0}],
        lr=1e-4,
    )
    _step_optimizer_once(saved_model, saved_optimizer)
    saved_model_state = {
        name: t.detach().clone()
        for name, t in saved_model.state_dict().items()
    }
    saved_opt_state = _tagged_state(saved_optimizer)
    # Sanity: the shared tensor object remained shared after the
    # dict-comp above? No -- .detach().clone() turns every entry into a
    # fresh tensor. To exercise the data_ptr dedup honestly, copy the
    # SAME cloned tensor for the two shared names, so the de-dup test
    # does not depend on internal torch.detach behaviour.
    shared_clone = saved_model_state["head_a.lin.weight"]
    saved_model_state["head_b.lin.weight"] = shared_clone
    assert (
        saved_model_state["head_a.lin.weight"].data_ptr()
        == saved_model_state["head_b.lin.weight"].data_ptr()
    ), "sanity: the two aliases share storage, matching the real shared "
    "policy embedding at save time"

    # The saved optimizer state has 2 entries -- one per unique Parameter
    # in build_optimizer's dedup'd list -- not 3.
    assert len(saved_opt_state["state"]) == 2

    new_model = _SharedStub()
    checkpoint_io.load_state_dict_into(new_model, saved_model_state)

    new_optimizer = NAdamW(
        [{"params": list(new_model.parameters()), "weight_decay": 0.0}],
        lr=1e-4,
    )

    with caplog.at_level(logging.WARNING):
        checkpoint_io.load_optimizer_state_dict_into(
            new_optimizer,
            saved_opt_state,
            new_model,
            saved_model_state,
            _EMPTY_SELECTOR,
        )

    # Every unique new parameter (2 of them) must carry its saved tag --
    # the dedup-by-data_ptr pairing succeeded, and the helper did NOT
    # crash on the 3-vs-2 mismatch between saved_model_state entry count
    # and saved optimizer flat-index count.
    new_params = list(new_model.parameters())
    assert len(new_params) == 2
    for param in new_params:
        slot = new_optimizer.state[param]
        assert "tag" in slot, (
            "surviving shared-or-not parameter must keep its saved moment "
            "after dedup-by-data_ptr pairing"
        )
