"""Tests for the ported losses and optimizer."""

import pathlib

import numpy as np
import pytest

torch = pytest.importorskip("torch")
jax = pytest.importorskip("jax")
optax = pytest.importorskip("optax")

import jax.numpy as jnp
from google.protobuf import text_format

from lczero_training.directml.losses import LczeroLoss
from lczero_training.directml.optimizer import NAdamW, selector_includes
from lczero_training.directml.training import training_segments
from proto.root_config_pb2 import RootConfig

_CONFIG_PATH = (
    pathlib.Path(__file__).resolve().parents[3]
    / "docs"
    / "example_kda_real_import.textproto"
)


def _config() -> RootConfig:
    config = RootConfig()
    text_format.Parse(_CONFIG_PATH.read_text(), config)
    return config


def test_describe_error_names_an_exception_that_carries_no_message():
    """A real run logged "Training stopped at step 197835:" and nothing else.

    That was a deliberate Ctrl-C -- KeyboardInterrupt stringifies to the
    empty string -- but it read exactly like a crash whose message had been
    lost, in a log whose entire job is telling those apart.
    """
    from lczero_training.directml.training import describe_error

    assert describe_error(KeyboardInterrupt()) == "KeyboardInterrupt"
    assert describe_error(RuntimeError("")) == "RuntimeError"
    assert (
        describe_error(RuntimeError("Not enough memory resources"))
        == "RuntimeError: Not enough memory resources"
    )


def test_training_segments_reach_absolute_target():
    segments = list(training_segments(1000, 1_000_000, 1000))

    assert len(segments) == 999
    assert set(segments) == {1000}
    assert 1000 + sum(segments) == 1_000_000


def test_training_segments_shortens_final_segment():
    assert list(training_segments(1000, 3500, 1000)) == [1000, 1000, 500]


# --------------------------------------------------------------------------
# Optimizer
# --------------------------------------------------------------------------


@pytest.mark.parametrize("weight_decay", [0.0, 1e-4])
@pytest.mark.parametrize("start_step", [0, 97000])
def test_nadamw_matches_optax(weight_decay, start_step):
    """The whole point of the custom optimizer: bit-comparable to Optax.

    Also run from step 97000, since that is where the real import starts and
    the bias corrections differ wildly between step 1 and step 97001.
    """
    beta1, beta2, eps, lr = 0.9, 0.98, 1e-7, 1e-4
    rng = np.random.default_rng(0)
    initial = rng.normal(size=(6, 5)).astype(np.float32)
    grads = [rng.normal(size=(6, 5)).astype(np.float32) for _ in range(5)]

    # --- optax
    jax_params = jnp.asarray(initial)
    tx = optax.nadamw(
        lr, b1=beta1, b2=beta2, eps=eps, weight_decay=weight_decay
    )
    state = tx.init(jax_params)
    if start_step:
        from lczero_training.training.optimizer import update_optimizer_step

        state = update_optimizer_step(state, start_step)
    for grad in grads:
        updates, state = tx.update(jnp.asarray(grad), state, jax_params)
        jax_params = optax.apply_updates(jax_params, updates)

    # --- ours
    param = torch.nn.Parameter(torch.from_numpy(initial.copy()))
    optimizer = NAdamW(
        [param],
        lr=lr,
        betas=(beta1, beta2),
        eps=eps,
        weight_decay=weight_decay,
    )
    if start_step:
        optimizer.set_step(start_step)
    for grad in grads:
        optimizer.zero_grad()
        param.grad = torch.from_numpy(grad.copy())
        optimizer.step()

    np.testing.assert_allclose(
        param.detach().numpy(), np.asarray(jax_params), rtol=1e-5, atol=1e-6
    )


def _nadamw_step_by_expressions(
    params,
    grads,
    state,
    *,
    beta1,
    beta2,
    eps,
    eps_root,
    lr,
    weight_decay,
):
    """NAdamW.step exactly as it was written before the buffer rewrite.

    The oracle for ``test_nadamw_is_bit_exact_after_the_buffer_rewrite``.
    Every line here allocated a fresh tensor -- which is what the rewrite
    removed and what this must go on proving it did not change. Do not
    "tidy" this: its value is being a frozen copy, so reordering an
    operation here to look nicer would destroy the comparison.
    """
    for param, grad in zip(params, grads):
        # Not setdefault: it evaluates its default eagerly, so every step
        # would build two throwaway zeros_like tensors and any allocation
        # count taken through this helper would be two per parameter high.
        if id(param) not in state:
            state[id(param)] = {
                "step": 0,
                "mu": torch.zeros_like(param),
                "nu": torch.zeros_like(param),
            }
        entry = state[id(param)]
        mu, nu = entry["mu"], entry["nu"]
        mu.mul_(beta1).add_(grad, alpha=1.0 - beta1)
        nu.mul_(beta2).addcmul_(grad, grad, value=1.0 - beta2)
        count = entry["step"] + 1
        entry["step"] = count

        mu_hat = mu / (1.0 - beta1 ** (count + 1))
        grad_hat = grad / (1.0 - beta1**count)
        blended = beta1 * mu_hat + (1.0 - beta1) * grad_hat
        nu_hat = nu / (1.0 - beta2**count)

        update = blended / (torch.sqrt(nu_hat + eps_root) + eps)
        if weight_decay != 0.0:
            update = update + weight_decay * param
        param.add_(update, alpha=-lr)


@pytest.mark.parametrize("weight_decay", [0.0, 1e-4])
@pytest.mark.parametrize("start_step", [0, 97000])
def test_nadamw_is_bit_exact_after_the_buffer_rewrite(weight_decay, start_step):
    """The rewrite removed ~10 allocations per parameter per step. It must
    not have moved a single bit of the result.

    ``test_nadamw_matches_optax`` above only holds this to rtol=1e-5, which
    is the right tolerance for a cross-framework comparison and far too
    loose for a refactor: a reassociated multiply would pass it while
    quietly changing the trajectory of a network 195,000 steps in. So this
    compares against a frozen copy of the old expressions and demands
    equality to the bit.

    Several shapes and several steps, because the bias corrections change
    every step and a 1-D parameter exercises different kernels from a 2-D one.
    """
    beta1, beta2, eps, eps_root, lr = 0.9, 0.98, 1e-7, 0.0, 1e-4
    rng = np.random.default_rng(7)
    shapes = [(6, 5), (17,), (3, 4, 4)]
    initial = [rng.normal(size=s).astype(np.float32) for s in shapes]
    grads = [
        [rng.normal(size=s).astype(np.float32) for s in shapes]
        for _ in range(6)
    ]

    ours = [torch.nn.Parameter(torch.from_numpy(a.copy())) for a in initial]
    optimizer = NAdamW(
        ours,
        lr=lr,
        betas=(beta1, beta2),
        eps=eps,
        weight_decay=weight_decay,
        eps_root=eps_root,
    )
    if start_step:
        optimizer.set_step(start_step)

    reference = [torch.from_numpy(a.copy()) for a in initial]
    reference_state: dict = {}
    if start_step:
        for param in reference:
            reference_state[id(param)] = {
                "step": start_step,
                "mu": torch.zeros_like(param),
                "nu": torch.zeros_like(param),
            }

    for step_grads in grads:
        optimizer.zero_grad()
        for param, grad in zip(ours, step_grads):
            param.grad = torch.from_numpy(grad.copy())
        optimizer.step()
        _nadamw_step_by_expressions(
            reference,
            [torch.from_numpy(g.copy()) for g in step_grads],
            reference_state,
            beta1=beta1,
            beta2=beta2,
            eps=eps,
            eps_root=eps_root,
            lr=lr,
            weight_decay=weight_decay,
        )

    for index, (mine, theirs) in enumerate(zip(ours, reference)):
        assert torch.equal(mine.detach(), theirs), (
            f"parameter {index} of shape {shapes[index]} diverged from the "
            f"pre-rewrite implementation; max |delta| "
            f"{(mine.detach() - theirs).abs().max().item():.3e}"
        )


def test_nadamw_reuses_its_scratch_buffers_across_steps():
    """The point of the rewrite. If a future edit reintroduces expression
    temporaries this stays green, so it checks the mechanism instead: the
    buffers exist, there are exactly two per parameter, and step() does not
    allocate new ones."""
    param = torch.nn.Parameter(torch.zeros(4, 4))
    optimizer = NAdamW([param], lr=1e-4)

    param.grad = torch.ones(4, 4)
    optimizer.step()
    first = optimizer._scratch[param]
    assert len(first) == 2

    param.grad = torch.ones(4, 4)
    optimizer.step()
    assert optimizer._scratch[param] is first, "buffers were reallocated"
    assert all(a is b for a, b in zip(optimizer._scratch[param], first))

    # And they are droppable, which is what the emergency checkpoint needs.
    optimizer.free_scratch()
    assert param not in optimizer._scratch


def test_scratch_buffers_stay_out_of_the_checkpoint():
    """They are ~48 MB of uninitialized memory for this model. In state_dict
    they would bloat every checkpoint and change its format."""
    param = torch.nn.Parameter(torch.zeros(8, 8))
    optimizer = NAdamW([param], lr=1e-4)
    param.grad = torch.ones(8, 8)
    optimizer.step()

    packed = optimizer.state_dict()
    keys = {key for entry in packed["state"].values() for key in entry}

    assert keys == {"step", "mu", "nu"}, f"unexpected checkpoint keys: {keys}"


def test_nadamw_differs_from_torch_nadam():
    """Guard the reason this optimizer exists at all.

    If a future PyTorch made torch.optim.NAdam Optax-equivalent, this test
    failing is the signal to delete the custom implementation.
    """
    rng = np.random.default_rng(1)
    initial = rng.normal(size=(4, 4)).astype(np.float32)
    grad = rng.normal(size=(4, 4)).astype(np.float32)

    ours = torch.nn.Parameter(torch.from_numpy(initial.copy()))
    theirs = torch.nn.Parameter(torch.from_numpy(initial.copy()))
    our_opt = NAdamW([ours], lr=1e-3, betas=(0.9, 0.98), eps=1e-7)
    their_opt = torch.optim.NAdam(
        [theirs], lr=1e-3, betas=(0.9, 0.98), eps=1e-7
    )
    for optimizer, param in ((our_opt, ours), (their_opt, theirs)):
        param.grad = torch.from_numpy(grad.copy())
        optimizer.step()

    assert not np.allclose(
        ours.detach().numpy(), theirs.detach().numpy(), rtol=1e-6, atol=1e-7
    ), "torch.optim.NAdam now matches Optax; the custom NAdamW is redundant"


def test_gradient_clipping_matches_optax():
    rng = np.random.default_rng(2)
    grads = [
        rng.normal(size=(4, 4)).astype(np.float32) * 20.0 for _ in range(2)
    ]
    max_norm = 10.0

    clipper = optax.clip_by_global_norm(max_norm)
    jax_grads = [jnp.asarray(g) for g in grads]
    expected, _ = clipper.update(jax_grads, clipper.init(jax_grads))

    tensors = [torch.from_numpy(g.copy()) for g in grads]
    params = [torch.nn.Parameter(t.clone()) for t in tensors]
    for param, tensor in zip(params, tensors):
        param.grad = tensor.clone()
    torch.nn.utils.clip_grad_norm_(params, max_norm)

    for param, want in zip(params, expected):
        np.testing.assert_allclose(
            param.grad.numpy(), np.asarray(want), rtol=1e-5, atol=1e-6
        )


# --------------------------------------------------------------------------
# Decay selector
# --------------------------------------------------------------------------


# --------------------------------------------------------------------------
# Learning-rate schedule
# --------------------------------------------------------------------------


def _schedule_proto(text: str):
    from google.protobuf import text_format as tf

    from proto.training_config_pb2 import LrSchedule

    rule = LrSchedule()
    tf.Parse(text, rule)
    return rule


SCHEDULE_CASES = [
    # Constant, as the real-import config uses.
    "starting_step: 0 duration_steps: 0 lr: 0.0001",
    # Linear warmup then a held rate.
    "starting_step: 0 duration_steps: 100 duration_steps: 100 "
    "lr: 0.0 lr: 0.001 lr: 0.001 transition: LINEAR transition: CONSTANT",
    # Cosine decay.
    "starting_step: 0 duration_steps: 500 lr: 0.001 lr: 0.00001 "
    "transition: COSINE",
    # Looping triangular cycle.
    "starting_step: 0 duration_steps: 50 duration_steps: 50 "
    "lr: 0.0001 lr: 0.001 lr: 0.0001 transition: LINEAR transition: LINEAR "
    "loop: true",
    # Starting part-way in, so steps before it use the pre-start rate.
    "starting_step: 1000 duration_steps: 200 lr: 0.002 lr: 0.0005 "
    "transition: LINEAR",
]


@pytest.mark.parametrize("text", SCHEDULE_CASES)
def test_lr_schedule_matches_jax(text):
    """The plain-Python schedule must track the jitted JAX one exactly."""
    from lczero_training.directml.lr_schedule import make_lr_schedule
    from lczero_training.training.lr_schedule import (
        make_lr_schedule as jax_make,
    )

    rules = [_schedule_proto(text)]
    ours = make_lr_schedule(rules)
    theirs = jax_make(rules)

    for step in [
        0,
        1,
        37,
        99,
        100,
        101,
        199,
        200,
        500,
        999,
        1000,
        1100,
        1200,
        5000,
    ]:
        np.testing.assert_allclose(
            ours(step),
            float(np.asarray(theirs(jnp.asarray(float(step))))),
            rtol=1e-6,
            atol=1e-9,
            err_msg=f"step {step} of {text!r}",
        )


def test_lr_schedule_matches_jax_for_multiple_rules():
    """Rule selection (largest starting_step at or below the step) must agree."""
    from lczero_training.directml.lr_schedule import make_lr_schedule
    from lczero_training.training.lr_schedule import (
        make_lr_schedule as jax_make,
    )

    rules = [
        _schedule_proto("starting_step: 0 duration_steps: 0 lr: 0.001"),
        _schedule_proto(
            "starting_step: 500 duration_steps: 500 lr: 0.001 lr: 0.0001 "
            "transition: LINEAR"
        ),
        _schedule_proto("starting_step: 1000 duration_steps: 0 lr: 0.00001"),
    ]
    ours = make_lr_schedule(rules)
    theirs = jax_make(rules)
    for step in [0, 250, 499, 500, 750, 999, 1000, 2000]:
        np.testing.assert_allclose(
            ours(step),
            float(np.asarray(theirs(jnp.asarray(float(step))))),
            rtol=1e-6,
            atol=1e-9,
            err_msg=f"step {step}",
        )


def test_decay_selector_matches_config_intent():
    selector = _config().training.optimizer.nadamw.decay_selector
    # Heads decay; biases, layer norms, and the input embedding do not.
    assert selector_includes(selector, "policy_heads.vanilla.q.weight")
    assert selector_includes(selector, "value_heads.winner.wdl.weight")
    assert selector_includes(selector, "movesleft_heads.main.out.weight")

    assert not selector_includes(selector, "policy_heads.vanilla.q.bias")
    assert not selector_includes(selector, "encoders.0.ln1.scale")
    assert not selector_includes(selector, "encoders.0.ln2.bias")
    assert not selector_includes(selector, "embedding.embedding.weight")
    # otherwise_include is false, so the tower body is undecayed.
    assert not selector_includes(selector, "encoders.0.mixer.q.weight")
    assert not selector_includes(selector, "encoders.3.mha.q.weight")


def test_decay_selector_agrees_with_jax_matcher():
    """Our 3.12 glob must agree with pathlib.PurePath.full_match.

    full_match only exists on 3.13, so compare against it only when the
    running interpreter has it.
    """
    from pathlib import PurePosixPath

    if not hasattr(PurePosixPath("a"), "full_match"):
        pytest.skip("PurePath.full_match requires Python 3.13")

    selector = _config().training.optimizer.nadamw.decay_selector
    names = [
        "policy_heads.vanilla.q.weight",
        "encoders.0.ln1.scale",
        "embedding.embedding.weight",
        "encoders.12.mixer.local_conv.bias",
        "value_heads.q.categorical.weight",
    ]
    for name in names:
        path = PurePosixPath(*name.split("."))
        expected = selector.otherwise_include
        for rule in selector.rule:
            if path.full_match(rule.match):
                expected = rule.include
                break
        assert selector_includes(selector, name) == expected, name


# --------------------------------------------------------------------------
# Gradient accumulation
# --------------------------------------------------------------------------


def _fixed_batches(count, batch_size, seed=0):
    """Deterministic batches, so two runs can be given identical data."""
    from lczero_training.directml.training import TrainingBatch

    generator = torch.Generator().manual_seed(seed)
    made = []
    for _ in range(count):
        logits = torch.randn(batch_size, 1858, generator=generator)
        made.append(
            TrainingBatch(
                inputs=torch.rand(batch_size, 112, 8, 8, generator=generator),
                probabilities=torch.softmax(logits, dim=-1),
                values=torch.rand(batch_size, 6, 3, generator=generator),
            )
        )
    return made


def _tiny_training_config():
    """The real architecture, shrunk until a CPU test is quick."""
    config = _config()
    config.model.embedding.dense_size = 8
    config.model.embedding.embedding_size = 16
    config.model.embedding.dff = 16
    del config.model.encoder.mixer_pattern[1:]
    config.model.encoder.num_blocks = 1
    config.model.encoder.d_model = 16
    config.model.encoder.dff = 16
    # Not shrinkable: the KDA mixer requires heads % len(directions) == 0,
    # and this config scans all 8 board directions.
    config.model.encoder.heads = 8
    config.model.encoder.kda.key_dim = 8
    config.model.encoder.kda.value_dim = 8
    config.model.encoder.kda.gate_rank = 8
    config.model.encoder.kda.chunk_size = 8
    config.model.shared_policy_embedding_size = 16
    for head in config.model.policy_head:
        head.d_model = 16
    for head in config.model.value_head:
        head.num_channels = 8
    for head in config.model.movesleft_head:
        head.num_channels = 8
    config.training.max_grad_norm = 0.0
    return config


def _grads_after_one_step(config, batches):
    """The gradient `train` produces for a single optimizer step."""
    from lczero_training.directml.model import LczeroModel
    from lczero_training.directml.training import train

    torch.manual_seed(1234)
    model = LczeroModel(config.model)
    optimizer = NAdamW(
        [{"params": list(model.parameters()), "weight_decay": 0.0}],
        lr=0.0,  # No movement, so the gradient is read at identical weights.
    )
    train(
        config=config,
        model=model,
        optimizer=optimizer,
        batches=iter(batches),
        device=torch.device("cpu"),
        start_step=0,
        steps=1,
        log_every=0,
        diagnostics=False,
    )
    return {
        name: parameter.grad.clone()
        for name, parameter in model.named_parameters()
        if parameter.grad is not None
    }


def test_step_releases_its_tensors_before_the_next_one():
    """Nothing from a finished step may still be referenced.

    `metrics`, `grad_norm` and `batch` stayed bound until the next
    iteration reassigned them, so the loop carried an extra step's tensors
    at all times. On DirectML a referenced block is one the allocator will
    not reuse, which is the shape of a slow creep to OOM over thousands of
    steps.

    Watches the first batch by weak reference: if the loop still holds it
    two steps later, it stays alive here too.
    """
    import gc
    import weakref

    from lczero_training.directml.model import LczeroModel
    from lczero_training.directml.training import train

    config = _tiny_training_config()
    torch.manual_seed(1234)
    model = LczeroModel(config.model)
    optimizer = NAdamW(
        [{"params": list(model.parameters()), "weight_decay": 0.0}], lr=0.0
    )

    batches = _fixed_batches(3, 4, seed=5)
    watched = weakref.ref(batches[0].inputs)

    def draining():
        # Pops as it yields, so the list stops holding a batch the moment
        # the loop takes it. Anything still alive afterwards is the loop's.
        while batches:
            yield batches.pop(0)

    train(
        config=config,
        model=model,
        optimizer=optimizer,
        batches=draining(),
        device=torch.device("cpu"),
        start_step=0,
        steps=3,
        log_every=0,
        diagnostics=False,
    )
    gc.collect()

    assert watched() is None, (
        "the first step's batch is still referenced after later steps"
    )


def _eval_steps(*, eval_every, steps, start_step, calls_per_segment=1):
    """The global steps `train` evaluates at, over one or more segments."""
    from lczero_training.directml.model import LczeroModel
    from lczero_training.directml.training import train

    config = _tiny_training_config()
    torch.manual_seed(1234)
    model = LczeroModel(config.model)
    optimizer = NAdamW(
        [{"params": list(model.parameters()), "weight_decay": 0.0}], lr=0.0
    )
    seen: list[int] = []
    step = start_step
    batches = iter(_fixed_batches(steps * calls_per_segment, 4, seed=7))
    for _ in range(calls_per_segment):
        step = train(
            config=config,
            model=model,
            optimizer=optimizer,
            batches=batches,
            device=torch.device("cpu"),
            start_step=step,
            steps=steps,
            log_every=0,
            diagnostics=False,
            eval_hook=seen.append,
            eval_every=eval_every,
        )
    return seen


def test_evaluation_fires_on_the_global_step_not_the_segment_index():
    """The cadence bug that made eval ten times more expensive than asked.

    `train` is called once per checkpoint interval, so the loop index
    restarts at 0 every 1,000 steps. The hook used to fire on
    `index % eval_every == 0 or last`, which for any eval_every above the
    interval meant the first and last step of every segment -- and every
    eval permanently cost the trainer device memory it could not reclaim.
    Four segments of 5 with eval_every 10 must evaluate at 10 and 20, not
    eight times.
    """
    seen = _eval_steps(
        eval_every=10, steps=5, start_step=0, calls_per_segment=4
    )

    assert seen == [10, 20]


def test_evaluation_counts_from_the_resumed_step():
    """A run resumed at 190,705 must still evaluate on multiples of 100, so
    the test series lands on the same steps across restarts."""
    seen = _eval_steps(
        eval_every=100, steps=120, start_step=190_705, calls_per_segment=1
    )

    assert seen == [190_800]


def test_evaluation_off_by_default():
    assert _eval_steps(eval_every=0, steps=4, start_step=0) == []


def test_gc_every_runs_a_collection_on_schedule():
    """The hook must fire, and only on the configured cadence."""
    import lczero_training.directml.training as training_module
    from lczero_training.directml.model import LczeroModel

    config = _tiny_training_config()
    torch.manual_seed(1234)
    model = LczeroModel(config.model)
    optimizer = NAdamW(
        [{"params": list(model.parameters()), "weight_decay": 0.0}], lr=0.0
    )

    calls: list[int] = []
    original = training_module.gc.collect

    def counting_collect(*args, **kwargs):
        calls.append(1)
        return original(*args, **kwargs)

    training_module.gc.collect = counting_collect
    try:
        training_module.train(
            config=config,
            model=model,
            optimizer=optimizer,
            batches=iter(_fixed_batches(6, 4, seed=6)),
            device=torch.device("cpu"),
            start_step=0,
            steps=6,
            log_every=0,
            diagnostics=False,
            gc_every=2,
        )
    finally:
        training_module.gc.collect = original

    # Steps 1..6 collecting when step % 2 == 0 -> steps 2, 4 and 6.
    assert len(calls) == 3, f"expected 3 collections, got {len(calls)}"


def test_gc_every_zero_never_collects():
    import lczero_training.directml.training as training_module
    from lczero_training.directml.model import LczeroModel

    config = _tiny_training_config()
    torch.manual_seed(1234)
    model = LczeroModel(config.model)
    optimizer = NAdamW(
        [{"params": list(model.parameters()), "weight_decay": 0.0}], lr=0.0
    )

    calls: list[int] = []
    original = training_module.gc.collect
    training_module.gc.collect = lambda *a, **k: (calls.append(1), 0)[1]
    try:
        training_module.train(
            config=config,
            model=model,
            optimizer=optimizer,
            batches=iter(_fixed_batches(3, 4, seed=7)),
            device=torch.device("cpu"),
            start_step=0,
            steps=3,
            log_every=0,
            diagnostics=False,
            gc_every=0,
        )
    finally:
        training_module.gc.collect = original

    assert calls == []


def test_accumulated_gradient_matches_the_whole_batch():
    """Four micro-batches of 8 must equal one batch of 32.

    This is the entire promise of the feature: the same update as a batch
    that does not fit in memory. A missing 1/accumulation divide, or a
    zero_grad in the wrong place, breaks it and nothing else would notice.
    """
    micro = _fixed_batches(4, 8, seed=7)
    whole = [
        type(micro[0])(
            inputs=torch.cat([b.inputs for b in micro]),
            probabilities=torch.cat([b.probabilities for b in micro]),
            values=torch.cat([b.values for b in micro]),
        )
    ]

    accumulated_config = _tiny_training_config()
    accumulated_config.training.gradient_accumulation_steps = 4
    accumulated = _grads_after_one_step(accumulated_config, micro)

    single = _grads_after_one_step(_tiny_training_config(), whole)

    assert set(accumulated) == set(single)
    for name, expected in single.items():
        torch.testing.assert_close(
            accumulated[name], expected, rtol=2e-4, atol=2e-6, msg=name
        )


def test_accumulation_consumes_one_batch_per_micro_step():
    """Steps count optimizer updates, not batches.

    The LR schedule, checkpoint cadence and export filenames are all keyed
    on the returned step, so an accumulating run must not inflate it.
    """
    from lczero_training.directml.training import train

    config = _tiny_training_config()
    config.training.gradient_accumulation_steps = 3
    batches = _fixed_batches(6, 4, seed=11)
    consumed = iter(batches)

    from lczero_training.directml.model import LczeroModel

    torch.manual_seed(1234)
    model = LczeroModel(config.model)
    optimizer = NAdamW(
        [{"params": list(model.parameters()), "weight_decay": 0.0}], lr=0.0
    )
    final = train(
        config=config,
        model=model,
        optimizer=optimizer,
        batches=consumed,
        device=torch.device("cpu"),
        start_step=100,
        steps=2,
        log_every=0,
        diagnostics=False,
    )

    assert final == 102, "2 steps must advance the step counter by 2"
    assert next(consumed, None) is None, "2 steps x 3 micro = all 6 batches"


def test_reported_metrics_average_over_the_micro_batches():
    """A reported loss must describe the effective batch, not the last
    micro-batch -- otherwise accumulation would improve the gradient while
    leaving the logs as noisy as before, which is half the point."""
    from lczero_training.directml.model import LczeroModel
    from lczero_training.directml.training import train

    config = _tiny_training_config()
    config.training.gradient_accumulation_steps = 4
    micro = _fixed_batches(4, 8, seed=7)

    seen: list[dict] = []

    torch.manual_seed(1234)
    model = LczeroModel(config.model)
    optimizer = NAdamW(
        [{"params": list(model.parameters()), "weight_decay": 0.0}], lr=0.0
    )
    train(
        config=config,
        model=model,
        optimizer=optimizer,
        batches=iter(micro),
        device=torch.device("cpu"),
        start_step=0,
        steps=1,
        log_every=0,
        reporters=[lambda step, scalars: seen.append(scalars)],
        report_every=1,
        diagnostics=False,
    )

    assert len(seen) == 1
    reported = seen[0]["total"]

    # The same weights scored on each micro-batch separately.
    torch.manual_seed(1234)
    reference = LczeroModel(config.model)
    reference.eval()
    loss_fn = LczeroLoss(config.training.losses)
    with torch.no_grad():
        per_micro = [
            float(loss_fn(reference(b.inputs), b, reference)[0]) for b in micro
        ]

    assert reported == pytest.approx(sum(per_micro) / len(per_micro), rel=1e-4)
    # And it is genuinely an average, not just the final micro-batch.
    assert reported != pytest.approx(per_micro[-1], rel=1e-6)


def test_accumulation_metrics_are_host_floats():
    """Metrics accumulated across micro-batches must be host Python floats,
    not device PyTorch tensors, preventing GPU allocation churn."""
    from lczero_training.directml.model import LczeroModel
    from lczero_training.directml.training import train

    config = _tiny_training_config()
    config.training.gradient_accumulation_steps = 2
    micro = _fixed_batches(2, 8, seed=42)

    seen: list[dict] = []
    model = LczeroModel(config.model)
    optimizer = NAdamW(
        [{"params": list(model.parameters()), "weight_decay": 0.0}], lr=0.0
    )
    train(
        config=config,
        model=model,
        optimizer=optimizer,
        batches=iter(micro),
        device=torch.device("cpu"),
        start_step=0,
        steps=1,
        log_every=0,
        reporters=[lambda step, scalars: seen.append(scalars)],
        report_every=1,
        diagnostics=True,
    )

    assert len(seen) == 1
    scalars = seen[0]
    for key, val in scalars.items():
        assert isinstance(val, float), f"{key} must be a float, got {type(val)}"


def test_release_to_host_clears_scratch_and_moves_tensors():
    """release_to_host must free optimizer scratch and move parameters to CPU."""
    from lczero_training.directml.model import LczeroModel
    from lczero_training.directml.training import release_to_host

    config = _tiny_training_config()
    model = LczeroModel(config.model)
    optimizer = NAdamW(
        [{"params": list(model.parameters()), "weight_decay": 0.0}], lr=0.0
    )
    # Populate scratch buffer
    for p in model.parameters():
        _ = optimizer._scratch_for(p)
    assert len(optimizer._scratch) > 0

    moved = release_to_host(model, optimizer)
    assert len(optimizer._scratch) == 0
    for p in model.parameters():
        assert p.device.type == "cpu"
