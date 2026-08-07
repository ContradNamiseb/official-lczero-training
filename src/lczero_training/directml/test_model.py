"""Parity tests for the PyTorch model against the JAX reference.

Built from the real target configuration in
docs/example_kda_real_import.textproto -- a 4-block hybrid tower (3 KDA +
1 MHA with shared Smolgen), 2 policy heads, 3 value heads, 1 moves-left
head -- so this exercises exactly the architecture the port has to train.
"""

import pathlib

import numpy as np
import pytest

torch = pytest.importorskip("torch")
jax = pytest.importorskip("jax")

import jax.numpy as jnp
from flax import nnx
from google.protobuf import text_format

from lczero_training.directml.model import LczeroModel as TorchModel
from lczero_training.model.model import LczeroModel as JaxModel
from lczero_training.model.policy_map import POLICY_MAP
from proto.root_config_pb2 import RootConfig

_CONFIG_PATH = (
    pathlib.Path(__file__).resolve().parents[3]
    / "docs"
    / "example_kda_real_import.textproto"
)


def _target_model_config():
    config = RootConfig()
    text_format.Parse(_CONFIG_PATH.read_text(), config)
    return config.model


def _all_heads_loss(prediction) -> "torch.Tensor":
    """Touch every output, so no parameter is legitimately gradient-free.

    The config has three value heads and two policy heads; a loss naming
    only one of each would leave the others unused and their parameters
    with grad None, which says nothing about correctness.
    """
    terms = []
    for logits in prediction.policy.values():
        terms.append(logits.square().sum())
    for wdl, error, categorical in prediction.value.values():
        terms.append(wdl.square().sum())
        if error is not None:
            terms.append(error.square().sum())
        if categorical is not None:
            terms.append(categorical.square().sum())
    for movesleft in prediction.movesleft.values():
        terms.append(movesleft.square().sum())
    return sum(terms[1:], terms[0])


def _assert_close(actual, expected, rtol=2e-4, atol=2e-5):
    np.testing.assert_allclose(
        np.asarray(actual, dtype=np.float64),
        np.asarray(expected, dtype=np.float64),
        rtol=rtol,
        atol=atol,
    )


# --------------------------------------------------------------------------
# torch -> jax weight copying
# --------------------------------------------------------------------------


def _set(param, value: np.ndarray) -> None:
    param.value = jnp.asarray(value)


def _copy_linear(torch_linear, jax_linear) -> None:
    # nn.Linear.weight is (out, in); the flax kernel is (in, out).
    _set(jax_linear.kernel, torch_linear.weight.detach().numpy().T)
    if torch_linear.bias is not None:
        _set(jax_linear.bias, torch_linear.bias.detach().numpy())


def _copy_layer_norm(torch_norm, jax_norm) -> None:
    _set(jax_norm.scale, torch_norm.scale.detach().numpy())
    _set(jax_norm.bias, torch_norm.bias.detach().numpy())


def _copy_conv(torch_conv, jax_conv) -> None:
    # torch (out, in/groups, kh, kw) -> flax (kh, kw, in/groups, out).
    weight = torch_conv.weight.detach().numpy()
    _set(jax_conv.kernel, weight.transpose(2, 3, 1, 0))
    _set(jax_conv.bias, torch_conv.bias.detach().numpy())


def _copy_ffn(torch_ffn, jax_ffn) -> None:
    _copy_linear(torch_ffn.linear1, jax_ffn.linear1)
    _copy_linear(torch_ffn.linear2, jax_ffn.linear2)


def _copy_kda(torch_mixer, jax_mixer) -> None:
    for name in ("q", "k", "v", "decay_a", "decay_b", "beta", "output_dense"):
        _copy_linear(getattr(torch_mixer, name), getattr(jax_mixer, name))
    if torch_mixer.output_gate:
        _copy_linear(torch_mixer.gate_a, jax_mixer.gate_a)
        _copy_linear(torch_mixer.gate_b, jax_mixer.gate_b)
    if torch_mixer.local_conv is not None:
        _copy_conv(torch_mixer.local_conv.conv, jax_mixer.local_conv.conv)
    if torch_mixer.rms_norm is not None:
        _set(
            jax_mixer.rms_norm_gammas,
            torch_mixer.rms_norm.scale.detach().numpy(),
        )
    _set(
        jax_mixer.log_decay.a_log, torch_mixer.log_decay.a_log.detach().numpy()
    )
    _set(
        jax_mixer.log_decay.dt_bias,
        torch_mixer.log_decay.dt_bias.detach().numpy(),
    )


def _copy_mha(torch_mha, jax_mha) -> None:
    for name in ("q", "k", "v", "output_dense"):
        _copy_linear(getattr(torch_mha, name), getattr(jax_mha, name))
    if torch_mha.smolgen is not None:
        _copy_linear(torch_mha.smolgen.compress, jax_mha.smolgen.compress)
        _copy_linear(torch_mha.smolgen.dense1, jax_mha.smolgen.dense1)
        _copy_linear(torch_mha.smolgen.dense2, jax_mha.smolgen.dense2)
        _copy_layer_norm(torch_mha.smolgen.ln1, jax_mha.smolgen.ln1)
        _copy_layer_norm(torch_mha.smolgen.ln2, jax_mha.smolgen.ln2)


def _copy_model(torch_model, jax_model) -> None:
    embedding = torch_model.embedding
    _copy_linear(embedding.preprocess, jax_model.embedding.preprocess)
    _copy_linear(embedding.embedding, jax_model.embedding.embedding)
    _copy_layer_norm(embedding.norm, jax_model.embedding.norm)
    _set(
        jax_model.embedding.ma_gating.mult_gate.gate,
        embedding.ma_gating.mult_gate.gate.detach().numpy(),
    )
    _set(
        jax_model.embedding.ma_gating.add_gate.gate,
        embedding.ma_gating.add_gate.gate.detach().numpy(),
    )
    _copy_ffn(embedding.ffn, jax_model.embedding.ffn)
    _copy_layer_norm(embedding.out_norm, jax_model.embedding.out_norm)

    # The shared Smolgen dense is one object referenced by every MHA block;
    # copying it once is deliberate.
    if torch_model.encoders.smolgen_shared_gen_dense is not None:
        jax_shared = None
        for jax_block in jax_model.encoders.encoders.layers:
            if not jax_block.is_kda and jax_block.mha.smolgen is not None:
                jax_shared = jax_block.mha.smolgen.weight_gen_dense
                break
        assert jax_shared is not None
        _copy_linear(torch_model.encoders.smolgen_shared_gen_dense, jax_shared)

    torch_blocks = list(torch_model.encoders.encoders)
    jax_blocks = list(jax_model.encoders.encoders.layers)
    assert len(torch_blocks) == len(jax_blocks)
    for torch_block, jax_block in zip(torch_blocks, jax_blocks):
        assert torch_block.is_kda == jax_block.is_kda
        if torch_block.is_kda:
            _copy_kda(torch_block.mixer, jax_block.mixer)
        else:
            _copy_mha(torch_block.mha, jax_block.mha)
        _copy_layer_norm(torch_block.ln1, jax_block.ln1)
        _copy_ffn(torch_block.ffn, jax_block.ffn)
        _copy_layer_norm(torch_block.ln2, jax_block.ln2)

    if torch_model.policy_embedding_shared is not None:
        _copy_linear(
            torch_model.policy_embedding_shared,
            jax_model.policy_embedding_shared,
        )
    for name, head in torch_model.policy_heads.items():
        jax_head = jax_model.policy_heads[name]
        if torch_model.policy_embedding_shared is None:
            _copy_linear(head.tokens, jax_head.tokens)
        _copy_linear(head.q, jax_head.q)
        _copy_linear(head.k, jax_head.k)
        _copy_linear(head.promotion_dense, jax_head.promotion_dense)

    for name, head in torch_model.value_heads.items():
        jax_head = jax_model.value_heads[name]
        _copy_linear(head.embed, jax_head.embed)
        _copy_linear(head.dense1, jax_head.dense1)
        _copy_linear(head.wdl, jax_head.wdl)
        if head.error is not None:
            _copy_linear(head.error, jax_head.error)
        if head.categorical is not None:
            _copy_linear(head.categorical, jax_head.categorical)

    for name, head in torch_model.movesleft_heads.items():
        jax_head = jax_model.movesleft_heads[name]
        _copy_linear(head.embed, jax_head.embed)
        _copy_linear(head.dense1, jax_head.dense1)
        _copy_linear(head.out, jax_head.out)


@pytest.fixture(scope="module")
def matched_models():
    config = _target_model_config()
    torch_model = TorchModel(config)
    torch_model.eval()
    jax_model = JaxModel(config=config, rngs=nnx.Rngs(params=0))
    _copy_model(torch_model, jax_model)
    return torch_model, jax_model, config


@pytest.fixture(scope="module")
def sample_inputs():
    rng = np.random.default_rng(3)
    return rng.normal(size=(2, 112, 8, 8)).astype(np.float32)


# --------------------------------------------------------------------------
# Parity
# --------------------------------------------------------------------------


def test_policy_map_matches_jax():
    from lczero_training.model.policy_head import _policy_map

    np.testing.assert_array_equal(
        np.asarray(_policy_map), np.asarray(POLICY_MAP)
    )
    assert len(POLICY_MAP) == 1858


def test_model_policy_matches_jax(matched_models, sample_inputs):
    torch_model, jax_model, _ = matched_models
    with torch.no_grad():
        actual = torch_model(torch.from_numpy(sample_inputs))
    for index in range(sample_inputs.shape[0]):
        expected = jax_model(jnp.asarray(sample_inputs[index]))
        for name, logits in actual.policy.items():
            _assert_close(logits[index].numpy(), expected.policy[name])


def test_model_value_matches_jax(matched_models, sample_inputs):
    torch_model, jax_model, _ = matched_models
    with torch.no_grad():
        actual = torch_model(torch.from_numpy(sample_inputs))
    for index in range(sample_inputs.shape[0]):
        expected = jax_model(jnp.asarray(sample_inputs[index]))
        for name, (wdl, error, categorical) in actual.value.items():
            exp_wdl, exp_error, exp_categorical = expected.value[name]
            _assert_close(wdl[index].numpy(), exp_wdl)
            assert (error is None) == (exp_error is None)
            if error is not None:
                _assert_close(error[index].numpy(), exp_error)
            assert (categorical is None) == (exp_categorical is None)
            if categorical is not None:
                _assert_close(categorical[index].numpy(), exp_categorical)


def test_model_movesleft_matches_jax(matched_models, sample_inputs):
    torch_model, jax_model, _ = matched_models
    with torch.no_grad():
        actual = torch_model(torch.from_numpy(sample_inputs))
    for index in range(sample_inputs.shape[0]):
        expected = jax_model(jnp.asarray(sample_inputs[index]))
        for name, movesleft in actual.movesleft.items():
            _assert_close(movesleft[index].numpy(), expected.movesleft[name])


def test_model_head_names_match_config(matched_models):
    torch_model, _, config = matched_models
    assert set(torch_model.policy_heads) == {h.name for h in config.policy_head}
    assert set(torch_model.value_heads) == {h.name for h in config.value_head}
    assert set(torch_model.movesleft_heads) == {
        h.name for h in config.movesleft_head
    }


def test_model_mixer_pattern_matches_config(matched_models):
    torch_model, _, config = matched_models
    blocks = list(torch_model.encoders.encoders)
    assert [b.is_kda for b in blocks] == [True, True, True, False]


def test_model_gradients_are_finite(matched_models, sample_inputs):
    torch_model, _, _ = matched_models
    x = torch.from_numpy(sample_inputs).requires_grad_(True)
    out = torch_model(x)
    loss = _all_heads_loss(out)
    loss.backward()
    assert x.grad is not None and bool(torch.isfinite(x.grad).all())
    for name, parameter in torch_model.named_parameters():
        assert parameter.grad is not None, name
        assert bool(torch.isfinite(parameter.grad).all()), name


# --------------------------------------------------------------------------
# DirectML leg
# --------------------------------------------------------------------------


def test_model_runs_on_directml(dml_device, sample_inputs):
    config = _target_model_config()
    model = TorchModel(config).to(dml_device)
    x = torch.from_numpy(sample_inputs).to(dml_device)
    out = model(x)
    loss = _all_heads_loss(out)
    loss.backward()
    assert bool(torch.isfinite(out.policy["vanilla"].detach()).all().cpu())
    for name, parameter in model.named_parameters():
        assert parameter.grad is not None, name
        assert bool(torch.isfinite(parameter.grad).all().cpu()), name


def test_model_directml_matches_cpu(sample_inputs, dml_device):
    config = _target_model_config()
    model = TorchModel(config)
    model.eval()
    x = torch.from_numpy(sample_inputs)
    with torch.no_grad():
        expected = model(x).policy["vanilla"].numpy()
        actual = model.to(dml_device)(x.to(dml_device))
        actual = actual.policy["vanilla"].cpu().numpy()
    _assert_close(actual, expected, rtol=1e-3, atol=1e-4)
