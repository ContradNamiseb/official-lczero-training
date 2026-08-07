"""Tests for the Leela -> PyTorch weight importer.

The real-network tests need an actual `.pb.gz`. Point LC0_TEST_NETWORK at
one to enable them; they skip otherwise, so CI without the file still runs
everything else.
"""

import os
import pathlib

import numpy as np
import pytest

torch = pytest.importorskip("torch")
jax = pytest.importorskip("jax")

import jax.numpy as jnp
from flax import nnx

from lczero_training.directml.leela_to_torch import (
    leela_config,
    leela_to_torch,
    read_leela_net,
)
from lczero_training.directml.model import LczeroModel as TorchModel
from lczero_training.directml.weight_map import (
    copy_torch_to_jax,
    iter_param_pairs,
)
from lczero_training.model.model import LczeroModel as JaxModel

_DEFAULT_NETWORK = pathlib.Path(
    r"C:\Users\Contrad\Documents\Code\repos\lc0-training\stable-branch\tf"
    r"\networks\kda-hybrid-128x4-3k1m-8h-small-gen-xpu"
    r"\kda-hybrid-128x4-3k1m-8h-small-gen-xpu-97000.pb.gz"
)


def _network_path() -> pathlib.Path:
    override = os.environ.get("LC0_TEST_NETWORK")
    return pathlib.Path(override) if override else _DEFAULT_NETWORK


@pytest.fixture(scope="module")
def imported():
    path = _network_path()
    if not path.exists():
        pytest.skip(f"no test network at {path}; set LC0_TEST_NETWORK")
    net = read_leela_net(str(path))
    return net, leela_to_torch(net)


def test_import_preserves_training_step(imported):
    net, result = imported
    assert result.training_steps == net.training_params.training_steps
    assert result.training_steps > 0


def test_import_consumes_every_weight_once(imported):
    """Each parameter must be visited exactly once, and be fully written."""
    _, result = imported
    jax_model = JaxModel(config=result.config, rngs=nnx.Rngs(params=0))
    names = [pair.name for pair in iter_param_pairs(result.model, jax_model)]
    assert len(names) == len(set(names)), "a parameter was visited twice"
    assert len(names) == result.weights_imported

    # Every parameter the torch model owns must be covered. Shared modules
    # (the policy embedding, the Smolgen generator) are one object reached
    # from several places, so compare by identity, not by name.
    covered = {
        id(pair.torch_param)
        for pair in iter_param_pairs(result.model, jax_model)
    }
    missing = [
        name
        for name, param in result.model.named_parameters()
        if id(param) not in covered
    ]
    assert not missing, f"parameters never imported: {missing}"


def test_import_matches_jax_predictions(imported):
    """The acceptance criterion: fixed-input parity with the JAX importer."""
    net, result = imported
    from lczero_training.convert.leela_to_jax import (
        LeelaImportOptions,
        leela_to_jax,
    )
    from proto import hlo_pb2

    jax_state = leela_to_jax(
        net,
        LeelaImportOptions(
            weights_dtype=hlo_pb2.XlaShapeProto.F32,
            compute_dtype=hlo_pb2.XlaShapeProto.F32,
        ),
    )
    jax_model = JaxModel(config=result.config, rngs=nnx.Rngs(params=0))
    nnx.update(jax_model, jax_state)

    rng = np.random.default_rng(11)
    inputs = rng.normal(size=(2, 112, 8, 8)).astype(np.float32)

    result.model.eval()
    with torch.no_grad():
        actual = result.model(torch.from_numpy(inputs))

    for index in range(inputs.shape[0]):
        expected = jax_model(jnp.asarray(inputs[index]))
        for name, logits in actual.policy.items():
            np.testing.assert_allclose(
                logits[index].numpy(),
                np.asarray(expected.policy[name]),
                rtol=2e-4,
                atol=2e-5,
            )
        for name, (wdl, _, _) in actual.value.items():
            np.testing.assert_allclose(
                wdl[index].numpy(),
                np.asarray(expected.value[name][0]),
                rtol=2e-4,
                atol=2e-5,
            )
        for name, movesleft in actual.movesleft.items():
            np.testing.assert_allclose(
                movesleft[index].numpy(),
                np.asarray(expected.movesleft[name]),
                rtol=2e-4,
                atol=2e-5,
            )


def test_config_matches_expected_is_enforced(imported):
    """A mismatched config must be refused, not silently accepted."""
    net, result = imported
    wrong = leela_config(net)
    wrong.encoder.num_blocks += 1
    with pytest.raises(ValueError, match="differs"):
        leela_to_torch(net, wrong)
    # ...unless explicitly overridden.
    leela_to_torch(net, wrong, ignore_config_mismatch=True)


def test_weight_map_roundtrips(imported):
    """torch -> jax -> torch must be lossless."""
    _, result = imported
    jax_model = JaxModel(config=result.config, rngs=nnx.Rngs(params=0))
    copy_torch_to_jax(result.model, jax_model)
    for pair in iter_param_pairs(result.model, jax_model):
        np.testing.assert_allclose(
            pair.jax_to_numpy(),
            pair.torch_param.detach().numpy(),
            rtol=1e-6,
            atol=1e-7,
            err_msg=pair.name,
        )


def test_imported_model_runs_on_directml(imported, dml_device):
    _, result = imported
    rng = np.random.default_rng(12)
    inputs = torch.from_numpy(
        rng.normal(size=(2, 112, 8, 8)).astype(np.float32)
    )
    model = TorchModel(result.config)
    model.load_state_dict(result.model.state_dict())
    model.eval()
    with torch.no_grad():
        expected = model(inputs).policy["vanilla"].numpy()
        actual = model.to(dml_device)(inputs.to(dml_device))
        actual = actual.policy["vanilla"].cpu().numpy()
    np.testing.assert_allclose(actual, expected, rtol=1e-3, atol=1e-4)
