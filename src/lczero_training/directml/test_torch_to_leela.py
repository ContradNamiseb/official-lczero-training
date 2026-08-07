"""Round-trip tests for the PyTorch -> Leela exporter (Phase 9)."""

import pathlib

import numpy as np
import pytest

torch = pytest.importorskip("torch")
jax = pytest.importorskip("jax")

from google.protobuf import text_format

from lczero_training.directml.leela_to_torch import leela_to_torch
from lczero_training.directml.model import LczeroModel
from lczero_training.directml.torch_to_leela import (
    torch_to_leela,
    write_leela_file,
)
from proto.root_config_pb2 import RootConfig

_CONFIG_PATH = (
    pathlib.Path(__file__).resolve().parents[3]
    / "docs"
    / "example_kda_real_import.textproto"
)


def _model_config():
    config = RootConfig()
    text_format.Parse(_CONFIG_PATH.read_text(), config)
    return config.model


@pytest.fixture(scope="module")
def trained_model():
    """A model with non-default weights, so a bad export cannot pass."""
    config = _model_config()
    model = LczeroModel(config)
    generator = torch.Generator().manual_seed(5)
    with torch.no_grad():
        for parameter in model.parameters():
            parameter.add_(
                torch.randn(
                    parameter.shape, generator=generator, dtype=parameter.dtype
                )
                * 0.02
            )
    model.eval()
    return model, config


def test_export_preserves_step_and_config(trained_model):
    model, config = trained_model
    net = torch_to_leela(model, config, training_steps=97100)
    assert net.training_params.training_steps == 97100
    reimported = leela_to_torch(net)
    assert reimported.config == config
    assert reimported.training_steps == 97100


def test_export_roundtrip_matches_within_quantization(trained_model):
    """torch -> .pb.gz -> torch, comparing predictions.

    Leela stores each layer as uint16 against a per-layer min/max, so this is
    lossy by construction; the tolerance is quantization noise, not slack.
    """
    model, config = trained_model
    net = torch_to_leela(model, config, training_steps=97100)
    reimported = leela_to_torch(net).model
    reimported.eval()

    rng = np.random.default_rng(21)
    inputs = torch.from_numpy(
        rng.normal(size=(2, 112, 8, 8)).astype(np.float32)
    )
    with torch.no_grad():
        before = model(inputs)
        after = reimported(inputs)

    for name, logits in before.policy.items():
        np.testing.assert_allclose(
            after.policy[name].numpy(), logits.numpy(), rtol=2e-2, atol=2e-3
        )
    for name, (wdl, _, _) in before.value.items():
        np.testing.assert_allclose(
            after.value[name][0].numpy(), wdl.numpy(), rtol=2e-2, atol=2e-3
        )
    for name, movesleft in before.movesleft.items():
        np.testing.assert_allclose(
            after.movesleft[name].numpy(),
            movesleft.numpy(),
            rtol=2e-2,
            atol=2e-3,
        )


def test_export_roundtrip_weights_match(trained_model):
    """Every parameter survives the round trip within quantization error."""
    model, config = trained_model
    net = torch_to_leela(model, config, training_steps=97100)
    reimported = leela_to_torch(net).model

    original = dict(model.named_parameters())
    restored = dict(reimported.named_parameters())
    assert set(original) == set(restored)
    for name, parameter in original.items():
        want = parameter.detach().numpy()
        got = restored[name].detach().numpy()
        # uint16 over a per-layer range: the error scales with the layer's
        # own spread, so compare against that rather than a fixed epsilon.
        spread = float(np.ptp(want)) or 1.0
        assert np.max(np.abs(got - want)) < spread / 1000.0, name


def test_written_file_reimports(trained_model, tmp_path):
    model, config = trained_model
    net = torch_to_leela(model, config, training_steps=97100)
    destination = tmp_path / "exported.pb.gz"
    write_leela_file(destination, net)
    assert destination.exists() and destination.stat().st_size > 0

    from lczero_training.directml.leela_to_torch import load_leela_file

    result = load_leela_file(str(destination), config)
    assert result.training_steps == 97100
