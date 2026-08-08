"""Shared fixtures for the DirectML tests.

Importing this file also performs the eager ``torch_directml`` import that
the autograd engine requires -- see ``ensure_initialized`` in device.py.
pytest loads conftest before any test module, so the import happens before
any test can trigger the first backward pass.
"""

import pathlib

import pytest

try:
    from lczero_training.directml import device as _device
except ImportError as error:  # torch / torch-directml not installed
    _device = None
    _IMPORT_ERROR = str(error)
else:
    _IMPORT_ERROR = None


@pytest.fixture(scope="session")
def dml_device():
    """A DirectML device, or a skip when the machine has no adapter."""
    if _device is None:
        pytest.skip(_IMPORT_ERROR)
    if _device.device_count() == 0:
        pytest.skip("no DirectML adapter present")
    return _device.get_device(0)


_REAL_CONFIG = (
    pathlib.Path(__file__).resolve().parents[3]
    / "docs"
    / "example_kda_real_import.textproto"
)


def make_tiny_config():
    """The shipped config, shrunk until a model built from it is instant.

    Derived from the real one rather than written by hand so that every
    section a test exercises -- losses, heads, the mixer pattern -- keeps the
    shape the trainer actually sees. Only the sizes change.
    """
    from google.protobuf import text_format

    from proto.root_config_pb2 import RootConfig

    config = RootConfig()
    text_format.Parse(_REAL_CONFIG.read_text(), config)
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


@pytest.fixture
def tiny_config():
    return make_tiny_config()
