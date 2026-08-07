"""Shared fixtures for the DirectML tests.

Importing this file also performs the eager ``torch_directml`` import that
the autograd engine requires -- see ``ensure_initialized`` in device.py.
pytest loads conftest before any test module, so the import happens before
any test can trigger the first backward pass.
"""

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
