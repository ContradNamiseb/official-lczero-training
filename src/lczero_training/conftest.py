"""Package-level pytest configuration.

The meson build symlinks the native module into this directory
(``_lczero_training.so`` -> ``build/release/...``). That link is created
from WSL, and on Windows it cannot even be stat'ed (WinError 1920), which
crashes pytest's collection of this package before any test runs. The
hook below runs before pytest's default ignore logic and drops any entry
the OS refuses to stat -- such a file is a build symlink, never a test
candidate.
"""

from __future__ import annotations

import pathlib

import pytest


def pytest_ignore_collect(
    collection_path: pathlib.Path, config: pytest.Config
) -> bool | None:
    try:
        collection_path.stat()
    except OSError:
        return True
    return None
