#!/usr/bin/env bash
# Create and verify the Intel Extension for TensorFlow (XPU) environment.
# Run inside native Ubuntu 22.04 or WSL2, not Windows.
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
venv="${repo_root}/.venv-xpu"
python_bin="${venv}/bin/python"
proto_root="${repo_root}"
# ITEX 2.15 supports Python 3.9-3.11 only; Ubuntu 24.04 defaults to 3.12.
base_python="${PYTHON_BIN:-python3.11}"

if ! command -v "${base_python}" >/dev/null 2>&1; then
    echo "${base_python} not found. Install it (Ubuntu 24.04: add the deadsnakes PPA) or set PYTHON_BIN." >&2
    exit 1
fi

if [ ! -x "${python_bin}" ]; then
    "${base_python}" -m venv "${venv}"
fi

"${python_bin}" -m pip install --upgrade pip
"${python_bin}" -m pip install -r "${repo_root}/tf/requirements-xpu.txt"

"${python_bin}" -m grpc_tools.protoc \
    "--proto_path=${proto_root}" \
    "--python_out=${repo_root}/tf" \
    "${proto_root}/proto/net.proto" \
    "${proto_root}/proto/chunk.proto"

cd "${repo_root}/tf"
"${python_bin}" - <<'PY'
import tensorflow as tf
from proto import net_pb2, chunk_pb2

devices = tf.config.list_physical_devices('XPU')
print('TensorFlow', tf.__version__)
print('XPUs:', devices)
print('Protobuf bindings: OK')
assert devices, (
    'Intel Extension for TensorFlow exposed no XPU device. Check the Intel '
    'GPU compute runtime and the oneAPI DPC++/oneMKL runtimes.')
PY
