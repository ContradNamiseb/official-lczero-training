$ErrorActionPreference = "Stop"

$repoRoot = Split-Path -Parent $PSScriptRoot
$environment = Join-Path $repoRoot ".venv-directml"
$python = Join-Path $environment "Scripts\python.exe"
$requirements = Join-Path $repoRoot "tf\requirements-directml.txt"
$protoRoot = $repoRoot
$tfRoot = Join-Path $repoRoot "tf"
$netProto = Join-Path $protoRoot "proto\net.proto"
$chunkProto = Join-Path $protoRoot "proto\chunk.proto"

if (-not (Test-Path $python)) {
    py -3.10 -m venv $environment
}

& $python -m pip install --upgrade pip
if ($LASTEXITCODE -ne 0) { throw "Failed to upgrade pip" }
& $python -m pip install -r $requirements
if ($LASTEXITCODE -ne 0) { throw "Failed to install DirectML requirements" }
& $python -m grpc_tools.protoc "--proto_path=$protoRoot" "--python_out=$tfRoot" $netProto $chunkProto
if ($LASTEXITCODE -ne 0) { throw "Failed to generate protobuf bindings" }
Push-Location $tfRoot
try {
    & $python -c "import tensorflow as tf; from proto import net_pb2, chunk_pb2; gpus = tf.config.list_physical_devices('GPU'); print('TensorFlow', tf.__version__); print('GPUs:', gpus); print('Protobuf bindings: OK'); assert gpus, 'DirectML did not expose a GPU device'"
    if ($LASTEXITCODE -ne 0) { throw "DirectML environment verification failed" }
} finally {
    Pop-Location
}