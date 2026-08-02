[CmdletBinding()]
param(
    [string]$Config = "tf\configs\kda-hybrid-directml.yaml",
    [ValidateRange(1, [int]::MaxValue)]
    [int]$TestSteps = 500,
    [ValidateRange(1, [int]::MaxValue)]
    [int]$NumTestPositions = 512,
    [switch]$DetailedSummaries,
    [switch]$SavedModelCheckpoints,
    [switch]$Setup,
    [switch]$VerifyOnly
)

$ErrorActionPreference = "Stop"

$repoRoot = Split-Path -Parent $PSScriptRoot
$setupScript = Join-Path $PSScriptRoot "setup-directml.ps1"
$python = Join-Path $repoRoot ".venv-directml\Scripts\python.exe"
$tfRoot = Join-Path $repoRoot "tf"

if ($Setup -or -not (Test-Path $python)) {
    & $setupScript
}

if (-not (Test-Path $python)) {
    throw "DirectML environment not found. Run scripts\setup-directml.ps1 first."
}

$configPath = if ([System.IO.Path]::IsPathRooted($Config)) {
    $Config
} else {
    Join-Path $repoRoot $Config
}
$configPath = [System.IO.Path]::GetFullPath($configPath)

if (-not (Test-Path $configPath -PathType Leaf)) {
    throw "Training config not found: $configPath"
}

& $python -c "import tensorflow as tf; gpus=tf.config.list_physical_devices('GPU'); print('TensorFlow', tf.__version__); print('GPUs:', gpus); assert gpus, 'DirectML did not expose a GPU device'"
if ($LASTEXITCODE -ne 0) { throw "DirectML GPU verification failed" }

& $python -c "import sys,yaml; c=yaml.safe_load(open(sys.argv[1], encoding='utf-8')); t=c['training']; m=c['model']; assert t['batch_size'] % t['num_batch_splits'] == 0; print('Config:', c['name']); print('Mixers:', m['encoder_mixer_pattern']); print('Microbatch:', t['batch_size'] // t['num_batch_splits'])" $configPath
if ($LASTEXITCODE -ne 0) { throw "Training config validation failed" }

if ($VerifyOnly) {
    Write-Output "DirectML environment and config verification passed."
    exit 0
}

& $python -c "import glob,os,sys,yaml; c=yaml.safe_load(open(sys.argv[1], encoding='utf-8')); d=c['dataset']; paths=d.get('input_train', []) + d.get('input_test', []); placeholders=[p for p in paths if '/path/to/' in p.replace('\\','/')]; assert not placeholders, 'Replace placeholder dataset paths: ' + ', '.join(placeholders); fast=d.get('fast_chunk_loading', True); missing=[p for p in paths if not (os.path.isdir(p.replace('*/','')) if fast else glob.glob(p))]; assert not missing, 'Dataset paths not found: ' + ', '.join(missing); print('Dataset roots verified:', len(paths))" $configPath
if ($LASTEXITCODE -ne 0) { throw "Dataset validation failed" }

Push-Location $tfRoot
try {
    Write-Output "Starting DirectML training. Press Ctrl+C to stop."
    Write-Output "Runtime monitoring: test every $TestSteps steps over $NumTestPositions positions."
    $trainArgs = @(
        ".\train.py",
        "--cfg", $configPath,
        "--test-steps", $TestSteps,
        "--num-test-positions", $NumTestPositions
    )
    if (-not $DetailedSummaries) {
        $trainArgs += "--disable-detailed-summaries"
    }
    if (-not $SavedModelCheckpoints) {
        $trainArgs += "--disable-saved-model-checkpointing"
    }
    & $python @trainArgs
    if ($LASTEXITCODE -ne 0) {
        throw "Training exited with code $LASTEXITCODE"
    }
} finally {
    Pop-Location
}