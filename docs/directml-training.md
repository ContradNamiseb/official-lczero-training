# KDA training with DirectML on Windows

**Scope: Windows, native PowerShell.** For training under WSL2 or Linux on an
Intel GPU, use the XPU runbook in [xpu-training.md](xpu-training.md) instead.
The two backends are independent, share the same model code, and can coexist in
one checkout.

This runbook starts, verifies, runs, monitors, pauses, and resumes local KDA
training on the Intel Iris Xe. Commands are written for PowerShell and assume
the repository root is the current directory.

Microsoft has discontinued `tensorflow-directml-plugin`. Its supported stack is
Python 3.10, `tensorflow-cpu==2.10`, and protobuf 3.19. The repository therefore
uses `.venv-directml` separately from its normal TensorFlow 2.14 environment.

## Validated local model

The local config is `tf/configs/kda-hybrid-directml.yaml`:

- embedding width: 512
- encoder layers: 2
- mixer pattern: one KDA layer, then one MHA layer
- effective batch: 16
- device microbatch: 1 (`batch_size / num_batch_splits`)
- precision: FP32
- optimizer: Nadam
- initial run: 5,000 steps

This exact model completed a synthetic forward pass, backward pass, and Nadam
update on the Iris Xe with 15,186,252 parameters and 106 finite gradients.
Models containing two or more KDA layers terminated inside the native DirectML
plugin during backpropagation, even at width 128. The 512x12 architecture in
`tf/configs/kda-hybrid.yaml` remains the dedicated-hardware target.

Do not change the DirectML config to `precision: half`. The plugin did not
complete the mixed-float16 KDA test reliably. Keep the device microbatch at one.

## One-time prerequisites

Confirm Windows sees the Intel adapter:

```powershell
Get-CimInstance Win32_VideoController |
    Select-Object Name, DriverVersion, AdapterRAM
```

Install Python 3.10 for the current user:

```powershell
winget install --id Python.Python.3.10 --exact --scope user
py -3.10 --version
```

The expected Python version is 3.10.x. Python 3.11 and newer cannot load the
DirectML plugin.

## Scripted setup

Allow local scripts in the current PowerShell process only, then run setup:

```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass -Force
.\scripts\setup-directml.ps1
```

The setup script:

1. Creates `.venv-directml` with Python 3.10 if needed.
2. Installs the pinned packages from `tf/requirements-directml.txt`.
3. Regenerates protobuf bindings with the pinned compatible compiler.
4. Imports TensorFlow and both bindings, then requires at least one `GPU` device.

A successful final check resembles:

```text
TensorFlow 2.10.0
GPUs: [PhysicalDevice(name='/physical_device:GPU:0', device_type='GPU')]
```

The setup script is safe to rerun after pulling dependency or protobuf changes.

## Manual setup

Use these commands instead of the setup script when each operation needs to be
run separately:

```powershell
py -3.10 -m venv .venv-directml
.\.venv-directml\Scripts\python.exe -m pip install --upgrade pip
.\.venv-directml\Scripts\python.exe -m pip install -r .\tf\requirements-directml.txt
```

Regenerate the Python protobuf bindings:

```powershell
$ProtoRoot = (Resolve-Path .).Path
$TfRoot = (Resolve-Path .\tf).Path
$Python = (Resolve-Path .\.venv-directml\Scripts\python.exe).Path

& $Python -m grpc_tools.protoc `
    "--proto_path=$ProtoRoot" `
    "--python_out=$TfRoot" `
    "$ProtoRoot\proto\net.proto" `
    "$ProtoRoot\proto\chunk.proto"
```

Verify TensorFlow and DirectML:

```powershell
.\.venv-directml\Scripts\python.exe -c `
  "import tensorflow as tf; print(tf.__version__); print(tf.config.list_physical_devices('GPU'))"
```

Verify the generated protobuf modules from the training directory:

```powershell
Set-Location .\tf
..\.venv-directml\Scripts\python.exe -c `
  "from proto import net_pb2, chunk_pb2; print('protobuf imports passed')"
Set-Location ..
```

## Training data layout

The DirectML config uses these local dataset roots:

```yaml
dataset:
    input_train:
        - 'C:/Users/Contrad/Documents/training-data/*/'
    input_test:
        - 'C:/Users/Contrad/Documents/training-test-data/*/'
```

The extracted chunks are split into immediate child folders containing at most
500 `.gz` files. This avoids very large Windows directories while matching the
non-recursive scan used by `fast_chunk_loading`. Keep each YAML entry ending in
`*/`; the loader searches those immediate shard folders for chunks.

The active split contains only v6/classic chunks so the winner, root-Q, and
short-term value heads all receive populated targets:

- 290,057 available training chunks
- 6,000 available test chunks
- 295,977 requested chunks, selecting all training and 5,920 test chunks
- 98% training and 2% testing

Older chunks without short-term targets are preserved outside the active roots
under `Documents/training-data-no-st-target`. Non-classic input formats are
preserved under `Documents/training-data-incompatible` and
`Documents/training-test-data-incompatible`.

V6 records contain short-term Q but no short-term draw probability. Therefore
this config enables `value_st_scalar_loss`, supervising the ST head's scalar Q
instead of forcing a synthetic zero-draw WDL target. Its error and categorical
Q auxiliaries remain enabled.

Keep training and test chunks separate. The current config allows fewer chunks
than `num_chunks`, so this first dataset is acceptable. Confirm files are
visible with:

```powershell
Get-ChildItem C:\Users\Contrad\Documents\training-data -Filter *.gz -Recurse |
    Select-Object -First 5 FullName
Get-ChildItem C:\Users\Contrad\Documents\training-test-data -Filter *.gz -Recurse |
    Select-Object -First 5 FullName
```

## Scripted verification and run

Verify the environment, GPU, and config without reading training data:

```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass -Force
.\scripts\run-directml-training.ps1 -VerifyOnly
```

Start training after configuring the dataset paths:

```powershell
.\scripts\run-directml-training.ps1
```

The launcher applies monitoring-only speed optimizations in memory; it does
not edit the YAML or alter the model/checkpoint structure. By default it tests
every 500 steps over 512 positions, skips full-model TensorBoard histograms and
update-ratio snapshots, and skips redundant SavedModel exports. TensorFlow
checkpoints and normal/SWA `.pb.gz` files are still written at the YAML's
configured checkpoint interval. The no-RMSNorm architecture uses a new network
name and starts a fresh checkpoint family; it must not restore older RMSNorm
checkpoints.

To restore the YAML's original monitoring and export behavior, run:

```powershell
.\scripts\run-directml-training.ps1 `
    -TestSteps 100 `
    -NumTestPositions 1024 `
    -DetailedSummaries `
    -SavedModelCheckpoints
```

Refresh dependencies and protobuf bindings before starting:

```powershell
.\scripts\run-directml-training.ps1 -Setup
```

Run another config by repository-relative or absolute path:

```powershell
.\scripts\run-directml-training.ps1 `
    -Config 'tf\configs\kda-hybrid-directml.yaml'
```

The launcher verifies the DirectML GPU, checks batch splitting, rejects
placeholder or missing dataset roots, changes to `tf`, and starts `train.py`.

## Manual startup and run

Activation is optional. The most reliable method is to call the environment's
Python executable explicitly:

```powershell
Set-Location .\tf
..\.venv-directml\Scripts\python.exe .\train.py `
    --cfg .\configs\kda-hybrid-directml.yaml
```

Alternatively, activate the environment first:

```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass -Force
.\.venv-directml\Scripts\Activate.ps1
Set-Location .\tf
python .\train.py --cfg .\configs\kda-hybrid-directml.yaml
```

Run from `tf`, not the repository root. The generated protobuf modules use
`tf` as their import root, and output paths are relative to that directory.

The DirectML config writes normal and SWA `.pb.gz` files at each checkpoint.
To export the latest checkpoint without continuing training, run:

```powershell
Set-Location .\tf
..\.venv-directml\Scripts\python.exe .\model_to_net.py `
    --cfg .\configs\kda-hybrid-directml.yaml
Set-Location ..
```

KDA nets use the experimental network format value `135`. Stock lc0 builds do
not support this format. The lc0 schema, weight loader, and selected inference
backend must implement the same KDA fields and recurrence before evaluating the
exported net.

## Monitor training

The terminal displays a progress bar, estimated time remaining, periodic test
metrics, and checkpoint messages. In another PowerShell terminal, start
TensorBoard from the repository root:

```powershell
.\.venv-directml\Scripts\tensorboard.exe `
    --logdir .\tf\leelalogs `
    --port 6006
```

Open `http://localhost:6006`. Logs for this config are under:

```text
tf/leelalogs/kda-hybrid-128x4-3k1m-logit-no-rmsnorm-train
tf/leelalogs/kda-hybrid-128x4-3k1m-logit-no-rmsnorm-test
tf/leelalogs/kda-hybrid-128x4-3k1m-logit-no-rmsnorm-swa-test
```

Watch policy/value losses, gradient stability, step time, and test metrics. The
iGPU shares system memory, so close GPU-heavy applications during training.

## Checkpoints and resume

Checkpoints, SavedModels, and protobuf nets are written under:

```text
tf/networks/kda-hybrid-128x4-3k1m-logit-no-rmsnorm
```

The config saves every 500 steps and at the final step. List recent files with:

```powershell
Get-ChildItem .\tf\networks\kda-hybrid-128x4-3k1m-logit-no-rmsnorm |
    Sort-Object LastWriteTime -Descending |
    Select-Object -First 15 LastWriteTime, Length, Name
```

Starting the same config again automatically restores the latest checkpoint in
that directory. Keep the same model architecture and `name` when resuming. Use
a new `name` when changing layer counts, dimensions, heads, or mixer patterns.

## Pause, stop, and restart

To pause between training steps without terminating Python, create `tf/stop`
from another terminal:

```powershell
New-Item .\tf\stop -ItemType File
```

Resume by deleting it:

```powershell
Remove-Item .\tf\stop
```

To stop training, wait for a `Model saved in file:` message when practical,
then press Ctrl+C in the training terminal. Work after the latest completed
checkpoint is not retained. Rerun the same launch command to resume.

If the machine restarts or the process exits unexpectedly, use the same command
again. The trainer restores the latest complete checkpoint automatically.

## Common failures

### DirectML reports no GPU

Update the Intel graphics driver, restart Windows, then rerun setup. Confirm
the output contains `GPU:0`, not only `CPU:0`.

### Python version is unsupported

Check the environment directly:

```powershell
.\.venv-directml\Scripts\python.exe --version
```

If it is not Python 3.10, delete only `.venv-directml` and rerun the setup
script. Do not delete the normal `.venv` environment.

### Protobuf import fails

Rerun `scripts/setup-directml.ps1`. It pins protobuf 3.19.6 and regenerates both
bindings with the matching compiler. Always launch training from `tf`.

### Dataset path is rejected

Replace the YAML placeholders, retain the trailing `*/`, and verify the parent
directory exists. The scripted launcher validates roots before training.

### Native process exits during backpropagation

Keep the validated `[kda, mha]` two-layer pattern, FP32 precision, and
microbatch one. The discontinued plugin terminates with deeper KDA stacks on
this adapter without producing a Python exception.

### Out of memory or severe desktop slowdown

Confirm `batch_size: 16` and `num_batch_splits: 16`. Their quotient must remain
one. Close browsers, games, video tools, and other GPU workloads. Reducing the
effective batch requires reducing both values together, for example 8 and 8.

### Start over without deleting training data

Stop training, rename the config's `name`, or move its directory under
`tf/networks`. Do not point a changed architecture at an old checkpoint name.