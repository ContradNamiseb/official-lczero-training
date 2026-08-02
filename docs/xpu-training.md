# KDA training on Intel GPUs with Intel Extension for TensorFlow (XPU)

**Scope: WSL2 or native Linux.** For training on Windows directly, use the
DirectML runbook in [directml-training.md](directml-training.md) instead. The
two backends are independent, share the same model code, and can coexist in one
checkout.

Intel Extension for TensorFlow (ITEX) plugs an Intel GPU into stock TensorFlow
as a PluggableDevice of type `XPU`. It runs on a maintained TensorFlow 2.15
with oneDNN kernels, where the DirectML path is stuck on an end-of-life plugin
and TensorFlow 2.10, so it is the faster of the two backends on paper.

## Platform requirements

ITEX XPU wheels are **Linux only**. On Windows the supported route is WSL2 with
Ubuntu; the GPU driver stays in Windows and the compute runtime is installed
inside WSL2.

- Windows 10/11 with WSL2 and Ubuntu 24.04
- Intel Arc / Iris Xe Windows graphics driver 31.0.101.5333 or newer
- Inside Ubuntu: Intel GPU compute runtime, oneAPI DPC++ and oneMKL runtimes
- **Python 3.11**, TensorFlow 2.15.0, `intel-extension-for-tensorflow[xpu]`

Two caveats to read before investing time in this path:

- ITEX 2.15 is validated against **Ubuntu 22.04 and Python 3.9-3.11**. Ubuntu
  24.04 ships Python 3.12, which ITEX will not install against, so a 3.11
  interpreter has to be added explicitly (below). The wheels are manylinux2014
  and work on 24.04 once the interpreter matches.
- Intel's verified ITEX hardware is the Data Center GPU Max and Flex series,
  with Arc A-series listed as experimental. `Intel(R) Iris(R) Xe Graphics` is
  a Tiger Lake integrated part. Intel's compute runtime does list Tiger Lake as
  supported under WSL, so the driver stack is fine, but ITEX kernel coverage
  and performance on it are not something Intel validates. Run `--verify-only`
  before committing to a long training run.

## One-time setup

Install WSL2 with Ubuntu 24.04 from PowerShell:

```powershell
wsl --install -d Ubuntu-24.04
```

Install the Windows graphics driver from Intel, then open the Ubuntu shell and
add the Intel compute runtime. Ubuntu 24.04 carries these in its own archive:

```bash
sudo apt-get update
sudo apt-get install -y intel-opencl-icd libze1 libze-intel-gpu1 clinfo
```

If those packages are missing or too old for your driver, use Intel's preview
PPA for 24.04 instead:

```bash
sudo add-apt-repository -y ppa:kobuk-team/intel-graphics
sudo apt-get update
sudo apt-get install -y intel-opencl-icd libze1 libze-intel-gpu1 clinfo
```

Add the oneAPI runtimes that ITEX links against. This repository is
distribution-independent, so the `all main` suite is correct on 24.04:

```bash
wget -O- https://apt.repos.intel.com/intel-gpg-keys/GPG-PUB-KEY-INTEL-SW-PRODUCTS.PUB |
    sudo gpg --dearmor --output /usr/share/keyrings/oneapi-archive-keyring.gpg
echo "deb [signed-by=/usr/share/keyrings/oneapi-archive-keyring.gpg] https://apt.repos.intel.com/oneapi all main" |
    sudo tee /etc/apt/sources.list.d/oneAPI.list
sudo apt-get update
sudo apt-get install -y intel-oneapi-runtime-dpcpp-cpp intel-oneapi-runtime-mkl
```

Add Python 3.11, since 24.04's default 3.12 cannot install ITEX:

```bash
sudo add-apt-repository -y ppa:deadsnakes/ppa
sudo apt-get update
sudo apt-get install -y python3.11 python3.11-venv python3.11-dev
```

Confirm the GPU is visible to the compute runtime before going further:

```bash
clinfo | grep -i "Device Name"
```

That must name your Intel GPU. If it lists nothing, the runtime is not seeing
the device through WSL and no amount of Python setup will help.

## Python environment

From the repository root inside Ubuntu:

```bash
./scripts/setup-xpu.sh
```

The script creates `.venv-xpu` with Python 3.11, installs
`tf/requirements-xpu.txt`, regenerates the protobuf bindings from `proto/`, and
requires at least one `XPU` device. Set `PYTHON_BIN` to override the
interpreter, for example `PYTHON_BIN=python3.10 ./scripts/setup-xpu.sh`.
A successful final check resembles:

```text
TensorFlow 2.15.0
XPUs: [PhysicalDevice(name='/physical_device:XPU:0', device_type='XPU')]
Protobuf bindings: OK
```

`.venv-xpu` is separate from `.venv-directml`; the two stacks pin incompatible
TensorFlow and protobuf versions and must not share a virtual environment.

## Training data layout

The XPU config uses WSL2 paths into the existing Windows dataset:

```yaml
dataset:
    input_train:
        - '/mnt/c/Users/Contrad/Documents/training-data/*/'
    input_test:
        - '/mnt/c/Users/Contrad/Documents/training-test-data/*/'
```

Reading chunks across the `/mnt/c` 9p mount is markedly slower than reading
from the WSL2 ext4 filesystem. If chunk loading becomes the bottleneck, copy
the dataset into the Linux filesystem (for example `~/training-data`) and point
the config there.

The chunk layout requirements are unchanged from the DirectML runbook: split
chunks into immediate child folders of at most 500 `.gz` files, keep every YAML
entry ending in `*/`, and keep training and test chunks separate.

## Verify and run

```bash
./scripts/run-xpu-training.sh --verify-only
./scripts/run-xpu-training.sh
```

From Windows PowerShell without opening a shell first:

```powershell
wsl -d Ubuntu-24.04 --cd ~/lc0-training/stable-branch -- ./scripts/run-xpu-training.sh
```

Options mirror the DirectML launcher: `--config`, `--test-steps`,
`--num-test-positions`, `--detailed-summaries`, `--saved-model-checkpoints`,
`--setup`. By default it tests every 500 steps over 512 positions and skips
full histograms and SavedModel exports; the following restores the config's
own monitoring behaviour:

```bash
./scripts/run-xpu-training.sh \
    --test-steps 100 \
    --num-test-positions 1024 \
    --detailed-summaries \
    --saved-model-checkpoints
```

## Performance notes

- `tf/configs/kda-hybrid-xpu.yaml` sets `precision: bfloat16`. The trainer maps
  this to Keras' `mixed_bfloat16` policy and disables loss scaling, since
  bfloat16 keeps the float32 exponent range. The KDA recurrence already casts
  its state, decay and normalisation to float32 internally, so the numerics of
  the mixer are unchanged. Fall back to `precision: single` if a run diverges.
- Device microbatch is `batch_size / num_batch_splits`. The DirectML config was
  forced down to 1 by plugin crashes; XPU has no such limit, so raise the
  microbatch until the device runs out of memory. This is the largest single
  throughput lever.
- `ITEX_VERBOSE=1` prints which kernels ITEX claims, which is the quickest way
  to spot an op falling back to CPU.
- Do not combine `ITEX_AUTO_MIXED_PRECISION=1` with `precision: bfloat16`. The
  Keras policy already casts the graph, and enabling both makes the numerics
  hard to reason about.

## Relationship to other configs

`tf/configs/kda-hybrid-xpu.yaml` uses the same 128x4 architecture, 3:1 KDA/MHA
mixer pattern and head set as the DirectML config, so runs are comparable. It
uses its own network name, which starts a fresh checkpoint family; it must not
restore DirectML checkpoints trained under a different precision policy unless
the architecture matches exactly.
