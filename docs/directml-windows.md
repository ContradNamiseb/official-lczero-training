# Training on native Windows with DirectML

Runs the whole pipeline — C++ data loader and PyTorch trainer — natively on
Windows against any DirectX 12 GPU, including Intel and AMD integrated
graphics. No WSL, no CUDA, no TensorFlow.

This is a parallel path to the JAX/CUDA pipeline in [README.md](README.md), not
a replacement. The two share the config format, the C++ loader sources, the
model definition and the `.pb.gz` network format, so a network trained here
loads in the JAX pipeline and vice versa.

The design notes and the defect log for the port are in
[directml_training_port.md](directml_training_port.md). Metric definitions are
in [metrics.md](metrics.md).

---

## What you need

| | |
|---|---|
| **Windows** | 10 or 11, x64 |
| **GPU** | anything with a DirectX 12 driver. Developed against Intel Iris Xe |
| **Visual Studio** | 2022 or newer, **Desktop development with C++** workload |
| **Python** | **3.12 exactly** — `torch-directml` publishes no 3.13 wheels |
| **uv** | [installation guide](https://docs.astral.sh/uv/#installation) |
| **RAM** | 16 GB comfortable. 12 GB works with a small batch; an iGPU shares system memory |

`protoc` is not needed separately — `grpc_tools` comes with the Python
dependencies.

> **Python 3.12 is not advisory.** `uv sync` elsewhere in this repo uses 3.13.
> The DirectML environment must be its own 3.12 venv, and the C++ extension has
> to be built against that same interpreter, since it is a pybind11 module.

---

## Setup

From the repo root, in **PowerShell**:

```powershell
git submodule update --init --recursive
uv python install 3.12
uv venv --python 3.12 .venv-directml
uv pip install --python .venv-directml -e ".[directml,dev]"
uv pip install --python .venv-directml meson ninja pybind11 tensorboardX textual
```

Generate the protobuf modules (they are gitignored and always built locally):

```powershell
.\.venv-directml\Scripts\python.exe -m grpc_tools.protoc --proto_path=. --proto_path=libs/lc0 --python_out=src/ --pyi_out=src/ proto/*.proto
```

Check the GPU is visible and that backward passes actually run:

```powershell
.\.venv-directml\Scripts\python.exe -m lczero_training.commands.directml_device
```

That prints the adapter and runs smoke tests for the operations the port is
known to be sensitive about. If it reports no adapter, update your GPU driver.

### Build the C++ loader

```powershell
.\scripts\build-windows.bat
```

The script finds Visual Studio itself, calls `vcvars64.bat`, configures meson
with `native-windows.ini`, builds, and copies
`_lczero_training.cp312-win_amd64.pyd` next to the Python package. zlib is
fetched automatically as a meson subproject (pinned to 1.3.1 with a verified
hash in `subprojects/zlib.wrap`).

* `.\scripts\build-windows.bat test` — also run the C++ test suite
* `.\scripts\build-windows.bat reconfigure` — wipe `build\windows` and start over

Verify:

```powershell
.\.venv-directml\Scripts\python.exe -c "from lczero_training.dataloader import make_dataloader; print('ok')"
```

---

## Configure

Start from [`kda_split.textproto`](kda_split.textproto), which is a complete
working config for a 128x4 KDA hybrid with a held-out test split. Edit at
minimum:

| field | meaning |
|---|---|
| `file_path_provider.directory` | where your `.tar` training data lives |
| `training.checkpoint.path` | where checkpoints go |
| `metrics.tensorboard_path` | where TensorBoard events go |
| `export.destination_filename` | `.pb.gz` output template, supports `{step}` and `{datetime}` |

Use forward slashes in paths. They work throughout on Windows and avoid
escaping problems in textproto strings.

Two settings deserve real attention:

**`chunk_pool_size` is a sampling window, not a buffer.** The pool draws from
the last N chunks it has seen, so anything smaller than your corpus silently
trains on a slice of it. Set it above your total chunk count: tars x 10,000 for
standard Leela data. At 346 tars that is 3,460,000. Getting this wrong does not
produce an error — it produces a run that quietly loops over a fraction of your
data.

**`gradient_accumulation_steps` buys effective batch size cheaply, but not
freely.** On an iGPU the batch ceiling is low (batch 48 already fails on 11.7 GB
shared) and a batch-32 gradient is mostly noise, so accumulation is the only way
to a usable batch. It sums N micro-batches into one optimizer step at
essentially the memory of a single one, and throughput is flat to slightly
better: 28.7 pos/sec at 1x versus 32.9 at 8x.

"Essentially" is doing work in that sentence. On a 11.7 GB machine with ~4.5 GB
free at rest, `8` ran 47 steps and then died with a host OOM; `4` is the shipped
default. **Raise it one notch at a time and give each setting a few thousand
steps before trusting it.** See [metrics.md](metrics.md#effective-batch).

### Adding a test split

Without held-out data you cannot tell learning from memorization. To fork an
existing single-output config into train/test branches:

```powershell
.\.venv-directml\Scripts\python.exe scripts\add_test_split.py --input docs\your_config.textproto --output docs\your_config_split.textproto --train-weight 95 --test-weight 5
```

---

## Run

### 1. Create a checkpoint

From an existing Leela network:

```powershell
.\.venv-directml\Scripts\python.exe -m lczero_training.commands.directml_init --config docs/kda_split.textproto --lczero-model path/to/net.pb.gz
```

Omit `--lczero-model` for random weights. The importer checks the network's
architecture against your `model` section and refuses on a mismatch; override
with `--ignore-config-mismatch` only if you know why they differ.

### 2. Train

The TUI is the normal way — it runs the trainer as a child process and renders
live metrics:

```powershell
.\.venv-directml\Scripts\python.exe -m lczero_training.commands.directml_tui --config docs/kda_split.textproto --logfile train.log -- --kda-chunk-size=8 --report-every=10 --target-step=1000000 --eval-every=5000 --eval-batches=50 "--output=C:/Users/you/networks/net-{step}.pb.gz"
```

Everything after the bare `--` is passed to the daemon. Use it rather than
`--daemon-flag`: argparse rejects option values that begin with a dash.

Press `q` to quit. `--io-dump FILE` records the raw daemon JSONL, which is the
tool to reach for if a panel renders no data.

Headless, without the TUI:

```powershell
.\.venv-directml\Scripts\python.exe -m lczero_training.commands.directml_train --config docs/kda_split.textproto --target-step 1000000
```

Useful flags on both:

| flag | effect |
|---|---|
| `--kda-chunk-size 8` | ~2.4x faster than the default 16 on DirectML. Mathematically identical |
| `--grad-accum N` | override `gradient_accumulation_steps` |
| `--eval-every N` | evaluate on the held-out split. Needs a split config |
| `--target-step N` | absolute step to stop at, checkpointing along the way |

The first batch takes a few minutes — the loader indexes every tar and fills
its shuffle pool before yielding anything. `First batch after Ns of data loader
startup` in the log marks the end of it, and that time is excluded from
ms/step.

### 3. Watch it

```powershell
.\.venv-directml\Scripts\python.exe -m tensorboard.main --logdir C:/Users/you/tensorboard
```

Two runs appear: `<name>-train` and `<name>-test`. **Read the test run.** The
train run reports a single micro-batch, and at batch 32 that is far too noisy
to show a real trend — see
[metrics.md](metrics.md#read--test-not-the-log) for what that looks like
in practice.

Tag names match the old TensorFlow pipeline where the metric exists in both, so
a DirectML run overlays directly on an old `leelalogs` run.

### 4. Export

Networks are exported automatically at every checkpoint when `--output` or
`export.destination_filename` is set. Manually:

```powershell
.\.venv-directml\Scripts\python.exe -m lczero_training.commands.directml_export --config docs/kda_split.textproto --output network.pb.gz
```

---

## Tests

```powershell
.\.venv-directml\Scripts\python.exe -m pytest src\lczero_training -q --ignore=src\lczero_training\_lczero_training.so
```

The `--ignore` is needed when a WSL-built `.so` symlink is present in the tree;
Windows cannot stat it and collection fails with `OSError: [WinError 1920]`.

---

## Troubleshooting

**`Not enough memory resources are available`** — a host allocation failed.
Two distinct causes, and they need different fixes.

*If it fails immediately*, the batch does not fit. Lower
`tensor_generator.batch_size` and raise `gradient_accumulation_steps` to
compensate; the effective batch is the product of the two.

*If it fails minutes into a run that started fine*, the working set does not
fit in physical memory. This is a capacity problem, not a leak — measured on a
128x4 model, batch 32, 346 tars, `chunk_pool_size: 3600000`:

| configuration | committed memory |
|---|---|
| trainer alone, synthetic batches | ramps for ~100 s, then plateaus at **3.80 GB** (+0.13 MB/s) |
| trainer + data loader | ramps for ~100 s, then plateaus at **5.46 GB** (+0.14 MB/s) |

Both plateau, at the same negligible rate. The loader accounts for the ~1.6 GB
difference; the trainer's own 3.8 GB is the dominant term.

The failure mode is what happens when 5.5 GB of working set meets ~4 GB of
free memory. Windows trims the resident set — RSS fell 4734 → 2502 MB across
one run while committed memory stayed flat — and once free physical memory
reaches a few hundred MB, a DirectML allocation fails, because GPU-shared
memory has to be resident and cannot be paged out. Free physical hit 238 MB
shortly before the failures here.

So the lever is free RAM, not run length or the accumulation setting. Close
other applications before a long run; a browser or editor holding 1-2 GB is
the difference between fitting and not. Two runs differing only in
`gradient_accumulation_steps` (8 and 4) failed at 380 s and 423 s, which is
the same wall in both cases.

**`aten::<op> ... falling back to CPU`** in the log — a real performance cliff,
not a warning to ignore. One fallback in the training step can dominate the
step time. Grep the log for `falling back`; the workarounds live in
`src/lczero_training/directml/layers.py`.

**Access violation (0xC0000005) in a backward pass** — `torch_directml` must be
imported before the first backward. `directml/device.py:ensure_initialized()`
handles it; if you write a standalone script, call it first.

**`OSError: [WinError 1920]`** — the WSL `.so` symlink again. Pass `--ignore` as
above.

**Log file looks like mojibake** — it is BOM-less UTF-8 and Notepad guesses
wrong. Open with VS Code, or `Get-Content -Encoding utf8`.

**Checkpoint digest mismatch after editing the config** — the digest covers the
`model` and `training.optimizer` sections. Changing either invalidates existing
checkpoints by design. `kda.chunk_size` and `gradient_accumulation_steps` are
deliberately excluded, so those are safe to change between runs.

---

## Known limitations

* `compute_dtype` must be `F32`. F16 is not implemented and the model raises
  rather than training something subtly wrong.
* Multi-GPU is not supported.
