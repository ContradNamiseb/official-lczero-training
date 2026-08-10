# KDA 256x10

Everything needed to train this architecture on someone else's hardware.
The config is [kda_256x10_a100.textproto](kda_256x10_a100.textproto).

**Ignore the DirectML material in this repo.** `docs/directml-*.md` and
`docs/kda_split.textproto` are a Windows/integrated-GPU port constrained by a
5,965 MB shared-memory cap. Every small number in there is a workaround for
hardware you do not have. This config is for the JAX/CUDA trainer.

---

## What the architecture is

Kimi Delta Attention as a sequence mixer over the 64 squares, in place of
most of the attention blocks.

|               |                                                           |
| ------------- | --------------------------------------------------------- |
| tower         | 10 encoder blocks, `d_model` 256, `dff` 256               |
| mixer pattern | `[KDA, KDA, KDA, MHA]` tiled → **8 KDA, 2 MHA**           |
| heads         | 16                                                        |
| KDA           | `key_dim` 32, `value_dim` 32, `gate_rank` 64              |
| board scan    | 8 directions — rank/file/diagonal/anti-diagonal, each way |
| parameters    | **14,866,540** (verified by building it)                  |

Each KDA block runs a gated delta-rule recurrence along the board in 8
directions, 2 heads per direction, instead of all-pairs attention. The two
MHA blocks keep smolgen. Heads, values and losses match the reference layout,
so exported nets load in lc0 as ordinary networks.

## Setup

```bash
git clone -b feature/directml-native-windows https://github.com/ContradNamiseb/official-lczero-training.git
cd official-lczero-training
uv sync --extra cuda
```

The C++ data loader needs building — see [loader.md](loader.md). Everything
else is Python.

## Configure

Open the config and search for **`CHANGEME`**. Five paths:

| field                                               | what                              |
| --------------------------------------------------- | --------------------------------- |
| `data_loader.stage[0].file_path_provider.directory` | directory of `.tar` training data |
| `training.checkpoint.path`                          | where checkpoints go              |
| `metrics.tensorboard_path`                          | TensorBoard logdir                |
| `export.destination_filename`                       | exported `.pb.gz` networks        |
| `chunk_rescorer.syzygy_paths`                       | optional, commented out           |

**Then fix `chunk_pool_size` (two places).** It is a sampling *window*, not a
buffer: the pool draws from the last N chunks it has seen, so anything below
the corpus size silently trains on a slice of it, and the indexing loop
discards the sources it never reached — oldest first, with no error and no
log line. Set it above `number_of_tars × 10000`. It is currently 10,000,000,
which covers a 1,000-tar corpus. This one has cost real training time here.

## Run

Create a checkpoint from random weights:

```bash
uv run lc0-init --config docs/kda_256x10_a100.textproto
```

Train, with the dashboard:

```bash
uv run lc0-tui --config docs/kda_256x10_a100.textproto
```

or headless:

```bash
uv run lc0-train --config docs/kda_256x10_a100.textproto
```

Networks export automatically every `steps_per_network` (5,000) steps.

## Multi-GPU

Automatic. When `jax.device_count() > 1` the trainer builds a mesh over
`jax.devices()` and shards the batch across it — you should see
`Multi-GPU training enabled: 4 devices detected` in the log.

**`batch_size` must stay divisible by the device count.** It is 4096 = 1,024
per GPU, which also matches ~1B positions over ~250k steps.

If the GPUs are underutilised, suspect the data loader before the model. At
4096 positions/step the pipeline has to sustain roughly **21,000
positions/second**, and its thread counts are the lever — `chunk_rescorer`
and `chunk_source_loader` are the usual bottlenecks. Raise
`shuffling_frame_sampler.threads` only with care: `reservoir_size_per_thread`
is *per thread*, so it multiplies host RAM.

## Two constraints that will not survive being changed

**`kda.key_dim` is capped at 32.** The SYCL inference kernel keeps a
per-work-item state array of that width. A net trained wider will train fine
and then fail to run on that backend.

**`heads` must be a multiple of `len(directions)`** — 16 and 8 here. The
model asserts on it at construction.

## Knobs worth sweeping

- **`kda.chunk_size`** (currently 64) is a pure speed knob. The chunked
  recurrence is mathematically identical at any value; it trades a larger
  per-chunk score matrix against fewer sequential steps. Memory grows with
  its square. 32/64/128 is a cheap sweep, and the right answer is
  hardware-specific — the DirectML path wants 8, an A100 should want much
  more.
- **`compute_dtype`** is `BF16`. Drop to `F32` if anything looks numerically
  off; that costs speed, not correctness.
- **`lr_schedule`** is warmup to 1e-3 over 2,000 steps, then cosine to 1e-5
  by 250k. Tuned by convention rather than by sweeping, so it is the least
  evidence-backed part of this config.

## What to compare

Read the **`-test`** TensorBoard run, not `-train`: the train run reports a
single batch and is far too noisy to show a trend. The metrics that matter
are in [metrics.md](metrics.md).

The KDA-specific health metrics are `KDA/blockN decay saturated %` and
`KDA/blockN output gate mean`. Sustained decay saturation above ~50% means
the recurrence is forgetting everything and the block has collapsed toward a
pointwise MLP — worth reporting even if the loss curve looks fine, because
it changes what the comparison means.
