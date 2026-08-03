# LCZero KDA-Hybrid Training Speed Improvement Plan

**Scope:** review of the `kda-hybrid` branch of `ContradNamiseb/lczero-training` with a focus on increasing training throughput (positions/sec) without compromising model quality or checkpoint compatibility.

**Review date:** 2026-08-03  
**Branch reviewed:** `https://github.com/ContradNamiseb/lczero-training/tree/kda-hybrid`  
**Repository path:** `C:\Users\Contrad\lczero-training`

---

## 1. Executive Summary

Training throughput is determined by three independent bottlenecks:

1. **Data ingestion** — how fast raw `.gz` chunks are decompressed, sampled, shuffled, and fed into TensorFlow.
2. **GPU work** — forward/backward time of the transformer body and auxiliary heads.
3. **Host overhead / I/O** — checkpoint writing, TensorBoard summaries, test passes, and Python-side control flow.

The current pipeline already contains several good practices (multi-worker chunk parsing, `tf.data.AUTOTUNE`, optional mixed precision, micro-batching via `num_batch_splits`, profiler hooks, and benchmark scripts). The biggest remaining wins are:

- **Eliminate CPU-side preprocessing from the critical path** by moving parsing into `tf.data` and/or using `tf.data.Dataset.interleave`/`prefetch` more aggressively.
- **Increase device microbatch size** on the XPU path, since DirectML is artificially capped at 1.
- **Reduce host overhead** by disabling SavedModel exports and full-model histograms during normal runs (the launch scripts already do this by default).
- **Simplify the model** where quality-neutral: remove auxiliary heads with zero loss weights, drop biases where already configurable, and consider swapping the last MHA for another KDA layer once DirectML stability allows.
- **Profile before large changes** with the existing `tf.profiler` and `tests/bench_*.py` tools.

---

## 2. Data Pipeline Bottlenecks and Fixes

### 2.1 Current pipeline architecture

- `train.py` builds `ChunkParser` objects that spawn `workers` child processes.
- Each child opens a `.gz` chunk, reads records, applies `sample` downsampling and diff-focus rejection, and ships raw byte records through a `multiprocessing.Pipe`.
- The parent multiplexes records into a `ShuffleBuffer`, converts them to tuples, batches them, and yields byte strings.
- `tf.data.Dataset.from_generator` consumes this Python generator, then `.map(parse_function)` decodes the byte strings into tensors.

This design has three inefficiencies:

1. **GIL/serialization tax**: every record crosses a `Pipe` and is copied into the shuffle buffer as raw bytes. For a batch size of 2048 and shuffle size of 1,000,000, the parent process spends a lot of time moving bytes.
2. **Generator cannot be parallelized by `tf.data`**: `from_generator` runs on the host in a single thread, so TensorFlow cannot autotune worker counts or prefetch depth.
3. **Parsing happens on the main Python thread**: `parse_function` runs inside `tf.data` but is still Python-host bound for each batch.

### 2.2 Recommended data pipeline changes

#### A. Benchmark the data pipeline in isolation

Use TensorFlow's built-in profiler around just the dataset:

```python
for _ in range(100):
    next(train_iter)
```

If this is slower than the GPU step time, the pipeline is the bottleneck.

#### B. Move record decoding into `tf.data` via `tf.py_function` or `tf.numpy_function`

Instead of yielding pre-batched byte strings from Python, yield **single** serialized records and let `tf.data` handle batching and decoding:

```python
def parse_single_record(record_bytes):
    # record_bytes is a scalar string tensor containing one v7 record
    # use tf.io.decode_raw, tf.slice, tf.reshape, etc.
    ...
    return planes, probs, winner, q, plies_left, st_q, opp_idx, next_idx

dataset = tf.data.Dataset.from_generator(parser.parse, output_types=tf.string)
dataset = dataset.map(parse_single_record, num_parallel_calls=tf.data.AUTOTUNE)
dataset = dataset.shuffle(shuffle_size)
dataset = dataset.batch(split_batch_size)
dataset = dataset.prefetch(tf.data.AUTOTUNE)
```

This removes the custom `batch_gen` Python loop and allows `tf.data` to vectorize decoding.

#### C. Replace multiprocessing pipes with `tf.data.Dataset.interleave`

If chunk reading is I/O bound, use one `tf.data` dataset per chunk file and interleave them:

```python
def make_chunk_dataset(filename):
    return tf.data.Dataset.from_generator(
        lambda f: single_file_gen(f),  # decompress + yield raw records
        args=(filename,),
        output_types=tf.string,
        output_shapes=(),
    )

dataset = tf.data.Dataset.from_tensor_slices(chunk_filenames)
dataset = dataset.interleave(
    make_chunk_dataset,
    cycle_length=num_readers,
    block_length=1,
    num_parallel_calls=tf.data.AUTOTUNE,
)
```

This keeps decompression and parsing inside TensorFlow's C++ runtime and lets it manage prefetch buffers per file.

#### D. Pre-decompress chunks to a fast local disk

On Windows with DirectML, gzip decompression is single-threaded Python. Pre-extracting `.gz` chunks to an NVMe/SSD or a RAM disk can remove a large CPU bottleneck, especially because the current `single_file_gen` reads only 256 records at a time. Even keeping the `.gz` format but using a faster disk helps.

#### E. Tune `shuffle_size` and `num_workers`

Current configs use:

| Config | `train_workers` | `test_workers` | `shuffle_size` |
|--------|-----------------|----------------|----------------|
| `kda-hybrid.yaml` | 8 | 4 | 1,000,000 |
| `kda-hybrid-directml.yaml` | 6 | 1 | 1,000 |
| `kda-hybrid-xpu.yaml` | 6 | 1 | 1,000 |

Observations:

- A 1,000,000-record shuffle buffer costs ~8.4 GB of host RAM (8396 bytes per v7 record). This is fine on a dedicated training box but may be excessive on laptops/integrated GPUs.
- `test_workers: 1` is fine because the test set is small and rarely the bottleneck.
- For DirectML on an iGPU with limited host memory, 1,000 is probably too small to fully decorrelate positions. A value between 10,000 and 100,000 is a better trade-off.

Recommendation: add a config note that `shuffle_size` should be scaled with available host memory, not copied blindly from the dedicated-GPU config.

#### F. Cache the chunk name list more aggressively

`fast_get_chunks` writes `chunknames.pkl` but the code currently has `if False and chunkfiles_name in subdirs:` guarding the fast path. This disables the cache read. The cache write still happens, so the file is created but never reused.

Fix:

```python
if chunkfiles_name in subdirs:
    # validate cache is not stale, then load it
```

For 290k chunks, directory enumeration alone can take many seconds per run.

---

## 3. Model Architecture Throughput

### 3.1 KDA recurrence

The KDA mixer is implemented as a custom `tf.keras.layers.Layer` called `KDARecurrence`. It uses a chunkwise-parallel form with:

- `KDA_CHUNK_SIZE = 16` (measured fastest on DirectML)
- four board traversal directions gathered/scattered via `tf.gather`/`tf.argsort`
- a Neumann series to invert the chunk-local lower-triangular system
- explicit Python `for index in range(chunks):` loop over chunks

This layer is the single most expensive operation. The current chunk size was tuned for DirectML; it may not be optimal for XPU/CUDA.

Recommendations:

1. **Benchmark `KDA_CHUNK_SIZE` per backend** using `tests/bench_mixers.py`. Values of 8, 16, 32, and 64 should be tested on the actual training hardware. The current 16 is a DirectML-specific result.
2. **Avoid the Python loop over chunks** if possible. For 64 tokens and chunk size 16, there are only 4 iterations, but the loop still forces graph partitioning. Consider a fully vectorized implementation that processes all chunks with `tf.scan` or `tf.while_loop` with `parallel_iterations=4`. On XPU/CUDA this may not matter; profile first.
3. **Fuse gather/recurrence/scatter into one custom op** for production. For inference this is already done in SYCL; for training, a CUDA/XPU custom op would remove Python overhead entirely.

### 3.2 Mixer pattern

Current configs use `[kda, kda, kda, mha]` repeated. Every fourth layer is full MHA, which:

- has global receptive field (good for quality),
- is typically slower than KDA on long sequences,
- currently uses `use_smolgen: true`, but smolgen only affects MHA layers.

Because smolgen is only active in MHA layers, a 3:1 hybrid gets much less benefit from smolgen than an all-MHA net while still paying its parameter cost in the shared generator. For the small DirectML/XPU configs (128-wide, 8 heads, smolgen channels 8) this is cheap, but for the 512x16 target it is material.

Recommendations:

1. **Ablation**: train a `[kda, kda, kda, kda]` variant (no MHA) and measure policy accuracy vs. throughput. If quality holds, remove MHA entirely and gain ~15–25% speed.
2. **If MHA is kept**, consider disabling smolgen on the hybrid pattern, because the MHA layer is only 1/4 of the body. The saved parameters and FLOPs can be reinvested in width or extra KDA layers.
3. **Place MHA layers strategically**. The last layer being MHA is a good default, but placing them more frequently early in the network (where tokens are less processed) may be more efficient.

### 3.3 Head dimensions

Default KDA dims:

```yaml
kda_key_dim: 32
kda_value_dim: 32
kda_gate_rank: 32
```

The recurrence state has shape `[batch, heads, key_dim, value_dim]`. For 16 heads this is `[B, 16, 32, 32]` per layer. This is small and unlikely to be the bottleneck, but:

- Smaller `kda_value_dim` directly reduces memory traffic and matmul cost.
- `kda_gate_rank` is used in two low-rank MLPs; reducing it to 16 or 8 cuts parameters with likely negligible quality impact.

Recommendation: run a small grid on the benchmark configs:

```yaml
kda_key_dim: [16, 24, 32]
kda_value_dim: [16, 24, 32]
kda_gate_rank: [16, 24, 32]
```

### 3.4 Biases

The code supports:

```yaml
omit_qkv_biases: true
omit_other_biases: false
```

Omitting all biases (`omit_other_biases: true`) removes a small number of FLOPs and memory accesses. With LayerNorm and gated KDA, biases are often redundant. Test quality before enabling.

### 3.5 Auxiliary heads

The default config enables several auxiliary heads:

- `policy_optimistic_st`
- `policy_soft`
- `value_q`
- `value_st`
- `moves_left`

Each head adds forward and backward computation. Heads with zero loss weights (`policy_opponent`, `policy_next`) should simply be disabled to avoid wasted compute.

Recommendation: keep only heads with non-zero loss weights in production configs. For example, if `policy_opponent: 0.0` and `policy_next: 0.0`, also set `model.policy_opponent: false` and `model.policy_next: false`.

### 3.6 Precision

Current settings:

| Config | Precision |
|--------|-----------|
| `kda-hybrid.yaml` | `half` |
| `kda-hybrid-directml.yaml` | `single` |
| `kda-hybrid-xpu.yaml` | `bfloat16` |

- `bfloat16` is the best throughput option on modern hardware (XPU, Ampere/Ada/Hopper) because it keeps the float32 exponent range without loss scaling.
- `half` requires dynamic loss scaling and can be slower on some kernels due to range checks.
- `single` is the fallback but halves throughput on ALU-bound kernels.

Recommendations:

1. **XPU**: keep `bfloat16`. Increase microbatch until OOM.
2. **DirectML**: the docs note mixed precision is unreliable with KDA; keep `single` until the plugin is fixed, but revisit with newer DirectML/plugin versions.
3. **CUDA (if used)**: use `bfloat16` on Ampere+ or `half` on Turing.

---

## 4. Training Loop and Optimizer

### 4.1 Batch splits and effective batch size

`num_batch_splits` controls gradient accumulation: each "training step" runs `num_batch_splits` forward/backward passes and accumulates gradients before applying them.

Trade-offs:

- More splits → smaller microbatch → lower device utilization but larger effective batch size.
- Fewer splits → larger microbatch → higher throughput up to memory limits.

Current configs:

| Config | `batch_size` | `num_batch_splits` | Microbatch |
|--------|--------------|--------------------|------------|
| `kda-hybrid.yaml` | 2048 | 1 | 2048 |
| `kda-hybrid-directml.yaml` | 512 | 4 | 128 |
| `kda-hybrid-xpu.yaml` | 512 | 2 | 256 |

The DirectML microbatch of 128 is very small; the docs say the plugin crashes above 1. This is a major throughput limit on Windows.

Recommendations:

1. Use `tests/bench_microbatch.py` to find the microbatch that saturates the device.
2. On XPU, raise `batch_size` and/or lower `num_batch_splits` until OOM, then back off 10%.
3. Consider using `tf.distribute.MirroredStrategy` with multiple XPUs/GPUs. The code already supports `gpu: 'all'` and `gpu: '0,1'`. On a multi-GPU box this is the easiest 2× speedup.

### 4.2 Optimizer

Current default is `optimizer: nadam`.

- NAdam maintains more optimizer state than SGD (two moments vs. one momentum vector), increasing memory use.
- For very large models, optimizer state can be 2–3× the model size in memory.
- SGD with Nesterov momentum is supported and uses less memory, but may require different hyperparameters.

Recommendation:

1. If memory is the bottleneck preventing larger microbatches, try `sgd` with `new_optimizer: true`.
2. If using mixed precision, ensure the optimizer state is kept in float32 to avoid instability.

### 4.3 SWA frequency

```yaml
swa: true
swa_steps: 100
swa_max_n: 10
```

SWA updates run on the host every 100 steps and copy all model weights. For large models this is cheap, but it is not free. If SWA is only needed for final model quality, increasing `swa_steps` to 500 or 1000 reduces host overhead.

### 4.4 Gradient clipping

`max_grad_norm: 1.0` is applied. Clipping by global norm requires a full-model gradient reduction; this is already in the TF graph and is cheap, but verify it does not dominate on tiny microbatches.

---

## 5. Host Overhead and I/O

### 5.1 Checkpointing

The training loop writes three things at `checkpoint_steps`:

1. `tf.train.CheckpointManager` checkpoint (required).
2. `tf.saved_model.save` of the full model (optional, controlled by `disable_saved_model_checkpointing`).
3. Leela `.pb.gz` weights (optional, controlled by `disable_pb_checkpointing`).

The launch scripts already disable SavedModel exports by default. This is correct and should remain the default.

Recommendations:

1. Keep `disable_saved_model_checkpointing: true` for normal training.
2. Set `checkpoint_steps` as high as stability allows (e.g., 5,000–50,000 steps) to reduce write stalls. The current 500-step DirectML checkpoint is very frequent.
3. Write checkpoints to a local NVMe drive, not a network share or the WSL2 `/mnt/c` mount.

### 5.2 TensorBoard summaries

`detailed_summaries` triggers:

- Full-model weight histograms every `train_avg_report_steps`.
- `compute_update_ratio` snapshots before/after every report step.

The launch scripts disable this by default. Keep it off for speed runs.

### 5.3 Test passes

The test loop evaluates `num_test_positions` positions every `test_steps`. Default in scripts is 512 positions every 500 steps, which is cheap. However, the YAML configs use 200 positions every 100 steps for DirectML/XPU, which is more frequent.

Recommendation: align test cadence with the launch scripts (500 steps, 512 positions) unless you are actively debugging.

### 5.4 Progress bar

The `rich` progress bar updates after every sub-batch split. On tiny microbatches this can add host overhead. It is already conditional on `rich` being installed; if throughput is critical, consider making it optional via a config flag.

---

## 6. Hardware and Backend-Specific Guidance

### 6.1 DirectML path (Windows, iGPU)

- Stuck on TensorFlow 2.10, Python 3.10, and protobuf 3.19.
- KDA backprop is unstable for models with two or more KDA layers at the tested widths.
- Microbatch is forced to 1 split-equivalent in practice.

Actions:

1. Keep the 128x4 3k1m model as the validated local target.
2. Monitor for DirectML plugin updates that fix KDA backprop; once fixed, move to larger microbatches and deeper/wider models.
3. Pre-decompress chunks to reduce CPU load, since the iGPU training is likely host-bound.

### 6.2 XPU path (WSL2/Linux, Intel GPU)

- Uses TensorFlow 2.15 + Intel Extension for TensorFlow.
- `bfloat16` is supported and recommended.
- The main lever is microbatch size.

Actions:

1. Copy the dataset into the WSL2 ext4 filesystem (`~/training-data`) instead of reading across `/mnt/c`.
2. Run `ITEX_VERBOSE=1` to confirm kernels are running on XPU, not CPU fallback.
3. Benchmark `num_batch_splits` values of 1, 2, 4 with `tests/bench_microbatch.py`.

### 6.3 CUDA path (Linux, NVIDIA GPU)

The repo is written for DirectML/XPU but the core TF code is backend-agnostic. For CUDA:

1. Install `tensorflow[and-cuda]` (the main `requirements.txt` already pins 2.14).
2. Use `precision: half` or `bfloat16` on Ampere+.
3. Use `gpu: 'all'` for multi-GPU scaling.
4. Consider XLA (`jit_compile=True`) if the KDA layer can be lowered without graph breaks.

---

## 7. Concrete Benchmark and Profiling Checklist

Before making large config changes, gather these numbers:

1. **Data-only throughput**

   ```python
   import time
   it = iter(tfprocess.train_dataset)
   t0 = time.time()
   for _ in range(100):
       next(it)
   print(f"data: {100 * batch_size / (time.time() - t0):.0f} pos/s")
   ```

2. **Model-only throughput**

   ```bash
   python tf/tests/bench_microbatch.py tf/configs/kda-hybrid-xpu.yaml 64 128 256 512
   ```

3. **Mixer breakdown**

   ```bash
   python tf/tests/bench_mixers.py tf/configs/kda-hybrid-xpu.yaml 256
   ```

4. **End-to-end profile**

   Set in YAML:

   ```yaml
   profile_step_freq: 100
   profile_step_offset: 10
   profile_step_count: 3
   ```

   Then open `leelalogs/<name>-profile` in TensorBoard's profile view.

---

## 8. Prioritized Action Plan

### Quick wins (low risk, immediate)

| # | Action | Expected impact | Files |
|---|--------|-----------------|-------|
| 1 | Disable SavedModel exports and detailed summaries in normal runs. | Reduced host I/O. Already default in scripts. | `scripts/run-*.ps1`, `scripts/run-*.sh` |
| 2 | Fix `chunknames.pkl` cache read path in `fast_get_chunks`. | Faster startup, less directory scanning. | `tf/train.py` |
| 3 | Disable auxiliary heads with zero loss weight (`policy_opponent`, `policy_next`). | Removed wasted forward/backward passes. | `tf/configs/*.yaml` |
| 4 | Increase DirectML/XPU `test_steps` to 500 and `num_test_positions` to 512. | Less frequent test overhead. | `tf/configs/*.yaml`, launch scripts |
| 5 | Raise `checkpoint_steps` to at least 5,000 for long runs. | Fewer write stalls. | `tf/configs/*.yaml` |

### Medium wins (requires validation)

| # | Action | Expected impact | Files |
|---|--------|-----------------|-------|
| 6 | Benchmark and tune `KDA_CHUNK_SIZE` per backend. | 5–20% mixer speedup. | `tf/tfprocess.py` |
| 7 | Tune `shuffle_size` for available host memory. | Better CPU/GPU overlap. | `tf/configs/*.yaml` |
| 8 | Benchmark `num_batch_splits` / microbatch on XPU. | Up to 2× device utilization. | `tf/configs/*.yaml`, `tf/tests/bench_microbatch.py` |
| 9 | Test `omit_other_biases: true` and `kda_gate_rank: 16`. | Fewer parameters/FLOPs. | `tf/configs/*.yaml` |
| 10 | Move record decoding into `tf.data.map` with `num_parallel_calls`. | Better CPU parallelism, less Python overhead. | `tf/chunkparser.py`, `tf/chunkparsefunc.py`, `tf/train.py` |

### Large wins (architecture changes, higher risk)

| # | Action | Expected impact | Files |
|---|--------|-----------------|-------|
| 11 | Ablation: all-KDA body (`[kda, kda, kda, kda]`). | Potentially 15–25% speedup if quality holds. | `tf/configs/*.yaml`, quality run |
| 12 | If MHA kept, evaluate disabling smolgen on 3:1 hybrid. | Saves smolgen compute with small quality risk. | `tf/configs/*.yaml` |
| 13 | Implement a fused KDA CUDA/XPU custom op for training. | Removes Python-loop overhead; largest possible speedup for the mixer. | New op/kernel code |
| 14 | Use `tf.distribute.MirroredStrategy` with multiple GPUs. | Near-linear scaling on multi-GPU nodes. | `tf/tfprocess.py`, config `gpu: 'all'` |
| 15 | Replace Python `from_generator` pipeline with `tf.data` interleave and optional TFRecord cache. | Decouples data loading from Python GIL; often 2–5× data throughput improvement. | `tf/train.py`, `tf/chunkparser.py` |

---

## 9. Config Presets to Test

### 9.1 XPU throughput preset

```yaml
training:
  precision: bfloat16
  batch_size: 1024
  num_batch_splits: 1          # raise until OOM, then back off
  test_steps: 500
  num_test_positions: 512
  checkpoint_steps: 5000
  disable_saved_model_checkpointing: true
  detailed_summaries: false
  shuffle_size: 50_000
model:
  omit_other_biases: true
  policy_opponent: false
  policy_next: false
```

### 9.2 DirectML stability preset

```yaml
training:
  precision: single
  batch_size: 512
  num_batch_splits: 4          # keep microbatch at 128
  test_steps: 500
  num_test_positions: 512
  checkpoint_steps: 5000
  disable_saved_model_checkpointing: true
  detailed_summaries: false
  shuffle_size: 10_000
model:
  policy_opponent: false
  policy_next: false
```

### 9.3 All-KDA ablation preset

```yaml
model:
  encoder_mixer_pattern: [kda, kda, kda, kda]
  use_smolgen: false           # irrelevant for KDA layers
```

Run this only after confirming it does not crash on the target backend.

---

## 10. What Not to Change

To avoid breaking existing functionality or other documentation:

- Do not modify `docs/directml-training.md`, `docs/xpu-training.md`, or `docs/model-design.md`.
- Do not change the protobuf schema (`proto/net.proto`) or `tf/net.py` serialization logic unless adding a new exported field is required.
- Do not remove the `multiprocessing`-based `ChunkParser` until a `tf.data` replacement is proven stable.
- Do not change the default `input_type`, network magic, or min_version unless the engine-side loader is updated.

---

## 11. Summary

The repository is already well organized for both DirectML and XPU training. The fastest path to higher throughput is:

1. **Fix the chunk-name cache** and **reduce test/checkpoint frequency**.
2. **Disable unused auxiliary heads** and **detailed summaries/SavedModel exports**.
3. **Maximize the device microbatch** on XPU, and **profile the KDA layer** for chunk-size tuning.
4. **Modernize the data pipeline** by moving decoding into `tf.data` and eventually replacing `from_generator`.
5. **Run architectural ablations** (all-KDA, no smolgen, smaller gate rank) with `tests/bench_mixers.py` and short training runs to verify quality.

Always measure before and after each change with the built-in profiler and benchmark scripts. Training speed is a system property of the data loader, model, optimizer state, and backend together; no single knob will give a large, safe win.
