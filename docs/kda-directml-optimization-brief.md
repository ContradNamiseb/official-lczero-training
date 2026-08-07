# Optimization brief: KDA recurrence on PyTorch/DirectML

A request for help making a Kimi Delta Attention mixer train faster and in
less memory on a DirectX 12 backend. Everything below is measured on the
target hardware, not estimated.

**Ask:** reduce the 3.80 GB memory floor and/or improve on 29.9 positions/sec.
The memory number is the binding constraint — the machine has ~4 GB free, so
runs die from allocation failure rather than finishing.

---

## 1. Hard constraints

These rule out most of the standard linear-attention toolbox. Please read
before proposing anything.

| | |
|---|---|
| Backend | **DirectML** (DirectX 12 compute), device type `privateuseone` |
| Stack | `torch==2.4.1+cpu`, `torch-directml==0.2.5.dev240914`, Python 3.12, Windows 11 |
| GPU | Intel Iris Xe (integrated, shares system RAM), 11.7 GB total system RAM |

**Not available:**

* **No Triton.** No CUDA, no ROCm, no HIP. Custom kernels are not an option
  in any language — DirectML exposes a fixed operator set through PyTorch's
  dispatcher and nothing else.
* **No `torch.compile`.** Verified: Dynamo raises
  `InternalTorchDynamoError: Cannot access storage of OpaqueTensorImpl`.
  DirectML tensors are opaque, so Inductor cannot trace them. No graph
  fusion, no kernel generation.
* **No `flash-linear-attention`, `fla`, `mamba-ssm`, `causal-conv1d`** or any
  package with compiled CUDA kernels.
* **No `torch.func` / `vmap`** over the recurrence (untested but expected to
  fail for the same opaque-storage reason).

**Silent performance cliff:** any aten op without a DirectML kernel falls
back to CPU, copying tensors across the bus each step. This is a warning, not
an error, and one fallback inside the training step dominates step time. Two
were found and removed this way (`aten::index_add.out` via the policy-map
gather, `aten::huber_loss` in the loss). **A proposal that introduces an
unsupported op is worse than no change.** Ops confirmed working and used
freely: `matmul`/`@`, elementwise arithmetic, `exp`, `log`, `softplus`,
`sigmoid`, `sqrt`, `clamp`, `sum`, `mean`, `reshape`, `permute`, `transpose`,
`cat`, `split`, `stack`, `index_select`, `cumsum` (positive dim only), `pad`.

**Known-broken ops** (each has a workaround in `directml/layers.py`):

| op | failure |
|---|---|
| `torch.flip` with a negative dim | access violation `0xC0000005`, kills the process |
| `torch.cumsum` backward | inherits the above — its backward flips along the recorded dim |
| `aten::index_add` | no kernel, CPU fallback (this is `index_select`'s generic backward) |
| `torch.eye` on device | broken fallback path |
| `F.layer_norm` backward | "tensor does not have a device" |
| `1.0 - tensor` (reflected scalar ops) | silently promotes to float64 |
| autograd generally | asserts unless `torch_directml` is imported before the first backward |

**Numerical parity is required.** This implementation must match a JAX/Flax
reference and a SYCL inference engine; the test suite asserts agreement to
`rtol=2e-4`. Algebraic rewrites are welcome but must be exactly equivalent or
provably within that tolerance. The 8 board-traversal permutation tables are
frozen byte-for-byte across three implementations and cannot change.

**Currently F32 only.** `compute_dtype: F16` is guarded off because it was
never implemented. However — see §6 — fp16 matmul and autocast both work at
the op level on this device, so this is a restriction of our code, not the
backend.

---

## 2. Model and shapes

A 128×4 chess transformer: 4 encoder blocks, the first 3 using KDA as the
token mixer, the 4th using standard MHA with Smolgen.

```
batch       B = 32      (48 already exhausts memory)
tokens      T = 64      (the 8x8 board; a fixed, tiny sequence length)
heads       H = 8       (one per board traversal direction)
key_dim     K = 32
value_dim   V = 32
chunk_size  C = 8       -> chunks = T / C = 8
d_model         128
gate_rank       32
```

Parameters: **6,012,892 (24.1 MB as f32)** — embedding 12.9 MB, value heads
6.4 MB, all 4 encoder blocks together only 3.3 MB.

Note the unusual regime: **sequence length is 64 and fixed**, batch is tiny,
and the model is small. This is not the long-sequence regime KDA is normally
tuned for. The recurrence runs 8 chunks of 8 tokens.

---

## 3. The anomaly worth explaining first

Measured steady-state committed memory for the trainer alone (synthetic
batches, no data loader, sampled past the ~100 s ramp to plateau):

**3,800 MB.**

The tensors do not come close to accounting for that:

| item | size |
|---|---|
| parameters | 24 MB |
| gradients | 24 MB |
| NAdamW state (mu + nu) | 48 MB |
| KDA activations saved for backward, **all 3 blocks** | 58 MB |
| everything else (embedding, heads, MHA block, losses) | ~100 MB |
| **accountable total** | **~250 MB** |

That is a **15x gap**. Per-KDA-block activation arithmetic, for the record:

```
chunked q/k/v/decay, each (B, chunks, H, C, K)   2.10 MB
cumulative + 3 decayed variants                  8.39 MB
attention matrices, 2 x (B, chunks, H, C, C)     1.05 MB
Neumann powers, ~3 x (B, chunks, H, C, C)        1.57 MB
state (B, H, K, V)                               1.05 MB
-> ~19.4 MB per block, 58.2 MB for three
```

**Question 1: where are the other 3.5 GB?** Candidates we cannot distinguish
from outside: DirectML allocator arena growth that never returns memory,
per-allocation padding/alignment, a separate host-side staging copy per
device tensor, or descriptor-heap overhead scaling with the number of live
allocations rather than bytes. The eager execution model means this graph
issues a *lot* of small allocations (see §4), so overhead that scales with
allocation *count* would fit the evidence.

If the answer is "allocation count", then the highest-value optimization is
not reducing FLOPs or bytes but **reducing the number of distinct tensors
created per step**, which is a very different target from normal kernel work.

---

## 4. The hot loop

`kda_recurrence` in `src/lczero_training/directml/kda.py`. It is the
chunkwise-parallel gated delta rule. Called 3× per forward pass.

```python
def kda_recurrence(query, key, value, log_decay, beta, chunk_size):
    # query/key/log_decay: (B, T, H, K)   value: (B, T, H, V)   beta: (B, T, H)
    query = query.float(); key = key.float()
    query = query / torch.sqrt(torch.clamp(
        torch.square(query).sum(dim=-1, keepdim=True), min=1e-12))
    key = key / torch.sqrt(torch.clamp(
        torch.square(key).sum(dim=-1, keepdim=True), min=1e-12))
    value = value.float()
    log_decay = torch.clamp(log_decay.float(), min=-10.0)
    beta = beta.float().unsqueeze(-1)

    batch, tokens, heads, key_dim = query.shape
    value_dim = value.shape[-1]
    # (padding elided: T=64 is divisible by C=8, so this never triggers)
    chunks = tokens // chunk_size

    def to_chunks(tensor, depth):
        tensor = tensor.reshape(batch, chunks, chunk_size, heads, depth)
        return tensor.permute(0, 1, 3, 2, 4)      # (B, chunks, H, C, depth)

    query = to_chunks(query, key_dim)
    key = to_chunks(key, key_dim)
    value = to_chunks(value, value_dim)
    log_decay = to_chunks(log_decay, key_dim)
    beta = to_chunks(beta, 1)

    cumulative = layers.cumsum(log_decay, 3)      # along the within-chunk axis
    final = cumulative[:, :, :, -1:, :]
    decayed_query = query * torch.exp(cumulative)
    decayed_key   = key   * torch.exp(cumulative)
    trailing_key  = key   * torch.exp(final - cumulative)

    # ---- HOT SPOT A: builds the two CxC intra-chunk matrices row by row.
    # chunk_size Python iterations, each issuing ~6 kernels on ragged slices.
    key_attention_rows = [query.new_zeros((batch, chunks, heads, 1, chunk_size))]
    query_attention_rows = []
    for row in range(chunk_size):
        cumulative_row = cumulative[:, :, :, row:row + 1, :]
        query_keys = key[:, :, :, :row + 1, :]
        query_decay = torch.exp(cumulative_row - cumulative[:, :, :, :row + 1, :])
        query_row = torch.sum(
            query[:, :, :, row:row + 1, :] * query_keys * query_decay, dim=-1)
        query_row = layers.pad_last(query_row, chunk_size - row - 1)
        query_attention_rows.append(query_row.unsqueeze(-2))

        if row:
            key_columns = key[:, :, :, :row, :]
            key_decay = torch.exp(cumulative_row - cumulative[:, :, :, :row, :])
            key_row = torch.sum(
                key[:, :, :, row:row + 1, :] * key_columns * key_decay, dim=-1)
            key_row = layers.pad_last(key_row, chunk_size - row)
            key_attention_rows.append(key_row.unsqueeze(-2))

    key_attention = torch.cat(key_attention_rows, dim=3)      # (B, chunks, H, C, C)
    query_attention = torch.cat(query_attention_rows, dim=3)

    # ---- HOT SPOT B: invert unit lower triangular (I + diag(beta) @ key_attention).
    # key_attention is strictly lower triangular hence nilpotent, so the
    # Neumann series terminates exactly. log2(C) = 3 doubling iterations.
    nilpotent = -beta * key_attention
    identity = layers.identity_matrix(chunk_size, dtype=query.dtype, device=query.device)
    inverse = identity + nilpotent
    power = nilpotent
    span = 2
    while span < chunk_size:
        power = power @ power
        inverse = inverse @ (identity + power)
        span *= 2

    beta_value = beta * value
    beta_key = beta * decayed_key
    final_decay = torch.exp(final).transpose(-1, -2)

    # ---- HOT SPOT C: the sequential inter-chunk scan. Genuinely sequential
    # in the state, 8 iterations, 4 matmuls each.
    state = query.new_zeros((batch, heads, key_dim, value_dim))
    outputs = []
    for index in range(chunks):
        delta = inverse[:, index] @ (beta_value[:, index] - beta_key[:, index] @ state)
        outputs.append(
            decayed_query[:, index] @ state + query_attention[:, index] @ delta)
        state = (state * final_decay[:, index]
                 + trailing_key[:, index].transpose(-1, -2) @ delta)

    output = torch.stack(outputs, dim=1) / math.sqrt(key_dim)
    output = output.permute(0, 1, 3, 2, 4)
    output = output.reshape(batch, chunks * chunk_size, heads, value_dim)
    return output[:tokens]
```

Rough kernel-launch budget per call: hot spot A issues ~8 × 6 = **48
launches on small ragged slices**; B issues ~6; C issues ~32. Times 3 blocks,
times forward and backward. In eager mode with no fusion, launch overhead and
allocation count plausibly dominate the actual arithmetic at these tiny
shapes.

### The 8-direction scan

Chess boards are not causal sequences, so the 8 heads are split across 8 board
traversal orders (rank/file/diagonal, forward and reverse). Rather than run
the recurrence 8 times, the `(token, head)` axis pair is flattened and a
single `index_select` permutes every head group into its own order at once:

```python
flat = tensor.reshape(batch, tokens * heads, depth)
permuted = layers.permute_along(flat, 1, order, inverse)   # custom autograd
```

`permute_along` exists because `index_select`'s generic backward is
`index_add`, which has no DirectML kernel. Since the index is a permutation,
the backward is another `index_select` by the inverse permutation. This is
already optimized and is probably not where the remaining win is.

---

## 5. Measured baseline and what is already ruled out

Trainer only, synthetic batches, batch 32, sampled to plateau:

| chunk_size | result |
|---|---|
| **8** | **3.80 GB, 1069 ms/step, 29.9 pos/sec** |
| 16 | out of memory |
| 32 | out of memory |
| 64 | out of memory |

Memory grows with the *square* of chunk size (the C×C matrices per head)
while the sequential-depth win is only linear, so 8 is both fastest and
smallest. Larger chunks are not a trade — they are strictly worse here.

Also ruled out by measurement:

* **Batch size** — 32 is the ceiling; 48 fails to allocate.
* **Gradient accumulation** — memory-neutral, as expected (each micro-batch's
  graph is freed by its own backward). Confirmed: 8 and 4 micro-batches gave
  the same failure point.
* **A memory leak** — both trainer-only and full-pipeline plateau and stay
  flat (+0.13 and +0.14 MB/s). The 3.80 GB is a steady-state working set, not
  growth.
* **The data loader** — accounts for a further 1.6 GB in the full pipeline,
  but is a separate C++ component and out of scope for this brief.

---

## 6. Specific questions

1. **The 15× memory gap (§3).** Is DirectML known to hold arena memory
   proportional to allocation *count* rather than bytes? If so, what
   reduces it — fewer/larger tensors, reusing preallocated buffers via
   `out=` variants, or something else? This is the highest-value question,
   because it decides whether to optimize bytes or tensor count.

2. **Hot spot A, vectorization.** Can the row-by-row construction of the two
   C×C matrices be replaced by a single masked formulation? The intent is:
   `A[i,j] = sum_d q[i,d] * k[j,d] * exp(cum[i,d] - cum[j,d])` for `j <= i`.
   Because the decay is *per-channel* (`cum` has a `key_dim` axis), this is
   not a plain `q @ k.T` — the decay does not factor out of the contraction.
   Writing it as `einsum('...id,...jd,...ijd->...ij')` materializes a
   `(B, chunks, H, C, C, K)` intermediate = 8.4 MB per tensor per block,
   trading 48 small kernels for one large one. **Is that trade favourable on
   a tile-based integrated GPU, or is the ragged-slice loop actually the
   lesser evil?** We have not measured it and would rather ask first.

3. **Hot spot B, the triangular inverse.** Is the Neumann doubling series
   the right approach at C=8, or is explicit forward substitution (8 rank-1
   updates) cheaper given that matmul on (8,8) matrices is likely
   launch-bound rather than FLOP-bound?

4. **Hot spot C, the sequential scan.** 8 sequential iterations with a
   carried `(B, H, K, V)` state. Is there a formulation with an associative
   combine that would let this run as a log-depth scan, given the state
   update is `state * decay + outer_product`? The decay is diagonal, which
   should make the transition matrices commute usefully.

5. **fp16.** We verified `torch.float16` matmul works on this device and
   `torch.amp.autocast_mode.is_autocast_available('privateuseone')` returns
   `True`. Our trainer currently forces f32. **Which parts of the gated delta
   rule are safe in fp16?** The concern is the exponentials: `log_decay` is
   clamped at -10 and `exp(cumulative)` spans a wide range within a chunk,
   and the normalization uses a `1e-12` floor on the squared norm. Would you
   keep `cumsum`/`exp`/normalization in f32 and cast only the matmuls?

6. **Op selection.** Given the fixed DirectML operator set, are any of the
   ops used above known to have poor kernels, with a cheaper equivalent?
   Particularly `cumsum` along a middle axis, ragged slicing along the
   token axis, and `torch.cat` of many small pieces.

---

## 7. Full source

The complete file is `src/lczero_training/directml/kda.py` (513 lines), with
DirectML-safe primitives in `src/lczero_training/directml/layers.py`. The
JAX/Flax reference this must match is `src/lczero_training/model/kda.py`.
`src/lczero_training/directml/test_kda.py` asserts parity between the two.
