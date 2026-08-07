# KDA recurrence optimization — results and handoff notes

Companion to [kda-directml-optimization-brief.md](kda-directml-optimization-brief.md).
That document asked how to make the chunkwise KDA recurrence use less memory;
this one records what was changed, what it achieved, and what it ruled out.
Written so an agent with no prior context can pick the work up from here.

**Repo / branch:** `lczero-training`, branch `feature/directml-native-windows`,
workspace root `official-training-branch`.
Python env: `.venv-directml/Scripts/python.exe` (Python 3.12).

**File changed:** `src/lczero_training/directml/kda.py`, function
`kda_recurrence`. Only this file was modified. Parity is enforced against the
JAX reference in `src/lczero_training/model/kda.py` and a sequential NumPy
reference in `src/lczero_training/directml/test_kda.py`.

## What was done

Three exact algebraic reassociations — no approximation. All 30 parity tests
pass at `rtol=1e-4`, including gradient checks.

1. **Intra-chunk attention (the brief's hot spot A).** The per-channel decay
   factors: `exp(cum[i] − cum[j]) = exp(cum[i]) · exp(−cum[j])`. The old code
   built the two C×C matrices row-by-row (~48 kernel launches on ragged
   slices). Now it is **one** matmul — concat decayed-query and decayed-key
   along the token axis, `(2C, K) @ (K, C)` — plus two host-built causal
   masks. The `(…, C, C, K)` einsum intermediate was *not* needed; the decay
   factors out of the contraction. Safety: the −10 log-decay floor bounds
   `cum ≥ C · (−10) = −80` at the shipped chunk size, keeping `exp(cum)` and
   `exp(−cum)` inside normal float32 range. This constraint is documented
   in the code — a deeper floor or a much larger chunk would overflow
   `exp(−cum)` and force a return to computing pairwise differences.

   > **Two defects corrected after review.** Both needed a saturated forget
   > gate to appear, and the shipped suites never produced one: they draw
   > `log_decay` from `[−2, −0.001]`, so `cumulative` never approaches the
   > −10 floor.
   >
   > **1 — the gradient.** Written first as `key / exp(cumulative)` rather
   > than `key * exp(−cumulative)`. Equal in the forward pass, but division
   > backpropagates as `−grad · key / exp(cum)²`, and `exp(−160)`
   > underflows float32 to *exactly zero* — so the `log_decay` gradient
   > became `−inf` and would have poisoned the gate on the next optimizer
   > step. Live runs report 6–24% of gate elements on the floor; the
   > failure starts at 25% saturation. It went unnoticed partly because
   > `test_recurrence_gradients_are_finite` never set `requires_grad` on
   > `log_decay` — the one gradient that blows up was not checked at all.
   >
   > **2 — the factored form itself is chunk-size-limited.** The safety
   > argument above holds at chunk_size 8, which is what was benchmarked
   > (`--chunk-size 8`). The *default* is 16, which doubles the bound to
   > `cum ≥ −160`, and `exp(160)` overflows float32 (ceiling `exp(88.7)`).
   > `test_import_matches_jax_predictions` — real imported weights, default
   > config — produced an **all-NaN policy output**: 644 infs and 412 NaNs
   > in `full_attention`, and masking cannot repair it because the overflow
   > is in `exp(−cumulative)` *before* the matmul, so `0 * inf` poisons
   > entries the mask keeps. `kda_recurrence` now checks
   > `_factored_decay_is_safe()` and falls back to the pairwise-difference
   > rows when the split would overflow: fast path at chunk_size ≤ 8,
   > correct path above it.
   >
   > Both are covered by `test_recurrence_is_finite_with_a_saturated_gate`,
   > parametrized over chunk_size {4, 8, 16} × saturation {25%, 50%, 100%}
   > and verified to fail without each fix.

2. **Sequential scan (hot spot C).** `delta = inverse @ (beta_value −
   beta_key @ state)` distributes into `inverse @ beta_value − (inverse @
   beta_key) @ state`, so both `inverse @ …` products hoist out of the loop
   and batch over all chunks (key/value halves concatenated into one
   `(C, K+V)` contraction). The state update becomes affine:
   `state' = state ⊙ final_decay + state_bias − state_matrix @ state` —
   **one matmul per chunk instead of four**.

3. **Chunk outputs.** Batched into **one** matmul after the loop via stacked
   carried states, instead of two per chunk. The Neumann identity and both
   causal masks are cached per `(chunk_size, dtype, device)` instead of being
   re-allocated on every call.

## Measured results

Benchmark: 3-mixer KDA stack (the KDA half of the target hybrid tower in
`docs/example_kda_real_import.textproto`), batch 32, chunk_size 8,
forward+backward, Intel Iris Xe / DirectML.

| | step time | GPU MiB |
|---|---|---|
| baseline | 888.4 ms | 402 |
| optimized | 887.1 ms (flat over a 20-iteration run) | 276 (−31%) |

Re-measured after the two corrections below, baseline and optimized back to
back in one machine state (absolute times drift several percent with
background load, so only same-session pairs are comparable):

| | step time | GPU MiB |
|---|---|---|
| baseline | 993.2 / 971.6 ms | 402 |
| optimized + both fixes | **940.8 / 948.5 ms** | **282** (−30%) |

**Interpretation:** a pure memory win; step time is unchanged. That is the
informative result. It means the recurrence was *not* launch-bound the way
the brief assumed, and the flat 888 ms is dominated by something the
recurrence does not own. Combined with the memory dropping exactly when the
per-call tensor count dropped (~30 tensors/call down to ~8), the evidence
points at **allocator-side overhead** — the brief's §6 Q1 / §3 "15× gap"
(3.8 GB committed vs ~250 MB accountable), where DirectML appears to hold
arena memory proportional to allocation *count* rather than bytes.

## How to reproduce / validate

- Tests:
  `pytest src/lczero_training/directml/test_kda.py -q --ignore=src/lczero_training/_lczero_training.so`
  (the `.so` is a dangling untracked symlink from a Linux build step that
  breaks pytest collection — safe to delete or ignore; not part of the
  change).
- Benchmark:
  `python -m lczero_training.directml.benchmark_kda --skip-jax --batches 32 --chunk-size 8 --iterations 20 --warmup 4`.
- Terminal gotcha in this workspace: the persistent PowerShell terminal
  corrupts after some commands (returns `CommandNotFoundException` for
  everything). Recover by launching a fresh `mode=async` command, and prefer
  a small `.cmd` wrapper file for multi-step invocations over nested quoting.

## Open levers — do NOT re-derive, read first

The recurrence math is now essentially minimal for this formulation; more
reassociation will not remove meaningful work. The next wins are elsewhere:

- **§6 Q1 — the allocator question (highest value).** Probe whether DirectML
  arena growth tracks allocation *count* vs bytes: time a loop allocating
  many small tensors vs few large ones of equal total bytes. If count-driven,
  the fix is buffer reuse (`out=` variants) or restructuring toward
  fewer/larger tensors — *not* more kernel work.
- **§6 Q5 — fp16.** Keep `cumsum`/`exp`/`1e-12`-floored normalization in f32
  (the risky parts), cast only the matmuls to fp16. fp16 matmul and autocast
  are confirmed working at the op level on this device. Unmeasured.
- **Full-trainer number.** The 888 ms / 276 MiB above is only the 3 KDA
  mixers. The 3.8 GB trainer floor includes the embedding, heads, MHA block,
  and the C++ data loader (1.6 GB, out of scope of the brief). End-to-end
  improvement will be smaller than the 31% mixer-local drop.

Hard constraints that must not be violated: no `torch.compile`, no Triton, no
custom-kernel packages (`fla`, `mamba-ssm`, `causal-conv1d`) — DirectML
exposes a fixed operator set and tensors are opaque. Do not change the 8
traversal tables or the −10 decay floor; both are frozen byte-for-byte across
three implementations (JAX reference, this port, and the SYCL engine), and
the test suite asserts they stay identical.
