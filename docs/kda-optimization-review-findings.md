# Review of the KDA recurrence optimization — findings and fixes

Written to be checked by someone else. Every claim below has a command you
can run to confirm or refute it, and a section at the end lists what I did
*not* verify. If you are giving a second opinion, the numbered defects are
measured facts; the judgement calls are flagged separately and are where
disagreement is most useful.

**Reviewed:** the changes described in
[kda-directml-recurrence-optimization.md](kda-directml-recurrence-optimization.md),
against `src/lczero_training/directml/kda.py` at commit `fa1acba`.

**Verdict:** the optimization is correct and worth keeping — three exact
algebraic reassociations, −30% mixer memory. It shipped with **two
numerical defects**, both of which would have produced NaN during training.
Both are now fixed; the memory win survives intact.

---

## 1. Summary

| | |
|---|---|
| Algebra | Correct. Verified by hand and against the sequential reference. |
| Memory | 402 → 282 MiB (−30%). Confirmed. |
| Step time | Unchanged within noise. Confirmed. |
| **Defect A** | Infinite gradient from 25% gate saturation. **Fixed.** |
| **Defect B** | All-NaN forward pass at `chunk_size` 16 (the default). **Fixed.** |

Both defects require a *saturated forget gate*. No existing test produces
one. That single gap is why 30 passing parity tests said nothing about
either.

---

## 2. The algebra is correct

Three reassociations, each checked by expanding both sides:

```
delta = inverse @ (βv − βk @ state)
      = inverse@βv − (inverse@βk) @ state                      ✓ distributes

state' = state⊙decay + tkᵀ @ delta
       = state⊙decay + (tkᵀ@inv@βv) − (tkᵀ@inv@βk) @ state     ✓ affine in state

output = dq @ state + qa @ delta
       = (dq − qa@inv@βk) @ state + qa@inv@βv                  ✓ same form
```

Shapes line up: `inverse_kb` is `(C, K+V)`, so slicing the last axis at
`key_dim` yields `(C, K)` and `(C, V)` as the code assumes. No objection
here.

---

## 3. Defect A — infinite gradient

### The change

```python
# as written
inverse_decayed_key = key / exp_cumulative
# fixed
inverse_decayed_key = key * torch.exp(-cumulative)
```

Identical in the forward pass. Not identical in the backward pass.

### Mechanism

`d(a/b)/db = −a/b²`. PyTorch evaluates that literally, so it computes
`exp(cumulative)²`. Within one chunk `cumulative` bottoms out at
`chunk_size × KDA_LOG_DECAY_FLOOR`, so the square reaches `exp(−160)` —
which **underflows float32 to exactly zero**, and the gradient becomes
`−inf`.

The multiply's derivative is `−key·exp(−cum)`: large, but finite.

### Evidence

```powershell
.\.venv-directml\Scripts\python.exe -c @"
import torch
cum = torch.tensor([-80.0], requires_grad=True)
k = torch.tensor([1.0], requires_grad=True)
e = torch.exp(cum)
print('exp(-80)**2 =', float(e*e))      # 0.0
y = k / e; y.backward()
print('grad via k/exp(cum):', float(cum.grad))    # -inf
cum2 = torch.tensor([-80.0], requires_grad=True)
k2 = torch.tensor([1.0], requires_grad=True)
(k2 * torch.exp(-cum2)).backward()
print('grad via k*exp(-cum):', float(cum2.grad))  # -5.54e+34, finite
"@
```

Full recurrence, original vs optimized, at increasing saturation:

| saturated | original | optimized as written |
|---|---|---|
| 0% | finite | finite |
| **25%** | finite | **−inf** |
| 50% | finite | −inf |
| 100% | finite | −inf |

### Why it matters here

Live training logs `KDA/block0 decay saturated %` at **6–24%**. The failure
threshold is 25%. This is not a corner case — it is the operating point.

---

## 4. Defect B — all-NaN forward at the default chunk size

### Mechanism

The whole optimization rests on splitting `exp(cum[i] − cum[j])` into
`exp(cum[i]) · exp(−cum[j])`. That is exact in real arithmetic. In float32
it is only valid while each *factor* is representable.

- `cumulative ≥ chunk_size × (−10)`
- so `exp(−cumulative)` reaches `exp(chunk_size × 10)`
- float32's ceiling is `exp(88.7)`

| chunk_size | needs | fits float32? |
|---|---|---|
| 8 (what was benchmarked) | exp(80) | yes |
| **16 (the default)** | **exp(160)** | **no — inf** |

The original handout says "the −10 log-decay floor bounds `cum ≥ C·(−10) =
−80` at the shipped chunk size". That is true at 8 and false at 16, and the
benchmark used `--chunk-size 8`.

**Masking cannot repair it.** The overflow happens in `exp(-cumulative)`,
*before* the matmul, so the reduction computes `0 × inf = NaN` inside
entries the causal mask keeps. Discarding the upper triangle afterwards is
too late.

### Evidence

`test_import_matches_jax_predictions` — real imported weights, default
config — returns an all-NaN policy tensor. Tracing the intermediates on the
tensors the real net feeds the mixer:

```
chunk_size      = 16
cumulative min  = -140.5
exp(-cumulative)= inf
full_attention  = 644 inf, 412 NaN
after masking   = 512 NaN
```

Reproduce:

```powershell
.\.venv-directml\Scripts\python.exe -m pytest `
  src\lczero_training\directml\test_leela_to_torch.py::test_import_matches_jax_predictions `
  -q --ignore=src\lczero_training\_lczero_training.so
```

Passes at `fa1acba`; fails with the un-guarded optimization.

### The fix

`kda_recurrence` now chooses its path:

```python
def _factored_decay_is_safe(chunk_size, dtype):
    largest_exponent = math.log(torch.finfo(dtype).max)
    return chunk_size * abs(KDA_LOG_DECAY_FLOOR) < largest_exponent - 8.0
```

- `chunk_size ≤ 8` → factored fast path (what your config runs)
- `chunk_size > 8` → original pairwise-difference rows, bounded at any size

The 8-unit headroom is because the matmul sums `key_dim` products, so the
individual factors must stay clear of the ceiling, not merely under it.
**This margin is a judgement call — see §7.**

---

## 5. Why 30 passing tests missed both

Two independent gaps, either of which alone would have hidden the defects:

1. **No test ever saturates the gate.** Every parity test draws
   `log_decay` from `[-2, -0.001]`, so `cumulative` reaches about −16 at
   chunk 8 and −32 at chunk 16. The safety argument is about −80 and −160.
   **The tests never enter the regime the claim is about.**
2. **`test_recurrence_gradients_are_finite` does not set `requires_grad`
   on `log_decay`.** The one gradient that goes infinite was never checked.

New coverage: `test_recurrence_is_finite_with_a_saturated_gate`,
parametrized over `chunk_size {4, 8, 16} × saturation {25%, 50%, 100%}`,
asserting finite forward *and* finite gradients on all five inputs.

I verified it bites, which a regression test must:

- reintroduce the division → all 9 cases fail
- disable the chunk-size guard → exactly the 3 `chunk_size=16` cases fail

---

## 6. Measurements

Baseline and optimized run back to back in one machine state. Absolute
times drift several percent with background load, so **only same-session
pairs are comparable** — I briefly mistook that drift for a regression
mid-review.

3 KDA mixers, batch 32, chunk_size 8, forward + backward, Iris Xe:

| | step time | GPU MiB |
|---|---|---|
| baseline (`fa1acba`) | 993.2 / 971.6 ms | 402 |
| optimized + both fixes | 940.8 / 948.5 ms | **282 (−30%)** |

```powershell
.\.venv-directml\Scripts\python.exe -m lczero_training.directml.benchmark_kda `
  --skip-jax --batches 32 --chunk-size 8 --iterations 20 --warmup 4
```

The memory win with unchanged step time is the interesting result: it means
the recurrence was never launch-bound, and supports the allocation-count
hypothesis in the optimization brief.

---

## 7. Judgement calls — the useful places to disagree

Everything above is measured. These are not:

1. **The 8-unit headroom in `_factored_decay_is_safe`.** Chosen so the
   `key_dim`-wide sum cannot push a factor over the ceiling. It is not
   derived from a tight bound. A rigorous bound would account for
   `key_dim` explicitly (`log(key_dim)` ≈ 3.5 at 32). 8 is comfortable but
   arbitrary.
2. **Falling back rather than restructuring.** An alternative is to shift
   the exponent by a per-chunk reference so both factors stay bounded at
   any chunk size. I did not pursue it because the shift cancels in the
   product — the discarded upper triangle still overflows — but I have not
   proved no stabilisation exists. If one does, the fast path could cover
   all chunk sizes.
3. **Keeping the slow path at all.** Since `chunk_size 8` is measurably
   optimal for both speed and memory (16/32/64 all OOM on this hardware),
   one could argue for rejecting unsafe chunk sizes outright instead of
   silently taking a slower path. I chose not to break the default config.
4. **The `-10` floor is assumed frozen.** Both defects scale with it. If
   the floor ever changes, `_factored_decay_is_safe` handles it
   automatically, but the JAX and SYCL implementations would need checking
   too.

---

## 8. What I did not verify

- **Whether the reassociation changes training dynamics.** Parity holds to
  `rtol=1e-4` per call, but I have not run a long training comparison. The
  rounding differs; over 100k steps that could diverge.
- **fp16.** Flagged as a next lever. Note it is worse than a precision
  question: fp16's maximum is 65504 = `exp(11)`, and this code needs
  `exp(80)` at chunk_size 8. Any fp16 cast on the decay path overflows on
  contact — the cumsum/exp/normalization must stay f32.
- **The allocator hypothesis itself.** Still unproven; the memory drop is
  consistent with it but does not establish it.
- **Whether 282 MiB is now the floor** for the mixer, or whether more is
  available.

---

## 9. Commands to check everything

```powershell
# All KDA tests, including the new saturation coverage
.\.venv-directml\Scripts\python.exe -m pytest src\lczero_training\directml\test_kda.py `
  -q --ignore=src\lczero_training\_lczero_training.so

# The import parity test that caught defect B
.\.venv-directml\Scripts\python.exe -m pytest `
  src\lczero_training\directml\test_leela_to_torch.py -q `
  --ignore=src\lczero_training\_lczero_training.so

# Full suite: 180 passed, 1 skipped
.\.venv-directml\Scripts\python.exe -m pytest src\lczero_training -q `
  --ignore=src\lczero_training\_lczero_training.so

# Benchmark
.\.venv-directml\Scripts\python.exe -m lczero_training.directml.benchmark_kda `
  --skip-jax --batches 32 --chunk-size 8 --iterations 20 --warmup 4
```

Commits: optimization plus both fixes in `3adc018`; the reviewed baseline
is `fa1acba`.
