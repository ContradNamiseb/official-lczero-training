# DirectML training: state of play and handoff

Everything a fresh session needs to continue without re-deriving it. Written
after a long debugging run; the measurements are real and the dead ends are
recorded so they are not re-explored.

**Branch:** `feature/directml-native-windows` on
`ContradNamiseb/official-lczero-training`. **Submodule:** `libs/lc0` at
`fd23ee1` on `ContradNamiseb/lc0`, branch `feature/kda-net-support`.

**Environment:** `.venv-directml` (Python 3.12 — `torch-directml` has no
3.13 wheels), `torch==2.4.1+cpu`, `torch-directml==0.2.5.dev240914`,
Intel Iris Xe, 11.7 GB shared system RAM.

---

## 1. Where training actually is

Net resumes from `C:/Users/Contrad/Documents/lc0-directml-checkpoint`,
currently around **step 191,000** (started this work at 102,708). Exports
land in `C:/Users/Contrad/Documents/lc0-directml-networks`.

Run to restart with:

```powershell
.\.venv-directml\Scripts\python.exe -m lczero_training.commands.directml_tui `
  --config docs/kda_split.textproto --logfile train.log -- `
  --kda-chunk-size=8 --report-every=10 --target-step=1000000 --gc-every=500 `
  --eval-every=5000 --eval-batches=50 `
  "--output=C:/Users/Contrad/Documents/lc0-directml-networks/kda-native-{step}.pb.gz"
```

**Is it learning?** Yes, but only the policy head, and slowly. Over 34,000
steps of held-out evaluation: policy loss −0.066 (t = −5.6), policy
accuracy +1.69pp (t = +4.0), thresholded@5 +2.49pp (t = +6.7). Value and
moves-left are statistically flat. The per-step log line is a single batch
of 32 and far too noisy to read — **always use the `-test` TensorBoard run**,
not the log. See `metrics.md`.

---

## 2. The one hard constraint

The trainer needs about **3.80 GB** committed on its own; with the data
loader the whole pipeline plateaus near **5.5 GB**. The machine has roughly
**4–5 GB free**. Everything difficult follows from that.

Measured, do not re-derive:

| lever | result |
|---|---|
| batch size | 32 is the ceiling. 48 fails to allocate. |
| `kda.chunk_size` | 8 is optimal on *both* speed and memory. 16/32/64 all OOM. |
| gradient accumulation | memory-neutral, as designed. 8 and 4 fail at the same wall. |
| KDA recurrence reassociation | **reverted** — see §4. |
| tensor arithmetic | ~250 MB accountable vs 3,800 MB committed. **93% is allocator overhead.** |

That last row is the real story: parameters are 24 MB, all KDA activations
across three blocks are 58 MB, optimizer state 48 MB. The model is not the
problem and never was.

---

## 3. Why it still OOMs, and the only remaining fix

Runs die after a few thousand steps with `Not enough memory resources are
available`. Best run so far: **36,397 steps over nine hours**. Most recent:
7,493 steps.

An independent review of DirectML memory management (see
`directml_memory_management_guide.pdf`, user-supplied) explains the
mechanism and matches what was measured here:

* `torch_directml` attaches through PyTorch's `PrivateUse1` backend. Its
  tensors are `OpaqueTensorImpl`, invisible to PyTorch's caching allocator.
* DX12 suballocates from coarse multi-megabyte heaps. A heap stays live
  until *every* suballocation in it is freed, so fragmentation strands
  memory that is nominally free.
* **The OS reclaims DX12 contexts only on process termination.**

Verified against the installed version — there is no escape hatch:

```
torch_directml.empty_cache       absent
torch_directml.memory_allocated  absent
torch_directml.synchronize       absent
torch._C._host_emptyCache        absent
```

Everything the guide recommends short of a subprocess is **already done**:

| guidance | status |
|---|---|
| `optimizer.zero_grad(set_to_none=True)` | done, always was |
| `del` locals each iteration | done — `predictions`, `loss`, `metrics`, `grad_norm`, `batch` |
| never accumulate un-detached loss | done — `losses.py` detaches every metric it returns |
| `torch.no_grad()` for evaluation | done — `evaluate()` is decorated |
| periodic `gc.collect()` | done — `--gc-every`, default 500 |

**What is left is the guide's only 100% technique: process isolation.**
Since DX12 memory returns to the OS solely on process exit, the robust
architecture is to stop fighting the allocator and restart around it:

1. **Evaluation in a subprocess.** `--eval-every` currently runs eval
   inside the training process, and the second data pipeline sits resident
   between evaluations. Spawning a worker that loads the exported `.pb.gz`,
   evaluates, writes TensorBoard and exits gives a guaranteed flush.
2. **Supervised restart of the trainer.** The daemon already checkpoints
   every `steps_per_network` (1,000) steps and resumes exactly. A wrapper
   that restarts it on OOM — or proactively every N thousand steps —
   converts an unavoidable leak into a scheduled, lossless event.

(2) is what makes unattended training work, and it depends on §4.1.

---

## 4. Open problems, ranked

### 4.1 The recovery checkpoint still fails — fix this first

On OOM the daemon tries to checkpoint what completed. It fails every time:

```
E Training stopped at step 191203: Not enough memory resources are available
E Could not save a recovery checkpoint at step 191203; progress since the
  last checkpoint is lost
```

`_emergency_save` in `directml/daemon.py` already clears the exception's
traceback frames (they pin the failed step's activations) and forces a
collection before saving, and it reports rather than failing silently. It
is still not enough. A save needs ~72 MB of host copies of the parameters
and both optimizer moments, on a device that has just run out.

Ideas not yet tried: stream the state dict tensor-by-tensor to disk instead
of building the whole dict; drop the optimizer state from the emergency
save (losing momentum is far better than losing the steps); free the model
from the device before copying.

There is also a secondary `RuntimeError: resource deadlock would occur`
during cleanup after some OOMs. Possibly the forced `gc.collect()` running
while loader threads hold locks — unconfirmed.

### 4.2 Where a NaN came from

One run died because a NaN reached the TUI. The model was healthy either
side of it, which does not fit a NaN loss (those poison the weights
permanently). Non-finite metrics are now logged with their step and name;
that evidence did not exist before. Watch for `Non-finite metric from the
daemon at step N`.

### 4.3 Three untrained heads

`optimistic_st`, `q` and `st` have no loss entries, so they get no gradient
and export at their imported values. **Deliberate.** lc0 defaults to
`policy_head=vanilla` everywhere, so they are loaded and ignored. Enabling
them costs ~10% throughput and, with the reference weights, would move 76%
of the gradient budget onto heads the engine does not use. Leave them.

### 4.4 lc0 has never loaded an exported net

The single unverified acceptance criterion. Export round-trips through our
own importer, but no lc0 binary has read one.

---

## 5. Traps that cost real time here

**PowerShell will corrupt source files.** `Get-Content | Set-Content
-Encoding utf8` reads UTF-8 as ANSI and rewrites it with a BOM, mangling
every non-ASCII character. Use the Edit tool, or
`[System.IO.File]::WriteAllText` with `UTF8Encoding($false)`.

**`_render` is Textual's internal API.** Naming a hook `_render` on a
Widget subclass breaks layout with `'Text' object has no attribute
'get_height'`. The panel hook here is `render_panel`.

**Textual dispatches handlers to every class in the MRO.** Instance
attributes are ignored and subclass overrides run *in addition to* the
original. Patch on the class (`monkeypatch.setattr(DirectMlTuiApp,
"on_load", ...)`) or tests silently spawn real daemons.

**`App.set_class` sets classes on the App, not the Screen.** Every
breakpoint rule reads `Screen.narrow`, so the whole responsive ladder
silently did nothing until this was found.

**Never assert on `export_screenshot()`.** It returns SVG; embedded
@font-face CSS survives tag-stripping and text is split across elements, so
multi-word probes can never match. Assert on widget state.

**`chunk_pool_size` is a sampling window, not a buffer.** Below the corpus
size the loader trims sources — oldest first, with no error. Keep it above
`tars × 10,000`. At 345 tars that is 3,600,000. A value of 500 once meant
training on 0.03% of the data with no indication anything was wrong.

**Benchmark the trainer, not the component.** A KDA reassociation measured
−30% memory on the mixer alone and made real training 50× worse (113 steps
→ 2). The microbenchmark counted allocations; the trainer cares about
retained bytes. Only same-session before/after pairs are comparable —
absolute times drift several percent with background load.

---

## 6. Reference

* `docs/directml-windows.md` — setup, build, run, troubleshooting.
* `docs/metrics.md` — every metric, direction, alarm thresholds.
* `docs/kda-directml-optimization-brief.md` — the constraints any KDA
  optimization must respect.
* `docs/kda-optimization-review-findings.md` — how the reassociation was
  reviewed and why it was reverted.
* `docs/kda_split.textproto` — the live config.

Tests: `.\.venv-directml\Scripts\python.exe -m pytest src\lczero_training -q`
(206 passed, 1 skipped). `src/lczero_training/conftest.py` handles the WSL
`.so` symlink, so no `--ignore` is needed any more.
