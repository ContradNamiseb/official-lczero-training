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
  --config docs/kda_split.textproto --supervise --logfile train.log -- `
  --kda-chunk-size=8 --report-every=10 --target-step=1000000 --gc-every=500 `
  --eval-every=5000 --eval-batches=50 `
  "--output=C:/Users/Contrad/Documents/lc0-directml-networks/kda-native-{step}.pb.gz"
```

`--supervise` is new and is what makes a long run survive; see §3.

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

1. **Evaluation in a subprocess.** Still open — see §4.2. `--eval-every`
   runs eval inside the training process, and the second data pipeline sits
   resident between evaluations. Spawning a worker that loads the exported
   `.pb.gz`, evaluates, writes TensorBoard and exits gives a guaranteed
   flush.
2. **Supervised restart of the trainer.** **Done.**
   `commands/directml_supervisor.py`, reached with
   `lc0-directml-tui --supervise` or run directly. The daemon already
   checkpointed every `steps_per_network` (1,000) steps and resumed
   exactly, so a process restart was already lossless; the supervisor turns
   that into the recovery mechanism. It relaunches after a crash and
   proactively every `--restart-every` steps (default 15,000, comfortably
   inside the 36,397-step best run) so the restart is scheduled rather than
   a failure.

   Details worth not rediscovering:

   * The daemon inherits the supervisor's stdin/stdout/stderr, so the JSONL
     protocol reaches the TUI untouched and there is **no relay code**. Both
     processes therefore have to keep stdout clean and log to stderr.
   * Each launch gets its own nearer `--target-step`, so it exits cleanly —
     checkpointing and exporting on the way out — instead of being killed
     at an arbitrary step. `partition_flags` is the one place that knows
     which flags belong to the supervisor rather than the daemon; the TUI
     imports it so the two cannot drift.
   * The child runs in a **Windows job object** with
     `JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE`. Terminating a process does not
     touch its children on Windows, and the TUI calls `terminate()` on
     quit — an orphaned trainer would keep holding gigabytes of DX12 memory
     while the replacement started. Verified by killing a supervisor with
     `TerminateProcess` and watching the child die with it.
   * A no-progress circuit breaker (`--max-stalls`, default 3) stops a
     misconfigured run relaunching forever.
   * `checkpoint.py` imports `torch` inside `save`/`load_latest` rather than
     at the top, so the supervisor can poll `latest_step` without paying a
     few hundred MB of RSS out of the trainer's headroom. Do not hoist it
     back.

---

## 4. Open problems, ranked

### 4.1 The recovery checkpoint — fixed, but unverified against a real OOM

`_emergency_save` in `directml/daemon.py` used to fail every time:

```
E Training stopped at step 191203: Not enough memory resources are available
E Could not save a recovery checkpoint at step 191203; progress since the
  last checkpoint is lost
```

Clearing the exception's traceback frames — which pin the failed step's
activations — was necessary but nowhere near sufficient. A save needs ~72 MB
of host copies of the parameters and both optimizer moments, all at once,
on a device that has just refused an allocation. There are now three stages,
cheapest first:

1. Clear the frames and collect, as before.
2. `training.release_to_host` — drop the gradients (same size as the
   parameters, no checkpoint wants them, and discarding them allocates
   nothing), then move parameters, buffers and optimizer moments to the
   host **one tensor at a time**. Each copy releases its device original,
   so pressure only ever falls; the transfer pays for itself after the
   first parameter. This is the change that should make it work. It sets
   `.data` in place rather than rebuilding the Parameters, because the
   optimizer's state is keyed on their identity.
3. Retry without the optimizer state if a save still fails. A momentum-less
   resume calls `optimizer.set_step(step)`, so the bias correction does not
   treat a network at step 191,000 as if it were on its first — without
   that the first updates would be full-rate sign steps.

The model is left on the host afterwards, so it must not be trained again;
every caller is on its way out of `run()`.

**Still unverified against a real out-of-memory crash.** The unit tests
cover the invariants and the fallback tier, but no CI machine has a
DirectML adapter to run out of. Watch the log on the next OOM: the success
line is `Saved recovery checkpoint at step N`, and the supervisor's next
launch should resume from that step rather than the last scheduled one.

The secondary `RuntimeError: resource deadlock would occur` during cleanup
after some OOMs is unchanged and still unexplained — possibly the forced
`gc.collect()` running while loader threads hold locks. It is caught and
reported by the `finally` in `run()`, so it costs nothing but a log line.

### 4.2 Evaluation still runs in the training process

§3.1, now the largest remaining piece of the memory work. `--eval-every`
holds a second data pipeline resident between evaluations, inside the
process that cannot give memory back.

### 4.3 Where a NaN came from

One run died because a NaN reached the TUI. The model was healthy either
side of it, which does not fit a NaN loss (those poison the weights
permanently). Non-finite metrics are now logged with their step and name;
that evidence did not exist before. Watch for `Non-finite metric from the
daemon at step N`.

### 4.4 Three untrained heads

`optimistic_st`, `q` and `st` have no loss entries, so they get no gradient
and export at their imported values. **Deliberate.** lc0 defaults to
`policy_head=vanilla` everywhere, so they are loaded and ignored. Enabling
them costs ~10% throughput and, with the reference weights, would move 76%
of the gradient budget onto heads the engine does not use. Leave them.

### 4.5 lc0 has never loaded an exported net

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
(222 passed, 1 skipped, ~3 min). `src/lczero_training/conftest.py` handles
the WSL `.so` symlink, so no `--ignore` is needed any more.
