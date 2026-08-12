# DirectML training: commands to copy and paste

Every command is a **single line** on purpose, so it pastes into `cmd` and
PowerShell alike — line continuations differ between the two (`^` vs a
backtick) and a wrapped command silently runs as two.

Run them from the repo root:
`C:\Users\Contrad\Documents\Code\repos\lc0-training\official-training-branch`

Paths below are already filled in for this machine. `python -m ...` is the way
in because the `lc0-directml-*` console scripts are **not** installed in
`.venv-directml`; `uv sync --extra directml` would install them and shorten
everything here.

---

## 1. Train

### With the dashboard (the usual way)

Live panels, and `--supervise` relaunches the trainer after a crash and
proactively every 15,000 steps. Press `q` to quit.

```powershell
.venv-directml\Scripts\python.exe -m lczero_training.commands.directml_tui --config docs/kda_split.textproto --supervise --logfile train.log -- --kda-chunk-size=8 --report-every=10 --target-step=1000000 --gc-every=500 --eval-every=5000 --eval-batches=50 "--output=C:/Users/Contrad/Documents/lc0-directml-networks/kda-native-{step}.pb.gz"
```

### Headless with restarts (unattended runs)

Same trainer, same restart logic, no dashboard.

**The redirection is not optional.** The daemon speaks the dashboard's JSONL
protocol on **stdout** and writes the readable log to **stderr**. With no TUI
attached, stdout is noise — park it in `run.out` and keep `run.log` for the
part you read.

```powershell
.venv-directml\Scripts\python.exe -m lczero_training.commands.directml_supervisor --config docs/kda_split.textproto --target-step 1000000 --restart-every 15000 -- --kda-chunk-size=8 --report-every=10 --gc-every=500 --eval-every=5000 --eval-batches=50 "--output=C:/Users/Contrad/Documents/lc0-directml-networks/kda-native-{step}.pb.gz" > run.out 2> run.log
```

Ctrl-C stops it, and the job object takes the trainer down with it — nothing
is left orphaned holding device memory.

### Plain (no supervisor, no protocol)

Logs straight to the console, nothing to redirect. **When this one hits an
out-of-memory error it stops** — you restart it yourself.

```powershell
.venv-directml\Scripts\python.exe -m lczero_training.commands.directml_train --config docs/kda_split.textproto --target-step 1000000 --kda-chunk-size=8 --report-every=10 --eval-every=5000 --eval-batches=50 "--output=C:/Users/Contrad/Documents/lc0-directml-networks/kda-native-{step}.pb.gz"
```

### Which one

| | supervisor | plain | dashboard |
|---|---|---|---|
| restarts after an OOM | **yes** | no | yes |
| restarts before hitting the wall | **yes** | no | yes |
| readable console | needs redirection | **yes, direct** | panels |
| extra memory | 23 MB | 0 | 50 + 23 MB |

Memory is a rounding error between these — measured, not guessed. Pick on
behaviour.

---

## 2. Watch it

Follow the log (PowerShell only — `cmd` has no `tail`):

```powershell
Get-Content -Wait -Tail 20 run.log
```

TensorBoard. **Read the `-test` run, not `-train`** — the train run reports a
single batch of 32 and is far too noisy to show a trend.

```powershell
.venv-directml\Scripts\tensorboard.exe --logdir C:\Users\Contrad\Documents\lc0-directml-tensorboard
```

`TensorFlow installation not found - running with reduced feature set` on
startup is expected. This pipeline writes only scalars, and the scalar plugin
is pure TensorBoard. Do not install TensorFlow to silence it.

Only the interesting lines out of a log:

```powershell
Select-String -Path run.log -Pattern "Launch |First batch|Training stopped|Memory at the failure|Only [0-9.]+ GB|Evaluated step|timed out|recovery checkpoint"
```

Current step, without opening anything:

```powershell
Get-ChildItem C:\Users\Contrad\Documents\lc0-directml-checkpoint\checkpoint-*.pt | Sort-Object Name | Select-Object -Last 3 Name, LastWriteTime
```

---

## 3. Is there room to train?

**There are two memory pools and only one of them is the constraint.** Getting
this wrong wastes a night: a run died reporting "Not enough memory resources
are available" with **6.16 GB of system RAM free** and the process at 1.70 GB
RSS. Both readings were true, because DirectML does not allocate from that
pool.

| pool | size here | who uses it |
|---|---|---|
| system RAM | 11.65 GB | everything; `psutil`, Task Manager |
| **GPU shared memory** | **~5,965 MB — half of RAM, capped by WDDM** | **DirectML. This is the one that runs out** |

The trainer needs ~5 GB of the second. Free RAM does not extend it, and
closing non-GPU applications does not help.

**The number that decides whether a run lives:**

```powershell
(Get-Counter "\GPU Adapter Memory(*)\Total Committed").CounterSamples | Measure-Object CookedValue -Sum | ForEach-Object { "GPU committed {0:N0} MB of ~5965 ({1:N0}%)" -f ($_.Sum/1MB), ($_.Sum/1MB/5965*100) }
```

Who is holding it, by process:

```powershell
(Get-Counter "\GPU Process Memory(*)\Total Committed").CounterSamples | Where-Object { $_.CookedValue -gt 50MB } | Sort-Object CookedValue -Descending | Select-Object -First 8 @{n='MB';e={[math]::Round($_.CookedValue/1MB)}}, InstanceName
```

Measured at a real failure: adapter at **5,865 MB of 5,965 — 98%** — while
6 GB of system RAM sat idle. The trainer's own share was 5,031 MB, and it
dropped to ~1,280 MB the moment the process exited, which is the only way
DirectML ever gives it back.

**The supervisor resets the GPU state before every launch, automatically.**
After a morning of failures, clearing the shader caches and restarting the
display driver produced a run that held 880 ms/step and *flat* GPU memory for
2,000 steps — so it is now part of the restart cycle rather than something to
remember. Controlled by `--reset-gpu-state`:

| value | what happens before each launch |
|---|---|
| `full` (default) | clear both shader caches **and** restart the display driver |
| `cache` | clear the caches only, no keystroke |
| `off` | nothing |

It runs at the one moment nothing of ours holds a device — the previous child
has exited, the next has not started — because a driver reset is system-wide:
the screen blanks for a moment and every GPU application on the machine has
its device reset. That is fine between launches and would not be fine during
one. A reset that fails is logged and the launch proceeds anyway.

For the paths with no supervisor (`directml_train`, a plain TUI run, or
tidying up after something died):

```powershell
.venv-directml\Scripts\python.exe scripts\reset_gpu_state.py
```

Same logic, plus a check that no trainer is running. `--dry-run` shows what it
would clear; `--no-driver-reset` skips the keystroke.

Which half actually does the work is not established: the caches are ~16 MB
of compiled shaders on disk, not GPU allocations, so their size does not
explain a memory symptom. The driver reset is the likelier half, because it
reclaims allocations left behind by processes that died badly. A clean exit
does return its own memory (the adapter drops from ~5,100 MB to ~1,200 MB),
but a process killed mid-flight may not. Treat this as a remedy after
crashes, not a ritual before every run.

System RAM still matters for speed (thrashing slows steps: ~930 ms/step with
3-4 GB free, ~4,400 ms with under 1.4 GB), just not for these crashes:

```powershell
$os = Get-CimInstance Win32_OperatingSystem; "available {0:N2} GB of {1:N2}" -f ($os.FreePhysicalMemory/1MB), ($os.TotalVisibleMemorySize/1MB)
```

The log now records both on every step line — `mem[...]` for system RAM and
`[gpu adapter N MB of ~5965 (P%)]` for the pool that matters — warns above
92% of the cap, prints both as the first line of any out-of-memory handler,
and charts `gpu_committed_mb` and `mem_available_gb` in TensorBoard.

---

## 4. Startup time

Every launch pays a data-loader startup before the first step, and the
supervisor pays it again on each restart. One config value dominates it:
`chunk_source_loader.threads`, which indexes the 345 tars across an 89 GB
corpus. Measured to the first batch, loader only, no trainer:

| `chunk_source_loader.threads` | time to first batch |
|---|---|
| 1 (the proto default) | 460 s |
| **4 (now set)** | **117 s** |
| 6 | 97 s |

Timed in the order 4, 1, 6, so 4 had the coldest file cache and 6 the
warmest — the 4-vs-1 gain is if anything understated, and 6's edge over 4 is
flattered. 4 matches the machine's physical core count. The threads idle on
the input queue once the scan is done, so this costs nothing while training.

**Raise it only there.** The other stages feed a trainer that is GPU-bound at
~880 ms/step, so more threads buy no throughput, and
`shuffling_frame_sampler.reservoir_size_per_thread` means extra sampler
threads cost host memory outright. `data_loader` is not part of the
checkpoint digest, so changing any of this is safe mid-run.

Time it yourself against a modified config:

```powershell
.venv-directml\Scripts\python.exe -c "import time; from google.protobuf import text_format; from lczero_training.dataloader import make_dataloader; from proto.root_config_pb2 import RootConfig; c=RootConfig(); text_format.Parse(open('docs/kda_split.textproto').read(), c); t=time.perf_counter(); l=make_dataloader(c.data_loader); l.get_next('train'); print(f'first batch after {time.perf_counter()-t:.1f}s'); l.stop()"
```

---

## 5. Evaluate by hand

Evaluation runs in a child process that exits, which is what returns its
memory. It runs on the **CPU** by default and takes ~15 s for 50 batches; on
the GPU, beside a running trainer, it once failed to finish in 900 s.

The trainer leaves each request on disk, in a directory named after a hash of
the config path — so find the newest rather than typing a hash. This prints
the path and remembers it as `$work` for the commands below:

```powershell
$work = (Get-ChildItem $env:TEMP\lc0-directml-eval-* -Directory | Sort-Object LastWriteTime | Select-Object -Last 1).FullName; $work
```

If an evaluation produced no numbers, the worker's own log says which stage it
reached — this is the first place to look, because a worker killed on timeout
may never reach the trainer's log:

```powershell
Get-Content $work\worker.log
```

Re-run that evaluation by hand. Takes ~15 s and rewrites the TensorBoard
`-test` point, so a skipped evaluation can be recovered:

```powershell
.venv-directml\Scripts\python.exe -m lczero_training.commands.directml_eval_worker --work-dir $work --log-file $work\worker.log --step 195000 --device cpu --kda-chunk-size 8
```

Set `--step` to the step the weights actually belong to, or the point lands on
the chart at the wrong place. The trainer logs it as
`Evaluating step N in a subprocess`.

Each request holds ~50 MB of batches and one directory is kept per config
path, so a couple of stale ones can accumulate. Safe to delete when nothing is
training:

```powershell
Remove-Item $env:TEMP\lc0-directml-eval-* -Recurse -Force
```

---

## 6. Checkpoints and networks

Export a `.pb.gz` from the latest checkpoint by hand:

```powershell
.venv-directml\Scripts\python.exe -m lczero_training.commands.directml_export --config docs/kda_split.textproto --output C:\Users\Contrad\Documents\lc0-directml-networks\manual-export.pb.gz
```

Start a checkpoint from an existing Leela network (only needed once):

```powershell
.venv-directml\Scripts\python.exe -m lczero_training.commands.directml_init --config docs/kda_split.textproto --lczero-model path\to\net.pb.gz
```

---

## 7. Tests

```powershell
.venv-directml\Scripts\python.exe -m pytest src\lczero_training -q
```

One file while working on it:

```powershell
.venv-directml\Scripts\python.exe -m pytest src\lczero_training\directml\test_subprocess_eval.py -q
```

---

## 8. Flags worth knowing

Everything after the bare `--` goes to the trainer. Supervisor flags go
**before** it — `--target-step` is read by the supervisor, which rewrites a
nearer one for each launch, so a copy after the `--` would quietly disable the
proactive restart. It refuses to start rather than let that happen.

| supervisor flag | what it does |
|---|---|
| `--target-step N` | the finish line. Required |
| `--restart-every N` | steps per launch before a planned restart (default 15000). `0` restarts only after a crash |
| `--max-launches N` | stop after N launches even while making progress. `0` = only `--target-step` ends it |
| `--max-stalls N` | give up after N launches that advance nothing (default 3) |
| `--reset-gpu-state M` | `full` (default) clears the shader caches and restarts the display driver before each launch; `cache` skips the keystroke; `off` does nothing. See §3 |

| trainer flag | what it does |
|---|---|
| `--kda-chunk-size 8` | ~2.4x faster than the default 16 and the lowest memory. Mathematically identical |
| `--grad-accum N` | effective batch = this x 32. Does **not** fix out-of-memory crashes |
| `--eval-every N` | evaluate every N **global** steps |
| `--eval-batches N` | batches per evaluation (default 50) |
| `--eval-device D` | `cpu` by default, and deliberately |
| `--eval-timeout S` | kill a stuck evaluation after S seconds and skip it (default 300) |
| `--eval-retries N` | extra attempts after a failed eval before giving up on that step (default 1). Cheap insurance, not a fix — see §9 |
| `--gc-every N` | force a Python collection every N steps (default 500) |
| `--report-every N` | metrics cadence to TensorBoard and the dashboard |
| `--nan-check M` | `report` (default) checks on the reporting cadence and stops on a bad gradient; `step` checks every step (+7%) and stops on the exact one; `skip` rides through bad gradients instead of stopping, up to `--max-skips`; `off` disables it |
| `--max-skips N` | with `--nan-check skip`, give up after this many skipped updates (default 20) |
| `--data-file-count N` | limit the visible corpus to this many tars per phase, to keep the loader's resident metadata small on a long run |
| `--data-phase-step-interval N` | steps per data phase (default 25000) |
| `--data-phase-shuffle` | randomize which tars each phase gets, instead of a fixed oldest-to-newest sequence. Still reproducible on resume — same step, same window — but a run that laps back to phase 0 gets a fresh shuffle rather than the exact partition it used last lap |

---

## 9. Gone wrong?

**Dashboard is blank / panels show `--`** — add `--io-dump dump.jsonl` to the
TUI command and check whether payloads are arriving at all.

**`Not enough memory resources are available`** — see §3. If it fails within a
few steps of starting, the machine is full; if it fails thousands of steps in,
that is the DirectML allocator and a restart is the cure, which is what
`--supervise` automates.

**`non-finite gradient at step N`** — the run diverged and stopped itself
before the optimizer could write NaN into the weights. The message names the
first bad gradients and says whether the weights were still clean, which
tells you if that step was the origin or an earlier one was.

Resume from the last checkpoint; it is guaranteed good, because
`checkpoint_io.save` refuses to write non-finite weights. Before this guard
existed a run diverged, carried on for three hours producing nothing, and
wrote six NaN checkpoints — with `max_to_keep` rotating the directory, the
last clean weights came within four writes of being deleted.

Check what is on disk before resuming:

```powershell
.venv-directml\Scripts\python.exe -c "import torch,glob,os; [print(os.path.basename(f), 'NaN' if any(torch.is_tensor(v) and not torch.isfinite(v).all() for v in torch.load(f,map_location='cpu',weights_only=False)['model_state'].values()) else 'clean') for f in sorted(glob.glob('C:/Users/Contrad/Documents/lc0-directml-checkpoint/checkpoint-*.pt'))]"
```

If it happens repeatedly, run with `--nan-check step` to catch the exact
step rather than the reporting window. It costs ~7% throughput (measured:
+37 ms on a 518 ms step), which is why `report` is the default.

**A run stopped and the last line mentions an evaluation** — it is not stuck.
The trainer waits up to `--eval-timeout` (times `1 + --eval-retries`), skips
the evaluation, and carries on. No steps are lost — the log will say
`did not succeed in N attempt(s); training continues` and keep going.

**Every automatic eval times out, always at exactly `--eval-timeout`, worker's
log never touched** — this happened for real (16 for 16 in one run) and the
cause is still open. Four faithful reproductions were tried — CPU device with
the real request files, concurrent with genuinely live training, through the
actual supervisor→daemon process chain, and with `--gc-every`/`--eval-every`
forced to the same step (production's defaults make that coincide on every
automatic eval by construction) — and all four succeeded in 10-15 s. So it is
real, but needs something a short test cannot compress, plausibly many hours
of live uptime. Two things now help without requiring the cause to be known:
the log names the worker's **PID** and a live CPU/thread snapshot the moment
the timeout fires, so a future occurrence — watched live — can be inspected
with `tasklist` / a debugger while it is still hung; and `--eval-retries`
(default 1) rides through it if the true cause turns out to be transient.
If it recurs, the PID and snapshot in the log are the next real lead.

**Checkpoint digest mismatch after editing the config** — the digest covers the
`model` and `training.optimizer` sections; changing either invalidates existing
checkpoints by design. `kda.chunk_size` and `gradient_accumulation_steps` are
deliberately excluded and safe to change between runs.

Fuller notes: [directml-windows.md](directml-windows.md) for setup and
troubleshooting, [metrics.md](metrics.md) for what every metric means,
[directml-session-handoff.md](directml-session-handoff.md) for the state of the
work.
