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

## 4. Evaluate by hand

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

## 5. Checkpoints and networks

Export a `.pb.gz` from the latest checkpoint by hand:

```powershell
.venv-directml\Scripts\python.exe -m lczero_training.commands.directml_export --config docs/kda_split.textproto --output C:\Users\Contrad\Documents\lc0-directml-networks\manual-export.pb.gz
```

Start a checkpoint from an existing Leela network (only needed once):

```powershell
.venv-directml\Scripts\python.exe -m lczero_training.commands.directml_init --config docs/kda_split.textproto --lczero-model path\to\net.pb.gz
```

---

## 6. Tests

```powershell
.venv-directml\Scripts\python.exe -m pytest src\lczero_training -q
```

One file while working on it:

```powershell
.venv-directml\Scripts\python.exe -m pytest src\lczero_training\directml\test_subprocess_eval.py -q
```

---

## 7. Flags worth knowing

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

| trainer flag | what it does |
|---|---|
| `--kda-chunk-size 8` | ~2.4x faster than the default 16 and the lowest memory. Mathematically identical |
| `--grad-accum N` | effective batch = this x 32. Does **not** fix out-of-memory crashes |
| `--eval-every N` | evaluate every N **global** steps |
| `--eval-batches N` | batches per evaluation (default 50) |
| `--eval-device D` | `cpu` by default, and deliberately |
| `--eval-timeout S` | kill a stuck evaluation after S seconds and skip it (default 300) |
| `--gc-every N` | force a Python collection every N steps (default 500) |
| `--report-every N` | metrics cadence to TensorBoard and the dashboard |

---

## 8. Gone wrong?

**Dashboard is blank / panels show `--`** — add `--io-dump dump.jsonl` to the
TUI command and check whether payloads are arriving at all.

**`Not enough memory resources are available`** — see §3. If it fails within a
few steps of starting, the machine is full; if it fails thousands of steps in,
that is the DirectML allocator and a restart is the cure, which is what
`--supervise` automates.

**A run stopped and the last line mentions an evaluation** — it is not stuck.
The trainer waits up to `--eval-timeout`, skips the evaluation, and carries on.
No steps are lost.

**Checkpoint digest mismatch after editing the config** — the digest covers the
`model` and `training.optimizer` sections; changing either invalidates existing
checkpoints by design. `kda.chunk_size` and `gradient_accumulation_steps` are
deliberately excluded and safe to change between runs.

Fuller notes: [directml-windows.md](directml-windows.md) for setup and
troubleshooting, [metrics.md](metrics.md) for what every metric means,
[directml-session-handoff.md](directml-session-handoff.md) for the state of the
work.
