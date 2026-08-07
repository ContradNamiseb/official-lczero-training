# Training metrics: what they are and how to read them

Every metric the native Windows DirectML trainer emits, what it means, and
which direction is good. Values shown as examples are real readings from the
`kda-hybrid-128x4-3k1m-8h` net around step 102,700 — use them as a rough
sense of scale, not as targets.

Where a metric also exists in the old TensorFlow pipeline, the tag name is
identical, so a DirectML run overlays directly on an old `leelalogs` run in
TensorBoard.

## Where they appear

| surface | what you get |
|---|---|
| **TensorBoard** | every metric, under `<tensorboard_path>/<config name>-train` and `-test` |
| **TUI** (`lc0-directml-tui`) | live pane with losses, grad norm, LR, throughput, ETA |
| **log file** | a subset, every `--log-every` steps |

`--report-every N` controls the TensorBoard/TUI cadence (default 10).
Metrics are only pulled off the GPU on reporting steps, so the other steps
cost nothing.

Two runs appear in TensorBoard when a test split is configured:

* **`-train`** — measured on the batch just trained on. Noisy by nature.
* **`-test`** — measured on held-out data every `--eval-every` steps, averaged
  over `--eval-batches` batches. Much smoother, and the only honest read on
  generalization.

**The gap between them is the point.** Train falling while test flattens or
rises is overfitting, and no single metric will tell you that on its own.

### Read `-test`, not the log

The log line and the `-train` run report **one micro-batch**. At batch 32 that
is 32 positions: `Policy Accuracy` can only take values that are multiples of
3.125%, and it swings about 8 percentage points between consecutive reports.
A real improvement of 2 points over 30,000 steps is invisible underneath that.

Measured over steps 102,709 to 136,709 of the live run, first quarter versus
last:

| tag | change | t |
|---|---|---|
| `Policy Loss` | −0.066 | −5.6 |
| `Policy Accuracy` | +1.69pp | +4.0 |
| `Thresholded Policy Accuracy @ 5` | +2.49pp | +6.7 |
| `Value Winner Loss` | −0.004 | −0.5 |
| `Moves Left Loss` | −0.001 | −0.1 |

`t` is the difference in standard errors; below about 2 it is noise. Policy was
learning the whole time and the log could not show it. **If you want to know
whether a run is working, open the `-test` run in TensorBoard.**

Once `gradient_accumulation_steps` is set, the reported **losses** improve too:
they are summed across every micro-batch and divided, so a run with
`gradient_accumulation_steps: 8` reports a loss measured on 256 positions
rather than 32.

The **diagnostics** — policy accuracy, entropy, value accuracy, KDA stats,
parameter norms — are still measured on a single micro-batch. Averaging them
across all 8 would cost 8x the kernels and 8x the allocation churn to sharpen
numbers that are observational rather than optimized, and parameter norms do
not vary within a step at all. So `Policy Accuracy` stays as noisy as it was;
read it from the `-test` run.

---

## Losses — the things being optimized

These are what the gradient actually minimizes. **Down is good** for all of
them, but only the trend matters; step-to-step noise is large at batch 32.

| tag | meaning | example |
|---|---|---|
| `Total Loss` | weighted sum of every configured loss | ~3.5 |
| `Policy Loss` | cross-entropy against the search's move distribution | ~2.2 |
| `Value Winner Loss` | cross-entropy against the game result (W/D/L) | ~0.7 |
| `Moves Left Loss` | Huber loss on plies remaining | ~0.5 |

`Total Loss` is the only one the optimizer sees directly. If it stalls while
the individual losses move in opposite directions, your loss weights are
fighting each other.

Additional loss tags appear if you configure them: `Value Q Loss`,
`Value ST Loss`, `Value Err L`, `Value Cat L`, `Value ST Err Loss`,
`Value ST Cat Loss`, `Policy Optimistic ST Loss`, `Reg term`. Your model
defines the `q`, `st` and `optimistic_st` heads but the shipped config does
not train them — add loss entries to enable both training and these tags.

---

## Policy diagnostics — is it picking the right move?

`Policy Loss` can improve while the move ranking does not. These separate
the two.

**`Policy Accuracy`** (%) — how often the top predicted move is the search's
top move. Example: **~30-45%**.
> **Up is good.** Very noisy at batch 32 (each step is 32 positions, so
> resolution is ~3%). Judge it over hundreds of steps, not step to step.

**`Thresholded Policy Accuracy @ 1 / 2 / 5 / 10`** (%) — how often the
*correct* move gets at least 1% / 2% / 5% / 10% predicted probability.
Example: **~95 / 90 / 75 / 57**.
> **Up is good**, and these are far less noisy than raw accuracy. `@ 1`
> answers "does it at least consider the right move?" — if that drops, the
> policy is confidently wrong, which is much worse than being uncertain.
> `@ 10` is the sharpest signal of real improvement.

**`Policy Entropy`** — spread of the predicted distribution, in nats.
Example: **~2.2**.
> **Neither direction is inherently good.** Falling means growing confidence,
> healthy while accuracy also rises — but collapsing toward 0 with flat
> accuracy means the net is confidently wrong and has stopped exploring.
> Rising toward `log(legal moves)` ≈ 3.5 means it is giving up and predicting
> near-uniform.

**`Policy UL`** (uniform loss) — cross-entropy against a flat distribution
over legal moves, i.e. distance from knowing nothing. Example: **~4.5**.
> **Up is good.** A net that has learned something should be far from
> uniform. Falling toward ~3.5 means it is collapsing toward guessing.

**`Policy SL`** (search loss) — `1 / P(best move)`, an estimate of how long a
search would take to find the best move. Example: **~14-22**.
> **Down is good.** Directly proportional to engine search effort, so this is
> the metric closest to playing strength. Very sensitive to outliers — one
> position where the right move gets ~0 probability dominates the average.

---

## Value diagnostics — is it judging positions?

**`Value Accuracy`** (%) — how often the predicted W/D/L argmax matches the
actual outcome. Example: **~65-80%**.
> **Up is good.** Bounded by how decidable the positions are: an early
> middlegame genuinely is uncertain, so this will never approach 100%.

**`MSE Loss`** — squared error between predicted and target WDL vectors.
Example: **~0.15-0.30**.
> **Down is good.** A calibration measure rather than a ranking one: it
> penalizes overconfidence in a way accuracy does not.

---

## Optimizer health

**`Gradient norm`** — global L2 norm of the gradient, **before** clipping.
Example: **~5-11**, against `max_grad_norm: 10.0`.
> **Neither direction is good; watch the relationship to your clip.**
> Consistently above the clip means most updates are being scaled down and
> the effective LR is lower than configured. Toward 0 means learning has
> stalled. Sudden spikes usually precede a loss blow-up.
> Yours sits near 10, so a meaningful fraction of steps are being clipped —
> that is not broken, but a lower LR or a higher clip would give the
> optimizer more faithful updates.

**`LR`** — current learning rate. Constant unless you configure a schedule.

**`ms_per_step`** — running average wall time per step. DirectML-only tag.
> **Down is good.** A sudden rise usually means memory pressure or a CPU
> fallback. If it climbs, grep the log for
> `not currently supported on the DML backend`.
>
> A step is one *optimizer* step, so raising
> `gradient_accumulation_steps` raises this proportionally and is not a
> regression — compare positions/sec, not ms/step, across settings.

### Effective batch

`training.gradient_accumulation_steps` (`--grad-accum` to override) sets how
many micro-batches are summed into one optimizer step. The effective batch is
that times the loader's `batch_size`, at the peak memory of a *single*
micro-batch. It is the TF pipeline's `num_batch_splits`.

This is the only way to enlarge the batch on an iGPU that is already at its
memory ceiling. Measured on the Iris Xe with this 128x4 model:

| batch | accum | effective | ms/step | pos/sec |
|---|---|---|---|---|
| 32 | 1 | 32 | 1114 | 28.7 |
| 32 | 2 | 64 | 2172 | 29.5 |
| 32 | 4 | 128 | 4029 | 31.8 |
| 32 | 8 | **256** | 7785 | **32.9** |
| 48 | 1 | — | *out of memory* | |

Throughput is flat-to-better, because one optimizer step is amortized over
several forwards. The gain is entirely in gradient quality per update.

The setting is not part of the checkpoint digest, so it can be changed
between runs without invalidating a checkpoint.

---

## Parameter norms

**`Params`**, **`Embedding params`**, **`Smolgen params`** — L2 norm over
each group. Example: **~216 / 147 / 28**.
> **Neither direction is good; watch for drift.** Slow growth is normal.
> Rapid growth means weight decay is not holding (the decay selector may not
> be matching what you think). Collapse toward 0 means weights are being
> regularized to death.
>
> Expect steady growth with the shipped config, not stability: its
> `decay_selector` has `otherwise_include` false and lists only the heads
> with `include: true`, so the embedding *and the whole encoder tower* are
> undecayed. Over steps 102.7k-136.7k `Embedding params` rose 148 -> 159
> (+7.4%). That is the selector working as written — worth knowing before
> you read it as a bug.
>
> Note: the TF pipeline logged these on its *test* writer; with no separate
> test model here they appear on both runs.

---

## KDA mixer internals

The KDA recurrence has failure modes none of the above will show you. All
three of these can degrade while `Total Loss` looks unremarkable, because
the rest of the network compensates. **If loss plateaus, check these first.**

Emitted per KDA block: `KDA/block0`, `block1`, `block2` (your tower is 3 KDA
blocks then 1 MHA block).

**`KDA/blockN decay saturated %`** — fraction of forget gates pinned at the
`-10` floor, meaning state fully wiped at that token. Example:
**19% / 11% / 6%**.
> **The single most important KDA metric. Watch it go up.**
> A slow drift is fine. A climb toward 60-80% means the recurrence has
> degenerated into a per-square feed-forward — the "remember along the scan"
> behaviour that makes KDA worth having is gone. **Alarm above ~50%.**

**`KDA/blockN beta mean`** — delta-rule write strength, sigmoid so 0-1.
Example: **0.95 / 0.92 / 0.90**.
> **Watch it go down.** Toward 0 the mixer has stopped writing to its state
> at all. Pinned at exactly 1.0 everywhere it overwrites completely each
> token and keeps no history. Mid-to-high and stable is healthy.
> **Alarm below ~0.3.**

**`KDA/blockN output gate mean`** — sigmoid gate on the mixer output.
Example: **0.97 / 0.97 / 0.95**.
> **Watch it go down.** Collapsing toward 0 means the block has become a
> pass-through: the mixer output is multiplied away no matter how good it is.
> **Alarm below ~0.3.**

**`KDA/blockN log decay mean`** — average log state retention per token.
Example: **-2.68 / -1.88 / -1.16**.
> **Neither direction is inherently good — watch the *shape* across blocks.**
> `exp(-2.68) ≈ 0.07` retained per token in block 0 versus
> `exp(-1.16) ≈ 0.31` in block 2: early blocks forget fast, later blocks
> remember longer. That division of labour is healthy. All three converging
> to one value, or block 0 heading toward 0 (never forgetting, state
> saturates), is worth investigating.

**`KDA/blockN recurrence rms`** / **`KDA/blockN output rms`** — magnitude of
the recurrence output, and of the mixer output after projection (before the
encoder block's DeepNorm alpha). Example: **~0.15** and **~0.45**.
> **Watch for collapse or blow-up, not direction.** Toward 0 the mixer
> contributes nothing to the residual stream. Growth by orders of magnitude
> signals instability — usually visible in `Gradient norm` first.

### The one chart to keep

Put `decay saturated %`, `beta mean` and `output gate mean` for all three
blocks on a single TensorBoard chart. If loss ever plateaus:

* all three flat → the mixer is fine, look at data or learning rate
* any one moved sharply → the mixer is the cause

Collection is off by default and enabled only on reporting steps, so 9 of
every 10 steps run the mixer completely untouched. Everything is captured
under `no_grad` and detached — none of it touches the gradient.

---

## Quick triage

| symptom | look at |
|---|---|
| loss falling, strength not improving | `-test` run; `Policy SL`; `Thresholded @ 10` |
| loss plateaued | the KDA chart above; `Gradient norm`; `LR` |
| loss rising / NaN | `Gradient norm` spikes; KDA `output rms` blow-up |
| train improving, test flat | overfitting — more data, or check `chunk_pool_size` |
| suddenly slower | `ms_per_step`; grep log for `not currently supported on the DML` |
| confidently wrong | `Policy Entropy` falling while `Policy Accuracy` flat |

## Related

* `docs/directml_training_port.md` — the port itself, including the DirectML
  defects worked around and the Phase 4 performance numbers.
* `scripts/add_test_split.py` — generate a config with a held-out split, which
  is what enables the `-test` run.
