#!/usr/bin/env python3
"""Measure the temporal Q signal across adjacent plies of a game.

Answers one question before any training-loop change is made: is the
"true continuation" signal -- how much a position's evaluation deviates
from its neighbours -- actually NEW information, or is it a proxy for
something the loader already computes?

The loader already has a per-position sampling weight
(ComputePositionSamplingWeight in csrc/loader/stages/position_sampling.cc)
built on |best_q - orig_q| and policy_kld. If the temporal signal is
strongly correlated with those, a temporal reweighting scheme buys nothing
that diff_focus cannot already express, and the caching work has no signal
to feed on. That is what the correlation table at the end is for.

This reads the corpus directly. It does not import torch, does not load a
network, and does not touch the training pipeline or any config.

  python scripts/measure_temporal_q.py <corpus-dir> [--games-per-tar 200]

THE SIGN CONVENTION IS THE WHOLE BALLGAME. root_q is from the
side-to-move's perspective, so it flips meaning every ply. Comparing raw
root_q[i] to root_q[i+1] measures mostly "whose turn is it", not "did the
evaluation change". Every frame is therefore folded to one fixed
perspective (q * (-1)^i) before any difference is taken. A global flip of
that perspective does not affect differences, so it does not matter
whether ply 0 is white or black.
"""

from __future__ import annotations

import argparse
import gzip
import math
import sys
import tarfile
from pathlib import Path

import numpy as np

# Field offsets within the packed V6 struct, from
# libs/lc0/src/trainingdata/trainingdata_v6.h. The struct is
# static_assert'd to 8356 bytes there; V7 extends it to 8396 with the V6
# fields at identical offsets, so the same offsets read both. Only the
# scalars needed here are mapped -- the 7432-byte probabilities array and
# the 832-byte plane bitboards are skipped rather than parsed.
_FIELDS = {
    "names": [
        "version",
        "invariance_info",
        "root_q",
        "best_q",
        "plies_left",
        "result_q",
        "played_q",
        "orig_q",
        "visits",
        "policy_kld",
    ],
    "formats": ["<u4", "u1", "<f4", "<f4", "<f4", "<f4", "<f4", "<f4", "<u4", "<f4"],
    "offsets": [0, 8278, 8280, 8284, 8304, 8308, 8316, 8328, 8340, 8348],
}
V6_SIZE = 8356
V7_SIZE = 8396

# invariance_info bit 5: game was adjudicated. Fishtest games are cut off
# around +4 pawns, so the final plies carry an evaluation that never
# resolved. Those tail plies are the ones most likely to show a large
# artificial jump, which would flatter the temporal signal.
ADJUDICATED_BIT = 1 << 5

MIN_PLIES = 3  # need i-1, i, i+1 for a curvature estimate


def _dtype(itemsize: int) -> np.dtype:
    return np.dtype({**_FIELDS, "itemsize": itemsize})


def parse_frames(raw: bytes) -> np.ndarray | None:
    """Decode one game's chunk into a structured array, or None if unusable."""
    for size in (V6_SIZE, V7_SIZE):
        if len(raw) % size == 0 and len(raw) >= size:
            frames = np.frombuffer(raw, dtype=_dtype(size))
            # Guard against a coincidental length match by sanity-checking
            # the version field, which must be 6 or 7 for every frame.
            versions = np.unique(frames["version"])
            if np.all((versions == 6) | (versions == 7)):
                return frames
    return None


def game_signals(frames: np.ndarray) -> dict[str, np.ndarray] | None:
    """Per-position temporal signals for one game.

    Returns arrays aligned to the interior plies [1 .. n-2], since a
    curvature estimate needs both neighbours.
    """
    n = len(frames)
    if n < MIN_PLIES:
        return None

    # Fold to one fixed perspective. See the module docstring -- without
    # this every measurement below is dominated by the side-to-move flip.
    sign = np.where(np.arange(n) % 2 == 0, 1.0, -1.0).astype(np.float32)
    q_fixed = frames["root_q"].astype(np.float32) * sign

    interior = slice(1, n - 1)

    # First difference: how far this position's eval sits from the
    # previous one. The "did something change" signal.
    step_back = np.abs(q_fixed[1 : n - 1] - q_fixed[0 : n - 2])
    step_fwd = np.abs(q_fixed[2:n] - q_fixed[1 : n - 1])
    adjacent = np.maximum(step_back, step_fwd)

    # Second difference: how far this position sits off the straight line
    # between its neighbours. This is the actual "true continuation" test
    # -- a position on a smooth evaluation trajectory has curvature near
    # zero even if the eval is steadily climbing, whereas a tactical spike
    # or a blunder shows up as a sharp deviation.
    curvature = np.abs(
        q_fixed[1 : n - 1] - 0.5 * (q_fixed[0 : n - 2] + q_fixed[2:n])
    )

    # The signal the loader can ALREADY weight on today, for comparison.
    best_q = frames["best_q"].astype(np.float32)[interior]
    orig_q = frames["orig_q"].astype(np.float32)[interior]
    diff_focus = np.abs(best_q - orig_q)  # NaN where orig_q was not cached

    return {
        "adjacent": adjacent,
        "curvature": curvature,
        "diff_focus": diff_focus,
        "policy_kld": frames["policy_kld"].astype(np.float32)[interior],
        "adjudicated": np.full(
            n - 2,
            bool(frames["invariance_info"][-1] & ADJUDICATED_BIT),
        ),
    }


def iter_games(corpus: Path, games_per_tar: int, tar_limit: int | None):
    """Yield raw chunk bytes, sampled across the whole corpus.

    Takes the first `games_per_tar` members of each tar rather than a
    random draw: tarfile reads sequentially, so this stops early instead
    of scanning 155 MB to reach a handful of scattered members. Sampling
    is spread across every tar, which is what keeps it representative --
    reading N games from one tar would not be.
    """
    tars = sorted(corpus.glob("*.tar"))
    if tar_limit:
        tars = tars[:tar_limit]

    if tars:
        for index, path in enumerate(tars, 1):
            print(
                f"  [{index}/{len(tars)}] {path.name}", file=sys.stderr, flush=True
            )
            try:
                with tarfile.open(path, "r|") as archive:  # streaming mode
                    taken = 0
                    for member in archive:
                        if taken >= games_per_tar:
                            break
                        if not member.isfile():
                            continue
                        handle = archive.extractfile(member)
                        if handle is None:
                            continue
                        try:
                            yield gzip.decompress(handle.read())
                        except (OSError, EOFError):
                            continue
                        taken += 1
            except tarfile.TarError as error:
                print(f"    skipped: {error}", file=sys.stderr)
        return

    # Loose .gz layout: the same corpus unpacked into one directory per
    # 6,000 games. Walk it per-directory for the same reason as above -- a
    # flat rglob would return all 6,000 of fishtest-see-0 before reaching
    # fishtest-see-1, which is a sample of one directory, not of the corpus.
    subdirs = sorted(d for d in corpus.iterdir() if d.is_dir())
    if tar_limit:
        subdirs = subdirs[:tar_limit]
    for index, subdir in enumerate(subdirs, 1):
        print(
            f"  [{index}/{len(subdirs)}] {subdir.name}", file=sys.stderr, flush=True
        )
        taken = 0
        for path in sorted(subdir.glob("*.gz")):
            if taken >= games_per_tar:
                break
            try:
                yield gzip.decompress(path.read_bytes())
            except (OSError, EOFError):
                continue
            taken += 1


class StreamStats:
    """Moments and percentiles over a stream, in constant memory.

    A full-corpus sweep is ~232M interior positions. Holding those to call
    np.percentile would cost ~4.6 GB across the four signals, which does
    not fit beside the iGPU's shared allocation on this machine. Percentiles
    therefore come from a fixed histogram: 20,000 bins over [0, 2] resolves
    to 1e-4, far finer than any decision made from these numbers, and the
    footprint is 160 KB per signal no matter how large the corpus grows.
    """

    def __init__(self, lo: float = 0.0, hi: float = 2.0, bins: int = 20000):
        self.lo, self.hi, self.bins = lo, hi, bins
        self.hist = np.zeros(bins, dtype=np.int64)
        self.count = 0
        self.total = 0.0
        self.total_sq = 0.0
        self.above_range = 0

    def update(self, values: np.ndarray) -> None:
        finite = values[np.isfinite(values)]
        if finite.size == 0:
            return
        wide = finite.astype(np.float64)
        self.count += finite.size
        self.total += float(wide.sum())
        self.total_sq += float(np.square(wide).sum())
        self.above_range += int((finite > self.hi).sum())
        scaled = (finite - self.lo) / (self.hi - self.lo) * self.bins
        index = np.clip(scaled.astype(np.int64), 0, self.bins - 1)
        self.hist += np.bincount(index, minlength=self.bins)

    @property
    def mean(self) -> float:
        return self.total / self.count if self.count else math.nan

    def percentile(self, pct: float) -> float:
        if self.count == 0:
            return math.nan
        target = self.count * pct / 100.0
        index = int(np.searchsorted(np.cumsum(self.hist), target))
        index = min(index, self.bins - 1)
        return self.lo + (index + 0.5) * (self.hi - self.lo) / self.bins

    def describe(self, name: str) -> None:
        if self.count == 0:
            print(f"  {name:<14} (no finite values)")
            return
        print(
            f"  {name:<14} mean {self.mean:7.4f}   "
            f"p50 {self.percentile(50):7.4f}  p75 {self.percentile(75):7.4f}  "
            f"p90 {self.percentile(90):7.4f}  p95 {self.percentile(95):7.4f}  "
            f"p99 {self.percentile(99):7.4f}"
        )

    def render(self, rows: int = 12, width: int = 48) -> None:
        if self.count == 0:
            return
        top = self.percentile(99)
        if top <= 0:
            print("    (all values zero)")
            return
        edges = np.linspace(0.0, top, rows + 1)
        cum = np.cumsum(self.hist)
        positions = np.clip(
            (edges / (self.hi - self.lo) * self.bins).astype(np.int64),
            0,
            self.bins - 1,
        )
        counts = np.diff(cum[positions])
        peak = counts.max() or 1
        shown = counts.sum() or 1
        for count, low, high in zip(counts, edges[:-1], edges[1:]):
            bar = "#" * int(width * count / peak)
            print(
                f"    {low:6.3f}-{high:6.3f} |{bar:<{width}} "
                f"{100.0 * count / shown:5.1f}%"
            )


class StreamCorr:
    """Pearson correlation from paired sums, in constant memory."""

    def __init__(self) -> None:
        self.n = 0
        self.sx = self.sy = self.sxx = self.syy = self.sxy = 0.0

    def update(self, a: np.ndarray, b: np.ndarray) -> None:
        ok = np.isfinite(a) & np.isfinite(b)
        if not ok.any():
            return
        x = a[ok].astype(np.float64)
        y = b[ok].astype(np.float64)
        self.n += x.size
        self.sx += float(x.sum())
        self.sy += float(y.sum())
        self.sxx += float(np.square(x).sum())
        self.syy += float(np.square(y).sum())
        self.sxy += float((x * y).sum())

    @property
    def r(self) -> float:
        if self.n < 2:
            return math.nan
        cov = self.sxy - self.sx * self.sy / self.n
        vx = self.sxx - self.sx * self.sx / self.n
        vy = self.syy - self.sy * self.sy / self.n
        if vx <= 0 or vy <= 0:
            return math.nan
        return cov / math.sqrt(vx * vy)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("corpus", type=Path)
    parser.add_argument("--games-per-tar", type=int, default=200)
    parser.add_argument("--tar-limit", type=int, default=None)
    parser.add_argument(
        "--drop-adjudicated",
        action="store_true",
        help="Exclude games the rescorer marked adjudicated. Their final "
        "plies end on an unresolved evaluation, which inflates the "
        "temporal signal for reasons that have nothing to do with the "
        "position being a genuine discontinuity.",
    )
    parser.add_argument("--save", type=Path, help="Write raw signals to .npz")
    args = parser.parse_args()

    if not args.corpus.is_dir():
        print(f"not a directory: {args.corpus}", file=sys.stderr)
        return 1

    keys = ("adjacent", "curvature", "diff_focus", "policy_kld")
    stats = {key: StreamStats() for key in keys}
    pairs = [
        ("curvature", "diff_focus"),
        ("curvature", "policy_kld"),
        ("adjacent", "diff_focus"),
        ("adjacent", "policy_kld"),
        ("curvature", "adjacent"),
    ]
    corrs = {pair: StreamCorr() for pair in pairs}
    games = positions = skipped = too_short = 0
    adjudicated_positions = 0
    orig_q_finite = 0

    print("reading corpus...", file=sys.stderr)
    for raw in iter_games(args.corpus, args.games_per_tar, args.tar_limit):
        frames = parse_frames(raw)
        if frames is None:
            skipped += 1
            continue
        signals = game_signals(frames)
        if signals is None:
            # Fewer than MIN_PLIES. Valid data, just too short for a
            # three-ply window -- NOT corrupt. Kept distinct from the
            # parse-failure count above for that reason.
            too_short += 1
            continue

        adjudicated = signals["adjudicated"]
        adjudicated_positions += int(adjudicated.sum())
        if args.drop_adjudicated and adjudicated.any():
            # Whole-game flag, so this drops the game outright.
            continue

        for key in keys:
            stats[key].update(signals[key])
        for pair in pairs:
            corrs[pair].update(signals[pair[0]], signals[pair[1]])

        orig_q_finite += int(np.isfinite(signals["diff_focus"]).sum())
        games += 1
        positions += len(signals["adjacent"])

    if games == 0:
        print("no usable games found", file=sys.stderr)
        return 1

    print(f"\n{'=' * 68}")
    print(f"games {games:,}   positions {positions:,}")
    print(f"skipped: {too_short:,} valid but <{MIN_PLIES} ply (NOT corrupt)"
          f"   |   {skipped:,} unparseable")
    print(
        f"adjudicated: {adjudicated_positions:,} positions "
        f"({100.0 * adjudicated_positions / max(positions, 1):.1f}%)"
        + ("  [excluded]" if args.drop_adjudicated else "")
    )
    print("=" * 68)

    print("\nDISTRIBUTIONS")
    stats["adjacent"].describe("adjacent |dq|")
    stats["curvature"].describe("curvature")
    stats["diff_focus"].describe("diff_focus")
    stats["policy_kld"].describe("policy_kld")

    print(
        f"\n  orig_q present for "
        f"{100.0 * orig_q_finite / max(positions, 1):.1f}% of positions "
        "(NaN means it was not in the eval cache; diff_focus silently "
        "falls back to default_weight for those)"
    )

    print("\nCURVATURE  (the 'true continuation' signal, clipped at p99)")
    stats["curvature"].render()

    print("\nCORRELATION  (does this duplicate what the loader already has?)")
    for left, right in pairs:
        print(f"  r({left:<10}, {right:<10}) = {corrs[(left, right)].r:+.4f}")

    # A corpus converted from PGN has no eval cache to repair against and
    # no search policy to diverge from, so both of the loader's existing
    # weight inputs can be entirely absent. That is not a parse failure --
    # it decides the question this script exists to answer, so say so
    # rather than leaving a table of NaNs to interpret.
    orig_q_absent = orig_q_finite == 0
    kld_absent = stats["policy_kld"].total == 0.0

    print(f"\n{'=' * 68}")
    print("READING THIS:")
    if orig_q_absent or kld_absent:
        print("  The existing diff_focus weighting is INERT on this corpus:")
        if orig_q_absent:
            print("    - orig_q is NaN for every position, and")
            print("      ComputePositionSamplingWeight returns default_weight")
            print("      immediately on NaN orig_q.")
        if kld_absent:
            print("    - policy_kld is 0.0 for every position.")
        print("  So the correlations above are NaN because there is nothing")
        print("  to correlate against, not because the measurement failed.")
        print("  Consequences:")
        print("    1. The temporal signal cannot be redundant -- it is the")
        print("       only per-position quality signal this corpus carries.")
        print("    2. A position_sampling{} block using only diff_focus_*")
        print("       knobs would do NOTHING here: every weight would come")
        print("       back as default_weight, i.e. uniform.")
        print("    3. Switching to position_count mode WITHOUT adding a")
        print("       temporal term is therefore strictly worse than the")
        print("       current rate mode -- it trades a clean disjoint")
        print("       partition for uniform sampling with replacement, and")
        print("       buys nothing in exchange.")
    else:
        print("  |r| > 0.7 against diff_focus or policy_kld -- the temporal")
        print("    signal is largely redundant. diff_focus can already")
        print("    express it, and no caching work is justified.")
        print("  |r| < 0.3 -- it is genuinely new information, and worth")
        print("    carrying into training.")
        print("  Between -- real but partly overlapping; check whether the")
        print("    tail (p95+) picks out different positions than diff_focus")
        print("    does, since the tail is what reweighting acts on.")
    print("=" * 68)

    if args.save:
        # Histograms, not raw values -- the raw arrays for a full sweep are
        # ~4.6 GB, which is the reason this script streams at all. These
        # reconstruct every number printed above.
        np.savez_compressed(
            args.save,
            games=games,
            positions=positions,
            adjudicated_positions=adjudicated_positions,
            orig_q_finite=orig_q_finite,
            **{f"hist_{key}": stats[key].hist for key in keys},
            **{f"count_{key}": stats[key].count for key in keys},
            **{f"total_{key}": stats[key].total for key in keys},
            **{f"totalsq_{key}": stats[key].total_sq for key in keys},
            hist_lo=0.0,
            hist_hi=2.0,
        )
        print(f"\nsaved histograms to {args.save}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
