#!/usr/bin/env python3
"""Scan the corpus for forfeited games and report the outcome breakdown.

A Fishtest game can end because a side crashed or lost on time. The result
is then awarded against that engine regardless of the position, so the
stored label contradicts the stored evaluation: the net is told a balanced
position is a dead loss. That is worse than useless data -- it is wrong
data, and under position_count sampling each such game is emitted
position_count times per epoch.

The 37 found previously were all 1-2 ply, which made them trivially
visible. A crash at move 20 would look like a normal game to any
length-based check, so this looks at the DISAGREEMENT between result and
evaluation instead, at whatever length it occurs.

  python scripts/scan_forfeit_games.py <corpus-dir> --workers 8 --out suspects.txt

HOW A LEGITIMATE DECISIVE GAME ENDS. Fishtest adjudicates once the score
passes roughly +4 pawns, so a real win ends with the final evaluation
already extreme, and a mate more so. A game whose final evaluation is
still near zero has not been decided by chess. The histogram this prints
shows the two populations; pick --eval-threshold where they separate
rather than trusting the default.

PERSPECTIVE. root_q and result_q are both relative to the side to move on
their own ply, so they can be compared directly WITHIN a ply without any
sign folding. Comparing across plies would need the usual (-1)^i fold;
this script deliberately only ever compares within one ply.
"""

from __future__ import annotations

import argparse
import collections
import gzip
import multiprocessing as mp
import sys
import tarfile
from pathlib import Path

import numpy as np

FRAME = np.dtype({
    "names": ["version", "invariance", "root_q", "best_q", "plies_left",
              "result_q", "result_d"],
    "formats": ["<u4", "u1", "<f4", "<f4", "<f4", "<f4", "<f4"],
    "offsets": [0, 8278, 8280, 8284, 8304, 8308, 8312],
    "itemsize": 8356,
})
ADJUDICATED_BIT = 1 << 5

# Histogram of final |root_q| for decisive games, so the threshold can be
# read off the data instead of assumed.
HIST_BINS = 20


def classify_game(frames: np.ndarray, eval_threshold: float):
    """Return (outcome, is_suspect, agreement, plies).

    outcome is +1 win / -1 loss / 0 draw, from the perspective of whoever
    moves on ply 0.

    `agreement` is root_q * result_q on the FINAL ply -- the evaluation
    projected onto the direction of the result. Both are relative to the
    same side to move on that ply, so this needs no sign folding.

      +1  the final position overwhelmingly supports the result (mate, or
          a Fishtest adjudication)
       0  the position is balanced but the result is decisive
      -1  the position says the winner was dead lost

    Magnitude alone (an earlier version of this used |root_q|) does NOT
    work: a game ending at +0.73 and won is a perfectly legitimate win
    that merely stopped short of the adjudication margin, while a game
    ending at -0.95 and won is a severe contradiction. Both have a large
    |root_q|; only the signed projection separates them.
    """
    n = len(frames)
    result_first = float(frames["result_q"][0])
    if result_first > 0.5:
        outcome = 1
    elif result_first < -0.5:
        outcome = -1
    else:
        outcome = 0

    last = n - 1
    result_last = float(frames["result_q"][last])
    eval_last = float(frames["root_q"][last])
    decisive = abs(result_last) > 0.99

    agreement = eval_last * result_last if decisive else None
    suspect = bool(decisive and agreement < eval_threshold)
    return outcome, suspect, agreement, n


def process_tar(args) -> dict:
    """Scan one tar. Runs in a worker; returns scalars plus a suspect list."""
    path, eval_threshold = args
    acc = {
        "games": 0, "positions": 0, "bad": 0,
        "win": 0, "loss": 0, "draw": 0,
        "adjudicated": 0,
        "suspects": [],
        "hist": np.zeros(HIST_BINS, dtype=np.int64),
        "suspect_plies": collections.Counter(),
    }
    try:
        with tarfile.open(path, "r|") as archive:
            for member in archive:
                if not member.isfile():
                    continue
                handle = archive.extractfile(member)
                if handle is None:
                    acc["bad"] += 1
                    continue
                try:
                    raw = gzip.decompress(handle.read())
                except (OSError, EOFError):
                    acc["bad"] += 1
                    continue
                if not raw or len(raw) % FRAME.itemsize:
                    acc["bad"] += 1
                    continue
                frames = np.frombuffer(raw, dtype=FRAME)
                if not np.all(np.isin(frames["version"], (6, 7))):
                    acc["bad"] += 1
                    continue

                outcome, suspect, final_abs, plies = classify_game(
                    frames, eval_threshold)
                acc["games"] += 1
                acc["positions"] += plies
                acc["win" if outcome == 1 else
                    "loss" if outcome == -1 else "draw"] += 1
                if int(frames["invariance"][-1]) & ADJUDICATED_BIT:
                    acc["adjudicated"] += 1
                if final_abs is not None:      # `agreement`, in [-1, 1]
                    idx = int((final_abs + 1.0) / 2.0 * HIST_BINS)
                    acc["hist"][min(max(idx, 0), HIST_BINS - 1)] += 1
                if suspect:
                    acc["suspects"].append(
                        f"{path.name}::{member.name}\t{plies}\t{final_abs:.3f}")
                    acc["suspect_plies"][min(plies, 200)] += 1
    except tarfile.TarError as error:
        acc["suspects"].append(f"{path.name}::<TAR ERROR: {error}>\t0\t0")
    return acc


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("corpus", type=Path)
    parser.add_argument("--workers", type=int, default=4,
                        help="Parallel tar readers. 8 measured ~2x faster "
                             "than serial here; the default stays at 4 so a "
                             "running trainer is not starved.")
    parser.add_argument("--eval-threshold", type=float, default=0.5,
                        help="A decisive game whose result-vs-eval agreement is "
                             "below this is flagged. Read the histogram "
                             "before trusting this value.")
    parser.add_argument("--tar-limit", type=int, default=None)
    parser.add_argument("--out", type=Path, help="Write the suspect list here.")
    args = parser.parse_args()

    tars = sorted(args.corpus.glob("*.tar"))
    if args.tar_limit:
        tars = tars[: args.tar_limit]
    if not tars:
        print(f"no .tar files in {args.corpus}", file=sys.stderr)
        return 1

    totals = collections.Counter()
    hist = np.zeros(HIST_BINS, dtype=np.int64)
    suspects: list[str] = []
    suspect_plies = collections.Counter()

    print(f"scanning {len(tars)} tars across {args.workers} workers...",
          file=sys.stderr)
    work = [(t, args.eval_threshold) for t in tars]
    with mp.Pool(args.workers) as pool:
        for done, acc in enumerate(
                pool.imap_unordered(process_tar, work, chunksize=1), 1):
            for key in ("games", "positions", "bad", "win", "loss", "draw",
                        "adjudicated"):
                totals[key] += acc[key]
            hist += acc["hist"]
            suspects.extend(acc["suspects"])
            suspect_plies.update(acc["suspect_plies"])
            if done % 20 == 0 or done == len(tars):
                print(f"  {done}/{len(tars)} tars   {totals['games']:,} games   "
                      f"{len(suspects):,} suspects", file=sys.stderr, flush=True)

    games = totals["games"]
    if games == 0:
        print("no usable games", file=sys.stderr)
        return 1

    print()
    print("=" * 70)
    print(f"games {games:,}   positions {totals['positions']:,}   "
          f"unreadable {totals['bad']:,}")
    print("=" * 70)

    print()
    print("OUTCOME BREAKDOWN (from the perspective of the ply-0 mover)")
    for label, key in (("win", "win"), ("loss", "loss"), ("draw", "draw")):
        n = totals[key]
        print(f"  {label:<6}{n:>12,}   {100.0*n/games:5.2f}%")
    print(f"  {'adjud.':<6}{totals['adjudicated']:>12,}   "
          f"{100.0*totals['adjudicated']/games:5.2f}%  (game marked adjudicated)")

    decisive = int(hist.sum())
    print()
    print("RESULT-vs-EVALUATION AGREEMENT OF DECISIVE GAMES  (root_q * result_q)")
    print("  +1 = the final position supports the result (mate/adjudication).")
    print("   0 = balanced position, decisive result.  -1 = winner was lost.")
    if decisive:
        peak = hist.max() or 1
        for i, count in enumerate(hist):
            lo, hi = -1.0 + 2.0 * i / HIST_BINS, -1.0 + 2.0 * (i + 1) / HIST_BINS
            bar = "#" * int(46 * count / peak)
            print(f"    {lo:+.2f}..{hi:+.2f} |{bar:<46} {100.0*count/decisive:6.3f}%")

    print()
    print(f"SUSPECTS (decisive, agreement < {args.eval_threshold})")
    print(f"  {len(suspects):,} games ({100.0*len(suspects)/games:.4f}%)")
    if suspect_plies:
        print("  by game length:")
        short = sum(c for p, c in suspect_plies.items() if p <= 4)
        mid = sum(c for p, c in suspect_plies.items() if 4 < p <= 40)
        long = sum(c for p, c in suspect_plies.items() if p > 40)
        print(f"    <=4 ply  {short:>8,}   (the trivially visible kind)")
        print(f"    5-40 ply {mid:>8,}   (invisible to a length check)")
        print(f"    >40 ply  {long:>8,}")
    print("=" * 70)

    if suspects and args.out:
        args.out.write_text(
            "# tar::member\tplies\tfinal_abs_eval" + chr(10) +
            chr(10).join(sorted(suspects)) + chr(10), encoding="utf-8")
        print()
        print(f"suspect list written to {args.out}")

    return 0


if __name__ == "__main__":
    mp.freeze_support()
    sys.exit(main())
