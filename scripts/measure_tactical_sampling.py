#!/usr/bin/env python3
"""Measure the tactical sampling terms over the whole corpus.

Replays the C++ weight formula from
csrc/loader/stages/position_sampling.cc against real games and reports
what fraction of sampled positions each candidate config would actually
deliver -- in particular the queen-endgame share, which is the number the
config in docs/kda_split.textproto is tuned against.

Everything is vectorised and streamed. The naive per-position loop takes
~11 hours over 2M games; this runs in roughly the time it takes to read
the tars. Memory is constant: only scalars are accumulated, never
per-position arrays.

Several configs are evaluated in ONE pass, because the expensive part is
decompressing and popcounting, not the weight arithmetic. Re-tuning
therefore does not mean re-reading 50 GB.

  python scripts/measure_tactical_sampling.py <corpus-dir> [--bad-list out.txt]

THE PERSPECTIVE FOLD MATTERS. Material balance, like root_q, is reported
from the side-to-move's point of view, so its sign flips every ply.
Comparing two plies without correcting for that measures the alternation,
not the material change. Same correction as MaterialFromPerspectiveOf().
"""

from __future__ import annotations

import argparse
import gzip
import multiprocessing as mp
import sys
import tarfile
from pathlib import Path

import numpy as np

FRAME = np.dtype({
    "names": ["version", "planes", "root_q"],
    "formats": ["<u4", ("<u8", 104), "<f4"],
    "offsets": [0, 7440, 8280],
    "itemsize": 8356,
})
PIECE_VALUE = np.array([1, 3, 3, 5, 9, 0], dtype=np.float32)

# Sacrifice detector, matching the proto defaults.
WINDOW, THRESHOLD, LOOKAHEAD, DECAY = 4, 2.0, 6, 0.8

# (label, material_max, require_queens, w_sacrifice, w_endgame, alpha, beta, tau)
CONFIGS = [
    ("uniform (today)",            40.0, True, 0.0, 0.0, 1.0, 1.0, 1.0),
    ("mat40 we3 a1.5 b0.2  <-cfg", 40.0, True, 1.0, 3.0, 1.5, 0.2, 2.0),
    ("mat40 we3 a1.0 b0.3",        40.0, True, 1.0, 3.0, 1.0, 0.3, 2.0),
    ("mat40 we4 a1.5 b0.2",        40.0, True, 1.0, 4.0, 1.5, 0.2, 2.0),
    ("mat30 we3 a1.5 b0.2",        30.0, True, 1.0, 3.0, 1.5, 0.2, 2.0),
    ("mat50 we3 a1.5 b0.2",        50.0, True, 1.0, 3.0, 1.5, 0.2, 2.0),
]


def popcount64(values: np.ndarray) -> np.ndarray:
    """Population count over a uint64 array.

    Hand-rolled because np.bitwise_count needs numpy >= 2.0 and this repo
    pins 1.26. The uint64 multiply is expected to wrap; that is what puts
    the byte sums into the top byte.
    """
    v = values.astype(np.uint64, copy=True)
    v -= (v >> np.uint64(1)) & np.uint64(0x5555555555555555)
    v = (v & np.uint64(0x3333333333333333)) + (
        (v >> np.uint64(2)) & np.uint64(0x3333333333333333))
    v = (v + (v >> np.uint64(4))) & np.uint64(0x0F0F0F0F0F0F0F0F)
    return ((v * np.uint64(0x0101010101010101)) >> np.uint64(56)).astype(np.int32)


def game_features(frames: np.ndarray):
    """Per-position material, queen count, sacrifice signal and folded eval."""
    n = len(frames)
    current = frames["planes"][:, :13]                 # current position only
    counts = popcount64(current[:, :12]).astype(np.float32)

    ours = counts[:, :6] @ PIECE_VALUE
    theirs = counts[:, 6:12] @ PIECE_VALUE
    material = ours - theirs                            # side-to-move relative
    total_material = ours + theirs
    queens = counts[:, 4] + counts[:, 10]

    index = np.arange(n)
    sign = np.where(index % 2 == 0, 1.0, -1.0).astype(np.float32)
    material_fixed = material * sign
    eval_fixed = frames["root_q"].astype(np.float32) * sign

    # Net material the mover at j gives up over the next WINDOW plies,
    # converted back into j's own perspective.
    end = np.minimum(index + WINDOW, n - 1)
    given_up = material - material_fixed[end] * sign
    candidate = np.where(given_up >= THRESHOLD, given_up, 0.0).astype(np.float32)

    # Propagate forward, decayed. Only same-parity offsets: a sacrifice is
    # credited to the side that made it, so the opponent's replies in
    # between are not boosted by it.
    sacrifice = np.zeros(n, dtype=np.float32)
    for offset in range(0, LOOKAHEAD + 1, 2):
        # Games shorter than the lookahead exist: without this guard
        # candidate[: n - offset] becomes a NEGATIVE slice, which silently
        # returns a longer array than expected and fails to broadcast.
        if offset >= n:
            break
        shifted = candidate if offset == 0 else np.concatenate(
            [np.zeros(offset, np.float32), candidate[: n - offset]])
        np.maximum(sacrifice, shifted * (DECAY ** offset), out=sacrifice)

    return total_material, queens, sacrifice, eval_fixed


def process_tar(path: Path) -> dict:
    """Accumulate one tar's statistics. Runs in a worker process.

    Returns scalars only -- never per-position arrays -- so the parent
    merges a few hundred bytes per tar regardless of corpus size, and each
    worker's peak memory is one game.
    """
    n_configs = len(CONFIGS)
    acc = {
        "games": 0, "positions": 0, "bad": [], "short": 0,
        "sampled_qe": np.zeros(n_configs), "min_accept": np.zeros(n_configs),
        "qe_share_sum": {}, "qe_games": {},
        "sac_positions": 0, "sac_win": 0, "sac_loss": 0,
    }
    for _, mx, *_ in CONFIGS:
        acc["qe_share_sum"].setdefault(mx, 0.0)
        acc["qe_games"].setdefault(mx, 0)

    def consume(name, raw):
        if raw is None or len(raw) == 0 or len(raw) % FRAME.itemsize:
            acc["bad"].append(name); return
        frames = np.frombuffer(raw, dtype=FRAME)
        if not np.all(np.isin(frames["version"], (6, 7))):
            acc["bad"].append(name); return
        if len(frames) < 3:
            # Valid data, just too short for a three-ply window. Counted
            # separately: calling these "unreadable" invites someone to
            # delete perfectly good 1-2 ply games.
            acc["short"] += 1
            return

        total_material, queens, sacrifice, eval_fixed = game_features(frames)
        acc["games"] += 1
        acc["positions"] += len(frames)

        for mx in acc["qe_share_sum"]:
            mask = (total_material <= mx) & (queens >= 1)
            acc["qe_share_sum"][mx] += float(mask.mean())
            acc["qe_games"][mx] += int(mask.any())

        active = sacrifice > 0
        if active.any():
            acc["sac_positions"] += int(active.sum())
            acc["sac_win"] += int((eval_fixed[active] > 0.15).sum())
            acc["sac_loss"] += int((eval_fixed[active] < -0.15).sum())

        for k, (_, mx, req_q, w_s, w_e, alpha, beta, tau) in enumerate(CONFIGS):
            qe = ((total_material <= mx) &
                  ((queens >= 1) if req_q else True)).astype(np.float32)
            if w_s == 0.0 and w_e == 0.0:
                acc["sampled_qe"][k] += float(qe.mean())
                acc["min_accept"][k] += 1.0
                continue
            mean = (w_s * sacrifice + w_e * qe) / (w_s + w_e)
            weight = np.minimum(mean * alpha + beta, tau)
            acc["sampled_qe"][k] += float((weight * qe).sum() / weight.sum())
            acc["min_accept"][k] += float(weight.min() / weight.max())

    try:
        with tarfile.open(path, "r|") as archive:
            for member in archive:
                if not member.isfile():
                    continue
                name = f"{path.name}::{member.name}"
                handle = archive.extractfile(member)
                if handle is None:
                    acc["bad"].append(name); continue
                try:
                    raw = gzip.decompress(handle.read())
                except (OSError, EOFError):
                    acc["bad"].append(name); continue
                consume(name, raw)
    except tarfile.TarError as error:
        acc["bad"].append(f"{path.name}::<TAR ERROR: {error}>")
    return acc


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("corpus", type=Path)
    parser.add_argument("--tar-limit", type=int, default=None)
    parser.add_argument("--workers", type=int, default=4,
                        help="Parallel tar readers. The work is CPU-bound on "
                             "gzip and popcount, not on disk, so this scales "
                             "with physical cores (4 here). Default 4 leaves "
                             "headroom for a running trainer.")
    parser.add_argument("--bad-list", type=Path,
                        help="Write identifiers of unreadable games here.")
    args = parser.parse_args()

    tars = sorted(args.corpus.glob("*.tar"))
    if args.tar_limit:
        tars = tars[: args.tar_limit]
    if not tars:
        print(f"no .tar files in {args.corpus}", file=sys.stderr)
        return 1

    n_configs = len(CONFIGS)
    sampled_qe = np.zeros(n_configs)
    min_accept = np.zeros(n_configs)
    qe_share_sum: dict[float, float] = {}
    qe_games: dict[float, int] = {}
    games = positions = short = 0
    sac_positions = sac_win = sac_loss = 0
    bad: list[str] = []

    print(f"reading {len(tars)} tars across {args.workers} workers...",
          file=sys.stderr)
    with mp.Pool(args.workers) as pool:
        for done, acc in enumerate(
                pool.imap_unordered(process_tar, tars, chunksize=1), 1):
            games += acc["games"]
            positions += acc["positions"]
            short += acc["short"]
            sampled_qe += acc["sampled_qe"]
            min_accept += acc["min_accept"]
            sac_positions += acc["sac_positions"]
            sac_win += acc["sac_win"]
            sac_loss += acc["sac_loss"]
            bad.extend(acc["bad"])
            for mx, value in acc["qe_share_sum"].items():
                qe_share_sum[mx] = qe_share_sum.get(mx, 0.0) + value
            for mx, value in acc["qe_games"].items():
                qe_games[mx] = qe_games.get(mx, 0) + value
            if done % 10 == 0 or done == len(tars):
                print(f"  {done}/{len(tars)} tars   {games:,} games",
                      file=sys.stderr, flush=True)

    if games == 0:
        print("no usable games", file=sys.stderr)
        return 1

    print()
    print(f"{'=' * 72}")
    print(f"games {games:,}   positions {positions:,}")
    print(f"skipped: {short:,} valid but <3 ply (not analysable, NOT corrupt)"
          f"   |   {len(bad):,} genuinely unreadable")
    print("=" * 72)

    print()
    print("QUEEN ENDGAME AVAILABILITY (the ceiling on any per-game weighting)")
    for mx in sorted(qe_share_sum):
        print(f"  material_max {mx:>5.0f}:  {100*qe_games[mx]/games:5.1f}% of games "
              f"contain one   |  {100*qe_share_sum[mx]/games:5.2f}% of positions")

    print()
    print("SACRIFICE EVENTS (selected on the event, so both outcomes appear)")
    if sac_positions:
        print(f"  {sac_positions:,} positions ({100*sac_positions/positions:.2f}%)")
        print(f"  sacrificer winning {100*sac_win/sac_positions:5.1f}%  |  "
              f"losing {100*sac_loss/sac_positions:5.1f}%  |  neutral "
              f"{100*(sac_positions-sac_win-sac_loss)/sac_positions:5.1f}%")

    print()
    print("SAMPLED QUEEN-ENDGAME SHARE BY CONFIG")
    for k, (label, *_rest) in enumerate(CONFIGS):
        print(f"  {label:<30}{100*sampled_qe[k]/games:6.2f}%   "
              f"min acceptance {100*min_accept[k]/games:5.1f}%")
    print("=" * 72)

    if bad:
        print()
        print(f"{len(bad)} unreadable game(s)")
        for name in sorted(bad)[:10]:
            print(f"  {name}")
        if len(bad) > 10:
            print(f"  ... and {len(bad)-10} more")
        if args.bad_list:
            args.bad_list.write_text(
                chr(10).join(sorted(bad)) + chr(10), encoding="utf-8")
            print()
            print(f"written to {args.bad_list}")

    return 0


if __name__ == "__main__":
    mp.freeze_support()
    sys.exit(main())
