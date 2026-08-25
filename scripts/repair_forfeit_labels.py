#!/usr/bin/env python3
"""Relabel forfeited games in place, without reconverting the corpus.

A game awarded against an engine that crashed or lost on time carries a
result that describes the engine, not the position. trainingdata-tool now
fixes this at conversion (-forfeit-repair), but reconverting 2M games from
PGN costs hours. This edits the affected chunks directly instead: it
touches only the games named in a suspect list, and only the label fields.

  # look first -- this is the default, nothing is written
  python scripts/repair_forfeit_labels.py <corpus> --suspects suspects.txt

  # then commit to it
  python scripts/repair_forfeit_labels.py <corpus> --suspects suspects.txt --apply

WHAT IT CHANGES, PER GAME
  result_q  negated on EVERY ply. The stored value is already relative to
            the side to move on its own ply, so a single negation carries
            the alternation with it.
  played_q  reassigned on the FINAL ply only, to the new result_q. The
            tool derives it there from the result; on every other ply it
            comes from the next position's eval and is unaffected.
  result_d  untouched. The game is still decisive, just decided the other
            way.
Nothing else in the frame is read or written.

A CAVEAT THIS SCRIPT CANNOT REMOVE. The C++ repair keys off the PGN's
Termination tag, which states outright that a game was a forfeit.
V6TrainingData does not carry that tag, so here the only available
evidence is that the final evaluation contradicts the result. That is
strong evidence but not proof: an engine that blunders into a lost
position while still believing it is winning produces the same signature.
Between strong engines such games are normally adjudicated long before
that, but the possibility is why --threshold defaults to a demanding
value and why --apply is not the default.
"""

from __future__ import annotations

import argparse
import collections
import gzip
import multiprocessing as mp
import os
import sys
import tarfile
from pathlib import Path

import numpy as np

ITEMSIZE = 8356
FRAME = np.dtype({
    "names": ["version", "best_q", "result_q", "result_d", "played_q"],
    "formats": ["<u4", "<f4", "<f4", "<f4", "<f4"],
    "offsets": [0, 8284, 8308, 8312, 8316],
    "itemsize": ITEMSIZE,
})


def repair_bytes(raw: bytes) -> tuple[bytes, float, float] | None:
    """Return (repaired_bytes, agreement_before, agreement_after).

    None if the chunk is unusable. Operates on a mutable copy of the
    original buffer so every byte this script does not explicitly set is
    preserved exactly -- rebuilding the frame from a parsed array would
    risk dropping the fields this dtype does not map.
    """
    if not raw or len(raw) % ITEMSIZE:
        return None
    buffer = bytearray(raw)
    frames = np.frombuffer(buffer, dtype=FRAME)
    if not frames.flags.writeable:
        raise RuntimeError("frame view is read-only; cannot repair in place")
    if not np.all(np.isin(frames["version"], (6, 7))):
        return None

    before = float(frames["best_q"][-1] * frames["result_q"][-1])
    frames["result_q"] *= -1.0
    frames["played_q"][-1] = frames["result_q"][-1]
    after = float(frames["best_q"][-1] * frames["result_q"][-1])
    return bytes(buffer), before, after


def process_tar(job) -> dict:
    """Repair one tar's suspect members. Runs in a worker process."""
    tar_path, members, apply_changes = job
    members = set(members)
    result = {
        "tar": tar_path.name, "repaired": 0, "kept": 0,
        "failed": [], "checks": [],
    }
    tmp = tar_path.with_suffix(tar_path.suffix + ".repair")
    try:
        with tarfile.open(tar_path, "r|") as rd, \
             tarfile.open(tmp, "w") as wr:
            for member in rd:
                if not member.isfile():
                    wr.addfile(member, rd.extractfile(member))
                    continue
                data = rd.extractfile(member).read()
                if member.name in members:
                    outcome = repair_bytes(gzip.decompress(data))
                    if outcome is None:
                        result["failed"].append(member.name)
                    else:
                        payload, before, after = outcome
                        data = gzip.compress(payload)
                        member.size = len(data)
                        result["repaired"] += 1
                        result["checks"].append((before, after))
                import io as _io
                wr.addfile(member, _io.BytesIO(data))
                result["kept"] += 1
    except Exception as error:                       # noqa: BLE001
        tmp.unlink(missing_ok=True)
        result["failed"].append(f"<{type(error).__name__}: {error}>")
        return result

    # Every repaired game must now agree with its own evaluation, and the
    # member count must reconcile. Only then is the original replaced.
    bad = [(b, a) for b, a in result["checks"] if not (b < 0 <= a)]
    if bad or result["repaired"] != len(members) or result["failed"]:
        tmp.unlink(missing_ok=True)
        result["failed"].append(
            f"<verification failed: repaired {result['repaired']} of "
            f"{len(members)}, {len(bad)} sign checks bad>")
        return result

    if apply_changes:
        os.replace(tmp, tar_path)
    else:
        tmp.unlink(missing_ok=True)
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("corpus", type=Path)
    parser.add_argument("--suspects", type=Path, required=True,
                        help="Output of scan_forfeit_games.py.")
    parser.add_argument("--threshold", type=float, default=-0.5,
                        help="Repair only games whose agreement is BELOW "
                             "this, i.e. where the final evaluation clearly "
                             "contradicts the result. Default -0.5. Raising "
                             "it toward 0 sweeps in games whose evaluation "
                             "was merely unconvinced, where the true result "
                             "is not actually knowable.")
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--apply", action="store_true",
                        help="Actually rewrite the archives. Without this "
                             "the run is a rehearsal: every tar is rebuilt "
                             "and verified, then discarded.")
    parser.add_argument("--loose-dir", type=Path,
                        help="Also repair the unpacked copy of the corpus.")
    args = parser.parse_args()

    rows = []
    for line in args.suspects.read_text(encoding="utf-8").splitlines():
        if line.startswith("#") or not line.strip():
            continue
        ident, _plies, agreement = line.split("\t")
        if float(agreement) < args.threshold:
            rows.append((ident, float(agreement)))

    if not rows:
        print(f"nothing below threshold {args.threshold}", file=sys.stderr)
        return 1

    by_tar: dict[str, list[str]] = collections.defaultdict(list)
    for ident, _ in rows:
        tar_name, member = ident.split("::")
        by_tar[tar_name].append(member)

    print(f"{len(rows):,} games below agreement {args.threshold}, "
          f"across {len(by_tar)} tars")
    print("REHEARSAL -- nothing will be written. Add --apply to commit."
          if not args.apply else "APPLYING -- archives will be rewritten.")
    print()

    jobs = [(args.corpus / name, members, args.apply)
            for name, members in sorted(by_tar.items())]
    repaired = failed = 0
    with mp.Pool(args.workers) as pool:
        for res in pool.imap_unordered(process_tar, jobs):
            status = "OK" if not res["failed"] else "FAILED"
            print(f"  {res['tar']:<26} repaired {res['repaired']:>4}  "
                  f"kept {res['kept']:>5}  [{status}]")
            for problem in res["failed"]:
                print(f"      {problem}")
            repaired += res["repaired"]
            failed += len(res["failed"])

    print()
    print(f"repaired {repaired:,} games; {failed} failure(s)")

    if args.loose_dir and args.apply and not failed:
        loose = 0
        for ident, _ in rows:
            _, member = ident.split("::")
            path = args.loose_dir / member
            if not path.is_file():
                continue
            outcome = repair_bytes(gzip.decompress(path.read_bytes()))
            if outcome is None:
                continue
            payload, before, after = outcome
            if before < 0 <= after:
                path.write_bytes(gzip.compress(payload))
                loose += 1
        print(f"loose copy: repaired {loose:,} games")

    if not args.apply:
        print()
        print("Rehearsal only. Every archive above was rebuilt and verified,")
        print("then thrown away. Re-run with --apply to keep the results.")
    return 1 if failed else 0


if __name__ == "__main__":
    mp.freeze_support()
    sys.exit(main())
