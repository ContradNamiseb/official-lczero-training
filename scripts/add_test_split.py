"""Add a hash-based train/test split to a loader config.

    python scripts/add_test_split.py IN.textproto OUT.textproto --test-weight 5

Inserts a ChunkSourceSplitter after chunk_source_loader and duplicates
every downstream stage into a "train_" and a "test_" branch, exposing the
two tensor generators as the `train` and `test` outputs.

The ratio is the splitter's `weight` pair, so it lives in the generated
config and can be edited there afterwards without re-running this.

The split is by chunk, hashed on (tar filename, chunk index): deterministic
across runs and restarts, drawn from every tar so both sides share a
distribution, and whole-game so no position leaks between them.
"""

import argparse
import copy
import sys

from google.protobuf import text_format

from proto.root_config_pb2 import RootConfig

SPLITTER = "chunk_source_splitter"
# Stages after the splitter get duplicated; everything at or before
# chunk_source_loader is shared.
SHARED = ("file_path_provider", "chunk_source_loader")


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source")
    parser.add_argument("destination")
    parser.add_argument(
        "--train-weight",
        type=int,
        default=95,
        help="Relative share of chunks used for training.",
    )
    parser.add_argument(
        "--test-weight",
        type=int,
        default=5,
        help="Relative share held out for evaluation.",
    )
    args = parser.parse_args(argv)

    if args.train_weight <= 0 or args.test_weight <= 0:
        parser.error("both weights must be positive")

    config = RootConfig()
    with open(args.source) as handle:
        text_format.Parse(handle.read(), config)

    stages = list(config.data_loader.stage)
    if any(s.name == SPLITTER for s in stages):
        print("config already has a splitter; nothing to do", file=sys.stderr)
        return 1

    shared = [s for s in stages if s.name in SHARED]
    downstream = [s for s in stages if s.name not in SHARED]
    if not downstream:
        parser.error("no stages found after chunk_source_loader")

    splitter = config.data_loader.stage.add()
    splitter.name = SPLITTER
    splitter.input.append(SHARED[-1])
    split_cfg = splitter.chunk_source_splitter
    for name in ("train", "test"):
        output = split_cfg.output.add()
        output.name = name
        output.queue_capacity = 16
    split_cfg.weight.extend([args.train_weight, args.test_weight])

    new_stages = list(shared) + [splitter]
    first_downstream = downstream[0].name
    for branch in ("train", "test"):
        for stage in downstream:
            clone = copy.deepcopy(stage)
            clone.name = f"{branch}_{stage.name}"
            for index, name in enumerate(clone.input):
                if name == first_downstream or name in {
                    s.name for s in downstream
                }:
                    clone.input[index] = f"{branch}_{name}"
            # The head of each branch reads its own splitter output.
            if stage.name == first_downstream:
                del clone.input[:]
                clone.input.append(f"{SPLITTER}.{branch}")
            new_stages.append(clone)

    del config.data_loader.stage[:]
    config.data_loader.stage.extend(new_stages)

    tail = downstream[-1].name
    del config.data_loader.output[:]
    config.data_loader.output.append(f"train:train_{tail}")
    config.data_loader.output.append(f"test:test_{tail}")

    with open(args.destination, "w") as handle:
        handle.write(text_format.MessageToString(config))

    total = args.train_weight + args.test_weight
    print(f"wrote {args.destination}")
    print(
        f"  split {args.train_weight}:{args.test_weight} "
        f"({args.test_weight / total:.1%} held out)"
    )
    print(f"  outputs: train:train_{tail}, test:test_{tail}")
    print(f"  stages: {len(new_stages)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
