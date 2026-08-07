"""Report DirectML adapters and run backward smoke tests.

    uv run lc0-directml-device

Phase 1 of docs/directml_training_port.md. Nothing else imports this
module, so a Linux environment without PyTorch is unaffected.
"""

import argparse
import sys


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--device",
        default=None,
        help="Device to test, e.g. directml:0 or cpu. Defaults to directml:0.",
    )
    parser.add_argument(
        "--skip-smoke-tests",
        action="store_true",
        help="Only list adapters; do not allocate or run backward passes.",
    )
    args = parser.parse_args()

    try:
        from lczero_training.directml import device as dml
    except ImportError as error:
        print(f"error: {error}", file=sys.stderr)
        return 2

    count = dml.device_count()
    print(f"DirectML device count: {count}")
    for index in range(count):
        print(f"  [{index}] {dml.adapter_name(index)}")
    if count == 0:
        print("error: no DirectML adapter found", file=sys.stderr)
        return 1

    if args.skip_smoke_tests:
        return 0

    device = dml.resolve_device(args.device)
    print(f"\nRunning smoke tests on {device}:")
    results = dml.run_smoke_tests(device)
    for result in results:
        status = "ok  " if result.passed else "FAIL"
        print(f"  [{status}] {result.name}: {result.detail}")

    failed = [result for result in results if not result.passed]
    if failed:
        print(f"\n{len(failed)} of {len(results)} smoke tests failed.")
        return 1
    print(f"\nAll {len(results)} smoke tests passed.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
