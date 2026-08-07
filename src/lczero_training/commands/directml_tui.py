"""Live TUI for native Windows DirectML training.

    uv run lc0-directml-tui --config docs/example_kda_directml.textproto

Spawns lc0-directml-daemon and renders its per-step metrics. The same
dashboard as lc0-tui, but pointed at the DirectML daemon and with a real
training pane instead of the JAX placeholder.
"""

import argparse
import logging
import sys

import anyio

from lczero_training.commands import configure_root_logging
from lczero_training.tui.directml_app import DirectMlTuiApp

DIRECTML_DAEMON = "lczero_training.commands.directml_daemon"


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, help="Root config textproto.")
    parser.add_argument("--logfile", help="Also write the daemon log here.")
    parser.add_argument(
        "--io-dump",
        dest="io_dump",
        help="Append the raw daemon JSONL to this file, for diagnosing "
        "panels that render no data.",
    )
    parser.add_argument(
        "--daemon-flag",
        action="append",
        default=[],
        dest="daemon_flags",
        help="Extra argument for the daemon (repeatable). Prefer `-- ...`.",
    )
    return parser


async def _amain(args: argparse.Namespace) -> None:
    app = DirectMlTuiApp(args=args)
    await app.run_async()


def main(argv: list[str] | None = None) -> int:
    configure_root_logging(logging.INFO)

    # Everything after a bare `--` goes straight to the daemon. argparse
    # will not accept a --daemon-flag value that begins with a dash unless
    # it is written --daemon-flag=--foo=1, which is unreadable; this is the
    # usual idiom and needs no quoting.
    argv = list(sys.argv[1:] if argv is None else argv)
    passthrough: list[str] = []
    if "--" in argv:
        index = argv.index("--")
        argv, passthrough = argv[:index], argv[index + 1 :]

    args = _build_parser().parse_args(argv)
    args.daemon_flags = list(args.daemon_flags) + passthrough
    # The daemon needs the config too -- it opens the pipeline.
    if not any(f.startswith("--config") for f in args.daemon_flags):
        args.daemon_flags = ["--config", args.config] + args.daemon_flags
    anyio.run(_amain, args)
    return 0


if __name__ == "__main__":
    sys.exit(main())
