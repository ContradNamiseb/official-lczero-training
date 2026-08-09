"""Clear the shader caches and restart the display driver, by hand.

    .venv-directml\\Scripts\\python.exe scripts\\reset_gpu_state.py

The supervisor already does this before every launch -- see
``--reset-gpu-state`` -- so this is for the paths that have no supervisor:
``directml_train``, a plain TUI run, or tidying up after something died
without one. The logic lives in ``directml/gpu_reset.py`` so both callers do
the same thing.

Unlike the supervisor, this cannot know that nothing holds a device, so it
checks. A driver reset is system-wide and would take a live run down with it.
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from lczero_training.directml import gpu_reset  # noqa: E402


def training_is_running() -> list[int]:
    """PIDs of any live trainer, excluding this script and its supervisor."""
    try:
        import psutil
    except ImportError:
        return []

    found = []
    for process in psutil.process_iter(["pid", "name", "cmdline"]):
        try:
            if process.info["pid"] == psutil.Process().pid:
                continue
            if not (process.info["name"] or "").lower().startswith("python"):
                continue
            command = process.info["cmdline"] or []
            # The daemon and the plain trainer. Not the supervisor, which
            # holds no device and is the thing that calls the reset itself.
            if any(
                "directml_daemon" in part or "directml_train" in part
                for part in command
            ):
                found.append(process.info["pid"])
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            continue
    return found


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Report what would be cleared and change nothing.",
    )
    parser.add_argument(
        "--no-driver-reset",
        action="store_true",
        help="Clear the caches but leave the display driver alone.",
    )
    args = parser.parse_args(argv)

    logging.basicConfig(
        level=logging.INFO, format="%(message)s", stream=sys.stderr
    )

    running = training_is_running()
    if running:
        print(
            f"A trainer is running (PID {', '.join(map(str, running))}). Stop "
            "it first: a driver reset is system-wide and would take the run "
            "down with it.",
            file=sys.stderr,
        )
        return 1

    from lczero_training.directml import gpu_memory

    print(f"Before: {gpu_memory.snapshot()}")
    if args.dry_run:
        gpu_reset.clear_shader_caches(dry_run=True)
        print("Dry run: nothing changed, driver not reset.")
        return 0

    gpu_reset.reset("cache" if args.no_driver_reset else "full")
    print(f"After:  {gpu_memory.snapshot()}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
