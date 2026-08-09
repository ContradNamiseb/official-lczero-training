"""Clear the shader caches and restart the display driver.

    .venv-directml\\Scripts\\python.exe scripts\\reset_gpu_state.py

An empirical remedy, not an understood one. After a morning of runs that
stalled and died, clearing the shader caches and resetting the driver
preceded a run that held 880 ms/step with flat GPU memory for 2,000 steps.
Which half did the work is not established, and the honest guess is the
driver reset: the caches are megabytes of compiled shaders on disk, not GPU
allocations, so their size does not explain a memory symptom. Restarting the
driver, on the other hand, reclaims allocations that badly-killed processes
left behind -- and this project has produced a great many of those. A clean
exit returns its own memory (measured: the adapter drops from ~5,100 MB to
~1,200 MB); one killed mid-allocation may not.

So run this after a bad patch, not as a ritual before every launch. It costs
a few seconds of shader recompilation on the next start.

The keystroke goes through ctypes rather than pyautogui. One hotkey does not
justify a GUI-automation dependency -- pyautogui pulls in pillow,
pygetwindow and friends -- in a virtualenv on a machine this short of
memory. It needs an interactive desktop session; over SSH or from a service
it will do nothing, which the script reports rather than hiding.
"""

from __future__ import annotations

import argparse
import ctypes
import os
import shutil
import sys
import time
from pathlib import Path

# Both caches. The Intel one is the larger of the two here and was missing
# from the first version of this script.
CACHE_PATHS = (
    Path(os.environ.get("LOCALAPPDATA", "")) / "D3DSCache",
    Path(os.environ.get("USERPROFILE", ""))
    / "AppData"
    / "LocalLow"
    / "Intel"
    / "ShaderCache",
)

_VK_SHIFT = 0x10
_VK_CONTROL = 0x11
_VK_LWIN = 0x5B
_VK_B = 0x42
_KEYEVENTF_KEYUP = 0x0002


def training_is_running() -> list[int]:
    """PIDs of any live trainer. Resetting the driver under one risks it."""
    try:
        import psutil
    except ImportError:
        return []

    found = []
    for process in psutil.process_iter(["pid", "name", "cmdline"]):
        try:
            if not (process.info["name"] or "").lower().startswith("python"):
                continue
            if any(
                "lczero_training" in part
                for part in (process.info["cmdline"] or [])
            ):
                found.append(process.info["pid"])
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            continue
    return found


def directory_size_mb(path: Path) -> float:
    total = 0
    for root, _, files in os.walk(path):
        for name in files:
            try:
                total += (Path(root) / name).stat().st_size
            except OSError:
                pass
    return total / 1024**2


def clear_cache(path: Path, dry_run: bool) -> None:
    """Empty a cache directory, keeping the directory itself.

    Contents rather than the folder: the driver expects it to exist and
    recreates entries inside it on demand. Files in use are skipped, which
    is normal -- something is always holding a shader cache open.
    """
    if not path.is_dir():
        print(f"  {path} -- not present, skipping")
        return

    size = directory_size_mb(path)
    if dry_run:
        print(f"  {path} -- would clear {size:.1f} MB")
        return

    skipped = 0
    for entry in path.iterdir():
        try:
            if entry.is_dir() and not entry.is_symlink():
                shutil.rmtree(entry)
            else:
                entry.unlink()
        except OSError:
            skipped += 1
    remaining = f", {skipped} in use and left alone" if skipped else ""
    print(f"  {path} -- cleared {size:.1f} MB{remaining}")


def restart_display_driver() -> None:
    """Win+Ctrl+Shift+B. The screen blanks for a second; nothing is lost."""
    if sys.platform != "win32":
        print("Not Windows; skipping the driver reset.")
        return
    user32 = ctypes.WinDLL("user32", use_last_error=True)
    # Held down in order, released in reverse, the way a person would.
    for key in (_VK_LWIN, _VK_CONTROL, _VK_SHIFT, _VK_B):
        user32.keybd_event(key, 0, 0, 0)
    for key in (_VK_B, _VK_SHIFT, _VK_CONTROL, _VK_LWIN):
        user32.keybd_event(key, 0, _KEYEVENTF_KEYUP, 0)
    print("Sent Win+Ctrl+Shift+B; the screen should blink.")


def gpu_state() -> str:
    """The pool that actually runs out. See directml/gpu_memory.py."""
    try:
        from lczero_training.directml import gpu_memory

        return gpu_memory.snapshot()
    except Exception:  # noqa: BLE001 - a report must not fail the reset
        return "gpu memory unavailable"


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

    running = training_is_running()
    if running:
        print(
            f"Training is running (PID {', '.join(map(str, running))}). Stop "
            "it first: resetting the driver under a live DirectML context is "
            "a good way to lose the run this is meant to protect.",
            file=sys.stderr,
        )
        return 1

    print(f"Before: {gpu_state()}")

    print("Shader caches:")
    for path in CACHE_PATHS:
        clear_cache(path, args.dry_run)

    if args.no_driver_reset or args.dry_run:
        print("Driver reset skipped.")
    else:
        restart_display_driver()
        # The reset is asynchronous; give it a moment before reading back.
        time.sleep(3)
        print(f"After:  {gpu_state()}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
