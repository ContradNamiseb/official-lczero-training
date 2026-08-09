"""Clear the shader caches and restart the display driver between launches.

The supervisor calls this before every daemon launch, which is the whole
point: a restart is already the only thing that returns DirectML memory to
the OS, and this makes the restart also clear whatever the driver was still
holding. Doing it by hand between runs worked; doing it automatically is
what makes an unattended run survive.

An empirical remedy, and worth being honest about which half does the work.
The caches are ~16 MB of compiled shaders on disk, not GPU allocations, so
their size does not explain a memory symptom. Restarting the display driver
is the likelier half: it reclaims allocations left behind by processes that
died badly, and a trainer killed mid-allocation is exactly that. A clean
exit already returns its own memory -- measured, the adapter drops from
~5,100 MB to ~1,200 MB -- but a killed one may not.

The driver reset is a **system-wide** action. The screen blanks for a
moment and every GPU application on the machine has its device reset, not
just ours. That is acceptable between training launches and would not be
acceptable during one, so nothing here may be called while a trainer holds a
device. The caller owns that guarantee; the supervisor has it by
construction, because it waits for the child to exit first.
"""

from __future__ import annotations

import ctypes
import logging
import os
import shutil
import sys
import time
from pathlib import Path

logger = logging.getLogger(__name__)

# Both of them. The Intel cache is the larger on this hardware (12.6 MB
# against 3.9 MB) and was missing from the first version of this.
CACHE_PATHS: tuple[Path, ...] = (
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

# The reset is asynchronous. Building a DirectML device while the driver is
# still coming back is a way to fail a launch that would otherwise have been
# fine, so the supervisor waits this long before spawning.
SETTLE_SECONDS = 5.0


def directory_size_mb(path: Path) -> float:
    total = 0
    for root, _, files in os.walk(path):
        for name in files:
            try:
                total += (Path(root) / name).stat().st_size
            except OSError:
                pass
    return total / 1024**2


def clear_shader_caches(dry_run: bool = False) -> float:
    """Empty the shader cache directories. Returns the MB cleared.

    Contents rather than the directories: the driver expects them to exist
    and recreates entries on demand. Files in use are skipped, which is
    normal -- something usually holds a shader cache open -- and is not a
    failure worth reporting up.
    """
    cleared = 0.0
    for path in CACHE_PATHS:
        if not path.is_dir():
            continue
        size = directory_size_mb(path)
        if dry_run:
            logger.info("Would clear %.1f MB from %s", size, path)
            cleared += size
            continue
        skipped = 0
        for entry in path.iterdir():
            try:
                if entry.is_dir() and not entry.is_symlink():
                    shutil.rmtree(entry)
                else:
                    entry.unlink()
            except OSError:
                skipped += 1
        cleared += size
        logger.info(
            "Cleared %.1f MB from %s%s",
            size,
            path,
            f" ({skipped} in use, left alone)" if skipped else "",
        )
    return cleared


def restart_display_driver() -> bool:
    """Send Win+Ctrl+Shift+B. Returns whether the keystroke was sent.

    Through ctypes rather than pyautogui: one keystroke does not justify a
    GUI-automation dependency, and this machine has no memory to spare for
    one. Needs an interactive desktop session -- from a service or over a
    bare SSH session the keys go nowhere, which is worth a log line rather
    than a crash.
    """
    if sys.platform != "win32":
        logger.info("Not Windows; skipping the display driver reset.")
        return False
    try:
        user32 = ctypes.WinDLL("user32", use_last_error=True)
        # Pressed in order and released in reverse, the way a person would.
        for key in (_VK_LWIN, _VK_CONTROL, _VK_SHIFT, _VK_B):
            user32.keybd_event(key, 0, 0, 0)
        for key in (_VK_B, _VK_SHIFT, _VK_CONTROL, _VK_LWIN):
            user32.keybd_event(key, 0, _KEYEVENTF_KEYUP, 0)
    except Exception:  # noqa: BLE001 - never fail a launch over this
        logger.warning(
            "Could not send the driver reset keystroke", exc_info=True
        )
        return False
    logger.info("Sent Win+Ctrl+Shift+B to restart the display driver.")
    return True


def reset(mode: str, *, settle: bool = True) -> None:
    """Run the reset. ``mode`` is "off", "cache" or "full".

    Never raises: this runs on the path to starting a training run, and
    failing to tidy up is not a reason to refuse to train.
    """
    if mode == "off":
        return

    from lczero_training.directml import gpu_memory

    try:
        logger.info("GPU before the reset: %s", gpu_memory.snapshot())
        clear_shader_caches()
        if mode == "full":
            if restart_display_driver() and settle:
                # Let the driver come back before anything asks it for a
                # device.
                time.sleep(SETTLE_SECONDS)
        logger.info("GPU after the reset:  %s", gpu_memory.snapshot())
    except Exception:  # noqa: BLE001
        logger.warning(
            "GPU state reset did not complete; continuing to launch anyway",
            exc_info=True,
        )
