"""Free physical memory, which is what every OOM in this port is really about.

DirectML reports the same sentence for every failure -- "Not enough memory
resources are available to complete this operation" -- and nothing at all
about the machine it happened on. So every out-of-memory crash here has been
diagnosed after the fact by inference, from a log that recorded the failure
and none of the conditions. One was blamed on a leak, one on the batch size,
one on a second process; the evidence for each was circumstantial because the
numbers were gone by the time anyone looked.

This records them. The figure that matters is **available physical memory**,
not the commit charge: GPU-shared memory has to be resident and cannot be
paged out, so a DirectML allocation fails when physical memory runs short
even though there is pagefile to spare. Runs have died with 238 MB free while
commit was nowhere near its limit.

Cheap enough to call on every reporting step -- psutil reads a counter, it
does not walk anything -- and it must never be the reason a step fails, so
every call here is best-effort.
"""

from __future__ import annotations

import logging
import time

logger = logging.getLogger(__name__)

_GB = 1024.0**3

# Below this, an allocation is close to failing and the log should say so
# before it does rather than after. Real failures here have landed around
# 240 MB free; this is roughly twice that, to leave a step or two of warning.
LOW_WATER_GB = 0.5


def snapshot() -> str:
    """One line describing the memory the next allocation has to come from."""
    try:
        import psutil

        virtual = psutil.virtual_memory()
        process = psutil.Process().memory_info()
        return (
            f"available {virtual.available / _GB:.2f} GB of "
            f"{virtual.total / _GB:.2f}; this process rss "
            f"{process.rss / _GB:.2f} GB, commit {process.vms / _GB:.2f} GB"
        )
    except Exception:  # noqa: BLE001 - diagnostics may never break a run
        return "memory unavailable"


def available_gb() -> float | None:
    """Available physical memory in GB, or None if it cannot be read."""
    try:
        import psutil

        return psutil.virtual_memory().available / _GB
    except Exception:  # noqa: BLE001
        return None


# Callers check every step, so the warning needs a floor on its own rate or
# it becomes a line a second in the log it is trying to make readable.
WARN_INTERVAL_SECONDS = 30.0
_last_warned = 0.0


def warn_if_low(context: str) -> None:
    """Log a warning while there is still room to notice it.

    The point is a line in the log *before* the allocation that fails, so a
    post-mortem can tell "the machine ran out" from "the allocator stranded
    what it had" -- two failures with the same message and different fixes.

    Safe to call on every step: it reads one counter, and it will not log
    more than once per ``WARN_INTERVAL_SECONDS``.
    """
    global _last_warned

    available = available_gb()
    if available is None or available >= LOW_WATER_GB:
        return
    now = time.monotonic()
    if now - _last_warned < WARN_INTERVAL_SECONDS:
        return
    _last_warned = now
    logger.warning(
        "Only %.2f GB of physical memory is available (%s). DirectML cannot "
        "page shared memory out, so an allocation is about to fail: close "
        "whatever else is resident, or lower the batch.",
        available,
        context,
    )
