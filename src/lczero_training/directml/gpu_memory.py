"""GPU memory, which is the pool that actually runs out.

``host_memory`` was the wrong instrument, and it took a crash report to see
it. A run died reporting "Not enough memory resources are available" with
**6.16 GB of system RAM free** and the process at 1.70 GB RSS. Both facts
were true, because they describe different pools:

* ``psutil`` sees system RAM. DirectML's D3D12 heaps barely appear in it --
  they are GPU allocations, and on an integrated adapter they come out of a
  carve-out that WDDM caps at **half of system RAM**, here ~5,965 MB.
* When that carve-out is full, every allocation fails, no matter how much
  system RAM is idle. Measured on this machine at the moment of a failure:
  adapter at 5,865 MB of 5,965, i.e. 98%, with 6 GB of RAM going spare.

So closing other applications does not help unless they are themselves using
the GPU, and "available memory" in the trainer's log was never the number to
watch. This module reports the one that is.

Read through PDH, the performance-counter API, because ``torch_directml``
exposes no memory query of its own (no ``memory_allocated``, no
``empty_cache``, no ``synchronize`` -- checked against 0.2.5.dev240914). The
query handle is opened once and reused: the first call costs ~230 ms of
counter setup and every call after it ~0.1 ms.

Windows publishes no counter for the *limit*, only for usage, so
``SHARED_LIMIT_MB`` is derived from the documented WDDM policy rather than
read from the driver. Treat it as a reference line, not a measurement.
"""

from __future__ import annotations

import ctypes
import logging
import os
import time
from ctypes import wintypes

logger = logging.getLogger(__name__)

_MB = 1024.0**2

PDH_FMT_LARGE = 0x00000400
_PDH_MORE_DATA = 0x800007D2

# Counter for this adapter's total commitment across every process.
_ADAPTER_PATH = r"\GPU Adapter Memory(*)\Total Committed"
# Per-process, so the trainer's own share can be told from the desktop's.
_PROCESS_PATH = r"\GPU Process Memory(*)\Total Committed"

# Warn with a step or two left rather than after the fact.
LOW_WATER_FRACTION = 0.92
WARN_INTERVAL_SECONDS = 30.0
_last_warned = 0.0


class _CounterValueItem(ctypes.Structure):
    """PDH_FMT_COUNTERVALUE_ITEM_W.

    The explicit pad matters: the value union is 8-byte aligned, so without
    it every ``largeValue`` is read from the wrong offset and the numbers
    look plausible while being wrong.
    """

    _fields_ = [
        ("szName", wintypes.LPWSTR),
        ("CStatus", wintypes.DWORD),
        ("_pad", ctypes.c_uint32),
        ("largeValue", ctypes.c_longlong),
    ]


def _shared_limit_mb() -> float | None:
    """The WDDM shared-memory ceiling: half of system RAM, by policy."""
    try:
        import psutil

        return psutil.virtual_memory().total / _MB / 2.0
    except Exception:  # noqa: BLE001
        return None


class _Reader:
    """One PDH query, opened lazily and kept open."""

    def __init__(self) -> None:
        self._pdh = None
        self._query = None
        self._counters: dict[str, wintypes.HANDLE] = {}
        self._broken = False

    def _open(self) -> bool:
        if self._broken:
            return False
        if self._query is not None:
            return True
        try:
            self._pdh = ctypes.WinDLL("pdh", use_last_error=True)
            query = wintypes.HANDLE()
            if self._pdh.PdhOpenQueryW(None, 0, ctypes.byref(query)) != 0:
                raise OSError("PdhOpenQueryW failed")
            self._query = query
            for path in (_ADAPTER_PATH, _PROCESS_PATH):
                counter = wintypes.HANDLE()
                status = self._pdh.PdhAddEnglishCounterW(
                    query, path, 0, ctypes.byref(counter)
                )
                if status != 0:
                    raise OSError(
                        f"PdhAddEnglishCounterW({path}) -> "
                        f"0x{status & 0xFFFFFFFF:X}"
                    )
                self._counters[path] = counter
        except Exception:  # noqa: BLE001
            # One warning, then stay quiet: this is a diagnostic, and a
            # machine without these counters must still train.
            logger.warning(
                "GPU memory counters are unavailable; the log will not be "
                "able to say how full the adapter was",
                exc_info=True,
            )
            self._broken = True
            self._query = None
            return False
        return True

    def read(self, path: str) -> list[tuple[str, int]]:
        """(instance name, bytes) for one counter. Empty on any failure."""
        if not self._open():
            return []
        try:
            if self._pdh.PdhCollectQueryData(self._query) != 0:
                return []
            counter = self._counters[path]
            size = wintypes.DWORD(0)
            count = wintypes.DWORD(0)
            status = self._pdh.PdhGetFormattedCounterArrayW(
                counter,
                PDH_FMT_LARGE,
                ctypes.byref(size),
                ctypes.byref(count),
                None,
            )
            if (status & 0xFFFFFFFF) != _PDH_MORE_DATA:
                return []
            buffer = ctypes.create_string_buffer(size.value)
            status = self._pdh.PdhGetFormattedCounterArrayW(
                counter,
                PDH_FMT_LARGE,
                ctypes.byref(size),
                ctypes.byref(count),
                buffer,
            )
            if status != 0:
                return []
            items = ctypes.cast(
                buffer,
                ctypes.POINTER(_CounterValueItem * count.value),
            ).contents
            return [(item.szName, item.largeValue) for item in items]
        except Exception:  # noqa: BLE001
            return []


_reader = _Reader()


def adapter_committed_mb() -> float | None:
    """Everything every process has committed on the GPU, in MB."""
    rows = _reader.read(_ADAPTER_PATH)
    if not rows:
        return None
    return sum(value for _, value in rows) / _MB


def process_committed_mb(pid: int | None = None) -> float | None:
    """This process's own GPU commitment, in MB.

    Filtered in Python rather than through a ``pid_N*`` counter path: PDH
    fails the whole collect when a wildcard matches no instance, and a
    process that has not touched the GPU yet has none.
    """
    rows = _reader.read(_PROCESS_PATH)
    if not rows:
        return None
    prefix = f"pid_{os.getpid() if pid is None else pid}_"
    mine = [value for name, value in rows if name.startswith(prefix)]
    return sum(mine) / _MB if mine else 0.0


def snapshot() -> str:
    """One line describing the pool a DirectML allocation comes from."""
    adapter = adapter_committed_mb()
    if adapter is None:
        return "gpu memory unavailable"
    mine = process_committed_mb()
    limit = _shared_limit_mb()
    text = f"gpu adapter {adapter:.0f} MB"
    if limit:
        text += f" of ~{limit:.0f} ({adapter / limit * 100:.0f}%)"
    if mine is not None:
        text += f"; this process {mine:.0f} MB"
    return text


def warn_if_low(context: str) -> None:
    """Warn while the adapter still has room to report it.

    Rate-limited, because the training loop calls this every step.
    """
    global _last_warned

    adapter = adapter_committed_mb()
    limit = _shared_limit_mb()
    if adapter is None or not limit:
        return
    if adapter < limit * LOW_WATER_FRACTION:
        return
    now = time.monotonic()
    if now - _last_warned < WARN_INTERVAL_SECONDS:
        return
    _last_warned = now
    logger.warning(
        "GPU adapter memory is at %.0f MB of ~%.0f (%.0f%%) at %s. This is "
        "the pool DirectML allocates from -- it is capped at half of system "
        "RAM and free RAM does not extend it. The next allocation may fail; "
        "a restart is what returns it.",
        adapter,
        limit,
        adapter / limit * 100,
        context,
    )
