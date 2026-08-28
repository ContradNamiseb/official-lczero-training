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

# ``ctypes.wintypes`` raises ValueError at import time on anything that is
# not Windows, so it cannot sit at module scope: training.py imports this
# module unconditionally, and a Linux/CUDA run would die here before it ever
# reached a device. The PDH reader below is the only user, and it already
# fails soft.
try:
    from ctypes import wintypes
except (ImportError, ValueError):  # pragma: no cover - platform dependent
    wintypes = None  # type: ignore[assignment]

import torch

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


# Defined only where wintypes exists. A ctypes.Structure evaluates its
# _fields_ at class-creation time, so this cannot be written defensively
# inside the class -- referencing wintypes.LPWSTR with wintypes set to None
# raises AttributeError during import, which is exactly the failure this
# module must not have on Linux.
#
# Leaving the name undefined off-Windows is safe because its only use is
# inside _Reader.read(), which is unreachable there: _open() returns False
# when wintypes is None and read() short-circuits on that.
if wintypes is not None:

    class _CounterValueItem(ctypes.Structure):
        """PDH_FMT_COUNTERVALUE_ITEM_W.

        The explicit pad matters: the value union is 8-byte aligned, so
        without it every ``largeValue`` is read from the wrong offset and
        the numbers look plausible while being wrong.
        """

        _fields_ = [
            ("szName", wintypes.LPWSTR),
            ("CStatus", wintypes.DWORD),
            ("_pad", ctypes.c_uint32),
            ("largeValue", ctypes.c_longlong),
        ]


def _shared_limit_mb() -> float | None:
    """The memory ceiling an allocation on this device comes out of.

    On DirectML this is the WDDM shared-memory carve-out -- half of system
    RAM by policy, derived rather than measured because Windows publishes no
    counter for it. On CUDA it is real dedicated VRAM, reported by the
    driver, so it is a measurement rather than a reference line.
    """
    # Defined before _cuda_active() textually, so the check is inlined here
    # rather than reaching forward to a helper that is not bound yet at
    # import time. (It would be by call time, but the forward reference
    # makes the file harder to read than the duplicated condition does.)
    if torch.cuda.is_available() and torch.cuda.device_count() > 0:
        try:
            return torch.cuda.mem_get_info()[1] / _MB
        except Exception:  # noqa: BLE001
            return None
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
        if wintypes is None:
            self._broken = True
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


# --------------------------------------------------------------- CUDA path
#
# Everything above this line reads PDH because torch_directml exposes no
# memory query at all. CUDA does, so on that backend the same three numbers
# come straight from the driver and the counter machinery is bypassed.
#
# The mapping is not quite one to one, and the difference matters when
# reading a log from both backends:
#
#   adapter_committed_mb   total - free from cudaMemGetInfo. Device-wide and
#                          across processes, which is what the PDH adapter
#                          counter measures too.
#   process_committed_mb   torch.cuda.memory_reserved(), i.e. what this
#                          process's caching allocator holds. Deliberately
#                          NOT memory_allocated(): reserved is the number
#                          comparable to a DirectML "commitment", since
#                          cached-but-free blocks are still held from the
#                          driver's point of view.
#   _shared_limit_mb       real VRAM, a hard number -- unlike the DirectML
#                          case, where the limit is derived from WDDM policy
#                          and is only a reference line.
#
# On CUDA the low-water warning is much less interesting than it is on
# DirectML: the caching allocator reclaims, so sitting at 92% is normal
# steady state rather than a symptom. It is left enabled because a genuine
# approach to the cap still precedes an OOM, but do not read it as the
# restart signal it is on the iGPU.


def _cuda_active() -> bool:
    """Whether to answer from CUDA rather than from PDH."""
    return torch.cuda.is_available() and torch.cuda.device_count() > 0


def _cuda_mem_info() -> tuple[float, float] | None:
    """(free_mb, total_mb) for the current CUDA device."""
    try:
        free, total = torch.cuda.mem_get_info()
        return free / _MB, total / _MB
    except Exception:  # noqa: BLE001 - diagnostics must never break a step
        return None


def adapter_committed_mb() -> float | None:
    """Everything every process has committed on the GPU, in MB."""
    if _cuda_active():
        info = _cuda_mem_info()
        return None if info is None else info[1] - info[0]
    rows = _reader.read(_ADAPTER_PATH)
    if not rows:
        return None
    return sum(value for _, value in rows) / _MB


def process_committed_mb(pid: int | None = None) -> float | None:
    """This process's own GPU commitment, in MB.

    Filtered in Python rather than through a ``pid_N*`` counter path: PDH
    fails the whole collect when a wildcard matches no instance, and a
    process that has not touched the GPU yet has none.

    The ``pid`` argument is meaningless on CUDA -- the allocator can only
    report on its own process -- so it is ignored there rather than
    pretending to honour it.
    """
    if _cuda_active():
        try:
            return torch.cuda.memory_reserved() / _MB
        except Exception:  # noqa: BLE001
            return None
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

    Silent on CUDA, and that is not an oversight. The number this compares
    against the cap is driver-level usage, which on CUDA includes everything
    PyTorch's caching allocator is holding -- and the allocator does not
    return blocks to the driver unless empty_cache() is called. A healthy
    CUDA run therefore climbs to near the cap and stays there, so the
    warning would fire every 30 seconds forever while describing nothing.
    The advice in the message is DirectML's, too: on that backend a restart
    really is the only thing that returns the memory, whereas on CUDA the
    allocator reuses it. A genuine CUDA OOM raises, loudly, at the
    allocation site.
    """
    global _last_warned

    if _cuda_active():
        return

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
