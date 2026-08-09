"""The GPU memory reader -- the pool that actually runs out.

Written after a crash report that made no sense until the right pool was
measured: "Not enough memory resources are available" with 6.16 GB of system
RAM free and the process at 1.70 GB RSS. The adapter was at 98% of its cap.

Most of this runs anywhere. The reads themselves only return numbers on a
Windows box with a GPU, so those assert the contract (a float or None, never
an exception) rather than a value.
"""

import ctypes
import logging
import sys

import pytest

from lczero_training.directml import gpu_memory


def test_the_counter_item_layout_matches_win32():
    """A mis-sized field here does not fail loudly -- it reads largeValue
    from the wrong offset and reports plausible, wrong megabytes. The union
    is 8-byte aligned after a DWORD status, which is what the pad is for."""
    if ctypes.sizeof(ctypes.c_void_p) != 8:
        pytest.skip("layout below is the 64-bit one")

    assert ctypes.sizeof(gpu_memory._CounterValueItem) == 24
    assert gpu_memory._CounterValueItem.szName.offset == 0
    assert gpu_memory._CounterValueItem.CStatus.offset == 8
    assert gpu_memory._CounterValueItem.largeValue.offset == 16


def test_reads_return_a_number_or_none_but_never_raise():
    """This runs on the out-of-memory path. A diagnostic that throws while
    the trainer is dying replaces the evidence with its own traceback."""
    for value in (
        gpu_memory.adapter_committed_mb(),
        gpu_memory.process_committed_mb(),
    ):
        assert value is None or isinstance(value, float)

    assert isinstance(gpu_memory.snapshot(), str)
    gpu_memory.warn_if_low("test")  # must not raise


def test_a_broken_reader_degrades_instead_of_failing(monkeypatch):
    """No PDH, no GPU counters, a locked-down machine: still trains."""
    monkeypatch.setattr(gpu_memory, "_reader", gpu_memory._Reader())
    monkeypatch.setattr(gpu_memory._reader, "_broken", True)

    assert gpu_memory.adapter_committed_mb() is None
    assert gpu_memory.process_committed_mb() is None
    assert gpu_memory.snapshot() == "gpu memory unavailable"
    gpu_memory.warn_if_low("broken reader")


def test_it_warns_near_the_cap(monkeypatch, caplog):
    monkeypatch.setattr(gpu_memory, "adapter_committed_mb", lambda: 5800.0)
    monkeypatch.setattr(gpu_memory, "_shared_limit_mb", lambda: 5965.0)
    monkeypatch.setattr(gpu_memory, "_last_warned", 0.0)

    with caplog.at_level(logging.WARNING):
        gpu_memory.warn_if_low("step 195743")

    assert any("5800" in record.message for record in caplog.records)
    assert any("step 195743" in record.message for record in caplog.records)
    # The whole point of the message: free RAM is not the lever.
    assert any(
        "half of system RAM" in record.message for record in caplog.records
    )


def test_it_stays_quiet_with_headroom(monkeypatch, caplog):
    monkeypatch.setattr(gpu_memory, "adapter_committed_mb", lambda: 1200.0)
    monkeypatch.setattr(gpu_memory, "_shared_limit_mb", lambda: 5965.0)
    monkeypatch.setattr(gpu_memory, "_last_warned", 0.0)

    with caplog.at_level(logging.WARNING):
        gpu_memory.warn_if_low("step 100")

    assert not caplog.records


def test_the_warning_is_rate_limited(monkeypatch, caplog):
    """Called every step, so once it trips it would otherwise bury the log."""
    monkeypatch.setattr(gpu_memory, "adapter_committed_mb", lambda: 5900.0)
    monkeypatch.setattr(gpu_memory, "_shared_limit_mb", lambda: 5965.0)
    monkeypatch.setattr(gpu_memory, "_last_warned", 0.0)

    with caplog.at_level(logging.WARNING):
        for step in range(200):
            gpu_memory.warn_if_low(f"step {step}")

    assert len(caplog.records) == 1


@pytest.mark.skipif(sys.platform != "win32", reason="PDH is Windows-only")
def test_the_real_counters_resolve_on_this_machine():
    """The wildcard paths and the sizing dance, against the live counters.

    Skipped off Windows. On a Windows box without a GPU the reader degrades
    to None, which is also a pass -- the failure this catches is a counter
    path that silently stopped resolving.
    """
    adapter = gpu_memory.adapter_committed_mb()
    if adapter is None:
        pytest.skip("no GPU performance counters on this machine")

    assert adapter > 0, "an adapter with zero committed bytes is implausible"
    mine = gpu_memory.process_committed_mb()
    assert mine is not None and mine >= 0.0
    assert "gpu adapter" in gpu_memory.snapshot()
