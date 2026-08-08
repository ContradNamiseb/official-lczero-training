"""The memory readings every OOM post-mortem in this port has lacked.

This code runs on the failure path, so the property that matters most is
that it cannot itself raise: a diagnostic that throws while the trainer is
already dying would replace the evidence with its own traceback.
"""

import logging

from lczero_training.directml import host_memory


def test_snapshot_reports_the_figure_that_actually_fails():
    """Available physical memory, not commit. GPU-shared memory has to be
    resident, so an allocation fails when physical runs short even with
    pagefile to spare -- runs here have died with 238 MB free."""
    line = host_memory.snapshot()

    assert "available" in line
    assert "rss" in line and "commit" in line


def test_snapshot_never_raises_when_psutil_is_unusable(monkeypatch):
    """Called from the out-of-memory handler, where anything may be broken."""
    import builtins

    real_import = builtins.__import__

    def refuse(name, *args, **kwargs):
        if name == "psutil":
            raise ImportError("no psutil here")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", refuse)

    assert host_memory.snapshot() == "memory unavailable"
    assert host_memory.available_gb() is None
    host_memory.warn_if_low("with no psutil")  # must not raise


def test_it_warns_below_the_low_water_mark(monkeypatch, caplog):
    monkeypatch.setattr(host_memory, "available_gb", lambda: 0.2)
    monkeypatch.setattr(host_memory, "_last_warned", 0.0)

    with caplog.at_level(logging.WARNING):
        host_memory.warn_if_low("step 27")

    assert any("0.20 GB" in record.message for record in caplog.records)
    assert any("step 27" in record.message for record in caplog.records)


def test_it_stays_quiet_when_there_is_headroom(monkeypatch, caplog):
    monkeypatch.setattr(host_memory, "available_gb", lambda: 4.5)
    monkeypatch.setattr(host_memory, "_last_warned", 0.0)

    with caplog.at_level(logging.WARNING):
        host_memory.warn_if_low("step 27")

    assert not caplog.records


def test_the_warning_is_rate_limited(monkeypatch, caplog):
    """The training loop calls this every step, because the launches it has
    to explain died at steps 7, 9 and 27 -- far inside the 250-step logging
    cadence. Warning every step once it trips would bury the log it exists
    to make readable."""
    monkeypatch.setattr(host_memory, "available_gb", lambda: 0.1)
    monkeypatch.setattr(host_memory, "_last_warned", 0.0)

    with caplog.at_level(logging.WARNING):
        for step in range(200):
            host_memory.warn_if_low(f"step {step}")

    assert len(caplog.records) == 1, (
        f"200 steps under the mark produced {len(caplog.records)} warnings"
    )
