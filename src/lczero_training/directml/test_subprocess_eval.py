"""Evaluation in a child process.

The end-to-end test really does spawn the worker, on the CPU with a tiny
model, because the parts worth doubting are the ones a fake cannot cover:
whether the request the trainer writes is one the worker can read, and
whether the scalars come back. The cheap tests around it cover the failure
paths, which must never take a training run down with them.
"""

import json

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from lczero_training.directml import subprocess_eval
from lczero_training.directml.conftest import make_tiny_config


@pytest.fixture(autouse=True)
def _bypass_memory_guard(monkeypatch, request):
    """Most tests here exercise the spawn mechanics, not the environmental
    low-memory skip. They run on machines whose own free memory is whatever
    it is -- the guard would fire on real-spawn tests for reasons that have
    nothing to do with the code under test, exactly as it fired here once:
    a CI box with 1.09 GB free where a tiny CPU worker spawns fine. Tests
    that *do* want the guard mark themselves via ``request.node`` markers
    or by not opting out below; the default for this module is bypass.

    Importing here keeps the bypass in lockstep with the import: if a new
    test forgets to override the guard it falls under the fixture."""
    from lczero_training.directml import host_memory

    marker = request.node.get_closest_marker("enable_memory_guard")
    if marker is None:
        monkeypatch.setattr(host_memory, "available_gb", lambda: None)


class _FakeLoader:
    """Yields batches shaped like the real loader's ``test`` output.

    Records the aliases it was asked for: the trainer must pull from the
    held-out split, and reading ``train`` here would silently evaluate on
    the training set.
    """

    def __init__(self, batch: int = 2):
        self.batch = batch
        self.aliases: list[str] = []
        self._rng = np.random.default_rng(0)

    def get_next(self, alias=""):
        self.aliases.append(alias)
        probabilities = self._rng.random((self.batch, 1858), dtype=np.float32)
        probabilities /= probabilities.sum(axis=1, keepdims=True)
        return (
            self._rng.random((self.batch, 112, 8, 8), dtype=np.float32),
            probabilities,
            self._rng.random((self.batch, 6, 3), dtype=np.float32),
        )


def test_batches_round_trip_through_the_request_file(tmp_path):
    loader = _FakeLoader(batch=3)
    path = tmp_path / "batches.npz"

    written = subprocess_eval.write_batches(loader, "test", 4, path)
    arrays, count = subprocess_eval.read_batches(path)
    recovered = list(arrays)

    assert written == 4
    assert count == 4
    assert loader.aliases == ["test"] * 4, "must read the held-out split"
    assert [tuple(a.shape for a in item) for item in recovered] == [
        ((3, 112, 8, 8), (3, 1858), (3, 6, 3))
    ] * 4


def _evaluate_through_a_real_worker(tmp_path, monkeypatch, device_spec):
    """Spawn the worker for real and return what came back."""
    from lczero_training.directml import host_memory
    from lczero_training.directml.model import LczeroModel

    config = make_tiny_config()
    config.name = "evaltest"
    config.metrics.tensorboard_path = str(tmp_path / "tensorboard")

    relayed: list[tuple[int, dict]] = []
    monkeypatch.setattr(
        subprocess_eval, "work_directory", lambda _: tmp_path / "work"
    )
    hook = subprocess_eval.make_eval_hook(
        config=config,
        model=LczeroModel(config.model),
        loader=_FakeLoader(batch=2),
        config_filepath=str(tmp_path / "config.textproto"),
        batch_count=2,
        device_spec=device_spec,
        kda_chunk_size=8,
        timeout=600.0,
        on_scalars=lambda step, scalars: relayed.append((step, scalars)),
    )
    hook(7000)
    return relayed


def _assert_a_usable_evaluation(relayed, tmp_path):
    assert relayed, "the worker produced no scalars"
    step, scalars = relayed[0]
    assert step == 7000
    assert "total" in scalars
    assert all(isinstance(value, float) for value in scalars.values())
    # The gate diagnostics have to survive the move out of process: the test
    # run is where KDA health is read from, and the mixers only capture them
    # when the worker asks.
    assert any(name.startswith("KDA/block") for name in scalars), sorted(
        scalars
    )
    assert (tmp_path / "work" / subprocess_eval.RESULT_FILE).exists()
    assert list((tmp_path / "tensorboard").glob("evaltest-test/*")), (
        "the worker must write the -test run itself"
    )


def test_the_worker_evaluates_and_reports_back(tmp_path, monkeypatch):
    """The whole path: request written here, worker spawned, scalars back.

    On the CPU, so it runs on a machine with no adapter at all.
    """
    relayed = _evaluate_through_a_real_worker(tmp_path, monkeypatch, "cpu")
    _assert_a_usable_evaluation(relayed, tmp_path)


def test_the_worker_evaluates_on_the_directml_device(
    tmp_path, monkeypatch, dml_device
):
    """The same path on the device it exists for.

    Worth its own test rather than trusting the CPU one: the worker performs
    the eager ``torch_directml`` import and the device resolution itself, and
    neither is exercised by ``--device cpu``. Skipped where there is no
    adapter.
    """
    relayed = _evaluate_through_a_real_worker(
        tmp_path, monkeypatch, f"directml:{dml_device.index or 0}"
    )
    _assert_a_usable_evaluation(relayed, tmp_path)


def test_importing_the_worker_does_not_import_torch():
    """The regression that cost fifteen minutes of a real run.

    The worker used to import ``directml.device`` -- and so torch and
    torch_directml -- at module scope, before ``basicConfig``. On a machine
    with the trainer holding 3.8 GB of shared memory that import did not
    return inside the 900 s timeout, and with no logging configured the
    worker was killed having emitted nothing at all: a stall with no
    evidence in it. Every heavy import has to happen inside ``main()``,
    after logging exists, and this is the only way to state that as a rule
    rather than a comment.

    Checked in a child interpreter because torch is certainly already
    imported in this one.
    """
    import subprocess
    import sys

    probe = (
        "import sys;"
        "import lczero_training.commands.directml_eval_worker;"
        "print(int('torch' in sys.modules))"
    )
    result = subprocess.run(
        [sys.executable, "-c", probe],
        capture_output=True,
        text=True,
        check=True,
    )
    assert result.stdout.strip() == "0", (
        "importing the eval worker pulled torch in at module scope; move it "
        "inside main(), after logging is configured"
    )


def test_the_worker_logs_each_stage_to_its_own_file(tmp_path, monkeypatch):
    """A worker killed on timeout may never reach the trainer's stderr, so
    the stage log has to be on disk -- and has to start before the import
    that is most likely to hang."""
    relayed = _evaluate_through_a_real_worker(tmp_path, monkeypatch, "cpu")
    assert relayed, "sanity: the worker should have succeeded"

    recorded = (tmp_path / "work" / subprocess_eval.WORKER_LOG_FILE).read_text(
        encoding="utf-8"
    )
    assert "Importing torch" in recorded
    assert "Evaluating" in recorded
    # The order is the point: a log line has to precede the risky import, not
    # follow it.
    assert recorded.index("starting in") < recorded.index("Importing torch")


def test_a_failing_worker_does_not_stop_training(tmp_path, monkeypatch, caplog):
    """Evaluation is observational. Nothing it does is worth a run."""
    import logging
    import subprocess

    config = make_tiny_config()
    model = torch.nn.Linear(2, 2)  # never used: the spawn fails first

    monkeypatch.setattr(
        subprocess_eval, "work_directory", lambda _: tmp_path / "work"
    )
    monkeypatch.setattr(
        subprocess,
        "run",
        lambda *a, **k: subprocess.CompletedProcess([], 3),
    )

    hook = subprocess_eval.make_eval_hook(
        config=config,
        model=model,
        loader=_FakeLoader(batch=2),
        config_filepath=str(tmp_path / "config.textproto"),
        batch_count=1,
        on_scalars=lambda step, scalars: pytest.fail("should not report"),
    )

    with caplog.at_level(logging.ERROR):
        hook(1234)  # must return normally

    assert any(
        "Evaluation at step 1234 failed" in record.message
        for record in caplog.records
    )


class _FakeHungProcess:
    """Stands in for a Popen handle that never returns.

    Real enough for spawn_and_wait to work with: a pid, a wait() that always
    times out, and a kill()/wait() pair for the cleanup path. Fake enough
    that no process is actually spawned, so the test stays fast and
    deterministic -- the same reason the old test faked subprocess.run.
    """

    def __init__(self, pid: int):
        self.pid = pid

    def wait(self, timeout=None):
        if timeout is not None:
            import subprocess

            raise subprocess.TimeoutExpired(cmd="worker", timeout=timeout)
        return 0  # the final, post-kill wait() in the cleanup path

    def kill(self):
        pass


def test_a_hung_worker_is_killed_and_the_eval_skipped(
    tmp_path, monkeypatch, caplog
):
    import logging

    config = make_tiny_config()
    spawned = []

    def fake_popen(command, **kwargs):
        process = _FakeHungProcess(pid=len(spawned) + 1000)
        spawned.append(process)
        return process

    monkeypatch.setattr(
        subprocess_eval, "work_directory", lambda _: tmp_path / "work"
    )
    monkeypatch.setattr(subprocess_eval.subprocess, "Popen", fake_popen)

    hook = subprocess_eval.make_eval_hook(
        config=config,
        model=torch.nn.Linear(2, 2),
        loader=_FakeLoader(batch=2),
        config_filepath=str(tmp_path / "config.textproto"),
        batch_count=1,
        timeout=1.0,
        retries=0,
        on_scalars=lambda step, scalars: pytest.fail("should not report"),
    )

    with caplog.at_level(logging.ERROR):
        hook(99)

    assert len(spawned) == 1, "retries=0 must mean exactly one attempt"
    assert any("timed out" in record.message for record in caplog.records)
    # Its PID is known even though it never finished -- the whole point of
    # spawning through Popen instead of subprocess.run(timeout=...).
    assert any(
        str(spawned[0].pid) in record.message for record in caplog.records
    )
    # No worker actually ran, so there is no log to quote -- but the message
    # must say so rather than leaving the reader wondering where to look.
    assert any(
        "no log" in record.message or "empty" in record.message
        for record in caplog.records
    ), "a timeout must point at the worker's log, or say there is none"


def test_a_hung_worker_is_retried_before_giving_up(
    tmp_path, monkeypatch, caplog
):
    """The default: one retry, so two attempts before the eval is skipped."""
    import logging

    config = make_tiny_config()
    spawned = []

    def fake_popen(command, **kwargs):
        process = _FakeHungProcess(pid=len(spawned) + 2000)
        spawned.append(process)
        return process

    monkeypatch.setattr(
        subprocess_eval, "work_directory", lambda _: tmp_path / "work"
    )
    monkeypatch.setattr(subprocess_eval.subprocess, "Popen", fake_popen)

    hook = subprocess_eval.make_eval_hook(
        config=config,
        model=torch.nn.Linear(2, 2),
        loader=_FakeLoader(batch=2),
        config_filepath=str(tmp_path / "config.textproto"),
        batch_count=1,
        timeout=1.0,
        # retries left at the default (1).
        on_scalars=lambda step, scalars: pytest.fail("should not report"),
    )

    with caplog.at_level(logging.ERROR):
        hook(99)

    assert len(spawned) == 2, "one retry means two spawn attempts"
    assert any(
        "did not succeed in 2 attempt" in record.message
        for record in caplog.records
    )


def test_a_worker_that_succeeds_on_retry_still_reports(tmp_path, monkeypatch):
    """The point of retrying: a transient failure must not cost the eval."""
    import json

    config = make_tiny_config()
    work = tmp_path / "work"
    work.mkdir()
    calls = {"n": 0}

    class _FlakyThenFine:
        """Times out on the first real attempt, succeeds on the second.

        ``wait()`` is also called once more per attempt with no timeout, to
        reap the process after ``kill()`` -- that call must always return
        promptly and must not itself count as an attempt.
        """

        def __init__(self, pid):
            self.pid = pid

        def wait(self, timeout=None):
            if timeout is None:
                return 0  # the post-kill reap
            calls["n"] += 1
            if calls["n"] == 1:
                import subprocess

                raise subprocess.TimeoutExpired(cmd="worker", timeout=timeout)
            (work / subprocess_eval.RESULT_FILE).write_text(
                json.dumps({"total": 1.5})
            )
            return 0

        def kill(self):
            pass

    monkeypatch.setattr(subprocess_eval, "work_directory", lambda _: work)
    monkeypatch.setattr(
        subprocess_eval.subprocess,
        "Popen",
        lambda command, **kwargs: _FlakyThenFine(pid=42),
    )

    reported = []
    hook = subprocess_eval.make_eval_hook(
        config=config,
        model=torch.nn.Linear(2, 2),
        loader=_FakeLoader(batch=2),
        config_filepath=str(tmp_path / "config.textproto"),
        batch_count=1,
        timeout=1.0,
        on_scalars=lambda step, scalars: reported.append((step, scalars)),
    )

    hook(99)

    assert calls["n"] == 2, "must have retried exactly once"
    assert reported == [(99, {"total": 1.5})]


def test_a_stale_result_is_never_reported_as_a_fresh_one(tmp_path, monkeypatch):
    """A worker that dies without writing must not hand back the last eval's
    numbers -- they would land on the chart at the wrong step."""
    import subprocess

    config = make_tiny_config()
    work = tmp_path / "work"
    work.mkdir()
    (work / subprocess_eval.RESULT_FILE).write_text(json.dumps({"total": 1.0}))

    monkeypatch.setattr(subprocess_eval, "work_directory", lambda _: work)
    monkeypatch.setattr(
        subprocess,
        "run",
        lambda *a, **k: subprocess.CompletedProcess([], 0),
    )

    hook = subprocess_eval.make_eval_hook(
        config=config,
        model=torch.nn.Linear(2, 2),
        loader=_FakeLoader(batch=2),
        config_filepath=str(tmp_path / "config.textproto"),
        batch_count=1,
        on_scalars=lambda step, scalars: pytest.fail("should not report"),
    )

    hook(500)

    assert not (work / subprocess_eval.RESULT_FILE).exists()


@pytest.mark.enable_memory_guard
def test_low_memory_skips_eval_without_attempting_to_spawn(
    tmp_path, monkeypatch, caplog
):
    """The hung-eval failure mode: the worker's python hangs at interpreter
    startup once system memory is too low for a second ``import torch``
    next to the trainer. Spawning in that state parks for the full
    timeout and is killed -- exactly the 22 hangs in a row in
    ``train-new.log`` before this guard existed. Skip up-front with a
    clear log line instead of paying that."""
    import logging

    config = make_tiny_config()
    spawned = []
    monkeypatch.setattr(
        subprocess_eval.subprocess,
        "Popen",
        lambda *a, **k: spawned.append(1) or pytest.fail("must not spawn"),
    )
    monkeypatch.setattr(
        subprocess_eval, "work_directory", lambda _: tmp_path / "work"
    )
    from lczero_training.directml import host_memory

    monkeypatch.setattr(host_memory, "available_gb", lambda: 1.2)
    monkeypatch.setattr(
        host_memory, "snapshot", lambda: "available 1.20 GB of 11.65"
    )

    hook = subprocess_eval.make_eval_hook(
        config=config,
        model=torch.nn.Linear(2, 2),
        loader=_FakeLoader(batch=2),
        config_filepath=str(tmp_path / "config.textproto"),
        batch_count=1,
        on_scalars=lambda step, scalars: pytest.fail("should not report"),
    )

    with caplog.at_level(logging.WARNING):
        hook(7001)

    assert spawned == [], "low memory must skip the spawn, not pay the timeout"
    assert any(
        "Skipping evaluation at step 7001" in record.message
        and "1.20 GB" in record.message
        for record in caplog.records
    ), "the skip reason must say which step and how much memory"


@pytest.mark.enable_memory_guard
def test_prepare_truncates_a_stale_worker_log_so_worker_tail_does_not_lie(
    tmp_path, monkeypatch, caplog
):
    """A worker that hangs at interpreter startup never reaches its own
    ``FileHandler(mode="w")``, so the parent has to truncate ``worker.log``
    itself. Otherwise ``worker_tail`` would report the previous successful
    eval's log lines as if the new worker had reached them -- which is
    exactly the misleading ``I0808 18:12 ... Evaluated step 195000`` block
    that appeared under every hung-eval post-mortem for a full week.

    Reproduces by leaving stale content behind, running the hook under the
    memory-starved path (which exits before any spawn), and confirming the
    file is left empty rather than stale."""
    import logging

    config = make_tiny_config()
    work = tmp_path / "work"
    work.mkdir()
    stale = "I9999 99:99:99 ... Evaluated step 12345 over 50 batch(es) ...\n"
    (work / subprocess_eval.WORKER_LOG_FILE).write_text(stale, encoding="utf-8")
    monkeypatch.setattr(subprocess_eval, "work_directory", lambda _: work)
    from lczero_training.directml import host_memory

    monkeypatch.setattr(host_memory, "available_gb", lambda: 0.4)

    hook = subprocess_eval.make_eval_hook(
        config=config,
        model=torch.nn.Linear(2, 2),
        loader=_FakeLoader(batch=2),
        config_filepath=str(tmp_path / "config.textproto"),
        batch_count=1,
        on_scalars=lambda step, scalars: pytest.fail("should not report"),
    )

    with caplog.at_level(logging.WARNING):
        hook(8000)

    # Stale content has to be gone before the next "worker got this far"
    # message can be believed. Empty (rather than absent) is fine: the
    # FileHandler would have truncated to empty if the worker had reached
    # it, and the diagnostic's empty-log branch handles that case honestly.
    fresh = (work / subprocess_eval.WORKER_LOG_FILE).read_text(
        encoding="utf-8"
    )
    assert "Evaluated step" not in fresh, (
        "the stale line from the previous successful eval must not survive "
        "into the next attempt's worker.log; a hung interpreter never "
        "truncates it itself, so prepare() does"
    )
    assert fresh == "", "prepare() leaves the file empty, not absent"
