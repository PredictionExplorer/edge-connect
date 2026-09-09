import fcntl
from concurrent.futures import ThreadPoolExecutor
import hashlib
from pathlib import Path
import sqlite3
import threading
from types import SimpleNamespace

import pytest
import torch

import startrain.actor as actor_module
import startrain.replay_store as replay_module
from startrain.config import load_config
from startrain.replay_store import ReplayStore, ReplayStoreCancelled
from test_pipeline_core import append_replay, make_replay_sample, run_identity


def replay_files(tmp_path, count=2):
    root = tmp_path / "replay"
    identity = run_identity(tmp_path)
    records = []
    with ReplayStore(root) as store:
        generation = store.lease_generation(identity, "actor-test")
        for index in range(count):
            records.append(
                append_replay(
                    store,
                    [
                        make_replay_sample(
                            identity=identity,
                            generation=generation,
                            game_id=f"cancel-{index}",
                        )
                    ],
                    identity,
                    generation=generation,
                    model_step=0,
                )
            )
    return root, records


def test_already_cancelled_startup_does_not_create_a_store(tmp_path):
    root = tmp_path / "not-created"
    with pytest.raises(ReplayStoreCancelled):
        ReplayStore(root, cancel_requested=lambda: True)
    assert not root.exists()


def test_startup_cancels_while_waiting_for_an_owned_reconciliation_lock(
    tmp_path, monkeypatch
):
    root, _ = replay_files(tmp_path)
    cancellation = threading.Event()
    waiting = threading.Event()
    failure = []
    lock = (root / ".reconcile.lock").open("r+")
    original_flock = fcntl.flock
    original_flock(lock.fileno(), fcntl.LOCK_EX)

    def observed_flock(descriptor, operation):
        if operation & fcntl.LOCK_NB:
            waiting.set()
        return original_flock(descriptor, operation)

    monkeypatch.setattr(fcntl, "flock", observed_flock)

    def open_store():
        store = object.__new__(ReplayStore)
        try:
            ReplayStore.__init__(store, root, cancel_requested=cancellation.is_set)
        except BaseException as error:
            failure.append(error)
            with pytest.raises(sqlite3.ProgrammingError, match="closed"):
                store.connection.execute("SELECT 1")

    worker = threading.Thread(target=open_store)
    worker.start()
    try:
        assert waiting.wait(5), "startup never reached the held flock"
        cancellation.set()
        worker.join(timeout=2)
        assert not worker.is_alive(), "cancelled startup remained blocked on flock"
        assert len(failure) == 1 and isinstance(failure[0], ReplayStoreCancelled)
    finally:
        cancellation.set()
        original_flock(lock.fileno(), fcntl.LOCK_UN)
        lock.close()
        worker.join(timeout=5)


class ObservedReader:
    def __init__(self, stream, cancel, sizes):
        self.stream, self.cancel, self.sizes = stream, cancel, sizes

    def __enter__(self):
        return self

    def __exit__(self, *args):
        return self.stream.__exit__(*args)

    def read(self, size):
        data = self.stream.read(size)
        self.sizes.append(len(data))
        self.cancel.set()
        return data


def cancel_on_read(monkeypatch, path, cancellation, sizes):
    original_open = Path.open

    def observed_open(selected, mode="r", *args, **kwargs):
        stream = original_open(selected, mode, *args, **kwargs)
        if selected == path and mode == "rb":
            return ObservedReader(stream, cancellation, sizes)
        return stream

    monkeypatch.setattr(Path, "open", observed_open)


def test_hash_cancellation_is_checked_between_bounded_chunks(tmp_path, monkeypatch):
    path = tmp_path / "large-shard"
    data = b"bounded-integrity" * (200_000)
    path.write_bytes(data)
    assert replay_module._sha256(path) == hashlib.sha256(data).hexdigest()
    cancel = threading.Event()
    reads = []
    cancel_on_read(monkeypatch, path, cancel, reads)
    with pytest.raises(ReplayStoreCancelled):
        replay_module._sha256(path, cancel_requested=cancel.is_set)
    assert reads == [1024 * 1024]


def test_cancelled_scan_closes_store_and_preserves_completed_quarantine_repairs(
    tmp_path, monkeypatch
):
    root, records = replay_files(tmp_path)
    records[0].path.write_bytes(b"corrupt data")
    marker = root / "restore-marker.json"
    marker.write_text("{}")
    cancel = threading.Event()
    reads = []
    cancel_on_read(monkeypatch, records[1].path, cancel, reads)
    store = object.__new__(ReplayStore)
    with pytest.raises(ReplayStoreCancelled):
        ReplayStore.__init__(store, root, cancel_requested=cancel.is_set)
    assert reads
    with pytest.raises(sqlite3.ProgrammingError, match="closed"):
        store.connection.execute("SELECT 1")
    assert marker.exists()
    with sqlite3.connect(root / "manifest.sqlite3") as connection:
        rows = connection.execute(
            "SELECT state, relative_path FROM shards ORDER BY id"
        ).fetchall()
    assert rows[0][0] == "quarantined"
    assert (root / rows[0][1]).read_bytes() == b"corrupt data"
    assert rows[1][0] == "ready"
    monkeypatch.undo()
    with ReplayStore(root, cancel_requested=lambda: False) as reopened:
        assert (
            reopened.connection.execute(
                "SELECT COUNT(*) FROM shards WHERE state='ready'"
            ).fetchone()[0]
            == 1
        )
    assert not marker.exists()


@pytest.mark.parametrize("cancellable", [False, True])
def test_successful_open_still_hashes_every_ready_shard_and_quarantines_corruption(
    tmp_path, monkeypatch, cancellable
):
    root, records = replay_files(tmp_path, count=3)
    records[1].path.write_bytes(b"changed after commit")
    original_hash = replay_module._sha256
    scanned = []

    def observe(path, **kwargs):
        scanned.append(path)
        return original_hash(path, **kwargs)

    monkeypatch.setattr(replay_module, "_sha256", observe)
    options = {"cancel_requested": lambda: False} if cancellable else {}
    with ReplayStore(root, **options) as store:
        assert set(scanned) == {record.path for record in records}
        assert store.reconciliation_metrics["corrupt_committed"] == 1
        assert (
            store.connection.execute(
                "SELECT COUNT(*) FROM shards WHERE state='ready'"
            ).fetchone()[0]
            == 2
        )


def test_explicit_reconciliation_cancellation_also_invalidates_the_store(tmp_path):
    root, _ = replay_files(tmp_path)
    store = ReplayStore(root)
    with pytest.raises(ReplayStoreCancelled):
        store.reconcile_orphans(cancel_requested=lambda: True)
    with pytest.raises(sqlite3.ProgrammingError, match="closed"):
        store.connection.execute("SELECT 1")


@pytest.mark.parametrize(
    "shutdown,integrity_error", [(True, False), (False, False), (True, True)]
)
def test_actor_only_swallows_explicit_cancellation_during_shutdown(
    tmp_path, monkeypatch, shutdown, integrity_error
):
    cancellation = threading.Event()
    phases = []
    actor = object.__new__(actor_module.ActorSupervisor)
    actor.gpu = SimpleNamespace(actor_cohorts=1)
    actor.registry = None
    actor.heartbeat = SimpleNamespace(
        start=lambda: None,
        advance=lambda **fields: None,
        close=lambda *, final_phase: phases.append(final_phase),
    )
    actor.provider = SimpleNamespace(wait_for_initial=lambda **kwargs: object())
    actor.candidate_provider = None
    actor._active_work_provider = None
    actor.replay_directory = tmp_path / "replay"
    actor.device = torch.device("cpu")

    def store(root, *, cancel_requested):
        assert root == actor.replay_directory
        if shutdown:
            cancellation.set()
        assert cancel_requested() is shutdown
        if integrity_error:
            raise ValueError("unrelated integrity failure")
        raise ReplayStoreCancelled("cancelled scan")

    monkeypatch.setattr(actor_module, "ReplayStore", store)
    if shutdown and not integrity_error:
        assert actor.run(stop_requested=cancellation.is_set) == 0
        assert phases == ["stopped"]
    else:
        expected = ValueError if integrity_error else ReplayStoreCancelled
        with pytest.raises(expected):
            actor.run(stop_requested=cancellation.is_set)
        assert phases == ["failed"]


def test_actor_uses_pure_startup_predicate_in_store_and_cancellation_handler(
    tmp_path, monkeypatch
):
    cancellation = threading.Event()
    actor = object.__new__(actor_module.ActorSupervisor)
    actor.gpu = SimpleNamespace(actor_cohorts=1)
    actor.registry = None
    phases = []
    actor.heartbeat = SimpleNamespace(
        start=lambda: None,
        advance=lambda **fields: None,
        close=lambda *, final_phase: phases.append(final_phase),
    )
    actor.provider = SimpleNamespace(wait_for_initial=lambda **kwargs: object())
    actor.candidate_provider = None
    actor._active_work_provider = None
    actor.replay_directory = tmp_path / "replay"
    actor.device = torch.device("cpu")

    def pause_aware_stop():
        raise AssertionError(
            "pause-aware stop must not run inside startup cancellation"
        )

    def store(root, *, cancel_requested):
        assert cancel_requested is not pause_aware_stop
        assert cancel_requested() is False
        cancellation.set()
        assert cancel_requested() is True
        raise ReplayStoreCancelled("cancelled during integrity scan")

    monkeypatch.setattr(actor_module, "ReplayStore", store)
    assert (
        actor.run(
            stop_requested=pause_aware_stop,
            startup_cancel_requested=cancellation.is_set,
        )
        == 0
    )
    assert phases == ["stopped"]


@pytest.mark.parametrize("waiting_on_lock", [False, True])
def test_shared_cohort_startup_never_parks_while_holding_or_waiting_on_reconcile_lock(
    tmp_path, monkeypatch, waiting_on_lock
):
    root, _ = replay_files(tmp_path)
    config = load_config(
        Path(__file__).parents[1] / "configs/h100-8gpu-variant-efficiency-stage-b.yaml"
    )
    gpu = next(row for row in config.orchestration.gpus if row.gpu_id == 7)
    stop = threading.Event()
    waiting = threading.Event()
    hashing = threading.Event()
    thread_state = threading.local()
    pause_calls = []
    original_flock = fcntl.flock
    lock = (root / ".reconcile.lock").open("r+")
    if waiting_on_lock:
        original_flock(lock.fileno(), fcntl.LOCK_EX)

    class PauseGate:
        def __init__(self, **kwargs):
            pass

        def checkpoint(self, identity):
            assert not getattr(thread_state, "lock_held", False)
            assert not getattr(thread_state, "lock_waiting", False)
            pause_calls.append(identity)
            stop.set()

        def finish(self, identity):
            pass

        def poll(self):
            return None

        def close(self):
            pass

    def observed_flock(descriptor, operation):
        try:
            result = original_flock(descriptor, operation)
        except BlockingIOError:
            thread_state.lock_waiting = True
            waiting.set()
            raise
        if operation & fcntl.LOCK_UN:
            thread_state.lock_held = False
        elif operation & fcntl.LOCK_EX:
            thread_state.lock_waiting = False
            thread_state.lock_held = True
        return result

    original_hash = replay_module._sha256

    def observed_hash(path, **kwargs):
        assert getattr(thread_state, "lock_held", False)
        hashing.set()
        return original_hash(path, **kwargs)

    monkeypatch.setattr(actor_module, "ActorPauseGate", PauseGate)
    monkeypatch.setattr(
        actor_module.ManifestModelProvider,
        "wait_for_initial",
        lambda self, **kwargs: object(),
    )
    monkeypatch.setattr(fcntl, "flock", observed_flock)
    monkeypatch.setattr(replay_module, "_sha256", observed_hash)
    supervisor = actor_module.ActorSupervisor(
        native_module=object(),
        experiment=config,
        gpu=gpu,
        replay_directory=root,
        manifest_path=tmp_path / "champion.json",
        candidate_manifest_path=tmp_path / "candidate.json",
        run_identity=run_identity(tmp_path),
        heartbeat_path=tmp_path / "parent.heartbeat.json",
        metrics_path=tmp_path / "parent.jsonl",
        device="cpu",
        gpu_pause_path=tmp_path / "arena-pause.json",
    )
    try:
        with ThreadPoolExecutor(max_workers=1) as pool:
            future = pool.submit(supervisor._run_cohorts, stop_requested=stop.is_set)
            if waiting_on_lock:
                assert waiting.wait(5)
                stop.set()
            assert future.result(timeout=5) == 0
        if waiting_on_lock:
            assert pause_calls == []
            assert not hashing.is_set()
        else:
            assert hashing.is_set()
            assert pause_calls  # Pause-aware checks still run after lock release.
    finally:
        stop.set()
        original_flock(lock.fileno(), fcntl.LOCK_UN)
        lock.close()
