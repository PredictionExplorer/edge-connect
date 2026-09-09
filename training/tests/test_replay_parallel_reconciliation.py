from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
import os
from pathlib import Path
import sqlite3
import threading

import pytest

import startrain.replay_store as replay_module
from startrain.replay_store import ReplayStore, ReplayStoreCancelled
from test_pipeline_core import append_replay, make_replay_sample, run_identity
from test_replay_startup_cancellation import replay_files


def open_metrics(root):
    with ReplayStore(root) as store:
        return dict(store.reconciliation_metrics)


def pause_after_hash(monkeypatch, path):
    hashed, resume = threading.Event(), threading.Event()
    original = replay_module._hash_file
    local = threading.local()
    counts = defaultdict(int)

    def observed(source, **kwargs):
        result = original(source, **kwargs)
        counts[source] += 1
        if source == path and not getattr(local, "paused", False):
            local.paused = True
            hashed.set()
            assert resume.wait(5), "test did not release checksum reader"
        return result

    monkeypatch.setattr(replay_module, "_hash_file", observed)
    return hashed, resume, counts


def test_independent_successful_opens_hash_every_snapshot_file_concurrently(
    tmp_path, monkeypatch
):
    root, records = replay_files(tmp_path, count=3)
    barrier = threading.Barrier(2)
    original = replay_module._hash_file
    scanned = defaultdict(list)

    def observed(path, **kwargs):
        scanned[threading.get_ident()].append(path)
        if path == records[0].path:
            barrier.wait(timeout=5)  # This deadlocks under the old exclusive scan.
        return original(path, **kwargs)

    monkeypatch.setattr(replay_module, "_hash_file", observed)
    with ThreadPoolExecutor(max_workers=2) as pool:
        results = list(pool.map(open_metrics, [root, root]))
    assert len(scanned) == 2
    assert all(
        set(paths) == {row.path for row in records} for paths in scanned.values()
    )
    assert all(
        result["missing_committed"] == result["corrupt_committed"] == 0
        for result in results
    )


def test_concurrent_append_and_gc_do_not_create_false_missing_quarantines(
    tmp_path, monkeypatch
):
    root, records = replay_files(tmp_path)
    identity = run_identity(tmp_path)
    with ReplayStore(root) as writer:
        hashed, resume, _ = pause_after_hash(monkeypatch, records[0].path)
        with ThreadPoolExecutor(max_workers=1) as pool:
            future = pool.submit(open_metrics, root)
            try:
                assert hashed.wait(5)
                generation = writer.lease_generation(identity, "actor-test")
                appended = append_replay(
                    writer,
                    [
                        make_replay_sample(
                            identity=identity,
                            generation=generation,
                            game_id="concurrent-append",
                        )
                    ],
                    identity,
                    generation=generation,
                    model_step=0,
                )
                collected = writer.collect_garbage(
                    run_id=identity.run_id,
                    generation_family=identity.generation_family,
                    retain_shards_per_ring=1,
                    dry_run=False,
                )
                assert collected["deleted_shards"] == 2
                resume.set()
                metrics = future.result(timeout=5)
                assert metrics["missing_committed"] == metrics["corrupt_committed"] == 0
                row = writer.connection.execute(
                    "SELECT relative_path, state FROM shards"
                ).fetchone()
                assert (
                    row["state"] == "ready"
                    and root / row["relative_path"] == appended.path
                )
                assert not list((root / "quarantine").iterdir())
            finally:
                resume.set()


@pytest.mark.parametrize("initially_corrupt", [False, True])
def test_changed_file_is_rehashed_before_any_quarantine_decision(
    tmp_path, monkeypatch, initially_corrupt
):
    root, records = replay_files(tmp_path, count=1)
    original_bytes = records[0].path.read_bytes()
    if initially_corrupt:
        records[0].path.write_bytes(b"initial corruption")
    hashed, resume, counts = pause_after_hash(monkeypatch, records[0].path)
    with ThreadPoolExecutor(max_workers=1) as pool:
        future = pool.submit(open_metrics, root)
        try:
            assert hashed.wait(5)
            records[0].path.write_bytes(
                original_bytes if initially_corrupt else b"new corruption"
            )
            resume.set()
            metrics = future.result(timeout=5)
        finally:
            resume.set()
    assert counts[records[0].path] >= 2
    with sqlite3.connect(root / "manifest.sqlite3") as connection:
        relative, state = connection.execute(
            "SELECT relative_path, state FROM shards"
        ).fetchone()
    if initially_corrupt:
        assert state == "ready" and metrics["corrupt_committed"] == 0
        assert records[0].path.read_bytes() == original_bytes
    else:
        assert state == "quarantined" and metrics["corrupt_committed"] == 1
        assert (root / relative).read_bytes() == b"new corruption"


def test_changed_ready_metadata_cannot_reuse_a_different_file_verdict(
    tmp_path, monkeypatch
):
    root, records = replay_files(tmp_path, count=1)
    replacement = root / "shards/replacement.npz"
    replacement.write_bytes(records[0].path.read_bytes())
    hashed, resume, counts = pause_after_hash(monkeypatch, records[0].path)
    with ThreadPoolExecutor(max_workers=1) as pool:
        future = pool.submit(open_metrics, root)
        try:
            assert hashed.wait(5)
            with sqlite3.connect(root / "manifest.sqlite3") as connection:
                connection.execute(
                    "UPDATE shards SET relative_path = ? WHERE id = ?",
                    ("shards/replacement.npz", records[0].shard_id),
                )
            resume.set()
            metrics = future.result(timeout=5)
        finally:
            resume.set()
    assert counts[replacement] == 1
    assert metrics["missing_committed"] == metrics["corrupt_committed"] == 0


def test_two_reconcilers_quarantine_corruption_exactly_once(tmp_path, monkeypatch):
    root, records = replay_files(tmp_path, count=1)
    records[0].path.write_bytes(b"corrupt")
    barrier = threading.Barrier(2)
    original = replay_module._hash_file

    def observed(path, **kwargs):
        result = original(path, **kwargs)
        barrier.wait(timeout=5)
        return result

    monkeypatch.setattr(replay_module, "_hash_file", observed)
    with ThreadPoolExecutor(max_workers=2) as pool:
        results = list(pool.map(open_metrics, [root, root]))
    assert sum(result["corrupt_committed"] for result in results) == 1
    assert sum(result["missing_committed"] for result in results) == 0
    assert len(list((root / "quarantine").iterdir())) == 1


def test_missing_file_that_reappears_is_checked_instead_of_quarantined(
    tmp_path, monkeypatch
):
    root, records = replay_files(tmp_path, count=1)
    content = records[0].path.read_bytes()
    records[0].path.unlink()
    missing, resume = threading.Event(), threading.Event()
    original = ReplayStore._observe_file
    first = True

    def observe(path, cancellation):
        nonlocal first
        result = original(path, cancellation)
        if first:
            first = False
            assert result is None
            missing.set()
            assert resume.wait(5)
        return result

    monkeypatch.setattr(ReplayStore, "_observe_file", staticmethod(observe))
    with ThreadPoolExecutor(max_workers=1) as pool:
        future = pool.submit(open_metrics, root)
        try:
            assert missing.wait(5)
            records[0].path.write_bytes(content)
            resume.set()
            metrics = future.result(timeout=5)
        finally:
            resume.set()
    assert metrics["missing_committed"] == metrics["corrupt_committed"] == 0


def test_new_restore_marker_aborts_without_committing_a_stale_repair(
    tmp_path, monkeypatch
):
    root, records = replay_files(tmp_path, count=1)
    records[0].path.write_bytes(b"corrupt before scan")
    hashed, resume, _ = pause_after_hash(monkeypatch, records[0].path)
    marker = root / "restore-marker.json"
    with ThreadPoolExecutor(max_workers=1) as pool:
        future = pool.submit(open_metrics, root)
        try:
            assert hashed.wait(5)
            marker.write_text('{"new_restore":true}')
            resume.set()
            with pytest.raises(RuntimeError, match="restore marker changed"):
                future.result(timeout=5)
        finally:
            resume.set()
    assert marker.exists() and records[0].path.exists()
    assert not list((root / "quarantine").iterdir())


def test_parallel_restore_reconciliation_preserves_orphans_and_clears_one_marker(
    tmp_path, monkeypatch
):
    root, records = replay_files(tmp_path, count=1)
    marker = root / "restore-marker.json"
    marker.write_text("{}")
    orphan = root / "shards/post-restore.npz"
    orphan.write_bytes(b"preserve this unreferenced file")
    barrier = threading.Barrier(2)
    original = replay_module._hash_file

    def observed(path, **kwargs):
        result = original(path, **kwargs)
        if path == records[0].path:
            barrier.wait(timeout=5)
        return result

    monkeypatch.setattr(replay_module, "_hash_file", observed)
    with ThreadPoolExecutor(max_workers=2) as pool:
        results = list(pool.map(open_metrics, [root, root]))
    assert sum(result["post_restore_orphans"] for result in results) == 1
    assert not marker.exists() and not orphan.exists()
    assert [path.read_bytes() for path in (root / "quarantine").iterdir()] == [
        b"preserve this unreferenced file"
    ]


def test_replaced_manifest_inode_aborts_before_repair(tmp_path, monkeypatch):
    root, records = replay_files(tmp_path, count=1)
    replacement = tmp_path / "replacement.sqlite3"
    with (
        sqlite3.connect(root / "manifest.sqlite3") as source,
        sqlite3.connect(replacement) as destination,
    ):
        source.backup(destination)
    hashed, resume, _ = pause_after_hash(monkeypatch, records[0].path)
    with ThreadPoolExecutor(max_workers=1) as pool:
        future = pool.submit(open_metrics, root)
        try:
            assert hashed.wait(5)
            os.replace(replacement, root / "manifest.sqlite3")
            resume.set()
            with pytest.raises(RuntimeError, match="manifest was replaced"):
                future.result(timeout=5)
        finally:
            resume.set()
    assert records[0].path.exists()


def test_quarantine_database_failure_restores_the_original_file(tmp_path):
    root, records = replay_files(tmp_path, count=1)
    records[0].path.write_bytes(b"corrupt")
    with sqlite3.connect(root / "manifest.sqlite3") as connection:
        connection.execute(
            "CREATE TRIGGER fail_quarantine BEFORE UPDATE OF state ON shards BEGIN SELECT RAISE(ABORT, 'injected repair failure'); END"
        )
    with pytest.raises(sqlite3.IntegrityError, match="injected repair failure"):
        ReplayStore(root)
    assert records[0].path.read_bytes() == b"corrupt"
    assert not list((root / "quarantine").iterdir())


def test_modification_during_a_file_read_cannot_reuse_its_old_digest(
    tmp_path, monkeypatch
):
    root, records = replay_files(tmp_path, count=1)
    target = records[0].path
    original_open = Path.open
    changed = False

    class Reader:
        def __init__(self, stream):
            self.stream = stream

        def __enter__(self):
            return self

        def __exit__(self, *args):
            return self.stream.__exit__(*args)

        def fileno(self):
            return self.stream.fileno()

        def read(self, size):
            nonlocal changed
            data = self.stream.read(size)
            if not changed:
                changed = True
                target.write_bytes(b"corruption introduced during read")
            return data

    def open_file(path, mode="r", *args, **kwargs):
        stream = original_open(path, mode, *args, **kwargs)
        return Reader(stream) if path == target and mode == "rb" else stream

    monkeypatch.setattr(Path, "open", open_file)
    result = open_metrics(root)
    assert result["corrupt_committed"] == 1
    assert not target.exists()


def test_continuously_changing_file_aborts_after_bounded_revalidation(
    tmp_path, monkeypatch
):
    root, records = replay_files(tmp_path, count=1)
    original = replay_module._hash_file
    reads = 0

    def changing(path, **kwargs):
        nonlocal reads
        result = original(path, **kwargs)
        reads += 1
        details = path.stat()
        os.utime(path, ns=(details.st_atime_ns, details.st_mtime_ns + 1_000_000))
        return result

    monkeypatch.setattr(replay_module, "_hash_file", changing)
    with pytest.raises(RuntimeError, match="changed repeatedly"):
        ReplayStore(root)
    assert reads == 4  # One unlocked scan, at most three serialized retries.
    assert records[0].path.exists()
    assert not list((root / "quarantine").iterdir())


def test_cancellation_between_serialized_repairs_preserves_committed_quarantines(
    tmp_path, monkeypatch
):
    root, records = replay_files(tmp_path, count=2)
    for index, record in enumerate(records):
        record.path.write_bytes(f"corrupt-{index}".encode())
    cancel = threading.Event()
    original = ReplayStore._repair_observation_locked

    def repair(self, *args, **kwargs):
        result = original(self, *args, **kwargs)
        cancel.set()
        return result

    monkeypatch.setattr(ReplayStore, "_repair_observation_locked", repair)
    with pytest.raises(ReplayStoreCancelled):
        ReplayStore(root, cancel_requested=cancel.is_set)
    with sqlite3.connect(root / "manifest.sqlite3") as connection:
        rows = connection.execute(
            "SELECT state, relative_path FROM shards ORDER BY id"
        ).fetchall()
    assert rows[0][0] == "quarantined" and rows[1][0] == "ready"
    assert (root / rows[0][1]).read_bytes() == b"corrupt-0"
    assert records[1].path.read_bytes() == b"corrupt-1"


def test_replaced_restore_marker_is_preserved_and_rejected(tmp_path, monkeypatch):
    root, records = replay_files(tmp_path, count=1)
    marker = root / "restore-marker.json"
    marker.write_text('{"restore":1}')
    hashed, resume, _ = pause_after_hash(monkeypatch, records[0].path)
    with ThreadPoolExecutor(max_workers=1) as pool:
        future = pool.submit(open_metrics, root)
        try:
            assert hashed.wait(5)
            replacement = root / "next-restore.json"
            replacement.write_text('{"restore":2}')
            os.replace(replacement, marker)
            resume.set()
            with pytest.raises(RuntimeError, match="restore marker changed"):
                future.result(timeout=5)
        finally:
            resume.set()
    assert marker.read_text() == '{"restore":2}'
