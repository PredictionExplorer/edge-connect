from __future__ import annotations

import hashlib
import json
import os
import time
from dataclasses import dataclass
from pathlib import Path

import pytest
import yaml

import scripts.training_disaster_recovery as recovery
from startrain.checkpoint import (
    MODEL_MANIFEST_FORMAT,
    MODEL_MANIFEST_VERSION,
    MODEL_POINTER_FORMAT,
    MODEL_POINTER_VERSION,
    RECOVERY_POINTER_FORMAT,
    RECOVERY_POINTER_VERSION,
    RESUME_CUTOVER_FORMAT,
    RESUME_CUTOVER_VERSION,
)
from startrain.contracts import FEATURE_SCHEMA_HASH, RULES_HASH_WIRE
from startrain.model import MODEL_SCHEMA_VERSION
from startrain.replay_store import ReplayStore
from startrain.runtime import RunIdentity, atomic_json


@dataclass(frozen=True)
class RecoveryFixture:
    root: Path
    profile: Path
    shard: Path
    manifest: Path
    checkpoint: Path
    recovery_checkpoint: Path


def _encoded(payload: dict[str, object]) -> bytes:
    return (
        json.dumps(payload, sort_keys=True, separators=(",", ":"), allow_nan=False)
        + "\n"
    ).encode()


def _write_json(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(_encoded(payload))


def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _content_addressed_file(
    directory: Path,
    *,
    prefix: str,
    suffix: str,
    data: bytes,
) -> tuple[Path, str]:
    digest = _sha256(data)
    path = directory / f"{prefix}{digest}{suffix}"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(data)
    return path, digest


def _fixture(tmp_path: Path) -> RecoveryFixture:
    root = tmp_path / "run"
    root.mkdir()
    identity = RunIdentity(
        root / "run.json",
        "run-disaster-test",
        "family-disaster-test",
        1_700_000_000_000_000_000,
    )
    atomic_json(
        identity.path,
        {
            "schema_version": 1,
            "run_id": identity.run_id,
            "generation_family": identity.generation_family,
            "created_ns": identity.created_ns,
        },
    )
    profile = root / "frozen-profile.yaml"
    profile.write_text(
        "orchestration:\n"
        "  run_id: run-disaster-test\n"
        "  directories:\n"
        f"    root: {root}\n",
        encoding="utf-8",
    )
    profile_digest = _sha256(profile.read_bytes())
    (root / "profile.sha256").write_text(
        f"{profile_digest}  {profile.name}\n",
        encoding="utf-8",
    )
    (root / "source-commit.txt").write_text(f"{'a' * 40}\n", encoding="utf-8")
    (root / "python-environment.txt").write_text(
        "Python 3.11.10\nstartrain==0.3.0\n",
        encoding="utf-8",
    )

    shard = root / "replay" / "shards" / "shard-000001.npz"
    shard.parent.mkdir(parents=True)
    shard.write_bytes(b"small immutable replay shard")
    shard_digest = _sha256(shard.read_bytes())
    with ReplayStore(root / "replay") as store:
        store.register_run(identity)
        store.connection.execute(
            """
            INSERT INTO shards(
                relative_path, created_ns, sample_count, ring,
                phase_min, phase_max, model_version, model_step,
                model_identity, run_id, generation_family, actor_id,
                generation, game_count, state, quarantine_reason,
                rules_hash, feature_schema_hash, checksum_sha256
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                str(shard.relative_to(root / "replay")),
                identity.created_ns + 1,
                4,
                10,
                0,
                10,
                "sha256-fixture",
                10,
                "sha256-fixture",
                identity.run_id,
                identity.generation_family,
                "actor-test",
                0,
                1,
                "ready",
                None,
                RULES_HASH_WIRE,
                f"{FEATURE_SCHEMA_HASH:016x}",
                shard_digest,
            ),
        )
        store.connection.execute(
            """
            UPDATE run_counters
            SET committed_samples = 4
            WHERE run_id = ? AND generation_family = ?
            """,
            (identity.run_id, identity.generation_family),
        )

    checkpoint, checkpoint_digest = _content_addressed_file(
        root / "learner" / "checkpoints",
        prefix="sha256-",
        suffix=".pt",
        data=b"dummy model checkpoint; hashing is sufficient for this fixture",
    )
    model_identity = f"sha256-{checkpoint_digest}"
    manifest_payload: dict[str, object] = {
        "format": MODEL_MANIFEST_FORMAT,
        "schema_version": MODEL_MANIFEST_VERSION,
        "rules_hash": RULES_HASH_WIRE,
        "feature_schema_hash": f"{FEATURE_SCHEMA_HASH:016x}",
        "model_schema_version": MODEL_SCHEMA_VERSION,
        "weights": "ema",
        "model_identity": model_identity,
        "model_version": model_identity,
        "model_step": 10,
        "run_id": identity.run_id,
        "generation_family": identity.generation_family,
        "created_ns": identity.created_ns + 2,
        "checkpoint": f"../checkpoints/{checkpoint.name}",
        "checkpoint_sha256": checkpoint_digest,
        "checkpoint_bytes": checkpoint.stat().st_size,
    }
    manifest_data = _encoded(manifest_payload)
    manifest, manifest_digest = _content_addressed_file(
        root / "learner" / "manifests",
        prefix="manifest-",
        suffix=".json",
        data=manifest_data,
    )
    pointer = {
        "format": MODEL_POINTER_FORMAT,
        "schema_version": MODEL_POINTER_VERSION,
        "role": "champion",
        "manifest": f"manifests/{manifest.name}",
        "manifest_sha256": manifest_digest,
        "manifest_bytes": manifest.stat().st_size,
        "model_identity": model_identity,
        "model_step": 10,
        "run_id": identity.run_id,
        "generation_family": identity.generation_family,
        "updated_ns": identity.created_ns + 3,
    }
    _write_json(root / "learner" / "champion.json", pointer)

    recovery_checkpoint, recovery_digest = _content_addressed_file(
        root / "learner" / "recovery",
        prefix="sha256-",
        suffix=".pt",
        data=b"dummy recovery checkpoint with optimizer metadata omitted",
    )
    recovery_pointer: dict[str, object] = {
        "format": RECOVERY_POINTER_FORMAT,
        "schema_version": RECOVERY_POINTER_VERSION,
        "checkpoint": f"recovery/{recovery_checkpoint.name}",
        "checkpoint_sha256": recovery_digest,
        "checkpoint_bytes": recovery_checkpoint.stat().st_size,
        "step": 10,
        "epoch": 1,
        "examples_consumed": 40,
        "run_id": identity.run_id,
        "generation_family": identity.generation_family,
        "updated_ns": identity.created_ns + 4,
    }
    _write_json(root / "learner" / "recovery.json", recovery_pointer)
    (root / "learner" / "recovery.journal.jsonl").write_bytes(
        _encoded(recovery_pointer)
    )
    cutover = {
        "format": RESUME_CUTOVER_FORMAT,
        "schema_version": RESUME_CUTOVER_VERSION,
        "checkpoint": f"recovery/{recovery_checkpoint.name}",
        "checkpoint_sha256": recovery_digest,
        "checkpoint_bytes": recovery_checkpoint.stat().st_size,
        "step": 10,
        "run_id": identity.run_id,
        "generation_family": identity.generation_family,
        "created_ns": identity.created_ns + 4,
    }
    _write_json(root / "learner" / "resume-cutover.json", cutover)
    _write_json(
        root / "learner" / "champion-warm-start.json",
        {
            "format": "startrain.champion-warm-start",
            "schema_version": 1,
            "status": "active",
            "run_id": identity.run_id,
            "generation_family": identity.generation_family,
            "source_model_identity": model_identity,
            "source_manifest": str(manifest),
            "source_manifest_sha256": manifest_digest,
            "source_manifest_bytes": manifest.stat().st_size,
            "checkpoint": f"recovery/{recovery_checkpoint.name}",
            "checkpoint_sha256": recovery_digest,
            "checkpoint_bytes": recovery_checkpoint.stat().st_size,
            "absolute_model_step": 10,
            "examples_consumed": 40,
        },
    )
    _write_json(
        root / "learner" / "cadence.json",
        {
            "schema_version": 1,
            "run_id": identity.run_id,
            "generation_family": identity.generation_family,
            "candidate_examples": 40,
            "selfplay_examples": None,
            "updated_ns": identity.created_ns + 4,
        },
    )
    _write_json(
        root / "learner" / "utd-segment.json",
        {
            "schema_version": 1,
            "run_id": identity.run_id,
            "generation_family": identity.generation_family,
            "target_updates_per_new_sample": 1.0,
            "baseline_examples_consumed": 40,
            "baseline_committed_replay_samples": 4,
            "created_ns": identity.created_ns + 4,
        },
    )
    (root / "learner" / "model-history.jsonl").write_bytes(
        _encoded(
            {
                "schema_version": 1,
                "run_id": identity.run_id,
                "generation_family": identity.generation_family,
                "model_identity": model_identity,
                "model_step": 10,
            }
        )
    )
    _write_json(
        root / "arena" / "promotion.json",
        {
            "schema_version": 1,
            "candidate_manifest": str(manifest),
            "champion_manifest": str(manifest),
            "result": "accepted",
        },
    )
    _write_json(
        root / "status" / "coordinator.json",
        {"schema_version": 1, "state": "running"},
    )
    (root / "logs").mkdir()
    (root / "logs" / "learner.log").write_text("not backed up\n", encoding="utf-8")
    return RecoveryFixture(
        root,
        profile,
        shard,
        manifest,
        checkpoint,
        recovery_checkpoint,
    )


def _snapshot(fixture: RecoveryFixture, backup_root: Path) -> Path:
    return recovery.create_snapshot(
        fixture.root,
        fixture.profile,
        backup_root,
        enforce_separate_filesystem=False,
    )


def _snapshot_payload(path: Path) -> dict[str, object]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    assert isinstance(payload, dict)
    return payload


def _publish_latest(
    backup_root: Path,
    snapshot: Path,
    payload: dict[str, object],
) -> None:
    data = snapshot.read_bytes()
    latest = {
        "report": recovery.LATEST_REPORT,
        "schema_version": recovery.SCHEMA_VERSION,
        "run_id": payload["run_id"],
        "generation_family": payload["generation_family"],
        "path": snapshot.name,
        "sha256": _sha256(data),
        "bytes": len(data),
        "created_ns": payload["created_ns"],
    }
    data = recovery._canonical_json(latest)
    (snapshot.parent / "latest.json").write_bytes(data)
    (backup_root / "latest.json").write_bytes(data)


def test_snapshot_verify_restore_and_deduplicate(tmp_path: Path) -> None:
    fixture = _fixture(tmp_path)
    backup_root = tmp_path / "backup"

    first = _snapshot(fixture, backup_root)
    first_report = recovery.verify_snapshot(first)
    assert recovery.verify_snapshot(backup_root / "latest.json")["status"] == "ok"
    first_payload = _snapshot_payload(first)
    first_catalog = first_payload["catalog"]
    assert isinstance(first_catalog, dict)
    assert first_report["status"] == "ok"
    assert first_report["run_id"] == "run-disaster-test"
    assert "replay/manifest.sqlite3" in first_catalog
    assert "replay/shards/shard-000001.npz" in first_catalog
    assert "learner/champion.json" in first_catalog
    assert "learner/recovery.json" in first_catalog
    assert "learner/resume-cutover.json" in first_catalog
    assert "logs/learner.log" not in first_catalog

    second = _snapshot(fixture, backup_root)
    second_payload = _snapshot_payload(second)
    second_catalog = second_payload["catalog"]
    assert isinstance(second_catalog, dict)
    assert (
        first_catalog["replay/shards/shard-000001.npz"]
        == second_catalog["replay/shards/shard-000001.npz"]
    )
    unique_objects = {
        entry["sha256"]
        for payload in (first_payload, second_payload)
        for entry in payload["catalog"].values()
    }
    objects = list((backup_root / "objects" / "sha256").glob("*/*"))
    assert len(objects) == len(unique_objects)

    destination = tmp_path / "restored"
    restored = recovery.restore_snapshot(
        second,
        destination,
        relocate_profile=True,
    )
    assert restored == destination
    assert (restored / "replay" / "manifest.sqlite3").is_file()
    assert (restored / "replay" / "initialized.json").is_file()
    assert (restored / "replay" / "shards" / fixture.shard.name).read_bytes() == (
        fixture.shard.read_bytes()
    )
    assert (restored / "learner" / "champion.json").read_bytes() == (
        fixture.root / "learner" / "champion.json"
    ).read_bytes()
    restored_journal = restored / "learner" / "recovery.journal.jsonl"
    assert restored_journal.is_file()
    journal_rows = [
        json.loads(line) for line in restored_journal.read_text().splitlines()
    ]
    assert (
        journal_rows[-1]["checkpoint_sha256"]
        == hashlib.sha256(fixture.recovery_checkpoint.read_bytes()).hexdigest()
    )
    marker = json.loads(
        (restored / "recovery" / "disaster-recovery-restore.json").read_text(
            encoding="utf-8"
        )
    )
    assert marker["recreated_initialized"] is False
    assert marker["relocated_profile"] == "profile-relocated.yaml"
    assert marker["relocated_metadata_paths"] > 0
    relocated_warm_start = json.loads(
        (restored / "learner" / "champion-warm-start.json").read_text()
    )
    assert relocated_warm_start["source_manifest"].startswith(str(restored))
    relocated = yaml.safe_load(
        (restored / "profile-relocated.yaml").read_text(encoding="utf-8")
    )
    assert relocated["orchestration"]["directories"]["root"] == str(restored)
    relocated_sha256 = hashlib.sha256(
        (restored / "profile-relocated.yaml").read_bytes()
    ).hexdigest()
    assert (restored / "profile-relocated.sha256").read_text().split()[0] == (
        relocated_sha256
    )
    assert (restored / "profile.sha256").read_text().split()[0] == relocated_sha256
    assert (restored / "recovery" / "original-profile.sha256").is_file()
    assert not (restored / "logs").exists()
    rebind = recovery.rebind_backup_namespace(backup_root, restored)
    assert rebind["changed"] is True
    assert recovery.verify_snapshot(second)["status"] == "ok"
    resnapshot = recovery.create_snapshot(
        restored,
        restored / "profile-relocated.yaml",
        backup_root,
        enforce_separate_filesystem=False,
    )
    assert recovery.verify_snapshot(resnapshot)["status"] == "ok"
    second_restore = recovery.restore_snapshot(
        resnapshot,
        tmp_path / "restored-twice",
        relocate_profile=True,
    )
    second_profile = yaml.safe_load(
        (second_restore / "profile-relocated.yaml").read_text(encoding="utf-8")
    )
    assert second_profile["orchestration"]["directories"]["root"] == str(second_restore)


def test_snapshot_skips_recovery_journal_checkpoint_removed_during_capture(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fixture = _fixture(tmp_path)
    current = json.loads(
        (fixture.root / "learner" / "recovery.json").read_text(encoding="utf-8")
    )
    stale_checkpoint, stale_digest = _content_addressed_file(
        fixture.root / "learner" / "recovery",
        prefix="sha256-",
        suffix=".pt",
        data=b"stale recovery checkpoint removed concurrently",
    )
    stale = {
        **current,
        "checkpoint": f"recovery/{stale_checkpoint.name}",
        "checkpoint_sha256": stale_digest,
        "checkpoint_bytes": stale_checkpoint.stat().st_size,
        "step": 9,
        "updated_ns": int(current["updated_ns"]) - 1,
    }
    journal = fixture.root / "learner" / "recovery.journal.jsonl"
    journal.write_bytes(_encoded(stale) + _encoded(current))

    original = recovery._pointer_checkpoint

    def remove_stale_checkpoint(*args, **kwargs):
        if kwargs.get("name") == "recovery journal line 1":
            stale_checkpoint.unlink()
        return original(*args, **kwargs)

    monkeypatch.setattr(recovery, "_pointer_checkpoint", remove_stale_checkpoint)

    snapshot = _snapshot(fixture, tmp_path / "backup")
    assert recovery.verify_snapshot(snapshot)["status"] == "ok"
    restored = recovery.restore_snapshot(
        snapshot,
        tmp_path / "restored-after-journal-race",
        relocate_profile=True,
    )
    rows = [
        json.loads(line)
        for line in (restored / "learner" / "recovery.journal.jsonl")
        .read_text()
        .splitlines()
    ]
    assert [row["checkpoint_sha256"] for row in rows] == [current["checkpoint_sha256"]]


def test_snapshot_omits_obsolete_warm_start_with_missing_checkpoint(
    tmp_path: Path,
) -> None:
    fixture = _fixture(tmp_path)
    marker_path = fixture.root / "learner" / "champion-warm-start.json"
    marker = json.loads(marker_path.read_text(encoding="utf-8"))
    obsolete, obsolete_digest = _content_addressed_file(
        fixture.root / "learner" / "recovery",
        prefix="sha256-",
        suffix=".pt",
        data=b"obsolete warm-start checkpoint superseded by resume cutover",
    )
    marker.update(
        {
            "checkpoint": f"recovery/{obsolete.name}",
            "checkpoint_sha256": obsolete_digest,
            "checkpoint_bytes": obsolete.stat().st_size,
        }
    )
    _write_json(marker_path, marker)
    obsolete.unlink()

    snapshot = _snapshot(fixture, tmp_path / "backup")
    payload = _snapshot_payload(snapshot)
    assert "learner/champion-warm-start.json" not in payload["catalog"]
    assert recovery.verify_snapshot(snapshot)["status"] == "ok"
    restored = recovery.restore_snapshot(
        snapshot,
        tmp_path / "restored-without-obsolete-warm-start",
        relocate_profile=True,
    )
    assert not (restored / "learner" / "champion-warm-start.json").exists()
    assert (restored / "learner" / "recovery.json").is_file()
    assert (restored / "learner" / "resume-cutover.json").is_file()


def test_snapshot_rejects_missing_shard_without_publishing_latest(
    tmp_path: Path,
) -> None:
    fixture = _fixture(tmp_path)
    fixture.shard.unlink()
    backup_root = tmp_path / "backup"

    with pytest.raises(recovery.DisasterRecoveryError, match="ready replay shard"):
        _snapshot(fixture, backup_root)

    assert not list((backup_root / "snapshots").glob("*/latest.json"))


def test_snapshot_rejects_cadence_ahead_of_durable_state(tmp_path: Path) -> None:
    fixture = _fixture(tmp_path)
    cadence_path = fixture.root / "learner" / "cadence.json"
    cadence = json.loads(cadence_path.read_text(encoding="utf-8"))
    cadence["candidate_examples"] = 80
    _write_json(cadence_path, cadence)
    backup_root = tmp_path / "backup"

    with pytest.raises(
        recovery.DisasterRecoveryError,
        match="cadence candidate_examples is ahead",
    ):
        recovery.create_snapshot(
            fixture.root,
            fixture.profile,
            backup_root,
            enforce_separate_filesystem=False,
        )

    assert not list((backup_root / "snapshots").glob("*/latest.json"))


def test_expected_backup_mount_must_be_mounted(tmp_path: Path) -> None:
    fixture = _fixture(tmp_path)

    with pytest.raises(
        recovery.DisasterRecoveryError,
        match="expected backup mount is not mounted",
    ):
        recovery.create_snapshot(
            fixture.root,
            fixture.profile,
            tmp_path / "backup",
            enforce_separate_filesystem=False,
            expected_backup_mount=tmp_path,
        )


def test_backup_namespace_rejects_second_root_with_same_run_id(tmp_path: Path) -> None:
    first_parent = tmp_path / "first"
    second_parent = tmp_path / "second"
    first_parent.mkdir()
    second_parent.mkdir()
    first = _fixture(first_parent)
    second = _fixture(second_parent)
    backup_root = tmp_path / "backup"
    _snapshot(first, backup_root)

    with pytest.raises(
        recovery.DisasterRecoveryError,
        match="different workload root",
    ):
        _snapshot(second, backup_root)


def test_interrupted_copy_and_tampered_object_fail_closed(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fixture = _fixture(tmp_path)
    interrupted_root = tmp_path / "interrupted"
    original = recovery._store_object
    calls = 0

    def interrupt(*args, **kwargs):
        nonlocal calls
        calls += 1
        if calls == 4:
            raise OSError("simulated interruption")
        return original(*args, **kwargs)

    monkeypatch.setattr(recovery, "_store_object", interrupt)
    with pytest.raises(OSError, match="simulated interruption"):
        _snapshot(fixture, interrupted_root)
    assert not list(interrupted_root.glob("snapshots/*/latest.json"))

    monkeypatch.setattr(recovery, "_store_object", original)
    backup_root = tmp_path / "backup"
    snapshot = _snapshot(fixture, backup_root)
    payload = _snapshot_payload(snapshot)
    catalog = payload["catalog"]
    assert isinstance(catalog, dict)
    shard_entry = catalog["replay/shards/shard-000001.npz"]
    shard_object = (
        backup_root
        / "objects"
        / "sha256"
        / shard_entry["sha256"][:2]
        / shard_entry["sha256"]
    )
    shard_object.chmod(0o644)
    shard_object.write_bytes(b"tampered")

    with pytest.raises(recovery.DisasterRecoveryError, match="byte length|SHA-256"):
        recovery.verify_snapshot(snapshot)
    with pytest.raises(recovery.DisasterRecoveryError, match="byte length|SHA-256"):
        recovery.restore_snapshot(snapshot, tmp_path / "must-not-exist")
    assert not (tmp_path / "must-not-exist").exists()


def test_stale_latest_pointer_is_rejected(tmp_path: Path) -> None:
    fixture = _fixture(tmp_path)
    backup_root = tmp_path / "backup"
    first = _snapshot(fixture, backup_root)
    second = _snapshot(fixture, backup_root)
    assert second != first

    _publish_latest(backup_root, first, _snapshot_payload(first))
    latest = first.parent / "latest.json"
    with pytest.raises(recovery.DisasterRecoveryError, match="stale"):
        recovery.verify_snapshot(latest)
    with pytest.raises(recovery.DisasterRecoveryError, match="stale"):
        recovery.verify_snapshot(backup_root / "latest.json")

    healed = _snapshot(fixture, backup_root)
    assert healed not in (first, second)
    assert recovery.verify_snapshot(latest)["status"] == "ok"
    assert recovery.verify_snapshot(backup_root / "latest.json")["status"] == "ok"


def test_uncommitted_newer_manifest_does_not_invalidate_latest(tmp_path: Path) -> None:
    fixture = _fixture(tmp_path)
    backup_root = tmp_path / "backup"
    first = _snapshot(fixture, backup_root)
    payload = _snapshot_payload(first)
    payload["created_ns"] = int(payload["created_ns"]) + 1
    data = _encoded(payload)
    digest = hashlib.sha256(data).hexdigest()
    uncommitted = first.parent / f"{payload['created_ns']}-{digest}.json"
    uncommitted.write_bytes(data)

    assert recovery.verify_snapshot(first.parent / "latest.json")["snapshot"] == str(
        first
    )
    assert not uncommitted.with_name(f"{uncommitted.name}.commit").exists()


def test_legacy_initialized_reconstruction_is_explicit(tmp_path: Path) -> None:
    fixture = _fixture(tmp_path)
    backup_root = tmp_path / "backup"
    (fixture.root / "replay" / "initialized.json").unlink()
    with pytest.raises(
        recovery.DisasterRecoveryError,
        match="required replay/initialized",
    ):
        _snapshot(fixture, backup_root)
    legacy = recovery.create_snapshot(
        fixture.root,
        fixture.profile,
        backup_root,
        enforce_separate_filesystem=False,
        allow_legacy_missing_initialized=True,
    )
    recovery.verify_snapshot(legacy)

    with pytest.raises(
        recovery.DisasterRecoveryError,
        match="recreate-initialized",
    ):
        recovery.restore_snapshot(legacy, tmp_path / "without-marker")
    restored = recovery.restore_snapshot(
        legacy,
        tmp_path / "legacy-restored",
        recreate_initialized=True,
    )
    initialized = json.loads(
        (restored / "replay" / "initialized.json").read_text(encoding="utf-8")
    )
    assert initialized["run_id"] == "run-disaster-test"
    marker = json.loads(
        (restored / "recovery" / "disaster-recovery-restore.json").read_text(
            encoding="utf-8"
        )
    )
    assert marker["recreated_initialized"] is True


def test_gc_dry_run_apply_and_integrity_safeguards(tmp_path: Path) -> None:
    fixture = _fixture(tmp_path)
    backup_root = tmp_path / "backup"
    first = _snapshot(fixture, backup_root)
    second = _snapshot(fixture, backup_root)
    assert first.exists() and second.exists()

    orphan_data = b"unreferenced object"
    orphan_digest = _sha256(orphan_data)
    orphan = backup_root / "objects" / "sha256" / orphan_digest[:2] / orphan_digest
    orphan.parent.mkdir(parents=True, exist_ok=True)
    orphan.write_bytes(orphan_data)
    old_ns = time.time_ns() - 10_000_000_000
    os.utime(orphan, ns=(old_ns, old_ns))

    dry_run = recovery.garbage_collect(
        backup_root,
        retain_latest=0,
        retain_hourly=0,
        retain_daily=0,
        retain_monthly=0,
        grace_seconds=0,
        now_ns=time.time_ns(),
    )
    assert dry_run["mode"] == "dry-run"
    assert dry_run["prunable_snapshots"] == 1
    assert dry_run["deletable_objects"] >= 1
    assert first.exists()
    assert orphan.exists()

    applied = recovery.garbage_collect(
        backup_root,
        retain_latest=0,
        retain_hourly=0,
        retain_daily=0,
        retain_monthly=0,
        grace_seconds=0,
        apply=True,
        now_ns=time.time_ns(),
    )
    assert applied["deleted_snapshots"] == 1
    assert not first.exists()
    assert second.exists()
    assert not orphan.exists()
    recovery.verify_snapshot(second)

    latest = second.parent / "latest.json"
    before = set((backup_root / "objects" / "sha256").glob("*/*"))
    latest.write_text("{}\n", encoding="utf-8")
    with pytest.raises(recovery.DisasterRecoveryError):
        recovery.garbage_collect(
            backup_root,
            retain_latest=0,
            retain_hourly=0,
            retain_daily=0,
            retain_monthly=0,
            grace_seconds=0,
            apply=True,
        )
    assert set((backup_root / "objects" / "sha256").glob("*/*")) == before


def test_snapshot_requires_a_separate_backup_filesystem(tmp_path: Path) -> None:
    fixture = _fixture(tmp_path)
    with pytest.raises(recovery.DisasterRecoveryError, match="different filesystem"):
        recovery.create_snapshot(
            fixture.root,
            fixture.profile,
            tmp_path / "same-filesystem-backup",
        )
