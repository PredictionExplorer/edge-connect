from __future__ import annotations

import hashlib
import json
import os
import sqlite3
from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import pytest
import torch
import yaml

from scripts import migrate_continuous_profile as migration
from startrain.config import load_config


@dataclass(frozen=True)
class _Fixture:
    root: Path
    old_profile: Path
    candidate_profile: Path
    target_name: str
    request: migration.MigrationRequest
    old_profile_bytes: bytes
    old_checksum_bytes: bytes
    old_source_commit_bytes: bytes
    checkpoint: Path


def _write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, sort_keys=True, separators=(",", ":")) + "\n",
        encoding="utf-8",
    )


def _fixture(tmp_path: Path) -> _Fixture:
    root = tmp_path / "active-run"
    root.mkdir()
    base_path = Path(__file__).parents[1] / "configs" / "h100-8gpu-throughput.yaml"
    old_raw = yaml.safe_load(base_path.read_text(encoding="utf-8"))
    run_id = "continuous-test-run"
    family = "family-continuous-test"
    created_ns = 11
    old_raw["orchestration"]["run_id"] = run_id
    old_raw["orchestration"]["directories"]["root"] = str(root)
    old_raw["orchestration"]["autonomous"] = {"enabled": False}
    old_raw["orchestration"]["promotion"]["finish_inflight_candidate"] = False
    old_raw["arena"]["continuation_pairs_per_ring"] = 150

    old_profile = root / "profile-throughput-v1.yaml"
    old_profile.write_text(yaml.safe_dump(old_raw, sort_keys=False), encoding="utf-8")
    old_profile.chmod(0o444)
    old_profile_bytes = old_profile.read_bytes()
    old_profile_sha256 = hashlib.sha256(old_profile_bytes).hexdigest()
    checksum_path = root / "profile.sha256"
    checksum_path.write_text(
        f"{old_profile_sha256}  {old_profile}\n",
        encoding="utf-8",
    )
    old_checksum_bytes = checksum_path.read_bytes()
    source_commit_path = root / "source-commit.txt"
    source_commit_path.write_text(f"{'a' * 40}\n", encoding="utf-8")
    old_source_commit_bytes = source_commit_path.read_bytes()

    target_raw = deepcopy(old_raw)
    target_raw["train"]["per_rank_batch_size"] += 256
    target_raw["learner"]["candidate_interval"] += 1_000
    target_raw["learner"]["max_replay_lag_steps"] += 1_000
    target_raw["orchestration"]["plateau"]["max_learner_champion_lag_steps"] = (
        target_raw["learner"]["max_replay_lag_steps"]
    )
    target_raw["orchestration"]["promotion"]["finish_inflight_candidate"] = True
    target_raw["arena"]["continuation_pairs_per_ring"] = 25
    candidate_profile = tmp_path / "profile-candidate.yaml"
    candidate_profile.write_text(
        yaml.safe_dump(target_raw, sort_keys=False),
        encoding="utf-8",
    )

    _write_json(
        root / "run.json",
        {
            "schema_version": 1,
            "run_id": run_id,
            "generation_family": family,
            "created_ns": created_ns,
        },
    )
    step = 100
    examples = 51_200
    _write_json(
        root / "status" / "learner.heartbeat.json",
        {
            "schema_version": 1,
            "worker": "learner",
            "pid": 999_999_999,
            "heartbeat_ns": 20,
            "phase": "stopped",
            "step": step + 10,
            "examples_consumed": examples + 5_120,
        },
    )

    checkpoint_directory = root / "learner" / "recovery"
    checkpoint_directory.mkdir(parents=True)
    staging_checkpoint = checkpoint_directory / "staging.pt"
    checkpoint_config = load_config(old_profile).as_dict()
    torch.save(
        {
            "format": "startrain.checkpoint",
            "version": 3,
            "step": step,
            "epoch": 7,
            "config": {
                section: checkpoint_config[section]
                for section in ("game", "model", "loss", "optimizer")
            },
            "extra": {
                "run_id": run_id,
                "generation_family": family,
                "examples_consumed": examples,
            },
            "model": {},
            "optimizer": {},
            "scheduler": {},
            "ema": {},
        },
        staging_checkpoint,
    )
    checkpoint_sha256 = hashlib.sha256(staging_checkpoint.read_bytes()).hexdigest()
    checkpoint = checkpoint_directory / f"sha256-{checkpoint_sha256}.pt"
    staging_checkpoint.replace(checkpoint)
    checkpoint_bytes = checkpoint.read_bytes()
    _write_json(
        root / "learner" / "recovery.json",
        {
            "format": "startrain.recovery-pointer",
            "schema_version": 1,
            "checkpoint": f"recovery/{checkpoint.name}",
            "checkpoint_sha256": checkpoint_sha256,
            "checkpoint_bytes": len(checkpoint_bytes),
            "step": step,
            "epoch": 7,
            "examples_consumed": examples,
            "run_id": run_id,
            "generation_family": family,
            "updated_ns": 21,
        },
    )

    champion_identity = f"sha256-{'c' * 64}"
    champion_manifest_payload = {
        "format": "startrain.model-manifest",
        "schema_version": 3,
        "model_identity": champion_identity,
        "model_version": champion_identity,
        "model_step": 90,
        "run_id": run_id,
        "generation_family": family,
    }
    champion_manifest_bytes = (
        json.dumps(
            champion_manifest_payload,
            sort_keys=True,
            separators=(",", ":"),
        )
        + "\n"
    ).encode()
    champion_manifest_sha256 = hashlib.sha256(champion_manifest_bytes).hexdigest()
    champion_manifest = (
        root / "learner" / "manifests" / f"manifest-{champion_manifest_sha256}.json"
    )
    champion_manifest.parent.mkdir(parents=True)
    champion_manifest.write_bytes(champion_manifest_bytes)
    _write_json(
        root / "learner" / "champion.json",
        {
            "format": "startrain.model-pointer",
            "schema_version": 2,
            "role": "champion",
            "manifest": f"manifests/{champion_manifest.name}",
            "manifest_sha256": champion_manifest_sha256,
            "manifest_bytes": len(champion_manifest_bytes),
            "model_identity": champion_identity,
            "model_step": 90,
            "run_id": run_id,
            "generation_family": family,
            "updated_ns": 23,
        },
    )

    replay = root / "replay"
    replay.mkdir()
    with sqlite3.connect(replay / "manifest.sqlite3") as connection:
        connection.executescript(
            """
            CREATE TABLE runs (
                run_id TEXT PRIMARY KEY,
                generation_family TEXT NOT NULL UNIQUE,
                created_ns INTEGER NOT NULL
            );
            CREATE TABLE run_counters (
                run_id TEXT NOT NULL,
                generation_family TEXT NOT NULL,
                committed_samples INTEGER NOT NULL,
                updated_ns INTEGER NOT NULL,
                history_complete INTEGER NOT NULL,
                PRIMARY KEY(run_id, generation_family)
            );
            """
        )
        connection.execute(
            """
            INSERT INTO runs(run_id, generation_family, created_ns)
            VALUES (?, ?, ?)
            """,
            (run_id, family, created_ns),
        )
        connection.execute(
            """
            INSERT INTO run_counters(
                run_id, generation_family, committed_samples, updated_ns,
                history_complete
            ) VALUES (?, ?, ?, ?, 1)
            """,
            (run_id, family, 60_000, 24),
        )
    _write_json(
        replay / "initialized.json",
        {
            "schema_version": 1,
            "run_id": run_id,
            "generation_family": family,
            "initialized_ns": 12,
        },
    )

    request = migration.MigrationRequest(
        run_root=root,
        old_profile=old_profile,
        new_profile=candidate_profile,
        target_profile_name="profile-throughput-v2.yaml",
        reason="reduce-arena-supersession-and-increase-batch",
        from_source_commit="a" * 40,
        to_source_commit="b" * 40,
    )
    return _Fixture(
        root=root,
        old_profile=old_profile,
        candidate_profile=candidate_profile,
        target_name=request.target_profile_name,
        request=request,
        old_profile_bytes=old_profile_bytes,
        old_checksum_bytes=old_checksum_bytes,
        old_source_commit_bytes=old_source_commit_bytes,
        checkpoint=checkpoint,
    )


def _snapshot(root: Path) -> dict[str, tuple[bytes, int]]:
    return {
        str(path.relative_to(root)): (path.read_bytes(), path.stat().st_mode & 0o777)
        for path in sorted(root.rglob("*"))
        if path.is_file()
    }


def test_dry_run_does_not_write(tmp_path: Path) -> None:
    fixture = _fixture(tmp_path)
    before = _snapshot(fixture.root)

    result = migration.migrate_continuous_profile(fixture.request)

    assert result["mode"] == "dry-run"
    assert result["boundary"]["learner_step"] == 100
    assert result["boundary"]["discarded_uncheckpointed_steps"] == 10
    assert {change["path"] for change in result["changes"]} == {
        "train.per_rank_batch_size",
        "learner.candidate_interval",
        "learner.max_replay_lag_steps",
        "orchestration.plateau.max_learner_champion_lag_steps",
        "orchestration.promotion.finish_inflight_candidate",
        "arena.continuation_pairs_per_ring",
    }
    assert _snapshot(fixture.root) == before
    assert not (fixture.root / fixture.target_name).exists()
    assert not (fixture.root / "continuous-migrations.jsonl").exists()
    assert not (fixture.root / "migration-backups").exists()


def test_apply_writes_immutable_profile_record_and_complete_backup(
    tmp_path: Path,
) -> None:
    fixture = _fixture(tmp_path)

    result = migration.migrate_continuous_profile(fixture.request, apply=True)

    target = fixture.root / fixture.target_name
    target_checksum = target.with_suffix(".sha256")
    target_digest = hashlib.sha256(target.read_bytes()).hexdigest()
    assert result["mode"] == "apply"
    assert target.read_bytes() == fixture.candidate_profile.read_bytes()
    assert target.stat().st_mode & 0o222 == 0
    assert target_checksum.stat().st_mode & 0o222 == 0
    assert target_checksum.read_text(encoding="utf-8") == (
        f"{target_digest}  {target}\n"
    )
    assert (fixture.root / "profile.sha256").read_bytes() == (
        target_checksum.read_bytes()
    )
    assert (fixture.root / "source-commit.txt").read_text(encoding="utf-8") == (
        f"{'b' * 40}\n"
    )
    assert fixture.old_profile.read_bytes() == fixture.old_profile_bytes
    assert fixture.old_profile.stat().st_mode & 0o222 == 0
    assert fixture.checkpoint.is_file()

    records = [
        json.loads(line)
        for line in (fixture.root / "continuous-migrations.jsonl")
        .read_text(encoding="utf-8")
        .splitlines()
    ]
    assert len(records) == 1
    record = records[0]
    assert record["from_source_commit"] == "a" * 40
    assert record["to_source_commit"] == "b" * 40
    assert record["from_profile"] == fixture.old_profile.name
    assert record["to_profile"] == fixture.target_name
    assert record["committed_replay_samples"] == 60_000
    assert record["recovery_checkpoint_sha256"] in fixture.checkpoint.name
    assert record["changes"] == result["changes"]

    backup = Path(result["backup_bundle"])
    assert backup.is_dir()
    assert (backup / fixture.old_profile.name).read_bytes() == fixture.old_profile_bytes
    assert (backup / "profile.sha256").read_bytes() == fixture.old_checksum_bytes
    assert (
        backup / "source-commit.txt"
    ).read_bytes() == fixture.old_source_commit_bytes
    for relative in (
        "run.json",
        "status/learner.heartbeat.json",
        "learner/recovery.json",
        "learner/champion.json",
        "replay/initialized.json",
    ):
        assert (backup / relative).is_file()
    backup_manifest = json.loads((backup / "manifest.json").read_text(encoding="utf-8"))
    assert backup_manifest["to_profile"] == fixture.target_name
    assert backup_manifest["validated_state"]["replay"]["history_complete"] is True
    assert not (fixture.root / "coordinator.lock").exists()


def test_additive_default_field_accepts_legacy_chain_hash(tmp_path: Path) -> None:
    fixture = _fixture(tmp_path)
    config = load_config(fixture.old_profile)
    materialized = migration.canonical_config_sha256(config)
    compatible = migration._compatible_source_config_sha256s(config)
    legacy_hashes = compatible - {materialized}
    assert len(legacy_hashes) == 1
    legacy_hash = legacy_hashes.pop()
    source_profile_sha256 = hashlib.sha256(fixture.old_profile_bytes).hexdigest()
    record = {
        "schema_version": 1,
        "timestamp_ns": 1,
        "run_id": "continuous-test-run",
        "generation_family": "family-continuous-test",
        "from_config_sha256": "d" * 64,
        "to_config_sha256": legacy_hash,
        "from_profile": "profile-legacy.yaml",
        "to_profile": fixture.old_profile.name,
        "from_profile_sha256": "e" * 64,
        "to_profile_sha256": source_profile_sha256,
        "learner_step": 90,
        "examples_consumed": 46_080,
        "committed_replay_samples": 50_000,
        "from_source_commit": "9" * 40,
        "to_source_commit": "a" * 40,
        "reason": "legacy-default-boundary",
        "changes": [
            {
                "path": "arena.continuation_pairs_per_ring",
                "from": None,
                "to": 150,
            }
        ],
    }
    (fixture.root / "continuous-migrations.jsonl").write_text(
        json.dumps(record, sort_keys=True, separators=(",", ":")) + "\n",
        encoding="utf-8",
    )

    result = migration.migrate_continuous_profile(fixture.request)

    assert result["mode"] == "dry-run"
    assert result["source"]["config_sha256"] == legacy_hash


def test_chained_apply_advances_profile_and_source_authority(tmp_path: Path) -> None:
    fixture = _fixture(tmp_path)
    migration.migrate_continuous_profile(fixture.request, apply=True)
    current_profile = fixture.root / fixture.target_name
    next_raw = yaml.safe_load(current_profile.read_text(encoding="utf-8"))
    next_raw["learner"]["candidate_interval"] += 1_000
    next_candidate = tmp_path / "profile-next-candidate.yaml"
    next_candidate.write_text(
        yaml.safe_dump(next_raw, sort_keys=False),
        encoding="utf-8",
    )
    request = migration.MigrationRequest(
        run_root=fixture.root,
        old_profile=current_profile,
        new_profile=next_candidate,
        target_profile_name="profile-throughput-v3.yaml",
        reason="extend-candidate-cadence",
        from_source_commit="b" * 40,
        to_source_commit="c" * 40,
    )

    migration.migrate_continuous_profile(request, apply=True)

    records = [
        json.loads(line)
        for line in (fixture.root / "continuous-migrations.jsonl")
        .read_text(encoding="utf-8")
        .splitlines()
    ]
    assert len(records) == 2
    assert records[1]["from_profile"] == fixture.target_name
    assert records[1]["from_config_sha256"] == records[0]["to_config_sha256"]
    assert records[1]["from_profile_sha256"] == records[0]["to_profile_sha256"]
    assert records[1]["from_source_commit"] == records[0]["to_source_commit"]
    assert (fixture.root / "source-commit.txt").read_text(encoding="utf-8") == (
        f"{'c' * 40}\n"
    )


def test_chained_migration_accepts_verified_plateau_cutover(tmp_path: Path) -> None:
    fixture = _fixture(tmp_path)
    first = migration.migrate_continuous_profile(fixture.request, apply=True)
    current_profile = fixture.root / fixture.target_name
    chain_head = json.loads(
        (fixture.root / "continuous-migrations.jsonl")
        .read_text(encoding="utf-8")
        .splitlines()[-1]
    )
    original = torch.load(fixture.checkpoint, map_location="cpu", weights_only=True)

    def checkpoint_at(step: int, examples: int) -> Path:
        payload = deepcopy(original)
        payload["step"] = step
        payload["extra"]["examples_consumed"] = examples
        staging = fixture.root / "learner" / "recovery" / f"staging-{step}.pt"
        torch.save(payload, staging)
        digest = hashlib.sha256(staging.read_bytes()).hexdigest()
        destination = staging.with_name(f"sha256-{digest}.pt")
        staging.replace(destination)
        return destination

    cutover_checkpoint = checkpoint_at(90, 46_080)
    recovery_checkpoint = checkpoint_at(95, 48_640)
    cutover_created_ns = int(chain_head["timestamp_ns"]) + 1
    _write_json(
        fixture.root / "learner" / "resume-cutover.json",
        {
            "format": "startrain.resume-cutover",
            "schema_version": 1,
            "checkpoint": f"recovery/{cutover_checkpoint.name}",
            "checkpoint_sha256": cutover_checkpoint.stem.removeprefix("sha256-"),
            "checkpoint_bytes": cutover_checkpoint.stat().st_size,
            "step": 90,
            "run_id": "continuous-test-run",
            "generation_family": "family-continuous-test",
            "created_ns": cutover_created_ns,
        },
    )
    _write_json(
        fixture.root / "learner" / "recovery.json",
        {
            "format": "startrain.recovery-pointer",
            "schema_version": 1,
            "checkpoint": f"recovery/{recovery_checkpoint.name}",
            "checkpoint_sha256": recovery_checkpoint.stem.removeprefix("sha256-"),
            "checkpoint_bytes": recovery_checkpoint.stat().st_size,
            "step": 95,
            "epoch": 8,
            "examples_consumed": 48_640,
            "run_id": "continuous-test-run",
            "generation_family": "family-continuous-test",
            "updated_ns": cutover_created_ns + 1,
        },
    )
    heartbeat = json.loads(
        (fixture.root / "status" / "learner.heartbeat.json").read_text()
    )
    heartbeat.update(
        {
            "step": 95,
            "examples_consumed": 48_640,
            "heartbeat_ns": cutover_created_ns + 2,
        }
    )
    _write_json(fixture.root / "status" / "learner.heartbeat.json", heartbeat)

    target = yaml.safe_load(current_profile.read_text(encoding="utf-8"))
    target["arena"]["continuation_pairs_per_ring"] = 20
    candidate = tmp_path / "profile-after-cutover.yaml"
    candidate.write_text(yaml.safe_dump(target, sort_keys=False), encoding="utf-8")
    request = migration.MigrationRequest(
        run_root=fixture.root,
        old_profile=current_profile,
        new_profile=candidate,
        target_profile_name="profile-throughput-v3.yaml",
        reason="migrate-after-plateau-cutover",
        from_source_commit="b" * 40,
        to_source_commit="c" * 40,
    )

    result = migration.migrate_continuous_profile(request)

    assert result["mode"] == "dry-run"
    record = result["boundary"]
    assert record["learner_step"] == 95
    plan = migration.plan_migration(request)
    assert plan.migration_record["boundary_reset_from_learner_step"] == 100
    assert plan.migration_record["resume_cutover_step"] == 90
    assert plan.migration_record["resume_cutover_created_ns"] == cutover_created_ns
    assert first["target"]["profile"] == current_profile.name


def test_live_coordinator_lock_is_rejected_without_writes(tmp_path: Path) -> None:
    fixture = _fixture(tmp_path)
    _write_json(
        fixture.root / "coordinator.lock",
        {"pid": os.getpid(), "created_ns": 30},
    )
    before = _snapshot(fixture.root)

    with pytest.raises(migration.MigrationError, match="is live"):
        migration.migrate_continuous_profile(fixture.request, apply=True)

    assert _snapshot(fixture.root) == before


def test_dead_coordinator_lock_is_replaced_and_backed_up(tmp_path: Path) -> None:
    fixture = _fixture(tmp_path)
    stale_lock = {"pid": 999_999_998, "created_ns": 30}
    _write_json(fixture.root / "coordinator.lock", stale_lock)

    result = migration.migrate_continuous_profile(fixture.request, apply=True)

    backup = Path(result["backup_bundle"])
    assert (
        json.loads((backup / "coordinator.lock").read_text(encoding="utf-8"))
        == stale_lock
    )
    assert not (fixture.root / "coordinator.lock").exists()


def test_disallowed_semantic_diff_is_rejected(tmp_path: Path) -> None:
    fixture = _fixture(tmp_path)
    target = yaml.safe_load(fixture.candidate_profile.read_text(encoding="utf-8"))
    target["optimizer"]["adamw_lr"] = 0.0004
    fixture.candidate_profile.write_text(
        yaml.safe_dump(target, sort_keys=False),
        encoding="utf-8",
    )
    before = _snapshot(fixture.root)

    with pytest.raises(migration.MigrationError, match="optimizer configuration"):
        migration.migrate_continuous_profile(fixture.request, apply=True)

    assert _snapshot(fixture.root) == before


def test_incomplete_replay_history_is_rejected(tmp_path: Path) -> None:
    fixture = _fixture(tmp_path)
    with sqlite3.connect(fixture.root / "replay" / "manifest.sqlite3") as connection:
        connection.execute("UPDATE run_counters SET history_complete = 0")
    before = _snapshot(fixture.root)

    with pytest.raises(migration.MigrationError, match="history_complete is false"):
        migration.migrate_continuous_profile(fixture.request, apply=True)

    assert _snapshot(fixture.root) == before


@pytest.mark.parametrize(
    ("mutate", "message"),
    [
        (
            lambda fixture: _rewrite_json_field(
                fixture.root / "learner" / "recovery.json",
                "run_id",
                "another-run",
            ),
            "recovery pointer identity",
        ),
        (
            lambda fixture: fixture.checkpoint.write_bytes(b"corrupted checkpoint"),
            "checkpoint byte length|checkpoint SHA-256",
        ),
        (
            lambda fixture: (fixture.root / "profile.sha256").write_text(
                f"{'0' * 64}  {fixture.old_profile}\n",
                encoding="utf-8",
            ),
            "checksum does not match",
        ),
    ],
)
def test_identity_or_hash_mismatch_is_rejected(
    tmp_path: Path,
    mutate: Callable[[_Fixture], object],
    message: str,
) -> None:
    fixture = _fixture(tmp_path)
    mutate(fixture)
    before = _snapshot(fixture.root)

    with pytest.raises(migration.MigrationError, match=message):
        migration.migrate_continuous_profile(fixture.request, apply=True)

    assert _snapshot(fixture.root) == before


def test_checkpoint_payload_identity_mismatch_is_rejected(tmp_path: Path) -> None:
    fixture = _fixture(tmp_path)
    payload = torch.load(fixture.checkpoint, map_location="cpu", weights_only=True)
    payload["extra"]["run_id"] = "another-run"
    replacement = fixture.checkpoint.with_name("replacement.pt")
    torch.save(payload, replacement)
    digest = hashlib.sha256(replacement.read_bytes()).hexdigest()
    renamed = fixture.checkpoint.with_name(f"sha256-{digest}.pt")
    replacement.replace(renamed)
    fixture.checkpoint.unlink()
    pointer = json.loads(
        (fixture.root / "learner" / "recovery.json").read_text(encoding="utf-8")
    )
    pointer.update(
        {
            "checkpoint": f"recovery/{renamed.name}",
            "checkpoint_sha256": digest,
            "checkpoint_bytes": renamed.stat().st_size,
        }
    )
    _write_json(fixture.root / "learner" / "recovery.json", pointer)

    with pytest.raises(migration.MigrationError, match="payload run identity"):
        migration.migrate_continuous_profile(fixture.request, apply=True)

    assert not (fixture.root / fixture.target_name).exists()
    assert not (fixture.root / "continuous-migrations.jsonl").exists()


def _rewrite_json_field(path: Path, key: str, value: object) -> None:
    payload = json.loads(path.read_text(encoding="utf-8"))
    payload[key] = value
    _write_json(path, payload)


def test_autonomous_profiles_are_explicitly_rejected(tmp_path: Path) -> None:
    fixture = _fixture(tmp_path)
    source = yaml.safe_load(fixture.old_profile.read_text(encoding="utf-8"))
    source["orchestration"]["autonomous"]["enabled"] = True
    fixture.old_profile.chmod(0o644)
    fixture.old_profile.write_text(
        yaml.safe_dump(source, sort_keys=False),
        encoding="utf-8",
    )
    before = _snapshot(fixture.root)

    with pytest.raises(migration.MigrationError, match="autonomous profiles"):
        migration.migrate_continuous_profile(fixture.request, apply=True)

    assert _snapshot(fixture.root) == before


def test_apply_rolls_back_all_partial_writes_on_failure(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fixture = _fixture(tmp_path)
    before = _snapshot(fixture.root)
    original = migration._atomic_write_bytes
    calls = 0

    def fail_after_log_append(
        path: Path,
        data: bytes,
        *,
        mode: int,
        overwrite: bool,
    ) -> None:
        nonlocal calls
        calls += 1
        if calls == 3:
            raise OSError("injected metadata write failure")
        original(path, data, mode=mode, overwrite=overwrite)

    monkeypatch.setattr(migration, "_atomic_write_bytes", fail_after_log_append)

    with pytest.raises(migration.MigrationError, match="was rolled back"):
        migration.migrate_continuous_profile(fixture.request, apply=True)

    assert calls == 3
    assert _snapshot(fixture.root) == before
    assert not (fixture.root / fixture.target_name).exists()
    assert not (fixture.root / "continuous-migrations.jsonl").exists()
    assert not (fixture.root / "migration-backups").exists()
    assert not (fixture.root / "coordinator.lock").exists()
