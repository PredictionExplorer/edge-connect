from __future__ import annotations

import hashlib
import json
import os
import sqlite3
from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path

import pytest
import yaml

from scripts import migrate_autonomous_profile as migration
from startrain.config import load_config


@dataclass(frozen=True)
class _Fixture:
    root: Path
    old_profile: Path
    candidate_profile: Path
    target_name: str
    request: migration.MigrationRequest
    source_config_sha256: str
    old_profile_bytes: bytes
    old_provenance_bytes: bytes


def _write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, sort_keys=True, separators=(",", ":")) + "\n",
        encoding="utf-8",
    )


def _fixture(tmp_path: Path) -> _Fixture:
    root = tmp_path / "active-run"
    root.mkdir()
    base_path = Path(__file__).parents[1] / "configs" / "h100-8gpu-autonomous.yaml"
    old_raw = yaml.safe_load(base_path.read_text(encoding="utf-8"))
    old_raw["orchestration"]["run_id"] = "autonomous-test-run"
    old_raw["orchestration"]["directories"]["root"] = str(root)
    old_raw["orchestration"]["gpus"][7]["actor_lanes"] = 1
    old_raw["orchestration"]["promotion"]["gpu_id"] = 7
    old_raw["orchestration"]["promotion"].pop("max_waves_per_lease")
    old_raw["orchestration"]["promotion"].pop("inter_wave_cooldown_seconds")
    old_raw["orchestration"].pop("historical_evaluation", None)
    old_raw["data"].pop("min_batches_for_workers", None)

    old_profile = root / "profile-cadence-v2.yaml"
    old_profile.write_text(yaml.safe_dump(old_raw, sort_keys=False), encoding="utf-8")
    old_profile.chmod(0o444)

    target_raw = deepcopy(old_raw)
    target_raw["data"]["min_batches_for_workers"] = 8
    target_raw["learner"]["target_updates_per_new_sample"] = 1.25
    target_raw["learner"]["candidate_interval_examples"] = 6_250_000
    target_raw["learner"]["selfplay_snapshot_interval_examples"] = 3_750_000
    target_raw["learner"]["selfplay_snapshot_warmup_interval_examples"] = 1_250_000
    target_raw["arena"]["pairs_per_ring"] = 5
    target_raw["arena"]["minimum_pairs_per_ring"] = 15
    target_raw["orchestration"]["historical_evaluation"] = {
        "enabled": True,
        "every_promotions": 2,
        "anchors_per_evaluation": 1,
        "pairs_per_ring": 5,
        "max_pairs_per_ring": 10,
    }
    candidate_profile = tmp_path / "profile-candidate.yaml"
    candidate_profile.write_text(
        yaml.safe_dump(target_raw, sort_keys=False),
        encoding="utf-8",
    )

    run_id = "autonomous-test-run"
    family = "family-autonomous-test"
    _write_json(
        root / "run.json",
        {
            "schema_version": 1,
            "run_id": run_id,
            "generation_family": family,
            "created_ns": 1,
        },
    )

    legacy_config = load_config(old_profile).as_dict()
    legacy_config["data"].pop("min_batches_for_workers")
    source_config_sha256 = hashlib.sha256(
        json.dumps(
            legacy_config,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    ).hexdigest()
    assert source_config_sha256 != migration.canonical_config_sha256(
        load_config(old_profile)
    )
    provenance_path = root / "autonomous-provenance.json"
    _write_json(
        provenance_path,
        {
            "schema_version": 1,
            "mode": "random-init-selfplay-only",
            "run_id": run_id,
            "generation_family": family,
            "train_seed": 17,
            "elo_anchor_step": 0,
            "external_weights": False,
            "external_replay": False,
            "external_positions": False,
            "config_sha256": source_config_sha256,
        },
    )
    (root / "autonomous-migrations.jsonl").write_text(
        json.dumps(
            {
                "schema_version": 1,
                "timestamp_ns": 10,
                "run_id": run_id,
                "generation_family": family,
                "from_config_sha256": "e" * 64,
                "to_config_sha256": source_config_sha256,
                "from_profile": "profile.yaml",
                "to_profile": old_profile.name,
                "learner_step": 80,
                "examples_consumed": 40_960,
                "from_source_commit": "a" * 40,
                "to_source_commit": "b" * 40,
                "reason": "decouple-selfplay-promotion-cadence",
            },
            sort_keys=True,
            separators=(",", ":"),
        )
        + "\n",
        encoding="utf-8",
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
            "step": step,
            "examples_consumed": 50_000,
        },
    )
    checkpoint_bytes = b"verified recovery checkpoint"
    checkpoint_sha256 = hashlib.sha256(checkpoint_bytes).hexdigest()
    checkpoint = root / "learner" / "recovery" / f"sha256-{checkpoint_sha256}.pt"
    checkpoint.parent.mkdir(parents=True)
    checkpoint.write_bytes(checkpoint_bytes)
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
    _write_json(
        root / "learner" / "cadence.json",
        {
            "schema_version": 1,
            "run_id": run_id,
            "generation_family": family,
            "candidate_examples": 50_000,
            "selfplay_examples": 49_000,
            "updated_ns": 22,
        },
    )
    champion_identity = f"sha256-{'c' * 64}"
    _write_json(
        root / "learner" / "champion.json",
        {
            "format": "startrain.model-pointer",
            "schema_version": 2,
            "role": "champion",
            "manifest": "manifests/manifest.json",
            "manifest_sha256": "d" * 64,
            "manifest_bytes": 123,
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
                generation_family TEXT NOT NULL
            );
            CREATE TABLE run_counters (
                run_id TEXT NOT NULL,
                generation_family TEXT NOT NULL,
                committed_samples INTEGER NOT NULL,
                history_complete INTEGER NOT NULL,
                PRIMARY KEY(run_id, generation_family)
            );
            """
        )
        connection.execute(
            "INSERT INTO runs(run_id, generation_family) VALUES (?, ?)",
            (run_id, family),
        )
        connection.execute(
            """
            INSERT INTO run_counters(
                run_id, generation_family, committed_samples, history_complete
            ) VALUES (?, ?, ?, 1)
            """,
            (run_id, family, 60_000),
        )

    old_profile_bytes = old_profile.read_bytes()
    (root / "profile.sha256").write_text(
        f"{hashlib.sha256(old_profile_bytes).hexdigest()}  {old_profile}\n",
        encoding="utf-8",
    )
    request = migration.MigrationRequest(
        run_root=root,
        old_profile=old_profile,
        new_profile=candidate_profile,
        target_profile_name="profile-elo-v3.yaml",
        reason="prospective-utd-and-evaluation-cadence",
        from_source_commit="b" * 40,
        to_source_commit="c" * 40,
    )
    return _Fixture(
        root=root,
        old_profile=old_profile,
        candidate_profile=candidate_profile,
        target_name=request.target_profile_name,
        request=request,
        source_config_sha256=source_config_sha256,
        old_profile_bytes=old_profile_bytes,
        old_provenance_bytes=provenance_path.read_bytes(),
    )


def _snapshot(root: Path) -> dict[str, tuple[bytes, int]]:
    return {
        str(path.relative_to(root)): (path.read_bytes(), path.stat().st_mode & 0o777)
        for path in sorted(root.rglob("*"))
        if path.is_file()
    }


def test_dry_run_does_not_mutate_run(tmp_path: Path) -> None:
    fixture = _fixture(tmp_path)
    before = _snapshot(fixture.root)

    result = migration.migrate_autonomous_profile(fixture.request)

    assert result["mode"] == "dry-run"
    assert result["source"]["config_sha256"] == fixture.source_config_sha256
    assert result["source"]["canonical_matches_authority"] is False
    assert _snapshot(fixture.root) == before
    assert not (fixture.root / fixture.target_name).exists()
    assert not (fixture.root / "learner" / "utd-segment.json").exists()
    assert not (fixture.root / "migration-backups").exists()


def test_dry_run_uses_recovery_when_stopped_heartbeat_is_slightly_ahead(
    tmp_path: Path,
) -> None:
    fixture = _fixture(tmp_path)
    heartbeat_path = fixture.root / "status" / "learner.heartbeat.json"
    heartbeat = json.loads(heartbeat_path.read_text(encoding="utf-8"))
    heartbeat["step"] += 50
    heartbeat["examples_consumed"] += 25_600
    _write_json(heartbeat_path, heartbeat)

    result = migration.migrate_autonomous_profile(fixture.request)

    assert result["boundary"]["learner_step"] == 100
    assert result["boundary"]["discarded_uncheckpointed_steps"] == 50


def test_cli_defaults_to_json_dry_run(tmp_path: Path, capsys) -> None:
    fixture = _fixture(tmp_path)
    before = _snapshot(fixture.root)

    exit_code = migration.main(
        [
            "--run-root",
            str(fixture.root),
            "--old-profile",
            str(fixture.old_profile),
            "--new-profile",
            str(fixture.candidate_profile),
            "--target-profile-name",
            fixture.target_name,
            "--reason",
            "prospective-utd-and-evaluation-cadence",
            "--from-source-commit",
            "b" * 40,
            "--to-source-commit",
            "c" * 40,
        ]
    )

    output = json.loads(capsys.readouterr().out)
    assert exit_code == 0
    assert output["status"] == "ok"
    assert output["mode"] == "dry-run"
    assert _snapshot(fixture.root) == before


def test_apply_writes_target_hash_utd_boundary_and_backup(tmp_path: Path) -> None:
    fixture = _fixture(tmp_path)

    result = migration.migrate_autonomous_profile(fixture.request, apply=True)

    target = fixture.root / fixture.target_name
    target_config_hash = migration.canonical_config_sha256(load_config(target))
    assert result["mode"] == "apply"
    assert target.read_bytes() == fixture.candidate_profile.read_bytes()
    assert target.stat().st_mode & 0o222 == 0

    provenance = json.loads(
        (fixture.root / "autonomous-provenance.json").read_text(encoding="utf-8")
    )
    assert provenance["config_sha256"] == target_config_hash
    utd = json.loads(
        (fixture.root / "learner" / "utd-segment.json").read_text(encoding="utf-8")
    )
    assert utd == {
        "schema_version": 1,
        "run_id": "autonomous-test-run",
        "generation_family": "family-autonomous-test",
        "target_updates_per_new_sample": 1.25,
        "baseline_examples_consumed": 51_200,
        "baseline_committed_replay_samples": 60_000,
        "created_ns": utd["created_ns"],
    }

    records = [
        json.loads(line)
        for line in (fixture.root / "autonomous-migrations.jsonl")
        .read_text(encoding="utf-8")
        .splitlines()
    ]
    assert len(records) == 2
    record = records[-1]
    assert record["from_config_sha256"] == fixture.source_config_sha256
    assert record["to_config_sha256"] == target_config_hash
    assert record["committed_replay_samples"] == 60_000
    assert record["learner_step"] == 100
    assert record["examples_consumed"] == 51_200
    assert record["champion_model_identity"] == f"sha256-{'c' * 64}"

    profile_digest = hashlib.sha256(target.read_bytes()).hexdigest()
    assert (fixture.root / "profile.sha256").read_text(encoding="utf-8") == (
        f"{profile_digest}  {target}\n"
    )
    assert target.with_suffix(".sha256").read_text(encoding="utf-8") == (
        f"{profile_digest}  {target}\n"
    )
    backup = Path(result["backup_bundle"])
    assert backup.is_dir()
    assert (backup / fixture.old_profile.name).read_bytes() == fixture.old_profile_bytes
    assert (
        backup / "autonomous-provenance.json"
    ).read_bytes() == fixture.old_provenance_bytes
    assert (backup / "learner" / "cadence.json").is_file()
    assert (backup / "learner" / "recovery.json").is_file()
    assert len(list((backup / "learner" / "recovery").glob("sha256-*.pt"))) == 1
    assert (backup / "learner" / "champion.json").is_file()
    assert (backup / "status" / "learner.heartbeat.json").is_file()
    assert (backup / "autonomous-migrations.jsonl").is_file()
    assert json.loads((backup / "manifest.json").read_text())["to_profile"] == (
        fixture.target_name
    )
    assert not (fixture.root / "coordinator.lock").exists()


def test_live_coordinator_lock_refuses_migration(tmp_path: Path) -> None:
    fixture = _fixture(tmp_path)
    _write_json(
        fixture.root / "coordinator.lock",
        {"pid": os.getpid(), "created_ns": 30},
    )
    before = _snapshot(fixture.root)

    with pytest.raises(migration.MigrationError, match="is live"):
        migration.migrate_autonomous_profile(fixture.request, apply=True)

    assert _snapshot(fixture.root) == before


def test_immutable_profile_diff_refuses_migration(tmp_path: Path) -> None:
    fixture = _fixture(tmp_path)
    target = yaml.safe_load(fixture.candidate_profile.read_text(encoding="utf-8"))
    target["optimizer"]["adamw_lr"] = 0.0004
    fixture.candidate_profile.write_text(
        yaml.safe_dump(target, sort_keys=False),
        encoding="utf-8",
    )
    before = _snapshot(fixture.root)

    with pytest.raises(migration.MigrationError, match="optimizer configuration"):
        migration.migrate_autonomous_profile(fixture.request, apply=True)

    assert _snapshot(fixture.root) == before


def test_incomplete_replay_history_refuses_migration(tmp_path: Path) -> None:
    fixture = _fixture(tmp_path)
    with sqlite3.connect(fixture.root / "replay" / "manifest.sqlite3") as connection:
        connection.execute("UPDATE run_counters SET history_complete = 0")
    before = _snapshot(fixture.root)

    with pytest.raises(migration.MigrationError, match="history_complete is false"):
        migration.migrate_autonomous_profile(fixture.request, apply=True)

    assert _snapshot(fixture.root) == before


def test_mismatched_source_profile_checksum_refuses_migration(tmp_path: Path) -> None:
    fixture = _fixture(tmp_path)
    (fixture.root / "profile.sha256").write_text(
        f"{'0' * 64}  {fixture.old_profile}\n",
        encoding="utf-8",
    )
    before = _snapshot(fixture.root)

    with pytest.raises(migration.MigrationError, match="not authenticated"):
        migration.migrate_autonomous_profile(fixture.request, apply=True)

    assert _snapshot(fixture.root) == before


def test_chained_migration_preserves_unchanged_utd_segment(tmp_path: Path) -> None:
    fixture = _fixture(tmp_path)
    migration.migrate_autonomous_profile(fixture.request, apply=True)
    first_profile = fixture.root / fixture.target_name
    first_segment = (fixture.root / "learner" / "utd-segment.json").read_bytes()
    second_raw = yaml.safe_load(first_profile.read_text(encoding="utf-8"))
    second_raw["orchestration"]["historical_evaluation"]["every_promotions"] = 3
    staged = tmp_path / "profile-second.yaml"
    staged.write_text(yaml.safe_dump(second_raw, sort_keys=False), encoding="utf-8")
    request = migration.MigrationRequest(
        run_root=fixture.root,
        old_profile=first_profile,
        new_profile=staged,
        target_profile_name="profile-elo-v4.yaml",
        reason="adjust-historical-evaluation-cadence",
        from_source_commit="c" * 40,
        to_source_commit="d" * 40,
    )

    migration.migrate_autonomous_profile(request, apply=True)

    assert (fixture.root / "learner" / "utd-segment.json").read_bytes() == first_segment


def test_gpu_topology_migration_is_narrowly_allowlisted(tmp_path: Path) -> None:
    run_root = tmp_path / "run"
    run_root.mkdir()
    profile = Path(__file__).parents[1] / "configs" / "h100-8gpu-autonomous.yaml"
    target_raw = yaml.safe_load(profile.read_text(encoding="utf-8"))
    target_raw["orchestration"]["run_id"] = "topology-migration"
    target_raw["orchestration"]["directories"]["root"] = str(run_root)
    source_raw = deepcopy(target_raw)
    source_raw["orchestration"]["gpus"][7]["actor_lanes"] = 1
    source_raw["orchestration"]["promotion"]["gpu_id"] = 7
    source_raw["orchestration"]["promotion"].pop("max_waves_per_lease")
    source_raw["orchestration"]["promotion"].pop("inter_wave_cooldown_seconds")
    source_raw["orchestration"]["historical_evaluation"]["enabled"] = True
    source = tmp_path / "source.yaml"
    target = tmp_path / "target.yaml"
    source.write_text(yaml.safe_dump(source_raw, sort_keys=False), encoding="utf-8")
    target.write_text(yaml.safe_dump(target_raw, sort_keys=False), encoding="utf-8")

    changes = migration._validate_profile_pair(
        load_config(source),
        load_config(target),
        run_root=run_root,
    )

    paths = {path for path, _, _ in changes}
    assert {
        "orchestration.gpus.7.actor_lanes",
        "orchestration.promotion.gpu_id",
        "orchestration.promotion.max_waves_per_lease",
        "orchestration.promotion.inter_wave_cooldown_seconds",
        "orchestration.historical_evaluation.enabled",
    } <= paths

    invalid_raw = deepcopy(target_raw)
    invalid_raw["orchestration"]["gpus"][7]["actor_lanes"] = 3
    invalid = tmp_path / "invalid.yaml"
    invalid.write_text(yaml.safe_dump(invalid_raw, sort_keys=False), encoding="utf-8")
    with pytest.raises(migration.MigrationError, match="one lane to two"):
        migration._validate_profile_pair(
            load_config(source),
            load_config(invalid),
            run_root=run_root,
        )

    unsafe_raw = deepcopy(target_raw)
    unsafe_raw["orchestration"]["promotion"]["max_waves_per_lease"] = 2
    unsafe = tmp_path / "unsafe.yaml"
    unsafe.write_text(yaml.safe_dump(unsafe_raw, sort_keys=False), encoding="utf-8")
    with pytest.raises(migration.MigrationError, match="learner-shared promotion"):
        migration._validate_profile_pair(
            load_config(source),
            load_config(unsafe),
            run_root=run_root,
        )

    measured_raw = deepcopy(target_raw)
    measured_raw["orchestration"]["historical_evaluation"]["enabled"] = True
    measured = tmp_path / "measured.yaml"
    measured.write_text(
        yaml.safe_dump(measured_raw, sort_keys=False),
        encoding="utf-8",
    )
    with pytest.raises(migration.MigrationError, match="learner-shared promotion"):
        migration._validate_profile_pair(
            load_config(source),
            load_config(measured),
            run_root=run_root,
        )
    with pytest.raises(migration.MigrationError, match="learner-shared promotion"):
        migration._validate_profile_pair(
            load_config(target),
            load_config(measured),
            run_root=run_root,
        )

    drift_raw = deepcopy(target_raw)
    drift_raw["orchestration"]["gpus"][7]["cpu_threads"] = 9
    drift = tmp_path / "drift.yaml"
    drift.write_text(yaml.safe_dump(drift_raw, sort_keys=False), encoding="utf-8")
    with pytest.raises(migration.MigrationError, match="immutable fields"):
        migration._validate_profile_pair(
            load_config(source),
            load_config(drift),
            run_root=run_root,
        )
