from __future__ import annotations

import hashlib
import json
import os
import sqlite3
from copy import deepcopy
from dataclasses import dataclass, replace
from pathlib import Path
from types import SimpleNamespace
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


def _fixture(
    tmp_path: Path, base_profile: str = "h100-8gpu-throughput.yaml"
) -> _Fixture:
    root = tmp_path / "active-run"
    root.mkdir()
    base_path = Path(__file__).parents[1] / "configs" / base_profile
    old_raw = yaml.safe_load(base_path.read_text(encoding="utf-8"))
    run_id = "continuous-test-run"
    family = "family-continuous-test"
    created_ns = 11
    old_raw["orchestration"]["run_id"] = run_id
    old_raw["orchestration"]["directories"]["root"] = str(root)
    old_raw["orchestration"]["autonomous"] = {"enabled": False}
    old_raw["orchestration"]["promotion"]["finish_inflight_candidate"] = False
    old_raw["arena"]["continuation_pairs_per_ring"] = min(
        150, old_raw["arena"]["max_pairs_per_ring"]
    )

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
    target_raw["arena"]["continuation_pairs_per_ring"] = min(
        25, target_raw["arena"]["max_pairs_per_ring"]
    )
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


def _source_only_request(fixture: _Fixture) -> migration.MigrationRequest:
    # The general migration fixture deliberately starts with an obsolete
    # promotion policy. A source-only fixture must already satisfy today's
    # unchanged validation contract on both sides of the runtime upgrade.
    raw = yaml.safe_load(fixture.old_profile.read_text())
    raw["orchestration"]["promotion"]["finish_inflight_candidate"] = True
    raw["arena"]["continuation_pairs_per_ring"] = 25
    fixture.old_profile.chmod(0o644)
    fixture.old_profile.write_text(yaml.safe_dump(raw, sort_keys=False))
    fixture.old_profile.chmod(0o444)
    digest = hashlib.sha256(fixture.old_profile.read_bytes()).hexdigest()
    (fixture.root / "profile.sha256").write_text(f"{digest}  {fixture.old_profile}\n")
    fixture.candidate_profile.chmod(0o644)
    fixture.candidate_profile.write_bytes(fixture.old_profile.read_bytes())
    return replace(fixture.request, reason="upgrade-runtime-with-unchanged-profile")


def test_source_only_dry_run_requires_no_configuration_change_or_writes(tmp_path):
    fixture = _fixture(tmp_path)
    request = _source_only_request(fixture)
    before = _snapshot(fixture.root)
    result = migration.migrate_continuous_profile(request)
    assert result["kind"] == "source-only"
    assert result["changes"] == []
    assert result["source"]["config_sha256"] == result["target"]["config_sha256"]
    assert result["source"]["source_commit"] == "a" * 40
    assert result["target"]["source_commit"] == "b" * 40
    assert _snapshot(fixture.root) == before


def test_source_only_apply_preserves_durable_training_and_utd_state(tmp_path):
    fixture = _fixture(tmp_path)
    _with_update_to_data(
        fixture,
        old_target=1.0,
        new_target=1.0,
        old_intervals=(2_000_000, 1_000_000),
        new_intervals=(2_000_000, 1_000_000),
        segment={
            "schema_version": 1,
            "run_id": "continuous-test-run",
            "generation_family": "family-continuous-test",
            "target_updates_per_new_sample": 1.0,
            "baseline_examples_consumed": 1024,
            "baseline_committed_replay_samples": 2048,
        },
    )
    request = _source_only_request(fixture)
    before = _snapshot(fixture.root)
    result = migration.migrate_continuous_profile(request, apply=True)
    target = fixture.root / fixture.target_name
    assert target.read_bytes() == fixture.old_profile.read_bytes()
    assert target.stat().st_mode & 0o222 == 0
    assert (fixture.root / "source-commit.txt").read_text() == f"{'b' * 40}\n"
    assert result["utd_segment"] is None
    for path, contents in before.items():
        if path.startswith(("learner/", "status/", "replay/")) or path == "run.json":
            assert _snapshot(fixture.root)[path] == contents
    record = json.loads((fixture.root / "continuous-migrations.jsonl").read_text())
    assert record["kind"] == "source-only"
    assert record["changes"] == []
    assert record["from_config_sha256"] == record["to_config_sha256"]
    assert record["learner_step"] == result["boundary"]["learner_step"]
    backup = Path(result["backup_bundle"])
    assert (
        backup / "source-commit.txt"
    ).read_bytes() == fixture.old_source_commit_bytes
    assert (backup / "learner/utd-segment.json").read_bytes() == before[
        "learner/utd-segment.json"
    ][0]


@pytest.mark.parametrize(
    "updates,message",
    [
        ({"to_source_commit": None}, "explicit --to-source-commit"),
        ({"to_source_commit": "a" * 40}, "different target source commit"),
        ({"to_source_commit": "a" * 7}, "different target source commit"),
        ({"to_source_commit": "not-a-commit"}, "to_source_commit"),
        ({"from_source_commit": "c" * 40}, "current source authority"),
    ],
)
def test_source_only_rejects_missing_same_or_mismatched_source(
    tmp_path, updates, message
):
    fixture = _fixture(tmp_path)
    request = replace(_source_only_request(fixture), **updates)
    before = _snapshot(fixture.root)
    with pytest.raises(migration.MigrationError, match=message):
        migration.migrate_continuous_profile(request, apply=True)
    assert _snapshot(fixture.root) == before


def test_source_only_requires_recorded_current_authority(tmp_path):
    fixture = _fixture(tmp_path)
    request = _source_only_request(fixture)
    (fixture.root / "source-commit.txt").unlink()
    with pytest.raises(migration.MigrationError, match="current source authority"):
        migration.plan_migration(request)


def test_source_only_preserves_lock_and_apply_fingerprint_checks(tmp_path):
    fixture = _fixture(tmp_path)
    request = _source_only_request(fixture)
    lock = fixture.root / "coordinator.lock"
    _write_json(lock, {"pid": os.getpid(), "created_ns": 30})
    with pytest.raises(migration.MigrationError, match="live"):
        migration.plan_migration(request)
    lock.unlink()
    plan = migration.plan_migration(request)
    (fixture.root / "source-commit.txt").write_text(f"{'c' * 40}\n")
    before = _snapshot(fixture.root)
    with pytest.raises(migration.MigrationError, match="validated input changed"):
        migration.apply_migration(plan)
    assert _snapshot(fixture.root) == before


def test_source_only_records_chain_into_another_source_and_profile_change(tmp_path):
    fixture = _fixture(tmp_path)
    first = migration.plan_migration(_source_only_request(fixture))
    migration.apply_migration(first)
    second = migration.plan_migration(
        replace(
            fixture.request,
            old_profile=first.target_profile,
            target_profile_name="profile-source-v3.yaml",
            from_source_commit="b" * 40,
            to_source_commit="c" * 40,
        )
    )
    migration.apply_migration(second)
    raw = yaml.safe_load(fixture.candidate_profile.read_text())
    raw["learner"]["candidate_interval"] += 1000
    fixture.candidate_profile.write_text(yaml.safe_dump(raw))
    third = migration.plan_migration(
        replace(
            fixture.request,
            old_profile=second.target_profile,
            target_profile_name="profile-source-v4.yaml",
            from_source_commit="c" * 40,
            to_source_commit="d" * 40,
        )
    )
    migration.apply_migration(third)
    records = [
        json.loads(line)
        for line in (fixture.root / "continuous-migrations.jsonl")
        .read_text()
        .splitlines()
    ]
    assert [record["kind"] for record in records] == [
        "source-only",
        "source-only",
        "profile",
    ]
    for previous, current in zip(records, records[1:]):
        assert current["from_source_commit"] == previous["to_source_commit"]
        assert current["from_config_sha256"] == previous["to_config_sha256"]
    assert third.source_config_sha256 != third.target_config_sha256


def test_source_only_run_passes_startup_preflight_and_disaster_snapshot(tmp_path):
    from scripts.preflight_run_state import run_state_preflight
    from scripts.training_disaster_recovery import create_snapshot, verify_snapshot
    from startrain.config_compatibility import compatible_config_epoch_payloads
    from test_run_state_preflight import _fixture as populated_run

    state = populated_run(tmp_path)
    raw = yaml.safe_load(
        (Path(__file__).parents[1] / "configs/h100-8gpu-throughput.yaml").read_text()
    )
    for section in ("game", "model", "loss", "optimizer"):
        raw[section] = state.experiment.as_dict()[section]
    raw["orchestration"]["run_id"] = state.identity.run_id
    raw["orchestration"]["directories"]["root"] = str(state.root)
    raw["orchestration"]["autonomous"] = {"enabled": False}
    old = state.root / "profile-source-before.yaml"
    old.write_text(yaml.safe_dump(raw, sort_keys=False))
    old.chmod(0o444)
    digest = hashlib.sha256(old.read_bytes()).hexdigest()
    (state.root / "profile.sha256").write_text(f"{digest}  {old}\n")
    (state.root / "source-commit.txt").write_text(f"{'a' * 40}\n")
    _write_json(
        state.root / "status/learner.heartbeat.json",
        {
            "schema_version": 1,
            "worker": "learner",
            "pid": 999_999_999,
            "heartbeat_ns": 20,
            "phase": "stopped",
            "step": 10,
            "examples_consumed": 100,
        },
    )
    run_state_preflight(state.root, old, apply=True)
    staged = tmp_path / "profile-source-staged.yaml"
    staged.write_bytes(old.read_bytes())
    request = migration.MigrationRequest(
        old_profile=old,
        new_profile=staged,
        target_profile_name="profile-source-after.yaml",
        reason="source-only-startup-and-snapshot-test",
        from_source_commit="a" * 40,
        to_source_commit="b" * 40,
    )
    migration.migrate_continuous_profile(request, apply=True)
    target = state.root / request.target_profile_name
    report = run_state_preflight(state.root, target)
    assert report["status"] == "ok"
    assert report["migrations"] == []
    config = load_config(target)
    assert config.as_dict() in list(compatible_config_epoch_payloads(config.as_dict()))
    snapshot = create_snapshot(
        state.root,
        target,
        tmp_path / "snapshot-backup",
        enforce_separate_filesystem=False,
    )
    assert verify_snapshot(snapshot)["status"] == "ok"
    payload = json.loads(snapshot.read_text())
    assert "continuous-migrations.jsonl" in payload["catalog"]


@pytest.mark.parametrize(
    "field,value",
    [
        ("kind", "profile"),
        ("to_source_commit", "a" * 40),
        ("to_config_sha256", "d" * 64),
    ],
)
def test_source_only_chain_rejects_forged_noop_or_configuration_change(
    tmp_path, field, value
):
    fixture = _fixture(tmp_path)
    plan = migration.plan_migration(_source_only_request(fixture))
    migration.apply_migration(plan)
    path = fixture.root / "continuous-migrations.jsonl"
    record = json.loads(path.read_text())
    record[field] = value
    _write_json(path, record)
    with pytest.raises(migration.MigrationError, match="migration.*invalid"):
        migration.plan_migration(
            replace(
                fixture.request,
                old_profile=plan.target_profile,
                target_profile_name="profile-forged-v3.yaml",
                from_source_commit="b" * 40,
                to_source_commit="c" * 40,
            )
        )


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


def test_efficiency_rollout_preserves_two_step_cadence_and_prospective_utd(
    tmp_path: Path,
) -> None:
    from dataclasses import replace

    fixture = _fixture(tmp_path, "h100-8gpu-variant-stage-a.yaml")
    old = yaml.safe_load(fixture.old_profile.read_text())
    freshness = deepcopy(old)
    freshness["learner"]["selfplay_snapshot_interval_examples"] = freshness["learner"][
        "selfplay_snapshot_warmup_interval_examples"
    ]
    freshness_input = tmp_path / "freshness-input.yaml"
    freshness_input.write_text(yaml.safe_dump(freshness))
    first = migration.plan_migration(
        replace(
            fixture.request,
            new_profile=freshness_input,
            target_profile_name="profile-freshness.yaml",
            reason="keep-selfplay-fresh",
        )
    )
    migration.apply_migration(first)
    # A stopped legacy evaluation is retained intact. The new balanced contract
    # uses distinct filenames, so none of these partial pairs can enter it.
    pending = fixture.root / "arena" / f"sha256-{'d' * 64}-vs-sha256-{'c' * 64}.json"
    _write_json(pending, {"result_kind": "promotion", "terminal": False, "pairs": []})
    _write_json(fixture.root / "arena/promotion-status.json", {"terminal": False})
    pending_bytes = pending.read_bytes()
    final = yaml.safe_load(
        (
            Path(__file__).parents[1]
            / "configs/h100-8gpu-variant-efficiency-stage-b.yaml"
        ).read_text()
    )
    final["orchestration"]["run_id"] = old["orchestration"]["run_id"]
    final["orchestration"]["directories"]["root"] = str(fixture.root)
    final["orchestration"]["autonomous"] = deepcopy(old["orchestration"]["autonomous"])
    final_input = tmp_path / "efficiency-input.yaml"
    final_input.write_text(yaml.safe_dump(final))
    second = migration.plan_migration(
        replace(
            fixture.request,
            old_profile=first.target_profile,
            new_profile=final_input,
            target_profile_name="profile-efficiency.yaml",
            reason="prospective-efficiency-cutover",
            from_source_commit="b" * 40,
        )
    )
    migration.apply_migration(second)
    segment = json.loads((fixture.root / "learner/utd-segment.json").read_text())
    assert segment["target_updates_per_new_sample"] == 1.5
    assert segment["baseline_committed_replay_samples"] == 60000
    assert segment["baseline_examples_consumed"] == 51200
    active = load_config(second.target_profile)
    assert active.model == load_config(fixture.old_profile).model
    assert active.arena.balanced_cells and active.orchestration.cpu_actors
    assert active.learner.selfplay_snapshot_interval_examples == 1500000
    records = [
        json.loads(line)
        for line in (fixture.root / "continuous-migrations.jsonl")
        .read_text()
        .splitlines()
    ]
    assert len(records) == 2
    assert records[1]["from_config_sha256"] == records[0]["to_config_sha256"]
    assert records[1]["evaluation_contract_transition"]["kind"] == "legacy_to_balanced"
    assert pending.read_bytes() == pending_bytes
    assert fixture.old_profile.read_bytes() == fixture.old_profile_bytes


def test_ring_allocation_migration_is_limited_to_the_validated_schedule() -> None:
    old = {
        "orchestration": {
            "ring_mixture": {
                "rings": [4, 6, 8, 10],
                "step_weights": [
                    {"from_step": 150_000, "weights": [0.15, 0.15, 0.2, 0.5]}
                ],
            }
        }
    }
    new = deepcopy(old)
    new["orchestration"]["ring_mixture"]["step_weights"] = [
        {"from_step": 0, "weights": [0.25, 0.25, 0.25, 0.25]}
    ]
    differences = list(migration._profile_diffs(old, new))
    assert len(differences) == 1
    assert differences[0][0] == ("orchestration", "ring_mixture", "step_weights")
    assert differences[0][0] in migration._ALLOWED_PROFILE_PATHS
    new["orchestration"]["ring_mixture"]["rings"] = [4, 6, 8]
    assert any(
        path not in migration._ALLOWED_PROFILE_PATHS
        for path, _, _ in migration._profile_diffs(old, new)
    )


@pytest.mark.parametrize("allocation", ["handicap_share", "segments"])
@pytest.mark.parametrize("arena_state", ["absent", "terminal", "running", "resumable"])
def test_variant_allocation_migration_requires_a_terminal_arena_boundary(
    tmp_path: Path, allocation: str, arena_state: str
) -> None:
    fixture = _fixture(tmp_path)
    target = yaml.safe_load(fixture.candidate_profile.read_text())
    if allocation == "handicap_share":
        target["arena"]["segment_handicap_classic_share"] = 0.5
    else:
        target["arena"]["segment_pairs_per_ring"] = {"classic": 2}
    fixture.candidate_profile.write_text(yaml.safe_dump(target, sort_keys=False))
    arena_root = fixture.root / "arena"
    if arena_state != "absent":
        _write_json(
            arena_root / "promotion-status.json",
            {"terminal": arena_state != "running", "decision": "continue"},
        )
    result_path = arena_root / f"sha256-{'d' * 64}-vs-sha256-{'c' * 64}.json"
    if arena_state in ("terminal", "resumable"):
        _write_json(
            result_path,
            {
                "result_kind": "promotion",
                "terminal": arena_state == "terminal",
                "pairs": [{"variant": "handicap-4-double"}],
            },
        )
    before = _snapshot(fixture.root)
    if arena_state in ("running", "resumable"):
        with pytest.raises(migration.MigrationError, match="terminal arena boundary"):
            migration.migrate_continuous_profile(fixture.request)
    else:
        plan = migration.plan_migration(fixture.request)
        if result_path.exists():
            assert result_path in {item.path for item in plan.input_fingerprints}
    assert _snapshot(fixture.root) == before


def test_variant_allocation_boundary_is_rechecked_before_apply(tmp_path: Path) -> None:
    fixture = _fixture(tmp_path)
    target = yaml.safe_load(fixture.candidate_profile.read_text())
    target["arena"]["segment_handicap_classic_share"] = 0.5
    fixture.candidate_profile.write_text(yaml.safe_dump(target, sort_keys=False))
    plan = migration.plan_migration(fixture.request)
    _write_json(
        fixture.root / "arena" / f"sha256-{'d' * 64}-vs-sha256-{'c' * 64}.json",
        {"terminal": False, "pairs": [{"variant": "handicap-4-double"}]},
    )
    with pytest.raises(migration.MigrationError, match="resumable arena evidence"):
        migration._assert_inputs_unchanged(plan, check_lock=True)


def test_variant_boundary_requires_terminal_current_champion_crossplay(
    tmp_path: Path,
) -> None:
    fixture = _fixture(tmp_path)
    target = yaml.safe_load(fixture.candidate_profile.read_text())
    target["arena"]["segment_handicap_classic_share"] = 0.5
    fixture.candidate_profile.write_text(yaml.safe_dump(target, sort_keys=False))
    result_path = (
        fixture.root
        / "arena"
        / f"crossplay-sha256-{'c' * 64}-vs-sha256-{'b' * 64}.json"
    )
    result = {"terminal": False, "result_kind": "historical_crossplay", "pairs": [{}]}
    _write_json(result_path, result)
    with pytest.raises(migration.MigrationError, match="resumable arena evidence"):
        migration.plan_migration(fixture.request)
    result["terminal"] = True
    _write_json(result_path, result)
    migration.plan_migration(fixture.request)


def test_variant_boundary_ignores_old_champions_and_unrelated_changes(
    tmp_path: Path,
) -> None:
    fixture = _fixture(tmp_path)
    arena_root = fixture.root / "arena"
    _write_json(arena_root / "promotion-status.json", {"terminal": False})
    # Existing unrelated migrations retain their boundary policy.
    migration.plan_migration(fixture.request)
    _write_json(arena_root / "promotion-status.json", {"terminal": True})
    _write_json(
        arena_root / f"sha256-{'d' * 64}-vs-sha256-{'b' * 64}.json",
        {
            "terminal": False,
            "pairs": [{"variant": "handicap-4-double"}],
        },
    )
    target = yaml.safe_load(fixture.candidate_profile.read_text())
    target["arena"]["segment_handicap_classic_share"] = 0.5
    fixture.candidate_profile.write_text(yaml.safe_dump(target, sort_keys=False))
    migration.plan_migration(fixture.request)


def test_additive_default_field_accepts_legacy_chain_hash(tmp_path: Path) -> None:
    """Every combination of absent defaulted fields is an acceptable chain head.

    A release that predates an additive field recorded the canonical hash
    without it; the migration chain must still accept that head when the
    source profile materializes the field's default.
    """

    fixture = _fixture(tmp_path)
    config = load_config(fixture.old_profile)
    materialized = migration.canonical_config_sha256(config)
    compatible = migration._compatible_source_config_sha256s(config)

    def hash_without(*paths: tuple[str, ...]) -> str:
        payload = deepcopy(config.as_dict())
        for path in paths:
            parent = payload
            for key in path[:-1]:
                parent = parent[key]
            del parent[path[-1]]
        return hashlib.sha256(migration._canonical_config_bytes(payload)).hexdigest()

    finish = ("orchestration", "promotion", "finish_inflight_candidate")
    count = ("orchestration", "plateau", "count_inconclusive_rejections")
    restore = ("orchestration", "plateau", "restore_learning_rate_scale")
    handicap = ("selfplay", "variants", "handicap_classic_share")
    arena_handicap = ("arena", "segment_handicap_classic_share")
    additive = [finish, count, restore, handicap, arena_handicap]
    expected = {materialized}
    for mask in range(1, 2 ** len(additive)):
        expected.add(
            hash_without(
                *(path for bit, path in enumerate(additive) if mask >> bit & 1)
            )
        )
    # Scheduling, actor pause and disabled topology preservation retain every
    # supported epoch. Removing the whole disabled inference service collapses
    # the otherwise independent pre/post-topology representations.
    assert expected <= compatible
    # Preserve the full 120-representation pre-neural-execution epoch, not just
    # the one legacy chain head below. The new disabled-field representations
    # add 64 distinct combinations after all earlier omissions are deduplicated.
    prior_payload = config.as_dict()
    prior_inference = prior_payload["orchestration"]["model_refresh"]["inference"]
    del prior_inference["compact_inference_gather"]
    del prior_inference["small_batch_graph_buckets"]
    prior = migration._compatible_source_config_sha256s(
        SimpleNamespace(as_dict=lambda: deepcopy(prior_payload))
    )
    assert len(prior) == 120 * len(expected)
    assert prior <= compatible
    assert len(compatible) == 184 * len(expected)
    # A profile that opts into a new field no longer matches releases that
    # never had it, but keeps the variants for the other additive fields.
    opted = yaml.safe_load(fixture.old_profile.read_text(encoding="utf-8"))
    opted["orchestration"]["plateau"]["count_inconclusive_rejections"] = True
    opted["orchestration"]["plateau"]["restore_learning_rate_scale"] = 0.5
    opted_path = tmp_path / "opted.yaml"
    opted_path.write_text(yaml.safe_dump(opted, sort_keys=False), encoding="utf-8")
    opted_config = load_config(opted_path)
    assert len(migration._compatible_source_config_sha256s(opted_config)) == 1472

    opted.setdefault("selfplay", {}).setdefault("variants", {})[
        "handicap_classic_share"
    ] = 0.5
    opted.setdefault("arena", {})["segment_handicap_classic_share"] = 0.5
    opted_path.write_text(yaml.safe_dump(opted, sort_keys=False), encoding="utf-8")
    assert (
        len(migration._compatible_source_config_sha256s(load_config(opted_path))) == 368
    )

    # The head a release without scheduling or plateau additions recorded.
    legacy_hash = hash_without(
        count,
        restore,
        handicap,
        arena_handicap,
        ("orchestration", "promotion", "session_seconds"),
        ("orchestration", "historical_evaluation", "session_seconds"),
        ("orchestration", "historical_evaluation", "cooldown_seconds"),
        ("orchestration", "promotion", "pause_strategy"),
        ("selfplay", "search_execution"),
        ("arena", "search_execution"),
        ("orchestration", "model_refresh", "inference", "compact_inference_gather"),
        ("orchestration", "model_refresh", "inference", "small_batch_graph_buckets"),
    )
    assert legacy_hash in compatible
    experimental = yaml.safe_load(fixture.old_profile.read_text(encoding="utf-8"))
    experimental["selfplay"]["search_execution"] = {"first_visit_batch_size": 2}
    experimental_path = tmp_path / "experimental-execution.yaml"
    experimental_path.write_text(yaml.safe_dump(experimental, sort_keys=False))
    assert legacy_hash not in migration._compatible_source_config_sha256s(
        load_config(experimental_path)
    )
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


def test_gate_budget_and_measurement_crossplay_are_migratable(tmp_path: Path) -> None:
    """The arena gate budget and measurement crossplay change arena evidence only."""

    fixture = _fixture(tmp_path)
    target = yaml.safe_load(fixture.candidate_profile.read_text(encoding="utf-8"))
    target["arena"]["simulations"] = 256
    target["arena"]["max_pairs_per_ring"] = 600
    target["orchestration"]["historical_evaluation"] = {
        "enabled": True,
        "every_promotions": 2,
        "anchors_per_evaluation": 1,
        "pairs_per_ring": 50,
        "max_pairs_per_ring": 100,
        "simulations": 1024,
        "max_considered": 32,
        "measure_direct_predecessor": True,
    }
    fixture.candidate_profile.write_text(
        yaml.safe_dump(target, sort_keys=False),
        encoding="utf-8",
    )

    result = migration.migrate_continuous_profile(fixture.request)

    changed = {change["path"] for change in result["changes"]}
    assert {
        "arena.simulations",
        "arena.max_pairs_per_ring",
        "orchestration.historical_evaluation.enabled",
        "orchestration.historical_evaluation.pairs_per_ring",
        "orchestration.historical_evaluation.max_pairs_per_ring",
        "orchestration.historical_evaluation.simulations",
        "orchestration.historical_evaluation.max_considered",
        "orchestration.historical_evaluation.measure_direct_predecessor",
    } <= changed
    assert not any(path.startswith("optimizer") for path in changed)


def test_evaluation_scheduling_migration_preserves_contract_and_pending_evidence(
    tmp_path: Path,
) -> None:
    from startrain.balanced_evaluation import evaluation_contract

    fixture = _fixture(tmp_path, "h100-8gpu-variant-efficiency-stage-b.yaml")
    old_config = load_config(fixture.old_profile)
    target = yaml.safe_load(fixture.old_profile.read_text())
    target["orchestration"]["promotion"].update(
        session_seconds=120.0, inter_wave_cooldown_seconds=600.0
    )
    target["orchestration"]["historical_evaluation"].update(
        session_seconds=150.0, cooldown_seconds=3600.0
    )
    fixture.candidate_profile.write_text(yaml.safe_dump(target, sort_keys=False))
    pending = fixture.root / "arena" / "pending-balanced.json"
    _write_json(
        pending,
        {
            "result_kind": "promotion",
            "terminal": False,
            "evaluation_contract": evaluation_contract(old_config.arena),
            "pairs": [{"pair_seed": 123, "score": 0.5}],
        },
    )
    pending_bytes = pending.read_bytes()

    result = migration.migrate_continuous_profile(fixture.request, apply=True)

    assert {change["path"] for change in result["changes"]} == {
        "orchestration.promotion.session_seconds",
        "orchestration.promotion.inter_wave_cooldown_seconds",
        "orchestration.historical_evaluation.session_seconds",
        "orchestration.historical_evaluation.cooldown_seconds",
    }
    active = load_config(fixture.root / fixture.target_name)
    assert evaluation_contract(active.arena) == evaluation_contract(old_config.arena)
    for section in ("game", "model", "optimizer", "learner", "train", "arena"):
        assert getattr(active, section) == getattr(old_config, section)
    assert active.orchestration.historical_evaluation.search_budget(active.arena) == (
        old_config.orchestration.historical_evaluation.search_budget(old_config.arena)
    )
    assert pending.read_bytes() == pending_bytes
    record = json.loads((fixture.root / "continuous-migrations.jsonl").read_text())
    assert "evaluation_contract_transition" not in record


def test_actor_suspend_migration_preserves_all_training_and_evaluation_state(
    tmp_path: Path,
) -> None:
    fixture = _fixture(tmp_path, "h100-8gpu-variant-efficiency-stage-a.yaml")
    old = load_config(fixture.old_profile)
    assert old.orchestration.promotion.pause_strategy == "terminate"
    target = yaml.safe_load(fixture.old_profile.read_text())
    target["orchestration"]["promotion"]["pause_strategy"] = "suspend"
    fixture.candidate_profile.write_text(yaml.safe_dump(target, sort_keys=False))
    preserved = {
        path: path.read_bytes()
        for path in (
            fixture.root / "run.json",
            fixture.root / "learner/recovery.json",
            fixture.root / "learner/champion.json",
            fixture.checkpoint,
        )
    }
    result = migration.migrate_continuous_profile(fixture.request, apply=True)

    assert result["changes"] == [
        {
            "path": "orchestration.promotion.pause_strategy",
            "from": "terminate",
            "to": "suspend",
        }
    ]
    active = load_config(fixture.root / fixture.target_name)
    for section in ("game", "model", "loss", "optimizer", "train", "learner", "arena"):
        assert getattr(active, section) == getattr(old, section)
    assert active.orchestration.historical_evaluation == (
        old.orchestration.historical_evaluation
    )
    assert all(path.read_bytes() == data for path, data in preserved.items())
    record = json.loads((fixture.root / "continuous-migrations.jsonl").read_text())
    assert "evaluation_contract_transition" not in record
    assert "utd_segment" not in record


def _with_update_to_data(
    fixture: _Fixture,
    *,
    old_target: float,
    new_target: float,
    old_intervals: tuple[int, int],
    new_intervals: tuple[int, int],
    segment: dict[str, object] | None,
) -> None:
    """Give both profiles UTD control and example cadence; seed the live segment."""

    for path, target, intervals in (
        (fixture.old_profile, old_target, old_intervals),
        (fixture.candidate_profile, new_target, new_intervals),
    ):
        raw = yaml.safe_load(path.read_text(encoding="utf-8"))
        raw["learner"]["target_updates_per_new_sample"] = target
        raw["learner"]["candidate_interval_examples"] = intervals[0]
        raw["learner"]["selfplay_snapshot_interval_examples"] = intervals[1]
        path.chmod(0o644)
        path.write_text(yaml.safe_dump(raw, sort_keys=False), encoding="utf-8")
        path.chmod(0o444)
    digest = hashlib.sha256(fixture.old_profile.read_bytes()).hexdigest()
    (fixture.root / "profile.sha256").write_text(
        f"{digest}  {fixture.old_profile}\n", encoding="utf-8"
    )
    segment_path = fixture.root / "learner" / "utd-segment.json"
    if segment is None:
        segment_path.unlink(missing_ok=True)
    else:
        _write_json(segment_path, segment)


def test_update_to_data_retarget_writes_prospective_segment(tmp_path: Path) -> None:
    """A UTD change starts a fresh segment at the durable boundary.

    The new ratio must apply only to samples generated from the boundary on;
    re-rating the historical ledger would grant a burst of updates over stale
    data. The previous segment is preserved in the rollback bundle.
    """

    fixture = _fixture(tmp_path)
    previous = {
        "schema_version": 1,
        "run_id": "continuous-test-run",
        "generation_family": "family-continuous-test",
        "target_updates_per_new_sample": 1.0,
        "baseline_examples_consumed": 1_024,
        "baseline_committed_replay_samples": 2_048,
        "created_ns": 5,
    }
    _with_update_to_data(
        fixture,
        old_target=1.0,
        new_target=1.5,
        old_intervals=(2_000_000, 1_000_000),
        new_intervals=(3_000_000, 1_500_000),
        segment=previous,
    )

    dry_run = migration.migrate_continuous_profile(fixture.request)
    assert {change["path"] for change in dry_run["changes"]} >= {
        "learner.target_updates_per_new_sample",
        "learner.candidate_interval_examples",
        "learner.selfplay_snapshot_interval_examples",
    }
    expected_segment = {
        "schema_version": 1,
        "run_id": "continuous-test-run",
        "generation_family": "family-continuous-test",
        "target_updates_per_new_sample": 1.5,
        "baseline_examples_consumed": 51_200,
        "baseline_committed_replay_samples": 60_000,
    }
    planned = dict(dry_run["utd_segment"])
    assert planned.pop("created_ns") > 0
    assert planned == expected_segment
    assert str(fixture.root / "learner" / "utd-segment.json") in dry_run["writes"]
    assert (
        json.loads(
            (fixture.root / "learner" / "utd-segment.json").read_text(encoding="utf-8")
        )
        == previous
    )

    result = migration.migrate_continuous_profile(fixture.request, apply=True)

    written = json.loads(
        (fixture.root / "learner" / "utd-segment.json").read_text(encoding="utf-8")
    )
    assert written == result["utd_segment"]
    assert written["created_ns"] == result["utd_segment"]["created_ns"]
    assert {key: written[key] for key in expected_segment} == expected_segment
    record = json.loads(
        (fixture.root / "continuous-migrations.jsonl").read_text(encoding="utf-8")
    )
    assert record["utd_segment"] == written
    backup = Path(result["backup_bundle"])
    assert (
        json.loads(
            (backup / "learner" / "utd-segment.json").read_text(encoding="utf-8")
        )
        == previous
    )


def test_unchanged_update_to_data_target_leaves_segment_alone(tmp_path: Path) -> None:
    fixture = _fixture(tmp_path)
    previous = {
        "schema_version": 1,
        "run_id": "continuous-test-run",
        "generation_family": "family-continuous-test",
        "target_updates_per_new_sample": 1.0,
        "baseline_examples_consumed": 1_024,
        "baseline_committed_replay_samples": 2_048,
    }
    _with_update_to_data(
        fixture,
        old_target=1.0,
        new_target=1.0,
        old_intervals=(2_000_000, 1_000_000),
        new_intervals=(2_000_000, 1_000_000),
        segment=previous,
    )

    result = migration.migrate_continuous_profile(fixture.request, apply=True)

    assert result["utd_segment"] is None
    assert str(fixture.root / "learner" / "utd-segment.json") not in result["writes"]
    assert (
        json.loads(
            (fixture.root / "learner" / "utd-segment.json").read_text(encoding="utf-8")
        )
        == previous
    )
    record = json.loads(
        (fixture.root / "continuous-migrations.jsonl").read_text(encoding="utf-8")
    )
    assert "utd_segment" not in record


@pytest.mark.parametrize(
    ("new_target", "new_intervals", "segment_target", "message"),
    [
        (1.5, (2_000_000, 1_000_000), 1.0, "must scale with the update-to-data"),
        (1.5, (3_000_000, 1_000_000), 1.0, "must scale with the update-to-data"),
        (1.5, (3_000_000, 1_500_000), 1.25, "does not match the source profile"),
    ],
)
def test_update_to_data_retarget_rejects_unsafe_inputs(
    tmp_path: Path,
    new_target: float,
    new_intervals: tuple[int, int],
    segment_target: float,
    message: str,
) -> None:
    fixture = _fixture(tmp_path)
    _with_update_to_data(
        fixture,
        old_target=1.0,
        new_target=new_target,
        old_intervals=(2_000_000, 1_000_000),
        new_intervals=new_intervals,
        segment={
            "schema_version": 1,
            "run_id": "continuous-test-run",
            "generation_family": "family-continuous-test",
            "target_updates_per_new_sample": segment_target,
            "baseline_examples_consumed": 1_024,
            "baseline_committed_replay_samples": 2_048,
        },
    )
    before = _snapshot(fixture.root)

    with pytest.raises(migration.MigrationError, match=message):
        migration.migrate_continuous_profile(fixture.request, apply=True)
    assert _snapshot(fixture.root) == before


def test_removing_update_to_data_control_is_rejected(tmp_path: Path) -> None:
    fixture = _fixture(tmp_path)
    _with_update_to_data(
        fixture,
        old_target=1.0,
        new_target=1.0,
        old_intervals=(2_000_000, 1_000_000),
        new_intervals=(2_000_000, 1_000_000),
        segment=None,
    )
    raw = yaml.safe_load(fixture.candidate_profile.read_text(encoding="utf-8"))
    raw["learner"]["target_updates_per_new_sample"] = None
    fixture.candidate_profile.chmod(0o644)
    fixture.candidate_profile.write_text(
        yaml.safe_dump(raw, sort_keys=False), encoding="utf-8"
    )

    with pytest.raises(migration.MigrationError, match="removing update-to-data"):
        migration.migrate_continuous_profile(fixture.request)


def test_plateau_inconclusive_counting_is_migratable(tmp_path: Path) -> None:
    fixture = _fixture(tmp_path)
    target = yaml.safe_load(fixture.candidate_profile.read_text(encoding="utf-8"))
    target["orchestration"]["plateau"]["count_inconclusive_rejections"] = True
    fixture.candidate_profile.write_text(
        yaml.safe_dump(target, sort_keys=False),
        encoding="utf-8",
    )

    result = migration.migrate_continuous_profile(fixture.request)

    assert "orchestration.plateau.count_inconclusive_rejections" in {
        change["path"] for change in result["changes"]
    }


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


def test_largest_board_cutover_preserves_pending_evidence_and_training_state(tmp_path):
    from startrain.balanced_evaluation import evaluation_contract

    fixture = _fixture(tmp_path, "h100-8gpu-variant-efficiency-stage-b.yaml")
    old = load_config(fixture.old_profile)
    target = yaml.safe_load(fixture.old_profile.read_text())
    target["orchestration"]["training_objective"] = "ring10_priority"
    target["orchestration"]["ring_mixture"]["step_weights"] = [
        {"from_step": 0, "weights": [0.05, 0.05, 0.05, 0.85]}
    ]
    target["selfplay"]["rings"] = 10
    target["arena"].update(rings=[10], required_regression_rings=[])
    fixture.candidate_profile.write_text(yaml.safe_dump(target))
    evidence = {
        "promotion-status.json": {"terminal": False, "candidate_step": 90},
        "pending-balanced.json": {
            "terminal": False,
            "result_kind": "promotion",
            "evaluation_contract": evaluation_contract(old.arena),
            "pairs": [{"ring": 4, "pair": 0}],
        },
        "pending-balanced.resume.json": {
            "arena_state": {"game_states": [{"actions": [1, 2, 3]}]}
        },
        "historical-old-champion.json": {
            "result_kind": "historical_crossplay",
            "terminal": False,
        },
        "orphan.resume.json": {"arena_state": {"game_states": []}},
    }
    for name, value in evidence.items():
        _write_json(fixture.root / "arena" / name, value)
    pinned = [fixture.root / "arena" / name for name in evidence]
    pinned.extend(
        fixture.root / name
        for name in ("run.json", "learner/recovery.json", "learner/champion.json")
    )
    before = {path: path.read_bytes() for path in pinned}
    checkpoint_before = fixture.checkpoint.read_bytes()

    plan = migration.plan_migration(fixture.request)
    migration.apply_migration(plan)

    assert {path: path.read_bytes() for path in pinned} == before
    assert fixture.checkpoint.read_bytes() == checkpoint_before
    active = load_config(plan.target_profile)
    assert active.model == old.model
    assert active.optimizer == old.optimizer
    assert active.learner == old.learner
    assert active.arena.rings == (10,)
    assert active.orchestration.ring_mixture.weights_for_step(100) == (
        0.05,
        0.05,
        0.05,
        0.85,
    )
    record = json.loads((fixture.root / "continuous-migrations.jsonl").read_text())
    transition = record["evaluation_contract_transition"]
    assert transition["kind"] == "balanced_scope_change"
    assert transition["from"]["identity"] != transition["to"]["identity"]
    assert len(transition["from"]["cells"]) == 24
    assert len(transition["to"]["cells"]) == 6
    assert set(transition["retained_evidence"]) == {
        f"arena/{name}" for name in evidence
    }
    assert "utd_segment" not in record


def test_same_scope_guard_change_still_requires_terminal_evidence(tmp_path):
    fixture = _fixture(tmp_path, "h100-8gpu-variant-efficiency-stage-b.yaml")
    raw = yaml.safe_load(fixture.old_profile.read_text())
    raw["arena"]["cell_regression_floor_elo"] = -80.0
    fixture.candidate_profile.write_text(yaml.safe_dump(raw))
    _write_json(fixture.root / "arena/promotion-status.json", {"terminal": False})
    before = _snapshot(fixture.root)
    with pytest.raises(migration.MigrationError, match="terminal arena boundary"):
        migration.plan_migration(fixture.request)
    assert _snapshot(fixture.root) == before


@pytest.mark.parametrize(
    "field,values",
    [
        ("preserve_broadcast_topology", (False, True)),
        ("compact_inference_gather", (True, False)),
        ("small_batch_graph_buckets", (True, False)),
    ],
)
def test_inference_execution_cutover_is_reversible_and_preserves_pending_work(
    tmp_path, field, values
):
    from dataclasses import replace

    from startrain.balanced_evaluation import evaluation_contract

    fixture = _fixture(tmp_path, "h100-8gpu-largest-board-priority.yaml")
    original = load_config(fixture.old_profile)
    evidence = {
        "arena/promotion-status.json": {"terminal": False, "candidate_step": 90},
        "arena/pending-balanced.json": {
            "terminal": False,
            "result_kind": "promotion",
            "evaluation_contract": evaluation_contract(original.arena),
            "pairs": [{"ring": 10, "pair": 0}],
        },
        "arena/pending-balanced.resume.json": {
            "arena_state": {"game_states": [{"actions": [1, 2, 3]}]}
        },
    }
    for name, payload in evidence.items():
        _write_json(fixture.root / name, payload)
    retained = [fixture.root / name for name in evidence]
    retained.extend(
        fixture.root / name
        for name in ("run.json", "learner/recovery.json", "learner/champion.json")
    )
    retained.append(fixture.checkpoint)
    before = {path: path.read_bytes() for path in retained}
    source_profile = fixture.old_profile
    source_commit = fixture.request.from_source_commit
    for index, enabled in enumerate(values):
        raw = yaml.safe_load(source_profile.read_text())
        raw["orchestration"]["model_refresh"]["inference"][field] = enabled
        fixture.candidate_profile.write_text(yaml.safe_dump(raw))
        request = replace(
            fixture.request,
            old_profile=source_profile,
            target_profile_name=f"profile-{field}-{index}.yaml",
            from_source_commit=source_commit,
            to_source_commit=str(index + 2) * 40,
        )
        plan = migration.plan_migration(request)
        migration.apply_migration(plan)
        changed = load_config(plan.target_profile)
        assert getattr(changed.orchestration.model_refresh.inference, field) is enabled
        assert changed.model == original.model
        assert changed.game == original.game
        assert changed.optimizer == original.optimizer
        assert changed.train == original.train
        assert changed.learner == original.learner
        assert changed.selfplay == original.selfplay
        assert changed.arena == original.arena
        assert evaluation_contract(changed.arena) == evaluation_contract(original.arena)
        assert {path: path.read_bytes() for path in retained} == before
        record = json.loads(
            (fixture.root / "continuous-migrations.jsonl").read_text().splitlines()[-1]
        )
        assert [change["path"] for change in record["changes"]] == [
            f"orchestration.model_refresh.inference.{field}"
        ]
        assert "utd_segment" not in record
        assert "evaluation_contract_transition" not in record
        source_profile = plan.target_profile
        source_commit = request.to_source_commit
