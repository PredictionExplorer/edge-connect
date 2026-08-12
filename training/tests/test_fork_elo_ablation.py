from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path

import pytest

from scripts.fork_elo_ablation import fork_elo_ablation, main
from scripts.prepare_elo_ablation import prepare_elo_ablation
from startrain.replay_store import ReplayStore
from startrain.runtime import load_run_identity

CONFIGS = Path(__file__).parents[1] / "configs"


def _write_json(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")


def _source_run(tmp_path: Path) -> Path:
    source = tmp_path / "source"
    _write_json(
        source / "run.json",
        {
            "schema_version": 1,
            "run_id": "shared-run",
            "generation_family": "shared-family",
            "created_ns": 1,
        },
    )
    _write_json(
        source / "learner" / "champion.json",
        {
            "model_identity": "champion-id",
            "model_step": 364_000,
            "updated_ns": 10,
        },
    )
    _write_json(
        source / "learner" / "candidate.json",
        {
            "model_identity": "candidate-id",
            "model_step": 392_000,
            "updated_ns": 11,
        },
    )
    _write_json(
        source / "learner" / "recovery.json",
        {
            "schema_version": 1,
            "format": "startrain.recovery-pointer",
            "run_id": "shared-run",
            "generation_family": "shared-family",
            "examples_consumed": 100,
        },
    )
    files = {
        "replay/shards/shard.npz": b"immutable replay",
        "learner/checkpoints/model.pt": b"immutable checkpoint",
        "learner/manifests/manifest.json": b"immutable manifest",
        "learner/recovery/recovery.pt": b"immutable recovery",
        "replay/manifest.sqlite3": b"mutable database",
        "status/coordinator.json": b"stale status",
        "logs/learner.log": b"old log",
        "metrics/actor.jsonl": b"old metrics\n",
        "learner/metrics.jsonl": b"old learner metrics\n",
        "strength-efficiency.json": b"old report",
    }
    for relative, content in files.items():
        path = source / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(content)
    return source


def _plan(tmp_path: Path, source: Path) -> Path:
    output = tmp_path / "profiles"
    prepare_elo_ablation(
        base_config=CONFIGS / "h100-8gpu-throughput.yaml",
        output_dir=output,
        run_root_parent=tmp_path / "runs",
        run_id="shared-run",
        source_run_root=source,
        prefix="pilot",
        seed=17,
        wall_budget_hours=8,
        leaf_budget=2_000_000_000,
        guard_floor_elo=-35,
        treatments=("control",),
    )
    return output / "ablation-plan.json"


def _winner_snapshot(source: Path) -> dict[str, object]:
    run_path = source / "run.json"
    champion_path = source / "learner" / "champion.json"
    run = json.loads(run_path.read_text(encoding="utf-8"))
    champion = json.loads(champion_path.read_text(encoding="utf-8"))
    return {
        "schema_version": 1,
        "status": "verified",
        "label": "selected",
        "run_root": str(source.resolve()),
        "run_identity": {
            key: run[key] for key in ("run_id", "generation_family", "created_ns")
        },
        "run_identity_artifact": {
            "path": str(run_path.resolve()),
            "sha256": hashlib.sha256(run_path.read_bytes()).hexdigest(),
        },
        "champion": champion,
        "champion_pointer_artifact": {
            "path": str(champion_path.resolve()),
            "sha256": hashlib.sha256(champion_path.read_bytes()).hexdigest(),
        },
        "source_anchor": {"model_identity": "older", "model_step": 0},
        "selection": "guarded_chronological_champion_frontier",
    }


def test_fork_links_immutable_artifacts_and_rotates_runtime(tmp_path: Path) -> None:
    source = _source_run(tmp_path)
    plan = _plan(tmp_path, source)

    metadata = fork_elo_ablation(
        source_run_root=source,
        plan_path=plan,
        treatment="control",
    )

    destination = tmp_path / "runs" / "pilot-control-seed17"
    assert metadata["treatment"] == "control"
    assert metadata["anchor"]["model_identity"] == "champion-id"
    assert metadata["measurement_started_ns"] is None
    assert (destination / "profile-elo-ablation.yaml").is_file()
    expected_checksum = f"{metadata['profile_sha256']}  profile-elo-ablation.yaml\n"
    assert (destination / "profile-elo-ablation.sha256").read_text() == (
        expected_checksum
    )
    assert (destination / "profile.sha256").read_text() == expected_checksum
    assert (destination / "ablation.json").is_file()
    assert not (destination / "coordinator.lock").exists()
    assert not list((destination / "status").iterdir())
    assert not list((destination / "logs").iterdir())
    assert not list((destination / "metrics").iterdir())
    assert (
        destination / "ablation-parent" / "learner-metrics.jsonl"
    ).read_bytes() == b"old learner metrics\n"
    assert (
        destination / "ablation-parent" / "strength-efficiency.json"
    ).read_bytes() == b"old report"

    linked = destination / "replay" / "shards" / "shard.npz"
    mutable = destination / "replay" / "manifest.sqlite3"
    assert (
        os.stat(linked).st_ino
        == os.stat(source / linked.relative_to(destination)).st_ino
    )
    assert (
        os.stat(mutable).st_ino
        != os.stat(source / mutable.relative_to(destination)).st_ino
    )


def test_fork_preserves_nested_ablation_ancestry(tmp_path: Path) -> None:
    source = _source_run(tmp_path)
    inherited = source / "ablation-parent" / "ancestor-marker.json"
    inherited.parent.mkdir()
    inherited.write_text('{"generation": 1}', encoding="utf-8")
    plan = _plan(tmp_path, source)

    fork_elo_ablation(
        source_run_root=source,
        plan_path=plan,
        treatment="control",
    )

    destination = tmp_path / "runs" / "pilot-control-seed17"
    preserved = destination / "ablation-parent" / "ancestor" / "ancestor-marker.json"
    assert preserved.read_text(encoding="utf-8") == '{"generation": 1}'
    assert (
        destination / "ablation-parent" / "learner-metrics.jsonl"
    ).read_bytes() == b"old learner metrics\n"


def test_fork_archives_incompatible_inherited_warm_start(tmp_path: Path) -> None:
    source = _source_run(tmp_path)
    _write_json(
        source / "learner" / "champion-warm-start.json",
        {
            "schema_version": 1,
            "status": "active",
            "source_model_identity": "older-champion",
            "absolute_model_step": 300_000,
        },
    )
    plan = _plan(tmp_path, source)

    metadata = fork_elo_ablation(
        source_run_root=source,
        plan_path=plan,
        treatment="control",
    )

    destination = tmp_path / "runs" / "pilot-control-seed17"
    assert not (destination / "learner" / "champion-warm-start.json").exists()
    archived = destination / "ablation-parent" / "inherited-champion-warm-start.json"
    assert json.loads(archived.read_text())["source_model_identity"] == "older-champion"
    rotation = metadata["inherited_champion_warm_start_rotation"]
    assert rotation["active_champion_model_identity"] == "champion-id"
    assert rotation["active_champion_model_step"] == 364_000


def test_fork_refuses_active_source_and_changed_profile(tmp_path: Path) -> None:
    source = _source_run(tmp_path)
    plan_path = _plan(tmp_path, source)
    (source / "coordinator.lock").write_text("locked", encoding="utf-8")
    with pytest.raises(RuntimeError, match="coordinator lock"):
        fork_elo_ablation(
            source_run_root=source,
            plan_path=plan_path,
            treatment="control",
        )
    (source / "coordinator.lock").unlink()

    plan = json.loads(plan_path.read_text())
    profile = Path(plan["treatments"][0]["profile"])
    profile.write_text(profile.read_text() + "\n", encoding="utf-8")
    with pytest.raises(ValueError, match="digest changed"):
        fork_elo_ablation(
            source_run_root=source,
            plan_path=plan_path,
            treatment="control",
        )


def test_fork_prepares_utd_segment_for_existing_run(tmp_path: Path) -> None:
    source = _source_run(tmp_path)
    (source / "replay" / "manifest.sqlite3").unlink()
    identity = load_run_identity(source / "run.json")
    with ReplayStore(source / "replay") as store:
        store.register_run(identity)
    output = tmp_path / "utd-profiles"
    prepare_elo_ablation(
        base_config=CONFIGS / "h100-8gpu-throughput.yaml",
        output_dir=output,
        run_root_parent=tmp_path / "runs",
        run_id="shared-run",
        source_run_root=source,
        prefix="utd",
        seed=17,
        wall_budget_hours=8,
        leaf_budget=2_000_000_000,
        guard_floor_elo=-35,
        treatments=("utd-1",),
    )

    metadata = fork_elo_ablation(
        source_run_root=source,
        plan_path=output / "ablation-plan.json",
        treatment="utd-1",
    )

    segment = metadata["utd_segment"]
    assert segment == {
        "schema_version": 1,
        "run_id": "shared-run",
        "generation_family": "shared-family",
        "target_updates_per_new_sample": 1.0,
        "baseline_examples_consumed": 100,
        "baseline_committed_replay_samples": 0,
        "created_ns": segment["created_ns"],
    }
    persisted = json.loads(
        (
            tmp_path / "runs" / "utd-utd-1-seed17" / "learner" / "utd-segment.json"
        ).read_text()
    )
    assert persisted == segment


def test_fork_preserves_verified_staged_winner_and_rejects_stale_source(
    tmp_path: Path,
) -> None:
    source = _source_run(tmp_path)
    snapshot = _winner_snapshot(source)
    output = tmp_path / "staged-profiles"
    prepare_elo_ablation(
        base_config=CONFIGS / "h100-8gpu-throughput.yaml",
        output_dir=output,
        run_root_parent=tmp_path / "staged-runs",
        run_id="shared-run",
        source_run_root=source,
        prefix="staged",
        seed=41,
        wall_budget_hours=1,
        leaf_budget=10,
        guard_floor_elo=-35,
        treatments=("control",),
        winner_snapshot=snapshot,
    )
    plan = output / "ablation-plan.json"

    metadata = fork_elo_ablation(
        source_run_root=source,
        plan_path=plan,
        treatment="control",
    )

    assert metadata["source_winner_snapshot"] == snapshot
    assert metadata["anchor"]["model_identity"] == "champion-id"

    stale_source = _source_run(tmp_path / "stale")
    stale_snapshot = _winner_snapshot(stale_source)
    stale_output = tmp_path / "stale-profiles"
    prepare_elo_ablation(
        base_config=CONFIGS / "h100-8gpu-throughput.yaml",
        output_dir=stale_output,
        run_root_parent=tmp_path / "stale-runs",
        run_id="shared-run",
        source_run_root=stale_source,
        prefix="stale",
        seed=43,
        wall_budget_hours=1,
        leaf_budget=10,
        guard_floor_elo=-35,
        treatments=("control",),
        winner_snapshot=stale_snapshot,
    )
    champion_path = stale_source / "learner" / "champion.json"
    champion = json.loads(champion_path.read_text(encoding="utf-8"))
    champion["model_step"] = 400_001
    champion_path.write_text(json.dumps(champion), encoding="utf-8")
    with pytest.raises(ValueError, match="artifact is stale"):
        fork_elo_ablation(
            source_run_root=stale_source,
            plan_path=stale_output / "ablation-plan.json",
            treatment="control",
        )


def test_fork_cli_reports_unknown_treatment_as_json(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    source = _source_run(tmp_path)
    plan = _plan(tmp_path, source)

    exit_code = main(
        [
            "--source-run-root",
            str(source),
            "--plan",
            str(plan),
            "--treatment",
            "missing",
        ]
    )

    assert exit_code == 2
    payload = json.loads(capsys.readouterr().out)
    assert payload["status"] == "error"
    assert "does not contain one treatment" in payload["error"]
