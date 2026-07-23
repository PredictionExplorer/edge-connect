from __future__ import annotations

import json
import os
from pathlib import Path

import pytest

from scripts.fork_elo_ablation import fork_elo_ablation, main
from scripts.prepare_elo_ablation import prepare_elo_ablation

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
