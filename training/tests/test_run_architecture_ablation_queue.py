from __future__ import annotations

import hashlib
import json
from pathlib import Path

from scripts import run_architecture_ablation_queue as architecture_queue
from scripts.prepare_elo_ablation import (
    RING10_ATTENTION_TREATMENTS,
    prepare_elo_ablation,
)
from scripts.prepare_scratch_architecture import prepare_scratch_root

CONFIGS = Path(__file__).parents[1] / "configs"


def _write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, sort_keys=True), encoding="utf-8")


def _prepared_plan(tmp_path: Path) -> Path:
    source = tmp_path / "source"
    _write_json(
        source / "learner" / "champion.json",
        {"model_identity": "baseline", "model_step": 0},
    )
    profiles = tmp_path / "profiles"
    prepare_elo_ablation(
        base_config=CONFIGS / "h100-8gpu-ring10-only.yaml",
        output_dir=profiles,
        run_root_parent=tmp_path / "runs",
        run_id="source-run",
        source_run_root=source,
        prefix="attention",
        seed=17,
        wall_budget_hours=0.001,
        leaf_budget=2_000_000_000,
        guard_floor_elo=-35,
        treatments=RING10_ATTENTION_TREATMENTS,
        guard_rings=(),
        suite="ring10-attention-reallocation",
    )
    plan_path = profiles / "ablation-plan.json"
    for treatment in RING10_ATTENTION_TREATMENTS:
        prepare_scratch_root(plan_path, treatment)
    return plan_path


def test_dedicated_architecture_queue_consumes_prepared_scratch_roots(
    tmp_path: Path,
    monkeypatch,
) -> None:
    plan_path = _prepared_plan(tmp_path)
    source_manifest = tmp_path / "source" / "baseline-manifest.json"
    source_manifest.write_text("baseline\n", encoding="utf-8")
    arena_calls: list[Path] = []

    def immutable_manifest(pointer: Path) -> Path:
        if pointer.name == "champion.json" and pointer.parent.parent.name == "source":
            return source_manifest
        return pointer

    def arm_runner(**arguments) -> dict[str, object]:
        treatment = arguments["treatment"]
        root = Path(str(treatment["run_root"]))
        manifest = root / "champion-manifest.json"
        manifest.write_text(str(treatment["treatment"]) + "\n", encoding="utf-8")
        return {
            "status": "completed",
            "champion_manifest": str(manifest),
            "champion_manifest_sha256": hashlib.sha256(
                manifest.read_bytes()
            ).hexdigest(),
            "champion_manifest_bytes": manifest.stat().st_size,
        }

    def arena_runner(**arguments) -> None:
        output = Path(str(arguments["output"]))
        output.write_text("{}\n", encoding="utf-8")
        arena_calls.append(output)

    monkeypatch.setattr(
        architecture_queue,
        "_immutable_manifest",
        immutable_manifest,
    )
    monkeypatch.setattr(
        architecture_queue,
        "_resolve_executable",
        lambda _value: "/arena",
    )
    monkeypatch.setattr(
        architecture_queue,
        "build_architecture_ablation_evidence",
        lambda _suite: {
            "format": "startrain-architecture-ablation-evidence",
            "diagnostic_only": True,
            "production_promotion_authorized": False,
            "adoption_authorized": False,
        },
    )

    result = architecture_queue.run_architecture_queue(
        plan_path=plan_path,
        state_path=tmp_path / "queue-state.json",
        evidence_directory=tmp_path / "evidence",
        orchestrator="/orchestrator",
        arena_executable="/arena",
        device="cuda",
        poll_seconds=1.0,
        execution_lock_path=tmp_path / "host-execution.lock",
        arm_runner=arm_runner,
        arena_runner=arena_runner,
    )

    assert result["status"] == "completed"
    assert len(arena_calls) == 3
    comparison = result["comparisons"]["ring10-attention-full-kv"]
    assert comparison["production_promotion_authorized"] is False
    assert comparison["adoption_authorized"] is False
    assert result["operator_review_required"] is True


def test_scratch_arm_rejects_prelaunch_contamination(tmp_path: Path) -> None:
    plan_path = _prepared_plan(tmp_path)
    plan = json.loads(plan_path.read_text(encoding="utf-8"))
    treatment = plan["treatments"][0]
    root = Path(str(treatment["run_root"]))
    (root / "replay").mkdir()

    try:
        architecture_queue.run_scratch_arm(
            plan_path=plan_path,
            plan_sha256=hashlib.sha256(plan_path.read_bytes()).hexdigest(),
            treatment=treatment,
            wall_budget_seconds=1.0,
            orchestrator="/unreachable",
            poll_seconds=1.0,
        )
    except architecture_queue.ArchitectureQueueError as error:
        assert "contaminated" in str(error)
    else:
        raise AssertionError("contaminated scratch root was accepted")
