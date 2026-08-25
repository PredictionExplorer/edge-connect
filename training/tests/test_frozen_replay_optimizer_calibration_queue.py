from __future__ import annotations

import json
from pathlib import Path

import pytest

import scripts.run_frozen_replay_optimizer_calibration_queue as queue
from scripts.prepare_elo_ablation import (
    RING10_OPTIMIZER_CALIBRATION_TREATMENTS,
)
from scripts.run_frozen_replay_optimizer_calibration import CONTROL_ARM


def _write_json(path: Path, document: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(document), encoding="utf-8")


def _plan(tmp_path: Path) -> Path:
    treatments = []
    for arm in RING10_OPTIMIZER_CALIBRATION_TREATMENTS:
        profile = tmp_path / "profiles" / f"{arm}.yaml"
        profile.parent.mkdir(parents=True, exist_ok=True)
        profile.write_text(f"arm: {arm}\n", encoding="utf-8")
        treatments.append(
            {
                "treatment": arm,
                "profile": str(profile),
                "profile_sha256": queue._sha256(profile),
                "run_root": str(tmp_path / "runs" / arm),
            }
        )
    plan = tmp_path / "profiles" / "ablation-plan.json"
    _write_json(
        plan,
        {
            "schema_version": 1,
            "report": "startrain-elo-ablation-plan",
            "suite": "ring10-optimizer-calibration",
            "wall_budget_seconds": 7200,
            "leaf_budget": 1,
            "treatments": treatments,
        },
    )
    return plan


def _run(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    *,
    selected: str,
    fallback: bool,
) -> tuple[dict[str, object], list[str]]:
    calls: list[str] = []

    def fake_run(settings):
        calls.append(settings.arm)
        result_path = settings.output_dir / "result.json"
        _write_json(result_path, {"arm": settings.arm, "status": "complete"})
        return {"arm": settings.arm, "status": "complete"}

    def fake_compare(_paths):
        return {
            "format": "fixture-comparison",
            "schema_version": 1,
            "status": "ok",
            "selection": {
                "selected_arm": selected,
                "selected_label": selected,
                "fallback_to_control": fallback,
            },
        }

    monkeypatch.setattr(queue, "run_calibration", fake_run)
    monkeypatch.setattr(queue, "compare_results", fake_compare)
    plan = _plan(tmp_path)
    state = queue.run_calibration_queue(
        plan_path=plan,
        champion=tmp_path / "champion.json",
        replay_root=tmp_path / "replay",
        replay_cutoff=42,
        output_root=tmp_path / "output",
        steps=10,
        device="cpu",
        budget_h100_hours=2,
        screen_plan_path=tmp_path / "screen" / "ablation-plan.json",
        screen_wall_budget_hours=8,
        screen_leaf_budget=2_000_000_000,
    )
    return state, calls


def test_queue_builds_control_vs_unique_winner_screen_plan(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    selected = "ring10-optimizer-clip-norm-2"
    state, calls = _run(
        tmp_path,
        monkeypatch,
        selected=selected,
        fallback=False,
    )

    assert state["status"] == "completed"
    assert calls == list(RING10_OPTIMIZER_CALIBRATION_TREATMENTS)
    screen_path = Path(state["screen_plan"]["path"])
    screen = json.loads(screen_path.read_text(encoding="utf-8"))
    assert [item["treatment"] for item in screen["treatments"]] == [
        CONTROL_ARM,
        selected,
    ]
    assert screen["wall_budget_seconds"] == 8 * 3600
    assert screen["leaf_budget"] == 2_000_000_000
    assert screen["frozen_calibration"]["selected_arm"] == selected

    resumed = queue.run_calibration_queue(
        plan_path=_plan(tmp_path),
        champion=tmp_path / "champion.json",
        replay_root=tmp_path / "replay",
        replay_cutoff=42,
        output_root=tmp_path / "output",
        steps=10,
        device="cpu",
        budget_h100_hours=2,
        screen_plan_path=screen_path,
        screen_wall_budget_hours=8,
        screen_leaf_budget=2_000_000_000,
    )
    assert resumed["status"] == "completed"
    assert calls == list(RING10_OPTIMIZER_CALIBRATION_TREATMENTS)


def test_queue_falls_back_without_screen_when_no_treatment_passes(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    state, _calls = _run(
        tmp_path,
        monkeypatch,
        selected=CONTROL_ARM,
        fallback=True,
    )

    assert state["status"] == "completed"
    assert state["screen_plan"] is None
    assert not (tmp_path / "screen" / "ablation-plan.json").exists()
