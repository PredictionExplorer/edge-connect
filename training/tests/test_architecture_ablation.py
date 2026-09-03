from __future__ import annotations

import json
from dataclasses import asdict, replace
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from scripts.compare_architecture_ablation import (
    EVIDENCE_FORMAT,
    build_architecture_ablation_evidence,
    main,
)
from scripts.run_architecture_ablation_queue import architecture_suite_document
from startrain.arena import ARENA_RESULT_SCHEMA_VERSION, ArenaGame, ArenaPair
from startrain.checkpoint import (
    ExponentialMovingAverage,
    extract_verified_manifest_config,
    sha256_file,
)
from startrain.config import load_config
from startrain.learner import ImmutableModelPublisher
from startrain.model import GraphResTNet
from startrain.runtime import RunIdentity


def _pin(path: Path) -> dict[str, object]:
    return {
        "path": str(path.resolve()),
        "sha256": sha256_file(path),
        "bytes": path.stat().st_size,
    }


def _publish_model(
    root: Path,
    *,
    role: str,
    model_config,
    experiment,
    seed: int,
):
    torch.manual_seed(seed)
    model = GraphResTNet(model_config)
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.001)
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer,
        lambda _step: 1.0,
    )
    ema = ExponentialMovingAverage(model)
    identity = RunIdentity(
        root / "run.json",
        f"run-{role}",
        f"family-{role}",
        1,
    )
    config = experiment.as_dict()
    config["model"] = asdict(model_config)
    manifest = ImmutableModelPublisher(root / "learner", identity).publish(
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
        ema=ema,
        step=1,
        epoch=0,
        config=config,
    )
    return manifest


def _games_for_pair(pair: ArenaPair) -> list[ArenaGame]:
    return [
        ArenaGame(
            ring=pair.ring,
            pair=pair.pair,
            candidate_player=candidate_player,
            opening_seed=pair.opening_seed,
            opening_action=pair.opening_action,
            forced_opening=pair.forced_opening,
            winner=(
                candidate_player
                if pair.outcomes[candidate_player] == 1
                else 1 - candidate_player
            ),
            outcome=pair.outcomes[candidate_player],
            searched_moves=3,
        )
        for candidate_player in (0, 1)
    ]


def _write_arena(
    path: Path,
    *,
    candidate,
    baseline,
    candidate_config,
    baseline_config,
    outcomes: tuple[tuple[int, int], ...],
) -> None:
    pairs = [
        ArenaPair(
            ring=4,
            pair=index,
            opening_seed=100 + index,
            opening_action=index,
            forced_opening=True,
            outcomes=pair_outcomes,
        )
        for index, pair_outcomes in enumerate(outcomes)
    ]
    games = [game for pair in pairs for game in _games_for_pair(pair)]
    path.write_text(
        json.dumps(
            {
                "schema_version": ARENA_RESULT_SCHEMA_VERSION,
                "result_kind": "architecture_evaluation",
                "evaluation_mode": "architecture",
                "candidate": candidate.model_identity,
                "baseline": baseline.model_identity,
                "candidate_manifest": str(
                    (candidate.artifact_manifest or candidate.path).resolve()
                ),
                "candidate_manifest_sha256": candidate.manifest_sha256,
                "candidate_manifest_bytes": candidate.manifest_bytes,
                "baseline_manifest": str(
                    (baseline.artifact_manifest or baseline.path).resolve()
                ),
                "baseline_manifest_sha256": baseline.manifest_sha256,
                "baseline_manifest_bytes": baseline.manifest_bytes,
                "model_configs": {
                    "candidate": candidate_config.model_config,
                    "baseline": baseline_config.model_config,
                },
                "evaluation_contract": candidate_config.evaluation_contract,
                "evaluator_contract": {
                    "device_type": "cpu",
                    "precision": "fp32",
                    "score_utility_weight": 0.0,
                    "compile_enabled": False,
                    "compile_dynamic": False,
                    "compile_mode": "default",
                    "fullgraph": True,
                },
                "diagnostic_only": True,
                "promotion_authorized": False,
                "adoption_authorized": False,
                "interrupted": False,
                "promotion": {
                    "decision": "evaluation",
                    "authorized": False,
                },
                "diagnostic_assessment": {"decision": "promote"},
                "evaluation_metrics": {
                    "requested_pairs": len(pairs),
                    "completed_pairs": len(pairs),
                },
                "arena_contract": {
                    "rings": [4],
                    "pairs_per_ring": len(pairs),
                    "simulations": 8,
                    "max_considered": 4,
                    "c_visit": 16.0,
                    "c_scale": 1.0,
                    "pair_chunk_size": None,
                },
                "search": {
                    "deterministic": True,
                    "simulations": 8,
                    "max_considered": 4,
                    "c_visit": 16.0,
                    "c_scale": 1.0,
                    "pie_rule": False,
                    "pair_chunk_size": None,
                },
                "pairs": [asdict(pair) for pair in pairs],
                "games": [asdict(game) for game in games],
            },
            sort_keys=True,
        ),
        encoding="utf-8",
    )


def _architecture_suite(tmp_path: Path) -> SimpleNamespace:
    experiment = load_config(Path(__file__).parents[1] / "configs" / "small.yaml")
    control_model = replace(
        experiment.model,
        width=16,
        rrt_groups=1,
        attention_heads=4,
        kv_heads=1,
    )
    treatment_model = replace(control_model, kv_heads=4)
    control = _publish_model(
        tmp_path / "control",
        role="control",
        model_config=control_model,
        experiment=experiment,
        seed=1,
    )
    treatment = _publish_model(
        tmp_path / "treatment",
        role="treatment",
        model_config=treatment_model,
        experiment=experiment,
        seed=2,
    )
    baseline = _publish_model(
        tmp_path / "baseline",
        role="baseline",
        model_config=control_model,
        experiment=experiment,
        seed=3,
    )
    manifests = {
        "control": control,
        "treatment": treatment,
        "baseline": baseline,
    }
    configs = {
        role: extract_verified_manifest_config(manifest)
        for role, manifest in manifests.items()
    }
    arenas = {
        "control_vs_baseline": tmp_path / "control-vs-baseline.json",
        "treatment_vs_baseline": tmp_path / "treatment-vs-baseline.json",
        "treatment_vs_control": tmp_path / "treatment-vs-control.json",
    }
    _write_arena(
        arenas["control_vs_baseline"],
        candidate=control,
        baseline=baseline,
        candidate_config=configs["control"],
        baseline_config=configs["baseline"],
        outcomes=((1, -1), (-1, 1)),
    )
    _write_arena(
        arenas["treatment_vs_baseline"],
        candidate=treatment,
        baseline=baseline,
        candidate_config=configs["treatment"],
        baseline_config=configs["baseline"],
        outcomes=((1, 1), (1, -1)),
    )
    _write_arena(
        arenas["treatment_vs_control"],
        candidate=treatment,
        baseline=control,
        candidate_config=configs["treatment"],
        baseline_config=configs["control"],
        outcomes=((1, -1), (1, 1)),
    )
    suite_path = tmp_path / "suite.json"
    suite = architecture_suite_document(
        suite_id="ring10-attention-reallocation-seed17",
        control_manifest=control.artifact_manifest or control.path,
        treatment_manifest=treatment.artifact_manifest or treatment.path,
        baseline_manifest=baseline.artifact_manifest or baseline.path,
        control_vs_baseline=arenas["control_vs_baseline"],
        treatment_vs_baseline=arenas["treatment_vs_baseline"],
        treatment_vs_control=arenas["treatment_vs_control"],
    )
    suite_path.write_text(json.dumps(suite, sort_keys=True), encoding="utf-8")
    return SimpleNamespace(
        path=suite_path,
        payload=suite,
        manifests=manifests,
        arenas=arenas,
    )


def test_architecture_ablation_emits_deterministic_diagnostic_evidence(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    suite = _architecture_suite(tmp_path)

    first = build_architecture_ablation_evidence(suite.path)
    second = build_architecture_ablation_evidence(suite.path)

    assert first == second
    assert first["format"] == EVIDENCE_FORMAT
    assert first["diagnostic_only"] is True
    assert first["production_promotion_authorized"] is False
    assert first["adoption_authorized"] is False
    assert first["direct_treatment_control_crossplay"] is True
    assert first["common_frozen_baseline"] == (
        suite.manifests["baseline"].model_identity
    )
    assert first["pair_validity"]["pair_count_per_comparison"] == 2
    assert set(first["comparisons"]) == {
        "control_vs_baseline",
        "treatment_vs_baseline",
        "treatment_vs_control",
    }
    assert all(
        comparison["complete_role_reversed_pairs"] is True
        for comparison in first["comparisons"].values()
    )

    output = tmp_path / "evidence.json"
    main(["--suite-manifest", str(suite.path), "--output", str(output)])
    summary = json.loads(capsys.readouterr().out)
    assert summary["production_promotion_authorized"] is False
    assert json.loads(output.read_text(encoding="utf-8")) == first


def test_architecture_ablation_rejects_incomplete_or_unpinned_evidence(
    tmp_path: Path,
) -> None:
    suite = _architecture_suite(tmp_path)
    direct_path = suite.arenas["treatment_vs_control"]
    direct = json.loads(direct_path.read_text(encoding="utf-8"))
    direct["games"].pop()
    direct_path.write_text(json.dumps(direct, sort_keys=True), encoding="utf-8")

    with pytest.raises(ValueError, match="byte length|SHA-256"):
        build_architecture_ablation_evidence(suite.path)

    suite.payload["arenas"]["treatment_vs_control"] = _pin(direct_path)
    suite.path.write_text(
        json.dumps(suite.payload, sort_keys=True),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="complete role-reversed"):
        build_architecture_ablation_evidence(suite.path)


def test_architecture_ablation_rejects_cherry_picked_opening_schedule(
    tmp_path: Path,
) -> None:
    suite = _architecture_suite(tmp_path)
    treatment_path = suite.arenas["treatment_vs_baseline"]
    treatment = json.loads(treatment_path.read_text(encoding="utf-8"))
    for pair in treatment["pairs"]:
        pair["opening_seed"] += 1
    for game in treatment["games"]:
        game["opening_seed"] += 1
    treatment_path.write_text(
        json.dumps(treatment, sort_keys=True),
        encoding="utf-8",
    )
    suite.payload["arenas"]["treatment_vs_baseline"] = _pin(treatment_path)
    suite.path.write_text(
        json.dumps(suite.payload, sort_keys=True),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="one complete opening schedule"):
        build_architecture_ablation_evidence(suite.path)


def test_architecture_ablation_rejects_evaluator_runtime_drift(
    tmp_path: Path,
) -> None:
    suite = _architecture_suite(tmp_path)
    treatment_path = suite.arenas["treatment_vs_baseline"]
    treatment = json.loads(treatment_path.read_text(encoding="utf-8"))
    treatment["evaluator_contract"]["precision"] = "bf16"
    treatment_path.write_text(
        json.dumps(treatment, sort_keys=True),
        encoding="utf-8",
    )
    suite.payload["arenas"]["treatment_vs_baseline"] = _pin(treatment_path)
    suite.path.write_text(
        json.dumps(suite.payload, sort_keys=True),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="one evaluator contract"):
        build_architecture_ablation_evidence(suite.path)


def test_architecture_ablation_rejects_manifest_checkpoint_identity_drift(
    tmp_path: Path,
) -> None:
    suite = _architecture_suite(tmp_path)
    control = suite.manifests["control"]
    artifact = control.artifact_manifest or control.path
    payload = json.loads(artifact.read_text(encoding="utf-8"))
    wrong_identity = "sha256-" + "f" * 64
    assert wrong_identity != control.model_identity
    payload["model_identity"] = wrong_identity
    payload["model_version"] = wrong_identity
    staged = artifact.parent / "forged.json"
    staged.write_text(json.dumps(payload, sort_keys=True), encoding="utf-8")
    forged = artifact.parent / f"manifest-{sha256_file(staged)}.json"
    staged.replace(forged)
    suite.payload["models"]["control"] = _pin(forged)
    suite.path.write_text(
        json.dumps(suite.payload, sort_keys=True),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="identity does not match"):
        build_architecture_ablation_evidence(suite.path)
