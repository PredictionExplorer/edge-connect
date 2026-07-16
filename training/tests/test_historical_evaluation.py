from __future__ import annotations

from pathlib import Path

from startrain.checkpoint import ModelManifest
from startrain.config import HistoricalEvaluationConfig
from startrain.historical_evaluation import (
    HISTORICAL_CROSSPLAY_RESULT_KIND,
    arena_result_kind,
    select_historical_evaluation,
)


def manifest(tmp_path: Path, *, identity: str, step: int) -> ModelManifest:
    return ModelManifest(
        path=tmp_path / f"manifest-{identity}.json",
        checkpoint=tmp_path / f"{identity}.pt",
        model_version=identity,
        model_step=step,
        published_ns=step + 1,
        model_identity=identity,
        checkpoint_sha256=f"{step:064x}",
        checkpoint_bytes=1,
        run_id="run-test",
        generation_family="family-test",
    )


def promotion(
    tmp_path: Path,
    *,
    candidate: str,
    baseline: str,
    completed_ns: int,
) -> tuple[Path, dict[str, object]]:
    return (
        tmp_path / f"{candidate}-vs-{baseline}.json",
        {
            "candidate": candidate,
            "baseline": baseline,
            "completed_ns": completed_ns,
            "promotion": {"decision": "promote"},
            "terminal": True,
        },
    )


def test_selects_nearest_non_adjacent_promoted_checkpoint(tmp_path: Path) -> None:
    zero = manifest(tmp_path, identity="model-zero", step=0)
    first = manifest(tmp_path, identity="model-first", step=10)
    champion = manifest(tmp_path, identity="model-champion", step=20)
    manifests = {
        item.model_identity: item for item in (zero, first, champion)
    }
    results = [
        promotion(
            tmp_path,
            candidate=first.model_identity,
            baseline=zero.model_identity,
            completed_ns=1,
        ),
        promotion(
            tmp_path,
            candidate=champion.model_identity,
            baseline=first.model_identity,
            completed_ns=2,
        ),
    ]

    plan = select_historical_evaluation(
        config=HistoricalEvaluationConfig(enabled=True),
        champion=champion,
        manifests=manifests,
        arena_results=results,
        results_directory=tmp_path,
    )

    assert plan is not None
    assert plan.candidate is champion
    assert plan.baseline is zero
    assert plan.previous is None
    assert plan.result_path.name.startswith("crossplay-model-champion-vs-model-zero")


def test_completed_crossplay_is_idempotent_and_partial_result_resumes(
    tmp_path: Path,
) -> None:
    zero = manifest(tmp_path, identity="model-zero", step=0)
    first = manifest(tmp_path, identity="model-first", step=10)
    champion = manifest(tmp_path, identity="model-champion", step=20)
    manifests = {
        item.model_identity: item for item in (zero, first, champion)
    }
    results = [
        promotion(
            tmp_path,
            candidate=first.model_identity,
            baseline=zero.model_identity,
            completed_ns=1,
        ),
        promotion(
            tmp_path,
            candidate=champion.model_identity,
            baseline=first.model_identity,
            completed_ns=2,
        ),
    ]
    crossplay_path = tmp_path / "crossplay.json"
    partial = {
        "result_kind": HISTORICAL_CROSSPLAY_RESULT_KIND,
        "candidate": champion.model_identity,
        "baseline": zero.model_identity,
        "terminal": False,
        "pairs": [],
    }

    resumed = select_historical_evaluation(
        config=HistoricalEvaluationConfig(enabled=True),
        champion=champion,
        manifests=manifests,
        arena_results=[*results, (crossplay_path, partial)],
        results_directory=tmp_path,
    )
    assert resumed is not None
    assert resumed.result_path == crossplay_path
    assert resumed.previous == partial

    partial["terminal"] = True
    assert (
        select_historical_evaluation(
            config=HistoricalEvaluationConfig(enabled=True),
            champion=champion,
            manifests=manifests,
            arena_results=[*results, (crossplay_path, partial)],
            results_directory=tmp_path,
        )
        is None
    )


def test_schedule_and_legacy_result_classification(tmp_path: Path) -> None:
    zero = manifest(tmp_path, identity="model-zero", step=0)
    first = manifest(tmp_path, identity="model-first", step=10)
    champion = manifest(tmp_path, identity="model-champion", step=20)
    results = [
        promotion(
            tmp_path,
            candidate=first.model_identity,
            baseline=zero.model_identity,
            completed_ns=1,
        ),
        promotion(
            tmp_path,
            candidate=champion.model_identity,
            baseline=first.model_identity,
            completed_ns=2,
        ),
    ]

    assert arena_result_kind({}) == "promotion"
    assert arena_result_kind({"result_kind": "historical_crossplay"}) == (
        HISTORICAL_CROSSPLAY_RESULT_KIND
    )
    assert (
        select_historical_evaluation(
            config=HistoricalEvaluationConfig(enabled=True, every_promotions=3),
            champion=champion,
            manifests={
                item.model_identity: item for item in (zero, first, champion)
            },
            arena_results=results,
            results_directory=tmp_path,
        )
        is None
    )
