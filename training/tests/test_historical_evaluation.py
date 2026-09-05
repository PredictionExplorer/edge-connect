from __future__ import annotations

import json
from pathlib import Path

from startrain.checkpoint import ModelManifest
from startrain.config import HistoricalEvaluationConfig
from startrain.historical_evaluation import (
    HISTORICAL_CROSSPLAY_RESULT_KIND,
    arena_result_kind,
    load_arena_results,
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


def crossplay(
    tmp_path: Path,
    *,
    candidate: str,
    baseline: str,
    terminal: bool,
) -> tuple[Path, dict[str, object]]:
    return (
        tmp_path / f"crossplay-{candidate}-vs-{baseline}.json",
        {
            "candidate": candidate,
            "baseline": baseline,
            "result_kind": HISTORICAL_CROSSPLAY_RESULT_KIND,
            "terminal": terminal,
            "pairs": [],
        },
    )


def test_selects_nearest_non_adjacent_promoted_checkpoint(tmp_path: Path) -> None:
    zero = manifest(tmp_path, identity="model-zero", step=0)
    first = manifest(tmp_path, identity="model-first", step=10)
    champion = manifest(tmp_path, identity="model-champion", step=20)
    manifests = {item.model_identity: item for item in (zero, first, champion)}
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
    manifests = {item.model_identity: item for item in (zero, first, champion)}
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


def test_current_measurement_link_follows_completed_older_links_at_own_budget(
    tmp_path: Path,
) -> None:
    from startrain.config import ArenaConfig

    zero = manifest(tmp_path, identity="model-zero", step=0)
    first = manifest(tmp_path, identity="model-first", step=10)
    champion = manifest(tmp_path, identity="model-champion", step=20)
    manifests = {item.model_identity: item for item in (zero, first, champion)}
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
        crossplay(
            tmp_path,
            candidate=first.model_identity,
            baseline=zero.model_identity,
            terminal=True,
        ),
    ]
    config = HistoricalEvaluationConfig(
        enabled=True,
        every_promotions=3,
        pairs_per_ring=50,
        max_pairs_per_ring=100,
        simulations=1024,
        max_considered=32,
        measure_direct_predecessor=True,
    )
    assert config.search_budget(ArenaConfig(simulations=256, max_considered=16)) == (
        1024,
        32,
    )
    assert HistoricalEvaluationConfig().search_budget(
        ArenaConfig(simulations=256, max_considered=16)
    ) == (256, 16)

    # The current direct predecessor link is due regardless of the anchor
    # cadence once earlier links are complete.
    plan = select_historical_evaluation(
        config=config,
        champion=champion,
        manifests=manifests,
        arena_results=results,
        results_directory=tmp_path,
    )
    assert plan is not None
    assert plan.kind == "measurement"
    assert plan.baseline is first
    assert plan.result_path.name == "crossplay-model-champion-vs-model-first.json"

    # A partial measurement resumes; a terminal one falls through to the
    # anchor cadence, which is not due at three promotions.
    partial = {
        "result_kind": HISTORICAL_CROSSPLAY_RESULT_KIND,
        "candidate": champion.model_identity,
        "baseline": first.model_identity,
        "terminal": False,
        "pairs": [],
    }
    resumed = select_historical_evaluation(
        config=config,
        champion=champion,
        manifests=manifests,
        arena_results=[*results, (plan.result_path, partial)],
        results_directory=tmp_path,
    )
    assert resumed is not None
    assert resumed.kind == "measurement"
    assert resumed.previous == partial
    partial["terminal"] = True
    assert (
        select_historical_evaluation(
            config=config,
            champion=champion,
            manifests=manifests,
            arena_results=[*results, (plan.result_path, partial)],
            results_directory=tmp_path,
        )
        is None
    )
    anchor = select_historical_evaluation(
        config=HistoricalEvaluationConfig(
            enabled=True,
            every_promotions=2,
            measure_direct_predecessor=True,
        ),
        champion=champion,
        manifests=manifests,
        arena_results=[*results, (plan.result_path, partial)],
        results_directory=tmp_path,
    )
    assert anchor is not None
    assert anchor.kind == "anchor"
    assert anchor.baseline is zero


def test_unfinished_older_transition_survives_champion_advance(tmp_path: Path) -> None:
    zero = manifest(tmp_path, identity="model-zero", step=0)
    first = manifest(tmp_path, identity="model-first", step=10)
    champion = manifest(tmp_path, identity="model-champion", step=20)
    manifests = {item.model_identity: item for item in (zero, first, champion)}
    old_path, partial = crossplay(
        tmp_path,
        candidate=first.model_identity,
        baseline=zero.model_identity,
        terminal=False,
    )
    # Preserve the existing evidence path, including its contract suffix.
    old_path = old_path.with_name(f"{old_path.stem}-contract.json")
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
        (old_path, partial),
    ]
    config = HistoricalEvaluationConfig(
        enabled=True, every_promotions=100, measure_direct_predecessor=True
    )

    plan = select_historical_evaluation(
        config=config,
        champion=champion,
        manifests=manifests,
        arena_results=results,
        results_directory=tmp_path,
    )
    assert plan is not None
    assert plan.kind == "measurement"
    assert plan.candidate is first
    assert plan.baseline is zero
    assert plan.result_path == old_path
    assert plan.previous == partial

    partial["terminal"] = True
    next_plan = select_historical_evaluation(
        config=config,
        champion=champion,
        manifests=manifests,
        arena_results=results,
        results_directory=tmp_path,
    )
    assert next_plan is not None
    assert next_plan.candidate is champion
    assert next_plan.baseline is first
    assert next_plan.previous is None


def test_measurement_backlog_order_is_stable_with_duplicate_promotions(
    tmp_path: Path,
) -> None:
    zero = manifest(tmp_path, identity="model-zero", step=0)
    first = manifest(tmp_path, identity="model-a", step=10)
    second = manifest(tmp_path, identity="model-b", step=10)
    champion = manifest(tmp_path, identity="model-champion", step=20)
    manifests = {item.model_identity: item for item in (zero, first, second, champion)}
    results = [
        promotion(
            tmp_path,
            candidate=champion.model_identity,
            baseline=second.model_identity,
            completed_ns=2,
        ),
        promotion(
            tmp_path,
            candidate=second.model_identity,
            baseline=first.model_identity,
            completed_ns=1,
        ),
        promotion(
            tmp_path,
            candidate=first.model_identity,
            baseline=zero.model_identity,
            completed_ns=1,
        ),
        promotion(
            tmp_path,
            candidate=first.model_identity,
            baseline=zero.model_identity,
            completed_ns=99,
        ),
    ]
    options = {
        "config": HistoricalEvaluationConfig(
            enabled=True, measure_direct_predecessor=True
        ),
        "champion": champion,
        "manifests": manifests,
        "results_directory": tmp_path,
    }
    # Missing links fill the oldest gap; timestamp and step ties use identity.
    for ordered in (results, list(reversed(results))):
        plan = select_historical_evaluation(arena_results=ordered, **options)
        assert plan is not None
        assert plan.candidate is first
        assert plan.baseline is zero

    # Any started link is resumed before an older missing one, and among
    # started links the oldest transition wins independent of file order.
    partials = [
        crossplay(
            tmp_path,
            candidate=champion.model_identity,
            baseline=second.model_identity,
            terminal=False,
        ),
        crossplay(
            tmp_path,
            candidate=second.model_identity,
            baseline=first.model_identity,
            terminal=False,
        ),
    ]
    for ordered in (results + partials, list(reversed(results + partials))):
        plan = select_historical_evaluation(arena_results=ordered, **options)
        assert plan is not None
        assert plan.candidate is second
        assert plan.baseline is first


def test_duplicate_terminal_crossplay_wins_over_partial_evidence(
    tmp_path: Path,
) -> None:
    zero = manifest(tmp_path, identity="model-zero", step=0)
    champion = manifest(tmp_path, identity="model-champion", step=10)
    results = [
        promotion(
            tmp_path,
            candidate=champion.model_identity,
            baseline=zero.model_identity,
            completed_ns=1,
        ),
        crossplay(
            tmp_path,
            candidate=champion.model_identity,
            baseline=zero.model_identity,
            terminal=True,
        ),
    ]
    stale_path, stale = crossplay(
        tmp_path,
        candidate=champion.model_identity,
        baseline=zero.model_identity,
        terminal=False,
    )
    results.append((stale_path.with_name("stale.json"), stale))
    for ordered in (results, list(reversed(results))):
        assert (
            select_historical_evaluation(
                config=HistoricalEvaluationConfig(
                    enabled=True, measure_direct_predecessor=True
                ),
                champion=champion,
                manifests={item.model_identity: item for item in (zero, champion)},
                arena_results=ordered,
                results_directory=tmp_path,
            )
            is None
        )


def test_resume_snapshots_are_excluded_from_arena_evidence(tmp_path: Path) -> None:
    zero = manifest(tmp_path, identity="model-zero", step=0)
    champion = manifest(tmp_path, identity="model-champion", step=10)
    path, payload = promotion(
        tmp_path,
        candidate=champion.model_identity,
        baseline=zero.model_identity,
        completed_ns=1,
    )
    resume_path = path.with_suffix(".resume.json")
    resume_path.write_text(json.dumps(payload), encoding="utf-8")
    assert load_arena_results(tmp_path) == []
    assert (
        select_historical_evaluation(
            config=HistoricalEvaluationConfig(
                enabled=True, measure_direct_predecessor=True
            ),
            champion=champion,
            manifests={item.model_identity: item for item in (zero, champion)},
            arena_results=[(resume_path, payload)],
            results_directory=tmp_path,
        )
        is None
    )

    path.write_text(json.dumps(payload), encoding="utf-8")
    terminal_path, terminal = crossplay(
        tmp_path,
        candidate=champion.model_identity,
        baseline=zero.model_identity,
        terminal=True,
    )
    terminal_path.with_suffix(".resume.json").write_text(
        json.dumps(terminal), encoding="utf-8"
    )
    assert load_arena_results(tmp_path) == [(path, payload)]
    plan = select_historical_evaluation(
        config=HistoricalEvaluationConfig(
            enabled=True, measure_direct_predecessor=True
        ),
        champion=champion,
        manifests={item.model_identity: item for item in (zero, champion)},
        arena_results=load_arena_results(tmp_path),
        results_directory=tmp_path,
    )
    assert plan is not None
    assert plan.kind == "measurement"
    assert plan.previous is None


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
            manifests={item.model_identity: item for item in (zero, first, champion)},
            arena_results=results,
            results_directory=tmp_path,
        )
        is None
    )
