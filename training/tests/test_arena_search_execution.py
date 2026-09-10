from dataclasses import replace
from concurrent.futures import ThreadPoolExecutor
import threading

import pytest

from startrain.arena import ArenaRunner
from startrain.config import ArenaConfig
from startrain.search_options import SearchExecutionConfig
from test_balanced_arena_batching import evaluator


def config(execution):
    return ArenaConfig(
        rings=(4,),
        pairs_per_ring=2,
        minimum_pairs_per_ring=2,
        simulations=4,
        max_considered=4,
        bootstrap_samples=200,
        search_execution=execution,
    )


@pytest.mark.native
def test_arena_first_visit_batching_preserves_fixed_search_results():
    native = pytest.importorskip("star_native")
    results = []
    for execution in (
        SearchExecutionConfig(),
        SearchExecutionConfig(first_visit_batch_size=4),
    ):
        adapter = evaluator()
        runner = ArenaRunner(
            native_module=native,
            candidate=adapter,
            baseline=adapter,
            config=config(execution),
            stable_pair_seeds=True,
        )
        results.append(runner.run())
        adapter.close()
    assert results[0]["games"] == results[1]["games"]
    assert results[0]["pairs"] == results[1]["pairs"]
    assert "execution" not in results[0]["search"]
    assert results[1]["search"]["execution"]["first_visit_batch_size"] == 4
    assert results[1]["search"]["deterministic"] is True


@pytest.mark.native
def test_arena_pooled_reuse_is_bounded_and_released_after_producers_drain(monkeypatch):
    native = pytest.importorskip("star_native")
    adapter = evaluator()
    runner = ArenaRunner(
        native_module=native,
        candidate=adapter,
        baseline=adapter,
        config=replace(
            config(
                SearchExecutionConfig(
                    first_visit_batch_size=2,
                    subtree_reuse=True,
                    subtree_reuse_max_nodes=256,
                )
            ),
            unforced_opening_fraction=0.9999,
        ),
        stable_pair_seeds=True,
    )
    contexts = []
    original_take = runner._search_sessions.take

    def take(states, context, pda):
        contexts.append(context)
        return original_take(states, context, pda)

    monkeypatch.setattr(runner._search_sessions, "take", take)
    result = runner.run()
    adapter.close()
    assert len(result["pairs"]) == 2
    metrics = result["evaluation_metrics"]["completed_search_cache"]
    assert metrics["puts"] > 0 and metrics["hits"] > 0
    assert metrics["entries"] == metrics["retained_nodes"] == 0
    assert metrics["max_nodes"] == 256
    # Both participant roles share the same adapter in this test. Distinct
    # contexts must therefore come from ordered logical pair/seat identities.
    assert len(set(contexts)) > 1
    assert all(":arena-group-" in context for context in contexts)
    assert result["search"]["deterministic"] is False
    assert (
        result["search"]["resume_search_policy"] == "fresh-root-statistics-after-resume"
    )


@pytest.mark.native
@pytest.mark.parametrize("failed", [False, True])
def test_interrupted_or_failed_arena_does_not_retain_search_sessions(failed):
    native = pytest.importorskip("star_native")
    adapter = evaluator()
    adapter.model.started = threading.Event()
    adapter.model.release = threading.Event()
    adapter.model.fail = failed
    stopping = threading.Event()
    runner = ArenaRunner(
        native_module=native,
        candidate=adapter,
        baseline=adapter,
        config=config(SearchExecutionConfig(subtree_reuse=True)),
    )
    try:
        with ThreadPoolExecutor(max_workers=1) as pool:
            future = pool.submit(runner.run, stop_requested=stopping.is_set)
            assert adapter.model.started.wait(3)
            stopping.set()
            adapter.model.release.set()
            if failed:
                with pytest.raises(ValueError, match="neural failure"):
                    future.result(timeout=10)
            else:
                assert future.result(timeout=10)["interrupted"]
        metrics = runner._search_sessions.metrics_snapshot()
        assert metrics["entries"] == metrics["retained_nodes"] == 0
        assert metrics["puts"] == 0
    finally:
        adapter.close()


def test_default_resume_contract_omits_additive_execution_defaults():
    adapter = evaluator()
    runner = ArenaRunner(
        native_module=object(),
        candidate=adapter,
        baseline=adapter,
        config=config(SearchExecutionConfig()),
        stable_pair_seeds=True,
    )
    runner._initialize_resume(None, lambda _: None)
    assert "search_execution" not in runner._resume_contract["config"]
    changed = replace(
        runner.config, search_execution=SearchExecutionConfig(first_visit_batch_size=2)
    )
    assert changed.search_execution.contract() is not None
