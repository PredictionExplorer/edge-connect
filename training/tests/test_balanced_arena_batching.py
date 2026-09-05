from __future__ import annotations

import threading
import time
from concurrent.futures import ThreadPoolExecutor

import pytest

from startrain.arena import ArenaRunner
from startrain.config import ArenaConfig
from startrain.inference import GraphInferenceAdapter, InferenceConfig
from test_inference_efficiency import ObservedNetwork


def evaluator() -> GraphInferenceAdapter:
    return GraphInferenceAdapter(
        ObservedNetwork(),
        config=InferenceConfig(precision="fp32"),
        model_version="sha256-" + "a" * 64,
        model_identity="sha256-" + "a" * 64,
    )


def config() -> ArenaConfig:
    return ArenaConfig(
        balanced_cells=True,
        pairs_per_ring=4,
        minimum_pairs_per_ring=4,
        simulations=2,
        max_considered=2,
    )


@pytest.mark.native
def test_parallel_balanced_groups_batch_neural_requests_without_changing_pairs() -> (
    None
):
    native = pytest.importorskip("star_native")
    counts = {4: 4, 6: 0, 8: 0, 10: 0}
    serial_adapter, shared_adapter = evaluator(), evaluator()
    serial = ArenaRunner(
        native_module=native,
        candidate=serial_adapter,
        baseline=serial_adapter,
        config=config(),
        parallel_variant_groups=1,
    ).run(pair_counts=counts)
    shared = ArenaRunner(
        native_module=native,
        candidate=shared_adapter,
        baseline=shared_adapter,
        config=config(),
    ).run(pair_counts=counts)

    def key(item):
        return item["ring"], item["variant"], item["pair"]

    assert sorted(shared["pairs"], key=key) == sorted(serial["pairs"], key=key)
    assert len(shared["pairs"]) == 24
    metrics = shared["evaluation_metrics"]["shared_inference"]
    assert metrics["neural_batches"] < metrics["submitted_requests"]
    assert metrics["submitted_requests"] == metrics["completed_requests"]
    assert metrics["pending_requests"] == metrics["failed_requests"] == 0
    assert set(shared_adapter.model.threads) == {"star-inference-owner"}
    assert max(shared_adapter.model.rows) > max(serial_adapter.model.rows)
    assert shared["search"]["variant_group_workers"] == 12
    assert shared["search"]["inference_execution"] == "shared_broker"
    assert serial["search"]["inference_execution"] == "serialized"


@pytest.mark.native
@pytest.mark.parametrize("fail", [False, True])
def test_balanced_groups_drain_all_inference_before_owner_shutdown(fail) -> None:
    native = pytest.importorskip("star_native")
    adapter = evaluator()
    adapter.model.started, adapter.model.release = threading.Event(), threading.Event()
    adapter.model.fail = fail
    stopped = threading.Event()
    runner = ArenaRunner(
        native_module=native, candidate=adapter, baseline=adapter, config=config()
    )
    with ThreadPoolExecutor(max_workers=1) as pool:
        future = pool.submit(
            runner.run,
            pair_counts={4: 4, 6: 0, 8: 0, 10: 0},
            stop_requested=stopped.is_set,
        )
        assert adapter.model.started.wait(3)
        broker = runner._shared_broker
        assert broker is not None
        deadline = time.monotonic() + 3
        while broker.metrics_snapshot()["submitted_requests"] < 2:
            assert time.monotonic() < deadline
            time.sleep(0.001)
        if not fail:
            stopped.set()
        adapter.model.release.set()
        if fail:
            with pytest.raises(ValueError, match="neural failure"):
                future.result(timeout=10)
        else:
            assert future.result(timeout=10)["interrupted"]
        assert not broker._thread.is_alive()
        assert runner._shared_broker is None
        metrics = broker.metrics_snapshot()
        assert metrics["pending_requests"] == 0
        assert (
            metrics["submitted_requests"]
            == metrics["completed_requests"] + metrics["failed_requests"]
        )


def test_balanced_group_parallelism_is_bounded() -> None:
    for count in (0, 13, True):
        with pytest.raises(ValueError, match="parallel_variant_groups"):
            ArenaRunner(
                native_module=object(),
                candidate=object(),
                baseline=object(),
                config=config(),
                parallel_variant_groups=count,
            )
