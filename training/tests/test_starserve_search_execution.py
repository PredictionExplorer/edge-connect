"""Opt-in server execution through the real native and inference boundaries."""

from dataclasses import replace
import threading

import pytest
import torch

import starserve.runtime as runtime_module
from starserve.config import (
    ServerConfigError,
    ServingInferenceConfig,
    load_server_config,
)
from starserve.runtime import AtomicModelManager, NativeAnalysisService, SearchCancelled
from starserve.schemas import AnalyzeRequest, AnalyzeResponse
from startrain.checkpoint import ModelManifest
from startrain.config import load_config
from startrain.model import GraphResTNet, ModelConfig
from startrain.search_options import FullSearchBudgetConfig, SearchExecutionConfig
from test_inference_efficiency import ObservedNetwork
from test_starserve import request_payload, server_config


@pytest.fixture
def execution_server(tmp_path, monkeypatch):
    native = pytest.importorskip("star_native")
    managers = []
    models = []
    selected = [1]

    def manifest(_path):
        step = selected[0]
        return ModelManifest(
            tmp_path / "manifest.json",
            tmp_path / f"{step}.pt",
            f"v{step}",
            step,
            step,
            role="champion",
            model_identity=f"sha256-{step}",
        )

    def make_model(_config):
        model = ObservedNetwork()
        models.append(model)
        return model

    monkeypatch.setattr(runtime_module, "GraphResTNet", make_model)
    monkeypatch.setattr(
        runtime_module,
        "load_ema_checkpoint",
        lambda path, **_: {"step": int(path.stem)},
    )

    def create(width=8, reuse=False, max_rows=8, max_nodes=4096, shared=True):
        config = server_config(
            tmp_path,
            search_execution=SearchExecutionConfig(
                first_visit_batch_size=width,
                subtree_reuse=reuse,
                subtree_reuse_max_nodes=max_nodes,
            ),
            inference=ServingInferenceConfig(
                cache_max_entries=0,
                cache_max_bytes=0,
                shared_batching=shared,
                max_batch_rows=max_rows,
            ),
        )
        experiment = load_config("configs/small.yaml")
        experiment = replace(
            experiment, train=replace(experiment.train, compile=False, precision="fp32")
        )
        manager = AtomicModelManager(
            config, experiment=experiment, manifest_reader=manifest
        )
        managers.append(manager)
        service = NativeAnalysisService(
            config, native_module=native, model_manager=manager
        )
        service.startup()
        with manager.lease() as lease:
            evaluator = lease.model.evaluator
            network = getattr(evaluator, "base", evaluator).model
        return service, manager, network

    yield create, selected
    for manager in managers:
        manager.close()


def analyze(service, payload=None, cancellation=None):
    payload = payload or request_payload()
    return service.analyze(
        AnalyzeRequest.model_validate(payload), cancellation or threading.Event()
    )


def descendant(payload, action):
    result = dict(payload)
    result["stones"] = list(payload["stones"])
    result["stones"][action] = 0
    result.update(
        to_move=1,
        moves_left=2,
        opening=False,
        history={
            "current_turn": [],
            "previous_turn": [action],
            "own_previous_turn": [],
            "handicap_stones": [action],
        },
    )
    return result


@pytest.mark.native
@pytest.mark.parametrize("shared,cap", [(False, 8), (True, 8), (True, 3)])
def test_server_prefetch_obeys_broker_cap_and_preserves_results(
    execution_server, shared, cap
):
    create, _ = execution_server
    baseline, _, serial_model = create(width=1, shared=shared)
    batched, manager, batch_model = create(max_rows=cap, shared=shared)
    payload = request_payload()
    payload["search"] = {"simulations": 16, "max_considered": 16, "seed": 17}
    before = analyze(baseline, payload)
    after = analyze(batched, payload)
    assert {k: v for k, v in before.items() if k != "timing_ms"} == {
        k: v for k, v in after.items() if k != "timing_ms"
    }
    assert serial_model.rows == [1] * 17
    assert sum(batch_model.rows) == 17
    assert max(batch_model.rows) <= cap
    assert len(batch_model.rows) < len(serial_model.rows)
    if cap == 8:
        assert batch_model.rows == [1, 8, 8]
    with manager.lease() as lease:
        if shared:
            assert lease.model.broker.max_batch_rows == cap
            assert lease.model.broker.max_wait_seconds == 0
        assert lease.model.search_cache is None


@pytest.mark.native
@pytest.mark.parametrize("invalidate", [None, "pda", "model", "unknown-history", "cap"])
def test_server_reuses_only_compatible_completed_descendants(
    execution_server, invalidate
):
    create, selected = execution_server
    service, manager, _ = create(
        reuse=True, max_nodes=1 if invalidate == "cap" else 4096
    )
    payload = request_payload()
    payload["search"] = {"simulations": 16, "max_considered": 2, "seed": 17}
    first = analyze(service, payload)
    with manager.lease() as lease:
        old_pool = lease.model.search_cache
    next_payload = descendant(payload, first["action"]["code"])
    next_payload["search"] = {"simulations": 7, "max_considered": 2, "seed": 19}
    if invalidate == "pda":
        next_payload["pda"] = 1
    elif invalidate == "unknown-history":
        next_payload["history"] = None
    elif invalidate == "model":
        selected[0] = 2
    second = analyze(service, next_payload)
    assert sum(second["root_visits"]) == 7
    with manager.lease() as lease:
        stats = lease.model.search_cache.metrics_snapshot()
        assert stats["hits"] == (1 if invalidate is None else 0)
        assert stats["retained_nodes"] <= stats["max_nodes"]
        if invalidate == "model":
            assert lease.model.search_cache is not old_pool
            assert old_pool.metrics_snapshot()["entries"] == 0


@pytest.mark.native
def test_cancelled_checked_out_search_is_not_retained(execution_server, monkeypatch):
    create, _ = execution_server
    service, manager, _ = create(reuse=True)
    payload = request_payload()
    payload["search"] = {"simulations": 16, "max_considered": 2, "seed": 17}
    first = analyze(service, payload)
    cancellation = threading.Event()
    with manager.lease() as lease:
        evaluator = lease.model.evaluator
        pool = lease.model.search_cache
    original = evaluator.evaluate_detailed

    def cancel_after_root(requests):
        result = original(requests)
        cancellation.set()
        return result

    monkeypatch.setattr(evaluator, "evaluate_detailed", cancel_after_root)
    with pytest.raises(SearchCancelled):
        analyze(service, descendant(payload, first["action"]["code"]), cancellation)
    stats = pool.metrics_snapshot()
    assert stats["hits"] == 1
    assert stats["entries"] == 0 and stats["retained_nodes"] == 0


@pytest.mark.native
def test_real_network_server_batching(execution_server, monkeypatch):
    create, _ = execution_server

    def model(_):
        torch.manual_seed(19)
        return GraphResTNet(
            ModelConfig(width=8, rrt_groups=1, attention_heads=2, kv_heads=1)
        )

    monkeypatch.setattr(runtime_module, "GraphResTNet", model)
    serial, _, _ = create(width=1)
    batched, _, _ = create(width=8)
    payload = request_payload()
    payload["search"] = {"simulations": 16, "max_considered": 16, "seed": 17}
    before, after = analyze(serial, payload), analyze(batched, payload)
    AnalyzeResponse.model_validate({**after, "request_id": "test"})
    assert after["action"] == before["action"]
    assert after["root_visits"] == before["root_visits"]
    assert after["root_q"] == pytest.approx(before["root_q"], abs=2e-6)
    assert after["root_policy"] == pytest.approx(before["root_policy"], abs=2e-6)


def test_serving_execution_config_is_explicit_and_preserves_exact_budgets(tmp_path):
    path = tmp_path / "server.yaml"
    path.write_text(
        "schema_version: 2\ndevice: cpu\nexperiment_config: experiment.yaml\nmodel_manifest: champion.json\n"
        "search_execution:\n  first_visit_batch_size: 8\n  subtree_reuse: true\n"
        "  subtree_reuse_max_nodes: 1024\ninference:\n  search_cache_entries: 4\n"
    )
    config = load_server_config(path)
    assert config.search_execution.first_visit_batch_size == 8
    assert config.search_execution.subtree_reuse
    assert config.inference.search_cache_entries == 4
    with pytest.raises(ServerConfigError, match="exact requested budgets"):
        replace(
            config,
            search_execution=SearchExecutionConfig(
                full_budget=FullSearchBudgetConfig(mode="root-entropy")
            ),
        )
    with pytest.raises(ServerConfigError, match="search_cache_entries"):
        ServingInferenceConfig(search_cache_entries=0)
