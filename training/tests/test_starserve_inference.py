from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace
from types import SimpleNamespace
import threading

import pytest
import torch

import starserve.runtime as runtime_module
from starserve.config import (
    LimitConfig,
    ServerConfigError,
    ServingInferenceConfig,
    load_server_config,
)
from starserve.runtime import AtomicModelManager, NativeAnalysisService, SearchCancelled
from starserve.schemas import AnalyzeRequest
from startrain.checkpoint import ModelManifest
from startrain.config import load_config
from startrain.features import encode_batch
from startrain.inference_batching import CohortInferenceAdapter
from test_inference_efficiency import ObservedNetwork, encoded_requests, position
from test_starserve import request_payload, server_config


@pytest.fixture
def model_manager(tmp_path, monkeypatch):
    models = []
    selected = [1]

    def make_model(_config):
        model = ObservedNetwork()
        models.append(model)
        return model

    def load_weights(checkpoint, *, model, **_kwargs):
        step = int(checkpoint.stem)
        with torch.no_grad():
            model.bias.fill_(step / 10)
        return {"step": step}

    def manifest(_path):
        step = selected[0]
        return ModelManifest(
            tmp_path / "manifest.json",
            tmp_path / f"{step}.pt",
            f"server-model-{step}",
            step,
            step,
            role="champion",
            model_identity=f"sha256-{step}",
        )

    monkeypatch.setattr(runtime_module, "GraphResTNet", make_model)
    monkeypatch.setattr(runtime_module, "load_ema_checkpoint", load_weights)
    monkeypatch.setattr(
        "startrain.inference.encode_native_feature_data", lambda data, **_: data.encoded
    )
    managers = []

    def create(*, concurrency=1, inference=None):
        configuration = server_config(
            tmp_path,
            limits=LimitConfig(max_concurrency=concurrency),
            inference=inference or ServingInferenceConfig(),
        )
        experiment = load_config("configs/small.yaml")
        experiment = replace(
            experiment, train=replace(experiment.train, compile=False, precision="fp32")
        )
        manager = AtomicModelManager(
            configuration, experiment=experiment, manifest_reader=manifest
        )
        managers.append(manager)
        return manager

    yield create, models, selected
    for manager in managers:
        manager.close()


def test_server_reuses_exact_predictions_and_discards_retired_model_cache(
    model_manager,
):
    create, models, selected = model_manager
    manager = create()
    request = encoded_requests(encode_batch([position()]))
    with manager.lease() as lease:
        bundle = lease.model
        assert isinstance(bundle.evaluator, CohortInferenceAdapter)
        assert bundle.broker.max_wait_seconds == 0
        first = bundle.evaluator.evaluate_detailed(request)
        repeated = bundle.evaluator.evaluate_detailed(
            encoded_requests(request.features.encoded, token_start=90)
        )
        assert repeated.response.tokens == [90]
        assert repeated.response.values == first.response.values
        assert repeated.response.policy_logits == first.response.policy_logits
        assert repeated.outcome_probabilities == first.outcome_probabilities
        assert models[0].rows == [1]
        stats = bundle.evaluator.base.efficiency_snapshot()
        assert stats["cache_hits"] == 1
        assert 0 < stats["cache_bytes"] <= manager.config.inference.cache_max_bytes
        # PDA is a model input and must never collide with the cached position.
        different = bundle.evaluator.evaluate(
            encoded_requests(encode_batch([position(pda=2)]))
        )
        assert different.values != first.response.values
        assert models[0].rows == [1, 1]
    selected[0] = 2
    with manager.lease() as lease:
        refreshed = lease.model.evaluator.evaluate(request)
        assert refreshed.values != first.response.values
        assert models[1].rows == [1]
        assert not bundle.broker._thread.is_alive()
        assert bundle.evaluator.base.efficiency_snapshot()["cache_entries"] == 0


def test_server_batches_concurrent_callers_and_routes_each_result(model_manager):
    create, models, _ = model_manager
    manager = create(
        concurrency=2,
        inference=ServingInferenceConfig(max_wait_seconds=0.5, max_batch_rows=2),
    )
    requests = [
        encoded_requests(encode_batch([position(pda=pda)]), token_start=token)
        for pda, token in ((0, 10), (2, 20))
    ]
    barrier = threading.Barrier(2)

    def analyze(request):
        with manager.lease() as lease:
            barrier.wait(timeout=3)
            assert lease.model.broker.max_wait_seconds == 0.5
            return lease.model.evaluator.evaluate(request)

    with ThreadPoolExecutor(max_workers=2) as pool:
        pending = [pool.submit(analyze, request) for request in requests]
        actual = [future.result(timeout=5) for future in pending]
    with manager.lease() as lease:
        adapter = lease.model.evaluator
        assert lease.model.broker.max_wait_seconds == 0
        assert [response.tokens for response in actual] == [[10], [20]]
        assert actual[0].values != actual[1].values
        assert models[0].rows == [2]
        assert models[0].threads == ["star-inference-owner"]
        # Repeating each row independently must recover its own original result.
        assert [adapter.evaluate(request) for request in requests] == actual
        assert models[0].rows == [2]


def test_server_prediction_storage_obeys_its_entry_and_byte_limits(model_manager):
    create, models, _ = model_manager
    manager = create(
        inference=ServingInferenceConfig(cache_max_entries=1, cache_max_bytes=1_000_000)
    )
    with manager.lease() as lease:
        for pda in (0, 2, 0):
            lease.model.evaluator.evaluate(
                encoded_requests(encode_batch([position(pda=pda)]))
            )
            stats = lease.model.evaluator.base.efficiency_snapshot()
            assert stats["cache_entries"] <= 1
            assert stats["cache_bytes"] <= 1_000_000
        assert stats["cache_evictions"] == 2
        assert models[0].rows == [1, 1, 1]


def test_cancellation_drains_owned_inference_before_model_shutdown(
    model_manager, monkeypatch
):
    create, models, _ = model_manager
    manager = create()
    manager.startup()
    started = models[0].started = threading.Event()
    release = models[0].release = threading.Event()
    cancelled = threading.Event()
    request = encoded_requests(encode_batch([position()]))

    class Search:
        def __init__(self, *_args, **_kwargs):
            pass

        def root_requests(self):
            return request

        def initialize_roots(self, *_args):
            raise AssertionError("cancelled inference must not enter the search")

    service = NativeAnalysisService(
        manager.config,
        native_module=SimpleNamespace(SearchBatch=Search),
        model_manager=manager,
    )
    monkeypatch.setattr(service, "_import_state", lambda _request: None)
    with ThreadPoolExecutor(max_workers=2) as pool:
        analysis = pool.submit(
            service.analyze, AnalyzeRequest.model_validate(request_payload()), cancelled
        )
        assert started.wait(3)
        cancelled.set()
        shutdown = pool.submit(service.shutdown)
        assert not shutdown.done()
        release.set()
        with pytest.raises(SearchCancelled):
            analysis.result(timeout=5)
        shutdown.result(timeout=5)
    assert manager.health()["active_requests"] == 0
    assert manager.health()["ready"] is False
    with pytest.raises(RuntimeError, match="closed"):
        with manager.lease():
            pass


@pytest.mark.parametrize(
    "changes",
    [
        {"cache_max_entries": -1},
        {"cache_max_bytes": 0},
        {"shared_batching": 1},
        {"max_batch_rows": 0},
        {"max_pending_requests": False},
        {"max_wait_seconds": float("nan")},
        {"max_wait_seconds": -0.1},
    ],
)
def test_server_inference_limits_reject_invalid_values(changes):
    with pytest.raises(ServerConfigError):
        ServingInferenceConfig(**changes)


def test_shipped_server_config_enables_bounded_inference():
    config = load_server_config("configs/starserve.yaml")
    assert config.inference == ServingInferenceConfig()
