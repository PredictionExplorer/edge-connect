from concurrent.futures import ThreadPoolExecutor
from dataclasses import FrozenInstanceError
import json
import threading

import pytest
import torch

from startrain.features import encode_batch
from startrain.inference import GraphInferenceAdapter, InferenceConfig
from startrain.inference_batching import BoundedInferenceBroker
from startrain.inference_graphs import _graph_storage_breakdown
from test_inference_efficiency import ObservedNetwork, encoded_requests, position
from test_inference_graphs import FakeBackend, cache


class AccountedBackend(FakeBackend):
    def capture(self, forward, args, kwargs):
        entry = super().capture(forward, args, kwargs)
        entry.private_pool_bytes = self.charge - 20
        entry.external_static_bytes = 20
        return entry


def test_storage_breakdown_uses_the_same_conservative_allocator_charges():
    private, external = _graph_storage_breakdown(
        [
            {
                "device": 0,
                "segment_pool_id": (1, 2),
                "address": 1000,
                "total_size": 100,
            },
            {
                "device": 0,
                "segment_pool_id": (0, 0),
                "address": 2000,
                "total_size": 1000,
                "blocks": [{"address": 2000, "size": 512}],
            },
        ],
        pool=(1, 2),
        device_index=0,
        storages=[(2000, 4), (2000, 4)],
    )
    assert (private, external) == (100, 512)


def test_residency_key_is_stable_for_values_and_distinguishes_original_layout():
    graphs = cache(AccountedBackend(), max_entries=4)

    def forward(value, *, ring, bias):
        return value + bias + ring

    graphs.run(("model",), forward, (torch.ones(2, 3),), {"ring": 4, "bias": 1})
    first = graphs.residency_snapshot()
    record = first[0]
    assert (record.rows, record.nodes) == (2, 3)
    assert len(record.key_sha256) == 64
    assert record.kwarg_names == ("bias", "ring")
    assert (
        record.charged_bytes,
        record.private_pool_bytes,
        record.external_static_bytes,
    ) == (100, 80, 20)
    graphs.run(("model",), forward, (torch.zeros(2, 3),), {"bias": 1, "ring": 4})
    assert graphs.residency_snapshot() is first
    graphs.run(("model",), forward, (torch.ones(3, 2).t(),), {"ring": 4, "bias": 1})
    second = graphs.residency_snapshot()
    assert len(second) == 2
    assert len({entry.key_sha256 for entry in second}) == 2
    with pytest.raises(FrozenInstanceError):
        record.rows = 99
    assert len(first) == 1


def test_residency_tracks_only_live_entries_after_eviction_and_clear():
    graphs = cache(AccountedBackend(), max_entries=2, max_bytes=200)
    for rows in (1, 2):
        graphs.run(("model",), lambda value: value, (torch.ones(rows, 3),), {})
    original = graphs.residency_snapshot()
    graphs.run(("model",), lambda value: value, (torch.ones(3, 3),), {})
    assert [record.rows for record in graphs.residency_snapshot()] == [2, 3]
    assert sum(record.charged_bytes for record in graphs.residency_snapshot()) == 200
    graphs.clear()
    assert graphs.residency_snapshot() == ()
    assert len(original) == 2


def test_rejected_capture_is_not_published_and_unknown_backend_breakdown_is_honest():
    rejected = cache(FakeBackend(charge=101), max_bytes=100)
    rejected.run(("model",), lambda value: value, (torch.ones(1, 2),), {})
    assert rejected.residency_snapshot() == ()
    unknown = cache()
    unknown.run(("model",), lambda value: value, (torch.ones(1, 2),), {})
    record = unknown.residency_snapshot()[0]
    assert record.charged_bytes == 100
    assert record.private_pool_bytes is record.external_static_bytes is None


def test_snapshots_do_not_wait_for_capture_or_access_live_graphs(monkeypatch):
    entered, release = threading.Event(), threading.Event()

    class BlockingBackend(AccountedBackend):
        def capture(self, *args):
            entered.set()
            assert release.wait(3)
            return super().capture(*args)

    graphs = cache(BlockingBackend())
    with ThreadPoolExecutor(max_workers=1) as executor:
        pending = executor.submit(
            graphs.run, ("model",), lambda value: value, (torch.ones(1, 2),), {}
        )
        assert entered.wait(3)
        try:
            assert graphs.residency_snapshot() == ()
        finally:
            release.set()
        pending.result(timeout=3)
    published = graphs.residency_snapshot()

    class NoReads(dict):
        def values(self):
            pytest.fail("heartbeat must not iterate a live graph cache")

        def __iter__(self):
            pytest.fail("heartbeat must not iterate a live graph cache")

    monkeypatch.setattr(graphs, "_entries", NoReads())
    monkeypatch.setattr(graphs, "_residency", NoReads())
    monkeypatch.setattr(torch.cuda, "memory_snapshot", lambda: pytest.fail("GPU read"))
    assert graphs.residency_snapshot() is published


def test_concurrent_snapshots_remain_immutable_during_capture_and_eviction():
    graphs = cache(AccountedBackend(), max_entries=2, max_bytes=200)
    start, done = threading.Event(), threading.Event()
    first_published, first_read = threading.Event(), threading.Event()
    snapshots = []

    def owner():
        assert start.wait(3)
        try:
            for index in range(50):
                graphs.run((str(index),), lambda value: value, (torch.ones(1, 2),), {})
                if index == 0:
                    first_published.set()
                    assert first_read.wait(3)
            graphs.clear()
        finally:
            done.set()

    with ThreadPoolExecutor(max_workers=1) as executor:
        pending = executor.submit(owner)
        start.set()
        assert first_published.wait(3)
        snapshots.append(graphs.residency_snapshot())
        first_read.set()
        while not done.wait(0.001):
            snapshots.append(graphs.residency_snapshot())
        pending.result(timeout=3)
    assert snapshots
    for snapshot in snapshots:
        assert len(snapshot) <= 2
        assert len({record.key_sha256 for record in snapshot}) == len(snapshot)
        assert all(record.charged_bytes == 100 for record in snapshot)
    assert graphs.residency_snapshot() == ()


def test_broker_residency_is_per_adapter_json_safe_and_separate_from_numeric_metrics(
    monkeypatch,
):
    monkeypatch.setattr(
        "startrain.inference.encode_native_feature_data", lambda data, **_: data.encoded
    )
    graph = GraphInferenceAdapter(
        ObservedNetwork(),
        model_identity="weights",
        model_version="version",
        model_step=17,
        config=InferenceConfig(deduplicate=False),
    )
    graph._graphs = cache(AccountedBackend())
    regular = GraphInferenceAdapter(ObservedNetwork(), model_identity="regular")
    request = encoded_requests(encode_batch([position()]))
    with BoundedInferenceBroker(max_batch_rows=1, max_wait_seconds=0) as broker:
        for adapter in (graph, graph, regular):
            broker.submit(adapter, request).result(timeout=3)
        snapshot = broker.metrics_snapshot()
        residency = snapshot["graph_residency"]
        assert len(residency) == 1
        assert residency[0]["model_identity"] == "weights"
        assert residency[0]["model_version"] == "version"
        assert residency[0]["model_step"] == 17
        assert residency[0]["entries"][0]["rows"] == 1
        assert residency[0]["entries"][0]["charged_bytes"] == 100
        assert all(
            type(value) in (int, float)
            for value in snapshot["physical_inference"].values()
        )
        json.dumps(snapshot, allow_nan=False)
        residency[0]["entries"][0]["rows"] = -100
        assert (
            broker.metrics_snapshot()["graph_residency"][0]["entries"][0]["rows"] == 1
        )
        graph.clear_inference_cache()
        assert broker.metrics_snapshot()["graph_residency"][0]["entries"] == []
