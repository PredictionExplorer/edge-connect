from concurrent.futures import ThreadPoolExecutor
from contextlib import nullcontext
import gc
import threading
import traceback
from types import SimpleNamespace
import weakref

import pytest
import torch

from startrain.features import encode_batch
from startrain.inference import GraphInferenceAdapter, InferenceConfig
from startrain.inference_batching import BoundedInferenceBroker
from startrain.inference_graphs import (
    BoundedInferenceGraphs,
    CaptureUnavailable,
    CudaBackend,
    _CaptureStreamPool,
    _CudaExecutable,
    _OwnedCaptureStream,
    _account_graph_storage,
    _bitwise_equal,
    _clone,
    _copy,
    _new_owned_capture_stream,
)
from test_inference_efficiency import ObservedNetwork, encoded_requests, position


class FakeExecutable:
    def __init__(self, forward, args, kwargs, charge):
        self.args = tuple(value.clone() for value in args)
        self.kwargs = {name: _clone(value) for name, value in kwargs.items()}
        self.forward = forward
        self.retained_bytes = charge
        self.warmup_calls = 2
        self.closed = False
        for _ in range(self.warmup_calls):
            output = forward(*self.args, **self.kwargs)
        self.reference_output = output.detach().clone()
        self.output = torch.empty_like(output)

    def replay(self, args, kwargs):
        assert not self.closed
        for target, source in zip(self.args, args, strict=True):
            _copy(target, source)
        for name in self.kwargs:
            _copy(self.kwargs[name], kwargs[name])
        self.output.copy_(self.forward(*self.args, **self.kwargs))
        return self.output

    def close(self):
        self.closed = True
        self.args = ()
        self.kwargs.clear()
        self.output = torch.empty(0)
        self.reference_output = None


class FakeBackend:
    def __init__(self, *, charge=100, error=None):
        self.charge = charge
        self.error = error
        self.attempts = 0
        self.entries = []
        self.threads = []

    def capture(self, forward, args, kwargs):
        self.attempts += 1
        self.threads.append(threading.current_thread().name)
        if self.error is not None:
            raise self.error
        entry = FakeExecutable(forward, args, kwargs, self.charge)
        self.entries.append(entry)
        return entry


def cache(backend=None, *, max_entries=2, max_bytes=1000):
    return BoundedInferenceGraphs(
        backend=backend or FakeBackend(), max_entries=max_entries, max_bytes=max_bytes
    )


def test_graph_replay_copies_inputs_and_nested_kwargs_without_aliasing_producers():
    graphs = cache()
    inputs = torch.arange(6.0).reshape(2, 3)
    bias = torch.ones(2, 3)

    def forward(value, *, biases, ring):
        return value + biases[0] + ring

    first = graphs.run(
        ("weights-a",), forward, (inputs,), {"biases": (bias,), "ring": 4}
    ).clone()
    inputs.add_(10)
    bias.mul_(2)
    second = graphs.run(
        ("weights-a",), forward, (inputs,), {"biases": (bias,), "ring": 4}
    ).clone()
    torch.testing.assert_close(second - first, torch.full_like(first, 11))
    assert graphs.captures == 1 and graphs.replays == 2
    assert graphs.warmup_calls == 3 and graphs.warmup_rows == 6
    assert graphs.validation_replays == 1 and graphs.validation_failures == 0


def test_graph_signature_separates_shape_dtype_stride_ring_and_identity():
    backend = FakeBackend()
    graphs = cache(backend, max_entries=16, max_bytes=10000)

    def forward(value, *, ring):
        return value + ring

    cases = (
        (("a",), torch.ones(2, 3), 4),
        (("b",), torch.ones(2, 3), 4),
        (("a",), torch.ones(3, 3), 4),
        (("a",), torch.ones(2, 3, dtype=torch.float64), 4),
        (("a",), torch.ones(3, 2).t(), 4),
        (("a",), torch.ones(2, 3), 6),
    )
    for namespace, value, ring in cases:
        torch.testing.assert_close(
            graphs.run(namespace, forward, (value,), {"ring": ring}), value + ring
        )
    assert graphs.captures == len(cases)


def test_graph_lru_and_memory_limits_close_evicted_executables():
    backend = FakeBackend(charge=100)
    graphs = cache(backend, max_entries=3, max_bytes=200)
    for name in ("a", "b", "a", "c"):
        graphs.run((name,), lambda value: value + 1, (torch.ones(2, 1),), {})
    assert graphs.snapshot()["graph_entries"] == 2
    assert graphs.retained_bytes == 200 and graphs.evictions == 1
    assert backend.entries[1].closed and not backend.entries[0].closed
    graphs.clear()
    assert all(entry.closed for entry in backend.entries)
    assert graphs.retained_bytes == graphs.snapshot()["graph_entries"] == 0


def test_oversized_graph_falls_back_once_and_negative_cache_is_bounded():
    backend = FakeBackend(charge=101)
    graphs = cache(backend, max_entries=1, max_bytes=100)
    for _ in range(3):
        torch.testing.assert_close(
            graphs.run(("a",), lambda x: x + 1, (torch.ones(1, 1),), {}),
            torch.full((1, 1), 2.0),
        )
    assert backend.attempts == 1 and graphs.fallbacks == 3
    assert backend.entries[0].closed and graphs.retained_bytes == 0
    for index in range(12):
        graphs.run((str(index),), lambda x: x, (torch.ones(1, 1),), {})
    assert graphs.snapshot()["graph_negative_entries"] == 8


def test_only_explicit_capture_unavailability_falls_back_and_model_errors_escape():
    backend = FakeBackend(error=CaptureUnavailable("unsupported", warmup_calls=2))
    graphs = cache(backend)
    for _ in range(2):
        assert graphs.run(("a",), lambda x: x + 2, (torch.ones(1, 1),), {}).item() == 3
    assert backend.attempts == 1 and graphs.warmup_calls == 2
    for error in (
        ValueError("invalid model output"),
        RuntimeError("illegal memory access"),
    ):
        graphs = cache(FakeBackend(error=error))
        with pytest.raises(type(error), match=str(error)):
            graphs.run(("a",), lambda x: x, (torch.ones(1, 1),), {})
        assert graphs.fallbacks == 0


def test_corrupt_capture_is_rejected_before_serving_and_falls_back_once(caplog):
    class CorruptBackend(FakeBackend):
        def capture(self, forward, args, kwargs):
            entry = super().capture(forward, args, kwargs)
            replay = entry.replay

            def corrupt(*args, **kwargs):
                return replay(*args, **kwargs).add_(0.25)

            entry.replay = corrupt
            return entry

    backend = CorruptBackend()
    graphs = cache(backend)
    for _ in range(2):
        output = graphs.run(("a",), lambda value: value + 2, (torch.ones(2, 1),), {})
        torch.testing.assert_close(output, torch.full((2, 1), 3.0), rtol=0, atol=0)
    assert backend.attempts == 1 and backend.entries[0].closed
    assert backend.entries[0].reference_output is None
    assert graphs.validation_failures == graphs.validation_replays == 1
    assert graphs.warmup_calls == 3 and graphs.warmup_rows == 6
    assert graphs.replays == 0 and graphs.retained_bytes == 0
    assert graphs.fallbacks == 2 and "bitwise output validation" in caplog.text


def test_validation_is_bitwise_including_signed_zero():
    assert not _bitwise_equal(torch.tensor([0.0]), torch.tensor([-0.0]))
    assert not _bitwise_equal(torch.ones(2), torch.ones(2, dtype=torch.float64))
    assert not _bitwise_equal(torch.ones(2), torch.ones(1, 2))
    assert _bitwise_equal(torch.arange(3.0), torch.arange(3.0))


def test_graph_memory_counts_own_pool_and_rounded_external_blocks_once():
    snapshot = [
        {"device": 0, "address": 1000, "total_size": 100, "segment_pool_id": (1, 2)},
        {"device": 0, "address": 2000, "total_size": 200, "segment_pool_id": (1, 2)},
        {
            "device": 0,
            "address": 3000,
            "total_size": 1024,
            "segment_pool_id": (0, 0),
            "blocks": [{"address": 3000, "size": 512}, {"address": 3512, "size": 256}],
        },
        {"device": 1, "address": 5000, "total_size": 10000, "segment_pool_id": (1, 2)},
    ]
    assert (
        _account_graph_storage(
            snapshot,
            pool=(1, 2),
            device_index=0,
            storages=((1000, 64), (3000, 32), (3000, 32), (3512, 13)),
        )
        == 1068
    )


@pytest.mark.parametrize(
    "snapshot", ([], [{"device": 0, "address": 1000, "total_size": 512}])
)
def test_graph_memory_requires_verifiable_allocator_ownership(snapshot):
    with pytest.raises(CaptureUnavailable, match="ownership|graph-pool"):
        _account_graph_storage(
            snapshot, pool=(1, 2), device_index=0, storages=((1000, 64),)
        )


def test_graph_keeps_parameter_storage_alive_until_synchronized_eviction():
    graphs = cache(max_entries=1)
    weight = torch.ones(8)
    reference = weakref.ref(weight)
    graphs.run(
        ("a",), lambda x: x + 1, (torch.ones(1, 1),), {}, lifetime_pins=(weight,)
    )
    del weight
    gc.collect()
    assert reference() is not None
    graphs.clear()
    gc.collect()
    assert reference() is None


def test_graph_execution_rejects_an_uncoordinated_second_owner():
    graphs = cache()
    graphs.run(("a",), lambda x: x, (torch.ones(1, 1),), {})
    with ThreadPoolExecutor(max_workers=1) as executor:
        with pytest.raises(RuntimeError, match="single inference owner"):
            executor.submit(
                graphs.run, ("a",), lambda x: x, (torch.ones(1, 1),), {}
            ).result()


def test_capture_stream_leases_reuse_released_handles_across_adapters():
    pool = _CaptureStreamPool(max_streams_per_device=3)
    created = []

    def factory():
        stream = SimpleNamespace(cuda_stream=len(created) % 3 + 1)
        created.append(stream)
        return stream

    first_backend = CudaBackend(torch.device("cuda:0"), stream_pool=pool)
    second_backend = CudaBackend(torch.device("cuda:0"), stream_pool=pool)
    hot = first_backend.stream_pool.acquire(0, factory)
    cold = second_backend.stream_pool.acquire(0, factory)
    assert hot.stream.cuda_stream != cold.stream.cuda_stream
    cold_handle = cold.stream.cuda_stream
    cold.release()
    for _ in range(96):
        reused = second_backend.stream_pool.acquire(0, factory)
        assert reused.stream.cuda_stream == cold_handle
        assert reused.stream.cuda_stream != hot.stream.cuda_stream
        reused.release()
    assert len(created) == 2
    hot.release()


def test_capture_stream_exhaustion_falls_back_without_aliasing_a_live_graph():
    pool = _CaptureStreamPool(max_streams_per_device=2)
    created = []

    def factory():
        stream = SimpleNamespace(cuda_stream=len(created) + 1)
        created.append(stream)
        return stream

    first = pool.acquire(0, factory)
    second = pool.acquire(0, factory)
    with pytest.raises(CaptureUnavailable, match="exclusively leased"):
        pool.acquire(0, factory)
    assert len(created) == 2
    first.release()
    replacement = pool.acquire(0, factory)
    first.release()  # A stale token cannot release the replacement lease.
    with pytest.raises(CaptureUnavailable, match="exclusively leased"):
        pool.acquire(0, factory)
    replacement.release()
    second.release()


def test_stream_pool_skips_round_robin_wrappers_for_already_leased_handles():
    pool = _CaptureStreamPool(max_streams_per_device=3)
    handles = iter((7, 7, 8))

    def factory():
        return SimpleNamespace(cuda_stream=next(handles))

    first = pool.acquire(0, factory)
    second = pool.acquire(0, factory)
    assert first.stream.cuda_stream == 7 and second.stream.cuda_stream == 8
    first.release()
    second.release()
    assert (
        CudaBackend(torch.device("cuda:0")).stream_pool
        is CudaBackend(torch.device("cuda:0")).stream_pool
    )


def test_cuda_executable_releases_stream_only_after_reset_and_destruction(monkeypatch):
    events = []
    pool = _CaptureStreamPool(max_streams_per_device=1)
    lease = pool.acquire(0, lambda: SimpleNamespace(cuda_stream=7))
    original_release = pool.release

    def release(selected):
        events.append("release")
        original_release(selected)

    monkeypatch.setattr(pool, "release", release)
    monkeypatch.setattr(
        torch.cuda, "synchronize", lambda device: events.append("synchronize")
    )

    class Graph:
        def reset(self):
            events.append("reset")

        def __del__(self):
            events.append("destroy")

    entry = _CudaExecutable(
        Graph(), (), {}, torch.empty(0), torch.device("cuda:0"), 0, 0, None, lease
    )
    entry.close()
    assert events == ["synchronize", "reset", "destroy", "release"]
    entry.close()
    assert events.count("release") == 1


def test_failed_graph_reset_does_not_release_its_live_stream(monkeypatch):
    pool = _CaptureStreamPool(max_streams_per_device=1)
    lease = pool.acquire(0, lambda: SimpleNamespace(cuda_stream=7))
    monkeypatch.setattr(torch.cuda, "synchronize", lambda device: None)

    class Graph:
        def reset(self):
            raise RuntimeError("poisoned CUDA context")

    entry = _CudaExecutable(
        Graph(), (), {}, torch.empty(0), torch.device("cuda:0"), 0, 0, None, lease
    )
    with pytest.raises(RuntimeError, match="poisoned"):
        entry.close()
    with pytest.raises(CaptureUnavailable, match="exclusively leased"):
        pool.acquire(0, lambda: SimpleNamespace(cuda_stream=7))


def test_owned_cuda_stream_uses_nonblocking_runtime_and_explicit_destruction(
    monkeypatch,
):
    import startrain.inference_graphs as module

    events = []
    runtime = SimpleNamespace(
        cudaStreamNonBlocking=1,
        cudaStreamCreateWithFlags=lambda flags: (
            events.append(("create", flags)) or (0, 7123)
        ),
        cudaStreamDestroy=lambda stream: events.append(("destroy", stream)) or (0,),
    )
    monkeypatch.setattr(module.importlib, "import_module", lambda name: runtime)
    monkeypatch.setattr(torch.cuda, "device", lambda device: nullcontext())
    monkeypatch.setattr(
        torch.cuda,
        "ExternalStream",
        lambda handle, device: SimpleNamespace(cuda_stream=handle, device=device),
    )
    monkeypatch.setattr(
        torch.cuda,
        "Stream",
        lambda **kwargs: pytest.fail(
            "owned capture must not use PyTorch pooled streams"
        ),
    )
    owned = _new_owned_capture_stream(3)
    assert owned.stream.cuda_stream == 7123 and owned.stream.device == 3
    owned.close()
    owned.close()
    assert events == [("create", 1), ("destroy", 7123)]


def test_missing_runtime_bindings_never_fall_back_to_pooled_streams(monkeypatch):
    import startrain.inference_graphs as module

    def missing(name):
        raise ImportError("no bindings")

    monkeypatch.setattr(module.importlib, "import_module", missing)
    monkeypatch.setattr(
        torch.cuda, "Stream", lambda **kwargs: pytest.fail("unsafe pooled fallback")
    )
    with pytest.raises(CaptureUnavailable, match="cuda.bindings.runtime"):
        _new_owned_capture_stream(0)


def test_idle_external_streams_are_bounded_and_destroyed_on_pool_cleanup():
    destroyed = []
    pool = _CaptureStreamPool(max_idle_per_device=2)

    def factory(index):
        return lambda: _OwnedCaptureStream(
            SimpleNamespace(cuda_stream=index), lambda: destroyed.append(index)
        )

    leases = [pool.acquire(0, factory(index)) for index in range(5)]
    for lease in leases:
        lease.release()
    assert len(destroyed) == 3 and len(pool._known) == 2
    pool.release_idle(0)
    assert sorted(destroyed) == list(range(5))
    assert not pool._known and not pool._leased


def test_external_stream_destruction_follows_graph_reset_and_destruction(monkeypatch):
    events = []
    pool = _CaptureStreamPool(max_idle_per_device=0)
    lease = pool.acquire(
        0,
        lambda: _OwnedCaptureStream(
            SimpleNamespace(cuda_stream=7), lambda: events.append("stream_destroy")
        ),
    )
    monkeypatch.setattr(
        torch.cuda, "synchronize", lambda device: events.append("synchronize")
    )

    class Graph:
        def reset(self):
            events.append("graph_reset")

        def __del__(self):
            events.append("graph_destroy")

    entry = _CudaExecutable(
        Graph(), (), {}, torch.empty(0), torch.device("cuda:0"), 0, 0, None, lease
    )
    entry.close()
    assert events == ["synchronize", "graph_reset", "graph_destroy", "stream_destroy"]


def test_failed_external_stream_destroy_quarantines_handle():
    pool = _CaptureStreamPool(max_streams_per_device=1, max_idle_per_device=0)

    def fail():
        raise RuntimeError("stream destroy failed")

    lease = pool.acquire(
        0, lambda: _OwnedCaptureStream(SimpleNamespace(cuda_stream=7), fail)
    )
    with pytest.raises(RuntimeError, match="destroy failed"):
        lease.release()
    with pytest.raises(CaptureUnavailable, match="exclusively leased"):
        pool.acquire(0, lambda: SimpleNamespace(cuda_stream=7))


def test_capture_exit_traceback_cannot_destroy_graph_after_stream_reuse(monkeypatch):
    import startrain.inference_graphs as module

    events = []
    pool = _CaptureStreamPool(max_idle_per_device=0)

    class Stream:
        cuda_stream = 7123

        def wait_stream(self, other):
            pass

        def synchronize(self):
            events.append("stream_sync")

    class Graph:
        def reset(self):
            events.append("graph_reset")

        def __del__(self):
            events.append("graph_destroy")

    class CaptureContext:
        def __init__(self, graph):
            self.graph = graph

        def __enter__(self):
            return self

        def __exit__(self, *unused):
            raise ValueError("capture end failed while retaining graph")

    monkeypatch.setattr(
        module,
        "_new_owned_capture_stream",
        lambda device: _OwnedCaptureStream(
            Stream(), lambda: events.append("stream_destroy")
        ),
    )
    monkeypatch.setattr(
        module, "_tensors", lambda value: []
    )  # CPU-only lifecycle test.
    monkeypatch.setattr(module, "_uncached_autocast", lambda device: nullcontext())
    monkeypatch.setattr(torch.cuda, "CUDAGraph", Graph)
    monkeypatch.setattr(torch.cuda, "stream", lambda stream: nullcontext())
    monkeypatch.setattr(torch.cuda, "current_stream", lambda device: None)
    monkeypatch.setattr(
        torch.cuda, "graph", lambda graph, stream: CaptureContext(graph)
    )
    backend = CudaBackend(torch.device("cuda:0"), warmup_calls=1, stream_pool=pool)
    with pytest.raises(ValueError, match="retaining graph") as captured:
        backend.capture(lambda value: value + 1, (torch.ones(1),), {})
    assert events == [
        "stream_sync",
        "graph_reset",
        "graph_destroy",
        "stream_sync",
        "stream_destroy",
    ]
    assert not pool._leased and not pool._known
    assert "__exit__" in [
        frame.name for frame in traceback.extract_tb(captured.value.__traceback__)
    ]


def test_validation_error_traceback_releases_wrapper_before_external_stream(
    monkeypatch,
):
    events = []
    pool = _CaptureStreamPool(max_idle_per_device=0)
    monkeypatch.setattr(torch.cuda, "synchronize", lambda device: events.append("sync"))

    class Graph:
        def replay(self):
            raise ValueError("validation replay model error")

        def reset(self):
            events.append("graph_reset")

        def __del__(self):
            events.append("graph_destroy")

    class Backend:
        def capture(self, forward, args, kwargs):
            lease = pool.acquire(
                0,
                lambda: _OwnedCaptureStream(
                    SimpleNamespace(cuda_stream=7),
                    lambda: events.append("stream_destroy"),
                ),
            )
            return _CudaExecutable(
                Graph(),
                (torch.ones(1),),
                {},
                torch.empty(1),
                torch.device("cuda:0"),
                100,
                0,
                torch.ones(1),
                lease,
            )

    graphs = cache(Backend())
    with pytest.raises(ValueError, match="validation replay model error") as captured:
        graphs.run(("a",), lambda value: value, (torch.ones(1),), {})
    assert events == ["sync", "graph_reset", "graph_destroy", "stream_destroy"]
    assert not pool._leased and not pool._known
    assert "replay" in [
        frame.name for frame in traceback.extract_tb(captured.value.__traceback__)
    ]


@pytest.fixture
def feature_requests(monkeypatch):
    monkeypatch.setattr(
        "startrain.inference.encode_native_feature_data",
        lambda data, **kwargs: data.encoded,
    )
    return encoded_requests


def test_adapter_prediction_bytes_survive_graph_replay_padding_and_weight_refresh(
    feature_requests, monkeypatch
):
    network = ObservedNetwork()
    adapter = GraphInferenceAdapter(
        network, model_identity="a", config=InferenceConfig(deduplicate=True)
    )
    backend = FakeBackend()
    adapter._graphs = cache(backend)
    monkeypatch.setattr(
        adapter, "_inference_batch_rows", lambda rows: 1 << (rows - 1).bit_length()
    )
    request = feature_requests(
        encode_batch([position(pda=value) for value in (0, 1, 2)])
    )
    first = adapter.evaluate_detailed(request)
    first_logits = list(first.response.policy_logits)
    second_request = feature_requests(
        encode_batch([position(pda=value) for value in (0, -1, -2)]), token_start=20
    )
    second = adapter.evaluate_detailed(second_request)
    assert second.response.tokens == [20, 21, 22]
    assert first.response.policy_logits == first_logits
    assert adapter.metrics_snapshot().graph_captures == 1
    assert adapter.metrics_snapshot().neural_padding_rows == 2
    with torch.no_grad():
        network.bias.add_(1)
    adapter.model_identity = "b"
    refreshed = adapter.evaluate_detailed(request)
    assert refreshed.response.policy_logits != first_logits
    assert adapter.metrics_snapshot().graph_captures == 2
    assert backend.entries[0].closed


def test_cpu_graph_flag_has_no_cuda_allocation_and_config_is_strict(
    feature_requests, monkeypatch
):
    monkeypatch.setattr(
        torch.cuda, "CUDAGraph", lambda: pytest.fail("CPU cannot capture CUDA")
    )
    adapter = GraphInferenceAdapter(
        ObservedNetwork(), model_identity="a", config=InferenceConfig(cuda_graphs=True)
    )
    assert adapter.evaluate(feature_requests(encode_batch([position()]))).tokens == [1]
    assert adapter.metrics_snapshot().graph_captures == 0
    for values in (
        {"cuda_graphs": 1},
        {"cuda_graph_max_entries": 0},
        {"cuda_graph_max_entries": 65},
        {"cuda_graph_max_bytes": 0},
    ):
        with pytest.raises(ValueError):
            InferenceConfig(**values)
    with pytest.raises(CaptureUnavailable, match="capture device"):
        CudaBackend(torch.device("cuda:0")).capture(
            lambda value: value, (torch.ones(1),), {}
        )


def test_broker_separates_physical_work_from_dispatches_and_counts_evicted_adapters(
    feature_requests,
):
    config = InferenceConfig(
        cache_max_entries=8, cache_max_bytes=2_000_000, deduplicate=True
    )
    first = GraphInferenceAdapter(ObservedNetwork(), model_identity="a", config=config)
    second = GraphInferenceAdapter(ObservedNetwork(), model_identity="b", config=config)
    request = feature_requests(encode_batch([position(), position()]))
    with BoundedInferenceBroker(max_batch_rows=2) as broker:
        for adapter in (first, first, second):
            assert broker.submit(adapter, request).result(timeout=3).tokens == [1, 2]
        snapshot = broker.metrics_snapshot()
    physical = snapshot["physical_inference"]
    assert snapshot["neural_batches"] == 3
    assert physical["neural_calls"] == 2 and physical["neural_rows"] == 2
    assert physical["cache_hits"] == 2 and physical["cache_misses"] == 4
    assert physical["deduplicated_rows"] == 2
    assert physical["total_neural_calls"] == 2
    assert physical["prepare_seconds"] > 0 and physical["key_seconds"] > 0


def test_broker_device_guard_serializes_capture_with_registry_work(feature_requests):
    adapter = GraphInferenceAdapter(
        ObservedNetwork(), model_identity="a", config=InferenceConfig(deduplicate=True)
    )
    backend = FakeBackend()
    adapter._graphs = cache(backend)
    request = feature_requests(encode_batch([position()]))
    with BoundedInferenceBroker(max_batch_rows=1) as broker:
        with broker.device_lock:
            future = broker.submit(adapter, request)
            assert not future.done() and backend.attempts == 0
        assert future.result(timeout=3).tokens == [1]
        physical = broker.metrics_snapshot()["physical_inference"]
    assert backend.threads == ["star-inference-owner"]
    assert physical["neural_calls"] == 1
    assert physical["graph_warmup_calls"] == 3
    assert physical["graph_validation_replays"] == 1
    assert physical["graph_validation_failures"] == 0
    assert physical["total_neural_calls"] == 4


@pytest.mark.cuda
def test_cuda_graph_reuse_keeps_inflight_inputs_and_outputs_isolated():
    pytest.importorskip("cuda.bindings.runtime")
    device = torch.device("cuda:0")
    graphs = BoundedInferenceGraphs(
        backend=CudaBackend(device), max_entries=2, max_bytes=128 * 1024**2
    )
    try:
        with torch.inference_mode():
            value = torch.arange(128, device=device).reshape(16, 8).float()
            first = graphs.run(("a",), lambda x: x.square() + 2, (value,), {}).clone()
            value.add_(3)
            second = graphs.run(("a",), lambda x: x.square() + 2, (value,), {}).clone()
            torch.testing.assert_close(
                first.cpu(), torch.arange(128).reshape(16, 8).float().square() + 2
            )
            torch.testing.assert_close(second.cpu(), value.cpu().square() + 2)
            assert (
                graphs.captures == 1
                and graphs.replays == 2
                and graphs.retained_bytes <= graphs.max_bytes
            )
            assert graphs.validation_replays == 1 and graphs.validation_failures == 0
    finally:
        graphs.clear()


@pytest.mark.cuda
def test_cuda_graph_stream_lifetimes_survive_ninety_six_cross_adapter_evictions():
    pytest.importorskip("cuda.bindings.runtime")
    device = torch.device("cuda:0")
    # Independent adapters must share the process-wide stream lease authority.
    hot = BoundedInferenceGraphs(
        backend=CudaBackend(device), max_entries=1, max_bytes=256 * 1024**2
    )
    cold = BoundedInferenceGraphs(
        backend=CudaBackend(device), max_entries=1, max_bytes=256 * 1024**2
    )
    try:
        with torch.inference_mode():
            weight = torch.randn(640, 640, device=device, dtype=torch.bfloat16) * 0.01
            hot_input = torch.randn(128, 640, device=device, dtype=torch.bfloat16)
            cold_input = torch.randn_like(hot_input)

            def forward(value):
                return value @ weight

            expected = forward(hot_input).clone()
            hot.run(("hot",), forward, (hot_input,), {}, lifetime_pins=(weight,))
            hot_entry = next(iter(hot._entries.values()))
            for index in range(96):
                cold_input.add_(0.0001)
                result = cold.run(
                    ("cold", index), forward, (cold_input,), {}, lifetime_pins=(weight,)
                )
                torch.testing.assert_close(result, forward(cold_input), rtol=0, atol=0)
                cold_entry = next(iter(cold._entries.values()))
                assert (
                    hot_entry.stream_lease.stream.cuda_stream
                    != cold_entry.stream_lease.stream.cuda_stream
                )
                # Force freed allocations to become unavailable; a surviving
                # graph must not refer to a workspace retired by its neighbor.
                if index % 8 == 0:
                    torch.cuda.empty_cache()
                torch.testing.assert_close(
                    hot.run(("hot",), forward, (hot_input,), {}),
                    expected,
                    rtol=0,
                    atol=0,
                )
            assert hot.captures == 1 and cold.captures == 96
            assert cold.evictions == 95
            assert hot.validation_failures == cold.validation_failures == 0
    finally:
        cold.clear()
        hot.clear()


@pytest.mark.cuda
@pytest.mark.native
def test_real_network_cuda_graphs_preserve_modes_boards_padding_and_cached_predictions():
    pytest.importorskip("cuda.bindings.runtime")
    from startrain.model import GraphResTNet, ModelConfig

    native = pytest.importorskip("star_native")
    torch.manual_seed(512)
    network = GraphResTNet(
        ModelConfig(width=16, rrt_groups=1, attention_heads=4, kv_heads=1)
    ).eval()
    baseline = GraphInferenceAdapter(
        network,
        device="cuda:0",
        model_identity="same",
        homogeneous_relational_bias=True,
        config=InferenceConfig(precision="bf16", deduplicate=True),
    )
    optimized = GraphInferenceAdapter(
        network,
        device="cuda:0",
        model_identity="same",
        homogeneous_relational_bias=True,
        config=InferenceConfig(
            precision="bf16",
            deduplicate=True,
            cuda_graphs=True,
            cache_max_entries=128,
            cache_max_bytes=16 * 1024**2,
            cuda_graph_max_entries=4,
            cuda_graph_max_bytes=512 * 1024**2,
        ),
    )
    try:
        for ring in (4, 6, 8, 10):
            for mode, handicap, pie in (
                ("classic", 1, False),
                ("double", 1, False),
                ("classic", 1, True),
                ("double", 1, True),
                ("classic", 9, False),
                ("double", 9, False),
            ):
                states = native.StateBatch(
                    ring, 3, mode=mode, handicap=handicap, pie=pie
                )
                request = native.SearchBatch(
                    states,
                    simulations=1,
                    max_considered=2,
                    pda_by_seat=[(0, 0), (1, -1), (2, -2)],
                ).root_requests()
                expected = baseline.evaluate_detailed(request)
                actual = optimized.evaluate_detailed(request)
                assert actual.response.tokens == expected.response.tokens
                assert (
                    actual.response.policy_offsets == expected.response.policy_offsets
                )
                torch.testing.assert_close(
                    torch.tensor(actual.response.policy_logits),
                    torch.tensor(expected.response.policy_logits),
                    rtol=0,
                    atol=0,
                )
                torch.testing.assert_close(
                    torch.tensor(actual.score_probabilities),
                    torch.tensor(expected.score_probabilities),
                    rtol=0,
                    atol=0,
                )
                assert actual.response.values == expected.response.values
        metrics = optimized.metrics_snapshot()
        assert metrics.graph_captures == 4 and metrics.graph_replays == 24
        assert (
            metrics.graph_validation_replays == 4
            and metrics.graph_validation_failures == 0
        )
        assert metrics.graph_fallbacks == 0 and metrics.neural_padding_rows == 24
        assert optimized.evaluate_detailed(request) == actual
        assert optimized.metrics_snapshot().graph_replays == 24
        assert optimized.metrics_snapshot().cache_hits == 3
        preserved = list(actual.response.policy_logits)
        with torch.no_grad():
            network.node_policy.bias.add_(0.125)
        optimized.model_identity = "updated-weights"
        expected = baseline.evaluate_detailed(request)
        refreshed = optimized.evaluate_detailed(request)
        assert refreshed == expected
        assert actual.response.policy_logits == preserved
        assert refreshed.response.policy_logits != preserved
        assert optimized.metrics_snapshot().graph_captures == 5
    finally:
        optimized.close()
        baseline.close()
