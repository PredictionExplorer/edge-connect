from concurrent.futures import ThreadPoolExecutor
import threading

import pytest

from startrain.features import encode_batch
from startrain.inference import DetailedInferenceResponse, GraphInferenceAdapter
from startrain.inference_batching import BoundedInferenceBroker
from test_inference_efficiency import (
    ObservedNetwork,
    cached_adapter,
    encoded_requests,
    position,
)


@pytest.fixture
def feature_requests(monkeypatch):
    monkeypatch.setattr(
        "startrain.inference.encode_native_feature_data", lambda data, **_: data.encoded
    )
    return encoded_requests


def test_broker_combines_compatible_cohorts_and_preserves_local_utility(
    feature_requests,
):
    base = cached_adapter()
    batch = encode_batch([position(), position(pda=2)])
    with BoundedInferenceBroker(max_batch_rows=4, max_wait_seconds=0.5) as broker:
        first = broker.submit(
            base, feature_requests(batch), include_details=True, score_utility_weight=0
        )
        second = broker.submit(
            base,
            feature_requests(batch, token_start=20),
            include_details=True,
            score_utility_weight=0.5,
        )
        left, right = first.result(timeout=3), second.result(timeout=3)
        assert isinstance(left, DetailedInferenceResponse) and isinstance(
            right, DetailedInferenceResponse
        )
        assert left.response.tokens == [1, 2] and right.response.tokens == [20, 21]
        assert left.response.policy_logits == right.response.policy_logits
        assert right.response.values == pytest.approx(
            [
                value + 0.5 * margin / 151
                for value, margin in zip(
                    left.outcome_values, left.score_expectations, strict=True
                )
            ],
            abs=1e-7,
        )
        assert base.model.rows == [2]
        assert base.model.threads == ["star-inference-owner"]
    metrics = broker.metrics_snapshot()
    assert metrics["neural_batches"] == 1 and metrics["batched_requests"] == 2


def test_complete_cache_keys_are_built_on_the_cpu_producer(
    feature_requests, monkeypatch
):
    base = cached_adapter()
    key_threads = []
    original = base._row_keys

    def record_keys(encoded, namespace):
        key_threads.append(threading.current_thread().name)
        return original(encoded, namespace)

    monkeypatch.setattr(base, "_row_keys", record_keys)
    request = feature_requests(encode_batch([position()]))
    with BoundedInferenceBroker(max_batch_rows=1) as broker:
        with ThreadPoolExecutor(
            max_workers=1, thread_name_prefix="cpu-cohort"
        ) as producer:
            response_future = producer.submit(broker.submit, base, request).result(
                timeout=2
            )
            assert response_future.result(timeout=2).tokens == [1]
    assert key_threads == ["cpu-cohort_0"]
    assert base.model.threads == ["star-inference-owner"]


def test_broker_never_merges_different_rings_or_adapters(feature_requests):
    first = cached_adapter()
    second = cached_adapter()
    with BoundedInferenceBroker(max_batch_rows=4, max_wait_seconds=0.05) as broker:
        jobs = [
            broker.submit(first, feature_requests(encode_batch([position(4)]))),
            broker.submit(first, feature_requests(encode_batch([position(6)]))),
            broker.submit(second, feature_requests(encode_batch([position(4)]))),
        ]
        assert all(job.result(timeout=3).tokens == [1] for job in jobs)
    assert broker.metrics_snapshot()["neural_batches"] == 3
    assert first.model.rows == [1, 1] and second.model.rows == [1]


def test_matching_requests_after_an_incompatible_ring_use_job_identity(
    feature_requests,
):
    base = cached_adapter()
    with BoundedInferenceBroker(max_batch_rows=2, max_wait_seconds=0.5) as broker:
        futures = [
            broker.submit(
                base, feature_requests(encode_batch([position(4)]), token_start=10)
            ),
            broker.submit(
                base, feature_requests(encode_batch([position(6)]), token_start=20)
            ),
            broker.submit(
                base,
                feature_requests(encode_batch([position(4, pda=2)]), token_start=30),
            ),
        ]
        assert [future.result(timeout=3).tokens for future in futures] == [
            [10],
            [20],
            [30],
        ]
    assert base.model.rows == [2, 1]
    assert broker.metrics_snapshot()["neural_batches"] == 2
    assert broker.metrics_snapshot()["worker_failures"] == 0


def test_scheduler_failure_fails_owned_and_queued_requests_and_closes_broker(
    feature_requests,
):
    class BrokenScheduler(BoundedInferenceBroker):
        def _next_batch(self):
            with self._condition:
                while len(self._queue) < 2 and not self._closed:
                    self._condition.wait()
                self._owned_jobs = [self._queue.popleft()]
                raise RuntimeError("injected scheduler failure")

    base = cached_adapter()
    broker = BrokenScheduler(max_pending_requests=2)
    request = feature_requests(encode_batch([position()]))
    first = broker.submit(base, request)
    second = broker.submit(base, request)
    try:
        for future in (first, second):
            with pytest.raises(RuntimeError, match="scheduler failure"):
                future.result(timeout=2)
        with pytest.raises(RuntimeError, match="closed"):
            broker.submit(base, request, timeout=0)
    finally:
        broker.shutdown()
    assert broker.metrics_snapshot()["worker_failures"] == 1
    assert broker.metrics_snapshot()["failed_requests"] == 2
    # Every admitted job relinquished its semaphore slot on the failure path.
    assert broker._slots.acquire(timeout=0)
    assert broker._slots.acquire(timeout=0)
    assert not broker._slots.acquire(timeout=0)


def test_broker_backpressure_cancellation_and_shutdown_release_waiters(
    feature_requests,
):
    base = GraphInferenceAdapter(ObservedNetwork(), model_identity="a")
    base.model.started = threading.Event()
    base.model.release = threading.Event()
    broker = BoundedInferenceBroker(
        max_batch_rows=2, max_pending_requests=2, max_wait_seconds=0
    )
    request = feature_requests(encode_batch([position()]))
    first = broker.submit(base, request)
    try:
        assert base.model.started.wait(2)
        second = broker.submit(base, request)
        with pytest.raises(TimeoutError, match="backpressure"):
            broker.submit(base, request, timeout=0)
        broker.shutdown(wait=False, cancel_pending=True)
        assert second.cancelled()
        with pytest.raises(RuntimeError, match="closed"):
            broker.submit(base, request)
    finally:
        base.model.release.set()
        broker.shutdown()
    assert first.result(timeout=1).tokens == [1]
    assert broker.metrics_snapshot()["cancelled_requests"] == 1


def test_broker_idle_proof_includes_active_owned_inference(feature_requests):
    base = GraphInferenceAdapter(ObservedNetwork(), model_identity="a")
    base.model.started = threading.Event()
    base.model.release = threading.Event()
    broker = BoundedInferenceBroker(max_wait_seconds=0)
    assert broker.is_idle()
    request = feature_requests(encode_batch([position()]))
    future = broker.submit(base, request)
    try:
        assert base.model.started.wait(2)
        assert broker.metrics_snapshot()["pending_requests"] == 0
        assert broker.metrics_snapshot()["active_requests"] == 1
        assert not broker.is_idle()
        # Closing admission does not make an in-flight inference safe to pause.
        broker.shutdown(wait=False)
        assert not broker.is_idle()
    finally:
        base.model.release.set()
        broker.shutdown()
    assert future.result(timeout=1).tokens == [1]
    assert broker.is_idle()
    assert broker.metrics_snapshot()["active_requests"] == 0


def test_shutdown_cancels_a_request_waiting_for_batch_partners(feature_requests):
    base = cached_adapter()
    broker = BoundedInferenceBroker(max_batch_rows=4, max_wait_seconds=0.5)
    future = broker.submit(base, feature_requests(encode_batch([position()])))
    broker.shutdown(cancel_pending=True)
    assert future.cancelled()
    assert base.model.rows == []


def test_neural_exception_reaches_callers_and_next_request_can_run(feature_requests):
    base = cached_adapter()
    base.model.fail = True
    request = feature_requests(encode_batch([position()]))
    with BoundedInferenceBroker(max_wait_seconds=0) as broker:
        with pytest.raises(ValueError, match="neural failure"):
            broker.submit(base, request).result(timeout=2)
        base.model.fail = False
        assert broker.submit(base, request).result(timeout=2).tokens == [1]
    assert broker.metrics_snapshot()["failed_requests"] == 1


def test_cohort_facades_pin_identity_and_do_not_mutate_other_cohorts(feature_requests):
    base = cached_adapter()
    with BoundedInferenceBroker(max_wait_seconds=0.05) as broker:
        first = broker.cohort_adapter(base, score_utility_weight=0)
        second = broker.cohort_adapter(base, score_utility_weight=0.5)
        request = feature_requests(encode_batch([position()]))
        with ThreadPoolExecutor(max_workers=2) as producers:
            left = producers.submit(first.evaluate, request)
            right = producers.submit(second.evaluate, request)
            assert left.result(timeout=3).values != right.result(timeout=3).values
        assert base.config.score_utility_weight == 0
        assert first.evaluator_calls == second.evaluator_calls == 1
        assert first.evaluator_rows == second.evaluator_rows == 1
        base.model_identity = "new-weights"
        with pytest.raises(RuntimeError, match="cohort model identity"):
            first.evaluate(request)


def test_queued_identity_change_is_rejected_before_model_execution(feature_requests):
    base = cached_adapter()
    with BoundedInferenceBroker(max_wait_seconds=0.2) as broker:
        future = broker.submit(base, feature_requests(encode_batch([position()])))
        base.model_identity = "new"
        with pytest.raises(RuntimeError, match="identity changed"):
            future.result(timeout=2)
    assert base.model.rows == []


def test_broker_rejects_oversized_and_mixed_ring_submissions(feature_requests):
    with BoundedInferenceBroker(max_batch_rows=2) as broker:
        base = cached_adapter()
        with pytest.raises(ValueError, match="max_batch_rows"):
            broker.submit(base, feature_requests(encode_batch([position()] * 3)))
        with pytest.raises(ValueError, match="homogeneous"):
            broker.submit(
                base, feature_requests(encode_batch([position(4), position(6)]))
            )
