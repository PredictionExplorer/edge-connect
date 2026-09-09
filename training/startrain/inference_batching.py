"""Bounded CPU-producer / single-owner neural inference batching.

Independent game cohorts submit owned host snapshots. Only the worker invokes
models or their caches; identities and board sizes never mix in a neural batch.
"""

from __future__ import annotations

from collections import deque
from concurrent.futures import Future, InvalidStateError
from dataclasses import asdict, dataclass, replace
import threading
import time
import weakref

from .inference import (
    DetailedInferenceResponse,
    GraphInferenceAdapter,
    InferenceMetrics,
    InferenceResponse,
    NativeEvalBatchProtocol,
    PreparedInferenceRequest,
)

InferenceResult = InferenceResponse | DetailedInferenceResponse


@dataclass(slots=True, eq=False)
class _Job:
    adapter: GraphInferenceAdapter
    prepared: PreparedInferenceRequest
    future: Future[InferenceResult]
    detailed: bool
    submitted: float

    @property
    def key(self) -> tuple[object, ...]:
        # Object identity prevents accidentally merging different model objects
        # that a caller labelled identically. An immutable registry can reuse
        # one adapter across cohorts to obtain batching/cache sharing.
        return (id(self.adapter), self.prepared.namespace, self.prepared.ring)


class BoundedInferenceBroker:
    """One neural worker, bounded outstanding jobs, and bounded batch wait.

    A registry owns adapter lifetimes and must keep their weights pinned until
    all associated futures finish. Closing this broker never closes adapters.
    """

    def __init__(
        self,
        *,
        max_batch_rows: int = 256,
        max_pending_requests: int = 16,
        max_wait_seconds: float = 0.002,
    ) -> None:
        for name, value in (
            ("max_batch_rows", max_batch_rows),
            ("max_pending_requests", max_pending_requests),
        ):
            if type(value) is not int or value <= 0:
                raise ValueError(f"{name} must be a positive integer")
        if isinstance(max_wait_seconds, bool) or not 0 <= max_wait_seconds <= 1:
            raise ValueError("max_wait_seconds must be in [0, 1]")
        self.max_batch_rows = max_batch_rows
        self.max_pending_requests = max_pending_requests
        self.max_wait_seconds = float(max_wait_seconds)
        self._slots = threading.BoundedSemaphore(max_pending_requests)
        self._condition = threading.Condition()
        # Registry model loads/evictions share this lock with the owner so no
        # other thread allocates CUDA tensors while a graph is being captured.
        self.device_lock = threading.RLock()
        self._queue: deque[_Job] = deque()
        self._owned_jobs: list[_Job] = []
        self._worker_failures = 0
        self._closed = False
        self._cancel_pending_on_close = False
        self._submitted = 0
        self._completed = 0
        self._failed = 0
        self._cancelled = 0
        self._batches = 0
        self._batched_requests = 0
        self._rows = 0
        self._queue_wait_seconds = 0.0
        self._worker_seconds = 0.0
        self._physical = {
            name: value
            for name, value in asdict(InferenceMetrics()).items()
            if name not in ("evaluator_calls", "evaluator_rows")
        }
        self._prepare_seconds = 0.0
        self._key_seconds = 0.0
        self._adapters: weakref.WeakSet[GraphInferenceAdapter] = weakref.WeakSet()
        self._thread = threading.Thread(
            target=self._run, name="star-inference-owner", daemon=True
        )
        self._thread.start()

    def submit(
        self,
        evaluator: GraphInferenceAdapter,
        requests: NativeEvalBatchProtocol,
        *,
        score_utility_weight: float | None = None,
        include_details: bool = False,
        timeout: float | None = None,
    ) -> Future[InferenceResult]:
        """Backpressure before host allocation; timeout covers queue admission."""

        with self._condition:
            if self._closed:
                raise RuntimeError("inference broker is closed")
        if len(requests) > self.max_batch_rows:
            raise ValueError("request exceeds inference max_batch_rows")
        if timeout is not None and (isinstance(timeout, bool) or timeout < 0):
            raise ValueError("submission timeout must be non-negative")
        if not self._slots.acquire(timeout=timeout):
            raise TimeoutError("inference queue backpressure timeout")
        try:
            with self._condition:
                if self._closed:
                    raise RuntimeError("inference broker is closed")
            future: Future[InferenceResult] = Future()
            if len(requests) == 0:
                # Empty responses are entirely CPU-side and contain no weight-
                # dependent values. Preserve the native protocol's validation.
                result = (
                    evaluator.evaluate_detailed(requests)
                    if include_details
                    else evaluator.evaluate(requests)
                )
                future.set_result(result)
                self._slots.release()
                return future
            prepared = evaluator.prepare_requests(
                requests, score_utility_weight=score_utility_weight
            )
            if prepared.ring is None:
                raise ValueError("queued requests must be homogeneous in ring")
            job = _Job(evaluator, prepared, future, include_details, time.monotonic())
            with self._condition:
                if self._closed:
                    raise RuntimeError(
                        "inference broker closed during request preparation"
                    )
                self._queue.append(job)
                self._adapters.add(evaluator)
                self._submitted += 1
                self._prepare_seconds += prepared.prepare_seconds
                self._key_seconds += prepared.key_seconds
                self._condition.notify_all()
            return future
        except BaseException:
            self._slots.release()
            raise

    def cohort_adapter(
        self,
        base_adapter: GraphInferenceAdapter,
        score_utility_weight: float | None = None,
    ) -> "CohortInferenceAdapter":
        return CohortInferenceAdapter(
            self, base_adapter, score_utility_weight=score_utility_weight
        )

    def _next_batch(self) -> list[_Job] | None:
        with self._condition:
            while not self._queue and not self._closed:
                self._condition.wait()
            if not self._queue:
                return None
            first = self._queue.popleft()
            jobs = [first]
            self._owned_jobs = jobs
            rows = first.prepared.rows
            deadline = first.submitted + self.max_wait_seconds
            while rows < self.max_batch_rows:
                # Preserve order of incompatible jobs. Scan only the bounded
                # existing queue, and never wait beyond the oldest job's limit.
                for job in tuple(self._queue):
                    if (
                        job.key == first.key
                        and rows + job.prepared.rows <= self.max_batch_rows
                    ):
                        self._queue.remove(job)
                        jobs.append(job)
                        rows += job.prepared.rows
                remaining = deadline - time.monotonic()
                if rows >= self.max_batch_rows or self._closed or remaining <= 0:
                    break
                self._condition.wait(remaining)
            return jobs

    def _run(self) -> None:
        try:
            self._run_batches()
        except BaseException as exc:
            # Fail scheduling/queue errors as well as model errors. A dead
            # worker must never leave accepted futures or blocked producers
            # waiting forever, including a batch already removed from queue.
            with self._condition:
                self._closed = True
                self._worker_failures += 1
                abandoned = [*self._owned_jobs, *self._queue]
                self._owned_jobs = []
                self._queue.clear()
                self._condition.notify_all()
            for job in abandoned:
                try:
                    if not job.future.done():
                        try:
                            job.future.set_exception(exc)
                        except InvalidStateError:
                            # A producer may cancel between the done check and
                            # failure publication. Still release every job.
                            pass
                        else:
                            with self._condition:
                                self._failed += 1
                    if job.future.cancelled():
                        with self._condition:
                            self._cancelled += 1
                finally:
                    self._slots.release()

    def _release_owned_job(self, job: _Job) -> None:
        with self._condition:
            self._owned_jobs.remove(job)
            self._condition.notify_all()
        self._slots.release()

    def is_idle(self) -> bool:
        """Prove no queued or active inference/compilation remains.

        Callers must first stop producers from submitting new work. An empty
        queue alone is insufficient because a worker may own an active batch.
        """
        with self._condition:
            return not self._queue and not self._owned_jobs

    def _run_batches(self) -> None:
        while True:
            jobs = self._next_batch()
            if jobs is None:
                return
            active: list[_Job] = []
            for job in tuple(jobs):
                with self._condition:
                    if self._cancel_pending_on_close:
                        job.future.cancel()
                    running = job.future.set_running_or_notify_cancel()
                if running:
                    active.append(job)
                else:
                    with self._condition:
                        self._cancelled += 1
                    self._release_owned_job(job)
            if not active:
                continue
            started = time.monotonic()
            try:
                adapter = active[0].adapter
                with self.device_lock:
                    before = adapter.metrics_snapshot()
                    try:
                        results = adapter.evaluate_prepared(
                            [job.prepared for job in active],
                            include_details=[job.detailed for job in active],
                        )
                    finally:
                        delta = asdict(adapter.metrics_snapshot().delta(before))
                        with self._condition:
                            for name in self._physical:
                                self._physical[name] += delta[name]
                if len(results) != len(active):
                    raise RuntimeError("inference result count does not match requests")
                for job, (response, details) in zip(active, results, strict=True):
                    result = details if job.detailed else response
                    if result is None:
                        raise RuntimeError("missing detailed inference result")
                    job.future.set_result(result)
                with self._condition:
                    self._completed += len(active)
            except BaseException as exc:
                for job in active:
                    if not job.future.done():
                        job.future.set_exception(exc)
                with self._condition:
                    self._failed += len(active)
            finally:
                with self._condition:
                    self._batches += 1
                    self._batched_requests += len(active)
                    self._rows += sum(job.prepared.rows for job in active)
                    self._queue_wait_seconds += sum(
                        started - job.submitted for job in active
                    )
                    self._worker_seconds += time.monotonic() - started
                for job in active:
                    self._release_owned_job(job)

    def metrics_snapshot(self) -> dict[str, object]:
        with self._condition:
            gauges = {
                name: 0
                for name in (
                    "graph_entries",
                    "graph_retained_bytes",
                    "graph_negative_entries",
                )
            }
            for adapter in tuple(self._adapters):
                current = adapter.efficiency_snapshot()
                for name in gauges:
                    gauges[name] += int(current.get(name, 0))
            return {
                "submitted_requests": self._submitted,
                "completed_requests": self._completed,
                "failed_requests": self._failed,
                "cancelled_requests": self._cancelled,
                "neural_batches": self._batches,
                "batched_requests": self._batched_requests,
                "requested_rows": self._rows,
                "pending_requests": len(self._queue),
                "active_requests": len(self._owned_jobs),
                "queue_wait_seconds": self._queue_wait_seconds,
                "worker_seconds": self._worker_seconds,
                "worker_failures": self._worker_failures,
                "physical_inference": {
                    **self._physical,
                    **gauges,
                    "prepare_seconds": self._prepare_seconds,
                    "key_seconds": self._key_seconds,
                    "total_neural_calls": self._physical["neural_calls"]
                    + self._physical["graph_warmup_calls"],
                    "total_neural_rows": self._physical["neural_rows"]
                    + self._physical["graph_warmup_rows"],
                },
            }

    def shutdown(self, wait: bool = True, *, cancel_pending: bool = False) -> None:
        with self._condition:
            self._closed = True
            if cancel_pending:
                self._cancel_pending_on_close = True
                while self._queue:
                    job = self._queue.popleft()
                    job.future.cancel()
                    self._cancelled += 1
                    self._slots.release()
            self._condition.notify_all()
        if wait:
            if threading.current_thread() is self._thread:
                raise RuntimeError("inference owner cannot join itself")
            self._thread.join()

    def __enter__(self) -> "BoundedInferenceBroker":
        return self

    def __exit__(self, *_: object) -> None:
        self.shutdown()


class CohortInferenceAdapter:
    """Synchronous search facade with immutable weights and local utility."""

    def __init__(
        self,
        broker: BoundedInferenceBroker,
        base: GraphInferenceAdapter,
        *,
        score_utility_weight: float | None = None,
    ) -> None:
        self.broker = broker
        self.base = base
        self._namespace = base.namespace
        self.config = base.config
        self.model_version = base.model_version
        self.model_step = base.model_step
        self.model_identity = base.model_identity
        self.device = base.device
        self._calls = 0
        self._rows = 0
        self._lock = threading.Lock()
        if score_utility_weight is not None:
            self.set_score_utility_weight(score_utility_weight)

    @property
    def evaluator_calls(self) -> int:
        return self._calls

    @property
    def evaluator_rows(self) -> int:
        return self._rows

    @property
    def last_feature_path(self) -> str | None:
        return self.base.last_feature_path

    @property
    def feature_path_counts(self) -> dict[str, int]:
        return self.base.feature_path_counts

    def metrics_snapshot(self) -> InferenceMetrics:
        return InferenceMetrics(evaluator_calls=self._calls, evaluator_rows=self._rows)

    def efficiency_snapshot(self) -> dict[str, int | float]:
        return self.base.efficiency_snapshot()

    def set_score_utility_weight(self, weight: float) -> None:
        with self._lock:
            self.config = replace(self.config, score_utility_weight=weight)

    def _evaluate(
        self, requests: NativeEvalBatchProtocol, *, detailed: bool
    ) -> InferenceResult:
        if self.base.namespace != self._namespace:
            raise RuntimeError(
                "cohort model identity changed before its games finished"
            )
        with self._lock:
            weight = self.config.score_utility_weight
        result = self.broker.submit(
            self.base,
            requests,
            score_utility_weight=weight,
            include_details=detailed,
        ).result()
        with self._lock:
            self._calls += 1
            self._rows += len(requests)
        return result

    def evaluate(self, requests: NativeEvalBatchProtocol) -> InferenceResponse:
        result = self._evaluate(requests, detailed=False)
        if not isinstance(result, InferenceResponse):
            raise RuntimeError("unexpected detailed inference response")
        return result

    def evaluate_detailed(
        self, requests: NativeEvalBatchProtocol
    ) -> DetailedInferenceResponse:
        result = self._evaluate(requests, detailed=True)
        if not isinstance(result, DetailedInferenceResponse):
            raise RuntimeError("missing detailed inference response")
        return result
