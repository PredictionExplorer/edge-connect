from __future__ import annotations

import json
import os
import sqlite3
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import fields, replace
from types import SimpleNamespace

import numpy as np
import pytest

from startrain.actor_pause import ActorPauseGate
from startrain.inference import InferenceResponse
from startrain.runtime import RunIdentity
from startrain.selfplay import SelfPlayActor, SelfPlayConfig, SelfPlayIdentity


class _PauseHarness:
    def __init__(self, tmp_path):
        self.stop = threading.Event()
        self.idle = threading.Event()
        self.idle.set()
        self.synchronizations = []
        self.pool = ThreadPoolExecutor(max_workers=2)
        self.gate = ActorPauseGate(
            request_path=tmp_path / "gpu-1.pause.json",
            gpu_id=1,
            worker_name="actor-gpu-1",
            run_identity=RunIdentity(tmp_path / "run.json", "run", "family", 1),
            cohort_ids=("cohort-0", "cohort-1"),
            stop_requested=self.stop.is_set,
            inference_idle=self.idle.is_set,
            synchronize=lambda: self.synchronizations.append("synchronized"),
            stale_seconds=30,
        )
        now = time.time_ns()
        self.request = {
            "schema_version": 1,
            "protocol": "coordinator-pause-v1",
            "token": "pause-token-1",
            "pid": os.getpid() + 1,
            "gpu_id": 1,
            "state": "requested",
            "requested_ns": now,
            "heartbeat_ns": now,
            "run_id": "run",
            "generation_family": "family",
        }
        self.ack = {
            "schema_version": 1,
            "protocol": "coordinator-pause-v1",
            "token": self.request["token"],
            "gpu_id": 1,
            "target_worker": "actor-gpu-1",
            "target_pid": os.getpid(),
            "ack_ns": now,
            "state": "waiting",
        }

    def write_request(self, **changes):
        self.gate.request_path.write_text(json.dumps(self.request | changes))

    def write_ack(self, **changes):
        self.gate.ack_path.write_text(json.dumps(self.ack | changes))

    def request_pause(self):
        self.write_request()
        self.write_ack()
        status = self.gate.poll()
        assert status is not None
        assert status["phase"] == "arena_gpu_quiescing"
        return status

    def await_parked(self, count):
        deadline = time.monotonic() + 5
        while time.monotonic() < deadline:
            status = self.gate.poll()
            if status is not None and status.get("parked_cohorts") == count:
                return status
            time.sleep(0.001)
        pytest.fail(f"cohorts did not reach {count} parked producers")

    def await_resumed(self, token="pause-token-1"):
        deadline = time.monotonic() + 5
        while time.monotonic() < deadline:
            status = self.gate.poll()
            assert status is not None
            assert status["actor_quiescent"] is False
            if status["phase"] == "shared_cohorts":
                assert status["last_resumed_lease_token"] == token
                assert status["parked_cohorts"] == 0
                return status
            assert status["phase"] == "arena_gpu_resuming"
            assert status["last_resumed_lease_token"] != token
            time.sleep(0.001)
        pytest.fail("cohorts did not acknowledge resuming")


@pytest.fixture
def pause(tmp_path):
    harness = _PauseHarness(tmp_path)
    try:
        yield harness
    finally:
        harness.gate.close()
        harness.pool.shutdown(wait=True, cancel_futures=True)


@pytest.mark.parametrize(
    "changes",
    [
        {"pid": 0},
        {"pid": True},
        {"gpu_id": 0},
        {"run_id": "foreign-run"},
        {"generation_family": "foreign-family"},
        {"state": "cancelled"},
        {"requested_ns": 1, "heartbeat_ns": 1},
        {"token": "short"},
    ],
)
def test_invalid_request_never_parks_cohorts(pause, changes):
    pause.write_request(**changes)
    pause.write_ack()

    assert pause.gate.poll() is None
    pause.pool.submit(pause.gate.checkpoint, "cohort-0").result(timeout=5)
    assert not pause.synchronizations


@pytest.mark.parametrize(
    "changes",
    [
        {"token": "foreign-token"},
        {"gpu_id": 0},
        {"target_pid": os.getpid() + 1},
        {"target_worker": "actor-gpu-2"},
        {"ack_ns": 1},
        {"state": "active"},
    ],
)
def test_only_matching_coordinator_waiting_ack_authorizes_pause(pause, changes):
    pause.write_request()
    assert pause.gate.poll() is None
    pause.write_ack(**changes)

    assert pause.gate.poll() is None
    pause.pool.submit(pause.gate.checkpoint, "cohort-0").result(timeout=5)
    assert not pause.synchronizations


@pytest.mark.parametrize("release_state", ["released", "recovered", "draining"])
@pytest.mark.parametrize("request_state", ["missing", "cancelled", "requested"])
def test_release_before_adoption_acknowledges_no_parked_cohorts(
    pause, release_state, request_state
):
    pause.write_request()
    pause.write_ack()
    # The coordinator cancels before the actor's first read of its waiting ack.
    if request_state == "missing":
        pause.gate.request_path.unlink()
    else:
        pause.write_request(state=request_state)
    pause.write_ack(state=release_state)

    status = pause.gate.poll()

    assert status is not None
    assert status["phase"] == "shared_cohorts"
    assert status["last_resumed_lease_token"] == pause.ack["token"]
    assert status["actor_quiescent"] is False
    assert status["parked_cohorts"] == 0
    assert status["cuda_synchronized"] is False
    assert pause.gate.poll() == status
    for cohort in ("cohort-0", "cohort-1"):
        pause.pool.submit(pause.gate.checkpoint, cohort).result(timeout=5)
    assert not pause.synchronizations


@pytest.mark.parametrize(
    "changes",
    [
        {"token": "short"},
        {"gpu_id": 0},
        {"target_pid": os.getpid() + 1},
        {"target_worker": "actor-gpu-2"},
        {"ack_ns": 0},
        {"ack_ns": 1},
        {"run_id": "foreign-run"},
        {"generation_family": "foreign-family"},
        {"schema_version": 2},
        {"protocol": "foreign-protocol"},
    ],
)
def test_invalid_release_without_adopted_lease_is_not_acknowledged(pause, changes):
    pause.write_ack(state="released", **changes)

    assert pause.gate.poll() is None
    pause.pool.submit(pause.gate.checkpoint, "cohort-0").result(timeout=5)
    assert not pause.synchronizations


def test_readiness_requires_all_live_cohorts_idle_inference_and_cuda_sync(pause):
    pause.idle.clear()
    initial = pause.request_pause()
    assert initial["lease_owner_pid"] == pause.request["pid"]
    assert initial["lease_requested_ns"] == pause.request["requested_ns"]
    first = pause.pool.submit(pause.gate.checkpoint, "cohort-0")
    status = pause.await_parked(1)
    assert status["phase"] == "arena_gpu_quiescing"
    assert not status["actor_quiescent"]
    assert not pause.synchronizations

    # A finished cohort has no work or cleanup left to park.
    pause.gate.finish("cohort-1")
    status = pause.gate.poll()
    assert status["live_cohorts"] == 1
    assert not status["inference_idle"]
    assert not status["cuda_synchronized"]
    assert not pause.synchronizations

    pause.idle.set()
    status = pause.gate.poll()
    assert status["phase"] == "arena_gpu_pause"
    assert status["actor_quiescent"] is True
    assert status["inference_idle"] is True
    assert status["cuda_synchronized"] is True
    assert pause.synchronizations == ["synchronized"]
    assert pause.gate.poll() == status
    assert pause.synchronizations == ["synchronized"]
    assert not first.done()


def test_sqlite_writer_commits_before_pause_readiness(pause, tmp_path):
    database = tmp_path / "replay.sqlite3"
    with sqlite3.connect(database) as connection:
        connection.execute("CREATE TABLE replay (value INTEGER)")
    transaction_started = threading.Event()
    allow_commit = threading.Event()

    def write_then_park():
        with sqlite3.connect(database) as connection:
            connection.execute("BEGIN IMMEDIATE")
            connection.execute("INSERT INTO replay VALUES (1)")
            transaction_started.set()
            assert allow_commit.wait(5)
            connection.commit()
        pause.gate.checkpoint("cohort-0")
        return "writer resumed"

    writer = pause.pool.submit(write_then_park)
    try:
        assert transaction_started.wait(5)
        pause.request_pause()
        other = pause.pool.submit(pause.gate.checkpoint, "cohort-1")
        status = pause.await_parked(1)
        assert status["phase"] == "arena_gpu_quiescing"
        assert not status["actor_quiescent"]
        assert not pause.synchronizations
        with sqlite3.connect(database, timeout=0) as competing:
            with pytest.raises(sqlite3.OperationalError, match="locked"):
                competing.execute("INSERT INTO replay VALUES (2)")

        allow_commit.set()
        status = pause.await_parked(2)
        assert status["phase"] == "arena_gpu_pause"
        assert pause.synchronizations == ["synchronized"]
        assert not writer.done() and not other.done()
        # The parked actor retains no SQLite writer lock.
        with sqlite3.connect(database, timeout=0) as competing:
            competing.execute("INSERT INTO replay VALUES (2)")
            assert competing.execute("SELECT value FROM replay").fetchall() == [
                (1,),
                (2,),
            ]
        pause.write_ack(state="released")
        pause.await_resumed()
        assert writer.result(timeout=5) == "writer resumed"
        other.result(timeout=5)
    finally:
        allow_commit.set()


@pytest.mark.parametrize("release_state", ["released", "recovered", "draining"])
def test_cancelled_or_orphaned_request_waits_for_matching_release(pause, release_state):
    pause.gate.finish("cohort-1")
    pause.request_pause()
    producer = pause.pool.submit(pause.gate.checkpoint, "cohort-0")
    pause.await_parked(1)

    pause.write_request(state="cancelled")
    assert pause.gate.poll()["phase"] == "arena_gpu_pause"
    pause.gate.request_path.unlink()
    pause.gate.ack_path.unlink()
    assert pause.gate.poll()["phase"] == "arena_gpu_pause"
    for foreign in (
        {"token": "foreign-token"},
        {"gpu_id": 0},
        {"target_pid": os.getpid() + 1},
        {"target_worker": "actor-gpu-2"},
        {"ack_ns": 1},
    ):
        pause.write_ack(state=release_state, **foreign)
        assert pause.gate.poll()["phase"] == "arena_gpu_pause"
        assert not producer.done()

    pause.write_ack(state=release_state)
    status = pause.await_resumed()
    producer.result(timeout=5)
    assert pause.gate.poll() == status


def test_new_lease_waits_for_cohorts_to_checkpoint_again(pause):
    pause.gate.finish("cohort-1")
    pause.request_pause()
    between_checkpoints = threading.Event()
    checkpoint_again = threading.Event()

    def two_checkpoints():
        pause.gate.checkpoint("cohort-0")
        between_checkpoints.set()
        assert checkpoint_again.wait(5)
        pause.gate.checkpoint("cohort-0")

    producer = pause.pool.submit(two_checkpoints)
    try:
        pause.await_parked(1)
        # Hold the condition only to force the release/adoption race: the old
        # checkpoint cannot leave its wait until this scheduling barrier opens.
        with pause.gate._condition:
            pause.write_ack(state="released")
            status = pause.gate.poll()
            assert status["phase"] == "arena_gpu_resuming"
            assert status["last_resumed_lease_token"] is None
            assert status["parked_cohorts"] == 1
            pause.write_request(token="pause-token-2")
            pause.write_ack(token="pause-token-2")
            assert pause.gate.poll() == status
        assert between_checkpoints.wait(5)
        pause.await_resumed()
        status = pause.gate.poll()
        assert status["phase"] == "arena_gpu_quiescing"
        assert status["lease_token"] == "pause-token-2"
        assert status["parked_cohorts"] == 0
        assert not status["actor_quiescent"]
        assert pause.synchronizations == ["synchronized"]

        checkpoint_again.set()
        assert pause.await_parked(1)["phase"] == "arena_gpu_pause"
        assert pause.synchronizations == ["synchronized", "synchronized"]
        pause.write_ack(token="pause-token-2", state="released")
        pause.await_resumed("pause-token-2")
        producer.result(timeout=5)
    finally:
        checkpoint_again.set()


@pytest.mark.parametrize("shutdown", ["stop", "close"])
def test_shutdown_unparks_without_coordinator_release(pause, shutdown):
    pause.gate.finish("cohort-1")
    pause.request_pause()
    producer = pause.pool.submit(pause.gate.checkpoint, "cohort-0")
    pause.await_parked(1)

    if shutdown == "stop":
        pause.stop.set()
    else:
        pause.gate.close()

    producer.result(timeout=5)
    assert pause.gate.poll() is None


class _UniformEvaluator:
    model_version = "uniform"
    model_step = 7
    model_identity = "uniform"

    def __init__(self):
        self.requests = []

    def evaluate(self, requests):
        self.requests.append(
            (
                tuple(requests.states.hashes),
                tuple(requests.legal_offsets),
                tuple(requests.legal_actions),
            )
        )
        return InferenceResponse(
            tokens=list(requests.tokens),
            values=[0.0] * len(requests),
            policy_offsets=list(requests.legal_offsets),
            policy_logits=[0.0] * len(requests.legal_actions),
        )


class _CapturingSink:
    def __init__(self):
        self.samples = []
        self.metadata = []

    def append(self, samples, **metadata):
        self.samples.extend(samples)
        self.metadata.append(metadata)
        return SimpleNamespace(sample_count=len(samples))


@pytest.mark.native
@pytest.mark.parametrize(
    "pause_at", [1, 2, 12], ids=["root-initialized", "leaf-submitted", "mid-game"]
)
def test_native_pause_preserves_search_replay_and_evaluation_counts(pause, pause_at):
    native = pytest.importorskip("star_native")
    config = SelfPlayConfig(
        rings=4,
        batch_size=1,
        games=1,
        fast_probability=0.0,
        full_probability=1.0,
        fast_simulations=8,
        full_simulations=8,
        simulation_reference_rings=4,
        max_considered=4,
        shard_size=128,
        seed=91,
    )
    identity = SelfPlayIdentity("run", "family", "actor-gpu-1", 3)
    baseline_evaluator, evaluator = _UniformEvaluator(), _UniformEvaluator()
    baseline_sink, sink = _CapturingSink(), _CapturingSink()
    baseline = SelfPlayActor(
        native, baseline_evaluator, baseline_sink, config, identity
    )
    expected = baseline.run()
    at_checkpoint = threading.Event()
    enter_checkpoint = threading.Event()
    checkpoint_calls = []
    progress = []
    pause.gate.finish("cohort-1")

    def checkpoint():
        checkpoint_calls.append(len(evaluator.requests))
        if len(checkpoint_calls) == pause_at:
            at_checkpoint.set()
            assert enter_checkpoint.wait(5)
        pause.gate.checkpoint("cohort-0")

    actor = SelfPlayActor(
        native, evaluator, sink, config, identity, pause_checkpoint=checkpoint
    )
    result = pause.pool.submit(
        actor.run, progress=lambda **event: progress.append(event)
    )
    try:
        assert at_checkpoint.wait(5)
        pause.request_pause()
        enter_checkpoint.set()
        assert pause.await_parked(1)["phase"] == "arena_gpu_pause"
        assert len(evaluator.requests) == pause_at
        assert actor.completed_decisions == 0
        if pause_at == 12:
            assert actor.full_decisions > 0
        assert not sink.samples and not result.done()
        assert actor.interrupted_cohorts == 0

        pause.write_ack(state="released")
        pause.await_resumed()
        assert result.result(timeout=10) == expected
    finally:
        enter_checkpoint.set()

    assert evaluator.requests == baseline_evaluator.requests
    assert checkpoint_calls == list(range(1, len(evaluator.requests) + 1))
    assert sink.metadata == baseline_sink.metadata
    assert len(sink.samples) == len(baseline_sink.samples) > 0
    for actual, expected_sample in zip(
        sink.samples, baseline_sink.samples, strict=True
    ):
        for field in fields(actual):
            actual_value = getattr(actual, field.name)
            expected_value = getattr(expected_sample, field.name)
            if isinstance(actual_value, np.ndarray):
                np.testing.assert_array_equal(actual_value, expected_value)
            else:
                assert actual_value == expected_value, field.name
    assert replace(actor.metrics_snapshot(), replay_append_seconds=0.0) == replace(
        baseline.metrics_snapshot(), replay_append_seconds=0.0
    )
    assert (
        actor.interrupted_cohorts == actor.dropped_games == actor.dropped_decisions == 0
    )
    assert all(event["phase"] != "selfplay_abort" for event in progress)
