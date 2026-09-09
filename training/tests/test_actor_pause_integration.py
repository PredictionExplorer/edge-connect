from __future__ import annotations

import json
import os
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pytest

from startrain.actor import ActorSupervisor
from startrain.config import load_config
from startrain.runtime import RunIdentity, atomic_json


def wait_for(predicate, seconds=5):
    deadline = time.monotonic() + seconds
    while time.monotonic() < deadline:
        value = predicate()
        if value:
            return value
        time.sleep(0.005)
    pytest.fail("actor pause transition timed out")


@pytest.mark.parametrize("shutdown_while_parked", [False, True])
def test_shared_actor_heartbeat_proves_parking_and_confirms_release(
    tmp_path,
    monkeypatch,
    shutdown_while_parked,
):
    config = load_config(
        Path(__file__).parents[1] / "configs/h100-8gpu-variant-efficiency-stage-b.yaml"
    )
    gpu = next(worker for worker in config.orchestration.gpus if worker.gpu_id == 7)
    request_path = tmp_path / "arena-gpu-pause.json"
    ack_path = tmp_path / "arena-gpu-pause.ack.json"
    heartbeat_path = tmp_path / "actor.heartbeat.json"
    supervisor = ActorSupervisor(
        native_module=object(),
        experiment=config,
        gpu=gpu,
        replay_directory=tmp_path / "replay",
        manifest_path=tmp_path / "champion.json",
        candidate_manifest_path=tmp_path / "candidate.json",
        run_identity=RunIdentity(tmp_path / "run.json", "run", "family", 1),
        heartbeat_path=heartbeat_path,
        metrics_path=tmp_path / "actor.jsonl",
        device="cpu",
        gpu_pause_path=request_path,
    )
    stop = threading.Event()
    counters = {}

    def child_run(child, *, stop_requested, startup_cancel_requested):
        assert startup_cancel_requested is not None
        counters[child.actor_id] = 0
        while not stop_requested():
            assert child._pause_checkpoint is not None
            child._pause_checkpoint()  # Same callback used between native leaves.
            if stop_requested():
                break
            counters[child.actor_id] += 1
            time.sleep(0.002)
        return counters[child.actor_id]

    monkeypatch.setattr(ActorSupervisor, "run", child_run)

    def heartbeat():
        return json.loads(heartbeat_path.read_text()) if heartbeat_path.exists() else {}

    with ThreadPoolExecutor(max_workers=1) as controller:
        future = controller.submit(supervisor._run_cohorts, stop_requested=stop.is_set)
        try:
            wait_for(lambda: len(counters) == 2 and min(counters.values()) >= 3)
            requested_ns = time.time_ns()
            token = "cooperative-integration-token"
            atomic_json(
                request_path,
                {
                    "schema_version": 1,
                    "protocol": "coordinator-pause-v1",
                    "token": token,
                    "pid": os.getpid(),
                    "gpu_id": 7,
                    "state": "requested",
                    "requested_ns": requested_ns,
                    "heartbeat_ns": requested_ns,
                },
            )
            acknowledgement = {
                "schema_version": 1,
                "protocol": "coordinator-pause-v1",
                "token": token,
                "gpu_id": 7,
                "target_pid": os.getpid(),
                "target_worker": "actor-gpu-7",
                "state": "waiting",
                "ack_ns": time.time_ns(),
            }
            atomic_json(ack_path, acknowledgement)
            parked = wait_for(
                lambda: value if (value := heartbeat()).get("actor_quiescent") else None
            )
            assert parked["phase"] == "arena_gpu_pause"
            assert parked["lease_token"] == token
            assert parked["parked_cohorts"] == parked["live_cohorts"] == 2
            assert parked["inference_idle"] and parked["cuda_synchronized"]
            assert parked["progress_ns"] >= requested_ns
            frozen = dict(counters)
            request_path.unlink()
            time.sleep(0.15)
            assert counters == frozen  # Request loss does not abandon ownership.
            if not shutdown_while_parked:
                release_ns = time.time_ns()
                atomic_json(
                    ack_path,
                    {**acknowledgement, "state": "released", "ack_ns": release_ns},
                )
                resumed = wait_for(
                    lambda: (
                        value
                        if (value := heartbeat()).get("last_resumed_lease_token")
                        == token
                        else None
                    )
                )
                assert resumed["phase"] == "shared_cohorts"
                assert not resumed["actor_quiescent"] and resumed["parked_cohorts"] == 0
                assert resumed["progress_ns"] >= release_ns
                wait_for(
                    lambda: all(counters[key] > value for key, value in frozen.items())
                )
                # Confirmation remains visible without any cohort progress
                # heartbeat, so the coordinator cannot miss a brief pulse.
                time.sleep(0.15)
                assert heartbeat()["last_resumed_lease_token"] == token
            stop.set()
            assert future.result(timeout=5) == sum(counters.values())
        finally:
            stop.set()
            future.result(timeout=5)
