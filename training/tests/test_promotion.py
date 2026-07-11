from __future__ import annotations

import json
from dataclasses import asdict, replace
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from startrain.checkpoint import (
    ExponentialMovingAverage,
    collect_model_garbage,
    load_model_manifest,
)
from startrain.arena import ARENA_RESULT_SCHEMA_VERSION, ArenaPair
from startrain.config import (
    ArenaConfig,
    PromotionConfig,
    SchedulerConfig,
    load_config,
)
from startrain.learner import ImmutableModelPublisher
from startrain.model import GraphResTNet
from startrain.optim import OptimizerConfig, build_optimizer
from startrain.orchestration import gpu_pause_ack_path
from startrain.promotion import CoordinatorPauseLease, PromotionSupervisor
from startrain.runtime import RunIdentity, atomic_json
from startrain.training import build_scheduler


def test_promotion_supervisor_bootstraps_and_only_promotes_arena_pass(
    tmp_path,
    monkeypatch,
) -> None:
    experiment = load_config(Path(__file__).parents[1] / "configs" / "small.yaml")
    experiment = replace(
        experiment,
        orchestration=replace(
            experiment.orchestration,
            promotion=PromotionConfig(
                enabled=True,
                gpu_id=0,
                cpu_threads=1,
                poll_seconds=0.01,
                bootstrap_initial_champion=True,
                device="cpu",
            ),
        ),
        arena=ArenaConfig(
            pairs_per_ring=100,
            minimum_pairs_per_ring=100,
            max_pairs_per_ring=100,
            simulations=1,
            max_considered=2,
            regression_floor_elo=-2_500.0,
            bootstrap_samples=200,
        ),
    )
    identity = RunIdentity(
        tmp_path / "run.json", "run-promotion", "family-promotion", 1
    )
    model = GraphResTNet(experiment.model)
    optimizer = build_optimizer(model, OptimizerConfig(kind="adamw"))
    scheduler = build_scheduler(
        optimizer, SchedulerConfig(warmup_steps=0, total_steps=10)
    )
    ema = ExponentialMovingAverage(model, decay=0.9)
    publisher = ImmutableModelPublisher(tmp_path / "learner", identity)
    first = publisher.publish(
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
        ema=ema,
        step=0,
        epoch=0,
        config=experiment.as_dict(),
    )
    with torch.no_grad():
        next(model.parameters()).add_(0.01)
    ema.update(model)
    second = publisher.publish(
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
        ema=ema,
        step=1,
        epoch=1,
        config=experiment.as_dict(),
    )
    with torch.no_grad():
        next(model.parameters()).add_(0.01)
    ema.update(model)
    newest = publisher.publish(
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
        ema=ema,
        step=2,
        epoch=2,
        config=experiment.as_dict(),
    )

    monkeypatch.setattr(
        "startrain.promotion.load_manifest_evaluator",
        lambda _experiment, manifest, device: SimpleNamespace(
            model_version=manifest.model_version,
            model_identity=manifest.model_identity,
        ),
    )
    decision = {"value": "promote"}

    class FakeArenaRunner:
        def __init__(self, **options):
            self.candidate = options["candidate"]
            self.baseline = options["baseline"]

        def run(self, **_options):
            outcomes = (1, 1) if decision["value"] == "promote" else (-1, -1)
            pairs = [
                ArenaPair(
                    ring,
                    pair,
                    pair,
                    0,
                    True,
                    outcomes,
                )
                for ring in experiment.arena.rings
                for pair in range(experiment.arena.pairs_per_ring)
            ]
            return {
                "schema_version": 1,
                "candidate": self.candidate.model_version,
                "baseline": self.baseline.model_version,
                "promotion": {"decision": decision["value"]},
                "pairs": [asdict(pair) for pair in pairs],
                "games": [],
            }

    monkeypatch.setattr("startrain.promotion.ArenaRunner", FakeArenaRunner)
    supervisor = PromotionSupervisor(
        experiment=experiment,
        run_identity=identity,
        candidate_path=tmp_path / "learner" / "candidate.json",
        champion_path=tmp_path / "learner" / "champion.json",
        results_directory=tmp_path / "arena",
        native_module=object(),
        device="cpu",
    )
    first_seed = supervisor._arena_config(second, first).seed
    newest_seed = supervisor._arena_config(newest, first).seed
    assert first_seed != newest_seed
    assert first_seed == supervisor._arena_config(second, first).seed
    assert supervisor.run(stop_requested=lambda: False, once=True) == 1
    champion = load_model_manifest(tmp_path / "learner" / "champion.json")
    assert champion.model_identity == newest.model_identity
    assert champion.model_identity != first.model_identity
    assert champion.role == "champion"
    superseded = (
        tmp_path / "arena" / f"{second.model_identity}-vs-{first.model_identity}.json"
    )
    assert json.loads(superseded.read_text())["promotion"]["decision"] == "superseded"

    with torch.no_grad():
        next(model.parameters()).add_(0.01)
    ema.update(model)
    rejected = publisher.publish(
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
        ema=ema,
        step=3,
        epoch=3,
        config=experiment.as_dict(),
    )
    decision["value"] = "reject"
    assert supervisor.run(stop_requested=lambda: False, once=True) == 1
    retained = load_model_manifest(tmp_path / "learner" / "champion.json")
    assert retained.model_identity == newest.model_identity
    assert retained.model_identity != rejected.model_identity
    dry_gc = collect_model_garbage(
        tmp_path / "learner",
        retain_candidate_manifests=1,
        dry_run=True,
        referenced_result_directory=tmp_path / "arena",
    )
    assert dry_gc["candidate_manifests"] >= 1
    assert dry_gc["deleted_manifests"] == 0


def test_inconclusive_candidate_persists_nonoverlapping_pairs_until_max(
    tmp_path,
    monkeypatch,
) -> None:
    experiment = load_config(Path(__file__).parents[1] / "configs" / "small.yaml")
    experiment = replace(
        experiment,
        orchestration=replace(
            experiment.orchestration,
            promotion=PromotionConfig(
                enabled=True,
                gpu_id=0,
                cpu_threads=1,
                poll_seconds=0.01,
                bootstrap_initial_champion=True,
                device="cpu",
            ),
        ),
        arena=ArenaConfig(
            rings=(3,),
            pairs_per_ring=2,
            minimum_pairs_per_ring=4,
            max_pairs_per_ring=6,
            simulations=1,
            max_considered=2,
            regression_floor_elo=-2_500.0,
            bootstrap_samples=200,
        ),
    )
    identity = RunIdentity(tmp_path / "run.json", "run-continue", "family-continue", 1)
    model = GraphResTNet(experiment.model)
    optimizer = build_optimizer(model, OptimizerConfig(kind="adamw"))
    scheduler = build_scheduler(
        optimizer, SchedulerConfig(warmup_steps=0, total_steps=10)
    )
    ema = ExponentialMovingAverage(model, decay=0.9)
    publisher = ImmutableModelPublisher(tmp_path / "learner", identity)
    champion = publisher.publish(
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
        ema=ema,
        step=0,
        epoch=0,
        config=experiment.as_dict(),
    )
    with torch.no_grad():
        next(model.parameters()).add_(0.01)
    ema.update(model)
    candidate = publisher.publish(
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
        ema=ema,
        step=1,
        epoch=1,
        config=experiment.as_dict(),
    )
    monkeypatch.setattr(
        "startrain.promotion.load_manifest_evaluator",
        lambda _experiment, manifest, device: SimpleNamespace(
            model_version=manifest.model_version,
            model_identity=manifest.model_identity,
        ),
    )

    class BalancedArena:
        def __init__(self, **options):
            self.candidate = options["candidate"]
            self.baseline = options["baseline"]

        def run(self, *, pair_starts, pair_counts, **_options):
            pairs = [
                ArenaPair(
                    ring,
                    pair,
                    pair,
                    0,
                    True,
                    (1, -1),
                )
                for ring, count in pair_counts.items()
                for pair in range(pair_starts[ring], pair_starts[ring] + count)
            ]
            return {
                "schema_version": 1,
                "candidate": self.candidate.model_version,
                "baseline": self.baseline.model_version,
                "pairs": [asdict(pair) for pair in pairs],
                "games": [],
                "promotion": {"decision": "continue"},
            }

    monkeypatch.setattr("startrain.promotion.ArenaRunner", BalancedArena)
    supervisor = PromotionSupervisor(
        experiment=experiment,
        run_identity=identity,
        candidate_path=tmp_path / "learner" / "candidate.json",
        champion_path=tmp_path / "learner" / "champion.json",
        results_directory=tmp_path / "arena",
        native_module=object(),
        device="cpu",
    )
    result_path = (
        tmp_path
        / "arena"
        / f"{candidate.model_identity}-vs-{champion.model_identity}.json"
    )
    newer = None
    for expected_pairs in (2, 4):
        assert supervisor.run(stop_requested=lambda: False, once=True) == 1
        progress = json.loads(result_path.read_text())
        assert progress["schema_version"] == ARENA_RESULT_SCHEMA_VERSION
        assert progress["terminal"] is False
        assert len(progress["pairs"]) == expected_pairs
        if expected_pairs == 2:
            with torch.no_grad():
                next(model.parameters()).add_(0.01)
            ema.update(model)
            newer = publisher.publish(
                model=model,
                optimizer=optimizer,
                scheduler=scheduler,
                ema=ema,
                step=2,
                epoch=2,
                config=experiment.as_dict(),
            )
            # Schema 1 stored the same pair records under the Hoeffding gate.
            # A resumed evaluation must consume them and rewrite schema 2.
            progress["schema_version"] = 1
            result_path.write_text(json.dumps(progress))
    assert supervisor.run(stop_requested=lambda: False, once=True) == 1
    terminal = json.loads(result_path.read_text())
    assert terminal["terminal"] is True
    assert terminal["promotion"]["decision"] == "reject_max_pairs"
    assert sorted(pair["pair"] for pair in terminal["pairs"]) == list(range(6))
    assert newer is not None
    assert supervisor.run(stop_requested=lambda: False, once=True) == 1
    newer_path = (
        tmp_path
        / "arena"
        / f"{newer.model_identity}-vs-{champion.model_identity}.json"
    )
    newer_progress = json.loads(newer_path.read_text())
    assert newer_progress["terminal"] is False
    assert len(newer_progress["pairs"]) == 2


class LeaseClock:
    def __init__(self, on_sleep=None) -> None:
        self.value = 0.0
        self.on_sleep = on_sleep
        self.sleep_calls = 0

    def __call__(self) -> float:
        return self.value

    def sleep(self, seconds: float) -> None:
        self.value += seconds
        self.sleep_calls += 1
        if self.on_sleep is not None:
            self.on_sleep()


def write_lease_ack(
    path: Path,
    *,
    token: str,
    state: str,
    gpu_id: int = 7,
    reason: str | None = None,
) -> None:
    atomic_json(
        path,
        {
            "schema_version": 1,
            "protocol": "coordinator-pause-v1",
            "token": token,
            "state": state,
            "gpu_id": gpu_id,
            "target_worker": "actor-gpu-7",
            "target_role": "actor",
            "coordinator_pid": 123,
            "ack_ns": 1,
            "reason": reason,
        },
    )


def test_tokenized_pause_lease_waits_for_matching_ack_and_persisted_result(
    tmp_path,
) -> None:
    request_path = tmp_path / "arena-gpu-pause.json"
    ack_path = gpu_pause_ack_path(request_path)
    events_path = tmp_path / "pause-events.jsonl"
    result_path = tmp_path / "arena-result.json"
    champion_path = tmp_path / "champion.json"
    lease_holder: dict[str, CoordinatorPauseLease] = {}
    acknowledgements = 0

    def on_sleep() -> None:
        nonlocal acknowledgements
        request = json.loads(request_path.read_text(encoding="utf-8"))
        lease = lease_holder["lease"]
        if request["state"] == "requested":
            acknowledgements += 1
            write_lease_ack(
                ack_path,
                token=("wrong-token" if acknowledgements == 1 else lease.token),
                state="ready",
            )
        elif request["state"] == "released":
            assert result_path.is_file()
            assert champion_path.is_file()
            write_lease_ack(ack_path, token=lease.token, state="released")

    clock = LeaseClock(on_sleep)
    lease = CoordinatorPauseLease(
        request_path=request_path,
        gpu_id=7,
        candidate_identity="candidate-token-test",
        ready_timeout_seconds=0.1,
        release_timeout_seconds=0.1,
        heartbeat_interval_seconds=60.0,
        poll_seconds=0.01,
        stop_requested=lambda: False,
        progress=None,
        events_path=events_path,
        clock=clock,
        sleep=clock.sleep,
    )
    lease_holder["lease"] = lease

    allocated = False
    with lease:
        allocated = True
        active = json.loads(request_path.read_text(encoding="utf-8"))
        assert active["token"] == lease.token
        assert active["state"] == "active"
        atomic_json(result_path, {"terminal": True, "decision": "promote"})
        atomic_json(champion_path, {"model_identity": "candidate-token-test"})

    assert allocated is True
    assert acknowledgements >= 2
    released = json.loads(request_path.read_text(encoding="utf-8"))
    assert released["state"] == "released"
    assert released["token"] == lease.token
    events = [
        json.loads(line)
        for line in events_path.read_text(encoding="utf-8").splitlines()
    ]
    assert [event["event"] for event in events] == [
        "pause_lease_requested",
        "pause_lease_ready",
        "pause_lease_release_requested",
        "pause_lease_released",
    ]


def test_pause_lease_ready_timeout_never_enters_allocation_scope(tmp_path) -> None:
    request_path = tmp_path / "arena-gpu-pause.json"
    clock = LeaseClock()
    lease = CoordinatorPauseLease(
        request_path=request_path,
        gpu_id=7,
        candidate_identity="candidate-timeout-test",
        ready_timeout_seconds=0.02,
        release_timeout_seconds=0.02,
        heartbeat_interval_seconds=60.0,
        poll_seconds=0.01,
        stop_requested=lambda: False,
        progress=None,
        events_path=tmp_path / "pause-events.jsonl",
        clock=clock,
        sleep=clock.sleep,
    )
    allocated = False

    with pytest.raises(TimeoutError, match="token-matched"):
        with lease:
            allocated = True

    assert allocated is False
    request = json.loads(request_path.read_text(encoding="utf-8"))
    assert request["token"] == lease.token
    assert request["state"] == "cancelled"
