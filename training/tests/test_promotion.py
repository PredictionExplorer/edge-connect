from __future__ import annotations

import json
from dataclasses import asdict, replace
from pathlib import Path
from types import SimpleNamespace

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
from startrain.promotion import PromotionSupervisor
from startrain.runtime import RunIdentity
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
    for expected_pairs in (2, 4):
        assert supervisor.run(stop_requested=lambda: False, once=True) == 1
        progress = json.loads(result_path.read_text())
        assert progress["schema_version"] == ARENA_RESULT_SCHEMA_VERSION
        assert progress["terminal"] is False
        assert len(progress["pairs"]) == expected_pairs
        if expected_pairs == 2:
            # Schema 1 stored the same pair records under the Hoeffding gate.
            # A resumed evaluation must consume them and rewrite schema 2.
            progress["schema_version"] = 1
            result_path.write_text(json.dumps(progress))
    assert supervisor.run(stop_requested=lambda: False, once=True) == 1
    terminal = json.loads(result_path.read_text())
    assert terminal["terminal"] is True
    assert terminal["promotion"]["decision"] == "reject_max_pairs"
    assert sorted(pair["pair"] for pair in terminal["pairs"]) == list(range(6))
    assert supervisor.run(stop_requested=lambda: False, once=True) == 0
