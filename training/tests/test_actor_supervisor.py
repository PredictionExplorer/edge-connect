from __future__ import annotations

import json
import random
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import pytest

from startrain.actor import ActorSupervisor, ManifestModelProvider
from startrain.config import (
    ConfigError,
    GPUWorkerConfig,
    ModelRefreshConfig,
    load_config,
)
from startrain.runtime import RunIdentity
from startrain.selfplay import SelfPlayMetrics


def test_actor_supervisor_refreshes_only_at_batch_boundaries_and_emits_metrics(
    tmp_path, monkeypatch
) -> None:
    experiment = load_config(Path(__file__).parents[1] / "configs" / "small.yaml")
    experiment = replace(
        experiment,
        orchestration=replace(
            experiment.orchestration,
            model_refresh=replace(
                experiment.orchestration.model_refresh,
                selfplay_source="candidate_champion_mix",
                candidate_probability=1.0,
            ),
        ),
    )
    identity = RunIdentity(
        tmp_path / "run.json",
        "run-actor-supervisor",
        "family-actor-supervisor",
        1,
    )
    evaluator = SimpleNamespace(
        model_version="sha256-" + "a" * 64,
        model_identity="sha256-" + "a" * 64,
        model_step=7,
        evaluator_calls=11,
        evaluator_rows=101,
    )
    provider_events: list[str] = []

    class FakeProvider:
        def __init__(self, *_args, **_kwargs) -> None:
            pass

        def wait_for_initial(self, **_kwargs):
            provider_events.append("initial")
            return evaluator

        def refresh(self):
            provider_events.append("refresh")
            return evaluator

    stopped = {"value": False}
    actor_events: list[tuple[int, int]] = []

    class FakeSelfPlayActor:
        def __init__(self, _native, selected, _store, config, actor_identity) -> None:
            assert selected is evaluator
            self.selected = selected
            actor_events.append((config.rings, actor_identity.generation))

        def run(self, **_kwargs):
            self.selected.evaluator_calls += 4
            self.selected.evaluator_rows += 120
            stopped["value"] = True
            return [
                SimpleNamespace(
                    winner=0,
                    samples=3,
                    policy_samples=1,
                    search_simulations=9,
                    model_version=evaluator.model_version,
                    model_identity=evaluator.model_identity,
                )
            ]

        def metrics_snapshot(self) -> SelfPlayMetrics:
            return SelfPlayMetrics(
                completed_decisions=3,
                full_decisions=1,
                fast_decisions=2,
                pass_decisions=1,
                policy_entropy_count=2,
                policy_entropy_sum=1.25,
                replay_append_calls=1,
                replay_append_bytes=2_048,
                replay_append_seconds=0.25,
            )

    monkeypatch.setattr("startrain.actor.ManifestModelProvider", FakeProvider)
    monkeypatch.setattr("startrain.actor.SelfPlayActor", FakeSelfPlayActor)
    supervisor = ActorSupervisor(
        native_module=object(),
        experiment=experiment,
        gpu=GPUWorkerConfig(
            gpu_id=2,
            role="actor",
            cpu_threads=1,
            actor_batch_size=1,
        ),
        replay_directory=tmp_path / "replay",
        manifest_path=tmp_path / "champion.json",
        candidate_manifest_path=tmp_path / "candidate.json",
        run_identity=identity,
        heartbeat_path=tmp_path / "heartbeat.json",
        metrics_path=tmp_path / "metrics.jsonl",
        device="cpu",
    )
    candidate = SimpleNamespace(
        run_id=identity.run_id,
        generation_family=identity.generation_family,
        model_step=7,
    )
    monkeypatch.setattr(supervisor, "_read_candidate", lambda: candidate)

    completed = supervisor.run(stop_requested=lambda: stopped["value"])
    assert completed == 1
    assert provider_events == ["initial", "initial", "refresh"]
    assert len(actor_events) == 1
    ring, generation = actor_events[0]
    assert 3 <= ring <= 12
    assert generation == 0

    metric = json.loads((tmp_path / "metrics.jsonl").read_text().strip())
    assert metric["games"] == 1
    assert metric["samples"] == 3
    assert metric["policy_samples"] == 1
    assert metric["policy_supervision_rate"] == 1 / 3
    assert metric["search_simulations"] == 9
    assert metric["search_simulations_per_second"] > 0
    assert metric["evaluator_calls"] == 4
    assert metric["evaluator_rows"] == 120
    assert metric["evaluator_rows_per_second"] > 0
    assert metric["attempted_decisions"] == 3
    assert metric["full_decisions"] == 1
    assert metric["fast_decisions"] == 2
    assert metric["pass_decisions"] == 1
    assert metric["pass_decision_rate"] == 1 / 3
    assert metric["game_lengths"] == [3]
    assert metric["game_length_distribution"] == {"3": 1}
    assert metric["mean_game_length"] == 3
    assert metric["policy_entropy_count"] == 2
    assert metric["policy_entropy_mean"] == 0.625
    assert metric["dropped_games"] == 0
    assert metric["dropped_decisions"] == 0
    assert metric["model_refresh_latency_seconds"] >= 0
    assert metric["replay_append_bytes"] == 2_048
    assert metric["replay_append_seconds"] == 0.25
    assert metric["peak_cuda_memory_bytes"] is None
    assert metric["peak_cuda_memory_allocated_bytes"] is None
    assert metric["peak_cuda_memory_reserved_bytes"] is None
    assert metric["model_step"] == 7
    heartbeat = json.loads((tmp_path / "heartbeat.json").read_text())
    assert heartbeat["phase"] == "stopped"


def test_actor_supervisor_records_interrupted_cohort_metrics(
    tmp_path, monkeypatch
) -> None:
    experiment = load_config(Path(__file__).parents[1] / "configs" / "small.yaml")
    identity = RunIdentity(
        tmp_path / "run.json",
        "run-actor-interrupt",
        "family-actor-interrupt",
        1,
    )
    evaluator = SimpleNamespace(
        model_version="sha256-" + "b" * 64,
        model_identity="sha256-" + "b" * 64,
        model_step=9,
        evaluator_calls=0,
        evaluator_rows=0,
    )

    class FakeProvider:
        def __init__(self, *_args, **_kwargs) -> None:
            pass

        def wait_for_initial(self, **_kwargs):
            return evaluator

        def refresh(self):
            return evaluator

    stopped = {"value": False}

    class InterruptedSelfPlayActor:
        def __init__(self, _native, selected, _store, _config, _identity) -> None:
            self.selected = selected

        def run(self, **_kwargs):
            self.selected.evaluator_calls += 3
            self.selected.evaluator_rows += 75
            stopped["value"] = True
            return []

        def metrics_snapshot(self) -> SelfPlayMetrics:
            return SelfPlayMetrics(
                full_decisions=3,
                fast_decisions=4,
                pass_decisions=2,
                interrupted_cohorts=1,
                dropped_games=2,
                dropped_decisions=7,
            )

    monkeypatch.setattr("startrain.actor.ManifestModelProvider", FakeProvider)
    monkeypatch.setattr(
        "startrain.actor.SelfPlayActor",
        InterruptedSelfPlayActor,
    )
    supervisor = ActorSupervisor(
        native_module=object(),
        experiment=experiment,
        gpu=GPUWorkerConfig(
            gpu_id=3,
            role="actor",
            cpu_threads=1,
            actor_batch_size=2,
        ),
        replay_directory=tmp_path / "replay",
        manifest_path=tmp_path / "champion.json",
        candidate_manifest_path=tmp_path / "candidate.json",
        run_identity=identity,
        heartbeat_path=tmp_path / "heartbeat.json",
        metrics_path=tmp_path / "metrics.jsonl",
        device="cpu",
    )
    monkeypatch.setattr(
        supervisor,
        "_read_candidate",
        lambda: SimpleNamespace(
            run_id=identity.run_id,
            generation_family=identity.generation_family,
            model_step=9,
        ),
    )

    assert supervisor.run(stop_requested=lambda: stopped["value"]) == 0
    metric = json.loads((tmp_path / "metrics.jsonl").read_text().strip())
    assert metric["games"] == 0
    assert metric["evaluator_calls"] == 3
    assert metric["evaluator_rows"] == 75
    assert metric["attempted_decisions"] == 7
    assert metric["pass_decision_rate"] == 2 / 7
    assert metric["interrupted_cohorts"] == 1
    assert metric["dropped_games"] == 2
    assert metric["dropped_decisions"] == 7


def test_manifest_provider_compiles_inference_model_when_profile_enables_it(
    tmp_path, monkeypatch
) -> None:
    experiment = load_config(Path(__file__).parents[1] / "configs" / "small.yaml")
    experiment = replace(experiment, train=replace(experiment.train, compile=True))
    identity = RunIdentity(
        tmp_path / "run.json",
        "run-provider",
        "family-provider",
        1,
    )
    pointer = tmp_path / "champion.json"
    pointer.write_text("{}")
    manifest = SimpleNamespace(
        role="champion",
        model_version="sha256-" + "d" * 64,
        model_identity="sha256-" + "d" * 64,
        model_step=12,
        checkpoint=tmp_path / "checkpoint.pt",
        checkpoint_sha256="d" * 64,
        checkpoint_bytes=1,
        run_id=identity.run_id,
        generation_family=identity.generation_family,
    )
    monkeypatch.setattr("startrain.actor.load_model_manifest", lambda _path: manifest)
    monkeypatch.setattr(
        "startrain.actor.load_ema_checkpoint",
        lambda *_args, **_kwargs: {"step": 12},
    )
    compile_calls = []

    def compile_model(model, **options):
        compile_calls.append(options)
        return model

    monkeypatch.setattr("startrain.actor.maybe_compile_model", compile_model)
    provider = ManifestModelProvider(
        experiment,
        pointer,
        device="cpu",
        run_identity=identity,
    )
    evaluator = provider.refresh()
    assert evaluator.model_identity == manifest.model_identity
    assert compile_calls == [{"enabled": True, "dynamic": True, "fullgraph": True}]
    candidate_provider = ManifestModelProvider(
        experiment,
        pointer,
        device="cpu",
        run_identity=identity,
        expected_role="candidate",
    )
    with pytest.raises(ValueError, match="expected a candidate"):
        candidate_provider.refresh()


def test_selfplay_model_source_selects_candidate_or_controlled_mix(tmp_path) -> None:
    experiment = load_config(Path(__file__).parents[1] / "configs" / "small.yaml")
    champion = SimpleNamespace(name="champion")
    candidate = SimpleNamespace(name="candidate")
    supervisor = object.__new__(ActorSupervisor)
    supervisor.provider = champion
    supervisor.candidate_provider = candidate
    supervisor.model_random = random.Random(3)

    candidate_refresh = replace(
        experiment.orchestration.model_refresh,
        selfplay_source="candidate",
    )
    supervisor.experiment = replace(
        experiment,
        orchestration=replace(
            experiment.orchestration,
            model_refresh=candidate_refresh,
        ),
    )
    assert supervisor._select_model_provider() == ("candidate", candidate)

    for probability, expected in ((1.0, candidate), (0.0, champion)):
        mixed = replace(
            candidate_refresh,
            selfplay_source="candidate_champion_mix",
            candidate_probability=probability,
        )
        supervisor.experiment = replace(
            experiment,
            orchestration=replace(
                experiment.orchestration,
                model_refresh=mixed,
            ),
        )
        assert supervisor._select_model_provider()[1] is expected

    with pytest.raises(ConfigError, match="candidate_probability"):
        ModelRefreshConfig(candidate_probability=1.1)
