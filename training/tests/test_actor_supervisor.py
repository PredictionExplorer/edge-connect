from __future__ import annotations

import json
import random
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from torch import nn

from startrain.actor import ActorSupervisor, HistoricalModelPool, ManifestModelProvider
from startrain.config import (
    ConfigError,
    GPUWorkerConfig,
    ModelRefreshConfig,
    load_config,
)
from startrain.features import EncodedBatch
from startrain.model import StarModelOutput
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
                policy_entropy_count=2,
                policy_entropy_sum=1.25,
                policy_weight_sum=0.5,
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
    assert ring in (4, 6, 8, 10)
    assert generation == 0

    metric = json.loads((tmp_path / "metrics.jsonl").read_text().strip())
    assert metric["games"] == 1
    assert metric["gpu_id"] == 2
    assert metric["batch_completed_ns"] >= metric["batch_started_ns"]
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
    assert "pass_decisions" not in metric
    assert metric["game_lengths"] == [3]
    assert metric["game_length_distribution"] == {"3": 1}
    assert metric["mean_game_length"] == 3
    assert metric["policy_entropy_count"] == 2
    assert metric["policy_entropy_mean"] == 0.625
    assert metric["policy_weight_count"] == 2
    assert metric["policy_weight_sum"] == 0.5
    assert metric["policy_weight_mean"] == 0.25
    assert metric["dropped_games"] == 0
    assert metric["dropped_decisions"] == 0
    assert metric["model_refresh_latency_seconds"] >= 0
    assert metric["replay_append_bytes"] == 2_048
    assert metric["replay_append_seconds"] == 0.25
    assert metric["peak_cuda_memory_bytes"] is None
    assert metric["peak_cuda_memory_allocated_bytes"] is None
    assert metric["peak_cuda_memory_reserved_bytes"] is None
    assert metric["model_step"] == 7
    assert metric["lane_id"] == 0
    heartbeat = json.loads((tmp_path / "heartbeat.json").read_text())
    assert heartbeat["phase"] == "stopped"


def test_actor_lane_identity_is_unique_and_range_checked(tmp_path) -> None:
    experiment = load_config(Path(__file__).parents[1] / "configs" / "small.yaml")
    identity = RunIdentity(tmp_path / "run.json", "run-lanes", "family-lanes", 1)
    gpu = GPUWorkerConfig(
        gpu_id=2,
        role="actor",
        cpu_threads=1,
        actor_batch_size=2,
        actor_lanes=3,
    )
    options = {
        "native_module": object(),
        "experiment": experiment,
        "gpu": gpu,
        "replay_directory": tmp_path / "replay",
        "manifest_path": tmp_path / "champion.json",
        "candidate_manifest_path": tmp_path / "candidate.json",
        "run_identity": identity,
        "heartbeat_path": tmp_path / "heartbeat.json",
        "metrics_path": tmp_path / "metrics.jsonl",
        "device": "cpu",
    }

    supervisor = ActorSupervisor(**options, lane_id=1)

    assert supervisor.actor_id == "actor-gpu-2-lane-1"
    assert supervisor.lane_id == 1
    with pytest.raises(ValueError, match="lane_id"):
        ActorSupervisor(**options, lane_id=3)


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
    assert "pass_decision_rate" not in metric
    assert metric["interrupted_cohorts"] == 1
    assert metric["dropped_games"] == 2
    assert metric["dropped_decisions"] == 7


def test_manifest_provider_reuses_compiled_evaluator_and_refreshes_weights(
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
    pointer.write_text("first", encoding="utf-8")
    first_manifest = SimpleNamespace(
        role="champion",
        model_version="sha256-" + "d" * 64,
        model_identity="sha256-" + "d" * 64,
        model_step=12,
        checkpoint=tmp_path / "checkpoint-first.pt",
        checkpoint_sha256="d" * 64,
        checkpoint_bytes=11,
        run_id=identity.run_id,
        generation_family=identity.generation_family,
    )
    second_manifest = SimpleNamespace(
        role="champion",
        model_version="sha256-" + "e" * 64,
        model_identity="sha256-" + "e" * 64,
        model_step=24,
        checkpoint=tmp_path / "checkpoint-second.pt",
        checkpoint_sha256="e" * 64,
        checkpoint_bytes=22,
        run_id=identity.run_id,
        generation_family=identity.generation_family,
    )
    manifests = {"first": first_manifest, "second-manifest": second_manifest}
    monkeypatch.setattr(
        "startrain.actor.load_model_manifest",
        lambda path: manifests[Path(path).read_text(encoding="utf-8")],
    )

    class TinyGraphModel(nn.Module):
        def __init__(self, _config) -> None:
            super().__init__()
            self.weight = nn.Parameter(torch.zeros(()))

        def forward(self, *arguments: torch.Tensor) -> StarModelOutput:
            node_features = arguments[0]
            legal_actions = arguments[-1]
            batch, nodes = node_features.shape[:2]
            value = self.weight.expand(batch)
            policy = self.weight.expand(batch, nodes).masked_fill(
                ~legal_actions, torch.finfo(node_features.dtype).min
            )
            return StarModelOutput(
                policy_logits=policy,
                outcome_logits=torch.stack((torch.zeros_like(value), value), dim=-1),
                score_margin_logits=torch.zeros(batch, 303, dtype=node_features.dtype),
                ownership_logits=torch.zeros(
                    batch, nodes, 3, dtype=node_features.dtype
                ),
                alive_logits=torch.zeros(batch, nodes, dtype=node_features.dtype),
                soft_policy_logits=policy,
            )

    class CompiledGraphModel(nn.Module):
        def __init__(self, raw_model: nn.Module) -> None:
            super().__init__()
            self.raw_model = raw_model
            self.train(raw_model.training)

        def forward(self, *arguments: torch.Tensor) -> StarModelOutput:
            return self.raw_model(*arguments)

    checkpoint_weights = {
        first_manifest.checkpoint: -2.0,
        second_manifest.checkpoint: 2.0,
    }
    checkpoint_steps = {
        first_manifest.checkpoint: first_manifest.model_step,
        second_manifest.checkpoint: second_manifest.model_step,
    }
    load_models: list[nn.Module] = []

    def load_checkpoint(source, *, model, **options):
        source = Path(source)
        load_models.append(model)
        assert options["expected_run_id"] == identity.run_id
        assert options["expected_generation_family"] == identity.generation_family
        assert options["expected_sha256"] in {"d" * 64, "e" * 64}
        assert options["expected_bytes"] in {11, 22}
        with torch.no_grad():
            model.weight.fill_(checkpoint_weights[source])
        return {"step": checkpoint_steps[source]}

    compile_calls: list[tuple[nn.Module, dict[str, object]]] = []

    def compile_model(model, **options):
        compile_calls.append((model, options))
        return CompiledGraphModel(model)

    monkeypatch.setattr("startrain.actor.GraphResTNet", TinyGraphModel)
    monkeypatch.setattr("startrain.actor.load_ema_checkpoint", load_checkpoint)
    monkeypatch.setattr("startrain.actor.maybe_compile_model", compile_model)
    encoded = EncodedBatch(
        node_features=torch.zeros(1, 1, 1),
        global_features=torch.zeros(1, 1),
        neighbor_index=torch.zeros(1, 1, 1, dtype=torch.int64),
        neighbor_mask=torch.ones(1, 1, 1, dtype=torch.bool),
        neighbor_edge_type=torch.zeros(1, 1, 1, dtype=torch.int64),
        node_mask=torch.ones(1, 1, dtype=torch.bool),
        legal_action_mask=torch.ones(1, 1, dtype=torch.bool),
        rings=torch.tensor([4], dtype=torch.int64),
    )
    monkeypatch.setattr(
        "startrain.inference.encode_native_state_data",
        lambda _states: encoded,
    )

    class Requests:
        tokens = [7]
        states = object()
        legal_offsets = [0, 1]
        legal_actions = [0]

        def __len__(self) -> int:
            return 1

    provider = ManifestModelProvider(
        experiment,
        pointer,
        device="cpu",
        run_identity=identity,
    )
    first_evaluator = provider.refresh()
    compiled_model = first_evaluator.model
    topology_cache = first_evaluator._topology_cache
    topology_cache[(4, 1, 1, 1)] = (torch.zeros(1),) * 4
    first_output = first_evaluator.evaluate(Requests())

    pointer.write_text("second-manifest", encoding="utf-8")
    second_evaluator = provider.refresh()
    second_output = second_evaluator.evaluate(Requests())

    assert second_evaluator is first_evaluator
    assert second_evaluator.model is compiled_model
    assert second_evaluator._topology_cache is topology_cache
    assert (4, 1, 1, 1) in second_evaluator._topology_cache
    assert load_models == [compile_calls[0][0], compile_calls[0][0]]
    assert compile_calls == [
        (
            load_models[0],
                {
                    "enabled": True,
                    "dynamic": True,
                    "fullgraph": True,
                    "mode": "default",
                    "recompile_limit": None,
                    "isolate_recompiles": False,
                },
        )
    ]
    assert first_output.values[0] < 0
    assert second_output.values[0] > 0
    assert second_evaluator.model_version == second_manifest.model_version
    assert second_evaluator.model_identity == second_manifest.model_identity
    assert second_evaluator.model_step == second_manifest.model_step
    assert second_evaluator.evaluator_calls == 2
    assert provider.manifest is second_manifest

    candidate_provider = ManifestModelProvider(
        experiment,
        pointer,
        device="cpu",
        run_identity=identity,
        expected_role="candidate",
    )
    with pytest.raises(ValueError, match="expected a candidate"):
        candidate_provider.refresh()


def test_manifest_provider_fails_closed_after_in_place_reload_error(
    tmp_path, monkeypatch
) -> None:
    experiment = load_config(Path(__file__).parents[1] / "configs" / "small.yaml")
    identity = RunIdentity(
        tmp_path / "run.json",
        "run-provider-failure",
        "family-provider-failure",
        1,
    )
    pointer = tmp_path / "champion.json"
    pointer.write_text("first", encoding="utf-8")
    first_manifest = SimpleNamespace(
        role="champion",
        model_version="sha256-" + "1" * 64,
        model_identity="sha256-" + "1" * 64,
        model_step=1,
        checkpoint=tmp_path / "checkpoint-first.pt",
        checkpoint_sha256="1" * 64,
        checkpoint_bytes=1,
        run_id=identity.run_id,
        generation_family=identity.generation_family,
    )
    broken_manifest = SimpleNamespace(
        role="champion",
        model_version="sha256-" + "2" * 64,
        model_identity="sha256-" + "2" * 64,
        model_step=2,
        checkpoint=tmp_path / "checkpoint-broken.pt",
        checkpoint_sha256="2" * 64,
        checkpoint_bytes=2,
        run_id=identity.run_id,
        generation_family=identity.generation_family,
    )
    manifests = {"first": first_manifest, "broken": broken_manifest}
    monkeypatch.setattr(
        "startrain.actor.load_model_manifest",
        lambda path: manifests[Path(path).read_text(encoding="utf-8")],
    )

    class TinyGraphModel(nn.Module):
        def __init__(self, _config) -> None:
            super().__init__()
            self.weight = nn.Parameter(torch.zeros(()))

    load_calls = 0

    def load_checkpoint(source, *, model, **_options):
        nonlocal load_calls
        load_calls += 1
        with torch.no_grad():
            model.weight.fill_(float(load_calls))
        if Path(source) == broken_manifest.checkpoint:
            raise ValueError("broken checkpoint")
        return {"step": first_manifest.model_step}

    compile_calls = 0

    def compile_model(model, **_options):
        nonlocal compile_calls
        compile_calls += 1
        return model

    monkeypatch.setattr("startrain.actor.GraphResTNet", TinyGraphModel)
    monkeypatch.setattr("startrain.actor.load_ema_checkpoint", load_checkpoint)
    monkeypatch.setattr("startrain.actor.maybe_compile_model", compile_model)
    provider = ManifestModelProvider(
        experiment,
        pointer,
        device="cpu",
        run_identity=identity,
    )
    evaluator = provider.refresh()

    pointer.write_text("broken", encoding="utf-8")
    with pytest.raises(ValueError, match="broken checkpoint"):
        provider.refresh()

    assert provider.manifest is first_manifest
    assert provider.evaluator is evaluator
    assert evaluator.model_version == first_manifest.model_version
    assert evaluator.model_step == first_manifest.model_step
    assert load_calls == 2
    assert compile_calls == 1

    pointer.write_text("first", encoding="utf-8")
    with pytest.raises(RuntimeError, match="unusable after a failed"):
        provider.refresh()
    assert load_calls == 2


def test_selfplay_model_source_selects_candidate_or_controlled_mix(tmp_path) -> None:
    experiment = load_config(Path(__file__).parents[1] / "configs" / "small.yaml")
    champion = SimpleNamespace(name="champion", manifest=None)
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

    historical = SimpleNamespace(name="history")
    observed_exclusions: list[set[str]] = []

    class FakeHistoryPool:
        def select(self, *, random_source, exclude):
            assert random_source is supervisor.model_random
            observed_exclusions.append(exclude)
            return historical

    supervisor.history_pool = FakeHistoryPool()
    supervisor._read_candidate = lambda: SimpleNamespace(model_identity="candidate-id")
    history_mix = replace(
        candidate_refresh,
        selfplay_source="candidate_champion_history_mix",
        candidate_probability=0.0,
        history_probability=1.0,
    )
    supervisor.experiment = replace(
        experiment,
        orchestration=replace(
            experiment.orchestration,
            model_refresh=history_mix,
        ),
    )
    assert supervisor._select_model_provider() == ("history", historical)
    assert observed_exclusions == [{"candidate-id"}]

    with pytest.raises(ConfigError, match="candidate_probability"):
        ModelRefreshConfig(candidate_probability=1.1)
    with pytest.raises(ConfigError, match="history mixture"):
        ModelRefreshConfig(
            selfplay_source="candidate_champion_history_mix",
            candidate_probability=0.5,
        )


def test_historical_pool_selects_log_spaced_checkpoints() -> None:
    pool = object.__new__(HistoricalModelPool)
    pool.pool_size = 3
    manifests = [
        SimpleNamespace(model_step=step, model_identity=f"model-{step}")
        for step in range(10)
    ]
    selected = pool._spaced_candidates(manifests)
    assert [item.model_step for item in selected] == [0, 4, 9]
