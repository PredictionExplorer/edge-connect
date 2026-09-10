from copy import deepcopy
from dataclasses import asdict, replace
import hashlib
import json
from pathlib import Path
import random
from types import SimpleNamespace

import pytest
import yaml

import startrain.selfplay as selfplay_module
import startrain.actor as actor_module
from scripts import migrate_continuous_profile as migration
from startrain.actor import resolve_actor_experiment
from startrain.config import ActorPipelineConfig, GPUWorkerConfig, load_config
from startrain.config_compatibility import (
    compatible_config_epoch_payloads,
    without_cohort_search_budget_defaults,
)
from startrain.runtime import RunIdentity
from startrain.selfplay import (
    SelfPlayActor,
    SelfPlayConfig,
    SelfPlayIdentity,
    SelfPlayMetrics,
)
from test_continuous_profile_migration import _fixture, _write_json
from test_selfplay_streaming import (
    Evaluator,
    Sink,
    VARIANTS,
    config,
    sample_fingerprints,
)


PROFILE = Path(__file__).parents[1] / "configs/h100-8gpu-largest-board-priority.yaml"


class RecordingActor(SelfPlayActor):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.waves = []
        self.seed_streams = {}

    def _seed(self, purpose, *parts):
        result = super()._seed(purpose, *parts)
        if purpose in ("search-game-v1", "pda-game-v1"):
            self.seed_streams[(purpose, *parts)] = result
        return result

    def _record_decisions(self, trajectories, positions, state_data, results, **kwargs):
        active = [
            index for index, terminal in enumerate(results.terminal) if not terminal
        ]
        full = [kwargs["full_search_by_row"][index] for index in active]
        for index, selected in zip(active, full, strict=True):
            base = self.config.simulation_budget(full=selected)
            pda = positions[index].pda
            if pda:
                high, low = self.config.playout_budgets(simulations=base, pda=abs(pda))
                expected = high if pda > 0 else low
            else:
                expected = base
            assert kwargs["budgets"][index] == expected
        self.waves.append(full)
        return super()._record_decisions(
            trajectories, positions, state_data, results, **kwargs
        )


def run(native, *, shared=True, rolling=True, variant=VARIANTS[0], **changes):
    sink = Sink()
    selected = config(
        games=4,
        batch_size=2,
        stream_completed_games=True,
        rolling_game_slots=rolling,
        seed_contract="game-v1",
        cohort_search_budgets=shared,
        **changes,
    ).with_variant(variant)
    worker = RecordingActor(
        native,
        Evaluator(),
        sink,
        selected,
        SelfPlayIdentity("budget-test", "budget-family", "budget-actor", 9),
    )
    summaries = worker.run()
    return worker, sink, summaries


@pytest.mark.native
@pytest.mark.parametrize("variant", VARIANTS, ids=lambda value: value.label)
def test_shared_wave_budgets_preserve_search_and_pda_streams(variant):
    native = pytest.importorskip("star_native")
    independent, _, baseline_games = run(native, shared=False, variant=variant)
    global_rng = random.getstate()
    shared, sink, shared_games = run(native, variant=variant)
    assert random.getstate() == global_rng
    assert shared.seed_streams == independent.seed_streams
    assert [
        (game.game_id, game.pda_seat0, game.pda_seat1) for game in shared_games
    ] == [(game.game_id, game.pda_seat0, game.pda_seat1) for game in baseline_games]
    assert all(len(set(wave)) == 1 for wave in shared.waves)
    assert {wave[0] for wave in shared.waves} == {False, True}
    assert all(
        "budget_schedule=cohort-v1" in sample.search_provenance
        for sample in sink.samples
    )
    assert shared.refilled_games == 2


@pytest.mark.native
@pytest.mark.parametrize("rolling", [False, True])
@pytest.mark.parametrize("full", [False, True])
def test_budget_probability_endpoints_and_repeatability(rolling, full):
    native = pytest.importorskip("star_native")
    results = []
    for _ in range(2):
        worker, sink, _ = run(
            native,
            rolling=rolling,
            full_probability=float(full),
            fast_probability=float(not full),
        )
        assert all(wave and set(wave) == {full} for wave in worker.waves)
        assert worker.full_decisions == (worker.completed_decisions if full else 0)
        assert worker.fast_decisions == (0 if full else worker.completed_decisions)
        results.append(sample_fingerprints(sink.samples))
    assert results[0] == results[1]


@pytest.mark.native
@pytest.mark.parametrize("rolling", [False, True])
def test_dedicated_generator_draws_exactly_once_per_search_wave(monkeypatch, rolling):
    native = pytest.importorskip("star_native")
    original_random = random.Random
    generators = []

    class ObservedRandom:
        def __init__(self, seed):
            self.seed = seed
            self.draws = []
            self.random = original_random(seed)
            generators.append(self)

        def getrandbits(self, bits):
            assert bits == 64
            result = self.random.getrandbits(bits)
            self.draws.append(result)
            return result

    monkeypatch.setattr(selfplay_module.random, "Random", ObservedRandom)
    worker, _, _ = run(native, rolling=rolling)
    assert len(generators) == (1 if rolling else 2)
    assert sum(len(generator.draws) for generator in generators) == len(worker.waves)
    threshold = int(worker.config.full_probability * (1 << 64))
    expected = [
        draw < threshold for generator in generators for draw in generator.draws
    ]
    assert [wave[0] for wave in worker.waves] == expected
    assert [generator.seed for generator in generators] == [
        worker._seed("cohort-budget-v1", cohort) for cohort in range(len(generators))
    ]


@pytest.mark.parametrize("kind", [SelfPlayConfig, ActorPipelineConfig])
@pytest.mark.parametrize("value", [0, 1, None, "true"])
def test_budget_option_requires_a_strict_boolean(kind, value):
    with pytest.raises(ValueError, match="boolean"):
        kind(seed_contract="game-v1", cohort_search_budgets=value)


@pytest.mark.parametrize("kind", [SelfPlayConfig, ActorPipelineConfig])
def test_budget_option_requires_explicit_game_seed_contract(kind):
    assert kind().cohort_search_budgets is False
    with pytest.raises(ValueError, match="game-v1"):
        kind(cohort_search_budgets=True)


def enabled_pipeline(**changes):
    return ActorPipelineConfig(
        compatible_work=True,
        stream_completed_games=True,
        rolling_game_slots=True,
        seed_contract="game-v1",
        games_per_task=128,
        cuda_graphs=True,
        **changes,
    )


def test_per_gpu_resolution_and_narrow_legacy_hash_compatibility(tmp_path):
    raw = yaml.safe_load(PROFILE.read_text())
    raw["orchestration"]["gpus"][7]["actor_pipeline"] = asdict(enabled_pipeline())
    path = tmp_path / "old-canary.yaml"
    path.write_text(yaml.safe_dump(raw))
    prior = load_config(path)
    payload = prior.as_dict()
    expected = deepcopy(payload)
    del expected["selfplay"]["cohort_search_budgets"]
    del expected["orchestration"]["gpus"][7]["actor_pipeline"]["cohort_search_budgets"]
    assert without_cohort_search_budget_defaults(payload) == expected
    checksum = hashlib.sha256(migration._canonical_config_bytes(expected)).hexdigest()
    assert checksum in migration._compatible_source_config_sha256s(prior)
    pipeline_gpu = replace(
        prior.orchestration.gpus[7],
        actor_pipeline=enabled_pipeline(cohort_search_budgets=True),
    )
    effective = resolve_actor_experiment(prior, pipeline_gpu)
    assert effective.selfplay.cohort_search_budgets is True
    assert effective.selfplay.seed_contract == "game-v1"
    assert prior.selfplay.cohort_search_budgets is False
    assert effective.model == prior.model and effective.optimizer == prior.optimizer


@pytest.mark.parametrize("value", [True, 0, 1, None, "false"])
def test_budget_compatibility_keeps_enabled_and_untyped_authority(value):
    payload = load_config(PROFILE).as_dict()
    payload["selfplay"]["cohort_search_budgets"] = value
    payload["orchestration"]["gpus"][7]["actor_pipeline"] = asdict(
        enabled_pipeline()
    ) | {"cohort_search_budgets": value}
    for representation in compatible_config_epoch_payloads(payload):
        for actual in (
            representation["selfplay"]["cohort_search_budgets"],
            representation["orchestration"]["gpus"][7]["actor_pipeline"][
                "cohort_search_budgets"
            ],
        ):
            assert actual == value and type(actual) is type(value)


def test_gpu_budget_only_migration_preserves_training_and_evaluation(tmp_path):
    raw = yaml.safe_load(PROFILE.read_text())
    raw["orchestration"]["gpus"][7]["actor_pipeline"] = asdict(enabled_pipeline())
    prior_path = tmp_path / "prior.yaml"
    prior_path.write_text(yaml.safe_dump(raw))
    fixture = _fixture(tmp_path, str(prior_path))
    before = load_config(fixture.old_profile)
    target = yaml.safe_load(fixture.old_profile.read_text())
    target["orchestration"]["gpus"][7]["actor_pipeline"]["cohort_search_budgets"] = True
    fixture.candidate_profile.write_text(yaml.safe_dump(target))
    _write_json(fixture.root / "arena/promotion-status.json", {"terminal": False})
    protected = [
        fixture.checkpoint,
        fixture.root / "learner/recovery.json",
        fixture.root / "learner/champion.json",
        fixture.root / "arena/promotion-status.json",
    ]
    data_before = {path: path.read_bytes() for path in protected}
    plan = migration.plan_migration(fixture.request)
    migration.apply_migration(plan)
    after = load_config(plan.target_profile)
    for field in (
        "model",
        "game",
        "optimizer",
        "train",
        "learner",
        "arena",
        "selfplay",
    ):
        assert getattr(after, field) == getattr(before, field)
    assert after.orchestration.gpus[:7] == before.orchestration.gpus[:7]
    assert after.orchestration.gpus[7].actor_pipeline.cohort_search_budgets is True
    assert {path: path.read_bytes() for path in protected} == data_before
    record = json.loads((fixture.root / "continuous-migrations.jsonl").read_text())
    assert [change["path"] for change in record["changes"]] == [
        "orchestration.gpus.7.actor_pipeline.cohort_search_budgets"
    ]
    assert (
        "utd_segment" not in record and "evaluation_contract_transition" not in record
    )


def test_actor_reports_effective_schedule_in_all_telemetry(tmp_path, monkeypatch):
    experiment = load_config(PROFILE.parent / "small.yaml")
    experiment = replace(
        experiment,
        selfplay=replace(
            experiment.selfplay,
            stream_completed_games=True,
            seed_contract="game-v1",
            cohort_search_budgets=True,
        ),
        orchestration=replace(
            experiment.orchestration,
            model_refresh=replace(
                experiment.orchestration.model_refresh,
                selfplay_source="candidate_champion_mix",
                candidate_probability=1.0,
            ),
        ),
    )
    evaluator = SimpleNamespace(
        model_version="sha256-" + "a" * 64,
        model_identity="sha256-" + "a" * 64,
        model_step=7,
        evaluator_calls=0,
        evaluator_rows=0,
    )
    stopped = False

    class Provider:
        def __init__(self, *args, **kwargs):
            pass

        def wait_for_initial(self, **kwargs):
            return evaluator

        def refresh(self):
            return evaluator

    class Task:
        def __init__(
            self, native, evaluator, store, configured, identity, *, source_role
        ):
            assert configured.cohort_search_budgets is True
            assert source_role == "candidate"

        def run(self, *, stop_requested, progress):
            nonlocal stopped
            evaluator.evaluator_calls += 3
            evaluator.evaluator_rows += 12
            progress(
                phase="selfplay_completed", completed_games=2, persisted_decisions=6
            )
            stopped = True
            return [
                SimpleNamespace(
                    winner=winner,
                    samples=3,
                    policy_samples=3,
                    search_simulations=12,
                )
                for winner in (0, 1)
            ]

        def metrics_snapshot(self):
            return SelfPlayMetrics(
                started_games=2,
                completed_games=2,
                completed_decisions=6,
                full_decisions=3,
                fast_decisions=3,
            )

    monkeypatch.setattr(actor_module, "ManifestModelProvider", Provider)
    monkeypatch.setattr(actor_module, "SelfPlayActor", Task)
    heartbeat = tmp_path / "actor.heartbeat.json"
    metrics = tmp_path / "actor.jsonl"
    supervisor = actor_module.ActorSupervisor(
        native_module=object(),
        experiment=experiment,
        gpu=GPUWorkerConfig(gpu_id=2, role="actor", cpu_threads=1, actor_batch_size=2),
        replay_directory=tmp_path / "replay",
        manifest_path=tmp_path / "champion.json",
        candidate_manifest_path=tmp_path / "candidate.json",
        run_identity=RunIdentity(
            tmp_path / "run.json", "budget-test", "budget-family", 1
        ),
        heartbeat_path=heartbeat,
        metrics_path=metrics,
        device="cpu",
        games_per_batch=2,
    )
    monkeypatch.setattr(
        supervisor,
        "_read_candidate",
        lambda: SimpleNamespace(
            run_id="budget-test", generation_family="budget-family", model_step=7
        ),
    )
    assert supervisor.run(stop_requested=lambda: stopped) == 1
    records = [json.loads(line) for line in metrics.read_text().splitlines()]
    assert any(record.get("record_kind") == "publication" for record in records)
    assert any("games" in record for record in records)
    assert all(record["cohort_search_budgets"] is True for record in records)
    assert json.loads(heartbeat.read_text())["cohort_search_budgets"] is True
