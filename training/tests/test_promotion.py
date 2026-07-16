from __future__ import annotations

import json
from contextlib import contextmanager
from dataclasses import asdict, replace
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

import startrain.promotion as promotion_module
from startrain.checkpoint import (
    ExponentialMovingAverage,
    collect_model_garbage,
    load_model_manifest,
    write_resume_cutover,
)
from startrain.arena import ARENA_RESULT_SCHEMA_VERSION, ArenaPair
from startrain.config import (
    ArenaConfig,
    HistoricalEvaluationConfig,
    PromotionConfig,
    SchedulerConfig,
    load_config,
)
from startrain.learner import ImmutableModelPublisher
from startrain.model import GraphResTNet
from startrain.optim import OptimizerConfig, build_optimizer
from startrain.orchestration import gpu_pause_ack_path
from startrain.promotion import (
    CoordinatorPauseLease,
    PromotionSupervisor,
    load_manifest_evaluator,
)
from startrain.runtime import RunIdentity, atomic_json
from startrain.training import build_scheduler


def test_arena_manifest_evaluator_uses_compiled_inference_model(
    tmp_path, monkeypatch: pytest.MonkeyPatch
) -> None:
    experiment = load_config(Path(__file__).parents[1] / "configs" / "small.yaml")
    experiment = replace(experiment, train=replace(experiment.train, compile=True))
    model = torch.nn.Linear(1, 1)
    compile_calls: list[dict[str, object]] = []
    manifest = SimpleNamespace(
        checkpoint=tmp_path / "checkpoint.pt",
        checkpoint_sha256="a" * 64,
        checkpoint_bytes=1,
        model_step=4,
        model_version="sha256-" + "a" * 64,
        model_identity="sha256-" + "a" * 64,
        run_id="run-compile",
        generation_family="family-compile",
    )
    monkeypatch.setattr(promotion_module, "GraphResTNet", lambda _config: model)
    monkeypatch.setattr(
        promotion_module,
        "load_ema_checkpoint",
        lambda *_args, **_kwargs: {"step": 4},
    )

    def compile_model(module, **options):
        assert module is model
        compile_calls.append(options)
        return module

    monkeypatch.setattr(promotion_module, "maybe_compile_model", compile_model)

    evaluator = load_manifest_evaluator(experiment, manifest, device="cpu")

    assert evaluator.model is model
    assert compile_calls == [
        {
            "enabled": True,
            "dynamic": True,
            "fullgraph": True,
            "mode": "default",
            "recompile_limit": None,
            "isolate_recompiles": False,
        }
    ]


def test_promotion_candidates_respect_durable_resume_cutover(tmp_path) -> None:
    experiment = load_config(Path(__file__).parents[1] / "configs" / "small.yaml")
    identity = RunIdentity(tmp_path / "run.json", "run-cutover", "family-cutover", 1)
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
    cutover = publisher.publish(
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
        ema=ema,
        step=1,
        epoch=1,
        config=experiment.as_dict(),
    )
    write_resume_cutover(
        publisher.root,
        manifest=cutover,
        run_id=identity.run_id,
        generation_family=identity.generation_family,
    )
    with torch.no_grad():
        next(model.parameters()).add_(0.01)
    ema.update(model)
    after = publisher.publish(
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
        ema=ema,
        step=2,
        epoch=2,
        config=experiment.as_dict(),
    )
    supervisor = PromotionSupervisor(
        experiment=experiment,
        run_identity=identity,
        candidate_path=publisher.candidate_path,
        champion_path=publisher.champion_path,
        results_directory=tmp_path / "arena",
        native_module=object(),
        device="cpu",
    )

    identities = {
        manifest.model_identity for manifest in supervisor._candidate_manifests()
    }
    assert first.model_identity not in identities
    assert identities == {cutover.model_identity, after.model_identity}
    atomic_json(
        supervisor.status_path,
        {
            "schema_version": 1,
            "candidate_identity": first.model_identity,
            "candidate_step": first.model_step,
            "champion_identity": cutover.model_identity,
            "champion_step": cutover.model_step,
            "decision": "reject",
            "terminal": True,
            "consecutive_terminal_rejections": 9,
            "cutover_created_ns": 0,
            "updated_ns": 1,
        },
    )
    supervisor._write_status(
        candidate=after,
        champion=cutover,
        decision="reject",
        terminal=True,
    )
    status = json.loads(supervisor.status_path.read_text())
    cutover_payload = json.loads((publisher.root / "resume-cutover.json").read_text())
    assert status["cutover_created_ns"] == cutover_payload["created_ns"]
    assert status["consecutive_terminal_rejections"] == 1
    collect_model_garbage(
        publisher.root,
        retain_candidate_manifests=1,
        dry_run=False,
    )
    assert cutover.checkpoint.is_file()
    collect_model_garbage(
        publisher.root,
        retain_candidate_manifests=1,
        dry_run=False,
    )
    assert cutover.checkpoint.is_file()


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
    progress: list[dict[str, object]] = []
    assert (
        supervisor.run(
            stop_requested=lambda: False,
            progress=lambda **details: progress.append(details),
            once=True,
        )
        == 1
    )
    phases = [item.get("phase") for item in progress]
    assert phases.index("arena") < phases.index("arena_loading_candidate")
    assert phases.index("arena_loading_candidate") < phases.index(
        "arena_loading_champion"
    )
    assert phases.index("arena_loading_champion") < phases.index("arena_search_start")
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
    idle_polls = 0
    idle_progress: list[dict[str, object]] = []

    def idle_sleep(_seconds: float) -> None:
        nonlocal idle_polls
        idle_polls += 1

    supervisor.sleep = idle_sleep
    assert (
        supervisor.run(
            stop_requested=lambda: idle_polls >= 2,
            progress=lambda **details: idle_progress.append(details),
        )
        == 0
    )
    assert idle_polls == 2
    assert [
        item["phase"]
        for item in idle_progress
        if item.get("phase") == "awaiting_new_candidate"
    ] == ["awaiting_new_candidate", "awaiting_new_candidate"]
    dry_gc = collect_model_garbage(
        tmp_path / "learner",
        retain_candidate_manifests=1,
        dry_run=True,
        referenced_result_directory=tmp_path / "arena",
    )
    assert dry_gc["candidate_manifests"] == 0
    assert dry_gc["deleted_manifests"] == 0


def test_old_arena_schema_is_rejected_before_new_candidate_evaluation(
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
            rings=(4,),
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
                "schema_version": ARENA_RESULT_SCHEMA_VERSION,
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
    assert supervisor.run(stop_requested=lambda: False, once=True) == 1
    progress = json.loads(result_path.read_text())
    assert progress["schema_version"] == ARENA_RESULT_SCHEMA_VERSION
    assert progress["terminal"] is False
    assert len(progress["pairs"]) == 2

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
    progress["schema_version"] = ARENA_RESULT_SCHEMA_VERSION - 1
    result_path.write_text(json.dumps(progress))

    assert supervisor.run(stop_requested=lambda: False, once=True) == 1
    rejected_old = json.loads(result_path.read_text())
    assert rejected_old["terminal"] is True
    assert rejected_old["promotion"]["decision"] == "superseded"
    newer_path = (
        tmp_path / "arena" / f"{newer.model_identity}-vs-{champion.model_identity}.json"
    )
    newer_progress = json.loads(newer_path.read_text())
    assert newer_progress["terminal"] is False
    assert len(newer_progress["pairs"]) == 2


def _promotion_wave_case(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> SimpleNamespace:
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
            rings=(4,),
            pairs_per_ring=2,
            minimum_pairs_per_ring=4,
            max_pairs_per_ring=6,
            simulations=1,
            max_considered=2,
            regression_floor_elo=-2_500.0,
            bootstrap_samples=200,
        ),
    )
    identity = RunIdentity(tmp_path / "run.json", "run-waves", "family-waves", 1)
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
    supervisor = PromotionSupervisor(
        experiment=experiment,
        run_identity=identity,
        candidate_path=publisher.candidate_path,
        champion_path=publisher.champion_path,
        results_directory=tmp_path / "arena",
        native_module=object(),
        device="cpu",
    )
    result_path = supervisor._result_path(candidate, champion)
    state = SimpleNamespace(
        lease_entries=0,
        evaluator_loads=[],
        runner_instances=0,
        wave_starts=[],
        persisted_pairs=[],
        wave_calls=0,
        stop=False,
        stop_after_wave=None,
        after_wave=None,
    )

    @contextmanager
    def single_lease(**_options):
        state.lease_entries += 1
        yield

    def load_evaluator(_experiment, manifest, *, device):
        assert device == "cpu"
        state.evaluator_loads.append(manifest.model_identity)
        return SimpleNamespace(
            model_version=manifest.model_version,
            model_identity=manifest.model_identity,
        )

    class WaveArena:
        def __init__(self, **options):
            state.runner_instances += 1
            self.candidate = options["candidate"]
            self.baseline = options["baseline"]

        def run(self, *, pair_starts, pair_counts, **_options):
            if result_path.is_file():
                persisted = json.loads(result_path.read_text(encoding="utf-8"))
                state.persisted_pairs.append(
                    [int(pair["pair"]) for pair in persisted["pairs"]]
                )
            else:
                state.persisted_pairs.append([])
            state.wave_starts.append(dict(pair_starts))
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
            state.wave_calls += 1
            if state.stop_after_wave == state.wave_calls:
                state.stop = True
            if state.after_wave is not None:
                state.after_wave(state.wave_calls)
            return {
                "schema_version": ARENA_RESULT_SCHEMA_VERSION,
                "candidate": self.candidate.model_version,
                "baseline": self.baseline.model_version,
                "pairs": [asdict(pair) for pair in pairs],
                "games": [],
                "promotion": {"decision": "continue"},
            }

    monkeypatch.setattr(supervisor, "_gpu_pause", single_lease)
    monkeypatch.setattr(promotion_module, "load_manifest_evaluator", load_evaluator)
    monkeypatch.setattr(promotion_module, "ArenaRunner", WaveArena)
    return SimpleNamespace(
        experiment=experiment,
        identity=identity,
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
        ema=ema,
        publisher=publisher,
        champion=champion,
        candidate=candidate,
        supervisor=supervisor,
        result_path=result_path,
        state=state,
    )


def test_promotion_runs_waves_in_one_lease_and_pins_result_manifests(
    tmp_path,
    monkeypatch,
) -> None:
    case = _promotion_wave_case(tmp_path, monkeypatch)

    def progress(**details) -> None:
        if details.get("phase") == "arena_terminal":
            case.state.stop = True

    assert (
        case.supervisor.run(
            stop_requested=lambda: case.state.stop,
            progress=progress,
        )
        == 3
    )

    assert case.state.lease_entries == 1
    assert case.state.runner_instances == 1
    assert case.state.evaluator_loads == [
        case.candidate.model_identity,
        case.champion.model_identity,
    ]
    assert case.state.wave_starts == [{4: 0}, {4: 2}, {4: 4}]
    assert case.state.persisted_pairs == [
        [],
        [0, 1],
        [0, 1, 2, 3],
    ]
    result = json.loads(case.result_path.read_text(encoding="utf-8"))
    pair_indices = [int(pair["pair"]) for pair in result["pairs"]]
    assert pair_indices == list(range(6))
    assert len(pair_indices) == len(set(pair_indices))
    assert result["terminal"] is True
    assert result["promotion"]["decision"] == "reject_max_pairs"
    assert result["result_kind"] == "promotion"
    assert (
        Path(result["candidate_manifest"])
        == (case.candidate.artifact_manifest or case.candidate.path).resolve()
    )
    assert (
        Path(result["champion_manifest"])
        == (case.champion.artifact_manifest or case.champion.path).resolve()
    )

    with torch.no_grad():
        next(case.model.parameters()).add_(0.01)
    case.ema.update(case.model)
    case.publisher.publish(
        model=case.model,
        optimizer=case.optimizer,
        scheduler=case.scheduler,
        ema=case.ema,
        step=2,
        epoch=2,
        config=case.experiment.as_dict(),
    )
    collect_model_garbage(
        case.publisher.root,
        retain_candidate_manifests=1,
        dry_run=False,
        referenced_result_directory=case.result_path.parent,
    )
    assert Path(result["candidate_manifest"]).is_file()
    assert case.candidate.checkpoint.is_file()


def test_promotion_wave_fills_minimum_without_overshooting() -> None:
    experiment = load_config(Path(__file__).parents[1] / "configs" / "small.yaml")
    supervisor = object.__new__(PromotionSupervisor)
    supervisor.experiment = replace(
        experiment,
        arena=ArenaConfig(
            rings=(4,),
            pairs_per_ring=15,
            minimum_pairs_per_ring=15,
            max_pairs_per_ring=200,
        ),
    )
    accumulated = [
        ArenaPair(4, pair, pair, 0, True, (1, -1)) for pair in range(10)
    ]

    starts, counts = supervisor._wave_plan(accumulated)

    assert starts == {4: 10}
    assert counts == {4: 5}
    accumulated.extend(
        ArenaPair(4, pair, pair, 0, True, (1, -1)) for pair in range(10, 15)
    )
    starts, counts = supervisor._wave_plan(accumulated)
    assert starts == {4: 15}
    assert counts == {4: 15}


def test_historical_crossplay_persists_bounded_waves_without_promoting(
    tmp_path,
    monkeypatch,
) -> None:
    case = _promotion_wave_case(tmp_path, monkeypatch)
    case.supervisor.experiment = replace(
        case.experiment,
        orchestration=replace(
            case.experiment.orchestration,
            historical_evaluation=HistoricalEvaluationConfig(
                enabled=True,
                every_promotions=2,
                anchors_per_evaluation=1,
                pairs_per_ring=5,
                max_pairs_per_ring=10,
            ),
        ),
    )
    crossplay_path = tmp_path / "arena" / "crossplay.json"
    assert not case.publisher.champion_path.exists()

    waves = case.supervisor._evaluate_historical_waves(
        candidate=case.candidate,
        baseline=case.champion,
        result_path=crossplay_path,
        previous=None,
        stop_requested=lambda: False,
        progress=None,
        once=False,
    )

    assert waves == 2
    result = json.loads(crossplay_path.read_text(encoding="utf-8"))
    assert result["result_kind"] == "historical_crossplay"
    assert result["promotion"]["decision"] == "evaluation"
    assert result["terminal"] is True
    assert [pair["pair"] for pair in result["pairs"]] == list(range(10))
    assert Path(result["candidate_manifest"]).is_file()
    assert Path(result["baseline_manifest"]).is_file()
    assert not case.publisher.champion_path.exists()


def test_promotion_stop_persists_wave_and_once_resumes_next_pair_indices(
    tmp_path,
    monkeypatch,
) -> None:
    case = _promotion_wave_case(tmp_path, monkeypatch)
    case.state.stop_after_wave = 1

    assert case.supervisor.run(stop_requested=lambda: case.state.stop) == 1
    first = json.loads(case.result_path.read_text(encoding="utf-8"))
    assert first["terminal"] is False
    assert [pair["pair"] for pair in first["pairs"]] == [0, 1]
    assert case.state.lease_entries == 1
    assert case.state.evaluator_loads == [
        case.candidate.model_identity,
        case.champion.model_identity,
    ]

    case.state.stop = False
    case.state.stop_after_wave = None
    assert (
        case.supervisor.run(
            stop_requested=lambda: case.state.stop,
            once=True,
        )
        == 1
    )
    resumed = json.loads(case.result_path.read_text(encoding="utf-8"))
    pair_indices = [int(pair["pair"]) for pair in resumed["pairs"]]
    assert pair_indices == [0, 1, 2, 3]
    assert len(pair_indices) == len(set(pair_indices))
    assert resumed["terminal"] is False
    assert case.state.wave_starts == [{4: 0}, {4: 2}]
    assert case.state.persisted_pairs == [[], [0, 1]]
    assert case.state.lease_entries == 2
    assert case.state.runner_instances == 2
    assert case.state.evaluator_loads == [
        case.candidate.model_identity,
        case.champion.model_identity,
        case.candidate.model_identity,
        case.champion.model_identity,
    ]


def test_newer_candidate_supersedes_between_session_waves(
    tmp_path,
    monkeypatch,
) -> None:
    case = _promotion_wave_case(tmp_path, monkeypatch)
    published = []

    def publish_newer(wave: int) -> None:
        if wave != 1:
            return
        with torch.no_grad():
            next(case.model.parameters()).add_(0.01)
        case.ema.update(case.model)
        published.append(
            case.publisher.publish(
                model=case.model,
                optimizer=case.optimizer,
                scheduler=case.scheduler,
                ema=case.ema,
                step=2,
                epoch=2,
                config=case.experiment.as_dict(),
            )
        )

    def progress(**details) -> None:
        if details.get("phase") == "candidate_superseded":
            case.state.stop = True

    case.state.after_wave = publish_newer
    assert (
        case.supervisor.run(
            stop_requested=lambda: case.state.stop,
            progress=progress,
        )
        == 1
    )

    result = json.loads(case.result_path.read_text(encoding="utf-8"))
    assert len(published) == 1
    assert case.state.wave_starts == [{4: 0}]
    assert case.state.lease_entries == 1
    assert case.state.evaluator_loads == [
        case.candidate.model_identity,
        case.champion.model_identity,
    ]
    assert [pair["pair"] for pair in result["pairs"]] == [0, 1]
    assert result["terminal"] is True
    assert result["promotion"] == {
        "decision": "superseded",
        "superseded_by": published[0].model_identity,
    }
    assert result["result_kind"] == "promotion"


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
