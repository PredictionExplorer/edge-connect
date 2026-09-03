import copy
from contextlib import nullcontext
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest
import torch

from scripts.validate_continuous_profile import validate_continuous_config
from startrain.checkpoint import (
    ExponentialMovingAverage,
    load_checkpoint,
    save_checkpoint,
)
from startrain.config import (
    ConfigError,
    DataConfig,
    HistoricalEvaluationConfig,
    ModelRefreshConfig,
    PlateauConfig,
    SchedulerConfig,
    load_config,
)
from startrain.export import ONNX_INPUT_NAMES, ONNXStarModel, export_onnx
from startrain.features import DoubleStarPosition, encode_batch
from startrain.model import GraphResTNet, ModelConfig
from startrain.optim import (
    MuonAdamW,
    OptimizerConfig,
    build_optimizer,
    split_decay_parameters,
)
from startrain.replay import ReplayBatch, ReplaySample, collate_replay_samples
from startrain.sampling import RingStratifiedSampler
from startrain.scoring import PlayerScore, ScoreResult
from startrain.topology import get_topology
from startrain.training import (
    DeviceBatchPrefetcher,
    NonFiniteTrainingError,
    TrainMetricAccumulator,
    TrainStepResult,
    build_scheduler,
    ema_effective_turnover,
    maybe_compile_model,
    scheduler_diagnostics,
    train_step,
    unwrap_model,
)


CONFIGS = Path(__file__).parents[1] / "configs"


def tiny_model() -> GraphResTNet:
    return GraphResTNet(
        ModelConfig(
            width=16,
            rrt_groups=5,
            attention_heads=4,
            kv_heads=1,
        )
    )


def test_scheduler_holds_minimum_rate_after_configured_horizon() -> None:
    parameter = torch.nn.Parameter(torch.ones(()))
    optimizer = torch.optim.SGD([parameter], lr=1.0)
    scheduler = build_scheduler(
        optimizer,
        SchedulerConfig(warmup_steps=0, total_steps=10, min_lr_ratio=0.05),
    )
    for _ in range(25):
        optimizer.step()
        scheduler.step()
    assert optimizer.param_groups[0]["lr"] == pytest.approx(0.05)


def test_scheduler_diagnostics_track_age_and_segment_position() -> None:
    parameter = torch.nn.Parameter(torch.ones(()))
    optimizer = torch.optim.SGD([parameter], lr=1.0)
    scheduler = build_scheduler(
        optimizer,
        SchedulerConfig(warmup_steps=2, total_steps=4, min_lr_ratio=0.1),
    )
    initial = scheduler_diagnostics(scheduler)
    assert initial.age_steps == 0
    assert initial.segment == "warmup"
    assert initial.segment_position == 0.0

    for _ in range(2):
        optimizer.step()
        scheduler.step()
    cosine = scheduler_diagnostics(scheduler)
    assert cosine.age_steps == 2
    assert cosine.segment == "cosine"
    assert cosine.segment_step == 0
    assert cosine.segment_position == 0.0

    for _ in range(2):
        optimizer.step()
        scheduler.step()
    floor = scheduler_diagnostics(scheduler)
    assert floor.age_steps == 4
    assert floor.segment == "floor"
    assert floor.segment_position == 1.0


def test_ema_effective_turnover_is_stable_and_strict() -> None:
    assert ema_effective_turnover(0.9, 0) == 0.0
    assert ema_effective_turnover(0.9, 2) == pytest.approx(0.19)
    assert ema_effective_turnover(0.0, 10) == 1.0
    with pytest.raises(ValueError, match="decay"):
        ema_effective_turnover(1.0, 1)
    for invalid in (-1, True, 1.5):
        with pytest.raises(ValueError, match="updates"):
            ema_effective_turnover(0.9, invalid)  # type: ignore[arg-type]


def sample(rings: int = 4) -> ReplaySample:
    topology = get_topology(rings)
    stones = torch.full((topology.n,), -1, dtype=torch.int8)
    stones[0] = 0
    position = DoubleStarPosition(
        rings=rings,
        stones=stones,
        to_move=1,
        moves_left=2,
        opening=False,
        terminal=False,
    )
    legal = stones.numpy() == -1
    policy = legal.astype(np.float32)
    policy /= policy.sum()
    return ReplaySample.from_position(
        position,
        policy=policy,
        final_score=ScoreResult(
            players=(
                PlayerScore(10, 3, 1, 1, 0, 11),
                PlayerScore(5, 2, 1, 0, 0, 5),
            ),
            node_owner=torch.zeros(topology.n, dtype=torch.int8),
            alive_stone=torch.zeros(topology.n, dtype=torch.bool),
            contested_peries=0,
            leader=0,
        ),
        search_provenance="mcts:test",
        policy_provenance="root-visits",
    )


def test_yaml_configs_load_strictly() -> None:
    small = load_config(CONFIGS / "small.yaml")
    h100 = load_config(CONFIGS / "h100.yaml")
    assert small.schema_version == h100.schema_version == 3
    assert small.model.rrt_groups == h100.model.rrt_groups == 5
    assert h100.model.kv_heads < h100.model.attention_heads
    assert small.train.precision == "fp32"
    assert h100.train.precision == "bf16"
    assert h100.train.global_batch_size(4) == (h100.train.per_rank_batch_size * 4)
    assert h100.profile == "standalone-smoke"
    assert h100.data.ring_stratified is False
    assert small.data.min_batches_for_workers == 32
    assert small.as_dict()["data"]["min_batches_for_workers"] == 32
    continuous = load_config(CONFIGS / "h100-8gpu.yaml")
    assert continuous.profile == "continuous"
    assert continuous.data.ring_stratified is True
    assert continuous.learner.use_ring_mixture_curriculum is False
    assert continuous.orchestration.model_refresh.selfplay_source == "champion"
    assert continuous.orchestration.model_refresh.candidate_probability == 0.8
    assert continuous.selfplay.record_fast_policy_targets is False
    assert continuous.selfplay.max_considered_cap == 64
    assert continuous.selfplay.clinch_finalization == "loser-fill"
    optimized = load_config(CONFIGS / "h100-8gpu-optimized.yaml")
    assert optimized.learner.use_ring_mixture_curriculum is True
    assert len(optimized.orchestration.actor_gpus) == 7
    assert {gpu.actor_batch_size for gpu in optimized.orchestration.actor_gpus} == {128}
    assert optimized.orchestration.actor_games_per_batch == 128
    assert optimized.orchestration.promotion.gpu_id == 7
    assert optimized.orchestration.promotion.pause_sharing_mode is True
    throughput = load_config(CONFIGS / "h100-8gpu-throughput.yaml")
    assert throughput.orchestration.training_objective == "generalist"
    assert throughput.as_dict()["orchestration"]["training_objective"] == "generalist"
    assert sum(gpu.actor_lanes for gpu in throughput.orchestration.actor_gpus) == 13
    assert throughput.data.shard_cache_size == 8
    assert throughput.orchestration.learner_gpus[0].cpu_affinity == "0-103"
    assert (
        next(
            gpu for gpu in throughput.orchestration.actor_gpus if gpu.gpu_id == 7
        ).actor_lanes
        == 1
    )
    assert throughput.selfplay.fast_policy_weight == 0.25
    assert throughput.selfplay.record_fast_policy_targets is True
    assert throughput.selfplay.considered_actions() == 16
    assert replace(throughput.selfplay, rings=10).considered_actions() == 27
    assert throughput.learner.unlimited is True
    assert throughput.orchestration.ring_mixture.weights_for_step(360_000) == (
        0.15,
        0.15,
        0.15,
        0.55,
    )
    assert throughput.orchestration.ring_mixture.weights_for_step(1_000_000) == (
        0.1,
        0.1,
        0.1,
        0.7,
    )
    assert throughput.orchestration.plateau.consecutive_terminal_rejections == 2
    assert throughput.orchestration.plateau.reset_learning_rate_scale == 0.5
    assert throughput.learner.candidate_interval == 28_000
    assert throughput.learner.max_replay_lag_steps == 60_000
    assert throughput.orchestration.plateau.max_learner_champion_lag_steps == 60_000
    assert throughput.orchestration.promotion.finish_inflight_candidate is True
    assert throughput.arena.continuation_pairs_per_ring == 25
    validate_continuous_config(throughput)
    ring10_only = load_config(CONFIGS / "h100-8gpu-ring10-only.yaml")
    assert ring10_only.game.rings == (4, 6, 8, 10)
    assert ring10_only.orchestration.training_objective == "ring10_only"
    assert ring10_only.orchestration.ring_mixture.rings == (4, 6, 8, 10)
    assert ring10_only.orchestration.ring_mixture.weights_for_step(0) == (
        0.0,
        0.0,
        0.0,
        1.0,
    )
    assert ring10_only.selfplay.rings == 10
    assert ring10_only.arena.rings == (10,)
    assert ring10_only.arena.continuation_pairs_per_ring == 50
    assert ring10_only.arena.required_regression_rings == ()
    assert ring10_only.arena.per_ring_regression_floor_elo == {}
    assert ring10_only.arena.promotion_pair_ratios == {}
    assert ring10_only.arena.weighted_initial_blocks == 0
    assert ring10_only.arena.weighted_continuation_blocks == 0
    assert ring10_only.arena.weighted_max_blocks == 0
    validate_continuous_config(ring10_only)
    learner_shared = load_config(CONFIGS / "h100-8gpu-learner-shared.yaml")
    assert [gpu.gpu_id for gpu in learner_shared.orchestration.actor_gpus] == list(
        range(1, 8)
    )
    assert sum(gpu.actor_lanes for gpu in learner_shared.orchestration.actor_gpus) == 14
    assert learner_shared.orchestration.promotion.gpu_id == 0
    assert learner_shared.orchestration.promotion.pause_sharing_mode is True
    assert learner_shared.orchestration.promotion.max_waves_per_lease == 1
    assert learner_shared.orchestration.promotion.inter_wave_cooldown_seconds == 1_800
    assert learner_shared.orchestration.historical_evaluation.enabled is False
    for field in (
        "game",
        "model",
        "loss",
        "optimizer",
        "train",
        "data",
        "selfplay",
        "learner",
        "arena",
    ):
        assert getattr(learner_shared, field) == getattr(throughput, field)
    for field in (
        "actor_games_per_batch",
        "ring_mixture",
        "model_refresh",
        "restart",
        "shutdown",
        "distributed",
        "hardware_health",
        "plateau",
        "retention",
    ):
        assert getattr(learner_shared.orchestration, field) == getattr(
            throughput.orchestration, field
        )
    validate_continuous_config(learner_shared)
    for unsafe_orchestration in (
        replace(
            learner_shared.orchestration,
            promotion=replace(
                learner_shared.orchestration.promotion,
                max_waves_per_lease=2,
            ),
        ),
        replace(
            learner_shared.orchestration,
            promotion=replace(
                learner_shared.orchestration.promotion,
                inter_wave_cooldown_seconds=1_799,
            ),
        ),
        replace(
            learner_shared.orchestration,
            historical_evaluation=HistoricalEvaluationConfig(enabled=True),
        ),
    ):
        with pytest.raises(ValueError, match="learner-shared promotion"):
            validate_continuous_config(
                replace(learner_shared, orchestration=unsafe_orchestration)
            )
    with pytest.raises(ValueError, match="resumable in-flight"):
        validate_continuous_config(
            replace(
                throughput,
                orchestration=replace(
                    throughput.orchestration,
                    promotion=replace(
                        throughput.orchestration.promotion,
                        finish_inflight_candidate=False,
                    ),
                ),
            )
        )
    with pytest.raises(ValueError, match="bounded continuation waves"):
        validate_continuous_config(
            replace(
                throughput,
                arena=replace(
                    throughput.arena,
                    continuation_pairs_per_ring=50,
                ),
            )
        )
    warmstart = load_config(CONFIGS / "h100-8gpu-champion-warmstart.yaml")
    assert warmstart.optimizer.adamw_lr == pytest.approx(1.5e-4)
    assert warmstart.optimizer.muon_lr == pytest.approx(1e-2)
    assert warmstart.learner.target_updates_per_new_sample == 1.0
    assert warmstart.learner.candidate_interval_examples == 5_000_000
    assert warmstart.learner.minimum_unique_samples_per_ring == 8_192
    assert warmstart.orchestration.model_refresh.selfplay_source == "champion"
    assert warmstart.arena.per_ring_regression_floor_elo == {
        4: -35.0,
        6: -35.0,
        8: -35.0,
    }
    validate_continuous_config(warmstart)
    autonomous = load_config(CONFIGS / "h100-8gpu-autonomous.yaml")
    assert autonomous.orchestration.autonomous.enabled is True
    assert autonomous.orchestration.model_refresh.selfplay_source == (
        "candidate_champion_history_mix"
    )
    assert autonomous.orchestration.model_refresh.history_probability == 0.25
    assert autonomous.orchestration.plateau.action == "reduce_lr_keep_weights"
    assert autonomous.learner.target_updates_per_new_sample == 1.0
    assert autonomous.learner.candidate_interval_examples == 5_000_000
    assert autonomous.learner.selfplay_snapshot_interval_examples == 3_000_000
    assert autonomous.learner.selfplay_snapshot_warmup_examples == 20_000_000
    assert autonomous.learner.selfplay_snapshot_warmup_interval_examples == 1_000_000
    assert autonomous.data.shards_per_batch == 4
    assert autonomous.arena.pairs_per_ring == 15
    assert autonomous.arena.minimum_pairs_per_ring == 15
    assert autonomous.orchestration.historical_evaluation.enabled is False
    assert autonomous.orchestration.historical_evaluation.every_promotions == 4
    assert autonomous.orchestration.historical_evaluation.pairs_per_ring == 10
    assert autonomous.orchestration.promotion.gpu_id == 0
    assert autonomous.orchestration.promotion.max_waves_per_lease == 1
    assert autonomous.orchestration.promotion.inter_wave_cooldown_seconds == 1_800
    assert sum(gpu.actor_lanes for gpu in autonomous.orchestration.actor_gpus) == 14
    assert replace(autonomous.selfplay, rings=10).considered_actions() == 53
    validate_continuous_config(autonomous)

    with pytest.raises(ValueError, match="cross-shard"):
        validate_continuous_config(
            replace(
                autonomous,
                data=replace(autonomous.data, shards_per_batch=1),
            )
        )
    with pytest.raises(ValueError, match="update-to-data"):
        validate_continuous_config(
            replace(
                autonomous,
                learner=replace(
                    autonomous.learner,
                    target_updates_per_new_sample=None,
                ),
            )
        )
    with pytest.raises(ValueError, match="historical evaluation"):
        validate_continuous_config(
            replace(
                autonomous,
                orchestration=replace(
                    autonomous.orchestration,
                    historical_evaluation=replace(
                        autonomous.orchestration.historical_evaluation,
                        enabled=True,
                        max_pairs_per_ring=15,
                    ),
                ),
            )
        )
    with pytest.raises(ValueError, match="learner-shared promotion"):
        validate_continuous_config(
            replace(
                autonomous,
                orchestration=replace(
                    autonomous.orchestration,
                    promotion=replace(
                        autonomous.orchestration.promotion,
                        max_waves_per_lease=2,
                    ),
                ),
            )
        )


def test_continuous_training_objective_mismatches_fail_closed(tmp_path: Path) -> None:
    generalist = load_config(CONFIGS / "h100-8gpu-throughput.yaml")
    ring10_only = load_config(CONFIGS / "h100-8gpu-ring10-only.yaml")

    with pytest.raises(ValueError, match="step-1000000 weights"):
        validate_continuous_config(
            replace(
                ring10_only,
                orchestration=replace(
                    ring10_only.orchestration,
                    training_objective="generalist",
                ),
            )
        )
    with pytest.raises(ValueError, match="exactly step-0 weights"):
        validate_continuous_config(
            replace(
                generalist,
                orchestration=replace(
                    generalist.orchestration,
                    training_objective="ring10_only",
                ),
            )
        )
    with pytest.raises(ValueError, match="single-ring unguarded"):
        validate_continuous_config(
            replace(
                ring10_only,
                arena=replace(ring10_only.arena, required_regression_rings=None),
            )
        )

    invalid = tmp_path / "invalid-objective.yaml"
    source = (CONFIGS / "h100-8gpu-throughput.yaml").read_text(encoding="utf-8")
    invalid.write_text(
        source.replace(
            "orchestration:\n  enabled: true\n",
            "orchestration:\n  enabled: true\n  training_objective: best_ring\n",
            1,
        ),
        encoding="utf-8",
    )
    with pytest.raises(ConfigError, match="training_objective"):
        load_config(invalid)

    malformed_ring10 = tmp_path / "malformed-ring10.yaml"
    ring10_source = (CONFIGS / "h100-8gpu-ring10-only.yaml").read_text(encoding="utf-8")
    malformed_ring10.write_text(
        ring10_source.replace("selfplay:\n  rings: 10\n", "selfplay:\n  rings: 4\n", 1),
        encoding="utf-8",
    )
    with pytest.raises(ConfigError, match="ring-10 self-play"):
        load_config(malformed_ring10)


def test_data_worker_window_threshold_loads_and_validates_strictly(
    tmp_path,
) -> None:
    source = (CONFIGS / "small.yaml").read_text(encoding="utf-8")
    configured = tmp_path / "worker-threshold.yaml"
    configured.write_text(
        source.replace(
            "  workers: 0\n",
            "  workers: 2\n  min_batches_for_workers: 17\n",
            1,
        ),
        encoding="utf-8",
    )

    experiment = load_config(configured)

    assert experiment.data.workers == 2
    assert experiment.data.min_batches_for_workers == 17
    assert experiment.as_dict()["data"]["min_batches_for_workers"] == 17
    for invalid in (0, -1, True, 2.5, "32"):
        with pytest.raises(ConfigError, match="worker settings"):
            DataConfig(min_batches_for_workers=invalid)  # type: ignore[arg-type]


def test_inference_compile_and_historical_evaluation_settings_load_strictly(
    tmp_path,
) -> None:
    source = (CONFIGS / "h100-8gpu-autonomous.yaml").read_text(encoding="utf-8")
    configured = tmp_path / "inference-evaluation.yaml"
    configured.write_text(
        source.replace(
            "    inference_compile_dynamic: true\n"
            "    inference_compile_mode: default\n",
            "    inference_compile_dynamic: false\n"
            "    inference_compile_mode: reduce-overhead\n",
            1,
        ),
        encoding="utf-8",
    )

    experiment = load_config(configured)

    assert experiment.orchestration.model_refresh.inference_compile_dynamic is False
    assert (
        experiment.orchestration.model_refresh.inference_compile_mode
        == "reduce-overhead"
    )
    assert experiment.orchestration.historical_evaluation.enabled is False
    assert experiment.orchestration.historical_evaluation.max_pairs_per_ring == 10
    with pytest.raises(ConfigError, match="inference_compile_mode"):
        ModelRefreshConfig(inference_compile_mode="fastest")  # type: ignore[arg-type]
    with pytest.raises(ConfigError, match="maximum"):
        HistoricalEvaluationConfig(pairs_per_ring=5, max_pairs_per_ring=4)


def test_yaml_parses_opt_in_learner_ring_mixture_curriculum(tmp_path) -> None:
    source = (CONFIGS / "small.yaml").read_text(encoding="utf-8")
    configured = tmp_path / "curriculum.yaml"
    configured.write_text(
        source.replace(
            "learner:\n",
            "learner:\n  use_ring_mixture_curriculum: true\n",
            1,
        ),
        encoding="utf-8",
    )

    experiment = load_config(configured)
    assert experiment.learner.use_ring_mixture_curriculum is True


def test_old_config_schema_and_noncanonical_rings_are_rejected(tmp_path) -> None:
    source = (CONFIGS / "small.yaml").read_text(encoding="utf-8")
    old = tmp_path / "old.yaml"
    old.write_text(source.replace("schema_version: 3", "schema_version: 2", 1))
    with pytest.raises(ConfigError, match="schema_version must be 3"):
        load_config(old)

    odd = tmp_path / "odd.yaml"
    odd.write_text(source.replace("rings: 4", "rings: 5", 1))
    with pytest.raises(ConfigError, match="one of"):
        load_config(odd)


def test_continuous_validator_requires_non_compounding_plateau_recovery() -> None:
    ring10_only = load_config(CONFIGS / "h100-8gpu-ring10-only.yaml")
    plateau = ring10_only.orchestration.plateau
    assert plateau.minimum_learning_rate_scale == 0.25
    assert plateau.restore_scale_on_promotion is True

    def with_plateau(**changes: object):
        return replace(
            ring10_only,
            orchestration=replace(
                ring10_only.orchestration,
                plateau=replace(plateau, **changes),
            ),
        )

    validate_continuous_config(with_plateau(action="reduce_lr_keep_weights"))
    validate_continuous_config(with_plateau(reset_learning_rate_scale=1.0))
    validate_continuous_config(
        with_plateau(
            action="reduce_lr_keep_weights",
            consecutive_terminal_rejections=3,
        )
    )
    for changes in (
        {"action": "pause"},
        {"action": "reduce_lr_keep_weights", "consecutive_terminal_rejections": 4},
        {"restore_scale_on_promotion": False},
        {"minimum_learning_rate_scale": 0.1, "reset_learning_rate_scale": 0.5},
    ):
        with pytest.raises(ValueError, match="non-compounding plateau recovery"):
            validate_continuous_config(with_plateau(**changes))
    with pytest.raises(ValueError, match="plateau"):
        validate_continuous_config(with_plateau(enabled=False))
    with pytest.raises(ConfigError, match="cannot be below"):
        PlateauConfig(reset_learning_rate_scale=0.1, minimum_learning_rate_scale=0.25)
    with pytest.raises(ConfigError, match="plateau policy settings are invalid"):
        PlateauConfig(minimum_learning_rate_scale=0.0)
    with pytest.raises(ConfigError, match="plateau booleans"):
        PlateauConfig(restore_scale_on_promotion="yes")  # type: ignore[arg-type]
    with pytest.raises(ConfigError, match="plateau booleans"):
        PlateauConfig(count_inconclusive_rejections=1)  # type: ignore[arg-type]
    assert PlateauConfig().count_inconclusive_rejections is False
    assert PlateauConfig().restore_learning_rate_scale == 1.0
    validate_continuous_config(
        with_plateau(action="reduce_lr_keep_weights", restore_learning_rate_scale=0.5)
    )
    with pytest.raises(ConfigError, match="restore_learning_rate_scale cannot"):
        PlateauConfig(restore_learning_rate_scale=0.1, minimum_learning_rate_scale=0.25)
    for invalid in (0.0, 1.5, True):
        with pytest.raises(ConfigError, match="plateau policy settings are invalid"):
            PlateauConfig(restore_learning_rate_scale=invalid)  # type: ignore[arg-type]
    validate_continuous_config(
        with_plateau(
            action="reduce_lr_keep_weights",
            count_inconclusive_rejections=True,
        )
    )


def test_plateau_candidate_cadence_fits_replay_lag() -> None:
    experiment = load_config(CONFIGS / "h100-8gpu.yaml")
    with pytest.raises(ConfigError, match="reset-triggering candidate"):
        replace(
            experiment,
            learner=replace(experiment.learner, candidate_interval=10_000),
        )
    with pytest.raises(ConfigError, match="reset-triggering candidate"):
        replace(
            experiment,
            learner=replace(
                experiment.learner,
                candidate_interval=1,
                candidate_interval_examples=10_000
                * experiment.train.per_rank_batch_size,
            ),
        )
    with pytest.raises(ConfigError, match="replay lag eligibility"):
        replace(
            experiment,
            learner=replace(
                experiment.learner,
                candidate_interval=20_000,
            ),
            orchestration=replace(
                experiment.orchestration,
                plateau=replace(
                    experiment.orchestration.plateau,
                    max_learner_champion_lag_steps=50_000,
                ),
            ),
        )


def test_optimizer_decay_groups_and_muon_selection() -> None:
    model = tiny_model()
    decay, no_decay = split_decay_parameters(model)
    no_decay_names = {name for name, _ in no_decay}
    assert "global_token" in no_decay_names
    assert all(parameter.ndim >= 2 for _, parameter in decay)
    assert any("edge_embedding" in name for name in no_decay_names)

    adamw = build_optimizer(model, OptimizerConfig(kind="adamw"))
    assert {float(group["weight_decay"]) for group in adamw.param_groups} == {
        0.0,
        0.01,
    }
    muon = build_optimizer(
        model,
        OptimizerConfig(kind="muon_adamw", min_muon_elements=1, muon_ns_steps=2),
    )
    assert isinstance(muon, MuonAdamW)
    assert any(group["algorithm"] == "muon" for group in muon.param_groups)
    assert any(
        group["algorithm"] == "adamw" and group["weight_decay"] == 0
        for group in muon.param_groups
    )


def test_compile_forwards_isolated_recompile_budget(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    model = tiny_model()
    calls: list[dict[str, object]] = []

    def compile_model(module, **options):
        assert module is model
        calls.append(options)
        return module

    monkeypatch.setattr(torch, "compile", compile_model)
    compiled = maybe_compile_model(
        model,
        enabled=True,
        dynamic=False,
        fullgraph=True,
        backend="eager",
        mode="reduce-overhead",
        recompile_limit=10,
        isolate_recompiles=True,
    )
    assert compiled is model
    assert calls == [
        {
            "dynamic": False,
            "fullgraph": True,
            "backend": "eager",
            "mode": "reduce-overhead",
            "recompile_limit": 10,
            "isolate_recompiles": True,
        }
    ]
    with pytest.raises(ValueError, match="recompile_limit"):
        maybe_compile_model(model, enabled=True, recompile_limit=0)
    with pytest.raises(ValueError, match="compile mode"):
        maybe_compile_model(model, enabled=True, mode="fastest")


def test_bf16_compiled_train_step_scheduler_and_checkpoint(tmp_path) -> None:
    torch.manual_seed(3)
    model = tiny_model()
    compiled = maybe_compile_model(
        model, enabled=True, dynamic=True, fullgraph=True, backend="eager"
    )
    optimizer = build_optimizer(model, OptimizerConfig(kind="adamw"))
    scheduler = build_scheduler(
        optimizer, SchedulerConfig(warmup_steps=1, total_steps=10)
    )
    ema = ExponentialMovingAverage(model, decay=0.9)
    result = train_step(
        compiled,
        collate_replay_samples([sample()]),
        optimizer,
        precision="bf16",
        gradient_clip_norm=0.5,
        scheduler=scheduler,
        ema=ema,
    )
    assert all(
        isinstance(value, torch.Tensor) for value in result.loss_tensors.values()
    )
    host_metrics = result.to_host()
    assert all(np.isfinite(value) for value in host_metrics.losses.values())
    assert np.isfinite(host_metrics.gradient_norm)
    assert result.losses == host_metrics.losses
    assert ema.num_updates == 1

    path = save_checkpoint(
        tmp_path / "checkpoint.pt",
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
        ema=ema,
        step=9,
        epoch=2,
        config={"schema_version": 3},
    )
    restored = tiny_model()
    restored_optimizer = build_optimizer(restored, OptimizerConfig(kind="adamw"))
    restored_scheduler = build_scheduler(
        restored_optimizer, SchedulerConfig(warmup_steps=1, total_steps=10)
    )
    restored_ema = ExponentialMovingAverage(restored, decay=0.9)
    metadata = load_checkpoint(
        path,
        model=restored,
        optimizer=restored_optimizer,
        scheduler=restored_scheduler,
        ema=restored_ema,
    )
    assert metadata["step"] == 9
    assert metadata["epoch"] == 2
    assert restored_scheduler.state_dict() == scheduler.state_dict()
    for expected, actual in zip(
        model.state_dict().values(), restored.state_dict().values(), strict=True
    ):
        torch.testing.assert_close(expected, actual)

    tampered_path = tmp_path / "tampered-checkpoint.pt"
    payload = torch.load(path, weights_only=True)
    payload["rules_hash"] = 1
    torch.save(payload, tampered_path)
    with pytest.raises(ValueError, match="rules hash"):
        load_checkpoint(tampered_path, model=tiny_model())

    old_path = tmp_path / "old-checkpoint.pt"
    payload = torch.load(path, weights_only=True)
    payload["version"] = 2
    torch.save(payload, old_path)
    with pytest.raises(ValueError, match="checkpoint version"):
        load_checkpoint(old_path, model=tiny_model())


def test_sampled_diagnostics_do_not_change_train_step_numerics() -> None:
    torch.manual_seed(29)
    baseline = tiny_model()
    instrumented = copy.deepcopy(baseline)
    baseline_optimizer = build_optimizer(
        baseline,
        OptimizerConfig(kind="adamw", adamw_lr=1e-3),
    )
    instrumented_optimizer = build_optimizer(
        instrumented,
        OptimizerConfig(kind="adamw", adamw_lr=1e-3),
    )
    scheduler_config = SchedulerConfig(warmup_steps=1, total_steps=10)
    baseline_scheduler = build_scheduler(baseline_optimizer, scheduler_config)
    instrumented_scheduler = build_scheduler(
        instrumented_optimizer,
        scheduler_config,
    )
    baseline_ema = ExponentialMovingAverage(baseline, decay=0.9)
    instrumented_ema = ExponentialMovingAverage(instrumented, decay=0.9)
    batch = collate_replay_samples([sample()])

    baseline_result = train_step(
        baseline,
        batch,
        baseline_optimizer,
        scheduler=baseline_scheduler,
        ema=baseline_ema,
    )
    instrumented_result = train_step(
        instrumented,
        batch,
        instrumented_optimizer,
        scheduler=instrumented_scheduler,
        ema=instrumented_ema,
        collect_diagnostics=True,
    )

    for name, value in baseline.state_dict().items():
        torch.testing.assert_close(
            value,
            instrumented.state_dict()[name],
            atol=0,
            rtol=0,
        )
    assert baseline_result.losses == {
        name: value
        for name, value in instrumented_result.losses.items()
        if not name.startswith("clinch_")
    }
    assert any(name.startswith("clinch_") for name in instrumented_result.loss_tensors)
    assert baseline_result.learning_rates == instrumented_result.learning_rates
    state_before_host_metrics = {
        name: value.detach().clone()
        for name, value in instrumented.state_dict().items()
    }
    host = instrumented_result.to_host()
    for name, value in instrumented.state_dict().items():
        torch.testing.assert_close(
            value,
            state_before_host_metrics[name],
            atol=0,
            rtol=0,
        )
    assert host.nonfinite_loss_count == 0
    assert host.nonfinite_gradient_count == 0
    assert host.gradient_pre_clip_norm == host.gradient_norm
    assert host.gradient_clip_coefficient is not None
    assert host.gradient_post_clip_norm == pytest.approx(
        host.gradient_norm * host.gradient_clip_coefficient
    )
    assert host.gradient_clip_threshold == 1.0
    assert host.gradient_clip_severity == pytest.approx(
        1.0 - host.gradient_clip_coefficient
    )
    assert host.gradient_clip_ratio == pytest.approx(host.gradient_norm)
    assert len(host.optimizer_groups) == len(instrumented_optimizer.param_groups)
    assert all(group.update_norm > 0 for group in host.optimizer_groups)
    assert host.scheduler is not None
    assert host.scheduler.age_steps == 1
    assert host.ema is not None
    assert host.ema.num_updates == 1
    assert host.ema.distance_norm > 0
    assert host.ema.effective_turnover == pytest.approx(0.1)


def test_gradient_clipping_metrics_cover_threshold_edges_and_nonfinite_values() -> None:
    def metrics(norm: float, *, threshold: float = 1.0):
        return TrainStepResult(
            loss_tensors={"total": torch.tensor(1.0)},
            gradient_norm_tensor=torch.tensor(norm),
            gradient_clipped_tensor=torch.tensor(norm > threshold),
            nonfinite_loss_count_tensor=torch.tensor(0),
            nonfinite_gradient_count_tensor=torch.tensor(0),
            learning_rates=(0.1,),
            gradient_clip_threshold=threshold,
        ).to_host()

    below = metrics(0.5)
    assert below.gradient_post_clip_norm == pytest.approx(0.5)
    assert below.gradient_clip_coefficient == 1.0
    assert below.gradient_clip_severity == 0.0
    assert below.gradient_clip_ratio == pytest.approx(0.5)

    above = metrics(2.0)
    expected_coefficient = 1.0 / (2.0 + 1e-6)
    assert above.gradient_post_clip_norm == pytest.approx(2.0 * expected_coefficient)
    assert above.gradient_clip_coefficient == pytest.approx(expected_coefficient)
    assert above.gradient_clip_severity == pytest.approx(1.0 - expected_coefficient)
    assert above.gradient_clip_ratio == pytest.approx(2.0)

    zero = metrics(0.0)
    assert zero.gradient_post_clip_norm == 0.0
    assert zero.gradient_clip_coefficient == 1.0
    assert zero.gradient_clip_severity == 0.0
    assert zero.gradient_clip_ratio == 0.0

    nonfinite = metrics(float("inf"))
    assert nonfinite.gradient_post_clip_norm is None
    assert nonfinite.gradient_clip_coefficient == 0.0
    assert nonfinite.gradient_clip_severity == 1.0
    assert nonfinite.gradient_clip_ratio is None


def test_train_metric_accumulator_reports_clipping_frequency_without_step_sync() -> (
    None
):
    def result(*, clipped: bool, loss_nonfinite: int = 0) -> TrainStepResult:
        return TrainStepResult(
            loss_tensors={"total": torch.tensor(1.0)},
            gradient_norm_tensor=torch.tensor(2.0),
            gradient_clipped_tensor=torch.tensor(clipped),
            nonfinite_loss_count_tensor=torch.tensor(loss_nonfinite),
            nonfinite_gradient_count_tensor=torch.tensor(0),
            learning_rates=(0.1,),
        )

    accumulator = TrainMetricAccumulator()
    accumulator.update(result(clipped=True))
    accumulator.update(result(clipped=False, loss_nonfinite=1))
    metrics = accumulator.to_host()

    assert metrics.steps == 2
    assert metrics.gradient_clipped_steps == 1
    assert metrics.gradient_clipping_frequency == pytest.approx(0.5)
    assert metrics.nonfinite_loss_count == 1
    assert metrics.nonfinite_gradient_count == 0
    accumulator.reset()
    assert accumulator.to_host().steps == 0


def test_public_train_step_validates_untrusted_target_weights() -> None:
    model = tiny_model()
    optimizer = build_optimizer(model, OptimizerConfig(kind="adamw"))
    batch = collate_replay_samples([sample()])
    assert batch.targets.policy_weight is not None
    batch.targets.policy_weight[0] = -1
    before = {
        name: value.detach().clone() for name, value in model.state_dict().items()
    }

    with pytest.raises(ValueError, match="policy weights"):
        train_step(model, batch, optimizer)

    for name, value in model.state_dict().items():
        torch.testing.assert_close(value, before[name])


def test_train_step_fails_all_ranks_before_optimizer_mutation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    model = tiny_model()
    optimizer = build_optimizer(model, OptimizerConfig(kind="adamw"))
    batch = collate_replay_samples([sample()])
    reductions: list[torch.Tensor] = []
    monkeypatch.setattr(torch.distributed, "is_available", lambda: True)
    monkeypatch.setattr(torch.distributed, "is_initialized", lambda: True)

    def reject_one_rank(tensor: torch.Tensor, **_kwargs) -> None:
        reductions.append(tensor.clone())
        if tensor.dtype == torch.int32:
            tensor[0] = 0
        else:
            tensor[0] = 1

    monkeypatch.setattr(torch.distributed, "all_reduce", reject_one_rank)
    before = {
        name: value.detach().clone() for name, value in model.state_dict().items()
    }

    with pytest.raises(
        NonFiniteTrainingError,
        match="at least one rank",
    ) as captured:
        train_step(model, batch, optimizer)

    assert captured.value.nonfinite_loss_count == 1
    assert captured.value.nonfinite_gradient_count == 0
    assert len(reductions) == 2
    for name, value in model.state_dict().items():
        torch.testing.assert_close(value, before[name])


def test_device_prefetcher_cpu_path_preserves_batches() -> None:
    batches = [collate_replay_samples([sample()]) for _ in range(3)]

    observed = list(DeviceBatchPrefetcher(batches, device="cpu"))

    assert [batch.feature_path for batch in observed] == [
        batch.feature_path for batch in batches
    ]
    for expected, actual in zip(batches, observed, strict=True):
        torch.testing.assert_close(
            expected.inputs.node_features, actual.inputs.node_features
        )
        torch.testing.assert_close(expected.targets.policy, actual.targets.policy)


def test_prefetch_transfer_reuses_homogeneous_topology_and_handles_mixed_rings() -> (
    None
):
    prefetcher = object.__new__(DeviceBatchPrefetcher)
    prefetcher.device = torch.device("cpu")
    prefetcher._topology_cache = {}

    homogeneous = collate_replay_samples([sample(4), sample(4)])
    first = prefetcher._to_device(homogeneous)
    second = prefetcher._to_device(homogeneous)

    assert len(prefetcher._topology_cache) == 1
    assert (
        first.inputs.neighbor_index.data_ptr()
        == second.inputs.neighbor_index.data_ptr()
    )
    assert (
        first.inputs.neighbor_mask.data_ptr() == second.inputs.neighbor_mask.data_ptr()
    )
    assert (
        first.inputs.neighbor_edge_type.data_ptr()
        == second.inputs.neighbor_edge_type.data_ptr()
    )
    assert first.inputs.node_mask.data_ptr() == second.inputs.node_mask.data_ptr()
    torch.testing.assert_close(first.targets.policy, homogeneous.targets.policy)

    mixed = collate_replay_samples([sample(4), sample(6)])
    transferred = prefetcher._to_device(mixed)
    assert transferred.inputs.rings.tolist() == [4, 6]
    assert transferred.inputs.max_nodes == get_topology(6).n
    assert len(prefetcher._topology_cache) == 1


def test_async_prefetch_hands_off_events_and_exhausts_cleanly(monkeypatch) -> None:
    class FakeEvent:
        def __init__(self, *, enable_timing: bool) -> None:
            assert enable_timing
            self.recorded_on: list[object] = []

        def record(self, stream: object) -> None:
            self.recorded_on.append(stream)

        def elapsed_time(self, _completed: object) -> float:
            return 2.5

    class FakeCurrentStream:
        def __init__(self) -> None:
            self.waited_for: list[object] = []

        def wait_stream(self, stream: object) -> None:
            self.waited_for.append(stream)

    fake_copy_stream = object()
    current_stream = FakeCurrentStream()
    monkeypatch.setattr(torch.cuda, "Event", FakeEvent)
    monkeypatch.setattr(torch.cuda, "stream", lambda stream: nullcontext(stream))
    monkeypatch.setattr(
        torch.cuda,
        "current_stream",
        lambda _device: current_stream,
    )
    recorded_batches: list[ReplayBatch] = []
    monkeypatch.setattr(
        ReplayBatch,
        "record_stream",
        lambda batch, _stream: recorded_batches.append(batch),
    )
    monkeypatch.setattr(
        DeviceBatchPrefetcher,
        "_to_device",
        lambda _prefetcher, source: source,
    )

    sources = [
        collate_replay_samples([sample()]),
        collate_replay_samples([sample()]),
    ]
    prefetcher = object.__new__(DeviceBatchPrefetcher)
    prefetcher._batches = iter(sources)
    prefetcher.device = torch.device("cpu")
    prefetcher._stream = fake_copy_stream
    prefetcher._consumed_copy_events = []
    prefetcher._next_copy_event = None
    prefetcher._topology_cache = {}
    prefetcher._next_batch = None
    prefetcher._next_source = None
    prefetcher._preload()

    assert next(prefetcher) is sources[0]
    assert next(prefetcher) is sources[1]
    with pytest.raises(StopIteration):
        next(prefetcher)
    assert recorded_batches == sources
    assert current_stream.waited_for == [fake_copy_stream, fake_copy_stream]
    assert prefetcher.pop_copy_seconds() == pytest.approx(0.005)
    assert prefetcher.pop_copy_events() == []


def test_training_rejects_invalid_step_controls_and_unwraps_nested_models() -> None:
    model = tiny_model()
    wrapper = torch.nn.Module()
    wrapper.module = model
    outer = torch.nn.Module()
    outer._orig_mod = wrapper
    assert unwrap_model(outer) is model

    optimizer = build_optimizer(model, OptimizerConfig(kind="adamw"))
    batch = collate_replay_samples([sample()])
    with pytest.raises(ValueError, match="precision"):
        train_step(model, batch, optimizer, precision="fp16")
    with pytest.raises(ValueError, match="gradient_clip_norm"):
        train_step(model, batch, optimizer, gradient_clip_norm=0)


@pytest.mark.cuda
def test_cuda_prefetcher_reuses_ring_topology_and_tracks_copy_time() -> None:
    sources = [
        collate_replay_samples([sample(), sample()]).pin_memory() for _ in range(3)
    ]
    assert sources[0].inputs.node_features.is_pinned()
    assert sources[0].inputs.legal_action_mask.is_pinned()
    assert not sources[0].inputs.neighbor_index.is_pinned()
    assert not sources[0].inputs.node_mask.is_pinned()
    prefetcher = DeviceBatchPrefetcher(sources, device="cuda")

    first = next(prefetcher)
    second = next(prefetcher)
    torch.cuda.synchronize()
    consumed_events = prefetcher.pop_copy_events()

    assert first.inputs.node_features.is_cuda
    assert second.targets.policy.is_cuda
    assert (
        first.inputs.neighbor_index.data_ptr()
        == second.inputs.neighbor_index.data_ptr()
    )
    assert first.inputs.node_mask.data_ptr() == second.inputs.node_mask.data_ptr()
    assert len(consumed_events) == 2
    assert sum(start.elapsed_time(end) for start, end in consumed_events) / 1_000 >= 0
    next(prefetcher)
    torch.cuda.synchronize()
    assert len(prefetcher.pop_copy_events()) == 1


def test_ema_foreach_update_matches_exact_lerp() -> None:
    model = tiny_model()
    ema = ExponentialMovingAverage(model, decay=0.75)
    before = {name: value.clone() for name, value in ema.shadow.items()}
    with torch.no_grad():
        for value in model.parameters():
            value.add_(0.125)

    ema.update(model)

    state = model.state_dict()
    assert ema.num_updates == 1
    for name, average in ema.shadow.items():
        expected = before[name].lerp(
            state[name].detach().to(dtype=average.dtype),
            0.25,
        )
        torch.testing.assert_close(average, expected)


def test_ring_stratified_sampler_remains_balanced() -> None:
    rings = [4] * 8 + [10] * 2
    sampler = RingStratifiedSampler(rings, num_samples=20, seed=11)
    sampled = [rings[index] for index in sampler]
    assert sampled.count(4) == sampled.count(10) == 10


def test_onnx_parity_when_runtime_is_available(tmp_path) -> None:
    runtime = pytest.importorskip("onnxruntime")
    model = tiny_model().eval()
    position = sample().to_position()
    batch = encode_batch([position])
    path = export_onnx(model, batch, tmp_path / "model.onnx")
    session = runtime.InferenceSession(str(path), providers=["CPUExecutionProvider"])
    topology4 = get_topology(4)
    position4 = DoubleStarPosition(
        rings=4,
        stones=torch.full((topology4.n,), -1, dtype=torch.int8),
        to_move=0,
        moves_left=2,
        opening=False,
        terminal=False,
    )
    for inference_batch in (batch, encode_batch([position4])):
        feed = {
            name: tensor.detach().cpu().numpy()
            for name, tensor in zip(
                ONNX_INPUT_NAMES, inference_batch.model_args(), strict=True
            )
        }
        actual = session.run(None, feed)
        with torch.no_grad():
            expected = ONNXStarModel(model)(*inference_batch.model_args())
        for expected_tensor, actual_array in zip(expected, actual, strict=True):
            np.testing.assert_allclose(
                expected_tensor.numpy(), actual_array, atol=2e-4, rtol=2e-4
            )
