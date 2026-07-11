"""Strict YAML loading into typed training configuration."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field, fields
from pathlib import Path
from typing import Any, Literal, TypeVar, cast

import yaml

from .losses import LossWeights
from .model import ModelConfig
from .optim import OptimizerConfig
from .selfplay import SelfPlayConfig

CONFIG_SCHEMA_VERSION = 2
_T = TypeVar("_T")


class ConfigError(ValueError):
    pass


@dataclass(frozen=True, slots=True)
class GameConfig:
    mode: str = "double"
    pie_rule: bool = False
    min_rings: int = 3
    max_rings: int = 12

    def __post_init__(self) -> None:
        if (
            self.mode != "double"
            or self.pie_rule
            or self.min_rings != 3
            or self.max_rings != 12
        ):
            raise ConfigError("only no-pie Double *Star rings 3..12 is supported")


@dataclass(frozen=True, slots=True)
class SchedulerConfig:
    warmup_steps: int = 1_000
    total_steps: int = 100_000
    min_lr_ratio: float = 0.1

    def __post_init__(self) -> None:
        if self.warmup_steps < 0 or self.total_steps <= self.warmup_steps:
            raise ConfigError("scheduler steps are invalid")
        if not 0 <= self.min_lr_ratio <= 1:
            raise ConfigError("min_lr_ratio must be in [0, 1]")


@dataclass(frozen=True, slots=True)
class TrainConfig:
    per_rank_batch_size: int = 32
    precision: Literal["fp32", "bf16"] = "fp32"
    compile: bool = False
    seed: int = 17
    ema_decay: float = 0.999
    gradient_clip_norm: float = 1.0
    scheduler: SchedulerConfig = SchedulerConfig()

    def __post_init__(self) -> None:
        if self.per_rank_batch_size <= 0 or self.gradient_clip_norm <= 0:
            raise ConfigError(
                "per_rank_batch_size and gradient_clip_norm must be positive"
            )
        if self.precision not in ("fp32", "bf16"):
            raise ConfigError("precision must be fp32 or bf16")
        if not 0 <= self.ema_decay < 1:
            raise ConfigError("ema_decay must be in [0, 1)")

    def global_batch_size(self, world_size: int) -> int:
        if world_size <= 0:
            raise ValueError("world_size must be positive")
        return self.per_rank_batch_size * world_size


@dataclass(frozen=True, slots=True)
class DataConfig:
    schema_version: int = 3
    ring_stratified: bool = True
    d5_augmentation: bool = True
    workers: int = 0
    prefetch_factor: int = 2
    pin_memory: bool = False
    shard_cache_size: int = 2

    def __post_init__(self) -> None:
        if self.schema_version != 3:
            raise ConfigError("data schema_version must be 3")
        if self.workers < 0 or self.prefetch_factor <= 0 or self.shard_cache_size <= 0:
            raise ConfigError("invalid data-loader worker settings")


@dataclass(frozen=True, slots=True)
class LearnerConfig:
    steps: int = 10_000
    minimum_replay_samples: int = 1
    recent_samples_per_ring: int = 10_000
    minimum_unique_samples_per_ring: int = 1
    use_ring_mixture_curriculum: bool = False
    max_replay_lag_steps: int = 50_000
    steps_per_window: int = 100
    candidate_interval: int = 1_000
    metrics_interval: int = 10
    replay_poll_seconds: float = 2.0
    replay_wait_timeout_seconds: float = 0.0
    resume_latest: bool = True
    device: str = "cpu"

    def __post_init__(self) -> None:
        if type(self.use_ring_mixture_curriculum) is not bool:
            raise ConfigError("use_ring_mixture_curriculum must be boolean")
        if type(self.resume_latest) is not bool:
            raise ConfigError("resume_latest must be boolean")
        values = (
            self.steps,
            self.minimum_replay_samples,
            self.recent_samples_per_ring,
            self.minimum_unique_samples_per_ring,
            self.steps_per_window,
            self.candidate_interval,
            self.metrics_interval,
        )
        if any(value <= 0 for value in values) or self.max_replay_lag_steps < 0:
            raise ConfigError("learner loop intervals and windows are invalid")
        if self.minimum_unique_samples_per_ring > self.recent_samples_per_ring:
            raise ConfigError("per-ring minimum cannot exceed the recent quota")
        if self.replay_poll_seconds <= 0 or self.replay_wait_timeout_seconds < 0:
            raise ConfigError("learner replay wait settings are invalid")
        if not self.device:
            raise ConfigError("learner device must be non-empty")


@dataclass(frozen=True, slots=True)
class GPUWorkerConfig:
    """One physical GPU assignment visible to exactly one worker job."""

    gpu_id: int
    role: Literal["learner", "actor"]
    cpu_threads: int
    actor_batch_size: int | None = None

    def __post_init__(self) -> None:
        if (
            isinstance(self.gpu_id, bool)
            or not isinstance(self.gpu_id, int)
            or self.gpu_id < 0
        ):
            raise ConfigError("gpu_id must be non-negative")
        if self.role not in ("learner", "actor"):
            raise ConfigError("GPU role must be learner or actor")
        if (
            isinstance(self.cpu_threads, bool)
            or not isinstance(self.cpu_threads, int)
            or self.cpu_threads <= 0
        ):
            raise ConfigError("GPU cpu_threads must be positive")
        if self.role == "actor":
            if (
                isinstance(self.actor_batch_size, bool)
                or not isinstance(self.actor_batch_size, int)
                or self.actor_batch_size <= 0
            ):
                raise ConfigError("actor GPUs require a positive actor_batch_size")
        elif self.actor_batch_size is not None:
            raise ConfigError("learner GPUs cannot set actor_batch_size")


@dataclass(frozen=True, slots=True)
class CurriculumStage:
    until_samples: int
    rings: tuple[int, ...]

    def __post_init__(self) -> None:
        if (
            isinstance(self.until_samples, bool)
            or not isinstance(self.until_samples, int)
            or self.until_samples <= 0
        ):
            raise ConfigError("curriculum until_samples must be positive")
        if not self.rings or any(ring < 3 or ring > 12 for ring in self.rings):
            raise ConfigError("curriculum rings must be in 3..12")
        if len(set(self.rings)) != len(self.rings):
            raise ConfigError("curriculum rings must be unique")


@dataclass(frozen=True, slots=True)
class RingMixtureConfig:
    rings: tuple[int, ...] = tuple(range(3, 13))
    curriculum: tuple[CurriculumStage, ...] = (
        CurriculumStage(until_samples=100_000, rings=(3, 4)),
        CurriculumStage(until_samples=500_000, rings=(3, 4, 5, 6)),
    )
    uniform_weight: float = 1.0
    deficit_weights: tuple[float, ...] = (1.0,) * 10

    def __post_init__(self) -> None:
        if self.rings != tuple(range(3, 13)):
            raise ConfigError(
                "orchestration must cover every ring in 3..12 exactly once"
            )
        if self.uniform_weight <= 0:
            raise ConfigError("ring uniform_weight must be positive")
        if len(self.deficit_weights) != len(self.rings) or any(
            weight < 0 for weight in self.deficit_weights
        ):
            raise ConfigError(
                "ring deficit_weights must match rings and be non-negative"
            )
        previous = 0
        for stage in self.curriculum:
            if stage.until_samples <= previous:
                raise ConfigError(
                    "curriculum stages must have increasing sample boundaries"
                )
            if any(ring not in self.rings for ring in stage.rings):
                raise ConfigError("curriculum stage contains an unavailable ring")
            previous = stage.until_samples

    def active_rings(self, total_samples: int) -> tuple[int, ...]:
        """Return the curriculum rings active at an aggregate sample count."""

        if (
            isinstance(total_samples, bool)
            or not isinstance(total_samples, int)
            or total_samples < 0
        ):
            raise ValueError(
                "total ring-mixture samples must be a non-negative integer"
            )
        for stage in self.curriculum:
            if total_samples < stage.until_samples:
                return stage.rings
        return self.rings


@dataclass(frozen=True, slots=True)
class ModelRefreshConfig:
    manifest_poll_seconds: float = 2.0
    startup_timeout_seconds: float = 600.0
    refresh_only_between_batches: bool = True
    selfplay_source: Literal["champion", "candidate", "candidate_champion_mix"] = (
        "champion"
    )
    candidate_probability: float = 0.8

    def __post_init__(self) -> None:
        if type(self.refresh_only_between_batches) is not bool:
            raise ConfigError("refresh_only_between_batches must be boolean")
        if self.manifest_poll_seconds <= 0 or self.startup_timeout_seconds <= 0:
            raise ConfigError("model refresh intervals must be positive")
        if self.selfplay_source not in (
            "champion",
            "candidate",
            "candidate_champion_mix",
        ):
            raise ConfigError("selfplay_source is invalid")
        if not 0.0 <= self.candidate_probability <= 1.0:
            raise ConfigError("candidate_probability must be in [0, 1]")
        if not self.refresh_only_between_batches:
            raise ConfigError(
                "actors may refresh models only between complete game batches"
            )


@dataclass(frozen=True, slots=True)
class RestartPolicyConfig:
    max_restarts: int = 5
    initial_backoff_seconds: float = 2.0
    maximum_backoff_seconds: float = 60.0
    stable_reset_seconds: float = 300.0

    def __post_init__(self) -> None:
        if (
            isinstance(self.max_restarts, bool)
            or not isinstance(self.max_restarts, int)
            or self.max_restarts < 0
        ):
            raise ConfigError("max_restarts must be non-negative")
        if (
            self.initial_backoff_seconds <= 0
            or self.maximum_backoff_seconds < self.initial_backoff_seconds
            or self.stable_reset_seconds <= 0
        ):
            raise ConfigError("restart backoff settings are invalid")


@dataclass(frozen=True, slots=True)
class RunDirectoryConfig:
    root: str = "runs/h100"
    replay: str = "replay"
    learner: str = "learner"
    logs: str = "logs"
    status: str = "status"
    metrics: str = "metrics"

    def __post_init__(self) -> None:
        values = (
            self.root,
            self.replay,
            self.learner,
            self.logs,
            self.status,
            self.metrics,
        )
        if any(
            not isinstance(value, str)
            or not value
            or Path(value).is_absolute()
            or ".." in Path(value).parts
            for value in values[1:]
        ):
            raise ConfigError("run subdirectories must be non-empty relative paths")
        if not isinstance(self.root, str) or not self.root:
            raise ConfigError("run root must be non-empty")


@dataclass(frozen=True, slots=True)
class ShutdownConfig:
    monitor_interval_seconds: float = 1.0
    heartbeat_interval_seconds: float = 5.0
    stale_heartbeat_seconds: float = 120.0
    stall_timeout_seconds: float = 600.0
    terminate_grace_seconds: float = 30.0
    kill_grace_seconds: float = 5.0

    def __post_init__(self) -> None:
        if (
            min(
                self.monitor_interval_seconds,
                self.heartbeat_interval_seconds,
                self.stale_heartbeat_seconds,
                self.stall_timeout_seconds,
                self.terminate_grace_seconds,
                self.kill_grace_seconds,
            )
            <= 0
        ):
            raise ConfigError("shutdown and heartbeat intervals must be positive")
        if self.stale_heartbeat_seconds <= self.heartbeat_interval_seconds:
            raise ConfigError("stale heartbeat threshold must exceed its interval")
        if self.stall_timeout_seconds <= self.stale_heartbeat_seconds:
            raise ConfigError("stall timeout must exceed stale heartbeat timeout")


@dataclass(frozen=True, slots=True)
class DistributedConfig:
    enabled: bool = False
    backend: Literal["nccl", "gloo"] = "nccl"

    def __post_init__(self) -> None:
        if type(self.enabled) is not bool:
            raise ConfigError("distributed.enabled must be boolean")
        if self.backend not in ("nccl", "gloo"):
            raise ConfigError("distributed backend must be nccl or gloo")


@dataclass(frozen=True, slots=True)
class PromotionConfig:
    enabled: bool = False
    gpu_id: int = 0
    cpu_threads: int = 4
    poll_seconds: float = 10.0
    bootstrap_initial_champion: bool = False
    device: str = "cuda"
    pause_sharing_mode: bool = False
    pause_ready_timeout_seconds: float = 1_200.0
    pause_release_timeout_seconds: float = 120.0
    final_drain_timeout_seconds: float = 7_200.0

    def __post_init__(self) -> None:
        if (
            type(self.enabled) is not bool
            or type(self.bootstrap_initial_champion) is not bool
            or type(self.pause_sharing_mode) is not bool
        ):
            raise ConfigError("promotion booleans must be boolean")
        if (
            type(self.gpu_id) is not int
            or self.gpu_id < 0
            or type(self.cpu_threads) is not int
            or self.cpu_threads <= 0
            or self.poll_seconds <= 0
            or self.pause_ready_timeout_seconds <= 0
            or self.pause_release_timeout_seconds <= 0
            or self.final_drain_timeout_seconds <= 0
        ):
            raise ConfigError(
                "promotion GPU, CPU, poll, and pause timeout settings are invalid"
            )
        if not isinstance(self.device, str) or not self.device:
            raise ConfigError("promotion device must be non-empty")


@dataclass(frozen=True, slots=True)
class PlateauConfig:
    enabled: bool = False
    max_learner_champion_lag_steps: int = 20_000
    consecutive_terminal_rejections: int = 3
    action: Literal["pause", "reset_from_champion"] = "pause"
    poll_seconds: float = 10.0

    def __post_init__(self) -> None:
        if type(self.enabled) is not bool:
            raise ConfigError("plateau.enabled must be boolean")
        if (
            self.max_learner_champion_lag_steps < 0
            or self.consecutive_terminal_rejections <= 0
            or self.poll_seconds <= 0
            or self.action not in ("pause", "reset_from_champion")
        ):
            raise ConfigError("plateau policy settings are invalid")


@dataclass(frozen=True, slots=True)
class RetentionConfig:
    enabled: bool = False
    dry_run: bool = True
    replay_shards_per_ring: int = 2_000
    candidate_manifests: int = 20
    gc_interval_windows: int = 10

    def __post_init__(self) -> None:
        if type(self.enabled) is not bool or type(self.dry_run) is not bool:
            raise ConfigError("retention booleans must be boolean")
        if (
            min(
                self.replay_shards_per_ring,
                self.candidate_manifests,
                self.gc_interval_windows,
            )
            <= 0
        ):
            raise ConfigError("retention counts must be positive")


@dataclass(frozen=True, slots=True)
class OrchestrationConfig:
    enabled: bool = False
    run_id: str | None = None
    gpus: tuple[GPUWorkerConfig, ...] = ()
    actor_games_per_batch: int = 256
    ring_mixture: RingMixtureConfig = RingMixtureConfig()
    model_refresh: ModelRefreshConfig = ModelRefreshConfig()
    restart: RestartPolicyConfig = RestartPolicyConfig()
    directories: RunDirectoryConfig = RunDirectoryConfig()
    shutdown: ShutdownConfig = ShutdownConfig()
    distributed: DistributedConfig = DistributedConfig()
    promotion: PromotionConfig = PromotionConfig()
    plateau: PlateauConfig = PlateauConfig()
    retention: RetentionConfig = RetentionConfig()

    def __post_init__(self) -> None:
        if type(self.enabled) is not bool:
            raise ConfigError("orchestration.enabled must be boolean")
        if self.run_id is not None:
            from .runtime import validate_identifier

            validate_identifier("run_id", self.run_id)
        if (
            isinstance(self.actor_games_per_batch, bool)
            or not isinstance(self.actor_games_per_batch, int)
            or self.actor_games_per_batch <= 0
        ):
            raise ConfigError("actor_games_per_batch must be positive")
        ids = [gpu.gpu_id for gpu in self.gpus]
        if len(ids) != len(set(ids)):
            raise ConfigError("a physical GPU may have only one configured role")
        learners = [gpu for gpu in self.gpus if gpu.role == "learner"]
        actors = [gpu for gpu in self.gpus if gpu.role == "actor"]
        if any(
            gpu.actor_batch_size is not None
            and gpu.actor_batch_size > self.actor_games_per_batch
            for gpu in actors
        ):
            raise ConfigError("actor game batches cannot be smaller than GPU batches")
        if self.enabled and (not learners or not actors):
            raise ConfigError("enabled orchestration requires learner and actor GPUs")
        if self.enabled and not self.promotion.enabled:
            raise ConfigError("enabled orchestration requires promotion supervision")
        actor_ids = {gpu.gpu_id for gpu in actors}
        learner_ids = {gpu.gpu_id for gpu in learners}
        promotion_overlap = (
            self.promotion.enabled and self.promotion.gpu_id in actor_ids | learner_ids
        )
        if promotion_overlap and not self.promotion.pause_sharing_mode:
            raise ConfigError("promotion GPU overlap requires pause-sharing mode")
        if (
            self.promotion.enabled
            and self.promotion.pause_sharing_mode
            and not promotion_overlap
        ):
            raise ConfigError(
                "pause-sharing mode requires exactly one learner or actor GPU overlap"
            )
        if not self.distributed.enabled and len(learners) > 1:
            raise ConfigError("multiple learner GPUs require distributed.enabled")
        if self.distributed.enabled and len(learners) < 2:
            raise ConfigError("distributed training requires at least two learner GPUs")
        if self.distributed.enabled and len({gpu.cpu_threads for gpu in learners}) != 1:
            raise ConfigError(
                "distributed learner GPUs require equal per-rank CPU budgets"
            )

    @property
    def learner_gpus(self) -> tuple[GPUWorkerConfig, ...]:
        return tuple(gpu for gpu in self.gpus if gpu.role == "learner")

    @property
    def actor_gpus(self) -> tuple[GPUWorkerConfig, ...]:
        return tuple(gpu for gpu in self.gpus if gpu.role == "actor")


@dataclass(frozen=True, slots=True)
class ArenaConfig:
    rings: tuple[int, ...] = tuple(range(3, 13))
    pairs_per_ring: int = 20
    simulations: int = 1_024
    max_considered: int = 32
    c_visit: float = 50.0
    c_scale: float = 1.0
    seed: int = 17
    null_elo: float = 0.0
    alternative_elo: float = 35.0
    alpha: float = 0.05
    beta: float = 0.05
    regression_floor_elo: float = -100.0
    per_ring_regression_floor_elo: dict[int, float] = field(default_factory=dict)
    confidence: float = 0.95
    bootstrap_samples: int = 2_000
    unforced_opening_fraction: float = 0.2
    minimum_pairs_per_ring: int = 40
    max_pairs_per_ring: int = 200

    def __post_init__(self) -> None:
        if (
            not self.rings
            or tuple(sorted(set(self.rings))) != self.rings
            or any(ring < 3 or ring > 12 for ring in self.rings)
        ):
            raise ConfigError("arena rings must be a sorted unique subset of 3..12")
        if self.pairs_per_ring < 2 or self.simulations <= 0:
            raise ConfigError(
                "arena requires at least two pairs per ring and positive simulations"
            )
        if self.max_considered <= 0 or self.c_visit <= 0 or self.c_scale <= 0:
            raise ConfigError("arena search settings must be positive")
        if self.alternative_elo <= self.null_elo:
            raise ConfigError("arena alternative_elo must exceed null_elo")
        if not 0 < self.alpha < 1 or not 0 < self.beta < 1:
            raise ConfigError("arena alpha and beta must be in (0, 1)")
        if not 0 < self.confidence < 1:
            raise ConfigError("arena confidence must be in (0, 1)")
        if self.bootstrap_samples < 200:
            raise ConfigError("arena bootstrap_samples must be at least 200")
        if not 0 < self.unforced_opening_fraction < 1:
            raise ConfigError("arena must include forced and unforced opening pairs")
        if not (
            self.pairs_per_ring > 0
            and self.minimum_pairs_per_ring >= self.pairs_per_ring
            and self.max_pairs_per_ring >= self.minimum_pairs_per_ring
        ):
            raise ConfigError(
                "arena pair round/minimum/maximum settings are inconsistent"
            )
        if any(ring not in self.rings for ring in self.per_ring_regression_floor_elo):
            raise ConfigError("arena regression floor has an unknown ring")


@dataclass(frozen=True, slots=True)
class ExperimentConfig:
    schema_version: int
    game: GameConfig
    model: ModelConfig
    loss: LossWeights
    optimizer: OptimizerConfig
    train: TrainConfig
    data: DataConfig
    selfplay: SelfPlayConfig
    learner: LearnerConfig
    orchestration: OrchestrationConfig = OrchestrationConfig()
    arena: ArenaConfig = ArenaConfig()
    profile: Literal["continuous", "standalone-smoke"] = "standalone-smoke"

    def __post_init__(self) -> None:
        if self.profile not in ("continuous", "standalone-smoke"):
            raise ConfigError("experiment profile is invalid")
        if self.profile == "continuous" and not self.orchestration.enabled:
            raise ConfigError("continuous profile requires orchestration")
        if (
            self.orchestration.plateau.enabled
            and self.orchestration.plateau.max_learner_champion_lag_steps
            > self.learner.max_replay_lag_steps
        ):
            raise ConfigError(
                "plateau lag cannot exceed learner replay lag eligibility"
            )
        plateau = self.orchestration.plateau
        if plateau.enabled and plateau.action == "reset_from_champion":
            rejection_span = (
                self.learner.candidate_interval
                * plateau.consecutive_terminal_rejections
            )
            if rejection_span > self.learner.max_replay_lag_steps:
                raise ConfigError(
                    "candidate rejection span cannot exceed replay lag eligibility"
                )
            if plateau.max_learner_champion_lag_steps < rejection_span:
                raise ConfigError(
                    "plateau lag must allow every reset-triggering candidate"
                )

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


def _construct(cls: type[_T], values: object) -> _T:
    if not isinstance(values, dict):
        raise ConfigError(f"{cls.__name__} must be a mapping")
    allowed = {field.name for field in fields(cast(Any, cls))}
    unknown = set(values) - allowed
    if unknown:
        raise ConfigError(f"unknown {cls.__name__} keys: {sorted(unknown)}")
    try:
        return cls(**values)
    except (TypeError, ValueError) as exc:
        raise ConfigError(f"invalid {cls.__name__}: {exc}") from exc


def _mapping(name: str, values: object) -> dict[str, Any]:
    if not isinstance(values, dict):
        raise ConfigError(f"{name} must be a mapping")
    return dict(values)


def _curriculum_values(values: object) -> dict[str, Any]:
    output = _mapping("curriculum stage", values)
    output["rings"] = tuple(output.get("rings", ()))
    return output


def load_config(path: str | Path) -> ExperimentConfig:
    with Path(path).open("r", encoding="utf-8") as stream:
        raw = yaml.safe_load(stream)
    if not isinstance(raw, dict):
        raise ConfigError("configuration root must be a mapping")
    required = {
        "schema_version",
        "game",
        "model",
        "loss",
        "optimizer",
        "train",
        "data",
        "selfplay",
        "learner",
    }
    optional = {"orchestration", "arena", "profile"}
    missing = required - set(raw)
    unknown = set(raw) - required - optional
    if missing or unknown:
        raise ConfigError(
            f"configuration has missing keys {sorted(missing)} "
            f"and unknown keys {sorted(unknown)}"
        )
    if raw["schema_version"] != CONFIG_SCHEMA_VERSION:
        raise ConfigError("configuration schema_version must be 2")

    optimizer_values = _mapping("optimizer", raw["optimizer"])
    if "betas" in optimizer_values:
        optimizer_values["betas"] = tuple(optimizer_values["betas"])
    train_values = _mapping("train", raw["train"])
    train_values["scheduler"] = _construct(
        SchedulerConfig, train_values.get("scheduler", {})
    )
    orchestration_values = _mapping("orchestration", raw.get("orchestration", {}))
    orchestration_values["gpus"] = tuple(
        _construct(GPUWorkerConfig, value)
        for value in orchestration_values.get("gpus", ())
    )
    ring_values = _mapping("ring_mixture", orchestration_values.get("ring_mixture", {}))
    ring_values["rings"] = tuple(ring_values.get("rings", tuple(range(3, 13))))
    ring_values["deficit_weights"] = tuple(
        ring_values.get("deficit_weights", (1.0,) * 10)
    )
    ring_values["curriculum"] = tuple(
        _construct(CurriculumStage, _curriculum_values(value))
        for value in ring_values.get(
            "curriculum",
            (
                {"until_samples": 100_000, "rings": (3, 4)},
                {"until_samples": 500_000, "rings": (3, 4, 5, 6)},
            ),
        )
    )
    orchestration_values["ring_mixture"] = _construct(RingMixtureConfig, ring_values)
    for key, cls in (
        ("model_refresh", ModelRefreshConfig),
        ("restart", RestartPolicyConfig),
        ("directories", RunDirectoryConfig),
        ("shutdown", ShutdownConfig),
        ("distributed", DistributedConfig),
        ("promotion", PromotionConfig),
        ("plateau", PlateauConfig),
        ("retention", RetentionConfig),
    ):
        orchestration_values[key] = _construct(cls, orchestration_values.get(key, {}))
    arena_values = _mapping("arena", raw.get("arena", {}))
    arena_values["rings"] = tuple(arena_values.get("rings", tuple(range(3, 13))))
    arena_values["per_ring_regression_floor_elo"] = {
        int(ring): float(value)
        for ring, value in _mapping(
            "per_ring_regression_floor_elo",
            arena_values.get("per_ring_regression_floor_elo", {}),
        ).items()
    }
    return ExperimentConfig(
        schema_version=CONFIG_SCHEMA_VERSION,
        game=_construct(GameConfig, raw["game"]),
        model=_construct(ModelConfig, raw["model"]),
        loss=_construct(LossWeights, raw["loss"]),
        optimizer=_construct(OptimizerConfig, optimizer_values),
        train=_construct(TrainConfig, train_values),
        data=_construct(DataConfig, raw["data"]),
        selfplay=_construct(SelfPlayConfig, raw["selfplay"]),
        learner=_construct(LearnerConfig, raw["learner"]),
        orchestration=_construct(OrchestrationConfig, orchestration_values),
        arena=_construct(ArenaConfig, arena_values),
        profile=raw.get("profile", "standalone-smoke"),
    )
