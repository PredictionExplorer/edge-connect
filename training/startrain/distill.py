"""Replay/teacher distillation into a smaller all-size browser model."""

from __future__ import annotations

import hashlib
import math
import os
import random
import re
import tempfile
import time
from dataclasses import asdict, dataclass, fields, replace
from pathlib import Path
from typing import Any, TypeVar, cast

import torch
import torch.nn.functional as functional
import yaml
from torch import Tensor

from .checkpoint import (
    ExponentialMovingAverage,
    load_ema_checkpoint,
    load_model_manifest,
    save_checkpoint,
)
from .config import CONFIG_SCHEMA_VERSION, GameConfig, load_config
from .contracts import (
    ACTION_LAYOUT_SCHEMA_ID,
    EXTERNAL_FEATURE_SCHEMA_ID,
    FEATURE_SCHEMA_HASH,
    FEATURE_SCHEMA_VERSION,
    RULES_HASH_WIRE,
    RULES_SCHEMA_ID,
)
from .export import ONNX_INPUT_NAMES, ONNX_OUTPUT_NAMES, export_onnx
from .features import (
    GLOBAL_FEATURE_DIM,
    NODE_FEATURE_DIM,
    DoubleStarPosition,
    encode_batch,
)
from .losses import LossWeights, compute_losses
from .model import MODEL_SCHEMA_VERSION, GraphResTNet, ModelConfig, StarModelOutput
from .replay import (
    ReplayBatch,
    ReplaySample,
    augment_sample,
    collate_replay_samples,
    read_replay_shard,
)
from .runtime import atomic_json
from .symmetry import deterministic_transform
from .topology import SUPPORTED_RINGS, get_topology

DISTILL_CONFIG_SCHEMA_VERSION = 2
BROWSER_MANIFEST_SCHEMA_VERSION = 2
BROWSER_MANIFEST_FORMAT = "startrain.browser-model"
_MODEL_VERSION = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$")
_T = TypeVar("_T")


class DistillationConfigError(ValueError):
    pass


@dataclass(frozen=True, slots=True)
class ReplaySourceConfig:
    sources: tuple[Path, ...]
    maximum_samples: int | None = None

    def __post_init__(self) -> None:
        if not self.sources:
            raise DistillationConfigError("at least one replay source is required")
        if self.maximum_samples is not None and (
            type(self.maximum_samples) is not int or self.maximum_samples <= 0
        ):
            raise DistillationConfigError("maximum_samples must be positive")


@dataclass(frozen=True, slots=True)
class TeacherConfig:
    experiment_config: Path | None = None
    model_manifest: Path | None = None

    def __post_init__(self) -> None:
        if (self.experiment_config is None) != (self.model_manifest is None):
            raise DistillationConfigError(
                "teacher experiment_config and model_manifest must be set together"
            )

    @property
    def enabled(self) -> bool:
        return self.model_manifest is not None


@dataclass(frozen=True, slots=True)
class DistillationTrainConfig:
    steps: int = 10_000
    batch_size: int = 64
    learning_rate: float = 3e-4
    weight_decay: float = 1e-2
    precision: str = "fp32"
    device: str = "cuda:0"
    seed: int = 17
    ema_decay: float = 0.999
    gradient_clip_norm: float = 1.0
    d5_augmentation: bool = True

    def __post_init__(self) -> None:
        if type(self.steps) is not int or self.steps <= 0:
            raise DistillationConfigError("distillation steps must be positive")
        if type(self.batch_size) is not int or self.batch_size <= 0:
            raise DistillationConfigError("distillation batch_size must be positive")
        if self.learning_rate <= 0 or self.weight_decay < 0:
            raise DistillationConfigError("optimizer settings are invalid")
        if self.precision not in ("fp32", "bf16"):
            raise DistillationConfigError("distillation precision must be fp32 or bf16")
        if not isinstance(self.device, str) or not self.device:
            raise DistillationConfigError("distillation device must be non-empty")
        if type(self.seed) is not int:
            raise DistillationConfigError("distillation seed must be an integer")
        if not 0 <= self.ema_decay < 1:
            raise DistillationConfigError("EMA decay must be in [0, 1)")
        if self.gradient_clip_norm <= 0:
            raise DistillationConfigError("gradient clip norm must be positive")
        if type(self.d5_augmentation) is not bool:
            raise DistillationConfigError("d5_augmentation must be boolean")


@dataclass(frozen=True, slots=True)
class DistillationLossConfig:
    policy: float = 1.0
    outcome: float = 1.0
    score_margin: float = 0.25
    ownership: float = 0.25
    alive: float = 0.1
    teacher_policy_kl: float = 0.0
    teacher_outcome_kl: float = 0.0
    teacher_score_margin_kl: float = 0.0
    teacher_ownership_kl: float = 0.0
    teacher_alive_kl: float = 0.0
    teacher_temperature: float = 2.0

    def __post_init__(self) -> None:
        values = (
            self.policy,
            self.outcome,
            self.score_margin,
            self.ownership,
            self.alive,
            self.teacher_policy_kl,
            self.teacher_outcome_kl,
            self.teacher_score_margin_kl,
            self.teacher_ownership_kl,
            self.teacher_alive_kl,
        )
        if any(not math.isfinite(value) or value < 0 for value in values):
            raise DistillationConfigError(
                "distillation loss weights must be non-negative"
            )
        if not math.isfinite(self.teacher_temperature) or self.teacher_temperature <= 0:
            raise DistillationConfigError("teacher_temperature must be positive")
        if not any(values):
            raise DistillationConfigError(
                "at least one distillation loss must be enabled"
            )

    @property
    def uses_teacher_logits(self) -> bool:
        return any(
            (
                self.teacher_policy_kl,
                self.teacher_outcome_kl,
                self.teacher_score_margin_kl,
                self.teacher_ownership_kl,
                self.teacher_alive_kl,
            )
        )

    def hard_target_weights(self) -> LossWeights:
        return LossWeights(
            policy=self.policy,
            outcome=self.outcome,
            score_margin=self.score_margin,
            ownership=self.ownership,
            alive=self.alive,
            soft_policy=0.0,
        )


@dataclass(frozen=True, slots=True)
class BrowserSearchConfig:
    simulations: int = 64
    max_considered: int = 16
    c_visit: float = 50.0
    c_scale: float = 1.0

    def __post_init__(self) -> None:
        if (
            type(self.simulations) is not int
            or self.simulations <= 0
            or type(self.max_considered) is not int
            or self.max_considered <= 0
            or self.c_visit <= 0
            or self.c_scale <= 0
        ):
            raise DistillationConfigError("recommended browser search is invalid")


@dataclass(frozen=True, slots=True)
class DistillationExportConfig:
    output_directory: Path
    model_version: str
    onnx_opset: int = 18
    recommended_search: BrowserSearchConfig = BrowserSearchConfig()

    def __post_init__(self) -> None:
        if not _MODEL_VERSION.fullmatch(self.model_version):
            raise DistillationConfigError(
                "model_version is not a safe artifact identifier"
            )
        if type(self.onnx_opset) is not int or self.onnx_opset < 18:
            raise DistillationConfigError("browser ONNX opset must be at least 18")


@dataclass(frozen=True, slots=True)
class DistillationConfig:
    replay: ReplaySourceConfig
    teacher: TeacherConfig
    student: ModelConfig
    train: DistillationTrainConfig
    loss: DistillationLossConfig
    export: DistillationExportConfig
    schema_version: int = DISTILL_CONFIG_SCHEMA_VERSION

    def __post_init__(self) -> None:
        if self.schema_version != DISTILL_CONFIG_SCHEMA_VERSION:
            raise DistillationConfigError("distillation schema_version must be 2")
        if (
            self.student.node_feature_dim != NODE_FEATURE_DIM
            or self.student.global_feature_dim != GLOBAL_FEATURE_DIM
        ):
            raise DistillationConfigError(
                "student feature dimensions must match the finalized feature schema"
            )
        if self.loss.uses_teacher_logits and not self.teacher.enabled:
            raise DistillationConfigError(
                "teacher-logit KL requires a teacher EMA manifest"
            )


@dataclass(frozen=True, slots=True)
class DistillationArtifacts:
    checkpoint: Path
    onnx: Path
    manifest: Path
    checkpoint_sha256: str
    onnx_sha256: str
    final_losses: dict[str, float]


def _mapping(name: str, value: object) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise DistillationConfigError(f"{name} must be a mapping")
    return dict(value)


def _construct(cls: type[_T], value: object) -> _T:
    values = _mapping(cls.__name__, value)
    allowed = {field.name for field in fields(cast(Any, cls))}
    unknown = set(values) - allowed
    if unknown:
        raise DistillationConfigError(f"unknown {cls.__name__} keys: {sorted(unknown)}")
    try:
        return cls(**values)
    except (TypeError, ValueError) as exc:
        raise DistillationConfigError(f"invalid {cls.__name__}: {exc}") from exc


def load_distillation_config(path: str | Path) -> DistillationConfig:
    source = Path(path)
    try:
        with source.open("r", encoding="utf-8") as stream:
            raw = yaml.safe_load(stream)
    except (OSError, yaml.YAMLError) as exc:
        raise DistillationConfigError(
            f"cannot read distillation configuration: {exc}"
        ) from exc
    values = _mapping("distillation configuration", raw)
    required = {
        "schema_version",
        "replay",
        "teacher",
        "student",
        "train",
        "loss",
        "export",
    }
    missing = required - set(values)
    unknown = set(values) - required
    if missing or unknown:
        raise DistillationConfigError(
            f"distillation configuration has missing keys {sorted(missing)} "
            f"and unknown keys {sorted(unknown)}"
        )
    root = source.resolve().parent

    replay_values = _mapping("replay", values["replay"])
    sources = replay_values.get("sources")
    if not isinstance(sources, list):
        raise DistillationConfigError("replay.sources must be a list")
    replay_values["sources"] = tuple(_resolve_path(root, item) for item in sources)

    teacher_values = _mapping("teacher", values["teacher"])
    for name in ("experiment_config", "model_manifest"):
        item = teacher_values.get(name)
        if item is not None:
            teacher_values[name] = _resolve_path(root, item)

    export_values = _mapping("export", values["export"])
    export_values["output_directory"] = _resolve_path(
        root, export_values.get("output_directory")
    )
    export_values["recommended_search"] = _construct(
        BrowserSearchConfig, export_values.get("recommended_search", {})
    )

    return DistillationConfig(
        schema_version=values["schema_version"],
        replay=_construct(ReplaySourceConfig, replay_values),
        teacher=_construct(TeacherConfig, teacher_values),
        student=_construct(ModelConfig, values["student"]),
        train=_construct(DistillationTrainConfig, values["train"]),
        loss=_construct(DistillationLossConfig, values["loss"]),
        export=_construct(DistillationExportConfig, export_values),
    )


def _resolve_path(root: Path, value: object) -> Path:
    if not isinstance(value, str) or not value:
        raise DistillationConfigError("artifact paths must be non-empty strings")
    path = Path(value)
    return (path if path.is_absolute() else root / path).resolve()


class DistillationRunner:
    def __init__(self, config: DistillationConfig) -> None:
        self.config = config

    def run(self) -> DistillationArtifacts:
        output = self.config.export.output_directory
        version = self.config.export.model_version
        checkpoint = output / f"{version}.pt"
        onnx_path = output / f"{version}.fp16.onnx"
        manifest_path = output / f"{version}.browser.json"
        existing = [
            path for path in (checkpoint, onnx_path, manifest_path) if path.exists()
        ]
        if existing:
            raise FileExistsError(
                "distillation artifacts are immutable; choose a new model_version: "
                + ", ".join(str(path) for path in existing)
            )
        samples = self._load_samples()
        train = self.config.train
        random.seed(train.seed)
        torch.manual_seed(train.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(train.seed)
        device = torch.device(train.device)
        if device.type == "cuda" and not torch.cuda.is_available():
            raise RuntimeError(
                f"configured CUDA device {train.device!r} is unavailable"
            )

        student = GraphResTNet(self.config.student).to(device)
        teacher, teacher_version = self._load_teacher(device)
        if (
            teacher is not None
            and student.parameter_count() >= teacher.parameter_count()
        ):
            raise ValueError("student architecture must be smaller than the teacher")

        optimizer = torch.optim.AdamW(
            student.parameters(),
            lr=train.learning_rate,
            weight_decay=train.weight_decay,
        )
        ema = ExponentialMovingAverage(student, decay=train.ema_decay)
        rng = random.Random(train.seed)
        final_losses: dict[str, float] = {}
        for step in range(train.steps):
            selected = [
                samples[rng.randrange(len(samples))] for _ in range(train.batch_size)
            ]
            if train.d5_augmentation:
                selected = [
                    augment_sample(
                        sample,
                        deterministic_transform(
                            seed=train.seed,
                            sample_index=index,
                            epoch=step,
                        ),
                    )
                    for index, sample in enumerate(selected)
                ]
            batch = collate_replay_samples(selected).to(device)
            final_losses = self._train_step(
                student,
                teacher,
                batch,
                optimizer,
                ema,
            )

        output.mkdir(parents=True, exist_ok=True)
        serialized = {
            "schema_version": CONFIG_SCHEMA_VERSION,
            "game": asdict(GameConfig()),
            "model": asdict(self.config.student),
            "distillation": {
                "train": asdict(self.config.train),
                "loss": asdict(self.config.loss),
                "teacher_model_version": teacher_version,
                "replay_samples": len(samples),
            },
        }
        save_checkpoint(
            checkpoint,
            model=student,
            optimizer=optimizer,
            ema=ema,
            step=train.steps,
            epoch=train.steps,
            config=serialized,
            extra={
                "model_version": version,
                "artifact_role": "browser-distilled",
                "teacher_model_version": teacher_version,
            },
        )

        export_model = GraphResTNet(self.config.student)
        export_model.load_state_dict(student.to("cpu").state_dict())
        ema.copy_to(export_model)
        export_model.half().eval()
        example = _browser_example_batch()
        _export_atomic_onnx(
            export_model,
            example,
            onnx_path,
            opset_version=self.config.export.onnx_opset,
        )
        validate_browser_onnx(onnx_path)

        checkpoint_sha256 = sha256_file(checkpoint)
        onnx_sha256 = sha256_file(onnx_path)
        manifest = self._browser_manifest(
            checkpoint=checkpoint,
            onnx_path=onnx_path,
            checkpoint_sha256=checkpoint_sha256,
            onnx_sha256=onnx_sha256,
            parameter_count=export_model.parameter_count(),
            replay_samples=len(samples),
            teacher_version=teacher_version,
        )
        atomic_json(manifest_path, manifest)
        return DistillationArtifacts(
            checkpoint=checkpoint,
            onnx=onnx_path,
            manifest=manifest_path,
            checkpoint_sha256=checkpoint_sha256,
            onnx_sha256=onnx_sha256,
            final_losses=final_losses,
        )

    def _load_samples(self) -> list[ReplaySample]:
        paths: list[Path] = []
        for source in self.config.replay.sources:
            if source.is_dir():
                paths.extend(sorted(source.glob("*.npz")))
            elif source.is_file():
                paths.append(source)
            else:
                raise FileNotFoundError(f"replay source does not exist: {source}")
        if not paths:
            raise ValueError("replay sources contain no .npz shards")
        samples: list[ReplaySample] = []
        for path in paths:
            samples.extend(read_replay_shard(path))
            maximum = self.config.replay.maximum_samples
            if maximum is not None and len(samples) >= maximum:
                samples = samples[:maximum]
                break
        if not samples:
            raise ValueError("replay sources contain no samples")
        return samples

    def _load_teacher(
        self, device: torch.device
    ) -> tuple[GraphResTNet | None, str | None]:
        teacher_config = self.config.teacher
        if not teacher_config.enabled:
            return None, None
        assert teacher_config.experiment_config is not None
        assert teacher_config.model_manifest is not None
        experiment = load_config(teacher_config.experiment_config)
        manifest = load_model_manifest(teacher_config.model_manifest)
        if manifest.role != "champion":
            raise ValueError("distillation teacher must be a champion pointer")
        teacher = GraphResTNet(experiment.model).to(device)
        metadata = load_ema_checkpoint(
            manifest.checkpoint,
            model=teacher,
            expected_model_config=asdict(experiment.model),
            expected_game_config=asdict(experiment.game),
            expected_run_id=manifest.run_id,
            expected_generation_family=manifest.generation_family,
            expected_sha256=manifest.checkpoint_sha256,
            expected_bytes=manifest.checkpoint_bytes,
            map_location=device,
        )
        if int(metadata["step"]) != manifest.model_step:
            raise ValueError("teacher manifest and EMA checkpoint identity disagree")
        teacher.eval()
        for parameter in teacher.parameters():
            parameter.requires_grad_(False)
        return teacher, manifest.model_version

    def _train_step(
        self,
        student: GraphResTNet,
        teacher: GraphResTNet | None,
        batch: ReplayBatch,
        optimizer: torch.optim.Optimizer,
        ema: ExponentialMovingAverage,
    ) -> dict[str, float]:
        train = self.config.train
        optimizer.zero_grad(set_to_none=True)
        autocast = train.precision == "bf16"
        with torch.autocast(
            device_type=next(student.parameters()).device.type,
            dtype=torch.bfloat16,
            enabled=autocast,
        ):
            student_output = student(*batch.inputs.model_args())
            hard = compute_losses(
                student_output,
                batch.targets,
                legal_action_mask=batch.inputs.legal_action_mask,
                node_mask=batch.inputs.node_mask,
                weights=self.config.loss.hard_target_weights(),
            )
            teacher_losses: dict[str, Tensor] = {}
            if teacher is not None and self.config.loss.uses_teacher_logits:
                with torch.no_grad():
                    teacher_output = teacher(*batch.inputs.model_args())
                teacher_losses = _teacher_kl_losses(
                    student_output,
                    teacher_output,
                    legal_action_mask=batch.inputs.legal_action_mask,
                    node_mask=batch.inputs.node_mask,
                    temperature=self.config.loss.teacher_temperature,
                )
            total = hard["total"]
            for name, weight in _teacher_weights(self.config.loss).items():
                if weight:
                    total = total + weight * teacher_losses[name]
        if not bool(torch.isfinite(total)):
            raise FloatingPointError("non-finite distillation loss")
        total.backward()
        gradient_norm = torch.nn.utils.clip_grad_norm_(
            student.parameters(),
            train.gradient_clip_norm,
            error_if_nonfinite=True,
        )
        optimizer.step()
        ema.update(student)
        output = {f"hard_{name}": float(value.detach()) for name, value in hard.items()}
        output.update(
            {
                f"teacher_{name}": float(value.detach())
                for name, value in teacher_losses.items()
            }
        )
        output["total"] = float(total.detach())
        output["gradient_norm"] = float(gradient_norm)
        return output

    def _browser_manifest(
        self,
        *,
        checkpoint: Path,
        onnx_path: Path,
        checkpoint_sha256: str,
        onnx_sha256: str,
        parameter_count: int,
        replay_samples: int,
        teacher_version: str | None,
    ) -> dict[str, object]:
        model = self.config.student
        return {
            "format": BROWSER_MANIFEST_FORMAT,
            "schema_version": BROWSER_MANIFEST_SCHEMA_VERSION,
            "model_version": self.config.export.model_version,
            "created_ns": time.time_ns(),
            "rules": {
                "schema_id": RULES_SCHEMA_ID,
                "hash": RULES_HASH_WIRE,
                "mode": "double",
                "pie_rule": False,
                "rings": list(SUPPORTED_RINGS),
            },
            "features": {
                "schema_id": EXTERNAL_FEATURE_SCHEMA_ID,
                "version": FEATURE_SCHEMA_VERSION,
                "hash": f"{FEATURE_SCHEMA_HASH:016x}",
                "node_feature_count": NODE_FEATURE_DIM,
                "global_feature_count": GLOBAL_FEATURE_DIM,
            },
            "actions": {"schema_id": ACTION_LAYOUT_SCHEMA_ID},
            "outcome": {
                "classes": ["loss", "win"],
                "value": "P(win)-P(loss)",
            },
            "architecture": {
                "name": "GraphResTNet",
                "schema_version": MODEL_SCHEMA_VERSION,
                "all_size": True,
                "parameter_count": parameter_count,
                "config": asdict(model),
            },
            "precision": "float16",
            "weights": "ema",
            "artifacts": {
                "onnx": {
                    **_artifact_entry(onnx_path, onnx_sha256),
                    "opset": self.config.export.onnx_opset,
                },
                "checkpoint": _artifact_entry(checkpoint, checkpoint_sha256),
            },
            "tensors": _browser_tensor_schema(model),
            "recommended_local_search": asdict(self.config.export.recommended_search),
            "training": {
                "steps": self.config.train.steps,
                "replay_samples": replay_samples,
                "teacher_model_version": teacher_version,
                "teacher_logit_kl": self.config.loss.uses_teacher_logits,
            },
        }


def _teacher_weights(config: DistillationLossConfig) -> dict[str, float]:
    return {
        "policy_kl": config.teacher_policy_kl,
        "outcome_kl": config.teacher_outcome_kl,
        "score_margin_kl": config.teacher_score_margin_kl,
        "ownership_kl": config.teacher_ownership_kl,
        "alive_kl": config.teacher_alive_kl,
    }


def _teacher_kl_losses(
    student: StarModelOutput,
    teacher: StarModelOutput,
    *,
    legal_action_mask: Tensor,
    node_mask: Tensor,
    temperature: float,
) -> dict[str, Tensor]:
    return {
        "policy_kl": _categorical_kl(
            student.policy_logits,
            teacher.policy_logits,
            valid=legal_action_mask,
            temperature=temperature,
        ),
        "outcome_kl": _categorical_kl(
            student.outcome_logits,
            teacher.outcome_logits,
            valid=None,
            temperature=temperature,
        ),
        "score_margin_kl": _categorical_kl(
            student.score_margin_logits,
            teacher.score_margin_logits,
            valid=None,
            temperature=temperature,
        ),
        "ownership_kl": _categorical_kl(
            student.ownership_logits,
            teacher.ownership_logits,
            valid=node_mask.unsqueeze(-1),
            temperature=temperature,
        ),
        "alive_kl": _bernoulli_kl(
            student.alive_logits,
            teacher.alive_logits,
            valid=node_mask,
            temperature=temperature,
        ),
    }


def _categorical_kl(
    student: Tensor,
    teacher: Tensor,
    *,
    valid: Tensor | None,
    temperature: float,
) -> Tensor:
    if valid is not None:
        if valid.shape == student.shape:
            minimum = torch.finfo(student.dtype).min
            student = student.masked_fill(~valid, minimum)
            teacher = teacher.masked_fill(~valid, minimum)
        elif valid.shape != (*student.shape[:-1], 1):
            raise ValueError("teacher categorical KL mask shape is invalid")
    teacher_log = functional.log_softmax(teacher.float() / temperature, dim=-1)
    teacher_probability = teacher_log.exp()
    student_log = functional.log_softmax(student.float() / temperature, dim=-1)
    values = (teacher_probability * (teacher_log - student_log)).sum(dim=-1)
    if valid is not None and valid.shape == (*student.shape[:-1], 1):
        mask = valid.squeeze(-1)
        values = values[mask]
    elif valid is not None and valid.shape == student.shape:
        rows = valid.any(dim=-1)
        values = values[rows]
    if values.numel() == 0:
        return student.sum() * 0.0
    return values.mean() * (temperature * temperature)


def _bernoulli_kl(
    student: Tensor,
    teacher: Tensor,
    *,
    valid: Tensor,
    temperature: float,
) -> Tensor:
    teacher_probability = torch.sigmoid(teacher.float() / temperature).clamp(
        1e-6, 1 - 1e-6
    )
    student_probability = torch.sigmoid(student.float() / temperature).clamp(
        1e-6, 1 - 1e-6
    )
    values = teacher_probability * torch.log(
        teacher_probability / student_probability
    ) + (1 - teacher_probability) * torch.log(
        (1 - teacher_probability) / (1 - student_probability)
    )
    selected = values[valid]
    if selected.numel() == 0:
        return student.sum() * 0.0
    return selected.mean() * (temperature * temperature)


def _browser_example_batch() -> Any:
    topology = get_topology(4)
    position = DoubleStarPosition(
        rings=4,
        stones=torch.full((topology.n,), -1, dtype=torch.int8),
        to_move=0,
        moves_left=1,
        opening=True,
        terminal=False,
    )
    return encode_batch([position], dtype=torch.float16)


def _export_atomic_onnx(
    model: GraphResTNet,
    example: Any,
    destination: Path,
    *,
    opset_version: int,
) -> None:
    temporary_name: str | None = None
    try:
        with tempfile.NamedTemporaryFile(
            prefix=f".{destination.name}.",
            suffix=".tmp.onnx",
            dir=destination.parent,
            delete=False,
        ) as temporary:
            temporary_name = temporary.name
        export_onnx(
            model,
            example,
            temporary_name,
            opset_version=opset_version,
        )
        with Path(temporary_name).open("rb") as stream:
            os.fsync(stream.fileno())
        os.replace(temporary_name, destination)
    finally:
        if temporary_name is not None and os.path.exists(temporary_name):
            os.unlink(temporary_name)


def _artifact_entry(path: Path, checksum: str) -> dict[str, object]:
    return {
        "file": path.name,
        "sha256": checksum,
        "bytes": path.stat().st_size,
    }


def _browser_tensor_schema(model: ModelConfig) -> dict[str, object]:
    inputs = {
        ONNX_INPUT_NAMES[0]: {
            "dtype": "float16",
            "shape": ["batch", "nodes", model.node_feature_dim],
        },
        ONNX_INPUT_NAMES[1]: {
            "dtype": "float16",
            "shape": ["batch", model.global_feature_dim],
        },
        ONNX_INPUT_NAMES[2]: {"dtype": "int64", "shape": ["batch", "nodes", "degree"]},
        ONNX_INPUT_NAMES[3]: {"dtype": "bool", "shape": ["batch", "nodes", "degree"]},
        ONNX_INPUT_NAMES[4]: {"dtype": "int64", "shape": ["batch", "nodes", "degree"]},
        ONNX_INPUT_NAMES[5]: {"dtype": "bool", "shape": ["batch", "nodes"]},
        ONNX_INPUT_NAMES[6]: {"dtype": "bool", "shape": ["batch", "nodes"]},
    }
    outputs = {
        ONNX_OUTPUT_NAMES[0]: {"dtype": "float16", "shape": ["batch", "nodes"]},
        ONNX_OUTPUT_NAMES[1]: {"dtype": "float16", "shape": ["batch", 2]},
        ONNX_OUTPUT_NAMES[2]: {
            "dtype": "float16",
            "shape": ["batch", model.score_margin_bins],
        },
        ONNX_OUTPUT_NAMES[3]: {
            "dtype": "float16",
            "shape": ["batch", "nodes", 3],
        },
        ONNX_OUTPUT_NAMES[4]: {"dtype": "float16", "shape": ["batch", "nodes"]},
        ONNX_OUTPUT_NAMES[5]: {"dtype": "float16", "shape": ["batch", "nodes"]},
    }
    return {"inputs": inputs, "outputs": outputs}


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def validate_browser_onnx(path: str | Path) -> None:
    try:
        import onnx
    except (ImportError, ModuleNotFoundError) as exc:
        raise RuntimeError("browser export validation requires the onnx extra") from exc
    model = onnx.load(Path(path), load_external_data=True)
    onnx.checker.check_model(model, full_check=True)
    inputs = [value.name for value in model.graph.input]
    outputs = [value.name for value in model.graph.output]
    if inputs != list(ONNX_INPUT_NAMES) or outputs != list(ONNX_OUTPUT_NAMES):
        raise RuntimeError(
            "exported ONNX tensor names do not match the browser contract"
        )
    float16 = onnx.TensorProto.FLOAT16
    for value in (model.graph.input[0], model.graph.input[1], *model.graph.output):
        if value.type.tensor_type.elem_type != float16:
            raise RuntimeError("browser ONNX feature and output tensors must be FP16")


def distill_main(argv: list[str] | None = None) -> None:
    import argparse
    import json

    parser = argparse.ArgumentParser(
        description="Distill replay and EMA teacher logits into a browser model"
    )
    parser.add_argument("--config", required=True)
    parser.add_argument("--steps", type=int)
    parser.add_argument("--device")
    arguments = parser.parse_args(argv)
    config = load_distillation_config(arguments.config)
    train = config.train
    if arguments.steps is not None:
        train = replace(train, steps=arguments.steps)
    if arguments.device is not None:
        train = replace(train, device=arguments.device)
    artifacts = DistillationRunner(replace(config, train=train)).run()
    print(
        json.dumps(
            {
                "checkpoint": str(artifacts.checkpoint),
                "checkpoint_sha256": artifacts.checkpoint_sha256,
                "onnx": str(artifacts.onnx),
                "onnx_sha256": artifacts.onnx_sha256,
                "manifest": str(artifacts.manifest),
                "final_losses": artifacts.final_losses,
            },
            sort_keys=True,
        )
    )
