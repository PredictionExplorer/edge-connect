#!/usr/bin/env python3
"""Run one bounded learner-only optimizer arm on immutable frozen replay."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import sqlite3
import time
from collections.abc import Iterator, Mapping, Sequence
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path

import torch
from torch import nn

from startrain.checkpoint import (
    ExponentialMovingAverage,
    ModelManifest,
    extract_verified_manifest_config,
    load_checkpoint,
    load_ema_checkpoint,
    load_ema_weights_for_warm_start,
    load_model_manifest,
    save_checkpoint,
    sha256_file,
    verify_file,
)
from startrain.config import ExperimentConfig, load_config
from startrain.contracts import FEATURE_SCHEMA_HASH, RULES_HASH, RULES_HASH_WIRE
from startrain.device import (
    enable_fast_math,
    resolve_device_string,
    resolve_precision,
    seed_all,
    synchronize_device,
)
from startrain.losses import compute_losses
from startrain.model import GraphResTNet
from startrain.optim import (
    OptimizerRoutingMetadata,
    build_optimizer,
    optimizer_routing_metadata,
)
from startrain.replay import (
    DecodedReplayShard,
    ReplayBatch,
    augment_sample,
    collate_replay_samples,
    decode_replay_shard,
)
from startrain.replay_store import MANIFEST_SCHEMA_VERSION
from startrain.runtime import atomic_json
from startrain.symmetry import deterministic_transform
from startrain.training import build_scheduler, maybe_compile_model, train_step

if __package__:
    from .prepare_elo_ablation import (
        RING10_OPTIMIZER_CALIBRATION_LABELS,
        RING10_OPTIMIZER_CALIBRATION_TREATMENTS,
    )
else:
    from prepare_elo_ablation import (  # type: ignore[no-redef]
        RING10_OPTIMIZER_CALIBRATION_LABELS,
        RING10_OPTIMIZER_CALIBRATION_TREATMENTS,
    )

FORMAT = "startrain.frozen-replay-optimizer-calibration"
SCHEMA_VERSION = 1
STATE_FORMAT = "startrain.frozen-replay-optimizer-calibration-state"
MAX_H100_HOURS_PER_ARM = 2.0
CONTROL_ARM = "ring10-optimizer-runtime-effective-control"
FOLLOW_ON_ARM = "ring10-optimizer-0.5x-effective-lr"


@dataclass(frozen=True, slots=True)
class CalibrationSettings:
    config: Path
    champion: Path
    replay_root: Path
    replay_cutoff: int
    output_dir: Path
    arm: str
    steps: int
    batch_size: int | None = None
    evaluation_batch_size: int = 64
    max_samples: int = 65_536
    holdout_fraction: float = 0.2
    seed: int = 17
    device: str | None = None
    budget_h100_hours: float = MAX_H100_HOURS_PER_ARM
    checkpoint_interval: int = 100
    stop_after_steps: int | None = None
    dry_run: bool = False


@dataclass(frozen=True, slots=True)
class ArtifactPin:
    path: Path
    sha256: str
    bytes: int

    def as_dict(self) -> dict[str, object]:
        return {
            "path": str(self.path),
            "sha256": self.sha256,
            "bytes": self.bytes,
        }


@dataclass(frozen=True, slots=True)
class ChampionPin:
    pointer: ArtifactPin
    manifest: ArtifactPin
    checkpoint: ArtifactPin
    model_identity: str
    model_step: int
    run_id: str
    generation_family: str
    loaded: ModelManifest

    def as_dict(self) -> dict[str, object]:
        return {
            "pointer": self.pointer.as_dict(),
            "manifest": self.manifest.as_dict(),
            "checkpoint": self.checkpoint.as_dict(),
            "model_identity": self.model_identity,
            "model_step": self.model_step,
            "run_id": self.run_id,
            "generation_family": self.generation_family,
            "weights": "ema",
        }


@dataclass(frozen=True, slots=True)
class FrozenShard:
    shard_id: int
    path: Path
    relative_path: str
    sample_count: int
    checksum_sha256: str
    model_step: int
    model_identity: str

    def hash_record(self) -> dict[str, object]:
        return {
            "id": self.shard_id,
            "relative_path": self.relative_path,
            "sample_count": self.sample_count,
            "checksum_sha256": self.checksum_sha256,
            "model_step": self.model_step,
            "model_identity": self.model_identity,
        }


@dataclass(frozen=True, slots=True)
class ReplayReference:
    shard: FrozenShard
    sample_index: int
    stable_id: str
    order_sha256: str

    def as_hash_record(self) -> dict[str, object]:
        return {
            "shard_id": self.shard.shard_id,
            "sample_index": self.sample_index,
            "stable_id": self.stable_id,
        }


@dataclass(frozen=True, slots=True)
class FrozenReplay:
    cutoff: int
    cutoff_sha256: str
    shards: tuple[FrozenShard, ...]
    train: tuple[ReplayReference, ...]
    holdout: tuple[ReplayReference, ...]
    train_sha256: str
    holdout_sha256: str
    partition_sha256: str
    decoded: Mapping[int, DecodedReplayShard]

    def partition_dict(self) -> dict[str, object]:
        return {
            "method": "bounded-latest-window-hash-order-exact-split-v1",
            "train_samples": len(self.train),
            "holdout_samples": len(self.holdout),
            "train_sha256": self.train_sha256,
            "holdout_sha256": self.holdout_sha256,
            "partition_sha256": self.partition_sha256,
            "disjoint": True,
        }


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--champion", required=True, type=Path)
    parser.add_argument("--replay-root", required=True, type=Path)
    parser.add_argument("--replay-cutoff", required=True, type=int)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument(
        "--arm",
        required=True,
        choices=RING10_OPTIMIZER_CALIBRATION_TREATMENTS,
    )
    parser.add_argument("--steps", required=True, type=int)
    parser.add_argument("--batch-size", type=int)
    parser.add_argument("--evaluation-batch-size", type=int, default=64)
    parser.add_argument("--max-samples", type=int, default=65_536)
    parser.add_argument("--holdout-fraction", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--device")
    parser.add_argument(
        "--budget-h100-hours",
        "--budget-hours",
        dest="budget_h100_hours",
        type=float,
        default=MAX_H100_HOURS_PER_ARM,
    )
    parser.add_argument("--checkpoint-interval", type=int, default=100)
    parser.add_argument(
        "--stop-after-steps",
        type=int,
        help="operational pause after this many new steps; the arm remains resumable",
    )
    parser.add_argument("--dry-run", action="store_true")
    return parser


def _settings(arguments: argparse.Namespace) -> CalibrationSettings:
    return CalibrationSettings(
        config=arguments.config.expanduser().resolve(),
        champion=arguments.champion.expanduser().resolve(),
        replay_root=arguments.replay_root.expanduser().resolve(),
        replay_cutoff=arguments.replay_cutoff,
        output_dir=arguments.output_dir.expanduser().resolve(),
        arm=arguments.arm,
        steps=arguments.steps,
        batch_size=arguments.batch_size,
        evaluation_batch_size=arguments.evaluation_batch_size,
        max_samples=arguments.max_samples,
        holdout_fraction=arguments.holdout_fraction,
        seed=arguments.seed,
        device=arguments.device,
        budget_h100_hours=arguments.budget_h100_hours,
        checkpoint_interval=arguments.checkpoint_interval,
        stop_after_steps=arguments.stop_after_steps,
        dry_run=arguments.dry_run,
    )


def _canonical(value: object) -> bytes:
    return json.dumps(
        value,
        allow_nan=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")


def _digest(value: object) -> str:
    return hashlib.sha256(_canonical(value)).hexdigest()


def _pin(path: Path) -> ArtifactPin:
    source = path.expanduser().resolve()
    if not source.is_file() or source.is_symlink():
        raise ValueError(f"artifact must be a regular non-symlink file: {source}")
    return ArtifactPin(source, sha256_file(source), source.stat().st_size)


def _verify_pin(pin: ArtifactPin) -> None:
    if pin.path.is_symlink():
        raise ValueError(f"pinned artifact became a symlink: {pin.path}")
    verify_file(
        pin.path,
        expected_sha256=pin.sha256,
        expected_bytes=pin.bytes,
    )


def _inside(path: Path, root: Path) -> bool:
    try:
        path.resolve().relative_to(root.resolve())
    except ValueError:
        return False
    return True


def validate_settings(settings: CalibrationSettings) -> None:
    if settings.arm not in RING10_OPTIMIZER_CALIBRATION_TREATMENTS:
        raise ValueError("arm is not in the frozen optimizer calibration suite")
    for name, path in (
        ("config", settings.config),
        ("champion", settings.champion),
    ):
        if not path.is_file():
            raise ValueError(f"{name} is not a file: {path}")
    if not settings.replay_root.is_dir():
        raise ValueError(f"replay root is not a directory: {settings.replay_root}")
    if settings.replay_cutoff <= 0:
        raise ValueError("replay cutoff must be positive")
    if settings.steps <= 0:
        raise ValueError("steps must be positive")
    if settings.batch_size is not None and settings.batch_size <= 0:
        raise ValueError("batch size must be positive")
    if settings.evaluation_batch_size <= 0:
        raise ValueError("evaluation batch size must be positive")
    if settings.max_samples < 3:
        raise ValueError("max samples must leave train and held-out data")
    if not 0 < settings.holdout_fraction < 1:
        raise ValueError("holdout fraction must be in (0, 1)")
    if settings.seed < 0:
        raise ValueError("seed must be non-negative")
    if (
        not math.isfinite(settings.budget_h100_hours)
        or settings.budget_h100_hours <= 0
        or settings.budget_h100_hours > MAX_H100_HOURS_PER_ARM
    ):
        raise ValueError("declared budget must be in (0, 2] H100-hours per arm")
    if settings.checkpoint_interval <= 0:
        raise ValueError("checkpoint interval must be positive")
    if settings.stop_after_steps is not None and settings.stop_after_steps <= 0:
        raise ValueError("stop-after-steps must be positive")
    if _inside(settings.output_dir, settings.replay_root):
        raise ValueError("output directory cannot be inside source replay")
    if settings.output_dir == settings.replay_root:
        raise ValueError("output directory cannot replace source replay")


def _validate_calibration_config(config: ExperimentConfig, arm: str) -> None:
    if config.orchestration.training_objective != "ring10_only":
        raise ValueError("optimizer calibration requires a ring10_only profile")
    if config.optimizer.kind != "muon_adamw":
        raise ValueError("optimizer calibration excludes AdamW-only profiles")
    if config.model.dropout != 0.0:
        raise ValueError("deterministic optimizer calibration requires zero dropout")
    expected_clip = {
        "ring10-optimizer-runtime-effective-control": 1.0,
        "ring10-optimizer-clip-norm-2": 2.0,
        "ring10-optimizer-clip-norm-5": 5.0,
        "ring10-optimizer-0.5x-effective-lr": 1.0,
    }[arm]
    if config.train.gradient_clip_norm != expected_clip:
        raise ValueError(f"{arm} has the wrong gradient clip norm")


def _champion_pin(path: Path, config: ExperimentConfig) -> ChampionPin:
    pointer = _pin(path)
    manifest = load_model_manifest(pointer.path)
    _verify_pin(pointer)
    artifact = (manifest.artifact_manifest or manifest.path).resolve()
    manifest_pin = _pin(artifact)
    checkpoint_pin = _pin(manifest.checkpoint)
    if (
        manifest_pin.sha256 != manifest.manifest_sha256
        or manifest_pin.bytes != manifest.manifest_bytes
        or checkpoint_pin.sha256 != manifest.checkpoint_sha256
        or checkpoint_pin.bytes != manifest.checkpoint_bytes
    ):
        raise ValueError("champion publication pins disagree")
    verified = extract_verified_manifest_config(manifest, require_ema=True)
    serialized = config.as_dict()
    if (
        verified.model_config != serialized["model"]
        or verified.game_config != serialized["game"]
    ):
        raise ValueError("champion architecture/game contract differs from config")
    return ChampionPin(
        pointer=pointer,
        manifest=manifest_pin,
        checkpoint=checkpoint_pin,
        model_identity=manifest.model_identity,
        model_step=manifest.model_step,
        run_id=manifest.run_id,
        generation_family=manifest.generation_family,
        loaded=manifest,
    )


@contextmanager
def open_replay_read_only(replay_root: Path) -> Iterator[sqlite3.Connection]:
    """Use both SQLite URI read-only mode and query_only enforcement."""

    manifest = replay_root.resolve() / "manifest.sqlite3"
    if not manifest.is_file() or manifest.is_symlink():
        raise ValueError(f"source replay manifest is unsafe: {manifest}")
    connection = sqlite3.connect(
        f"{manifest.as_uri()}?mode=ro",
        uri=True,
        timeout=30.0,
        isolation_level=None,
    )
    connection.row_factory = sqlite3.Row
    try:
        connection.execute("PRAGMA query_only=ON")
        connection.execute("BEGIN")
        yield connection
    finally:
        connection.close()


def _replay_metadata(connection: sqlite3.Connection) -> dict[str, str]:
    rows = connection.execute(
        """
        SELECT key, value
        FROM store_metadata
        WHERE key IN (
            'manifest_schema_version', 'rules_hash', 'feature_schema_hash'
        )
        """
    )
    metadata = {str(row["key"]): str(row["value"]) for row in rows}
    expected = {
        "manifest_schema_version": str(MANIFEST_SCHEMA_VERSION),
        "rules_hash": RULES_HASH_WIRE,
        "feature_schema_hash": f"{FEATURE_SCHEMA_HASH:016x}",
    }
    if metadata != expected:
        raise ValueError("source replay metadata is incompatible")
    return metadata


def _frozen_shard(
    replay_root: Path,
    row: sqlite3.Row,
) -> FrozenShard:
    relative = Path(str(row["relative_path"]))
    if relative.is_absolute():
        raise ValueError("source replay contains an absolute shard path")
    path = (replay_root / relative).resolve()
    if not _inside(path, replay_root):
        raise ValueError("source replay shard escapes its root")
    sample_count = int(row["sample_count"])
    checksum = str(row["checksum_sha256"])
    if (
        sample_count <= 0
        or len(checksum) != 64
        or any(character not in "0123456789abcdef" for character in checksum)
    ):
        raise ValueError("source replay shard metadata is invalid")
    return FrozenShard(
        shard_id=int(row["id"]),
        path=path,
        relative_path=str(relative),
        sample_count=sample_count,
        checksum_sha256=checksum,
        model_step=int(row["model_step"]),
        model_identity=str(row["model_identity"]),
    )


def _stable_reference(
    shard: FrozenShard, sample_index: int, seed: int
) -> ReplayReference:
    stable_id = f"{shard.shard_id}:{sample_index}:{shard.checksum_sha256}"
    order = hashlib.sha256(
        f"optimizer-calibration-v1:{seed}:{stable_id}".encode()
    ).hexdigest()
    return ReplayReference(shard, sample_index, stable_id, order)


def _partition_hash(references: Sequence[ReplayReference]) -> str:
    return _digest([reference.as_hash_record() for reference in references])


def freeze_replay(
    settings: CalibrationSettings,
    champion: ChampionPin,
    *,
    decode: bool,
) -> FrozenReplay:
    root = settings.replay_root.resolve()
    with open_replay_read_only(root) as connection:
        metadata = _replay_metadata(connection)
        rows = connection.execute(
            """
            SELECT id, relative_path, sample_count, checksum_sha256,
                   model_step, model_identity
            FROM shards
            WHERE state = 'ready'
              AND id <= ?
              AND ring = 10
              AND run_id = ?
              AND generation_family = ?
              AND rules_hash = ?
              AND feature_schema_hash = ?
            ORDER BY id ASC
            """,
            (
                settings.replay_cutoff,
                champion.run_id,
                champion.generation_family,
                f"{RULES_HASH:016x}",
                f"{FEATURE_SCHEMA_HASH:016x}",
            ),
        ).fetchall()
    shards = tuple(_frozen_shard(root, row) for row in rows)
    if not shards or shards[-1].shard_id != settings.replay_cutoff:
        raise ValueError(
            "replay cutoff must identify the latest eligible ready ring-10 shard"
        )
    cutoff_document = {
        "schema_version": 1,
        "metadata": metadata,
        "cutoff": settings.replay_cutoff,
        "run_id": champion.run_id,
        "generation_family": champion.generation_family,
        "shards": [shard.hash_record() for shard in shards],
    }
    cutoff_sha256 = _digest(cutoff_document)

    window: list[FrozenShard] = []
    capacity = 0
    for shard in reversed(shards):
        window.append(shard)
        capacity += shard.sample_count
        if capacity >= settings.max_samples:
            break
    references = [
        _stable_reference(shard, sample_index, settings.seed)
        for shard in reversed(window)
        for sample_index in range(shard.sample_count)
    ]
    references.sort(key=lambda reference: (reference.order_sha256, reference.stable_id))
    selected = references[: settings.max_samples]
    if len(selected) < 3:
        raise ValueError("frozen replay has fewer than three eligible samples")
    holdout_count = round(len(selected) * settings.holdout_fraction)
    holdout_count = max(1, min(len(selected) - 1, holdout_count))
    partition_order = sorted(
        selected,
        key=lambda reference: hashlib.sha256(
            f"holdout-v1:{settings.seed}:{reference.stable_id}".encode()
        ).hexdigest(),
    )
    holdout_ids = {reference.stable_id for reference in partition_order[:holdout_count]}
    train = tuple(
        sorted(
            (
                reference
                for reference in selected
                if reference.stable_id not in holdout_ids
            ),
            key=lambda reference: (reference.order_sha256, reference.stable_id),
        )
    )
    holdout = tuple(
        sorted(
            (reference for reference in selected if reference.stable_id in holdout_ids),
            key=lambda reference: (reference.order_sha256, reference.stable_id),
        )
    )
    if {reference.stable_id for reference in train} & holdout_ids:
        raise RuntimeError("frozen replay partitions overlap")
    batch_size = (
        settings.batch_size or load_config(settings.config).train.per_rank_batch_size
    )
    if len(train) < batch_size:
        raise ValueError("frozen train partition is smaller than one learner batch")
    train_sha256 = _partition_hash(train)
    holdout_sha256 = _partition_hash(holdout)
    partition_sha256 = _digest(
        {
            "method": "bounded-latest-window-hash-order-exact-split-v1",
            "train_sha256": train_sha256,
            "holdout_sha256": holdout_sha256,
            "cutoff_sha256": cutoff_sha256,
        }
    )

    decoded: dict[int, DecodedReplayShard] = {}
    selected_shards = {
        reference.shard.shard_id: reference.shard for reference in (*train, *holdout)
    }
    for shard in selected_shards.values():
        if not shard.path.is_file() or shard.path.is_symlink():
            raise ValueError(f"source replay shard is unsafe: {shard.path}")
        if sha256_file(shard.path) != shard.checksum_sha256:
            raise ValueError(f"source replay shard hash failed: {shard.path}")
        if decode:
            materialized = decode_replay_shard(shard.path)
            if len(materialized) != shard.sample_count:
                raise ValueError("source replay shard count disagrees with manifest")
            if int(materialized.arrays["rings"][0]) != 10:
                raise ValueError("optimizer calibration replay must contain ring 10")
            decoded[shard.shard_id] = materialized
    return FrozenReplay(
        cutoff=settings.replay_cutoff,
        cutoff_sha256=cutoff_sha256,
        shards=shards,
        train=train,
        holdout=holdout,
        train_sha256=train_sha256,
        holdout_sha256=holdout_sha256,
        partition_sha256=partition_sha256,
        decoded=decoded,
    )


def _materialize(
    replay: FrozenReplay,
    references: Sequence[ReplayReference],
    *,
    start: int,
    batch_size: int,
    seed: int,
    augment: bool,
) -> ReplayBatch:
    samples = []
    for offset in range(batch_size):
        position = start + offset
        reference = references[position % len(references)]
        decoded = replay.decoded.get(reference.shard.shard_id)
        if decoded is None:
            raise RuntimeError("selected replay shard was not decoded")
        sample = decoded.sample(reference.sample_index)
        if augment:
            sample = augment_sample(
                sample,
                deterministic_transform(
                    seed=seed,
                    sample_index=int(reference.order_sha256[:16], 16),
                    epoch=position // len(references),
                ),
            )
        samples.append(sample)
    return collate_replay_samples(samples)


def _config_contract(config: ExperimentConfig) -> dict[str, object]:
    serialized = config.as_dict()
    return {
        "model": serialized["model"],
        "game": serialized["game"],
        "loss": serialized["loss"],
        "optimizer": serialized["optimizer"],
        "train": {
            key: serialized["train"][key]
            for key in (
                "per_rank_batch_size",
                "precision",
                "compile",
                "ema_decay",
                "ema_half_life_examples",
                "gradient_clip_norm",
                "scheduler",
            )
        },
    }


def _run_contract(
    settings: CalibrationSettings,
    config: ExperimentConfig,
    config_pin: ArtifactPin,
    champion: ChampionPin,
    replay: FrozenReplay,
    *,
    batch_size: int,
) -> dict[str, object]:
    selected_shards = {
        reference.shard.shard_id: reference.shard
        for reference in (*replay.train, *replay.holdout)
    }
    return {
        "format": FORMAT,
        "schema_version": SCHEMA_VERSION,
        "arm": settings.arm,
        "label": RING10_OPTIMIZER_CALIBRATION_LABELS[settings.arm],
        "phase": "follow_on" if settings.arm == FOLLOW_ON_ARM else "primary",
        "config": config_pin.as_dict(),
        "config_contract": _config_contract(config),
        "champion": champion.as_dict(),
        "replay": {
            "root": str(settings.replay_root),
            "manifest_open_mode": "ro",
            "query_only": True,
            "cutoff": replay.cutoff,
            "cutoff_sha256": replay.cutoff_sha256,
            "eligible_shards": len(replay.shards),
            "selected_shards": [
                {
                    "id": shard.shard_id,
                    "path": str(shard.path),
                    "sha256": shard.checksum_sha256,
                    "bytes": shard.path.stat().st_size,
                }
                for shard in sorted(
                    selected_shards.values(),
                    key=lambda value: value.shard_id,
                )
            ],
        },
        "partition": replay.partition_dict(),
        "training": {
            "steps": settings.steps,
            "batch_size": batch_size,
            "seed": settings.seed,
            "budget_h100_hours": settings.budget_h100_hours,
            "h100_count": 1,
            "budget_enforced_against_total_arm_wall_clock": True,
            "fresh_optimizer": True,
            "fresh_scheduler": True,
            "initial_weights": "champion_ema",
        },
        "evaluation": {
            "batch_size": settings.evaluation_batch_size,
            "augmentation": False,
            "composite": (
                "policy_weight*policy + soft_policy_weight*soft_policy + "
                "outcome_weight*outcome + score_margin_weight*score_margin"
            ),
            "observation_unit": "deterministic-held-out-batch",
        },
    }


def _state_path(settings: CalibrationSettings) -> Path:
    return settings.output_dir / "state.json"


def _plan_path(settings: CalibrationSettings) -> Path:
    return settings.output_dir / "run-plan.json"


def _load_json(path: Path) -> dict[str, object]:
    loaded = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(loaded, dict):
        raise ValueError(f"{path} must contain a JSON object")
    return loaded


def _new_state(contract_sha256: str) -> dict[str, object]:
    return {
        "format": STATE_FORMAT,
        "schema_version": SCHEMA_VERSION,
        "status": "ready",
        "run_contract_sha256": contract_sha256,
        "measurement_started_ns": time.time_ns(),
        "completed_steps": 0,
        "examples_consumed": 0,
        "elapsed_seconds": 0.0,
        "training_seconds": 0.0,
        "loss_sums": {},
        "gradient_norm_sum": 0.0,
        "gradient_clipped_steps": 0,
        "nonfinite_loss_count": 0,
        "nonfinite_gradient_count": 0,
        "checkpoint": None,
        "first_step_optimizer_diagnostics": [],
    }


def _validated_state(
    settings: CalibrationSettings,
    contract_sha256: str,
) -> dict[str, object]:
    state_path = _state_path(settings)
    plan_path = _plan_path(settings)
    if not state_path.exists():
        if settings.output_dir.exists() and any(settings.output_dir.iterdir()):
            raise ValueError("output directory is non-empty but has no resumable state")
        settings.output_dir.mkdir(parents=True, exist_ok=True)
        return _new_state(contract_sha256)
    if not plan_path.is_file():
        raise ValueError("resumable calibration state has no frozen run plan")
    state = _load_json(state_path)
    plan = _load_json(plan_path)
    if (
        state.get("format") != STATE_FORMAT
        or state.get("schema_version") != SCHEMA_VERSION
        or state.get("run_contract_sha256") != contract_sha256
        or plan.get("semantic_sha256") != contract_sha256
    ):
        raise ValueError("resumable calibration state differs from requested run")
    return state


def _int_field(mapping: Mapping[str, object], name: str) -> int:
    value = mapping.get(name)
    if type(value) is not int or value < 0:
        raise ValueError(f"state {name} is invalid")
    return value


def _float_field(mapping: Mapping[str, object], name: str) -> float:
    value = mapping.get(name)
    if isinstance(value, bool) or not isinstance(value, int | float):
        raise ValueError(f"state {name} is invalid")
    result = float(value)
    if not math.isfinite(result) or result < 0:
        raise ValueError(f"state {name} is invalid")
    return result


def _checkpoint_pin_from_state(state: Mapping[str, object]) -> ArtifactPin | None:
    raw = state.get("checkpoint")
    if raw is None:
        return None
    if not isinstance(raw, Mapping):
        raise ValueError("state checkpoint pin is invalid")
    path = raw.get("path")
    digest = raw.get("sha256")
    size = raw.get("bytes")
    if (
        not isinstance(path, str)
        or not isinstance(digest, str)
        or type(size) is not int
    ):
        raise ValueError("state checkpoint pin is incomplete")
    pin = ArtifactPin(Path(path).resolve(), digest, size)
    _verify_pin(pin)
    return pin


def _runtime_optimizer_metadata(
    optimizer: torch.optim.Optimizer,
) -> OptimizerRoutingMetadata:
    metadata = optimizer_routing_metadata(optimizer)
    if (
        metadata.requested_kind != "muon_adamw"
        or metadata.implementation != "muon_adamw"
        or metadata.fallback_used
    ):
        raise ValueError("optimizer calibration requires runtime Muon+AdamW routing")
    return metadata


def _initialize_training_state(
    *,
    config: ExperimentConfig,
    champion: ChampionPin,
    device: torch.device,
    precision: str,
    compile_enabled: bool,
    state: Mapping[str, object],
    contract_sha256: str,
) -> tuple[
    GraphResTNet,
    nn.Module,
    torch.optim.Optimizer,
    torch.optim.lr_scheduler.LRScheduler,
    ExponentialMovingAverage,
    OptimizerRoutingMetadata,
]:
    seed_all(config.train.seed)
    model = GraphResTNet(config.model).to(device)
    ema = ExponentialMovingAverage(
        model,
        decay=config.train.resolved_ema_decay(1),
    )
    serialized = config.as_dict()
    checkpoint_pin = _checkpoint_pin_from_state(state)
    if checkpoint_pin is None:
        load_ema_weights_for_warm_start(
            champion.checkpoint.path,
            model=model,
            ema=ema,
            expected_model_config=serialized["model"],
            expected_game_config=serialized["game"],
            map_location=device,
            expected_run_id=champion.run_id,
            expected_generation_family=champion.generation_family,
            expected_sha256=champion.checkpoint.sha256,
            expected_bytes=champion.checkpoint.bytes,
        )
        optimizer = build_optimizer(model, config.optimizer)
        scheduler = build_scheduler(optimizer, config.train.scheduler)
    else:
        optimizer = build_optimizer(model, config.optimizer)
        scheduler = build_scheduler(optimizer, config.train.scheduler)
        metadata = load_checkpoint(
            checkpoint_pin.path,
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            ema=ema,
            map_location=device,
            expected_model_config=serialized["model"],
            expected_game_config=serialized["game"],
            expected_sha256=checkpoint_pin.sha256,
            expected_bytes=checkpoint_pin.bytes,
        )
        extra = metadata.get("extra")
        if (
            not isinstance(extra, Mapping)
            or extra.get("run_contract_sha256") != contract_sha256
            or int(metadata["step"]) != _int_field(state, "completed_steps")
        ):
            raise ValueError("resume checkpoint does not match calibration state")
    routing = _runtime_optimizer_metadata(optimizer)
    compiled = maybe_compile_model(
        model,
        enabled=compile_enabled,
        dynamic=False,
        fullgraph=True,
    )
    compiled.train()
    return model, compiled, optimizer, scheduler, ema, routing


def _save_progress(
    *,
    settings: CalibrationSettings,
    config: ExperimentConfig,
    model: GraphResTNet,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LRScheduler,
    ema: ExponentialMovingAverage,
    state: dict[str, object],
    contract_sha256: str,
) -> ArtifactPin:
    step = _int_field(state, "completed_steps")
    checkpoint = settings.output_dir / "checkpoints" / f"step-{step:012d}.pt"
    if not checkpoint.exists():
        save_checkpoint(
            checkpoint,
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            ema=ema,
            step=step,
            config=config.as_dict(),
            extra={
                "run_contract_sha256": contract_sha256,
                "arm": settings.arm,
                "source_champion_sha256": sha256_file(settings.champion),
            },
        )
    pin = _pin(checkpoint)
    state["checkpoint"] = pin.as_dict()
    atomic_json(_state_path(settings), state)
    return pin


def _component_losses(
    model: nn.Module,
    batch: ReplayBatch,
    config: ExperimentConfig,
    *,
    device: torch.device,
    precision: str,
) -> dict[str, float]:
    moved = batch.to(device)
    with (
        torch.no_grad(),
        torch.autocast(
            device_type=device.type,
            dtype=torch.bfloat16,
            enabled=precision == "bf16",
        ),
    ):
        output = model(*moved.inputs.model_args())
        losses = compute_losses(
            output,
            moved.targets,
            legal_action_mask=moved.inputs.legal_action_mask,
            node_mask=moved.inputs.node_mask,
            weights=config.loss,
            validate_targets=False,
        )
    host = {name: float(value.detach().float().cpu()) for name, value in losses.items()}
    if any(not math.isfinite(value) for value in host.values()):
        raise FloatingPointError("held-out evaluation produced non-finite loss")
    weights = config.loss
    policy = weights.policy * host["policy"] + weights.soft_policy * host["soft_policy"]
    value = (
        weights.outcome * host["outcome"] + weights.score_margin * host["score_margin"]
    )
    return {
        "policy": policy,
        "value": value,
        "composite": policy + value,
    }


def _evaluate_holdout(
    *,
    config: ExperimentConfig,
    champion: ChampionPin,
    replay: FrozenReplay,
    model: GraphResTNet,
    ema: ExponentialMovingAverage,
    device: torch.device,
    precision: str,
    batch_size: int,
) -> dict[str, object]:
    serialized = config.as_dict()
    reference = GraphResTNet(config.model).to(device)
    load_ema_checkpoint(
        champion.checkpoint.path,
        model=reference,
        expected_model_config=serialized["model"],
        expected_game_config=serialized["game"],
        map_location=device,
        expected_run_id=champion.run_id,
        expected_generation_family=champion.generation_family,
        expected_sha256=champion.checkpoint.sha256,
        expected_bytes=champion.checkpoint.bytes,
    )
    candidate = GraphResTNet(config.model).to(device)
    candidate.load_state_dict(model.state_dict())
    ema.copy_to(candidate)
    reference.eval()
    candidate.eval()

    observations: list[dict[str, object]] = []
    for start in range(0, len(replay.holdout), batch_size):
        count = min(batch_size, len(replay.holdout) - start)
        batch = _materialize(
            replay,
            replay.holdout,
            start=start,
            batch_size=count,
            seed=config.train.seed,
            augment=False,
        )
        reference_losses = _component_losses(
            reference,
            batch,
            config,
            device=device,
            precision=precision,
        )
        candidate_losses = _component_losses(
            candidate,
            batch,
            config,
            device=device,
            precision=precision,
        )
        observations.append(
            {
                "index": len(observations),
                "samples": count,
                "reference": reference_losses,
                "candidate": candidate_losses,
                "composite_improvement": (
                    reference_losses["composite"] - candidate_losses["composite"]
                ),
            }
        )
    if not observations:
        raise ValueError("held-out partition produced no evaluation observations")

    def observation_samples(observation: Mapping[str, object]) -> int:
        value = observation.get("samples")
        if type(value) is not int or value <= 0:
            raise ValueError("held-out observation sample count is invalid")
        return value

    total_samples = sum(
        observation_samples(observation) for observation in observations
    )

    def observation_component(
        observation: Mapping[str, object],
        side: str,
        component: str,
    ) -> float:
        value = cast_mapping(observation[side]).get(component)
        if (
            isinstance(value, bool)
            or not isinstance(value, int | float)
            or not math.isfinite(float(value))
        ):
            raise ValueError("held-out observation component is invalid")
        return float(value)

    def weighted(side: str, component: str) -> float:
        return (
            sum(
                observation_samples(observation)
                * observation_component(observation, side, component)
                for observation in observations
            )
            / total_samples
        )

    reference_aggregate = {
        component: weighted("reference", component)
        for component in ("policy", "value", "composite")
    }
    candidate_aggregate = {
        component: weighted("candidate", component)
        for component in ("policy", "value", "composite")
    }
    return {
        "finite": True,
        "samples": total_samples,
        "batches": len(observations),
        "observation_unit": "deterministic-held-out-batch",
        "reference": reference_aggregate,
        "candidate": candidate_aggregate,
        "composite_improvement": (
            reference_aggregate["composite"] - candidate_aggregate["composite"]
        ),
        "observations": observations,
    }


def cast_mapping(value: object) -> Mapping[str, object]:
    if not isinstance(value, Mapping):
        raise TypeError("expected a mapping")
    return value


def _verify_sources(
    settings: CalibrationSettings,
    champion: ChampionPin,
    replay: FrozenReplay,
) -> None:
    for pin in (champion.pointer, champion.manifest, champion.checkpoint):
        _verify_pin(pin)
    current = freeze_replay(settings, champion, decode=False)
    if (
        current.cutoff_sha256 != replay.cutoff_sha256
        or current.partition_sha256 != replay.partition_sha256
    ):
        raise ValueError("source replay cutoff changed during calibration")


def _plan_payload(
    contract: Mapping[str, object],
    contract_sha256: str,
    *,
    status: str,
) -> dict[str, object]:
    payload = {
        **dict(contract),
        "status": status,
        "run_contract_sha256": contract_sha256,
        "semantic_sha256": contract_sha256,
        "diagnostic_only": True,
        "production_promotion_authorized": False,
    }
    normalized = json.loads(_canonical(payload))
    if not isinstance(normalized, dict):
        raise RuntimeError("calibration plan normalization failed")
    return normalized


def run_calibration(settings: CalibrationSettings) -> dict[str, object]:
    validate_settings(settings)
    config_pin = _pin(settings.config)
    config = load_config(settings.config)
    _verify_pin(config_pin)
    _validate_calibration_config(config, settings.arm)
    batch_size = settings.batch_size or config.train.per_rank_batch_size
    champion = _champion_pin(settings.champion, config)
    replay = freeze_replay(settings, champion, decode=not settings.dry_run)
    contract = _run_contract(
        settings,
        config,
        config_pin,
        champion,
        replay,
        batch_size=batch_size,
    )
    contract_sha256 = _digest(contract)
    plan = _plan_payload(
        contract,
        contract_sha256,
        status="dry_run" if settings.dry_run else "frozen",
    )
    if settings.dry_run:
        return plan

    state = _validated_state(settings, contract_sha256)
    plan_path = _plan_path(settings)
    if not plan_path.exists():
        atomic_json(plan_path, plan)
        atomic_json(_state_path(settings), state)
    if state.get("status") == "complete":
        result_path = settings.output_dir / "result.json"
        if not result_path.is_file():
            raise ValueError("complete calibration state has no result artifact")
        return _load_json(result_path)

    requested_device = settings.device or config.learner.device
    device = torch.device(resolve_device_string(requested_device))
    precision = resolve_precision(config.train.precision, device)
    compile_enabled = bool(config.train.compile) if device.type == "cuda" else False
    enable_fast_math(device)
    (
        model,
        compiled,
        optimizer,
        scheduler,
        ema,
        routing,
    ) = _initialize_training_state(
        config=config,
        champion=champion,
        device=device,
        precision=precision,
        compile_enabled=compile_enabled,
        state=state,
        contract_sha256=contract_sha256,
    )

    completed = _int_field(state, "completed_steps")
    measurement_started_ns = _int_field(state, "measurement_started_ns")
    if measurement_started_ns <= 0 or measurement_started_ns > time.time_ns():
        raise ValueError("state measurement start is invalid")

    def wall_elapsed_seconds() -> float:
        return (time.time_ns() - measurement_started_ns) / 1e9

    training_seconds = _float_field(state, "training_seconds")
    loss_sums_raw = state.get("loss_sums")
    if not isinstance(loss_sums_raw, Mapping):
        raise ValueError("state loss_sums is invalid")
    loss_sums = {str(key): float(value) for key, value in loss_sums_raw.items()}
    gradient_sum = _float_field(state, "gradient_norm_sum")
    clipped_steps = _int_field(state, "gradient_clipped_steps")
    nonfinite_loss = _int_field(state, "nonfinite_loss_count")
    nonfinite_gradient = _int_field(state, "nonfinite_gradient_count")
    budget_seconds = settings.budget_h100_hours * 3600.0
    invocation_steps = 0
    state["status"] = "running"
    atomic_json(_state_path(settings), state)

    while completed < settings.steps:
        elapsed_now = wall_elapsed_seconds()
        if elapsed_now >= budget_seconds:
            state["status"] = "budget_exhausted"
            break
        if (
            settings.stop_after_steps is not None
            and invocation_steps >= settings.stop_after_steps
        ):
            state["status"] = "paused"
            break
        batch = _materialize(
            replay,
            replay.train,
            start=completed * batch_size,
            batch_size=batch_size,
            seed=settings.seed,
            augment=config.data.d5_augmentation,
        )
        step_started = time.perf_counter()
        step_result = train_step(
            compiled,
            batch,
            optimizer,
            loss_weights=config.loss,
            precision=precision,
            gradient_clip_norm=config.train.gradient_clip_norm,
            scheduler=scheduler,
            ema=ema,
            trusted_batch=True,
            collect_diagnostics=completed == 0,
        )
        host = step_result.to_host()
        synchronize_device(device)
        training_seconds += time.perf_counter() - step_started
        if any(
            not math.isfinite(value) for value in host.losses.values()
        ) or not math.isfinite(host.gradient_norm):
            raise FloatingPointError("calibration training produced non-finite metrics")
        for name, value in host.losses.items():
            loss_sums[name] = loss_sums.get(name, 0.0) + value
        gradient_sum += host.gradient_norm
        clipped_steps += int(host.gradient_clipped)
        nonfinite_loss += host.nonfinite_loss_count
        nonfinite_gradient += host.nonfinite_gradient_count
        if completed == 0:
            state["first_step_optimizer_diagnostics"] = [
                group.as_dict() for group in host.optimizer_groups
            ]
        completed += 1
        invocation_steps += 1
        state.update(
            {
                "completed_steps": completed,
                "examples_consumed": completed * batch_size,
                "training_seconds": training_seconds,
                "loss_sums": loss_sums,
                "gradient_norm_sum": gradient_sum,
                "gradient_clipped_steps": clipped_steps,
                "nonfinite_loss_count": nonfinite_loss,
                "nonfinite_gradient_count": nonfinite_gradient,
            }
        )
        if completed % settings.checkpoint_interval == 0:
            state["elapsed_seconds"] = wall_elapsed_seconds()
            _save_progress(
                settings=settings,
                config=config,
                model=model,
                optimizer=optimizer,
                scheduler=scheduler,
                ema=ema,
                state=state,
                contract_sha256=contract_sha256,
            )

    state["elapsed_seconds"] = wall_elapsed_seconds()
    if completed == settings.steps and float(state["elapsed_seconds"]) > budget_seconds:
        state["status"] = "budget_exhausted"
    elif completed == settings.steps:
        state["status"] = "evaluating"
    checkpoint_pin = _save_progress(
        settings=settings,
        config=config,
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
        ema=ema,
        state=state,
        contract_sha256=contract_sha256,
    )
    if completed != settings.steps or state["status"] == "budget_exhausted":
        return {
            **plan,
            "status": state["status"],
            "progress": {
                "completed_steps": completed,
                "target_steps": settings.steps,
                "elapsed_seconds": state["elapsed_seconds"],
                "checkpoint": checkpoint_pin.as_dict(),
            },
        }

    heldout = _evaluate_holdout(
        config=config,
        champion=champion,
        replay=replay,
        model=model,
        ema=ema,
        device=device,
        precision=precision,
        batch_size=settings.evaluation_batch_size,
    )
    state["elapsed_seconds"] = wall_elapsed_seconds()
    if float(state["elapsed_seconds"]) > budget_seconds:
        state["status"] = "budget_exhausted"
        atomic_json(_state_path(settings), state)
        return {
            **plan,
            "status": "budget_exhausted",
            "progress": {
                "completed_steps": completed,
                "target_steps": settings.steps,
                "elapsed_seconds": state["elapsed_seconds"],
                "checkpoint": checkpoint_pin.as_dict(),
            },
        }
    _verify_sources(settings, champion, replay)
    _verify_pin(config_pin)
    mean_losses = {name: total / completed for name, total in sorted(loss_sums.items())}
    training_finite = (
        nonfinite_loss == 0
        and nonfinite_gradient == 0
        and all(math.isfinite(value) for value in mean_losses.values())
        and math.isfinite(gradient_sum)
    )
    if not training_finite:
        raise FloatingPointError("calibration finite-training gate failed")
    throughput = (completed * batch_size) / training_seconds
    result: dict[str, object] = {
        **plan,
        "status": "complete",
        "device": {
            "requested": requested_device,
            "resolved": str(device),
            "precision": precision,
            "compile": compile_enabled,
        },
        "optimizer": {
            "fresh_from_champion_ema": True,
            "source_optimizer_loaded": False,
            "source_scheduler_loaded": False,
            "routing": routing.as_dict(include_parameter_names=True),
            "first_step_diagnostics": state["first_step_optimizer_diagnostics"],
        },
        "training": {
            **cast_mapping(contract["training"]),
            "completed_steps": completed,
            "examples_consumed": completed * batch_size,
            "elapsed_seconds": state["elapsed_seconds"],
            "measured_training_seconds": training_seconds,
            "examples_per_second": throughput,
            "finite": training_finite,
            "nonfinite_loss_count": nonfinite_loss,
            "nonfinite_gradient_count": nonfinite_gradient,
            "mean_losses": mean_losses,
            "mean_gradient_norm": gradient_sum / completed,
            "gradient_clipped_steps": clipped_steps,
            "gradient_clipping_frequency": clipped_steps / completed,
        },
        "heldout": heldout,
        "candidate_checkpoint": checkpoint_pin.as_dict(),
    }
    normalized = json.loads(_canonical(result))
    if not isinstance(normalized, dict):
        raise RuntimeError("calibration result normalization failed")
    result = normalized
    result["result_sha256"] = _digest(result)
    result_path = settings.output_dir / "result.json"
    atomic_json(result_path, result)
    state.update(
        {
            "status": "complete",
            "result": _pin(result_path).as_dict(),
        }
    )
    atomic_json(_state_path(settings), state)
    return result


def _error_payload(error: BaseException) -> dict[str, object]:
    return {
        "format": FORMAT,
        "schema_version": SCHEMA_VERSION,
        "status": "error",
        "error": {
            "type": type(error).__name__,
            "message": str(error),
        },
        "production_promotion_authorized": False,
    }


def main(argv: list[str] | None = None) -> int:
    settings = _settings(_parser().parse_args(argv))
    try:
        result = run_calibration(settings)
    except (
        FloatingPointError,
        OSError,
        RuntimeError,
        sqlite3.Error,
        TypeError,
        ValueError,
    ) as error:
        print(json.dumps(_error_payload(error), sort_keys=True))
        return 2
    print(json.dumps(result, allow_nan=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
