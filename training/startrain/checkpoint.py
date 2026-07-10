"""Atomic checkpoints and float32 exponential moving averages."""

from __future__ import annotations

import json
import hashlib
import os
import re
import tempfile
import time
from collections.abc import Iterator, Mapping
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from torch import Tensor, nn

from .contracts import (
    FEATURE_SCHEMA_HASH,
    RULES_HASH,
    RULES_HASH_WIRE,
    RULES_SCHEMA_ID,
)
from .runtime import atomic_json, validate_identifier

CHECKPOINT_FORMAT = "startrain.checkpoint"
CHECKPOINT_VERSION = 2
EMA_VERSION = 1
MODEL_MANIFEST_FORMAT = "startrain.model-manifest"
MODEL_POINTER_FORMAT = "startrain.model-pointer"
MODEL_MANIFEST_VERSION = 2


@dataclass(frozen=True, slots=True)
class ModelManifest:
    path: Path
    checkpoint: Path
    model_version: str
    model_step: int
    published_ns: int
    model_identity: str = "manual"
    checkpoint_sha256: str = ""
    checkpoint_bytes: int = 0
    run_id: str = "manual"
    generation_family: str = "manual"
    artifact_manifest: Path | None = None
    manifest_sha256: str = ""
    manifest_bytes: int = 0
    role: str = "direct"


class ExponentialMovingAverage:
    def __init__(
        self,
        model: nn.Module,
        *,
        decay: float = 0.999,
        device: torch.device | str | None = None,
    ) -> None:
        if not 0.0 <= decay < 1.0:
            raise ValueError("EMA decay must be in [0, 1)")
        self.decay = float(decay)
        self.num_updates = 0
        target_device = torch.device(device) if device is not None else None
        self.shadow: dict[str, Tensor] = {}
        for name, value in model.state_dict().items():
            if value.is_floating_point():
                self.shadow[name] = value.detach().to(
                    device=target_device or value.device,
                    dtype=torch.float32,
                    copy=True,
                )

    @torch.no_grad()
    def update(self, model: nn.Module) -> None:
        model_state = model.state_dict()
        if self.shadow.keys() != {
            name for name, value in model_state.items() if value.is_floating_point()
        }:
            raise ValueError("model state does not match EMA state")
        self.num_updates += 1
        for name, average in self.shadow.items():
            source = (
                model_state[name]
                .detach()
                .to(device=average.device, dtype=average.dtype)
            )
            average.lerp_(source, 1.0 - self.decay)

    @torch.no_grad()
    def copy_to(self, model: nn.Module) -> None:
        model_state = model.state_dict()
        for name, average in self.shadow.items():
            if name not in model_state:
                raise ValueError(f"model is missing EMA value {name}")
            model_state[name].copy_(
                average.to(
                    device=model_state[name].device, dtype=model_state[name].dtype
                )
            )

    @contextmanager
    def average_parameters(self, model: nn.Module) -> Iterator[None]:
        backup = {
            name: value.detach().clone()
            for name, value in model.state_dict().items()
            if value.is_floating_point()
        }
        self.copy_to(model)
        try:
            yield
        finally:
            model_state = model.state_dict()
            with torch.no_grad():
                for name, value in backup.items():
                    model_state[name].copy_(value)

    def state_dict(self) -> dict[str, Any]:
        return {
            "version": EMA_VERSION,
            "decay": self.decay,
            "num_updates": self.num_updates,
            "shadow": self.shadow,
        }

    def load_state_dict(self, state: Mapping[str, Any]) -> None:
        if int(state.get("version", -1)) != EMA_VERSION:
            raise ValueError("unsupported EMA state version")
        loaded_shadow = state.get("shadow")
        if not isinstance(loaded_shadow, Mapping):
            raise ValueError("EMA shadow state is missing")
        if set(loaded_shadow) != set(self.shadow):
            raise ValueError("EMA keys do not match the model")
        self.decay = float(state["decay"])
        self.num_updates = int(state["num_updates"])
        for name, average in self.shadow.items():
            value = loaded_shadow[name]
            if not isinstance(value, Tensor) or value.shape != average.shape:
                raise ValueError(f"invalid EMA tensor for {name}")
            average.copy_(value.to(device=average.device, dtype=average.dtype))


def save_checkpoint(
    destination: str | Path,
    *,
    model: nn.Module,
    step: int,
    epoch: int = 0,
    optimizer: torch.optim.Optimizer | None = None,
    scheduler: torch.optim.lr_scheduler.LRScheduler | None = None,
    ema: ExponentialMovingAverage | None = None,
    config: Mapping[str, Any] | None = None,
    extra: Mapping[str, Any] | None = None,
) -> Path:
    if step < 0 or epoch < 0:
        raise ValueError("step and epoch must be non-negative")
    destination = Path(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "format": CHECKPOINT_FORMAT,
        "version": CHECKPOINT_VERSION,
        "rules_schema": RULES_SCHEMA_ID,
        "rules_hash": RULES_HASH,
        "rules_hash_wire": RULES_HASH_WIRE,
        "feature_schema_hash": FEATURE_SCHEMA_HASH,
        "step": int(step),
        "epoch": int(epoch),
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict() if optimizer is not None else None,
        "scheduler": scheduler.state_dict() if scheduler is not None else None,
        "ema": ema.state_dict() if ema is not None else None,
        "config": dict(config or {}),
        "extra": dict(extra or {}),
    }
    temporary_name: str | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="wb",
            prefix=f".{destination.name}.",
            suffix=".tmp",
            dir=destination.parent,
            delete=False,
        ) as temporary:
            temporary_name = temporary.name
            torch.save(payload, temporary)
            temporary.flush()
            os.fsync(temporary.fileno())
        os.replace(temporary_name, destination)
        descriptor = os.open(destination.parent, os.O_RDONLY)
        try:
            os.fsync(descriptor)
        finally:
            os.close(descriptor)
    finally:
        if temporary_name is not None and os.path.exists(temporary_name):
            os.unlink(temporary_name)
    return destination


def load_checkpoint(
    source: str | Path,
    *,
    model: nn.Module,
    optimizer: torch.optim.Optimizer | None = None,
    scheduler: torch.optim.lr_scheduler.LRScheduler | None = None,
    ema: ExponentialMovingAverage | None = None,
    map_location: torch.device | str = "cpu",
    strict: bool = True,
    use_ema_weights: bool = False,
    require_ema: bool = False,
    expected_model_config: Mapping[str, Any] | None = None,
    expected_game_config: Mapping[str, Any] | None = None,
    expected_run_id: str | None = None,
    expected_generation_family: str | None = None,
    expected_sha256: str | None = None,
    expected_bytes: int | None = None,
) -> dict[str, Any]:
    checkpoint_path = Path(source)
    if expected_sha256 is not None or expected_bytes is not None:
        verify_file(
            checkpoint_path,
            expected_sha256=expected_sha256,
            expected_bytes=expected_bytes,
        )
    payload = torch.load(checkpoint_path, map_location=map_location, weights_only=True)
    _validate_checkpoint_payload(
        payload,
        expected_model_config=expected_model_config,
        expected_game_config=expected_game_config,
        expected_run_id=expected_run_id,
        expected_generation_family=expected_generation_family,
    )
    ema_payload = payload["ema"]
    if (use_ema_weights or require_ema) and ema_payload is None:
        raise ValueError("checkpoint has no EMA weights")
    model.load_state_dict(payload["model"], strict=strict)
    if use_ema_weights:
        evaluation_ema = ExponentialMovingAverage(model)
        evaluation_ema.load_state_dict(ema_payload)
        evaluation_ema.copy_to(model)
    if optimizer is not None:
        if payload["optimizer"] is None:
            raise ValueError("checkpoint has no optimizer state")
        optimizer.load_state_dict(payload["optimizer"])
    if scheduler is not None:
        if payload["scheduler"] is None:
            raise ValueError("checkpoint has no scheduler state")
        scheduler.load_state_dict(payload["scheduler"])
    if ema is not None:
        if payload["ema"] is None:
            raise ValueError("checkpoint has no EMA state")
        ema.load_state_dict(payload["ema"])
    return {
        "step": int(payload["step"]),
        "epoch": int(payload["epoch"]),
        "config": payload["config"],
        "extra": payload["extra"],
    }


def load_ema_checkpoint(
    source: str | Path,
    *,
    model: nn.Module,
    expected_model_config: Mapping[str, Any],
    expected_game_config: Mapping[str, Any],
    map_location: torch.device | str = "cpu",
    expected_run_id: str | None = None,
    expected_generation_family: str | None = None,
    expected_sha256: str | None = None,
    expected_bytes: int | None = None,
) -> dict[str, Any]:
    """Load actor/evaluation weights, rejecting raw-weight or config drift."""

    return load_checkpoint(
        source,
        model=model,
        map_location=map_location,
        use_ema_weights=True,
        require_ema=True,
        expected_model_config=expected_model_config,
        expected_game_config=expected_game_config,
        expected_run_id=expected_run_id,
        expected_generation_family=expected_generation_family,
        expected_sha256=expected_sha256,
        expected_bytes=expected_bytes,
    )


def load_model_manifest(path: str | Path) -> ModelManifest:
    source = Path(path)
    payload = _read_json(source, "model publication")
    if payload.get("format") == MODEL_POINTER_FORMAT:
        if payload.get("schema_version") != 1:
            raise ValueError("unsupported model pointer")
        role = payload.get("role")
        if role not in ("candidate", "champion"):
            raise ValueError("model pointer role must be candidate or champion")
        manifest_value = payload.get("manifest")
        if not isinstance(manifest_value, str) or not manifest_value:
            raise ValueError("model pointer manifest path is invalid")
        artifact = Path(manifest_value)
        if not artifact.is_absolute():
            artifact = source.parent / artifact
        artifact = artifact.resolve()
        verify_file(
            artifact,
            expected_sha256=_sha256_text(payload.get("manifest_sha256")),
            expected_bytes=_positive_int(
                "manifest_bytes", payload.get("manifest_bytes")
            ),
        )
        manifest_payload = _read_json(artifact, "immutable model manifest")
        manifest = _parse_model_manifest(
            source=source,
            artifact=artifact,
            payload=manifest_payload,
            role=role,
        )
        expected_pointer = {
            "model_identity": manifest.model_identity,
            "model_step": manifest.model_step,
            "run_id": manifest.run_id,
            "generation_family": manifest.generation_family,
        }
        if any(payload.get(key) != value for key, value in expected_pointer.items()):
            raise ValueError("model pointer identity disagrees with its manifest")
        return manifest
    if payload.get("format") == MODEL_MANIFEST_FORMAT:
        return _parse_model_manifest(
            source=source,
            artifact=source,
            payload=payload,
            role="direct",
        )
    raise ValueError("legacy mutable model manifests are not supported")


def write_model_pointer(
    destination: str | Path,
    manifest: ModelManifest,
    *,
    role: str,
    promotion_result: str | None = None,
) -> Path:
    if role not in ("candidate", "champion"):
        raise ValueError("model pointer role must be candidate or champion")
    artifact = manifest.artifact_manifest or manifest.path
    verify_file(
        artifact,
        expected_sha256=manifest.manifest_sha256,
        expected_bytes=manifest.manifest_bytes,
    )
    verify_file(
        manifest.checkpoint,
        expected_sha256=manifest.checkpoint_sha256,
        expected_bytes=manifest.checkpoint_bytes,
    )
    payload: dict[str, object] = {
        "format": MODEL_POINTER_FORMAT,
        "schema_version": 1,
        "role": role,
        "manifest": os.path.relpath(
            artifact.resolve(), Path(destination).parent.resolve()
        ),
        "manifest_sha256": manifest.manifest_sha256,
        "manifest_bytes": manifest.manifest_bytes,
        "model_identity": manifest.model_identity,
        "model_step": manifest.model_step,
        "run_id": manifest.run_id,
        "generation_family": manifest.generation_family,
        "updated_ns": time.time_ns(),
    }
    if promotion_result is not None:
        payload["promotion_result"] = (
            promotion_result
            if promotion_result == "bootstrap"
            else os.path.relpath(
                Path(promotion_result).resolve(),
                Path(destination).parent.resolve(),
            )
        )
    atomic_json(destination, payload)
    return Path(destination)


def _parse_model_manifest(
    *,
    source: Path,
    artifact: Path,
    payload: Mapping[str, Any],
    role: str,
) -> ModelManifest:
    if (
        payload.get("format") != MODEL_MANIFEST_FORMAT
        or payload.get("schema_version") != MODEL_MANIFEST_VERSION
    ):
        raise ValueError("unsupported immutable model manifest")
    if payload.get("rules_hash") != RULES_HASH_WIRE:
        raise ValueError("model manifest rules hash is incompatible")
    if payload.get("feature_schema_hash") != f"{FEATURE_SCHEMA_HASH:016x}":
        raise ValueError("model manifest feature schema is incompatible")
    if payload.get("weights") != "ema":
        raise ValueError("model manifest must explicitly publish EMA weights")
    model_identity = validate_identifier(
        "model_identity", payload.get("model_identity")
    )
    if not model_identity.startswith("sha256-"):
        raise ValueError("model identity must be content-addressed")
    model_version = validate_identifier("model_version", payload.get("model_version"))
    if model_version != model_identity:
        raise ValueError("model version must equal its immutable model identity")
    run_id = validate_identifier("run_id", payload.get("run_id"))
    family = validate_identifier("generation_family", payload.get("generation_family"))
    model_step = _nonnegative_int("model_step", payload.get("model_step"))
    published_ns = _positive_int("created_ns", payload.get("created_ns"))
    checkpoint_value = payload.get("checkpoint")
    if not isinstance(checkpoint_value, str) or not checkpoint_value:
        raise ValueError("model manifest checkpoint path is invalid")
    checkpoint = Path(checkpoint_value)
    if not checkpoint.is_absolute():
        checkpoint = artifact.parent / checkpoint
    checkpoint = checkpoint.resolve()
    checkpoint_sha256 = _sha256_text(payload.get("checkpoint_sha256"))
    checkpoint_bytes = _positive_int(
        "checkpoint_bytes", payload.get("checkpoint_bytes")
    )
    verify_file(
        checkpoint,
        expected_sha256=checkpoint_sha256,
        expected_bytes=checkpoint_bytes,
    )
    if checkpoint.name != f"sha256-{checkpoint_sha256}.pt":
        raise ValueError("checkpoint filename is not content-addressed")
    manifest_sha256 = sha256_file(artifact)
    if artifact.name != f"manifest-{manifest_sha256}.json":
        raise ValueError("immutable model manifest filename is not content-addressed")
    return ModelManifest(
        path=source,
        checkpoint=checkpoint,
        model_version=model_version,
        model_step=model_step,
        published_ns=published_ns,
        model_identity=model_identity,
        checkpoint_sha256=checkpoint_sha256,
        checkpoint_bytes=checkpoint_bytes,
        run_id=run_id,
        generation_family=family,
        artifact_manifest=artifact,
        manifest_sha256=manifest_sha256,
        manifest_bytes=artifact.stat().st_size,
        role=role,
    )


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def verify_file(
    path: str | Path,
    *,
    expected_sha256: str | None,
    expected_bytes: int | None,
) -> None:
    source = Path(path)
    if not source.is_file():
        raise ValueError(f"published artifact does not exist: {source}")
    if expected_bytes is not None and source.stat().st_size != expected_bytes:
        raise ValueError(f"published artifact byte length failed: {source}")
    if expected_sha256 is not None and sha256_file(source) != expected_sha256:
        raise ValueError(f"published artifact SHA-256 failed: {source}")


def _read_json(path: Path, name: str) -> dict[str, Any]:
    try:
        with path.open("r", encoding="utf-8") as stream:
            payload = json.load(stream)
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"cannot read {name} {path}: {exc}") from exc
    if not isinstance(payload, dict):
        raise ValueError(f"{name} must be a JSON object")
    return payload


def _sha256_text(value: object) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise ValueError("SHA-256 must be 64 lowercase hexadecimal characters")
    return value


def _positive_int(name: str, value: object) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"{name} must be a positive integer")
    return value


def _nonnegative_int(name: str, value: object) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(f"{name} must be a non-negative integer")
    return value


def latest_checkpoint(directory: str | Path) -> Path | None:
    root = Path(directory)
    for candidate_pointer in (
        root / "candidate.json",
        root.parent / "candidate.json",
    ):
        if candidate_pointer.is_file():
            return load_model_manifest(candidate_pointer).checkpoint
    candidates: list[tuple[int, Path]] = []
    for pattern in ("checkpoint-*.pt", "step-*.pt"):
        for path in root.glob(pattern):
            if not path.is_file():
                continue
            match = re.search(r"(\d+)\.pt$", path.name)
            if match:
                candidates.append((int(match.group(1)), path))
    if not candidates:
        return None
    return max(candidates, key=lambda value: (value[0], value[1].name))[1]


def collect_model_garbage(
    root: str | Path,
    *,
    retain_candidate_manifests: int,
    dry_run: bool,
    referenced_result_directory: str | Path | None = None,
) -> dict[str, int]:
    if retain_candidate_manifests <= 0:
        raise ValueError("retain_candidate_manifests must be positive")
    directory = Path(root)
    manifests = []
    invalid_manifests = 0
    for path in (directory / "manifests").glob("manifest-*.json"):
        try:
            manifests.append(load_model_manifest(path))
        except ValueError:
            invalid_manifests += 1
            continue
    manifests.sort(key=lambda item: (item.model_step, item.model_identity))
    protected_paths: set[Path] = set()
    for pointer_name in ("candidate.json", "champion.json"):
        pointer = directory / pointer_name
        if pointer.is_file():
            manifest = load_model_manifest(pointer)
            protected_paths.add((manifest.artifact_manifest or manifest.path).resolve())
    protected_paths.update(
        (item.artifact_manifest or item.path).resolve()
        for item in manifests[-retain_candidate_manifests:]
    )
    if referenced_result_directory is not None:
        for result_path in Path(referenced_result_directory).glob("*.json"):
            try:
                payload = _read_json(result_path, "arena result")
            except ValueError:
                continue
            for key in ("candidate_manifest", "champion_manifest"):
                value = payload.get(key)
                if isinstance(value, str) and value:
                    path = Path(value)
                    if not path.is_absolute():
                        path = result_path.parent / path
                    protected_paths.add(path.resolve())
    candidates = [
        item
        for item in manifests
        if (item.artifact_manifest or item.path).resolve() not in protected_paths
    ]
    retained_checkpoints = {
        item.checkpoint.resolve() for item in manifests if item not in candidates
    }
    checkpoint_candidates = {
        item.checkpoint.resolve()
        for item in candidates
        if item.checkpoint.resolve() not in retained_checkpoints
    }
    referenced_checkpoints = {item.checkpoint.resolve() for item in manifests}
    checkpoint_candidates.update(
        path.resolve()
        for path in (directory / "checkpoints").glob("sha256-*.pt")
        if path.resolve() not in referenced_checkpoints
    )
    bytes_reclaimable = sum(
        path.stat().st_size
        for path in (
            *((item.artifact_manifest or item.path) for item in candidates),
            *checkpoint_candidates,
        )
        if path.is_file()
    )
    metrics = {
        "candidate_manifests": len(candidates),
        "candidate_checkpoints": len(checkpoint_candidates),
        "candidate_bytes": bytes_reclaimable,
        "deleted_manifests": 0,
        "deleted_checkpoints": 0,
        "deleted_bytes": 0,
        "dry_run": int(dry_run),
        "invalid_manifests": invalid_manifests,
    }
    if dry_run:
        return metrics
    for item in candidates:
        (item.artifact_manifest or item.path).unlink(missing_ok=True)
    for path in checkpoint_candidates:
        path.unlink(missing_ok=True)
    metrics["deleted_manifests"] = len(candidates)
    metrics["deleted_checkpoints"] = len(checkpoint_candidates)
    metrics["deleted_bytes"] = bytes_reclaimable
    return metrics


def _validate_checkpoint_payload(
    payload: object,
    *,
    expected_model_config: Mapping[str, Any] | None,
    expected_game_config: Mapping[str, Any] | None,
    expected_run_id: str | None,
    expected_generation_family: str | None,
) -> None:
    if not isinstance(payload, dict) or payload.get("format") != CHECKPOINT_FORMAT:
        raise ValueError("not a startrain checkpoint")
    if payload.get("version") != CHECKPOINT_VERSION:
        raise ValueError("unsupported checkpoint version")
    if payload.get("rules_schema") != RULES_SCHEMA_ID:
        raise ValueError("checkpoint rules schema is incompatible")
    if payload.get("rules_hash") != RULES_HASH:
        raise ValueError("checkpoint rules hash is incompatible")
    if payload.get("rules_hash_wire") != RULES_HASH_WIRE:
        raise ValueError("checkpoint rules hash identifier is incompatible")
    if payload.get("feature_schema_hash") != FEATURE_SCHEMA_HASH:
        raise ValueError("checkpoint feature schema hash is incompatible")
    required = {
        "model",
        "optimizer",
        "scheduler",
        "ema",
        "step",
        "epoch",
        "config",
        "extra",
    }
    if not required <= payload.keys():
        raise ValueError("checkpoint payload is incomplete")
    if not isinstance(payload["model"], Mapping):
        raise ValueError("checkpoint model state is invalid")
    if not isinstance(payload["extra"], Mapping):
        raise ValueError("checkpoint extra metadata is invalid")
    for name in ("step", "epoch"):
        value = payload[name]
        if isinstance(value, bool) or not isinstance(value, int) or value < 0:
            raise ValueError(f"checkpoint {name} is invalid")
    config = payload.get("config")
    if not isinstance(config, Mapping):
        raise ValueError("checkpoint configuration is missing")
    if expected_model_config is not None:
        actual_model = config.get("model")
        if not isinstance(actual_model, Mapping) or dict(actual_model) != dict(
            expected_model_config
        ):
            raise ValueError("checkpoint model/feature configuration is incompatible")
    if expected_game_config is not None:
        actual_game = config.get("game")
        if not isinstance(actual_game, Mapping) or dict(actual_game) != dict(
            expected_game_config
        ):
            raise ValueError("checkpoint game/rules configuration is incompatible")
    extra = payload["extra"]
    if expected_run_id is not None and extra.get("run_id") != expected_run_id:
        raise ValueError("checkpoint run_id is incompatible")
    if (
        expected_generation_family is not None
        and extra.get("generation_family") != expected_generation_family
    ):
        raise ValueError("checkpoint generation family is incompatible")
