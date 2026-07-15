"""Atomic checkpoints and float32 exponential moving averages."""

from __future__ import annotations

import json
import hashlib
import os
import re
import tempfile
import time
from collections.abc import Callable, Iterator, Mapping
from contextlib import contextmanager
from dataclasses import dataclass, replace
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
from .model import MODEL_SCHEMA_VERSION
from .runtime import append_jsonl, atomic_json, validate_identifier

CHECKPOINT_FORMAT = "startrain.checkpoint"
CHECKPOINT_VERSION = 3
EMA_VERSION = 1
MODEL_MANIFEST_FORMAT = "startrain.model-manifest"
MODEL_POINTER_FORMAT = "startrain.model-pointer"
MODEL_MANIFEST_VERSION = 3
MODEL_POINTER_VERSION = 2
RECOVERY_POINTER_FORMAT = "startrain.recovery-pointer"
RECOVERY_POINTER_VERSION = 1
RESUME_CUTOVER_FORMAT = "startrain.resume-cutover"
RESUME_CUTOVER_VERSION = 1


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


@dataclass(frozen=True, slots=True)
class ResumeCheckpoint:
    checkpoint: Path
    checkpoint_sha256: str
    checkpoint_bytes: int
    step: int
    epoch: int
    run_id: str
    generation_family: str
    source: str
    manifest_path: Path | None = None


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
        groups: dict[
            tuple[torch.device, torch.dtype],
            tuple[list[Tensor], list[Tensor]],
        ] = {}
        for name, average in self.shadow.items():
            source = (
                model_state[name]
                .detach()
                .to(device=average.device, dtype=average.dtype)
            )
            averages, sources = groups.setdefault(
                (average.device, average.dtype),
                ([], []),
            )
            averages.append(average)
            sources.append(source)
        for averages, sources in groups.values():
            foreach_lerp = getattr(torch, "_foreach_lerp_", None)
            if callable(foreach_lerp):
                foreach_lerp(averages, sources, 1.0 - self.decay)
            else:
                for average, source in zip(averages, sources, strict=True):
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
        "model_schema_version": MODEL_SCHEMA_VERSION,
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
    metadata_validator: Callable[[Mapping[str, Any]], object] | None = None,
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
    metadata = {
        "step": int(payload["step"]),
        "epoch": int(payload["epoch"]),
        "config": payload["config"],
        "extra": payload["extra"],
    }
    if metadata_validator is not None:
        metadata_validator(metadata)
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
    return metadata


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
        if payload.get("schema_version") != MODEL_POINTER_VERSION:
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
        "schema_version": MODEL_POINTER_VERSION,
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


def write_recovery_checkpoint(
    root: str | Path,
    *,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LRScheduler,
    ema: ExponentialMovingAverage,
    step: int,
    epoch: int,
    config: Mapping[str, Any],
    run_id: str,
    generation_family: str,
    examples_consumed: int,
    global_batch_size: int,
) -> ResumeCheckpoint:
    directory = Path(root)
    recovery_directory = directory / "recovery"
    recovery_directory.mkdir(parents=True, exist_ok=True)
    run_id = validate_identifier("run_id", run_id)
    family = validate_identifier("generation_family", generation_family)
    staged = recovery_directory / f".recovery-{step:012d}.staging.pt"
    save_checkpoint(
        staged,
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
        ema=ema,
        step=step,
        epoch=epoch,
        config=config,
        extra={
            "training_step_version": f"step-{step:012d}",
            "run_id": run_id,
            "generation_family": family,
            "examples_consumed": examples_consumed,
            "global_batch_size": global_batch_size,
        },
    )
    checkpoint_sha256 = sha256_file(staged)
    checkpoint = recovery_directory / f"sha256-{checkpoint_sha256}.pt"
    checkpoint_bytes = staged.stat().st_size
    if checkpoint.exists():
        verify_file(
            checkpoint,
            expected_sha256=checkpoint_sha256,
            expected_bytes=checkpoint_bytes,
        )
        staged.unlink()
    else:
        os.replace(staged, checkpoint)
        descriptor = os.open(recovery_directory, os.O_RDONLY)
        try:
            os.fsync(descriptor)
        finally:
            os.close(descriptor)
    payload: dict[str, object] = {
        "format": RECOVERY_POINTER_FORMAT,
        "schema_version": RECOVERY_POINTER_VERSION,
        "checkpoint": os.path.relpath(checkpoint, directory),
        "checkpoint_sha256": checkpoint_sha256,
        "checkpoint_bytes": checkpoint_bytes,
        "step": step,
        "epoch": epoch,
        "examples_consumed": examples_consumed,
        "run_id": run_id,
        "generation_family": family,
        "updated_ns": time.time_ns(),
    }
    append_jsonl(directory / "recovery.journal.jsonl", payload, durable=True)
    atomic_json(directory / "recovery.json", payload)
    return _parse_recovery_pointer(
        directory / "recovery.json",
        payload,
        expected_run_id=run_id,
        expected_generation_family=family,
        source_name="recovery",
    )


def write_resume_cutover(
    root: str | Path,
    *,
    manifest: ModelManifest | ResumeCheckpoint,
    run_id: str,
    generation_family: str,
    created_ns: int | None = None,
) -> ResumeCheckpoint:
    directory = Path(root)
    run_id = validate_identifier("run_id", run_id)
    family = validate_identifier("generation_family", generation_family)
    if manifest.run_id != run_id or manifest.generation_family != family:
        raise ValueError("resume cutover manifest belongs to another run")
    step = manifest.model_step if isinstance(manifest, ModelManifest) else manifest.step
    payload: dict[str, object] = {
        "format": RESUME_CUTOVER_FORMAT,
        "schema_version": RESUME_CUTOVER_VERSION,
        "checkpoint": os.path.relpath(manifest.checkpoint, directory),
        "checkpoint_sha256": manifest.checkpoint_sha256,
        "checkpoint_bytes": manifest.checkpoint_bytes,
        "step": step,
        "run_id": run_id,
        "generation_family": family,
        "created_ns": time.time_ns() if created_ns is None else created_ns,
    }
    path = directory / "resume-cutover.json"
    atomic_json(path, payload)
    return _parse_resume_cutover(
        path,
        payload,
        expected_run_id=run_id,
        expected_generation_family=family,
    )


def discover_resume_checkpoints(
    root: str | Path,
    *,
    run_id: str,
    generation_family: str,
) -> tuple[list[ResumeCheckpoint], list[str]]:
    directory = Path(root)
    run_id = validate_identifier("run_id", run_id)
    family = validate_identifier("generation_family", generation_family)
    discovered: list[ResumeCheckpoint] = []
    failures: list[str] = []
    cutover: ResumeCheckpoint | None = None
    cutover_ns = 0
    cutover_path = directory / "resume-cutover.json"
    if cutover_path.is_file():
        try:
            cutover_payload = _read_json(cutover_path, "resume cutover")
            cutover = _parse_resume_cutover(
                cutover_path,
                cutover_payload,
                expected_run_id=run_id,
                expected_generation_family=family,
                verify_artifact=False,
            )
            cutover_ns = _positive_int("created_ns", cutover_payload.get("created_ns"))
            discovered.append(cutover)
        except ValueError as exc:
            failures.append(f"resume-cutover.json: {exc}")
            return [], failures

    recovery_pointer = directory / "recovery.json"
    if recovery_pointer.exists():
        try:
            recovery_payload = _read_json(recovery_pointer, "recovery pointer")
            if _artifact_is_after_cutover(
                recovery_payload,
                cutover=cutover,
                cutover_ns=cutover_ns,
                timestamp_key="updated_ns",
            ):
                discovered.append(
                    _parse_recovery_pointer(
                        recovery_pointer,
                        recovery_payload,
                        expected_run_id=run_id,
                        expected_generation_family=family,
                        source_name="recovery",
                        verify_artifact=False,
                    )
                )
        except ValueError as exc:
            failures.append(f"recovery.json: {exc}")

    journal_path = directory / "recovery.journal.jsonl"
    if journal_path.is_file():
        try:
            journal_lines = journal_path.read_text(encoding="utf-8").splitlines()
        except OSError as exc:
            failures.append(f"recovery journal: {exc}")
        else:
            first_index = max(0, len(journal_lines) - 64)
            recent_lines = journal_lines[first_index:]
            for index, line in reversed(
                list(enumerate(recent_lines, start=first_index + 1))
            ):
                try:
                    payload = json.loads(line)
                    if not isinstance(payload, dict):
                        raise ValueError("entry is not an object")
                    if not _artifact_is_after_cutover(
                        payload,
                        cutover=cutover,
                        cutover_ns=cutover_ns,
                        timestamp_key="updated_ns",
                    ):
                        continue
                    discovered.append(
                        _parse_recovery_pointer(
                            journal_path,
                            payload,
                            expected_run_id=run_id,
                            expected_generation_family=family,
                            source_name=f"recovery-journal:{index}",
                            verify_artifact=False,
                        )
                    )
                except (json.JSONDecodeError, ValueError) as exc:
                    failures.append(f"recovery journal line {index}: {exc}")

    candidate_pointer = directory / "candidate.json"
    candidate_valid = False
    if candidate_pointer.is_file():
        try:
            manifest = load_model_manifest(candidate_pointer)
            if manifest.run_id != run_id or manifest.generation_family != family:
                raise ValueError("run identity mismatch")
            if _manifest_is_after_cutover(
                manifest, cutover=cutover, cutover_ns=cutover_ns
            ):
                discovered.append(
                    _resume_from_manifest(manifest, source="candidate.json")
                )
                candidate_valid = True
            else:
                failures.append("candidate.json: predates the durable resume cutover")
        except ValueError as exc:
            failures.append(f"candidate.json: {exc}")

    if not candidate_valid:
        for manifest_path in (directory / "manifests").glob("manifest-*.json"):
            try:
                manifest = load_model_manifest(manifest_path)
                if manifest.run_id != run_id or manifest.generation_family != family:
                    continue
                if not _manifest_is_after_cutover(
                    manifest, cutover=cutover, cutover_ns=cutover_ns
                ):
                    continue
                discovered.append(
                    _resume_from_manifest(manifest, source="manifest-history")
                )
            except ValueError as exc:
                failures.append(f"{manifest_path.name}: {exc}")

    champion_pointer = directory / "champion.json"
    if champion_pointer.is_file():
        try:
            manifest = load_model_manifest(champion_pointer)
            if manifest.run_id != run_id or manifest.generation_family != family:
                raise ValueError("run identity mismatch")
            if not _manifest_is_after_cutover(
                manifest, cutover=cutover, cutover_ns=cutover_ns
            ):
                raise ValueError("champion predates the durable resume cutover")
            discovered.append(_resume_from_manifest(manifest, source="champion.json"))
        except ValueError as exc:
            failures.append(f"champion.json: {exc}")

    unique: dict[tuple[Path, int], ResumeCheckpoint] = {}
    source_priority = {
        "cutover": 5,
        "recovery": 4,
        "recovery-journal": 4,
        "candidate.json": 3,
        "manifest-history": 2,
        "champion.json": 1,
    }
    for candidate in discovered:
        key = (candidate.checkpoint.resolve(), candidate.step)
        current = unique.get(key)
        candidate_priority = source_priority.get(
            candidate.source.split(":", maxsplit=1)[0], 0
        )
        current_priority = (
            source_priority.get(current.source.split(":", maxsplit=1)[0], 0)
            if current is not None
            else -1
        )
        if current is None or candidate_priority > current_priority:
            if candidate.manifest_path is None and current is not None:
                candidate = replace(candidate, manifest_path=current.manifest_path)
            unique[key] = candidate
        elif current.manifest_path is None and candidate.manifest_path is not None:
            unique[key] = replace(current, manifest_path=candidate.manifest_path)
    ordered = sorted(
        unique.values(),
        key=lambda item: (
            item.step,
            source_priority.get(item.source.split(":", maxsplit=1)[0], 0),
            item.checkpoint.name,
        ),
        reverse=True,
    )
    return ordered, failures


def collect_recovery_garbage(
    root: str | Path, *, retain_checkpoints: int, dry_run: bool
) -> dict[str, int]:
    if retain_checkpoints <= 0:
        raise ValueError("retain_checkpoints must be positive")
    directory = Path(root)
    cutover: ResumeCheckpoint | None = None
    cutover_ns = 0
    cutover_invalid = False
    cutover_path = directory / "resume-cutover.json"
    if cutover_path.is_file():
        try:
            raw_cutover = _read_json(cutover_path, "resume cutover")
            cutover = _parse_resume_cutover(
                cutover_path,
                raw_cutover,
                expected_run_id=validate_identifier(
                    "run_id", raw_cutover.get("run_id")
                ),
                expected_generation_family=validate_identifier(
                    "generation_family", raw_cutover.get("generation_family")
                ),
            )
            cutover_ns = _positive_int("created_ns", raw_cutover.get("created_ns"))
        except ValueError:
            cutover_invalid = True
    payloads: list[ResumeCheckpoint] = []
    seen_checkpoints: set[Path] = set()

    def add_if_valid(source: Path, raw: Mapping[str, Any], source_name: str) -> None:
        if not _artifact_is_after_cutover(
            raw,
            cutover=cutover,
            cutover_ns=cutover_ns,
            timestamp_key="updated_ns",
        ):
            return
        item = _parse_recovery_pointer(
            source,
            raw,
            expected_run_id=validate_identifier("run_id", raw.get("run_id")),
            expected_generation_family=validate_identifier(
                "generation_family", raw.get("generation_family")
            ),
            source_name=source_name,
            verify_artifact=False,
        )
        resolved = item.checkpoint.resolve()
        if resolved in seen_checkpoints:
            return
        verify_file(
            item.checkpoint,
            expected_sha256=item.checkpoint_sha256,
            expected_bytes=item.checkpoint_bytes,
        )
        seen_checkpoints.add(resolved)
        payloads.append(item)

    pointer = directory / "recovery.json"
    if pointer.is_file():
        try:
            raw = _read_json(pointer, "recovery pointer")
            add_if_valid(pointer, raw, "recovery")
        except ValueError:
            pass
    journal = directory / "recovery.journal.jsonl"
    if journal.is_file():
        lines = journal.read_text(encoding="utf-8").splitlines()[-64:]
        for line in reversed(lines):
            if len(payloads) >= retain_checkpoints:
                break
            try:
                raw = json.loads(line)
                if not isinstance(raw, dict):
                    continue
                add_if_valid(journal, raw, "recovery-journal")
            except (json.JSONDecodeError, ValueError):
                continue
    payloads.sort(key=lambda item: (item.step, item.checkpoint.name), reverse=True)
    protected = {item.checkpoint.resolve() for item in payloads[:retain_checkpoints]}
    if (
        cutover is not None
        and cutover.checkpoint.parent == (directory / "recovery").resolve()
    ):
        protected.add(cutover.checkpoint.resolve())
    recovery_files = list((directory / "recovery").glob("sha256-*.pt"))
    candidates = [path for path in recovery_files if path.resolve() not in protected]
    bytes_reclaimable = sum(path.stat().st_size for path in candidates)
    metrics = {
        "recovery_checkpoints": len(candidates),
        "recovery_bytes": bytes_reclaimable,
        "deleted_recovery_checkpoints": 0,
        "deleted_recovery_bytes": 0,
        "dry_run": int(dry_run),
        "valid_recovery_checkpoints": len(protected),
        "gc_skipped": 0,
    }
    if cutover_invalid or (recovery_files and not protected):
        metrics["gc_skipped"] = 1
        return metrics
    if dry_run:
        return metrics
    for path in candidates:
        path.unlink(missing_ok=True)
    metrics["deleted_recovery_checkpoints"] = len(candidates)
    metrics["deleted_recovery_bytes"] = bytes_reclaimable
    return metrics


def _parse_resume_cutover(
    source: Path,
    payload: Mapping[str, Any],
    *,
    expected_run_id: str,
    expected_generation_family: str,
    verify_artifact: bool = True,
) -> ResumeCheckpoint:
    if (
        payload.get("format") != RESUME_CUTOVER_FORMAT
        or payload.get("schema_version") != RESUME_CUTOVER_VERSION
    ):
        raise ValueError("unsupported resume cutover")
    run_id = validate_identifier("run_id", payload.get("run_id"))
    family = validate_identifier("generation_family", payload.get("generation_family"))
    if run_id != expected_run_id or family != expected_generation_family:
        raise ValueError("resume cutover run identity mismatch")
    checkpoint_value = payload.get("checkpoint")
    if not isinstance(checkpoint_value, str) or not checkpoint_value:
        raise ValueError("resume cutover checkpoint path is invalid")
    checkpoint = Path(checkpoint_value)
    if not checkpoint.is_absolute():
        checkpoint = source.parent / checkpoint
    checkpoint = checkpoint.resolve()
    allowed_directories = {
        (source.parent / "checkpoints").resolve(),
        (source.parent / "recovery").resolve(),
    }
    if checkpoint.parent not in allowed_directories:
        raise ValueError("resume cutover checkpoint escaped its artifact directories")
    checkpoint_sha256 = _sha256_text(payload.get("checkpoint_sha256"))
    checkpoint_bytes = _positive_int(
        "checkpoint_bytes", payload.get("checkpoint_bytes")
    )
    if verify_artifact:
        verify_file(
            checkpoint,
            expected_sha256=checkpoint_sha256,
            expected_bytes=checkpoint_bytes,
        )
    _positive_int("created_ns", payload.get("created_ns"))
    return ResumeCheckpoint(
        checkpoint=checkpoint,
        checkpoint_sha256=checkpoint_sha256,
        checkpoint_bytes=checkpoint_bytes,
        step=_nonnegative_int("step", payload.get("step")),
        epoch=0,
        run_id=run_id,
        generation_family=family,
        source="cutover",
    )


def _artifact_is_after_cutover(
    payload: Mapping[str, Any],
    *,
    cutover: ResumeCheckpoint | None,
    cutover_ns: int,
    timestamp_key: str,
) -> bool:
    if cutover is None:
        return True
    if payload.get("checkpoint_sha256") == cutover.checkpoint_sha256:
        return True
    timestamp = payload.get(timestamp_key)
    return (
        isinstance(timestamp, int)
        and not isinstance(timestamp, bool)
        and timestamp >= cutover_ns
    )


def _manifest_is_after_cutover(
    manifest: ModelManifest,
    *,
    cutover: ResumeCheckpoint | None,
    cutover_ns: int,
) -> bool:
    return (
        cutover is None
        or manifest.checkpoint_sha256 == cutover.checkpoint_sha256
        or manifest.published_ns >= cutover_ns
    )


def _parse_recovery_pointer(
    source: Path,
    payload: Mapping[str, Any],
    *,
    expected_run_id: str,
    expected_generation_family: str,
    source_name: str,
    verify_artifact: bool = True,
) -> ResumeCheckpoint:
    if (
        payload.get("format") != RECOVERY_POINTER_FORMAT
        or payload.get("schema_version") != RECOVERY_POINTER_VERSION
    ):
        raise ValueError("unsupported recovery pointer")
    run_id = validate_identifier("run_id", payload.get("run_id"))
    family = validate_identifier("generation_family", payload.get("generation_family"))
    if run_id != expected_run_id or family != expected_generation_family:
        raise ValueError("recovery pointer run identity mismatch")
    checkpoint_value = payload.get("checkpoint")
    if not isinstance(checkpoint_value, str) or not checkpoint_value:
        raise ValueError("recovery checkpoint path is invalid")
    checkpoint = Path(checkpoint_value)
    base = source.parent
    if not checkpoint.is_absolute():
        checkpoint = base / checkpoint
    checkpoint = checkpoint.resolve()
    recovery_directory = (base / "recovery").resolve()
    if checkpoint.parent != recovery_directory:
        raise ValueError("recovery checkpoint escaped its artifact directory")
    checkpoint_sha256 = _sha256_text(payload.get("checkpoint_sha256"))
    checkpoint_bytes = _positive_int(
        "checkpoint_bytes", payload.get("checkpoint_bytes")
    )
    if verify_artifact:
        verify_file(
            checkpoint,
            expected_sha256=checkpoint_sha256,
            expected_bytes=checkpoint_bytes,
        )
    return ResumeCheckpoint(
        checkpoint=checkpoint,
        checkpoint_sha256=checkpoint_sha256,
        checkpoint_bytes=checkpoint_bytes,
        step=_nonnegative_int("step", payload.get("step")),
        epoch=_nonnegative_int("epoch", payload.get("epoch")),
        run_id=run_id,
        generation_family=family,
        source=source_name,
    )


def _resume_from_manifest(manifest: ModelManifest, *, source: str) -> ResumeCheckpoint:
    return ResumeCheckpoint(
        checkpoint=manifest.checkpoint,
        checkpoint_sha256=manifest.checkpoint_sha256,
        checkpoint_bytes=manifest.checkpoint_bytes,
        step=manifest.model_step,
        epoch=0,
        run_id=manifest.run_id,
        generation_family=manifest.generation_family,
        source=source,
        manifest_path=manifest.artifact_manifest or manifest.path,
    )


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
    if payload.get("model_schema_version") != MODEL_SCHEMA_VERSION:
        raise ValueError("model manifest schema is incompatible")
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
    cutover_checkpoints: set[Path] = set()
    cutover_path = directory / "resume-cutover.json"
    if cutover_path.is_file():
        cutover_payload = _read_json(cutover_path, "resume cutover")
        cutover = _parse_resume_cutover(
            cutover_path,
            cutover_payload,
            expected_run_id=validate_identifier(
                "run_id", cutover_payload.get("run_id")
            ),
            expected_generation_family=validate_identifier(
                "generation_family",
                cutover_payload.get("generation_family"),
            ),
        )
        cutover_checkpoints.add(cutover.checkpoint.resolve())
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
    retained_checkpoints.update(cutover_checkpoints)
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
        and path.resolve() not in retained_checkpoints
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
    if payload.get("model_schema_version") != MODEL_SCHEMA_VERSION:
        raise ValueError("checkpoint model schema is incompatible")
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
