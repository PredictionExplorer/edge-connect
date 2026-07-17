#!/usr/bin/env python3
"""Safely migrate a stopped autonomous run to a new frozen profile."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import shutil
import sqlite3
import subprocess
import sys
import tempfile
import time
import uuid
from collections.abc import Iterator, Mapping, Sequence
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any, NoReturn, TypeGuard

from startrain.config import ExperimentConfig, load_config

MIGRATION_SCHEMA_VERSION = 1
UTD_SEGMENT_SCHEMA_VERSION = 1
_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_COMMIT = re.compile(r"^[0-9a-f]{7,64}$")
_PROFILE_NAME = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,122}\.ya?ml$")
_IDENTIFIER = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$")
_MISSING = object()

_ALLOWED_PROFILE_PATHS = {
    ("data", "min_batches_for_workers"),
    ("learner", "target_updates_per_new_sample"),
    ("learner", "candidate_interval"),
    ("learner", "candidate_interval_examples"),
    ("learner", "selfplay_snapshot_interval_examples"),
    ("learner", "selfplay_snapshot_warmup_examples"),
    ("learner", "selfplay_snapshot_warmup_interval_examples"),
    ("orchestration", "model_refresh", "inference_compile_dynamic"),
    ("orchestration", "model_refresh", "inference_compile_mode"),
    ("orchestration", "promotion", "gpu_id"),
    ("orchestration", "promotion", "max_waves_per_lease"),
    ("orchestration", "promotion", "inter_wave_cooldown_seconds"),
    ("arena", "pairs_per_ring"),
    ("arena", "minimum_pairs_per_ring"),
}


class MigrationError(RuntimeError):
    """A fail-closed migration validation error."""


@dataclass(frozen=True, slots=True)
class MigrationRequest:
    old_profile: Path
    new_profile: Path
    target_profile_name: str
    reason: str
    run_root: Path | None = None
    from_source_commit: str | None = None
    to_source_commit: str | None = None


@dataclass(frozen=True, slots=True)
class _BackupArtifact:
    source: Path
    relative_path: Path
    data: bytes
    mode: int


@dataclass(frozen=True, slots=True)
class MigrationPlan:
    run_root: Path
    old_profile: Path
    new_profile: Path
    target_profile: Path
    target_profile_checksum: Path
    target_profile_bytes: bytes
    target_profile_mode: int
    backup_directory: Path
    backup_artifacts: tuple[_BackupArtifact, ...]
    input_hashes: tuple[tuple[Path, str], ...]
    expected_absent: tuple[Path, ...]
    migration_log_existed: bool
    migration_record: Mapping[str, object]
    provenance_payload: Mapping[str, object]
    utd_segment_payload: Mapping[str, object]
    profile_sha256_bytes: bytes
    source_canonical_sha256: str
    source_authoritative_sha256: str
    target_canonical_sha256: str
    source_profile_sha256: str
    target_profile_sha256: str
    changes: tuple[tuple[str, object, object], ...]
    learner_step: int
    examples_consumed: int
    committed_replay_samples: int
    champion_model_identity: str
    coordinator_lock_status: str

    def output(self, *, mode: str, backup: Path | None = None) -> dict[str, object]:
        writes = [
            str(self.target_profile),
            str(self.target_profile_checksum),
            str(self.run_root / "learner" / "utd-segment.json"),
            str(self.run_root / "autonomous-migrations.jsonl"),
            str(self.run_root / "autonomous-provenance.json"),
            str(self.run_root / "profile.sha256"),
        ]
        return {
            "status": "ok",
            "mode": mode,
            "run_id": self.migration_record["run_id"],
            "generation_family": self.migration_record["generation_family"],
            "coordinator_lock": self.coordinator_lock_status,
            "source": {
                "profile": self.old_profile.name,
                "profile_sha256": self.source_profile_sha256,
                "config_sha256": self.source_authoritative_sha256,
                "config_sha256_authority": "autonomous-provenance.json/migration-chain",
                "current_canonical_sha256": self.source_canonical_sha256,
                "canonical_matches_authority": (
                    self.source_canonical_sha256 == self.source_authoritative_sha256
                ),
            },
            "target": {
                "profile": self.target_profile.name,
                "profile_sha256": self.target_profile_sha256,
                "config_sha256": self.target_canonical_sha256,
            },
            "boundary": {
                "learner_step": self.learner_step,
                "examples_consumed": self.examples_consumed,
                "committed_replay_samples": self.committed_replay_samples,
                "champion_model_identity": self.champion_model_identity,
                "discarded_uncheckpointed_steps": self.migration_record.get(
                    "discarded_uncheckpointed_steps", 0
                ),
                "target_updates_per_new_sample": self.utd_segment_payload[
                    "target_updates_per_new_sample"
                ],
            },
            "changes": [
                {"path": path, "from": old, "to": new}
                for path, old, new in self.changes
            ],
            "backup_bundle": str(backup or self.backup_directory),
            "writes": writes,
        }


def _sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _verified_profile_checksum(
    run_root: Path,
    profile: Path,
    *,
    actual_sha256: str,
) -> Path:
    candidates = (profile.with_suffix(".sha256"), run_root / "profile.sha256")
    failures = []
    for path in candidates:
        if not path.is_file():
            continue
        text = _read_bytes(path, "profile checksum").decode("utf-8").strip()
        parts = text.split(maxsplit=1)
        try:
            recorded = _sha256_text("recorded profile checksum", parts[0])
        except (IndexError, MigrationError) as exc:
            failures.append(f"{path}: {exc}")
            continue
        if recorded != actual_sha256:
            failures.append(f"{path}: checksum does not match {profile.name}")
            continue
        if len(parts) == 2:
            recorded_path = Path(parts[1].strip())
            if recorded_path.name != profile.name:
                failures.append(f"{path}: checksum names {recorded_path.name}")
                continue
        return path
    detail = "; ".join(failures) if failures else "no checksum file exists"
    raise MigrationError(f"source frozen profile is not authenticated: {detail}")


def canonical_config_sha256(config: ExperimentConfig) -> str:
    """Use the coordinator's canonical autonomous-profile hash algorithm."""

    encoded = json.dumps(
        config.as_dict(),
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _json_bytes(payload: Mapping[str, object]) -> bytes:
    return (json.dumps(payload, sort_keys=True, separators=(",", ":")) + "\n").encode(
        "utf-8"
    )


def _read_bytes(path: Path, name: str) -> bytes:
    if path.is_symlink():
        raise MigrationError(f"{name} may not be a symbolic link: {path}")
    try:
        data = path.read_bytes()
    except OSError as exc:
        raise MigrationError(f"cannot read {name} {path}: {exc}") from exc
    if not path.is_file():
        raise MigrationError(f"{name} is not a regular file: {path}")
    return data


def _read_json(path: Path, name: str) -> tuple[dict[str, Any], bytes]:
    data = _read_bytes(path, name)
    try:
        payload = json.loads(data)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise MigrationError(f"cannot parse {name} {path}: {exc}") from exc
    if not isinstance(payload, dict):
        raise MigrationError(f"{name} must be a JSON object")
    return payload, data


def _sha256_text(name: str, value: object) -> str:
    if not isinstance(value, str) or _SHA256.fullmatch(value) is None:
        raise MigrationError(f"{name} must be a lowercase SHA-256")
    return value


def _commit_text(name: str, value: object) -> str:
    if not isinstance(value, str) or _COMMIT.fullmatch(value) is None:
        raise MigrationError(f"{name} must be a 7-64 character lowercase Git hash")
    return value


def _identifier(name: str, value: object) -> str:
    if not isinstance(value, str) or _IDENTIFIER.fullmatch(value) is None:
        raise MigrationError(f"{name} is invalid")
    return value


def _nonnegative_int(name: str, value: object) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise MigrationError(f"{name} must be a non-negative integer")
    return value


def _positive_int(name: str, value: object) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise MigrationError(f"{name} must be a positive integer")
    return value


def _profile_name(name: object) -> str:
    if not isinstance(name, str) or _PROFILE_NAME.fullmatch(name) is None:
        raise MigrationError(
            "profile names must be safe .yaml/.yml basenames under the run root"
        )
    return name


def _load_profile(path: Path, name: str) -> ExperimentConfig:
    if path.is_symlink():
        raise MigrationError(f"{name} may not be a symbolic link: {path}")
    try:
        return load_config(path)
    except (OSError, ValueError) as exc:
        raise MigrationError(f"cannot load {name} {path}: {exc}") from exc


def _configured_root(config: ExperimentConfig) -> Path:
    return Path(config.orchestration.directories.root).expanduser().resolve()


def _is_sequence(value: object) -> TypeGuard[Sequence[object]]:
    return isinstance(value, Sequence) and not isinstance(
        value, (str, bytes, bytearray)
    )


def _profile_diffs(
    old: object,
    new: object,
    path: tuple[str, ...] = (),
) -> Iterator[tuple[tuple[str, ...], object, object]]:
    if isinstance(old, Mapping) and isinstance(new, Mapping):
        keys = sorted(set(old) | set(new), key=str)
        for key in keys:
            old_value = old.get(key, _MISSING)
            new_value = new.get(key, _MISSING)
            yield from _profile_diffs(
                old_value,
                new_value,
                (*path, str(key)),
            )
        return
    if _is_sequence(old) and _is_sequence(new):
        old_values = list(old)
        new_values = list(new)
        for index in range(max(len(old_values), len(new_values))):
            old_value = old_values[index] if index < len(old_values) else _MISSING
            new_value = new_values[index] if index < len(new_values) else _MISSING
            yield from _profile_diffs(
                old_value,
                new_value,
                (*path, str(index)),
            )
        return
    if old != new:
        yield path, old, new


def _json_value(value: object) -> object:
    return "<missing>" if value is _MISSING else value


def _is_allowed_profile_path(path: tuple[str, ...]) -> bool:
    if path in _ALLOWED_PROFILE_PATHS:
        return True
    if (
        len(path) == 4
        and path[:2] == ("orchestration", "gpus")
        and path[2].isdigit()
        and path[3] == "actor_lanes"
    ):
        return True
    # Evaluation configuration is intentionally isolated from training and replay.
    # Supporting this named subtree keeps future typed evaluation settings migratable
    # without opening arbitrary orchestration fields.
    return (
        len(path) >= 1
        and path[0] == "evaluation"
        or len(path) >= 2
        and path[:2]
        in {
            ("orchestration", "evaluation"),
            ("orchestration", "historical_evaluation"),
        }
    )


def _validate_gpu_topology_pair(
    old: ExperimentConfig,
    new: ExperimentConfig,
) -> None:
    old_gpus = old.orchestration.gpus
    new_gpus = new.orchestration.gpus
    if len(old_gpus) != len(new_gpus):
        raise MigrationError("GPU topology cannot add or remove physical GPUs")
    changed_lanes = []
    for index, (before, after) in enumerate(zip(old_gpus, new_gpus, strict=True)):
        immutable = (
            "gpu_id",
            "role",
            "cpu_threads",
            "actor_batch_size",
            "cpu_affinity",
        )
        if any(getattr(before, name) != getattr(after, name) for name in immutable):
            raise MigrationError(
                f"GPU topology changed immutable fields at orchestration.gpus.{index}"
            )
        if before.actor_lanes != after.actor_lanes:
            changed_lanes.append(before.gpu_id)

    old_promotion = old.orchestration.promotion
    new_promotion = new.orchestration.promotion
    old_by_id = {gpu.gpu_id: gpu for gpu in old_gpus}
    new_by_id = {gpu.gpu_id: gpu for gpu in new_gpus}
    new_shared = new_by_id.get(new_promotion.gpu_id)
    if new_shared is not None and new_shared.role == "learner" and (
        new_promotion.max_waves_per_lease != 1
        or new_promotion.inter_wave_cooldown_seconds < 1_800
        or new.orchestration.historical_evaluation.enabled
    ):
        raise MigrationError(
            "learner-shared promotion requires one-wave leases, a 30-minute "
            "cooldown, and disabled historical evaluation"
        )
    if old_promotion.gpu_id == new_promotion.gpu_id:
        if changed_lanes:
            raise MigrationError("actor lane changes require moving promotion off that GPU")
        return
    if (
        not old_promotion.pause_sharing_mode
        or not new_promotion.pause_sharing_mode
    ):
        raise MigrationError("GPU topology migration must retain pause sharing")
    old_shared = old_by_id.get(old_promotion.gpu_id)
    if old_shared is None or old_shared.role != "actor":
        raise MigrationError("source promotion GPU must be an actor GPU")
    if new_shared is None or new_shared.role != "learner":
        raise MigrationError("target promotion GPU must be a learner GPU")
    if (
        changed_lanes != [old_shared.gpu_id]
        or old_shared.actor_lanes != 1
        or new_by_id[old_shared.gpu_id].actor_lanes != 2
    ):
        raise MigrationError(
            "topology migration may only raise the former arena actor "
            "from one lane to two"
        )


def _validate_profile_pair(
    old: ExperimentConfig,
    new: ExperimentConfig,
    *,
    run_root: Path,
) -> tuple[tuple[str, object, object], ...]:
    for label, config in (("old", old), ("new", new)):
        if (
            config.profile != "continuous"
            or not config.orchestration.enabled
            or not config.orchestration.autonomous.enabled
            or not config.learner.unlimited
            or not config.learner.resume_latest
        ):
            raise MigrationError(
                f"{label} profile is not an active resumable autonomous profile"
            )

    old_run_id = old.orchestration.run_id
    new_run_id = new.orchestration.run_id
    if not old_run_id or old_run_id != new_run_id:
        raise MigrationError("old and new profiles must have the same explicit run_id")
    old_root = _configured_root(old)
    new_root = _configured_root(new)
    if old_root != new_root or old_root != run_root:
        raise MigrationError(
            "old profile, new profile, and requested run root must resolve identically"
        )
    if old.schema_version != new.schema_version:
        raise MigrationError("configuration schema_version is immutable")
    if old.data.schema_version != new.data.schema_version:
        raise MigrationError("replay data schema_version is immutable")
    if old.train.seed != new.train.seed:
        raise MigrationError("training seed is immutable")
    for section in ("game", "model", "loss", "optimizer"):
        if getattr(old, section) != getattr(new, section):
            raise MigrationError(f"{section} configuration is immutable")
    _validate_gpu_topology_pair(old, new)

    differences = tuple(_profile_diffs(old.as_dict(), new.as_dict()))
    disallowed = [
        ".".join(path)
        for path, _, _ in differences
        if not _is_allowed_profile_path(path)
    ]
    if disallowed:
        raise MigrationError(
            "profile changes immutable or replay-critical fields: "
            + ", ".join(disallowed)
        )
    if not differences:
        raise MigrationError("new profile has no semantic changes")
    return tuple(
        (".".join(path), _json_value(old_value), _json_value(new_value))
        for path, old_value, new_value in differences
    )


def _pid_is_live(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


def _coordinator_lock_status(run_root: Path) -> tuple[str, bytes | None]:
    path = run_root / "coordinator.lock"
    if not path.exists():
        return "absent", None
    payload, data = _read_json(path, "coordinator lock")
    pid = _positive_int("coordinator lock pid", payload.get("pid"))
    _positive_int("coordinator lock created_ns", payload.get("created_ns"))
    if _pid_is_live(pid):
        raise MigrationError(f"coordinator lock PID {pid} is live")
    return f"stale-dead-pid:{pid}", data


def _validate_run_identity(payload: Mapping[str, object]) -> tuple[str, str]:
    if payload.get("schema_version") != 1:
        raise MigrationError("run.json schema is incompatible")
    run_id = _identifier("run_id", payload.get("run_id"))
    family = _identifier("generation_family", payload.get("generation_family"))
    _positive_int("run identity created_ns", payload.get("created_ns"))
    return run_id, family


def _validate_provenance(
    payload: Mapping[str, object],
    *,
    run_id: str,
    generation_family: str,
    config: ExperimentConfig,
) -> str:
    required = {
        "schema_version",
        "mode",
        "run_id",
        "generation_family",
        "train_seed",
        "elo_anchor_step",
        "external_weights",
        "external_replay",
        "external_positions",
        "config_sha256",
    }
    if set(payload) != required:
        raise MigrationError("autonomous provenance fields are incompatible")
    expected = {
        "schema_version": 1,
        "mode": "random-init-selfplay-only",
        "run_id": run_id,
        "generation_family": generation_family,
        "train_seed": config.train.seed,
        "elo_anchor_step": config.orchestration.autonomous.elo_anchor_step,
        "external_weights": False,
        "external_replay": False,
        "external_positions": False,
    }
    if any(payload.get(key) != value for key, value in expected.items()):
        raise MigrationError("autonomous provenance disagrees with the source run")
    return _sha256_text("provenance config_sha256", payload.get("config_sha256"))


def _read_migration_chain(
    path: Path,
    *,
    run_id: str,
    generation_family: str,
    provenance_sha256: str,
    old_profile_name: str,
) -> tuple[tuple[dict[str, Any], ...], bytes | None]:
    if not path.exists():
        return (), None
    data = _read_bytes(path, "autonomous migration chain")
    records: list[dict[str, Any]] = []
    required = {
        "schema_version",
        "timestamp_ns",
        "run_id",
        "generation_family",
        "from_config_sha256",
        "to_config_sha256",
        "from_profile",
        "to_profile",
        "learner_step",
        "examples_consumed",
        "from_source_commit",
        "to_source_commit",
        "reason",
    }
    for line_number, raw_line in enumerate(data.splitlines(), start=1):
        if not raw_line.strip():
            raise MigrationError(
                f"autonomous migration chain line {line_number} is empty"
            )
        try:
            record = json.loads(raw_line)
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise MigrationError(
                f"autonomous migration chain line {line_number} is invalid: {exc}"
            ) from exc
        if not isinstance(record, dict) or not required <= set(record):
            raise MigrationError(
                f"autonomous migration chain line {line_number} fields are invalid"
            )
        if record.get("schema_version") != MIGRATION_SCHEMA_VERSION:
            raise MigrationError("autonomous migration schema version is incompatible")
        if (
            record.get("run_id") != run_id
            or record.get("generation_family") != generation_family
        ):
            raise MigrationError("autonomous migration chain run identity mismatch")
        _positive_int("migration timestamp_ns", record.get("timestamp_ns"))
        _sha256_text("migration from_config_sha256", record.get("from_config_sha256"))
        _sha256_text("migration to_config_sha256", record.get("to_config_sha256"))
        _profile_name(record.get("from_profile"))
        _profile_name(record.get("to_profile"))
        _nonnegative_int("migration learner_step", record.get("learner_step"))
        _nonnegative_int("migration examples_consumed", record.get("examples_consumed"))
        if "committed_replay_samples" in record:
            _nonnegative_int(
                "migration committed_replay_samples",
                record.get("committed_replay_samples"),
            )
        _commit_text("migration from_source_commit", record.get("from_source_commit"))
        _commit_text("migration to_source_commit", record.get("to_source_commit"))
        reason = record.get("reason")
        if not isinstance(reason, str) or not reason.strip() or "\n" in reason:
            raise MigrationError("migration reason is invalid")
        if records:
            previous = records[-1]
            if (
                record["timestamp_ns"] <= previous["timestamp_ns"]
                or record["from_config_sha256"] != previous["to_config_sha256"]
                or record["from_profile"] != previous["to_profile"]
                or record["from_source_commit"] != previous["to_source_commit"]
            ):
                raise MigrationError("autonomous migration chain is discontinuous")
        records.append(record)
    if not records:
        raise MigrationError("autonomous migration chain exists but is empty")
    last = records[-1]
    if last["to_config_sha256"] != provenance_sha256:
        raise MigrationError(
            "autonomous migration chain does not end at provenance config_sha256"
        )
    if last["to_profile"] != old_profile_name:
        raise MigrationError(
            "source profile name does not match the migration chain head"
        )
    return tuple(records), data


def _validate_heartbeat(payload: Mapping[str, object]) -> tuple[int, int | None]:
    if payload.get("schema_version") != 1 or payload.get("worker") != "learner":
        raise MigrationError("learner heartbeat identity is incompatible")
    pid = _positive_int("learner heartbeat pid", payload.get("pid"))
    if _pid_is_live(pid):
        raise MigrationError(f"learner heartbeat PID {pid} is still live")
    _positive_int("learner heartbeat_ns", payload.get("heartbeat_ns"))
    phase = payload.get("phase")
    if not isinstance(phase, str) or not phase:
        raise MigrationError("learner heartbeat phase is invalid")
    step = _nonnegative_int("learner heartbeat step", payload.get("step"))
    examples_value = payload.get("examples_consumed")
    examples = (
        None
        if examples_value is None
        else _nonnegative_int("learner heartbeat examples_consumed", examples_value)
    )
    return step, examples


def _validate_recovery(
    payload: Mapping[str, object],
    *,
    path: Path,
    run_id: str,
    generation_family: str,
) -> tuple[int, int, str, Path]:
    if (
        payload.get("format") != "startrain.recovery-pointer"
        or payload.get("schema_version") != 1
        or payload.get("run_id") != run_id
        or payload.get("generation_family") != generation_family
    ):
        raise MigrationError("learner recovery pointer is incompatible")
    step = _nonnegative_int("recovery step", payload.get("step"))
    _nonnegative_int("recovery epoch", payload.get("epoch"))
    examples = _nonnegative_int(
        "recovery examples_consumed", payload.get("examples_consumed")
    )
    _positive_int("recovery updated_ns", payload.get("updated_ns"))
    checkpoint_sha256 = _sha256_text(
        "recovery checkpoint_sha256", payload.get("checkpoint_sha256")
    )
    checkpoint_bytes = _positive_int(
        "recovery checkpoint_bytes", payload.get("checkpoint_bytes")
    )
    checkpoint_value = payload.get("checkpoint")
    if not isinstance(checkpoint_value, str) or not checkpoint_value:
        raise MigrationError("recovery checkpoint path is invalid")
    checkpoint = Path(checkpoint_value)
    if not checkpoint.is_absolute():
        checkpoint = path.parent / checkpoint
    if checkpoint.is_symlink():
        raise MigrationError("recovery checkpoint may not be a symbolic link")
    checkpoint = checkpoint.resolve()
    expected_directory = (path.parent / "recovery").resolve()
    if (
        checkpoint.parent != expected_directory
        or checkpoint.name != f"sha256-{checkpoint_sha256}.pt"
    ):
        raise MigrationError("recovery checkpoint escaped its content-addressed root")
    if not checkpoint.is_file() or checkpoint.stat().st_size != checkpoint_bytes:
        raise MigrationError("recovery checkpoint byte length is invalid")
    if _sha256_file(checkpoint) != checkpoint_sha256:
        raise MigrationError("recovery checkpoint SHA-256 is invalid")
    return step, examples, checkpoint_sha256, checkpoint


def _validate_cadence(
    payload: Mapping[str, object],
    *,
    run_id: str,
    generation_family: str,
) -> tuple[int, int | None]:
    if (
        payload.get("schema_version") != 1
        or payload.get("run_id") != run_id
        or payload.get("generation_family") != generation_family
    ):
        raise MigrationError("learner cadence state is incompatible")
    candidate = _nonnegative_int(
        "cadence candidate_examples", payload.get("candidate_examples")
    )
    selfplay_value = payload.get("selfplay_examples")
    selfplay = (
        None
        if selfplay_value is None
        else _nonnegative_int("cadence selfplay_examples", selfplay_value)
    )
    _positive_int("cadence updated_ns", payload.get("updated_ns"))
    return candidate, selfplay


def _validate_champion(
    payload: Mapping[str, object],
    *,
    run_id: str,
    generation_family: str,
) -> tuple[int, str, str]:
    if (
        payload.get("format") != "startrain.model-pointer"
        or payload.get("schema_version") != 2
        or payload.get("role") != "champion"
        or payload.get("run_id") != run_id
        or payload.get("generation_family") != generation_family
    ):
        raise MigrationError("learner champion pointer is incompatible")
    step = _nonnegative_int("champion model_step", payload.get("model_step"))
    identity = _identifier("champion model_identity", payload.get("model_identity"))
    if not identity.startswith("sha256-") or _SHA256.fullmatch(identity[7:]) is None:
        raise MigrationError("champion model_identity is not content-addressed")
    manifest_sha256 = _sha256_text(
        "champion manifest_sha256", payload.get("manifest_sha256")
    )
    _positive_int("champion manifest_bytes", payload.get("manifest_bytes"))
    _positive_int("champion updated_ns", payload.get("updated_ns"))
    manifest = payload.get("manifest")
    if not isinstance(manifest, str) or not manifest:
        raise MigrationError("champion manifest path is invalid")
    return step, identity, manifest_sha256


def _read_replay_boundary(
    run_root: Path,
    *,
    run_id: str,
    generation_family: str,
) -> int:
    path = run_root / "replay" / "manifest.sqlite3"
    if not path.is_file():
        raise MigrationError(f"replay manifest is missing: {path}")
    uri = f"{path.resolve().as_uri()}?mode=ro"
    try:
        with sqlite3.connect(uri, uri=True, timeout=30.0) as connection:
            connection.execute("PRAGMA query_only = ON")
            run = connection.execute(
                "SELECT generation_family FROM runs WHERE run_id = ?",
                (run_id,),
            ).fetchone()
            if run is None or run[0] != generation_family:
                raise MigrationError("replay ledger run identity mismatch")
            row = connection.execute(
                """
                SELECT committed_samples, history_complete
                FROM run_counters
                WHERE run_id = ? AND generation_family = ?
                """,
                (run_id, generation_family),
            ).fetchone()
    except sqlite3.Error as exc:
        raise MigrationError(
            f"cannot read replay ledger in read-only mode: {exc}"
        ) from exc
    if row is None:
        raise MigrationError("replay ledger has no cumulative run counter")
    committed = _nonnegative_int("committed replay samples", row[0])
    if row[1] != 1:
        raise MigrationError("committed replay history_complete is false")
    return committed


def _source_commit_from_file(run_root: Path) -> str | None:
    path = run_root / "source-commit.txt"
    if not path.is_file():
        return None
    try:
        value = path.read_text(encoding="utf-8").strip().split()[0]
    except (OSError, IndexError, UnicodeDecodeError) as exc:
        raise MigrationError(f"cannot read source-commit.txt: {exc}") from exc
    return _commit_text("source-commit.txt", value)


def _current_git_commit() -> str:
    training_root = Path(__file__).resolve().parents[1]
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=training_root,
            check=True,
            capture_output=True,
            text=True,
            timeout=10,
        )
    except (OSError, subprocess.SubprocessError) as exc:
        raise MigrationError(
            "cannot infer target source commit; pass --to-source-commit"
        ) from exc
    return _commit_text("current Git commit", result.stdout.strip())


def _resolve_source_commits(
    request: MigrationRequest,
    *,
    run_root: Path,
    chain: Sequence[Mapping[str, object]],
) -> tuple[str, str]:
    chain_source = str(chain[-1]["to_source_commit"]) if chain else None
    from_commit = (
        _commit_text("from_source_commit", request.from_source_commit)
        if request.from_source_commit is not None
        else chain_source or _source_commit_from_file(run_root)
    )
    if from_commit is None:
        raise MigrationError(
            "source commit is unknown; pass --from-source-commit or provide "
            "source-commit.txt"
        )
    if chain_source is not None and from_commit != chain_source:
        raise MigrationError("from_source_commit disagrees with migration chain head")
    to_commit = (
        _commit_text("to_source_commit", request.to_source_commit)
        if request.to_source_commit is not None
        else _current_git_commit()
    )
    return from_commit, to_commit


def _artifact(path: Path, *, run_root: Path, name: str) -> _BackupArtifact:
    resolved = path.resolve()
    try:
        relative = resolved.relative_to(run_root)
    except ValueError as exc:
        raise MigrationError(f"{name} is outside the run root: {resolved}") from exc
    data = _read_bytes(resolved, name)
    return _BackupArtifact(
        source=resolved,
        relative_path=relative,
        data=data,
        mode=resolved.stat().st_mode & 0o777,
    )


def plan_migration(request: MigrationRequest) -> MigrationPlan:
    old_profile = request.old_profile.expanduser().resolve()
    new_profile = request.new_profile.expanduser().resolve()
    target_name = _profile_name(request.target_profile_name)
    reason = request.reason.strip()
    if not reason or "\n" in reason or len(reason) > 256:
        raise MigrationError("migration reason must be 1-256 characters on one line")

    old_config = _load_profile(old_profile, "old profile")
    new_config = _load_profile(new_profile, "new profile")
    configured_root = _configured_root(old_config)
    run_root = (
        configured_root
        if request.run_root is None
        else request.run_root.expanduser().resolve()
    )
    if not run_root.is_dir() or run_root.is_symlink():
        raise MigrationError(f"run root must be an existing real directory: {run_root}")
    if old_profile.parent != run_root:
        raise MigrationError("old frozen profile must be directly under the run root")
    target_profile = run_root / target_name
    if target_profile == old_profile:
        raise MigrationError("target profile name must differ from the source profile")
    if target_profile.exists():
        raise MigrationError(f"target frozen profile already exists: {target_profile}")
    if new_profile == target_profile:
        raise MigrationError(
            "new profile must be staged outside its install destination"
        )

    changes = _validate_profile_pair(old_config, new_config, run_root=run_root)
    lock_status, lock_data = _coordinator_lock_status(run_root)

    run_path = run_root / "run.json"
    provenance_path = run_root / "autonomous-provenance.json"
    heartbeat_path = run_root / "status" / "learner.heartbeat.json"
    recovery_path = run_root / "learner" / "recovery.json"
    cadence_path = run_root / "learner" / "cadence.json"
    champion_path = run_root / "learner" / "champion.json"
    migrations_path = run_root / "autonomous-migrations.jsonl"

    run_payload, _ = _read_json(run_path, "run identity")
    run_id, generation_family = _validate_run_identity(run_payload)
    if old_config.orchestration.run_id != run_id:
        raise MigrationError("profile run_id does not match run.json")

    provenance_payload, _ = _read_json(provenance_path, "autonomous provenance")
    source_authoritative_sha256 = _validate_provenance(
        provenance_payload,
        run_id=run_id,
        generation_family=generation_family,
        config=old_config,
    )
    chain, migration_chain_data = _read_migration_chain(
        migrations_path,
        run_id=run_id,
        generation_family=generation_family,
        provenance_sha256=source_authoritative_sha256,
        old_profile_name=old_profile.name,
    )
    from_source_commit, to_source_commit = _resolve_source_commits(
        request,
        run_root=run_root,
        chain=chain,
    )

    heartbeat_payload, _ = _read_json(heartbeat_path, "learner heartbeat")
    recovery_payload, _ = _read_json(recovery_path, "learner recovery")
    cadence_payload, _ = _read_json(cadence_path, "learner cadence")
    champion_payload, champion_data = _read_json(champion_path, "learner champion")

    heartbeat_step, heartbeat_examples = _validate_heartbeat(heartbeat_payload)
    (
        learner_step,
        examples_consumed,
        recovery_checkpoint_sha256,
        recovery_checkpoint,
    ) = _validate_recovery(
        recovery_payload,
        path=recovery_path,
        run_id=run_id,
        generation_family=generation_family,
    )
    candidate_examples, selfplay_examples = _validate_cadence(
        cadence_payload,
        run_id=run_id,
        generation_family=generation_family,
    )
    champion_step, champion_identity, champion_manifest_sha256 = _validate_champion(
        champion_payload,
        run_id=run_id,
        generation_family=generation_family,
    )
    recovery_interval = old_config.learner.recovery_interval_steps
    discarded_uncheckpointed_steps = heartbeat_step - learner_step
    if discarded_uncheckpointed_steps < 0:
        raise MigrationError("learner heartbeat is behind the recovery boundary")
    if (
        discarded_uncheckpointed_steps
        and (
            recovery_interval is None
            or discarded_uncheckpointed_steps >= recovery_interval
        )
    ):
        raise MigrationError(
            "learner heartbeat is too far ahead of the recovery boundary"
        )
    if candidate_examples > examples_consumed or (
        selfplay_examples is not None and selfplay_examples > examples_consumed
    ):
        raise MigrationError("learner cadence is ahead of the recovery boundary")
    if champion_step > learner_step:
        raise MigrationError("champion step is ahead of the recovery boundary")

    committed_replay_samples = _read_replay_boundary(
        run_root,
        run_id=run_id,
        generation_family=generation_family,
    )
    if chain:
        chain_head = chain[-1]
        if learner_step < int(chain_head["learner_step"]) or examples_consumed < int(
            chain_head["examples_consumed"]
        ):
            raise MigrationError("learner boundary moves behind the migration chain")
        previous_replay = chain_head.get("committed_replay_samples")
        if (
            isinstance(previous_replay, int)
            and not isinstance(previous_replay, bool)
            and committed_replay_samples < previous_replay
        ):
            raise MigrationError("replay boundary moves behind the migration chain")
    target = new_config.learner.target_updates_per_new_sample
    if target is None:
        raise MigrationError("target profile must use update-to-data control")
    old_target = old_config.learner.target_updates_per_new_sample
    if old_target is None:
        raise MigrationError("source profile must use update-to-data control")
    previous_utd_segment: dict[str, object] | None = None
    previous_utd_path = run_root / "learner" / "utd-segment.json"
    if float(old_target) == float(target) and previous_utd_path.is_file():
        payload, _ = _read_json(previous_utd_path, "previous UTD segment")
        required = {
            "schema_version",
            "run_id",
            "generation_family",
            "target_updates_per_new_sample",
            "baseline_examples_consumed",
            "baseline_committed_replay_samples",
        }
        if not required <= set(payload):
            raise MigrationError("previous UTD segment is incomplete")
        if (
            payload.get("schema_version") != UTD_SEGMENT_SCHEMA_VERSION
            or payload.get("run_id") != run_id
            or payload.get("generation_family") != generation_family
            or payload.get("target_updates_per_new_sample") != float(target)
        ):
            raise MigrationError("previous UTD segment is incompatible")
        previous_examples = _nonnegative_int(
            "previous UTD baseline examples",
            payload.get("baseline_examples_consumed"),
        )
        previous_samples = _nonnegative_int(
            "previous UTD baseline replay samples",
            payload.get("baseline_committed_replay_samples"),
        )
        if (
            previous_examples > examples_consumed
            or previous_samples > committed_replay_samples
        ):
            raise MigrationError("previous UTD segment is ahead of the boundary")
        previous_utd_segment = payload

    old_profile_bytes = _read_bytes(old_profile, "old profile")
    new_profile_bytes = _read_bytes(new_profile, "new profile")
    source_profile_sha256 = _sha256_bytes(old_profile_bytes)
    target_profile_sha256 = _sha256_bytes(new_profile_bytes)
    source_profile_checksum = _verified_profile_checksum(
        run_root,
        old_profile,
        actual_sha256=source_profile_sha256,
    )
    target_profile_checksum = target_profile.with_suffix(".sha256")
    source_canonical_sha256 = canonical_config_sha256(old_config)
    target_canonical_sha256 = canonical_config_sha256(new_config)
    if target_canonical_sha256 == source_authoritative_sha256:
        raise MigrationError("target canonical hash does not advance provenance")

    timestamp_ns = time.time_ns()
    if chain and timestamp_ns <= int(chain[-1]["timestamp_ns"]):
        raise MigrationError("system clock does not advance the migration chain")
    migration_record: dict[str, object] = {
        "schema_version": MIGRATION_SCHEMA_VERSION,
        "timestamp_ns": timestamp_ns,
        "run_id": run_id,
        "generation_family": generation_family,
        "from_config_sha256": source_authoritative_sha256,
        "to_config_sha256": target_canonical_sha256,
        "from_profile": old_profile.name,
        "to_profile": target_name,
        "learner_step": learner_step,
        "examples_consumed": examples_consumed,
        "discarded_uncheckpointed_steps": discarded_uncheckpointed_steps,
        "heartbeat_examples_consumed": heartbeat_examples,
        "from_source_commit": from_source_commit,
        "to_source_commit": to_source_commit,
        "reason": reason,
        "committed_replay_samples": committed_replay_samples,
        "target_updates_per_new_sample": float(target),
        "from_profile_sha256": source_profile_sha256,
        "to_profile_sha256": target_profile_sha256,
        "recovery_checkpoint_sha256": recovery_checkpoint_sha256,
        "champion_model_identity": champion_identity,
        "champion_model_step": champion_step,
        "champion_manifest_sha256": champion_manifest_sha256,
        "champion_pointer_sha256": _sha256_bytes(champion_data),
    }
    updated_provenance = dict(provenance_payload)
    updated_provenance["config_sha256"] = target_canonical_sha256
    utd_segment: dict[str, object] = previous_utd_segment or {
        "schema_version": UTD_SEGMENT_SCHEMA_VERSION,
        "run_id": run_id,
        "generation_family": generation_family,
        "target_updates_per_new_sample": float(target),
        "baseline_examples_consumed": examples_consumed,
        "baseline_committed_replay_samples": committed_replay_samples,
        "created_ns": timestamp_ns,
    }

    required_paths = (
        (old_profile, "old profile"),
        (run_path, "run identity"),
        (provenance_path, "autonomous provenance"),
        (heartbeat_path, "learner heartbeat"),
        (recovery_path, "learner recovery"),
        (recovery_checkpoint, "learner recovery checkpoint"),
        (cadence_path, "learner cadence"),
        (champion_path, "learner champion"),
        (source_profile_checksum, "source profile checksum"),
    )
    optional_paths = (
        *(
            ((run_root / "profile.sha256", "legacy profile checksum"),)
            if source_profile_checksum != run_root / "profile.sha256"
            else ()
        ),
        (run_root / "source-commit.txt", "source commit"),
        (run_root / "status" / "coordinator.json", "coordinator status"),
        (run_root / "learner" / "candidate.json", "learner candidate"),
        (
            run_root / "learner" / "selfplay" / "candidate.json",
            "self-play candidate",
        ),
        (run_root / "learner" / "resume-cutover.json", "resume cutover"),
        (run_root / "learner" / "utd-segment.json", "previous UTD segment"),
    )
    artifacts = [
        _artifact(path, run_root=run_root, name=name) for path, name in required_paths
    ]
    for path, name in optional_paths:
        if path.exists():
            artifacts.append(_artifact(path, run_root=run_root, name=name))
    if migration_chain_data is not None:
        artifacts.append(
            _artifact(
                migrations_path,
                run_root=run_root,
                name="autonomous migration chain",
            )
        )
    if lock_data is not None:
        artifacts.append(
            _artifact(
                run_root / "coordinator.lock",
                run_root=run_root,
                name="stale coordinator lock",
            )
        )
    artifacts.sort(key=lambda artifact: str(artifact.relative_path))

    input_paths = {artifact.source for artifact in artifacts}
    input_paths.add(new_profile)
    input_hash_values = {path: _sha256_file(path) for path in input_paths}
    input_hash_values[recovery_checkpoint] = recovery_checkpoint_sha256
    input_hashes = tuple(
        sorted(input_hash_values.items(), key=lambda item: str(item[0]))
    )
    expected_absent = [target_profile, target_profile_checksum]
    for path in (
        migrations_path,
        run_root / "profile.sha256",
        run_root / "learner" / "utd-segment.json",
    ):
        if not path.exists():
            expected_absent.append(path)

    profile_sha256_bytes = (
        f"{target_profile_sha256}  {target_profile}\n".encode("utf-8")
    )
    backup_directory = (
        run_root / "migration-backups" / f"{timestamp_ns}-{target_profile.stem}"
    )
    return MigrationPlan(
        run_root=run_root,
        old_profile=old_profile,
        new_profile=new_profile,
        target_profile=target_profile,
        target_profile_checksum=target_profile_checksum,
        target_profile_bytes=new_profile_bytes,
        target_profile_mode=0o444,
        backup_directory=backup_directory,
        backup_artifacts=tuple(artifacts),
        input_hashes=input_hashes,
        expected_absent=tuple(expected_absent),
        migration_log_existed=migration_chain_data is not None,
        migration_record=migration_record,
        provenance_payload=updated_provenance,
        utd_segment_payload=utd_segment,
        profile_sha256_bytes=profile_sha256_bytes,
        source_canonical_sha256=source_canonical_sha256,
        source_authoritative_sha256=source_authoritative_sha256,
        target_canonical_sha256=target_canonical_sha256,
        source_profile_sha256=source_profile_sha256,
        target_profile_sha256=target_profile_sha256,
        changes=changes,
        learner_step=learner_step,
        examples_consumed=examples_consumed,
        committed_replay_samples=committed_replay_samples,
        champion_model_identity=champion_identity,
        coordinator_lock_status=lock_status,
    )


def _fsync_directory(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _atomic_write_bytes(path: Path, data: bytes, *, mode: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_name: str | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="wb",
            prefix=f".{path.name}.",
            suffix=".tmp",
            dir=path.parent,
            delete=False,
        ) as stream:
            temporary_name = stream.name
            os.fchmod(stream.fileno(), mode)
            stream.write(data)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary_name, path)
        temporary_name = None
        _fsync_directory(path.parent)
    finally:
        if temporary_name is not None:
            Path(temporary_name).unlink(missing_ok=True)


def _append_jsonl(path: Path, payload: Mapping[str, object]) -> None:
    existed = path.exists()
    prefix = b""
    if existed and path.stat().st_size:
        with path.open("rb") as stream:
            stream.seek(-1, os.SEEK_END)
            if stream.read(1) != b"\n":
                prefix = b"\n"
    data = prefix + _json_bytes(payload)
    descriptor = os.open(
        path,
        os.O_APPEND | os.O_CREAT | os.O_WRONLY,
        0o644,
    )
    try:
        written = 0
        while written < len(data):
            count = os.write(descriptor, data[written:])
            if count <= 0:
                raise OSError("short autonomous migration record write")
            written += count
        os.fsync(descriptor)
    finally:
        os.close(descriptor)
    if not existed:
        _fsync_directory(path.parent)


def _write_new_backup_file(path: Path, data: bytes, mode: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor = os.open(path, os.O_CREAT | os.O_EXCL | os.O_WRONLY, mode)
    try:
        written = 0
        while written < len(data):
            count = os.write(descriptor, data[written:])
            if count <= 0:
                raise OSError("short backup write")
            written += count
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _create_backup(plan: MigrationPlan) -> Path:
    destination = plan.backup_directory
    if destination.exists():
        raise MigrationError(f"backup destination already exists: {destination}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    staging = destination.parent / (
        f".{destination.name}.tmp-{os.getpid()}-{uuid.uuid4().hex}"
    )
    staging.mkdir(mode=0o700)
    try:
        manifest_files: list[dict[str, object]] = []
        for artifact in plan.backup_artifacts:
            target = staging / artifact.relative_path
            _write_new_backup_file(target, artifact.data, artifact.mode)
            manifest_files.append(
                {
                    "path": str(artifact.relative_path),
                    "bytes": len(artifact.data),
                    "sha256": _sha256_bytes(artifact.data),
                }
            )
        manifest = {
            "schema_version": 1,
            "timestamp_ns": plan.migration_record["timestamp_ns"],
            "run_id": plan.migration_record["run_id"],
            "generation_family": plan.migration_record["generation_family"],
            "from_profile": plan.old_profile.name,
            "to_profile": plan.target_profile.name,
            "files": manifest_files,
        }
        _write_new_backup_file(staging / "manifest.json", _json_bytes(manifest), 0o444)
        directories = sorted(
            (path for path in staging.rglob("*") if path.is_dir()),
            key=lambda path: len(path.parts),
            reverse=True,
        )
        for directory in directories:
            _fsync_directory(directory)
        _fsync_directory(staging)
        os.replace(staging, destination)
        _fsync_directory(destination.parent)
    finally:
        if staging.exists():
            shutil.rmtree(staging)
    return destination


def _assert_inputs_unchanged(plan: MigrationPlan, *, check_lock: bool) -> None:
    if check_lock:
        _coordinator_lock_status(plan.run_root)
    for path, expected_sha256 in plan.input_hashes:
        if not path.is_file() or _sha256_file(path) != expected_sha256:
            raise MigrationError(f"validated input changed before apply: {path}")
    for path in plan.expected_absent:
        if path.exists():
            raise MigrationError(f"validated output appeared before apply: {path}")
    replay_samples = _read_replay_boundary(
        plan.run_root,
        run_id=str(plan.migration_record["run_id"]),
        generation_family=str(plan.migration_record["generation_family"]),
    )
    if replay_samples != plan.committed_replay_samples:
        raise MigrationError("committed replay count changed before apply")


@contextmanager
def _coordinator_write_guard(run_root: Path) -> Iterator[None]:
    path = run_root / "coordinator.lock"
    status, _ = _coordinator_lock_status(run_root)
    if status.startswith("stale-dead-pid:"):
        path.unlink()
        _fsync_directory(run_root)
    token = uuid.uuid4().hex
    payload = _json_bytes(
        {
            "pid": os.getpid(),
            "created_ns": time.time_ns(),
            "owner": "autonomous-profile-migration",
            "token": token,
        }
    )
    try:
        descriptor = os.open(path, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o644)
    except FileExistsError as exc:
        raise MigrationError("coordinator lock appeared before apply") from exc
    try:
        os.write(descriptor, payload)
        os.fsync(descriptor)
    finally:
        os.close(descriptor)
    _fsync_directory(run_root)
    try:
        yield
    finally:
        try:
            current, _ = _read_json(path, "migration coordinator lock")
        except MigrationError:
            current = {}
        if current.get("token") == token and current.get("pid") == os.getpid():
            path.unlink(missing_ok=True)
            _fsync_directory(run_root)


def apply_migration(plan: MigrationPlan) -> dict[str, object]:
    """Apply a fully validated migration plan under a coordinator exclusion lock."""

    _assert_inputs_unchanged(plan, check_lock=True)
    with _coordinator_write_guard(plan.run_root):
        _assert_inputs_unchanged(plan, check_lock=False)
        backup = _create_backup(plan)
        _atomic_write_bytes(
            plan.target_profile,
            plan.target_profile_bytes,
            mode=plan.target_profile_mode,
        )
        _atomic_write_bytes(
            plan.target_profile_checksum,
            plan.profile_sha256_bytes,
            mode=0o444,
        )
        _atomic_write_bytes(
            plan.run_root / "learner" / "utd-segment.json",
            _json_bytes(plan.utd_segment_payload),
            mode=0o644,
        )
        _append_jsonl(
            plan.run_root / "autonomous-migrations.jsonl",
            plan.migration_record,
        )
        _atomic_write_bytes(
            plan.run_root / "autonomous-provenance.json",
            _json_bytes(plan.provenance_payload),
            mode=0o644,
        )
        _atomic_write_bytes(
            plan.run_root / "profile.sha256",
            plan.profile_sha256_bytes,
            mode=0o644,
        )
    return plan.output(mode="apply", backup=backup)


def migrate_autonomous_profile(
    request: MigrationRequest,
    *,
    apply: bool = False,
) -> dict[str, object]:
    plan = plan_migration(request)
    if apply:
        return apply_migration(plan)
    return plan.output(mode="dry-run")


class _JSONArgumentParser(argparse.ArgumentParser):
    def error(self, message: str) -> NoReturn:
        raise MigrationError(message)


def _parser() -> argparse.ArgumentParser:
    parser = _JSONArgumentParser(description=__doc__)
    parser.add_argument("--run-root", type=Path)
    parser.add_argument(
        "--old-profile",
        "--source-profile",
        dest="old_profile",
        type=Path,
        required=True,
    )
    parser.add_argument(
        "--new-profile",
        "--target-profile",
        dest="new_profile",
        type=Path,
        required=True,
    )
    parser.add_argument(
        "--target-profile-name",
        "--new-profile-name",
        "--target-name",
        "--profile-name",
        dest="target_profile_name",
        required=True,
    )
    parser.add_argument("--reason", required=True)
    parser.add_argument("--from-source-commit")
    parser.add_argument("--to-source-commit", "--source-commit")
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--dry-run", action="store_true")
    mode.add_argument("--apply", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    mode = "dry-run"
    try:
        arguments = _parser().parse_args(argv)
        mode = "apply" if arguments.apply else "dry-run"
        request = MigrationRequest(
            old_profile=arguments.old_profile,
            new_profile=arguments.new_profile,
            target_profile_name=arguments.target_profile_name,
            reason=arguments.reason,
            run_root=arguments.run_root,
            from_source_commit=arguments.from_source_commit,
            to_source_commit=arguments.to_source_commit,
        )
        result = migrate_autonomous_profile(request, apply=arguments.apply)
    except Exception as exc:
        error = {
            "status": "error",
            "mode": mode,
            "error_type": type(exc).__name__,
            "error": str(exc),
        }
        print(json.dumps(error, sort_keys=True), file=sys.stderr)
        return 2
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
