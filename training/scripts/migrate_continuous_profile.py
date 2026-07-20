#!/usr/bin/env python3
"""Safely migrate a stopped, non-autonomous continuous run profile."""

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

import torch

from startrain.checkpoint import CHECKPOINT_FORMAT, CHECKPOINT_VERSION
from startrain.config import ExperimentConfig, load_config

if __package__:
    from scripts.validate_continuous_profile import validate_continuous_config
else:
    from validate_continuous_profile import validate_continuous_config


MIGRATION_SCHEMA_VERSION = 1
_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_COMMIT = re.compile(r"^[0-9a-f]{7,64}$")
_PROFILE_NAME = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,122}\.ya?ml$")
_IDENTIFIER = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$")
_MISSING = object()

_ALLOWED_PROFILE_PATHS = {
    ("train", "per_rank_batch_size"),
    ("learner", "candidate_interval"),
    ("learner", "max_replay_lag_steps"),
    ("orchestration", "plateau", "max_learner_champion_lag_steps"),
    ("orchestration", "promotion", "finish_inflight_candidate"),
    ("arena", "continuation_pairs_per_ring"),
}


class MigrationError(RuntimeError):
    """A fail-closed migration validation or transaction error."""


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
class _InputFingerprint:
    path: Path
    sha256: str
    size: int
    mode: int


@dataclass(frozen=True, slots=True)
class _FileState:
    path: Path
    data: bytes | None
    mode: int | None


@dataclass(frozen=True, slots=True)
class _ReplayBoundary:
    committed_samples: int
    updated_ns: int


@dataclass(frozen=True, slots=True)
class MigrationPlan:
    run_root: Path
    old_profile: Path
    new_profile: Path
    target_profile: Path
    target_profile_checksum: Path
    target_profile_bytes: bytes
    backup_directory: Path
    backup_artifacts: tuple[_BackupArtifact, ...]
    backup_evidence: Mapping[str, object]
    input_fingerprints: tuple[_InputFingerprint, ...]
    expected_absent: tuple[Path, ...]
    migration_record: Mapping[str, object]
    changes: tuple[tuple[str, object, object], ...]
    source_config_sha256: str
    target_config_sha256: str
    source_profile_sha256: str
    target_profile_sha256: str
    profile_sha256_bytes: bytes
    source_commit_bytes: bytes
    learner_step: int
    examples_consumed: int
    heartbeat_step: int
    recovery_interval_steps: int
    run_created_ns: int
    committed_replay_samples: int
    replay_updated_ns: int
    champion_model_identity: str
    coordinator_lock_status: str

    def output(self, *, mode: str, backup: Path | None = None) -> dict[str, object]:
        return {
            "status": "ok",
            "mode": mode,
            "run_id": self.migration_record["run_id"],
            "generation_family": self.migration_record["generation_family"],
            "coordinator_lock": self.coordinator_lock_status,
            "source": {
                "profile": self.old_profile.name,
                "profile_sha256": self.source_profile_sha256,
                "config_sha256": self.source_config_sha256,
                "source_commit": self.migration_record["from_source_commit"],
            },
            "target": {
                "profile": self.target_profile.name,
                "profile_sha256": self.target_profile_sha256,
                "config_sha256": self.target_config_sha256,
                "source_commit": self.migration_record["to_source_commit"],
            },
            "boundary": {
                "learner_step": self.learner_step,
                "heartbeat_step": self.heartbeat_step,
                "discarded_uncheckpointed_steps": (
                    self.heartbeat_step - self.learner_step
                ),
                "recovery_interval_steps": self.recovery_interval_steps,
                "examples_consumed": self.examples_consumed,
                "committed_replay_samples": self.committed_replay_samples,
                "champion_model_identity": self.champion_model_identity,
            },
            "changes": [
                {"path": path, "from": old, "to": new}
                for path, old, new in self.changes
            ],
            "backup_bundle": str(backup or self.backup_directory),
            "writes": [
                str(self.target_profile),
                str(self.target_profile_checksum),
                str(self.run_root / "continuous-migrations.jsonl"),
                str(self.run_root / "profile.sha256"),
                str(self.run_root / "source-commit.txt"),
            ],
        }


def _sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _sha256_file(path: Path) -> str:
    if path.is_symlink():
        raise MigrationError(f"hashed input may not be a symbolic link: {path}")
    digest = hashlib.sha256()
    try:
        with path.open("rb") as stream:
            for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                digest.update(chunk)
    except OSError as exc:
        raise MigrationError(f"cannot hash input {path}: {exc}") from exc
    if not path.is_file():
        raise MigrationError(f"hashed input is not a regular file: {path}")
    return digest.hexdigest()


def canonical_config_sha256(config: ExperimentConfig) -> str:
    """Return the stable hash of a fully materialized profile."""

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


def _resolved_input_file(path: Path, name: str) -> Path:
    expanded = path.expanduser()
    if expanded.is_symlink():
        raise MigrationError(f"{name} may not be a symbolic link: {expanded}")
    try:
        resolved = expanded.resolve(strict=True)
    except OSError as exc:
        raise MigrationError(f"cannot resolve {name} {expanded}: {exc}") from exc
    if not resolved.is_file():
        raise MigrationError(f"{name} is not a regular file: {resolved}")
    return resolved


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
        for key in sorted(set(old) | set(new), key=str):
            yield from _profile_diffs(
                old.get(key, _MISSING),
                new.get(key, _MISSING),
                (*path, str(key)),
            )
        return
    if _is_sequence(old) and _is_sequence(new):
        old_values = list(old)
        new_values = list(new)
        for index in range(max(len(old_values), len(new_values))):
            yield from _profile_diffs(
                old_values[index] if index < len(old_values) else _MISSING,
                new_values[index] if index < len(new_values) else _MISSING,
                (*path, str(index)),
            )
        return
    if old != new:
        yield path, old, new


def _json_value(value: object) -> object:
    return "<missing>" if value is _MISSING else value


def _validate_profile_pair(
    old: ExperimentConfig,
    new: ExperimentConfig,
    *,
    run_root: Path,
) -> tuple[tuple[str, object, object], ...]:
    for label, config in (("old", old), ("new", new)):
        if config.orchestration.autonomous.enabled:
            raise MigrationError(
                f"{label} profile is autonomous; autonomous profiles are not supported"
            )
        if (
            config.profile != "continuous"
            or not config.orchestration.enabled
            or not config.learner.unlimited
            or not config.learner.resume_latest
        ):
            raise MigrationError(
                f"{label} profile is not a resumable non-autonomous continuous profile"
            )

    try:
        validate_continuous_config(new)
    except ValueError as exc:
        raise MigrationError(f"new profile fails continuous validation: {exc}") from exc

    if (
        not old.orchestration.run_id
        or old.orchestration.run_id != new.orchestration.run_id
    ):
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

    differences = tuple(_profile_diffs(old.as_dict(), new.as_dict()))
    disallowed = [
        ".".join(path)
        for path, _, _ in differences
        if path not in _ALLOWED_PROFILE_PATHS
    ]
    if disallowed:
        raise MigrationError(
            "profile changes immutable or unsupported fields: " + ", ".join(disallowed)
        )
    if not differences:
        raise MigrationError("new profile has no semantic changes")
    return tuple(
        (".".join(path), _json_value(before), _json_value(after))
        for path, before, after in differences
    )


def _pid_is_live(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


def _coordinator_lock_status(run_root: Path) -> tuple[str, bytes | None, int | None]:
    path = run_root / "coordinator.lock"
    if not path.exists():
        return "absent", None, None
    payload, data = _read_json(path, "coordinator lock")
    pid = _positive_int("coordinator lock pid", payload.get("pid"))
    _positive_int("coordinator lock created_ns", payload.get("created_ns"))
    if _pid_is_live(pid):
        raise MigrationError(f"coordinator lock PID {pid} is live")
    return f"stale-dead-pid:{pid}", data, path.stat().st_mode & 0o777


def _validate_run_identity(payload: Mapping[str, object]) -> tuple[str, str, int]:
    if payload.get("schema_version") != 1:
        raise MigrationError("run.json schema is incompatible")
    run_id = _identifier("run_id", payload.get("run_id"))
    family = _identifier("generation_family", payload.get("generation_family"))
    created_ns = _positive_int("run identity created_ns", payload.get("created_ns"))
    return run_id, family, created_ns


def _parse_checksum(
    path: Path,
    *,
    profile: Path,
    actual_sha256: str,
) -> None:
    text = _read_bytes(path, "profile checksum").decode("utf-8").strip()
    parts = text.split(maxsplit=1)
    try:
        recorded = _sha256_text("recorded profile checksum", parts[0])
    except IndexError as exc:
        raise MigrationError(f"profile checksum is empty: {path}") from exc
    if recorded != actual_sha256:
        raise MigrationError(f"profile checksum does not match {profile.name}: {path}")
    if len(parts) == 2 and Path(parts[1].strip()).name != profile.name:
        raise MigrationError(f"profile checksum names another profile: {path}")


def _verified_profile_checksums(
    run_root: Path,
    profile: Path,
    *,
    actual_sha256: str,
) -> tuple[Path, ...]:
    named = profile.with_suffix(".sha256")
    active = run_root / "profile.sha256"
    paths = tuple(path for path in (named, active) if path.is_file())
    if not paths:
        raise MigrationError("source frozen profile has no checksum")
    for path in paths:
        _parse_checksum(path, profile=profile, actual_sha256=actual_sha256)
    return paths


def _validate_change_list(value: object) -> None:
    if not isinstance(value, list) or not value:
        raise MigrationError("continuous migration changes are invalid")
    seen: set[str] = set()
    allowed = {".".join(path) for path in _ALLOWED_PROFILE_PATHS}
    for change in value:
        if not isinstance(change, dict) or set(change) != {"path", "from", "to"}:
            raise MigrationError("continuous migration change entry is invalid")
        path = change.get("path")
        if not isinstance(path, str) or path not in allowed or path in seen:
            raise MigrationError("continuous migration change path is invalid")
        seen.add(path)


def _read_migration_chain(
    path: Path,
    *,
    run_id: str,
    generation_family: str,
    source_config_sha256: str,
    source_profile_name: str,
    source_profile_sha256: str,
) -> tuple[tuple[dict[str, Any], ...], bytes | None]:
    if not path.exists():
        return (), None
    data = _read_bytes(path, "continuous migration chain")
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
        "from_profile_sha256",
        "to_profile_sha256",
        "learner_step",
        "examples_consumed",
        "committed_replay_samples",
        "from_source_commit",
        "to_source_commit",
        "reason",
        "changes",
    }
    for line_number, raw_line in enumerate(data.splitlines(), start=1):
        if not raw_line.strip():
            raise MigrationError(
                f"continuous migration chain line {line_number} is empty"
            )
        try:
            record = json.loads(raw_line)
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise MigrationError(
                f"continuous migration chain line {line_number} is invalid: {exc}"
            ) from exc
        if not isinstance(record, dict) or not required <= set(record):
            raise MigrationError(
                f"continuous migration chain line {line_number} fields are invalid"
            )
        if record.get("schema_version") != MIGRATION_SCHEMA_VERSION:
            raise MigrationError("continuous migration schema version is incompatible")
        if (
            record.get("run_id") != run_id
            or record.get("generation_family") != generation_family
        ):
            raise MigrationError("continuous migration chain run identity mismatch")
        _positive_int("migration timestamp_ns", record.get("timestamp_ns"))
        _sha256_text("migration from_config_sha256", record.get("from_config_sha256"))
        _sha256_text("migration to_config_sha256", record.get("to_config_sha256"))
        _sha256_text("migration from_profile_sha256", record.get("from_profile_sha256"))
        _sha256_text("migration to_profile_sha256", record.get("to_profile_sha256"))
        _profile_name(record.get("from_profile"))
        _profile_name(record.get("to_profile"))
        _nonnegative_int("migration learner_step", record.get("learner_step"))
        _nonnegative_int("migration examples_consumed", record.get("examples_consumed"))
        _nonnegative_int(
            "migration committed_replay_samples",
            record.get("committed_replay_samples"),
        )
        _commit_text("migration from_source_commit", record.get("from_source_commit"))
        _commit_text("migration to_source_commit", record.get("to_source_commit"))
        reason = record.get("reason")
        if not isinstance(reason, str) or not reason.strip() or "\n" in reason:
            raise MigrationError("continuous migration reason is invalid")
        _validate_change_list(record.get("changes"))
        if records:
            previous = records[-1]
            if (
                record["timestamp_ns"] <= previous["timestamp_ns"]
                or record["from_config_sha256"] != previous["to_config_sha256"]
                or record["from_profile"] != previous["to_profile"]
                or record["from_profile_sha256"] != previous["to_profile_sha256"]
                or record["from_source_commit"] != previous["to_source_commit"]
                or record["learner_step"] < previous["learner_step"]
                or record["examples_consumed"] < previous["examples_consumed"]
                or record["committed_replay_samples"]
                < previous["committed_replay_samples"]
            ):
                raise MigrationError("continuous migration chain is discontinuous")
        records.append(record)
    if not records:
        raise MigrationError("continuous migration chain exists but is empty")
    head = records[-1]
    if (
        head["to_config_sha256"] != source_config_sha256
        or head["to_profile"] != source_profile_name
        or head["to_profile_sha256"] != source_profile_sha256
    ):
        raise MigrationError("source profile does not match the migration chain head")
    return tuple(records), data


def _source_commit_from_file(run_root: Path) -> tuple[str | None, bytes | None]:
    path = run_root / "source-commit.txt"
    if not path.exists():
        return None, None
    data = _read_bytes(path, "source commit")
    try:
        value = data.decode("utf-8").strip().split()[0]
    except (UnicodeDecodeError, IndexError) as exc:
        raise MigrationError(f"cannot parse source-commit.txt: {exc}") from exc
    return _commit_text("source-commit.txt", value), data


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
) -> tuple[str, str, bytes | None]:
    file_commit, file_data = _source_commit_from_file(run_root)
    chain_commit = str(chain[-1]["to_source_commit"]) if chain else None
    if (
        chain_commit is not None
        and file_commit is not None
        and file_commit != chain_commit
    ):
        raise MigrationError("source-commit.txt disagrees with migration chain head")
    authority = chain_commit or file_commit
    if request.from_source_commit is not None:
        from_commit = _commit_text("from_source_commit", request.from_source_commit)
        if authority is not None and from_commit != authority:
            raise MigrationError(
                "from_source_commit disagrees with current source authority"
            )
    else:
        from_commit = authority
    if from_commit is None:
        raise MigrationError(
            "source commit is unknown; pass --from-source-commit or provide "
            "source-commit.txt"
        )
    to_commit = (
        _commit_text("to_source_commit", request.to_source_commit)
        if request.to_source_commit is not None
        else _current_git_commit()
    )
    return from_commit, to_commit, file_data


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
) -> tuple[int, int, str, int, Path]:
    if (
        payload.get("format") != "startrain.recovery-pointer"
        or payload.get("schema_version") != 1
        or payload.get("run_id") != run_id
        or payload.get("generation_family") != generation_family
    ):
        raise MigrationError("learner recovery pointer identity is incompatible")
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
    recovery_directory = path.parent / "recovery"
    if recovery_directory.is_symlink():
        raise MigrationError("recovery checkpoint directory may not be a symbolic link")
    checkpoint = checkpoint.resolve()
    expected_directory = recovery_directory.resolve()
    if (
        checkpoint.parent != expected_directory
        or checkpoint.name != f"sha256-{checkpoint_sha256}.pt"
    ):
        raise MigrationError("recovery checkpoint escaped its content-addressed root")
    if not checkpoint.is_file() or checkpoint.stat().st_size != checkpoint_bytes:
        raise MigrationError("recovery checkpoint byte length is invalid")
    if _sha256_file(checkpoint) != checkpoint_sha256:
        raise MigrationError("recovery checkpoint SHA-256 is invalid")
    return step, examples, checkpoint_sha256, checkpoint_bytes, checkpoint


def _validate_recovery_checkpoint_payload(
    checkpoint: Path,
    *,
    config: ExperimentConfig,
    run_id: str,
    generation_family: str,
    step: int,
    examples_consumed: int,
) -> None:
    try:
        payload = torch.load(checkpoint, map_location="cpu", weights_only=True)
    except (OSError, RuntimeError, TypeError, ValueError) as exc:
        raise MigrationError(f"cannot load recovery checkpoint payload: {exc}") from exc
    if (
        not isinstance(payload, Mapping)
        or payload.get("format") != CHECKPOINT_FORMAT
        or payload.get("version") != CHECKPOINT_VERSION
        or payload.get("step") != step
    ):
        raise MigrationError("recovery checkpoint payload is incompatible")
    extra = payload.get("extra")
    if (
        not isinstance(extra, Mapping)
        or extra.get("run_id") != run_id
        or extra.get("generation_family") != generation_family
        or extra.get("examples_consumed") != examples_consumed
    ):
        raise MigrationError("recovery checkpoint payload run identity is incompatible")
    checkpoint_config = payload.get("config")
    serialized = config.as_dict()
    if not isinstance(checkpoint_config, Mapping) or any(
        checkpoint_config.get(section) != serialized[section]
        for section in ("game", "model", "loss", "optimizer")
    ):
        raise MigrationError(
            "recovery checkpoint experiment configuration is incompatible"
        )
    if any(
        payload.get(name) is None for name in ("model", "optimizer", "scheduler", "ema")
    ):
        raise MigrationError("recovery checkpoint training state is incomplete")


def _validate_champion(
    payload: Mapping[str, object],
    *,
    path: Path,
    run_id: str,
    generation_family: str,
) -> tuple[int, str, str, Path]:
    if (
        payload.get("format") != "startrain.model-pointer"
        or payload.get("schema_version") != 2
        or payload.get("role") != "champion"
        or payload.get("run_id") != run_id
        or payload.get("generation_family") != generation_family
    ):
        raise MigrationError("learner champion pointer identity is incompatible")
    step = _nonnegative_int("champion model_step", payload.get("model_step"))
    identity = _identifier("champion model_identity", payload.get("model_identity"))
    if not identity.startswith("sha256-") or _SHA256.fullmatch(identity[7:]) is None:
        raise MigrationError("champion model_identity is not content-addressed")
    manifest_sha256 = _sha256_text(
        "champion manifest_sha256", payload.get("manifest_sha256")
    )
    manifest_bytes = _positive_int(
        "champion manifest_bytes", payload.get("manifest_bytes")
    )
    _positive_int("champion updated_ns", payload.get("updated_ns"))
    manifest_value = payload.get("manifest")
    if not isinstance(manifest_value, str) or not manifest_value:
        raise MigrationError("champion manifest path is invalid")
    manifest = Path(manifest_value)
    if not manifest.is_absolute():
        manifest = path.parent / manifest
    if manifest.is_symlink():
        raise MigrationError("champion manifest may not be a symbolic link")
    manifest_directory = path.parent / "manifests"
    if manifest_directory.is_symlink():
        raise MigrationError("champion manifest directory may not be a symbolic link")
    manifest = manifest.resolve()
    expected_directory = manifest_directory.resolve()
    if (
        manifest.parent != expected_directory
        or manifest.name != f"manifest-{manifest_sha256}.json"
    ):
        raise MigrationError("champion manifest escaped its content-addressed root")
    if not manifest.is_file() or manifest.stat().st_size != manifest_bytes:
        raise MigrationError("champion manifest byte length is invalid")
    if _sha256_file(manifest) != manifest_sha256:
        raise MigrationError("champion manifest SHA-256 is invalid")
    manifest_payload, _ = _read_json(manifest, "champion manifest")
    expected = {
        "model_identity": identity,
        "model_version": identity,
        "model_step": step,
        "run_id": run_id,
        "generation_family": generation_family,
    }
    if any(manifest_payload.get(key) != value for key, value in expected.items()):
        raise MigrationError("champion pointer identity disagrees with its manifest")
    return step, identity, manifest_sha256, manifest


def _read_replay_boundary(
    run_root: Path,
    *,
    run_id: str,
    generation_family: str,
    created_ns: int,
) -> _ReplayBoundary:
    path = run_root / "replay" / "manifest.sqlite3"
    if (
        path.parent.is_symlink()
        or path.is_symlink()
        or not path.is_file()
        or path.resolve().parent != path.parent.resolve()
    ):
        raise MigrationError(f"replay manifest is missing or unsafe: {path}")
    uri = f"{path.resolve().as_uri()}?mode=ro"
    try:
        with sqlite3.connect(uri, uri=True, timeout=30.0) as connection:
            connection.execute("PRAGMA query_only = ON")
            run = connection.execute(
                """
                SELECT generation_family, created_ns
                FROM runs
                WHERE run_id = ?
                """,
                (run_id,),
            ).fetchone()
            if run is None or run[0] != generation_family or run[1] != created_ns:
                raise MigrationError("replay ledger run registration mismatch")
            row = connection.execute(
                """
                SELECT committed_samples, updated_ns, history_complete
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
    updated_ns = _positive_int("replay counter updated_ns", row[1])
    if row[2] != 1:
        raise MigrationError("committed replay history_complete is false")
    return _ReplayBoundary(committed, updated_ns)


def _artifact(path: Path, *, run_root: Path, name: str) -> _BackupArtifact:
    data = _read_bytes(path, name)
    resolved = path.resolve()
    try:
        relative = resolved.relative_to(run_root)
    except ValueError as exc:
        raise MigrationError(f"{name} is outside the run root: {resolved}") from exc
    return _BackupArtifact(
        source=resolved,
        relative_path=relative,
        data=data,
        mode=resolved.stat().st_mode & 0o777,
    )


def _fingerprint(path: Path) -> _InputFingerprint:
    if path.is_symlink() or not path.is_file():
        raise MigrationError(f"validated input is missing or unsafe: {path}")
    stat = path.stat()
    return _InputFingerprint(
        path=path,
        sha256=_sha256_file(path),
        size=stat.st_size,
        mode=stat.st_mode & 0o777,
    )


def plan_migration(request: MigrationRequest) -> MigrationPlan:
    old_profile = _resolved_input_file(request.old_profile, "old profile")
    new_profile = _resolved_input_file(request.new_profile, "new profile")
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
    target_checksum = target_profile.with_suffix(".sha256")
    if target_profile == old_profile:
        raise MigrationError("target profile name must differ from the source profile")
    if target_checksum == old_profile.with_suffix(".sha256"):
        raise MigrationError("target profile must use a distinct checksum basename")
    if target_profile.exists() or target_checksum.exists():
        raise MigrationError("target frozen profile or checksum already exists")
    if new_profile == target_profile:
        raise MigrationError(
            "new profile must be staged outside its install destination"
        )

    changes = _validate_profile_pair(old_config, new_config, run_root=run_root)
    lock_status, lock_data, _ = _coordinator_lock_status(run_root)

    old_profile_bytes = _read_bytes(old_profile, "old profile")
    new_profile_bytes = _read_bytes(new_profile, "new profile")
    source_profile_sha256 = _sha256_bytes(old_profile_bytes)
    target_profile_sha256 = _sha256_bytes(new_profile_bytes)
    source_checksums = _verified_profile_checksums(
        run_root,
        old_profile,
        actual_sha256=source_profile_sha256,
    )
    source_config_sha256 = canonical_config_sha256(old_config)
    target_config_sha256 = canonical_config_sha256(new_config)
    if source_config_sha256 == target_config_sha256:
        raise MigrationError("target canonical hash does not change the profile")

    run_path = run_root / "run.json"
    heartbeat_path = run_root / "status" / "learner.heartbeat.json"
    recovery_path = run_root / "learner" / "recovery.json"
    champion_path = run_root / "learner" / "champion.json"
    migrations_path = run_root / "continuous-migrations.jsonl"

    run_payload, _ = _read_json(run_path, "run identity")
    run_id, generation_family, created_ns = _validate_run_identity(run_payload)
    if old_config.orchestration.run_id != run_id:
        raise MigrationError("profile run_id does not match run.json")

    chain, migration_chain_data = _read_migration_chain(
        migrations_path,
        run_id=run_id,
        generation_family=generation_family,
        source_config_sha256=source_config_sha256,
        source_profile_name=old_profile.name,
        source_profile_sha256=source_profile_sha256,
    )
    from_source_commit, to_source_commit, source_commit_data = _resolve_source_commits(
        request, run_root=run_root, chain=chain
    )

    heartbeat_payload, _ = _read_json(heartbeat_path, "learner heartbeat")
    recovery_payload, recovery_data = _read_json(recovery_path, "learner recovery")
    champion_payload, champion_data = _read_json(champion_path, "learner champion")

    heartbeat_step, heartbeat_examples = _validate_heartbeat(heartbeat_payload)
    (
        learner_step,
        examples_consumed,
        recovery_checkpoint_sha256,
        recovery_checkpoint_bytes,
        recovery_checkpoint,
    ) = _validate_recovery(
        recovery_payload,
        path=recovery_path,
        run_id=run_id,
        generation_family=generation_family,
    )
    _validate_recovery_checkpoint_payload(
        recovery_checkpoint,
        config=old_config,
        run_id=run_id,
        generation_family=generation_family,
        step=learner_step,
        examples_consumed=examples_consumed,
    )
    champion_step, champion_identity, champion_manifest_sha256, champion_manifest = (
        _validate_champion(
            champion_payload,
            path=champion_path,
            run_id=run_id,
            generation_family=generation_family,
        )
    )
    recovery_interval = old_config.learner.recovery_interval_steps
    if recovery_interval is None:
        raise MigrationError("source profile has no recovery interval")
    heartbeat_lag = heartbeat_step - learner_step
    if heartbeat_lag < 0:
        raise MigrationError("learner heartbeat is behind the recovery boundary")
    if heartbeat_lag >= recovery_interval:
        raise MigrationError("learner heartbeat lag is not below the recovery interval")
    if champion_step > learner_step:
        raise MigrationError("champion step is ahead of the recovery boundary")

    replay_boundary = _read_replay_boundary(
        run_root,
        run_id=run_id,
        generation_family=generation_family,
        created_ns=created_ns,
    )
    if chain:
        head = chain[-1]
        if (
            learner_step < int(head["learner_step"])
            or examples_consumed < int(head["examples_consumed"])
            or replay_boundary.committed_samples < int(head["committed_replay_samples"])
        ):
            raise MigrationError("durable boundary moves behind the migration chain")

    timestamp_ns = time.time_ns()
    if chain and timestamp_ns <= int(chain[-1]["timestamp_ns"]):
        raise MigrationError("system clock does not advance the migration chain")
    change_records = [
        {"path": path, "from": before, "to": after} for path, before, after in changes
    ]
    migration_record: dict[str, object] = {
        "schema_version": MIGRATION_SCHEMA_VERSION,
        "timestamp_ns": timestamp_ns,
        "run_id": run_id,
        "generation_family": generation_family,
        "from_config_sha256": source_config_sha256,
        "to_config_sha256": target_config_sha256,
        "from_profile": old_profile.name,
        "to_profile": target_name,
        "from_profile_sha256": source_profile_sha256,
        "to_profile_sha256": target_profile_sha256,
        "from_source_commit": from_source_commit,
        "to_source_commit": to_source_commit,
        "reason": reason,
        "changes": change_records,
        "learner_step": learner_step,
        "heartbeat_step": heartbeat_step,
        "heartbeat_examples_consumed": heartbeat_examples,
        "discarded_uncheckpointed_steps": heartbeat_lag,
        "examples_consumed": examples_consumed,
        "committed_replay_samples": replay_boundary.committed_samples,
        "replay_counter_updated_ns": replay_boundary.updated_ns,
        "recovery_checkpoint_sha256": recovery_checkpoint_sha256,
        "recovery_checkpoint_bytes": recovery_checkpoint_bytes,
        "recovery_pointer_sha256": _sha256_bytes(recovery_data),
        "champion_model_identity": champion_identity,
        "champion_model_step": champion_step,
        "champion_manifest_sha256": champion_manifest_sha256,
        "champion_pointer_sha256": _sha256_bytes(champion_data),
    }

    required_paths = [
        (old_profile, "old profile"),
        (run_path, "run identity"),
        (heartbeat_path, "learner heartbeat"),
        (recovery_path, "learner recovery"),
        (champion_path, "learner champion"),
        (champion_manifest, "champion manifest"),
        *((path, "source profile checksum") for path in source_checksums),
    ]
    optional_paths = (
        (run_root / "source-commit.txt", "source commit"),
        (run_root / "status" / "coordinator.json", "coordinator status"),
        (run_root / "learner" / "candidate.json", "learner candidate"),
        (run_root / "arena" / "promotion-status.json", "promotion status"),
        (run_root / "replay" / "initialized.json", "replay initialization"),
    )
    artifacts_by_source: dict[Path, _BackupArtifact] = {}
    for path, name in required_paths:
        artifact = _artifact(path, run_root=run_root, name=name)
        artifacts_by_source[artifact.source] = artifact
    for path, name in optional_paths:
        if path.exists():
            artifact = _artifact(path, run_root=run_root, name=name)
            artifacts_by_source[artifact.source] = artifact
    if migration_chain_data is not None:
        artifact = _artifact(
            migrations_path,
            run_root=run_root,
            name="continuous migration chain",
        )
        artifacts_by_source[artifact.source] = artifact
    if lock_data is not None:
        artifact = _artifact(
            run_root / "coordinator.lock",
            run_root=run_root,
            name="stale coordinator lock",
        )
        artifacts_by_source[artifact.source] = artifact
    artifacts = tuple(
        sorted(artifacts_by_source.values(), key=lambda item: str(item.relative_path))
    )

    input_paths = {artifact.source for artifact in artifacts}
    input_paths.update({new_profile, recovery_checkpoint})
    # The stale lock is deliberately replaced by the migration guard.
    input_paths.discard((run_root / "coordinator.lock").resolve())
    fingerprints = tuple(
        sorted(
            (_fingerprint(path) for path in input_paths),
            key=lambda item: str(item.path),
        )
    )

    expected_absent = [target_profile, target_checksum]
    backup_directory = (
        run_root / "migration-backups" / f"{timestamp_ns}-{target_profile.stem}"
    )
    expected_absent.append(backup_directory)
    for path in (
        migrations_path,
        run_root / "profile.sha256",
        run_root / "source-commit.txt",
    ):
        if not path.exists():
            expected_absent.append(path)

    profile_sha256_bytes = f"{target_profile_sha256}  {target_profile}\n".encode(
        "utf-8"
    )
    source_commit_bytes = f"{to_source_commit}\n".encode("utf-8")
    backup_evidence: dict[str, object] = {
        "recovery": {
            "pointer": str(recovery_path.relative_to(run_root)),
            "checkpoint": str(recovery_checkpoint.relative_to(run_root)),
            "checkpoint_sha256": recovery_checkpoint_sha256,
            "checkpoint_bytes": recovery_checkpoint_bytes,
            "step": learner_step,
            "examples_consumed": examples_consumed,
            "run_id": run_id,
            "generation_family": generation_family,
        },
        "champion": {
            "pointer": str(champion_path.relative_to(run_root)),
            "model_identity": champion_identity,
            "model_step": champion_step,
            "manifest_sha256": champion_manifest_sha256,
        },
        "replay": {
            "manifest": "replay/manifest.sqlite3",
            "run_id": run_id,
            "generation_family": generation_family,
            "committed_samples": replay_boundary.committed_samples,
            "counter_updated_ns": replay_boundary.updated_ns,
            "history_complete": True,
        },
    }
    if source_commit_data is not None:
        backup_evidence["source_commit_sha256"] = _sha256_bytes(source_commit_data)

    return MigrationPlan(
        run_root=run_root,
        old_profile=old_profile,
        new_profile=new_profile,
        target_profile=target_profile,
        target_profile_checksum=target_checksum,
        target_profile_bytes=new_profile_bytes,
        backup_directory=backup_directory,
        backup_artifacts=artifacts,
        backup_evidence=backup_evidence,
        input_fingerprints=fingerprints,
        expected_absent=tuple(expected_absent),
        migration_record=migration_record,
        changes=changes,
        source_config_sha256=source_config_sha256,
        target_config_sha256=target_config_sha256,
        source_profile_sha256=source_profile_sha256,
        target_profile_sha256=target_profile_sha256,
        profile_sha256_bytes=profile_sha256_bytes,
        source_commit_bytes=source_commit_bytes,
        learner_step=learner_step,
        examples_consumed=examples_consumed,
        heartbeat_step=heartbeat_step,
        recovery_interval_steps=recovery_interval,
        run_created_ns=created_ns,
        committed_replay_samples=replay_boundary.committed_samples,
        replay_updated_ns=replay_boundary.updated_ns,
        champion_model_identity=champion_identity,
        coordinator_lock_status=lock_status,
    )


def _fsync_directory(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _atomic_replace_bytes(
    path: Path,
    data: bytes,
    *,
    mode: int,
    overwrite: bool,
) -> None:
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
        if overwrite:
            os.replace(temporary_name, path)
            temporary_name = None
        else:
            try:
                os.link(temporary_name, path)
            except FileExistsError as exc:
                raise MigrationError(
                    f"refusing to overwrite immutable output: {path}"
                ) from exc
            os.unlink(temporary_name)
            temporary_name = None
        _fsync_directory(path.parent)
    finally:
        if temporary_name is not None:
            Path(temporary_name).unlink(missing_ok=True)


def _atomic_write_bytes(
    path: Path,
    data: bytes,
    *,
    mode: int,
    overwrite: bool,
) -> None:
    """Write one migration output atomically; kept separate for fault injection."""

    _atomic_replace_bytes(path, data, mode=mode, overwrite=overwrite)


def _append_jsonl(path: Path, payload: Mapping[str, object]) -> None:
    if path.is_symlink():
        raise MigrationError("continuous migration chain may not be a symbolic link")
    existed = path.exists()
    if existed and not path.is_file():
        raise MigrationError("continuous migration chain is not a regular file")
    prefix = b""
    if existed and path.stat().st_size:
        with path.open("rb") as stream:
            stream.seek(-1, os.SEEK_END)
            if stream.read(1) != b"\n":
                prefix = b"\n"
    data = prefix + _json_bytes(payload)
    flags = os.O_APPEND | os.O_CREAT | os.O_WRONLY | getattr(os, "O_NOFOLLOW", 0)
    descriptor = os.open(path, flags, 0o644)
    try:
        written = 0
        while written < len(data):
            count = os.write(descriptor, data[written:])
            if count <= 0:
                raise OSError("short continuous migration record write")
            written += count
        os.fsync(descriptor)
    finally:
        os.close(descriptor)
    if not existed:
        _fsync_directory(path.parent)


def _write_new_backup_file(path: Path, data: bytes, mode: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor = os.open(
        path,
        os.O_CREAT | os.O_EXCL | os.O_WRONLY | getattr(os, "O_NOFOLLOW", 0),
        mode,
    )
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
                    "mode": f"{artifact.mode:04o}",
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
            "validated_state": plan.backup_evidence,
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
    for expected in plan.input_fingerprints:
        path = expected.path
        if path.is_symlink() or not path.is_file():
            raise MigrationError(f"validated input changed before apply: {path}")
        stat = path.stat()
        if (
            stat.st_size != expected.size
            or stat.st_mode & 0o777 != expected.mode
            or _sha256_file(path) != expected.sha256
        ):
            raise MigrationError(f"validated input changed before apply: {path}")
    for path in plan.expected_absent:
        if path.exists():
            raise MigrationError(f"validated output appeared before apply: {path}")
    replay = _read_replay_boundary(
        plan.run_root,
        run_id=str(plan.migration_record["run_id"]),
        generation_family=str(plan.migration_record["generation_family"]),
        created_ns=plan.run_created_ns,
    )
    if (
        replay.committed_samples != plan.committed_replay_samples
        or replay.updated_ns != plan.replay_updated_ns
    ):
        raise MigrationError("replay ledger boundary changed before apply")


@contextmanager
def _coordinator_write_guard(run_root: Path) -> Iterator[None]:
    path = run_root / "coordinator.lock"
    status, stale_data, stale_mode = _coordinator_lock_status(run_root)
    if status.startswith("stale-dead-pid:"):
        path.unlink()
        _fsync_directory(run_root)
    token = uuid.uuid4().hex
    payload = _json_bytes(
        {
            "pid": os.getpid(),
            "created_ns": time.time_ns(),
            "owner": "continuous-profile-migration",
            "token": token,
        }
    )
    try:
        descriptor = os.open(
            path,
            os.O_CREAT | os.O_EXCL | os.O_WRONLY | getattr(os, "O_NOFOLLOW", 0),
            0o644,
        )
    except Exception:
        if stale_data is not None and not path.exists():
            _atomic_replace_bytes(
                path,
                stale_data,
                mode=stale_mode or 0o644,
                overwrite=False,
            )
        raise
    try:
        os.write(descriptor, payload)
        os.fsync(descriptor)
    finally:
        os.close(descriptor)
    _fsync_directory(run_root)
    succeeded = False
    try:
        yield
        succeeded = True
    finally:
        try:
            current, _ = _read_json(path, "migration coordinator lock")
        except MigrationError:
            current = {}
        owns_lock = current.get("token") == token and current.get("pid") == os.getpid()
        if owns_lock:
            if succeeded or stale_data is None:
                path.unlink(missing_ok=True)
                _fsync_directory(run_root)
            else:
                _atomic_replace_bytes(
                    path,
                    stale_data,
                    mode=stale_mode or 0o644,
                    overwrite=True,
                )


def _capture_file_state(path: Path) -> _FileState:
    if not path.exists():
        return _FileState(path, None, None)
    data = _read_bytes(path, "transaction output")
    return _FileState(path, data, path.stat().st_mode & 0o777)


def _restore_file_state(state: _FileState) -> None:
    if state.data is None:
        if state.path.exists():
            if state.path.is_dir() and not state.path.is_symlink():
                raise MigrationError(
                    f"rollback output became a directory: {state.path}"
                )
            state.path.unlink()
            _fsync_directory(state.path.parent)
        return
    _atomic_replace_bytes(
        state.path,
        state.data,
        mode=state.mode or 0o644,
        overwrite=True,
    )


def _rollback_outputs(
    plan: MigrationPlan,
    states: Sequence[_FileState],
    *,
    backup_parent_existed: bool,
) -> tuple[str, ...]:
    failures: list[str] = []
    for state in reversed(states):
        try:
            _restore_file_state(state)
        except Exception as exc:
            failures.append(f"{state.path}: {exc}")
    try:
        if plan.backup_directory.exists():
            if plan.backup_directory.is_symlink():
                plan.backup_directory.unlink()
            else:
                shutil.rmtree(plan.backup_directory)
            _fsync_directory(plan.backup_directory.parent)
        backup_parent = plan.backup_directory.parent
        if (
            not backup_parent_existed
            and backup_parent.is_dir()
            and not any(backup_parent.iterdir())
        ):
            backup_parent.rmdir()
            _fsync_directory(backup_parent.parent)
    except Exception as exc:
        failures.append(f"{plan.backup_directory}: {exc}")
    return tuple(failures)


def apply_migration(plan: MigrationPlan) -> dict[str, object]:
    """Apply a validated migration atomically under coordinator exclusion."""

    _assert_inputs_unchanged(plan, check_lock=True)
    with _coordinator_write_guard(plan.run_root):
        _assert_inputs_unchanged(plan, check_lock=False)
        mutable_paths = (
            plan.target_profile,
            plan.target_profile_checksum,
            plan.run_root / "continuous-migrations.jsonl",
            plan.run_root / "profile.sha256",
            plan.run_root / "source-commit.txt",
        )
        states = tuple(_capture_file_state(path) for path in mutable_paths)
        backup_parent_existed = plan.backup_directory.parent.exists()
        try:
            backup = _create_backup(plan)
            _atomic_write_bytes(
                plan.target_profile,
                plan.target_profile_bytes,
                mode=0o444,
                overwrite=False,
            )
            _atomic_write_bytes(
                plan.target_profile_checksum,
                plan.profile_sha256_bytes,
                mode=0o444,
                overwrite=False,
            )
            _append_jsonl(
                plan.run_root / "continuous-migrations.jsonl",
                plan.migration_record,
            )
            _atomic_write_bytes(
                plan.run_root / "profile.sha256",
                plan.profile_sha256_bytes,
                mode=0o644,
                overwrite=True,
            )
            _atomic_write_bytes(
                plan.run_root / "source-commit.txt",
                plan.source_commit_bytes,
                mode=0o644,
                overwrite=True,
            )
        except Exception as exc:
            rollback_failures = _rollback_outputs(
                plan,
                states,
                backup_parent_existed=backup_parent_existed,
            )
            if rollback_failures:
                raise MigrationError(
                    "migration apply failed and rollback was incomplete: "
                    + "; ".join(rollback_failures)
                ) from exc
            raise MigrationError(
                f"migration apply failed and was rolled back: {exc}"
            ) from exc
    return plan.output(mode="apply", backup=backup)


def migrate_continuous_profile(
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
        result = migrate_continuous_profile(request, apply=arguments.apply)
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
