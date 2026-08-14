#!/usr/bin/env python3
"""Run, resume, verify, and finalize a durable Elo-ablation queue."""

from __future__ import annotations

import argparse
import fcntl
import hashlib
import json
import math
import os
import re
import shlex
import sqlite3
import subprocess
import time
from collections.abc import Callable, Iterator, Mapping
from contextlib import contextmanager, nullcontext
from pathlib import Path
from typing import Any, Protocol

from startrain.config import load_config
from startrain.runtime import atomic_json, load_run_identity

if __package__:
    from .compare_elo_ablation import (
        DEFAULT_GUARD_FLOOR_ELO,
        DEFAULT_GUARD_RINGS,
        DEFAULT_PROVISIONED_GPUS,
        build_elo_ablation_comparison,
    )
    from .run_elo_ablation import (
        BUDGET_COMPLETION,
        FATAL_ORCHESTRATOR_EXIT,
        RUNNER_ERROR,
        TRANSIENT_CRASH,
        run_elo_ablation,
    )
    from .replay_manifest_backup import create_backup_with_evidence
else:
    from compare_elo_ablation import (
        DEFAULT_GUARD_FLOOR_ELO,
        DEFAULT_GUARD_RINGS,
        DEFAULT_PROVISIONED_GPUS,
        build_elo_ablation_comparison,
    )
    from run_elo_ablation import (
        BUDGET_COMPLETION,
        FATAL_ORCHESTRATOR_EXIT,
        RUNNER_ERROR,
        TRANSIENT_CRASH,
        run_elo_ablation,
    )
    from replay_manifest_backup import create_backup_with_evidence

SCHEMA_VERSION = 1
DEPLOYMENT_REPORT = "startrain-elo-ablation-deployment"
QUEUE_REPORT = "startrain-elo-ablation-queue"
CONTINUITY_HANDOFF_REPORT = "startrain-continuity-handoff-request"
_SEMANTIC_JSON_PIN = "json-fields-v1"
_ARM_STATUSES = frozenset({"pending", "running", "completed", "failed", "quarantined"})
_ISOLATED_FAILURE_DOMAINS = frozenset({"arm", "run", "workload"})
_VALID_INTEGRITY_STATUSES = frozenset(
    {"ok", "pass", "passed", "valid", "verified", "healthy"}
)
_CLEAN_TEARDOWN_STATUSES = frozenset(
    {"not_required", "clean", "complete", "completed", "released"}
)
_COMMIT_PATTERN = re.compile(r"(?:[0-9a-f]{40}|[0-9a-f]{64})")
_SHA256_PATTERN = re.compile(r"[0-9a-f]{64}")
_EXECUTION_LOCK_LEASE_TOKEN = object()
_BACKUP_SCRIPT_NAME = "replay_manifest_backup.py"
_BASE_SCRIPT_NAMES = (
    "check_training_ipc.py",
    "run_elo_ablation_queue.py",
    "run_staged_elo_pipeline.py",
    "run_elo_ablation.py",
    "compare_elo_ablation.py",
    "preflight_run_state.py",
    _BACKUP_SCRIPT_NAME,
)


class AblationQueueError(RuntimeError):
    """Base error for queue, manifest, and finalization failures."""


class QueueBusyError(AblationQueueError):
    """Raised when another queue process owns the exclusive state lock."""


class ExecutionLockLease:
    """Opaque proof that this process currently owns one host execution lock."""

    __slots__ = ("_active", "_descriptor", "_token", "path")

    def __init__(
        self,
        path: Path,
        descriptor: int,
        *,
        token: object,
    ) -> None:
        if token is not _EXECUTION_LOCK_LEASE_TOKEN:
            raise TypeError("execution lock leases can only be acquired")
        self.path = path
        self._descriptor = descriptor
        self._token = token
        self._active = True

    def _invalidate(self) -> None:
        self._active = False

    def verify(self, expected_path: Path) -> None:
        if (
            self._token is not _EXECUTION_LOCK_LEASE_TOKEN
            or not self._active
            or self.path != expected_path.expanduser().resolve()
        ):
            raise AblationQueueError(
                "active execution lock lease differs from the deployment manifest"
            )
        try:
            os.fstat(self._descriptor)
        except OSError as error:
            raise AblationQueueError(
                "execution lock lease is no longer active"
            ) from error


class ArmRunner(Protocol):
    def __call__(
        self,
        *,
        config_path: Path,
        orchestrator: str,
        poll_seconds: float,
    ) -> dict[str, object]: ...


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    manifest = subparsers.add_parser(
        "manifest",
        help="generate a revision-locked deployment manifest",
    )
    manifest.add_argument("--plan", type=Path, required=True)
    manifest.add_argument("--output", type=Path, required=True)
    manifest.add_argument(
        "--training-dir",
        type=Path,
        default=Path(__file__).resolve().parents[1],
    )
    manifest.add_argument("--queue-unit", type=Path, required=True)
    manifest.add_argument("--finalize-unit", type=Path, required=True)
    manifest.add_argument("--environment-file", type=Path, required=True)
    manifest.add_argument("--state", type=Path, required=True)
    manifest.add_argument("--comparison-output", type=Path, required=True)
    manifest.add_argument("--continuity-handoff-output", type=Path)
    manifest.add_argument("--execution-lock", type=Path)
    manifest.add_argument("--source-commit")
    manifest.add_argument("--orchestrator", default="startrain-orchestrate")
    manifest.add_argument("--poll-seconds", type=float, default=5.0)
    manifest.add_argument("--max-transient-retries", type=int, default=2)
    manifest.add_argument("--retry-delay-seconds", type=float, default=30.0)
    manifest.add_argument("--continue-after-fatal", action="store_true")
    manifest.add_argument("--replay-backup-service-unit", type=Path)
    manifest.add_argument("--replay-backup-timer-unit", type=Path)
    manifest.add_argument("--replay-backup-interval-seconds", type=float)
    manifest.add_argument("--replay-backup-retain", type=int, default=3)
    manifest.add_argument(
        "--replay-backup-max-total-bytes",
        type=int,
        default=20 * 1024 * 1024 * 1024,
    )
    manifest.add_argument(
        "--provisioned-gpus",
        type=int,
        default=DEFAULT_PROVISIONED_GPUS,
    )

    for name in ("verify", "run", "finalize"):
        command = subparsers.add_parser(name)
        command.add_argument("--manifest", type=Path, required=True)
    return parser


def _read_json(path: Path) -> dict[str, Any]:
    try:
        loaded = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise AblationQueueError(
            f"cannot read JSON object {path}: {type(error).__name__}: {error}"
        ) from error
    if not isinstance(loaded, dict):
        raise AblationQueueError(f"{path} must contain a JSON object")
    return loaded


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    try:
        with path.open("rb") as stream:
            for block in iter(lambda: stream.read(1024 * 1024), b""):
                digest.update(block)
    except OSError as error:
        raise AblationQueueError(f"cannot hash {path}: {error}") from error
    return digest.hexdigest()


def _mapping(value: object, name: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise AblationQueueError(f"{name} must be an object")
    return value


def _list(value: object, name: str) -> list[Any]:
    if not isinstance(value, list):
        raise AblationQueueError(f"{name} must be a list")
    return value


def _string(value: object, name: str) -> str:
    if not isinstance(value, str) or not value:
        raise AblationQueueError(f"{name} must be a non-empty string")
    return value


def _commit(value: object, name: str) -> str:
    commit = _string(value, name).lower()
    if not _COMMIT_PATTERN.fullmatch(commit):
        raise AblationQueueError(f"{name} must be a full hexadecimal Git commit")
    return commit


def _positive_float(value: object, name: str, *, allow_zero: bool = False) -> float:
    if (
        isinstance(value, bool)
        or not isinstance(value, int | float)
        or not math.isfinite(float(value))
        or (float(value) < 0 if allow_zero else float(value) <= 0)
    ):
        qualifier = "non-negative" if allow_zero else "positive"
        raise AblationQueueError(f"{name} must be finite and {qualifier}")
    return float(value)


def _nonnegative_integer(value: object, name: str) -> int:
    if type(value) is not int or value < 0:
        raise AblationQueueError(f"{name} must be a non-negative integer")
    return value


def _positive_integer(value: object, name: str) -> int:
    parsed = _nonnegative_integer(value, name)
    if parsed == 0:
        raise AblationQueueError(f"{name} must be positive")
    return parsed


def _replay_backup_policy(
    queue: Mapping[str, object],
) -> dict[str, object] | None:
    raw = queue.get("replay_backup")
    if raw is None:
        return None
    policy = _mapping(raw, "replay backup policy")
    if policy.get("enabled") is not True:
        raise AblationQueueError("replay backup policy must be enabled or omitted")
    return {
        "enabled": True,
        "interval_seconds": _positive_float(
            policy.get("interval_seconds"),
            "replay backup interval seconds",
        ),
        "retain": _positive_integer(
            policy.get("retain"),
            "replay backup retention count",
        ),
        "max_total_bytes": _positive_integer(
            policy.get("max_total_bytes"),
            "replay backup maximum bytes",
        ),
    }


def _required_script_names(manifest: Mapping[str, object]) -> tuple[str, ...]:
    _mapping(manifest.get("queue"), "deployment queue")
    return _BASE_SCRIPT_NAMES


def _required_unit_names(manifest: Mapping[str, object]) -> set[str]:
    queue = _mapping(manifest.get("queue"), "deployment queue")
    names = {"queue", "finalize"}
    if _replay_backup_policy(queue) is not None:
        names.update({"replay_backup_service", "replay_backup_timer"})
    return names


def _artifact(name: str, path: Path) -> dict[str, object]:
    absolute = Path(os.path.abspath(path.expanduser()))
    if not absolute.is_file():
        raise AblationQueueError(f"{name} does not exist: {absolute}")
    return {
        "name": name,
        "path": str(absolute),
        "sha256": _sha256(absolute),
    }


def _replay_initialization_fields(path: Path) -> dict[str, object]:
    payload = _read_json(path)
    schema_version = payload.get("schema_version")
    if type(schema_version) is not int or schema_version != 1:
        raise AblationQueueError("replay initialization schema is unsupported")
    return {
        "schema_version": schema_version,
        "run_id": _string(
            payload.get("run_id"),
            "replay initialization run id",
        ),
        "generation_family": _string(
            payload.get("generation_family"),
            "replay initialization generation family",
        ),
    }


def _replay_initialization_artifact(path: Path) -> dict[str, object]:
    absolute = Path(os.path.abspath(path.expanduser()))
    if not absolute.is_file():
        raise AblationQueueError(f"replay_initialization does not exist: {absolute}")
    return {
        "name": "replay_initialization",
        "path": str(absolute),
        "pin": _SEMANTIC_JSON_PIN,
        "fields": _replay_initialization_fields(absolute),
    }


def _unit_artifact(name: str, path: Path) -> dict[str, object]:
    absolute = Path(os.path.abspath(path.expanduser()))
    try:
        contents = absolute.read_text(encoding="utf-8")
    except OSError as error:
        raise AblationQueueError(
            f"cannot read {name} unit {absolute}: {error}"
        ) from error
    if re.search(r"@[A-Z][A-Z0-9_]*@", contents):
        raise AblationQueueError(f"{name} unit still contains template placeholders")
    return _artifact(name, absolute)


def _unit_value(path: Path, name: str) -> str:
    try:
        lines = path.expanduser().resolve().read_text(encoding="utf-8").splitlines()
    except OSError as error:
        raise AblationQueueError(f"cannot read unit {path}: {error}") from error
    values = [
        line.partition("=")[2].strip()
        for line in lines
        if line.partition("=")[0].strip() == name
    ]
    if len(values) != 1 or not values[0]:
        raise AblationQueueError(f"unit {path} must define exactly one {name}")
    return values[0]


def _validate_replay_backup_units(
    *,
    service_unit: Path,
    timer_unit: Path,
    backup_script: Path,
    state_path: Path,
    policy: Mapping[str, object],
) -> None:
    try:
        command = shlex.split(_unit_value(service_unit, "ExecStart"))
    except ValueError as error:
        raise AblationQueueError(
            f"replay backup ExecStart is invalid: {error}"
        ) from error
    if "backup-active-arm" not in command:
        raise AblationQueueError("replay backup service must run backup-active-arm")
    script_tokens = [
        token for token in command if Path(token).name == "replay_manifest_backup.py"
    ]
    if len(script_tokens) != 1 or Path(script_tokens[0]).expanduser().resolve() != (
        backup_script.expanduser().resolve()
    ):
        raise AblationQueueError(
            "replay backup service script differs from the deployed revision"
        )

    def option(name: str) -> str:
        try:
            index = command.index(name)
            return command[index + 1]
        except (ValueError, IndexError) as error:
            raise AblationQueueError(
                f"replay backup service is missing {name}"
            ) from error

    configured_state = Path(option("--queue-state")).expanduser().resolve()
    if configured_state != state_path:
        raise AblationQueueError("replay backup service queue state path differs")
    interval = _positive_float(
        policy.get("interval_seconds"),
        "replay backup interval seconds",
    )
    retain = _positive_integer(
        policy.get("retain"),
        "replay backup retention count",
    )
    max_total_bytes = _positive_integer(
        policy.get("max_total_bytes"),
        "replay backup maximum bytes",
    )
    if float(option("--interval-seconds")) != interval:
        raise AblationQueueError("replay backup service interval differs")
    if int(option("--retain")) != retain:
        raise AblationQueueError("replay backup service retention differs")
    if int(option("--max-total-bytes")) != max_total_bytes:
        raise AblationQueueError("replay backup service byte cap differs")
    timer_interval = _unit_value(timer_unit, "OnUnitActiveSec")
    expected_interval = f"{interval:g}s"
    if timer_interval != expected_interval:
        raise AblationQueueError(
            "replay backup timer interval differs from the manifest policy"
        )
    if _unit_value(timer_unit, "Unit") != service_unit.expanduser().resolve().name:
        raise AblationQueueError("replay backup timer targets a different service")


def _git_revision(training_dir: Path) -> tuple[str, bool]:
    try:
        commit = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=training_dir,
            check=True,
            capture_output=True,
            text=True,
            timeout=10,
        ).stdout.strip()
        status = subprocess.run(
            ["git", "status", "--porcelain", "--untracked-files=all"],
            cwd=training_dir,
            check=True,
            capture_output=True,
            text=True,
            timeout=30,
        ).stdout
    except (OSError, subprocess.SubprocessError) as error:
        raise AblationQueueError(
            f"cannot determine deployed source revision: {error}"
        ) from error
    return _commit(commit, "current source commit"), not bool(status.strip())


def _plan_treatments(plan: Mapping[str, object]) -> list[dict[str, str]]:
    raw_treatments = _list(plan.get("treatments"), "plan treatments")
    if len(raw_treatments) < 2:
        raise AblationQueueError("an ablation queue requires at least two treatments")
    treatments: list[dict[str, str]] = []
    labels: set[str] = set()
    roots: set[Path] = set()
    for index, raw_treatment in enumerate(raw_treatments):
        treatment = _mapping(raw_treatment, f"plan treatment {index}")
        label = _string(treatment.get("treatment"), "plan treatment label")
        profile = _string(treatment.get("profile"), f"{label} plan profile")
        profile_sha256 = _string(
            treatment.get("profile_sha256"),
            f"{label} plan profile digest",
        )
        root = (
            Path(_string(treatment.get("run_root"), f"{label} run root"))
            .expanduser()
            .resolve()
        )
        if label in labels:
            raise AblationQueueError(f"duplicate treatment label: {label}")
        if root in roots:
            raise AblationQueueError(f"duplicate treatment run root: {root}")
        if not _SHA256_PATTERN.fullmatch(profile_sha256):
            raise AblationQueueError(f"{label} plan profile digest is invalid")
        labels.add(label)
        roots.add(root)
        treatments.append(
            {
                "treatment": label,
                "profile": str(Path(profile).expanduser().resolve()),
                "profile_sha256": profile_sha256,
                "run_root": str(root),
            }
        )
    return treatments


def _seed_snapshot(source_root: Path) -> dict[str, object]:
    if not source_root.is_dir():
        raise AblationQueueError(f"seed snapshot does not exist: {source_root}")
    if (source_root / "coordinator.lock").exists():
        raise AblationQueueError("seed snapshot still has a coordinator lock")
    identity = load_run_identity(source_root / "run.json")
    required = (
        ("run_identity", source_root / "run.json"),
        ("champion_pointer", source_root / "learner" / "champion.json"),
    )
    optional = (
        ("candidate_pointer", source_root / "learner" / "candidate.json"),
        ("recovery_pointer", source_root / "learner" / "recovery.json"),
        ("replay_manifest", source_root / "replay" / "manifest.sqlite3"),
        ("replay_initialization", source_root / "replay" / "initialized.json"),
        ("source_commit", source_root / "source-commit.txt"),
    )
    artifacts = [_artifact(name, path) for name, path in required]
    for name, path in optional:
        if not path.is_file():
            continue
        artifact = (
            _replay_initialization_artifact(path)
            if name == "replay_initialization"
            else _artifact(name, path)
        )
        artifacts.append(artifact)
    champion = _read_json(source_root / "learner" / "champion.json")
    initialization = next(
        (
            _mapping(artifact.get("fields"), "replay initialization fields")
            for artifact in artifacts
            if artifact.get("name") == "replay_initialization"
        ),
        None,
    )
    if initialization is not None and (
        initialization.get("run_id") != identity.run_id
        or initialization.get("generation_family") != identity.generation_family
    ):
        raise AblationQueueError(
            "replay initialization identity disagrees with the seed run"
        )
    return {
        "root": str(source_root),
        "coverage": "identity, model pointers, recovery pointer, and replay ledger",
        "run_identity": {
            "run_id": identity.run_id,
            "generation_family": identity.generation_family,
            "created_ns": identity.created_ns,
        },
        "champion": {
            "model_identity": champion.get("model_identity"),
            "model_step": champion.get("model_step"),
        },
        "artifacts": artifacts,
    }


def _profile_manifest_entry(
    treatment: Mapping[str, str],
    *,
    seed: Mapping[str, object],
    expected_seed: int,
    source_winner_snapshot: object,
    futility_policy: object,
) -> dict[str, object]:
    label = treatment["treatment"]
    plan_profile = Path(treatment["profile"])
    if (
        not plan_profile.is_file()
        or _sha256(plan_profile) != treatment["profile_sha256"]
    ):
        raise AblationQueueError(
            f"{label} plan profile is missing or changed: {plan_profile}"
        )
    run_root = Path(treatment["run_root"])
    profile = run_root / "profile-elo-ablation.yaml"
    metadata_path = run_root / "ablation.json"
    if not run_root.is_dir():
        raise AblationQueueError(f"{label} run root does not exist: {run_root}")
    metadata = _read_json(metadata_path)
    if metadata.get("report") != "startrain-elo-ablation-branch":
        raise AblationQueueError(f"{label} ablation metadata report is invalid")
    if metadata.get("treatment") != label:
        raise AblationQueueError(f"{label} ablation metadata treatment disagrees")
    seed_identity = _mapping(seed.get("run_identity"), "seed run identity")
    for metadata_name, seed_name in (
        ("source_run_id", "run_id"),
        ("source_generation_family", "generation_family"),
        ("source_created_ns", "created_ns"),
    ):
        if metadata.get(metadata_name) != seed_identity.get(seed_name):
            raise AblationQueueError(
                f"{label} metadata does not match seed snapshot {seed_name}"
            )
    seed_root = Path(_string(seed.get("root"), "seed root")).expanduser().resolve()
    metadata_source = metadata.get("source_run_root")
    if (
        not isinstance(metadata_source, str)
        or Path(metadata_source).expanduser().resolve() != seed_root
    ):
        raise AblationQueueError(f"{label} metadata seed root disagrees")
    anchor = _mapping(metadata.get("anchor"), f"{label} ablation anchor")
    seed_champion = _mapping(seed.get("champion"), "seed champion")
    state_artifacts = []
    selection = metadata.get("source_manifest_selection")
    expected_anchor: Mapping[str, object] = seed_champion
    if selection is not None:
        selected = _mapping(selection, f"{label} manifest selection")
        selected_manifest = _mapping(
            selected.get("selected_manifest"),
            f"{label} selected manifest",
        )
        if selected.get("status") != "verified":
            raise AblationQueueError(f"{label} manifest selection is not verified")
        expected_anchor = selected_manifest
        installed_snapshot = Path(
            _string(
                selected.get("installed_snapshot"),
                f"{label} installed selection snapshot",
            )
        ).resolve()
        if installed_snapshot != run_root / "selection-snapshot.json":
            raise AblationQueueError(
                f"{label} installed selection snapshot path is invalid"
            )
        snapshot_artifact = _artifact("manifest_selection", installed_snapshot)
        if snapshot_artifact["sha256"] != selected.get(
            "source_snapshot_sha256"
        ) or snapshot_artifact["bytes"] != selected.get("source_snapshot_bytes"):
            raise AblationQueueError(
                f"{label} installed selection snapshot differs from its source"
            )
        cutover_path = run_root / "learner" / "selection-cutover.json"
        cutover = _read_json(cutover_path)
        if (
            cutover.get("format") != "startrain.selection-cutover"
            or cutover.get("schema_version") != 1
            or cutover.get("status") != "active"
            or cutover.get("selected_model_identity")
            != selected_manifest.get("model_identity")
            or cutover.get("selected_model_step") != selected_manifest.get("model_step")
        ):
            raise AblationQueueError(
                f"{label} selected manifest lacks an active warm-start cutover"
            )
        state_artifacts.extend(
            (
                snapshot_artifact,
                _artifact("selection_cutover", cutover_path),
            )
        )
    if any(
        anchor.get(field) != expected_anchor.get(field)
        for field in ("model_identity", "model_step")
    ):
        raise AblationQueueError(f"{label} anchor differs from its verified source")
    if metadata.get("source_winner_snapshot") != source_winner_snapshot:
        raise AblationQueueError(f"{label} staged winner snapshot disagrees with plan")
    if metadata.get("futility_policy") != futility_policy:
        raise AblationQueueError(f"{label} futility policy disagrees with plan")
    installed = _artifact("installed_profile", profile)
    if installed["sha256"] != treatment["profile_sha256"]:
        raise AblationQueueError(f"{label} installed profile differs from the plan")
    configured_profile = metadata.get("profile")
    if (
        not isinstance(configured_profile, str)
        or Path(configured_profile).expanduser().resolve() != profile.resolve()
        or metadata.get("profile_sha256") != installed["sha256"]
    ):
        raise AblationQueueError(f"{label} metadata profile authority disagrees")
    experiment = load_config(profile)
    if (
        Path(experiment.orchestration.directories.root).expanduser().resolve()
        != run_root
    ):
        raise AblationQueueError(f"{label} profile run root disagrees")
    if (
        experiment.train.seed != expected_seed
        or experiment.selfplay.seed != expected_seed
        or experiment.arena.seed != expected_seed
    ):
        raise AblationQueueError(
            f"{label} profile seed contract disagrees with the ablation plan"
        )
    warm_start_path = run_root / "learner" / "champion-warm-start.json"
    if warm_start_path.is_file():
        warm_start = _read_json(warm_start_path)
        if (
            warm_start.get("format") != "startrain.champion-warm-start"
            or warm_start.get("schema_version") != 1
            or warm_start.get("status") != "active"
            or warm_start.get("source_model_identity")
            != expected_anchor.get("model_identity")
            or warm_start.get("absolute_model_step")
            != expected_anchor.get("model_step")
        ):
            raise AblationQueueError(f"{label} champion warm start is incompatible")
        checkpoint_value = _string(
            warm_start.get("checkpoint"),
            f"{label} warm-start checkpoint",
        )
        checkpoint = (run_root / "learner" / checkpoint_value).resolve()
        recovery_root = (run_root / "learner" / "recovery").resolve()
        if checkpoint.parent != recovery_root:
            raise AblationQueueError(
                f"{label} warm-start checkpoint escaped the recovery directory"
            )
        state_artifacts.extend(
            (
                _artifact("champion_warm_start", warm_start_path),
                _artifact("champion_warm_start_checkpoint", checkpoint),
            )
        )
    return {
        "treatment": label,
        "run_root": str(run_root),
        "profile": installed,
        "plan_profile": _artifact("plan_profile", plan_profile),
        "ablation_metadata": {
            "path": str(metadata_path),
            "report": metadata["report"],
            "treatment": label,
            "source_run_root": metadata.get("source_run_root"),
            "source_run_id": metadata.get("source_run_id"),
            "source_generation_family": metadata.get("source_generation_family"),
            "source_created_ns": metadata.get("source_created_ns"),
            "source_winner_snapshot": metadata.get("source_winner_snapshot"),
            "source_manifest_selection": selection,
            "training_objective": metadata.get("training_objective"),
            "promotion_objective": metadata.get("promotion_objective"),
            "per_ring_guarantees": metadata.get("per_ring_guarantees"),
            "futility_policy": metadata.get("futility_policy"),
            "anchor": metadata.get("anchor"),
        },
        "seed_contract": {
            "train_seed": experiment.train.seed,
            "selfplay_seed": experiment.selfplay.seed,
            "arena_seed": experiment.arena.seed,
        },
        "state_artifacts": state_artifacts,
    }


def generate_deployment_manifest(
    *,
    plan_path: Path,
    output_path: Path,
    training_dir: Path,
    queue_unit: Path,
    finalize_unit: Path,
    environment_file: Path,
    state_path: Path,
    comparison_output: Path,
    continuity_handoff_output: Path | None = None,
    execution_lock_path: Path | None = None,
    source_commit: str | None = None,
    orchestrator: str = "startrain-orchestrate",
    poll_seconds: float = 5.0,
    max_transient_retries: int = 2,
    retry_delay_seconds: float = 30.0,
    continue_after_fatal: bool = False,
    provisioned_gpus: int = DEFAULT_PROVISIONED_GPUS,
    replay_backup_service_unit: Path | None = None,
    replay_backup_timer_unit: Path | None = None,
    replay_backup_interval_seconds: float | None = None,
    replay_backup_retain: int = 3,
    replay_backup_max_total_bytes: int = 20 * 1024 * 1024 * 1024,
) -> dict[str, object]:
    """Create a manifest that pins every launch-critical deployment artifact."""
    output = output_path.expanduser().resolve()
    if output.exists():
        raise FileExistsError(f"deployment manifest already exists: {output}")
    training = training_dir.expanduser().resolve()
    if not training.is_dir():
        raise AblationQueueError(f"training directory does not exist: {training}")
    plan_file = plan_path.expanduser().resolve()
    plan = _read_json(plan_file)
    if plan.get("report") != "startrain-elo-ablation-plan":
        raise AblationQueueError("unsupported ablation plan")
    plan_seed = _positive_integer(plan.get("seed"), "ablation plan seed")
    treatments = _plan_treatments(plan)
    source_root = (
        Path(_string(plan.get("source_run_root"), "plan source run root"))
        .expanduser()
        .resolve()
    )
    seed = _seed_snapshot(source_root)
    profiles = [
        _profile_manifest_entry(
            treatment,
            seed=seed,
            expected_seed=plan_seed,
            source_winner_snapshot=plan.get("source_winner_snapshot"),
            futility_policy=plan.get("futility_policy"),
        )
        for treatment in treatments
    ]
    if source_commit is None:
        resolved_commit, clean = _git_revision(training)
        if not clean:
            raise AblationQueueError(
                "source tree is dirty; commit the deployment revision first"
            )
    else:
        resolved_commit = _commit(source_commit, "source commit")
    if not orchestrator:
        raise AblationQueueError("orchestrator must be non-empty")
    poll = _positive_float(poll_seconds, "poll seconds")
    retries = _nonnegative_integer(
        max_transient_retries,
        "maximum transient retries",
    )
    retry_delay = _positive_float(
        retry_delay_seconds,
        "retry delay seconds",
        allow_zero=True,
    )
    if type(continue_after_fatal) is not bool:
        raise AblationQueueError("continue-after-fatal policy must be boolean")
    if type(provisioned_gpus) is not int or provisioned_gpus <= 0:
        raise AblationQueueError("provisioned GPUs must be a positive integer")
    replay_backup = None
    if replay_backup_interval_seconds is None:
        if (
            replay_backup_service_unit is not None
            or replay_backup_timer_unit is not None
        ):
            raise AblationQueueError(
                "replay backup units require a configured backup interval"
            )
    else:
        if replay_backup_service_unit is None or replay_backup_timer_unit is None:
            raise AblationQueueError(
                "replay backup requires both service and timer units"
            )
        replay_backup = {
            "enabled": True,
            "interval_seconds": _positive_float(
                replay_backup_interval_seconds,
                "replay backup interval seconds",
            ),
            "retain": _positive_integer(
                replay_backup_retain,
                "replay backup retention count",
            ),
            "max_total_bytes": _positive_integer(
                replay_backup_max_total_bytes,
                "replay backup maximum bytes",
            ),
        }
    resolved_state = state_path.expanduser().resolve()
    resolved_comparison = comparison_output.expanduser().resolve()
    resolved_handoff = (
        continuity_handoff_output.expanduser().resolve()
        if continuity_handoff_output is not None
        else resolved_state.with_name("continuity-handoff-request.json")
    )
    resolved_execution_lock = (
        execution_lock_path.expanduser().resolve()
        if execution_lock_path is not None
        else resolved_state.parent.parent
        / "edgeconnect-startrain-ablation-execution.lock"
    )
    queue_outputs = {
        resolved_state,
        resolved_comparison,
        resolved_handoff,
        resolved_execution_lock,
    }
    if len(queue_outputs) != 4 or output in queue_outputs:
        raise AblationQueueError(
            "manifest, queue state, comparison output, continuity handoff, "
            "and execution lock paths must be distinct"
        )
    raw_guard_rings = plan.get("guard_rings", list(DEFAULT_GUARD_RINGS))
    guard_rings = _list(raw_guard_rings, "plan guard rings")
    if any(type(ring) is not int or ring <= 0 for ring in guard_rings):
        raise AblationQueueError("plan guard rings must be positive integers")
    if len(set(guard_rings)) != len(guard_rings):
        raise AblationQueueError("plan guard rings must be unique")
    guard_floor = plan.get("guard_floor_elo", DEFAULT_GUARD_FLOOR_ELO)
    if (
        isinstance(guard_floor, bool)
        or not isinstance(guard_floor, int | float)
        or not math.isfinite(float(guard_floor))
        or float(guard_floor) >= 0
    ):
        raise AblationQueueError("plan guard floor must be finite and negative")
    scripts = [
        _artifact(name.removesuffix(".py"), training / "scripts" / name)
        for name in _BASE_SCRIPT_NAMES
    ]
    units = [
        _unit_artifact("queue", queue_unit),
        _unit_artifact("finalize", finalize_unit),
    ]
    if replay_backup is not None:
        assert replay_backup_service_unit is not None
        assert replay_backup_timer_unit is not None
        _validate_replay_backup_units(
            service_unit=replay_backup_service_unit,
            timer_unit=replay_backup_timer_unit,
            backup_script=training / "scripts" / _BACKUP_SCRIPT_NAME,
            state_path=resolved_state,
            policy=replay_backup,
        )
        units.extend(
            (
                _unit_artifact(
                    "replay_backup_service",
                    replay_backup_service_unit,
                ),
                _unit_artifact(
                    "replay_backup_timer",
                    replay_backup_timer_unit,
                ),
            )
        )
    manifest: dict[str, object] = {
        "schema_version": SCHEMA_VERSION,
        "report": DEPLOYMENT_REPORT,
        "created_ns": time.time_ns(),
        "source": {
            "commit": resolved_commit,
            "training_dir": str(training),
            "clean_tree_required": True,
        },
        "plan": _artifact("ablation_plan", plan_file),
        "seed_snapshot": seed,
        "profiles": profiles,
        "scripts": scripts,
        "units": units,
        "environment": _artifact("environment", environment_file),
        "queue": {
            "seed": plan_seed,
            "state_path": str(resolved_state),
            "comparison_output": str(resolved_comparison),
            "continuity_handoff_output": str(resolved_handoff),
            "execution_lock_path": str(resolved_execution_lock),
            "orchestrator": orchestrator,
            "poll_seconds": poll,
            "policy": {
                "max_transient_retries": retries,
                "retry_delay_seconds": retry_delay,
                "continue_after_fatal": continue_after_fatal,
            },
            "comparison": {
                "provisioned_gpus": provisioned_gpus,
                "guard_rings": guard_rings,
                "guard_floor_elo": float(guard_floor),
            },
        },
    }
    if replay_backup is not None:
        queue_manifest = _mapping(manifest["queue"], "deployment queue")
        queue_manifest["replay_backup"] = replay_backup
    output.parent.mkdir(parents=True, exist_ok=True)
    atomic_json(output, manifest)
    return manifest


def _load_deployment_manifest(
    path: Path,
    *,
    expected_sha256: str | None = None,
) -> tuple[Path, dict[str, Any]]:
    manifest_path = path.expanduser().resolve()
    try:
        payload = manifest_path.read_bytes()
        manifest = json.loads(payload)
    except (OSError, json.JSONDecodeError) as error:
        raise AblationQueueError(
            f"cannot read JSON object {manifest_path}: {type(error).__name__}: {error}"
        ) from error
    if not isinstance(manifest, dict):
        raise AblationQueueError(f"{manifest_path} must contain a JSON object")
    observed_sha256 = hashlib.sha256(payload).hexdigest()
    if expected_sha256 is not None and observed_sha256 != expected_sha256:
        raise AblationQueueError("deployment manifest differs from its external pin")
    if (
        manifest.get("schema_version") != SCHEMA_VERSION
        or manifest.get("report") != DEPLOYMENT_REPORT
    ):
        raise AblationQueueError("unsupported ablation deployment manifest")
    _mapping(manifest.get("source"), "deployment source")
    _mapping(manifest.get("plan"), "deployment plan")
    _mapping(manifest.get("seed_snapshot"), "deployment seed snapshot")
    _list(manifest.get("profiles"), "deployment profiles")
    _list(manifest.get("scripts"), "deployment scripts")
    _list(manifest.get("units"), "deployment units")
    _mapping(manifest.get("environment"), "deployment environment")
    _mapping(manifest.get("queue"), "deployment queue")
    return manifest_path, manifest


def _verify_artifact(raw: object, name: str) -> Path:
    artifact = _mapping(raw, name)
    path = Path(
        os.path.abspath(
            Path(_string(artifact.get("path"), f"{name} path")).expanduser()
        )
    )
    expected = _string(artifact.get("sha256"), f"{name} digest")
    if not _SHA256_PATTERN.fullmatch(expected):
        raise AblationQueueError(f"{name} digest is invalid")
    if not path.is_file():
        raise AblationQueueError(f"{name} is missing: {path}")
    actual = _sha256(path)
    if actual != expected:
        raise AblationQueueError(
            f"{name} digest mismatch: expected {expected}, observed {actual}"
        )
    return path


def _verify_seed_artifact(raw: object, name: str) -> Path:
    artifact = _mapping(raw, name)
    pin = artifact.get("pin")
    if pin is None:
        return _verify_artifact(artifact, name)
    if pin != _SEMANTIC_JSON_PIN:
        raise AblationQueueError(f"{name} pin mode is unsupported")
    if artifact.get("name") != "replay_initialization":
        raise AblationQueueError(f"{name} semantic pin is not permitted")
    path = Path(
        os.path.abspath(
            Path(_string(artifact.get("path"), f"{name} path")).expanduser()
        )
    )
    if not path.is_file():
        raise AblationQueueError(f"{name} is missing: {path}")
    expected = dict(_mapping(artifact.get("fields"), f"{name} fields"))
    observed = _replay_initialization_fields(path)
    if observed != expected:
        raise AblationQueueError(
            f"{name} semantic fields changed: expected {expected}, observed {observed}"
        )
    return path


def _verify_manifest_artifacts(manifest: Mapping[str, object]) -> int:
    count = 0
    _verify_artifact(manifest.get("plan"), "ablation plan")
    count += 1
    seed = _mapping(manifest.get("seed_snapshot"), "seed snapshot")
    for index, artifact in enumerate(_list(seed.get("artifacts"), "seed artifacts")):
        _verify_seed_artifact(artifact, f"seed artifact {index}")
        count += 1
    profiles = _list(manifest.get("profiles"), "deployment profiles")
    for index, raw_profile in enumerate(profiles):
        profile = _mapping(raw_profile, f"deployment profile {index}")
        _verify_artifact(profile.get("profile"), f"profile {index}")
        _verify_artifact(profile.get("plan_profile"), f"plan profile {index}")
        count += 2
        for artifact_index, artifact in enumerate(
            _list(profile.get("state_artifacts"), f"profile {index} state artifacts")
        ):
            _verify_artifact(
                artifact,
                f"profile {index} state artifact {artifact_index}",
            )
            count += 1
    scripts = _list(manifest.get("scripts"), "deployment scripts")
    required_scripts = _required_script_names(manifest)
    if len(scripts) != len(required_scripts):
        raise AblationQueueError("deployment must pin all ablation scripts")
    script_names = {
        _string(_mapping(script, "script").get("name"), "script name")
        for script in scripts
    }
    if script_names != {name.removesuffix(".py") for name in required_scripts}:
        raise AblationQueueError("deployment script set is incomplete")
    for index, script in enumerate(scripts):
        _verify_artifact(script, f"script {index}")
        count += 1
    units = _list(manifest.get("units"), "deployment units")
    unit_names = {
        _string(_mapping(unit, "unit").get("name"), "unit name") for unit in units
    }
    if unit_names != _required_unit_names(manifest):
        raise AblationQueueError("deployment unit set is incomplete")
    for index, unit in enumerate(units):
        _verify_artifact(unit, f"unit {index}")
        count += 1
    _verify_artifact(manifest.get("environment"), "environment file")
    return count + 1


def _seed_artifacts_match(
    frozen_value: object,
    current_value: object,
) -> bool:
    frozen = _list(frozen_value, "frozen seed artifacts")
    current = _list(current_value, "current seed artifacts")
    if len(frozen) != len(current):
        return False
    current_by_name = {
        _string(
            _mapping(artifact, "current seed artifact").get("name"), "artifact name"
        ): _mapping(artifact, "current seed artifact")
        for artifact in current
    }
    if len(current_by_name) != len(current):
        return False
    for raw_frozen in frozen:
        frozen_artifact = _mapping(raw_frozen, "frozen seed artifact")
        name = _string(frozen_artifact.get("name"), "frozen artifact name")
        current_artifact = current_by_name.get(name)
        if current_artifact is None:
            return False
        if name == "replay_initialization" and frozen_artifact.get("pin") is None:
            path = Path(
                _string(
                    current_artifact.get("path"),
                    "current replay initialization path",
                )
            )
            observed: Mapping[str, object] = _artifact(name, path)
        else:
            observed = current_artifact
        if dict(observed) != dict(frozen_artifact):
            return False
    return True


def _verify_manifest_semantics(manifest: Mapping[str, object]) -> None:
    seed = _mapping(manifest.get("seed_snapshot"), "seed snapshot")
    source_root = Path(_string(seed.get("root"), "seed root")).expanduser().resolve()
    current_seed = _seed_snapshot(source_root)
    if current_seed["run_identity"] != seed.get("run_identity"):
        raise AblationQueueError("seed snapshot run identity changed")
    if current_seed["champion"] != seed.get("champion"):
        raise AblationQueueError("seed snapshot champion changed")
    if not _seed_artifacts_match(
        seed.get("artifacts"),
        current_seed["artifacts"],
    ):
        raise AblationQueueError("seed snapshot artifact set changed")
    plan_path = Path(
        _string(
            _mapping(manifest.get("plan"), "deployment plan").get("path"),
            "plan path",
        )
    )
    plan = _read_json(plan_path)
    plan_seed = _positive_integer(plan.get("seed"), "ablation plan seed")
    queue = _mapping(manifest.get("queue"), "deployment queue")
    if queue.get("seed") != plan_seed:
        raise AblationQueueError("deployment queue seed differs from the plan")
    planned_source = (
        Path(_string(plan.get("source_run_root"), "plan source run root"))
        .expanduser()
        .resolve()
    )
    if planned_source != source_root:
        raise AblationQueueError("seed snapshot root differs from the plan")
    planned = {entry["treatment"]: entry for entry in _plan_treatments(plan)}
    raw_profiles = _list(manifest.get("profiles"), "deployment profiles")
    labels: set[str] = set()
    roots: set[Path] = set()
    for index, raw_profile in enumerate(raw_profiles):
        profile_entry = _mapping(raw_profile, f"deployment profile {index}")
        label = _string(profile_entry.get("treatment"), f"profile {index} treatment")
        run_root = (
            Path(_string(profile_entry.get("run_root"), f"{label} run root"))
            .expanduser()
            .resolve()
        )
        if label in labels or run_root in roots:
            raise AblationQueueError("deployment profiles contain duplicates")
        labels.add(label)
        roots.add(run_root)
        planned_entry = planned.get(label)
        if planned_entry is None or Path(planned_entry["run_root"]) != run_root:
            raise AblationQueueError(f"{label} deployment differs from the plan")
        profile_artifact = _mapping(profile_entry.get("profile"), f"{label} profile")
        profile_path = Path(_string(profile_artifact.get("path"), f"{label} profile"))
        if profile_artifact.get("sha256") != planned_entry["profile_sha256"]:
            raise AblationQueueError(f"{label} profile digest differs from the plan")
        if profile_path.resolve() != run_root / "profile-elo-ablation.yaml":
            raise AblationQueueError(f"{label} installed profile path changed")
        plan_profile = _mapping(
            profile_entry.get("plan_profile"),
            f"{label} plan profile",
        )
        if Path(
            _string(plan_profile.get("path"), f"{label} plan profile path")
        ).expanduser().resolve() != Path(planned_entry["profile"]):
            raise AblationQueueError(f"{label} plan profile path changed")
        experiment = load_config(profile_path)
        if (
            Path(experiment.orchestration.directories.root).expanduser().resolve()
            != run_root
        ):
            raise AblationQueueError(f"{label} profile run root changed")
        seed_contract = _mapping(
            profile_entry.get("seed_contract"),
            f"{label} seed contract",
        )
        expected_seed_contract = {
            "train_seed": plan_seed,
            "selfplay_seed": plan_seed,
            "arena_seed": plan_seed,
        }
        if (
            dict(seed_contract) != expected_seed_contract
            or experiment.train.seed != plan_seed
            or experiment.selfplay.seed != plan_seed
            or experiment.arena.seed != plan_seed
        ):
            raise AblationQueueError(f"{label} profile seed contract changed")
        frozen_metadata = _mapping(
            profile_entry.get("ablation_metadata"),
            f"{label} ablation metadata",
        )
        metadata_path = Path(
            _string(frozen_metadata.get("path"), f"{label} metadata path")
        )
        if metadata_path.resolve() != run_root / "ablation.json":
            raise AblationQueueError(f"{label} ablation metadata path changed")
        metadata = _read_json(metadata_path)
        for field in (
            "report",
            "treatment",
            "source_run_root",
            "source_run_id",
            "source_generation_family",
            "source_created_ns",
            "source_winner_snapshot",
            "source_manifest_selection",
            "training_objective",
            "promotion_objective",
            "per_ring_guarantees",
            "futility_policy",
            "anchor",
        ):
            if metadata.get(field) != frozen_metadata.get(field):
                raise AblationQueueError(
                    f"{label} immutable ablation metadata field changed: {field}"
                )
        metadata_source = metadata.get("source_run_root")
        if (
            not isinstance(metadata_source, str)
            or Path(metadata_source).expanduser().resolve() != source_root
        ):
            raise AblationQueueError(f"{label} metadata seed root changed")
        anchor = _mapping(metadata.get("anchor"), f"{label} ablation anchor")
        seed_champion = _mapping(seed.get("champion"), "seed champion")
        selection = metadata.get("source_manifest_selection")
        expected_anchor: Mapping[str, object] = seed_champion
        if selection is not None:
            selection_metadata = _mapping(
                selection,
                f"{label} manifest selection",
            )
            expected_anchor = _mapping(
                selection_metadata.get("selected_manifest"),
                f"{label} selected manifest",
            )
        if any(
            anchor.get(field) != expected_anchor.get(field)
            for field in ("model_identity", "model_step")
        ):
            raise AblationQueueError(f"{label} anchor changed from verified source")
        if (
            not isinstance(metadata.get("profile"), str)
            or Path(str(metadata["profile"])).expanduser().resolve()
            != profile_path.resolve()
            or metadata.get("profile_sha256") != profile_artifact.get("sha256")
        ):
            raise AblationQueueError(f"{label} metadata profile authority changed")
    if labels != set(planned):
        raise AblationQueueError("deployment profiles do not cover the full plan")


def verify_deployment_manifest(
    manifest_path: Path,
    *,
    current_source_commit: str | None = None,
    source_tree_clean: bool | None = None,
    expected_manifest_sha256: str | None = None,
) -> dict[str, object]:
    """Verify revision identity and every launch-critical artifact digest."""
    resolved_path, manifest = _load_deployment_manifest(
        manifest_path,
        expected_sha256=expected_manifest_sha256,
    )
    source = _mapping(manifest.get("source"), "deployment source")
    expected_commit = _commit(source.get("commit"), "deployment source commit")
    training_dir = (
        Path(_string(source.get("training_dir"), "deployment training directory"))
        .expanduser()
        .resolve()
    )
    if current_source_commit is None:
        observed_commit, observed_clean = _git_revision(training_dir)
    else:
        observed_commit = _commit(current_source_commit, "current source commit")
        observed_clean = True if source_tree_clean is None else source_tree_clean
    if observed_commit != expected_commit:
        raise AblationQueueError(
            "mixed-revision launch refused: current source commit "
            f"{observed_commit} != deployment commit {expected_commit}"
        )
    if source.get("clean_tree_required") is not True or observed_clean is not True:
        raise AblationQueueError(
            "mixed-revision launch refused: source tree is not clean"
        )
    artifact_count = _verify_manifest_artifacts(manifest)
    scripts = _list(manifest.get("scripts"), "deployment scripts")
    for raw_script in scripts:
        script = _mapping(raw_script, "deployment script")
        name = _string(script.get("name"), "deployment script name")
        expected_path = training_dir / "scripts" / f"{name}.py"
        observed_path = Path(
            _string(script.get("path"), f"{name} script path")
        ).resolve()
        if observed_path != expected_path:
            raise AblationQueueError(f"{name} script is outside the training revision")
    _verify_manifest_semantics(manifest)
    if current_source_commit is None:
        final_commit, final_clean = _git_revision(training_dir)
        if final_commit != expected_commit or not final_clean:
            raise AblationQueueError(
                "mixed-revision launch refused: source changed during verification"
            )
    return {
        "schema_version": SCHEMA_VERSION,
        "report": DEPLOYMENT_REPORT,
        "status": "verified",
        "manifest": str(resolved_path),
        "source_commit": expected_commit,
        "artifact_count": artifact_count,
    }


def _queue_config(manifest: Mapping[str, object]) -> dict[str, Any]:
    queue = _mapping(manifest.get("queue"), "deployment queue")
    _string(queue.get("state_path"), "queue state path")
    _string(queue.get("comparison_output"), "comparison output")
    handoff_output = queue.get("continuity_handoff_output")
    if handoff_output is not None:
        _string(handoff_output, "continuity handoff output")
    _string(queue.get("execution_lock_path"), "execution lock path")
    _string(queue.get("orchestrator"), "queue orchestrator")
    _positive_float(queue.get("poll_seconds"), "queue poll seconds")
    policy = _mapping(queue.get("policy"), "queue policy")
    _nonnegative_integer(
        policy.get("max_transient_retries"),
        "maximum transient retries",
    )
    _positive_float(
        policy.get("retry_delay_seconds"),
        "retry delay seconds",
        allow_zero=True,
    )
    if type(policy.get("continue_after_fatal")) is not bool:
        raise AblationQueueError("continue-after-fatal policy must be boolean")
    _replay_backup_policy(queue)
    comparison = _mapping(queue.get("comparison"), "comparison configuration")
    if (
        type(comparison.get("provisioned_gpus")) is not int
        or comparison["provisioned_gpus"] <= 0
    ):
        raise AblationQueueError(
            "comparison provisioned GPUs must be a positive integer"
        )
    guard_rings = _list(comparison.get("guard_rings"), "comparison guard rings")
    if any(type(ring) is not int or ring <= 0 for ring in guard_rings):
        raise AblationQueueError("comparison guard rings must be positive integers")
    if len(set(guard_rings)) != len(guard_rings):
        raise AblationQueueError("comparison guard rings must be unique")
    guard_floor = comparison.get("guard_floor_elo")
    if (
        isinstance(guard_floor, bool)
        or not isinstance(guard_floor, int | float)
        or not math.isfinite(float(guard_floor))
        or float(guard_floor) >= 0
    ):
        raise AblationQueueError("comparison guard floor must be finite and negative")
    return queue


def _manifest_arms(manifest: Mapping[str, object]) -> list[dict[str, str]]:
    arms: list[dict[str, str]] = []
    for index, raw_profile in enumerate(
        _list(manifest.get("profiles"), "deployment profiles")
    ):
        profile = _mapping(raw_profile, f"deployment profile {index}")
        profile_artifact = _mapping(profile.get("profile"), f"profile {index}")
        arms.append(
            {
                "treatment": _string(
                    profile.get("treatment"),
                    f"profile {index} treatment",
                ),
                "profile": _string(
                    profile_artifact.get("path"),
                    f"profile {index} path",
                ),
                "run_root": _string(
                    profile.get("run_root"),
                    f"profile {index} run root",
                ),
            }
        )
    return arms


def _initial_state(
    manifest_path: Path,
    manifest: Mapping[str, object],
) -> dict[str, object]:
    now_ns = time.time_ns()
    queue = _queue_config(manifest)
    arms = [
        {
            **arm,
            "status": "pending",
            "attempts": 0,
            "transient_failures": 0,
            "last_started_ns": None,
            "last_stopped_ns": None,
            "last_outcome": None,
            "last_exit_code": None,
            "failure": None,
            "failure_domain": None,
            "failure_phase": None,
            "structured_lifecycle": False,
            "measurement_cutoff_ns": None,
            "resource_released_ns": None,
            "teardown_status": None,
            "teardown": None,
            "integrity_status": None,
            "integrity": None,
            "completion_status": None,
            "lifecycle_warnings": [],
            "quarantine": None,
            "replay_restore": None,
            "replay_backup": None,
        }
        for arm in _manifest_arms(manifest)
    ]
    return {
        "schema_version": SCHEMA_VERSION,
        "report": QUEUE_REPORT,
        "manifest": str(manifest_path),
        "manifest_sha256": _sha256(manifest_path),
        "created_ns": now_ns,
        "updated_ns": now_ns,
        "queue_status": "pending",
        "queue_error": None,
        "policy": queue["policy"],
        "arms": arms,
        "finalization": {
            "status": "pending",
            "attempts": 0,
            "comparison_output": queue["comparison_output"],
            "comparison_status": None,
            "comparison_sha256": None,
            "selector_summary": None,
            "started_ns": None,
            "stopped_ns": None,
            "error": None,
        },
        "continuity_handoff": {
            "status": "not_requested",
            "requested": False,
            "action": None,
            "requested_action": None,
            "queue_status": None,
            "path": str(_continuity_handoff_path(manifest)),
            "requested_ns": None,
            "reason": None,
            "error": None,
        },
    }


def _state_path(manifest: Mapping[str, object]) -> Path:
    queue = _queue_config(manifest)
    return (
        Path(_string(queue.get("state_path"), "queue state path"))
        .expanduser()
        .resolve()
    )


def _continuity_handoff_path(manifest: Mapping[str, object]) -> Path:
    queue = _queue_config(manifest)
    configured = queue.get("continuity_handoff_output")
    if configured is None:
        return _state_path(manifest).with_name("continuity-handoff-request.json")
    return Path(_string(configured, "continuity handoff output")).expanduser().resolve()


def _load_or_create_state(
    state_path: Path,
    *,
    manifest_path: Path,
    manifest: Mapping[str, object],
) -> dict[str, Any]:
    if not state_path.exists():
        state = _initial_state(manifest_path, manifest)
        _save_state(state_path, state)
        return state
    state = _read_json(state_path)
    if (
        state.get("schema_version") != SCHEMA_VERSION
        or state.get("report") != QUEUE_REPORT
    ):
        raise AblationQueueError("unsupported ablation queue state")
    if state.get("queue_status") not in {"pending", "running", "completed", "failed"}:
        raise AblationQueueError("queue state has an invalid overall status")
    if state.get("manifest") != str(manifest_path) or state.get(
        "manifest_sha256"
    ) != _sha256(manifest_path):
        raise AblationQueueError("queue state belongs to a different deployment")
    if state.get("policy") != _queue_config(manifest).get("policy"):
        raise AblationQueueError("queue state policy differs from the deployment")
    raw_arms = _list(state.get("arms"), "queue state arms")
    expected = _manifest_arms(manifest)
    if len(raw_arms) != len(expected):
        raise AblationQueueError("queue state arm count differs from the deployment")
    for index, (raw_arm, expected_arm) in enumerate(
        zip(raw_arms, expected, strict=True)
    ):
        arm = _mapping(raw_arm, f"queue arm {index}")
        if any(arm.get(name) != value for name, value in expected_arm.items()):
            raise AblationQueueError(f"queue arm {index} identity changed")
        if arm.get("status") not in _ARM_STATUSES:
            raise AblationQueueError(f"queue arm {index} has an invalid status")
        _nonnegative_integer(arm.get("attempts"), f"queue arm {index} attempts")
        _nonnegative_integer(
            arm.get("transient_failures"),
            f"queue arm {index} transient failures",
        )
        for field in (
            "failure_domain",
            "failure_phase",
            "structured_lifecycle",
            "measurement_cutoff_ns",
            "resource_released_ns",
            "teardown_status",
            "teardown",
            "integrity_status",
            "integrity",
            "completion_status",
            "lifecycle_warnings",
            "quarantine",
            "replay_restore",
            "replay_backup",
        ):
            arm.setdefault(field, None)
    finalization = _mapping(state.get("finalization"), "queue finalization")
    if finalization.get("status") not in {"pending", "running", "completed", "failed"}:
        raise AblationQueueError("queue finalization has an invalid status")
    _nonnegative_integer(
        finalization.get("attempts"),
        "queue finalization attempts",
    )
    finalization.setdefault("comparison_sha256", None)
    finalization.setdefault("selector_summary", None)
    raw_handoff = state.get("continuity_handoff")
    if raw_handoff is None:
        state["continuity_handoff"] = {
            "status": "not_requested",
            "requested": False,
            "action": None,
            "requested_action": None,
            "queue_status": None,
            "path": str(_continuity_handoff_path(manifest)),
            "requested_ns": None,
            "reason": None,
            "error": None,
        }
    else:
        handoff = _mapping(raw_handoff, "continuity handoff")
        if handoff.get("status") not in {
            "not_requested",
            "pending",
            "requested",
            "blocked",
            "failed",
        }:
            raise AblationQueueError("continuity handoff has an invalid status")
        expected_handoff = str(_continuity_handoff_path(manifest))
        if handoff.get("path") != expected_handoff:
            raise AblationQueueError("continuity handoff path differs from deployment")
    return state


def _save_state(path: Path, state: dict[str, Any]) -> None:
    state["updated_ns"] = time.time_ns()
    atomic_json(path, state)


@contextmanager
def _exclusive_lock(lock_path: Path, owner: str) -> Iterator[Path]:
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    descriptor = os.open(lock_path, os.O_RDWR | os.O_CREAT, 0o600)
    try:
        try:
            fcntl.flock(descriptor, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as error:
            raise QueueBusyError(f"another {owner} owns {lock_path}") from error
        yield lock_path
    finally:
        try:
            fcntl.flock(descriptor, fcntl.LOCK_UN)
        finally:
            os.close(descriptor)


@contextmanager
def exclusive_queue_lock(state_path: Path) -> Iterator[Path]:
    """Lock one queue's durable state against runners and finalizers."""
    with _exclusive_lock(
        Path(f"{state_path}.lock"),
        "ablation queue process",
    ) as lock_path:
        yield lock_path


@contextmanager
def exclusive_execution_lock(lock_path: Path) -> Iterator[ExecutionLockLease]:
    """Prevent distinct ablation deployments from sharing one host."""
    resolved = lock_path.expanduser().resolve()
    resolved.parent.mkdir(parents=True, exist_ok=True)
    descriptor = os.open(resolved, os.O_RDWR | os.O_CREAT, 0o600)
    lease: ExecutionLockLease | None = None
    try:
        try:
            fcntl.flock(descriptor, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as error:
            raise QueueBusyError(
                f"another ablation execution owns {resolved}"
            ) from error
        lease = ExecutionLockLease(
            resolved,
            descriptor,
            token=_EXECUTION_LOCK_LEASE_TOKEN,
        )
        yield lease
    finally:
        if lease is not None:
            lease._invalidate()
        try:
            fcntl.flock(descriptor, fcntl.LOCK_UN)
        finally:
            os.close(descriptor)


def _state_arms(state: Mapping[str, object]) -> list[dict[str, Any]]:
    return [
        _mapping(arm, f"queue arm {index}")
        for index, arm in enumerate(_list(state.get("arms"), "queue state arms"))
    ]


def _nested_mapping(value: object) -> dict[str, object]:
    return dict(value) if isinstance(value, Mapping) else {}


def _first_present(
    sources: tuple[Mapping[str, object], ...],
    *names: str,
) -> object:
    for source in sources:
        for name in names:
            if name in source:
                return source[name]
    return None


def _status_name(value: object) -> str | None:
    if isinstance(value, str) and value:
        return value.lower().replace("-", "_").replace(" ", "_")
    return None


def _lifecycle_view(payload: Mapping[str, object]) -> dict[str, object]:
    lifecycle = _nested_mapping(payload.get("lifecycle"))
    measurement = _nested_mapping(lifecycle.get("measurement"))
    if not measurement:
        measurement = _nested_mapping(payload.get("measurement"))
    teardown = _nested_mapping(
        _first_present(
            (payload, lifecycle),
            "teardown",
            "measurement_teardown",
        )
    )
    integrity = _nested_mapping(
        _first_present(
            (payload, lifecycle, teardown),
            "integrity",
            "post_cutoff_integrity",
            "measurement_integrity",
        )
    )
    measurement_sources = (measurement, payload, lifecycle)
    status = _status_name(
        _first_present(measurement_sources, "measurement_status", "status")
    )
    outcome = _status_name(
        _first_present(measurement_sources, "measurement_outcome", "outcome")
    )
    cutoff_ns = _first_present(
        measurement_sources,
        "measurement_cutoff_ns",
        "cutoff_ns",
        "measurement_stopped_ns",
        "stopped_ns",
    )
    resource_released_ns = _first_present(
        (payload, lifecycle, teardown),
        "resource_released_ns",
        "released_ns",
    )
    raw_teardown_status = _first_present(
        (payload, lifecycle),
        "teardown_status",
    )
    if isinstance(raw_teardown_status, Mapping):
        raw_teardown_status = raw_teardown_status.get("status")
    if raw_teardown_status is None:
        raw_teardown_status = teardown.get("status")
    teardown_status = _status_name(raw_teardown_status)
    raw_integrity_status = _first_present(
        (payload, lifecycle, teardown),
        "integrity_status",
    )
    if isinstance(raw_integrity_status, Mapping):
        raw_integrity_status = raw_integrity_status.get("status")
    if raw_integrity_status is None:
        raw_integrity_status = integrity.get("status")
    integrity_status = _status_name(raw_integrity_status)
    failure_domain = _status_name(
        _first_present(
            measurement_sources,
            "failure_domain",
            "domain",
        )
    )
    failure_phase = _status_name(
        _first_present(
            measurement_sources,
            "failure_phase",
            "phase",
        )
    )
    structured = any(
        name in payload
        for name in (
            "lifecycle",
            "measurement",
            "measurement_cutoff_ns",
            "resource_released_ns",
            "teardown",
            "teardown_status",
            "integrity",
            "integrity_status",
            "failure_domain",
            "failure_phase",
        )
    ) or any((measurement, teardown, integrity))
    return {
        "structured": structured,
        "status": status,
        "outcome": outcome,
        "cutoff_ns": cutoff_ns if type(cutoff_ns) is int else None,
        "resource_released_ns": (
            resource_released_ns if type(resource_released_ns) is int else None
        ),
        "stop_reason": _first_present(
            measurement_sources,
            "measurement_stop_reason",
            "stop_reason",
        ),
        "exit_code": _first_present(
            measurement_sources,
            "measurement_exit_code",
            "exit_code",
        ),
        "attempt_count": _first_present(
            measurement_sources,
            "measurement_attempt_count",
            "attempt_count",
            "attempt",
        ),
        "failure": _first_present(
            measurement_sources,
            "measurement_failure",
            "failure",
        ),
        "completion_status": _status_name(
            _first_present(
                measurement_sources,
                "measurement_completion_status",
                "completion_status",
            )
        ),
        "warnings": _first_present(
            measurement_sources,
            "measurement_warnings",
            "warnings",
        ),
        "failure_domain": failure_domain,
        "failure_phase": failure_phase,
        "teardown_status": teardown_status,
        "teardown": teardown or None,
        "integrity_status": integrity_status,
        "integrity": integrity or None,
    }


def _integrity_is_valid(lifecycle: Mapping[str, object]) -> bool:
    integrity = _nested_mapping(lifecycle.get("integrity"))
    explicit_valid = integrity.get("valid")
    if type(explicit_valid) is bool:
        return explicit_valid
    status = _status_name(lifecycle.get("integrity_status"))
    return status in _VALID_INTEGRITY_STATUSES


def _teardown_has_warning(lifecycle: Mapping[str, object]) -> bool:
    status = _status_name(lifecycle.get("teardown_status"))
    return status is not None and status not in _CLEAN_TEARDOWN_STATUSES


def _is_budget_completion(lifecycle: Mapping[str, object]) -> bool:
    if lifecycle.get("outcome") != BUDGET_COMPLETION:
        return False
    if lifecycle.get("status") not in {
        "complete",
        "completed",
        "completed_with_teardown_failure",
        "completed_with_teardown_warning",
    }:
        return False
    if lifecycle.get("completion_status") not in {
        None,
        "complete",
        "completed",
        "complete_with_warning",
        "completed_with_teardown_failure",
        "completed_with_teardown_warning",
    }:
        return False
    if not lifecycle.get("structured"):
        return True
    cutoff_ns = lifecycle.get("cutoff_ns")
    released_ns = lifecycle.get("resource_released_ns")
    if (
        isinstance(cutoff_ns, bool)
        or not isinstance(cutoff_ns, int)
        or isinstance(released_ns, bool)
        or not isinstance(released_ns, int)
        or released_ns < cutoff_ns
    ):
        return False
    integrity_status = _status_name(lifecycle.get("integrity_status"))
    if integrity_status is not None and not _integrity_is_valid(lifecycle):
        return False
    if _teardown_has_warning(lifecycle):
        phase = _status_name(lifecycle.get("failure_phase"))
        if phase in {"measurement", "pre_cutoff", "pre_budget"}:
            return False
        return _integrity_is_valid(lifecycle)
    return True


def _record_lifecycle(arm: dict[str, Any], lifecycle: Mapping[str, object]) -> None:
    arm["last_outcome"] = lifecycle.get("outcome")
    arm["last_exit_code"] = lifecycle.get("exit_code")
    arm["measurement_cutoff_ns"] = lifecycle.get("cutoff_ns")
    arm["resource_released_ns"] = lifecycle.get("resource_released_ns")
    arm["last_stopped_ns"] = lifecycle.get("resource_released_ns") or lifecycle.get(
        "cutoff_ns"
    )
    arm["failure"] = lifecycle.get("failure")
    arm["failure_domain"] = lifecycle.get("failure_domain")
    arm["failure_phase"] = lifecycle.get("failure_phase")
    arm["structured_lifecycle"] = lifecycle.get("structured") is True
    arm["teardown_status"] = lifecycle.get("teardown_status")
    arm["teardown"] = lifecycle.get("teardown")
    arm["integrity_status"] = lifecycle.get("integrity_status")
    arm["integrity"] = lifecycle.get("integrity")
    arm["completion_status"] = lifecycle.get("completion_status") or (
        "complete_with_warning"
        if _is_budget_completion(lifecycle) and _teardown_has_warning(lifecycle)
        else ("complete" if _is_budget_completion(lifecycle) else None)
    )
    raw_warnings = lifecycle.get("warnings")
    arm["lifecycle_warnings"] = (
        list(raw_warnings) if isinstance(raw_warnings, list) else []
    )


def _quarantine_arm(
    arm: dict[str, Any],
    lifecycle: Mapping[str, object],
) -> None:
    recorded_ns = time.time_ns()
    arm["status"] = "quarantined"
    arm["quarantine"] = {
        "schema_version": SCHEMA_VERSION,
        "status": "quarantined",
        "isolated": True,
        "recorded_ns": recorded_ns,
        "failure_domain": lifecycle.get("failure_domain"),
        "failure_phase": lifecycle.get("failure_phase"),
        "outcome": lifecycle.get("outcome"),
        "reason": lifecycle.get("failure")
        or lifecycle.get("outcome")
        or "isolated arm failure",
        "run_root": arm.get("run_root"),
    }


def _reconcile_arms(state: dict[str, Any]) -> None:
    for arm in _state_arms(state):
        if arm.get("status") in {"completed", "failed", "quarantined"}:
            continue
        metadata_path = Path(str(arm["run_root"])) / "ablation.json"
        metadata = _read_json(metadata_path)
        lifecycle = _lifecycle_view(metadata)
        attempt_count = lifecycle.get("attempt_count")
        if type(attempt_count) is int:
            arm["attempts"] = max(int(arm.get("attempts", 0)), attempt_count)
        _record_lifecycle(arm, lifecycle)
        if _is_budget_completion(lifecycle):
            arm["status"] = "completed"
        elif lifecycle.get("failure_domain") in _ISOLATED_FAILURE_DOMAINS and (
            lifecycle.get("status") == "failed"
            or lifecycle.get("outcome") in {FATAL_ORCHESTRATOR_EXIT, RUNNER_ERROR}
        ):
            _quarantine_arm(arm, lifecycle)
        elif lifecycle.get("status") == "failed" or lifecycle.get("outcome") in {
            BUDGET_COMPLETION,
            FATAL_ORCHESTRATOR_EXIT,
            RUNNER_ERROR,
        }:
            arm["status"] = "failed"
        else:
            arm["status"] = "pending"


def _forced_ineligible(state: Mapping[str, object]) -> dict[str, str]:
    forced = {}
    for arm in _state_arms(state):
        if arm.get("status") != "completed":
            detail = arm.get("failure") or arm.get("last_outcome") or "not completed"
            quarantine = arm.get("quarantine")
            if isinstance(quarantine, Mapping):
                detail = quarantine.get("reason") or detail
            forced[str(arm["treatment"])] = (
                f"queue arm is {arm.get('status')}: {detail}"
            )
    return forced


def _reconciled_queue_status(state: Mapping[str, object]) -> str:
    statuses = {str(arm.get("status")) for arm in _state_arms(state)}
    if statuses == {"completed"}:
        return "completed"
    if statuses & {"failed", "quarantined"}:
        return "failed"
    return "pending"


def _continuity_reason(state: Mapping[str, object]) -> str:
    statuses = {str(arm.get("status")) for arm in _state_arms(state)}
    if "quarantined" in statuses:
        return "queue_completed_with_quarantined_arms"
    if state.get("queue_status") == "completed":
        return "queue_completed"
    if state.get("queue_status") == "failed":
        return "queue_failed"
    return "queue_interrupted_or_pending"


def _selector_summary(report: Mapping[str, object]) -> dict[str, object]:
    raw_selector = report.get("selector")
    selector = raw_selector if isinstance(raw_selector, Mapping) else {}
    raw_snapshot = selector.get("winner_snapshot")
    snapshot = raw_snapshot if isinstance(raw_snapshot, Mapping) else {}
    raw_champion = snapshot.get("champion")
    champion = raw_champion if isinstance(raw_champion, Mapping) else {}
    return {
        "status": selector.get("status", "unavailable"),
        "ranking_objective": selector.get("ranking_objective"),
        "ranking_metric": selector.get("ranking_metric"),
        "winner_label": snapshot.get("label"),
        "winner_model_identity": champion.get("model_identity"),
        "winner_model_step": champion.get("model_step"),
        "single_seed_evidence_only": True,
        "adoption_authorized": False,
    }


def _comparison_handoff_evidence(
    finalization: Mapping[str, object],
) -> dict[str, object]:
    return {
        "path": finalization.get("comparison_output"),
        "sha256": finalization.get("comparison_sha256"),
        "status": finalization.get("comparison_status"),
        "selector_summary": finalization.get("selector_summary"),
        "adoption_authorized": False,
    }


def _request_continuity_handoff(
    *,
    state_path: Path,
    state: dict[str, Any],
    manifest: Mapping[str, object],
) -> dict[str, object]:
    """Write a durable request; an external controller decides what to start."""
    requested_ns = time.time_ns()
    reason = _continuity_reason(state)
    output = _continuity_handoff_path(manifest)
    finalization = _mapping(state.get("finalization"), "queue finalization")
    unreleased = [
        {
            "treatment": arm.get("treatment"),
            "run_root": arm.get("run_root"),
        }
        for arm in _state_arms(state)
        if arm.get("last_started_ns") is not None
        and arm.get("structured_lifecycle") is True
        and arm.get("resource_released_ns") is None
    ]
    if unreleased:
        blocked = {
            "schema_version": SCHEMA_VERSION,
            "report": CONTINUITY_HANDOFF_REPORT,
            "status": "blocked",
            "requested": False,
            "action": None,
            "requested_action": None,
            "requested_ns": requested_ns,
            "reason": "resources_not_released",
            "terminal_reason": "resources_not_released",
            "failure_domain": "host",
            "source": {
                "kind": "elo_ablation_queue",
                "manifest": state.get("manifest"),
                "queue_state": str(state_path),
            },
            "queue_status": state.get("queue_status"),
            "finalization": {
                "status": finalization.get("status"),
                "error": finalization.get("error"),
                "comparison": _comparison_handoff_evidence(finalization),
            },
            "unreleased_arms": unreleased,
            "requires_safe_workload": True,
        }
        output.parent.mkdir(parents=True, exist_ok=True)
        atomic_json(output, blocked)
        handoff = _mapping(state.get("continuity_handoff"), "continuity handoff")
        handoff.update(
            {
                "status": "blocked",
                "requested": False,
                "action": None,
                "requested_action": None,
                "queue_status": state.get("queue_status"),
                "path": str(output),
                "requested_ns": requested_ns,
                "reason": "resources_not_released",
                "error": None,
            }
        )
        _save_state(state_path, state)
        return blocked
    quarantined = []
    for arm in _state_arms(state):
        if arm.get("status") != "quarantined":
            continue
        quarantined.append(
            {
                "treatment": arm.get("treatment"),
                "run_root": arm.get("run_root"),
                "quarantine": arm.get("quarantine"),
            }
        )
    request: dict[str, object] = {
        "schema_version": SCHEMA_VERSION,
        "report": CONTINUITY_HANDOFF_REPORT,
        "status": "requested",
        "requested": True,
        "action": "request_fallback",
        "requested_action": "reconcile_training_continuity",
        "requested_ns": requested_ns,
        "reason": reason,
        "terminal_reason": reason,
        "failure_domain": (
            "run_failure" if state.get("queue_status") == "failed" else "queue_complete"
        ),
        "source": {
            "kind": "elo_ablation_queue",
            "manifest": state.get("manifest"),
            "queue_state": str(state_path),
            "queue_status": state.get("queue_status"),
            "queue_error": state.get("queue_error"),
        },
        "finalization": {
            "status": finalization.get("status"),
            "comparison_status": finalization.get("comparison_status"),
            "comparison_output": finalization.get("comparison_output"),
            "comparison_sha256": finalization.get("comparison_sha256"),
            "comparison": _comparison_handoff_evidence(finalization),
            "selector_summary": finalization.get("selector_summary"),
            "error": finalization.get("error"),
        },
        "quarantined_arms": quarantined,
        "requires_safe_workload": state.get("queue_status") != "running",
        "adoption_authorized": False,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    atomic_json(output, request)
    handoff = _mapping(state.get("continuity_handoff"), "continuity handoff")
    handoff.update(
        {
            "status": "requested",
            "requested": True,
            "action": "request_fallback",
            "requested_action": "reconcile_training_continuity",
            "queue_status": state.get("queue_status"),
            "path": str(output),
            "requested_ns": requested_ns,
            "reason": reason,
            "error": None,
        }
    )
    _save_state(state_path, state)
    return request


def _finalize_locked(
    *,
    state_path: Path,
    state: dict[str, Any],
    manifest: Mapping[str, object],
) -> dict[str, object]:
    finalization = _mapping(state.get("finalization"), "queue finalization")
    started_ns = time.time_ns()
    finalization.update(
        {
            "status": "running",
            "attempts": int(finalization.get("attempts", 0)) + 1,
            "comparison_sha256": None,
            "selector_summary": None,
            "started_ns": started_ns,
            "stopped_ns": None,
            "error": None,
        }
    )
    _save_state(state_path, state)
    queue = _queue_config(manifest)
    comparison_config = _mapping(queue.get("comparison"), "comparison configuration")
    runs = {
        str(arm["treatment"]): Path(str(arm["run_root"])) for arm in _state_arms(state)
    }
    try:
        report = build_elo_ablation_comparison(
            runs,
            provisioned_gpus=int(comparison_config["provisioned_gpus"]),
            guard_rings=tuple(int(ring) for ring in comparison_config["guard_rings"]),
            guard_floor_elo=float(comparison_config["guard_floor_elo"]),
            forced_ineligible=_forced_ineligible(state),
        )
        report["queue"] = {
            "seed": _positive_integer(
                queue.get("seed"),
                "deployment queue seed",
            ),
            "source_commit": _commit(
                _mapping(manifest.get("source"), "deployment source").get("commit"),
                "deployment source commit",
            ),
            "manifest": state.get("manifest"),
            "manifest_sha256": state.get("manifest_sha256"),
            "state_path": str(state_path),
            "queue_status": state.get("queue_status"),
            "arms": [
                {
                    key: arm.get(key)
                    for key in (
                        "treatment",
                        "status",
                        "attempts",
                        "transient_failures",
                        "last_outcome",
                        "last_exit_code",
                        "failure",
                        "failure_domain",
                        "failure_phase",
                        "measurement_cutoff_ns",
                        "resource_released_ns",
                        "teardown_status",
                        "teardown",
                        "integrity_status",
                        "integrity",
                        "completion_status",
                        "lifecycle_warnings",
                        "quarantine",
                        "replay_restore",
                        "replay_backup",
                    )
                }
                for arm in _state_arms(state)
            ],
        }
        output = (
            Path(_string(queue.get("comparison_output"), "comparison output"))
            .expanduser()
            .resolve()
        )
        atomic_json(output, report)
        comparison_sha256 = _sha256(output)
        selector_summary = _selector_summary(report)
    except (OSError, TypeError, ValueError, AblationQueueError) as error:
        finalization.update(
            {
                "status": "failed",
                "stopped_ns": time.time_ns(),
                "error": f"{type(error).__name__}: {error}",
            }
        )
        _save_state(state_path, state)
        raise AblationQueueError(f"final comparison failed: {error}") from error
    finalization.update(
        {
            "status": "completed",
            "comparison_status": report.get("status"),
            "comparison_sha256": comparison_sha256,
            "selector_summary": selector_summary,
            "stopped_ns": time.time_ns(),
            "error": None,
        }
    )
    _save_state(state_path, state)
    return report


def finalize_ablation_queue(manifest_path: Path) -> dict[str, object]:
    """Idempotently rebuild the final report, including failed and pending arms."""
    resolved_manifest, manifest = _load_deployment_manifest(manifest_path)
    state_path = _state_path(manifest)
    with exclusive_queue_lock(state_path):
        state = _load_or_create_state(
            state_path,
            manifest_path=resolved_manifest,
            manifest=manifest,
        )
        _reconcile_arms(state)
        _ensure_terminal_replay_backups(
            state,
            policy=_replay_backup_policy(_queue_config(manifest)),
        )
        if state.get("queue_status") in {"pending", "running"}:
            state["queue_status"] = _reconciled_queue_status(state)
        _save_state(state_path, state)
        try:
            return _finalize_locked(
                state_path=state_path,
                state=state,
                manifest=manifest,
            )
        finally:
            _request_continuity_handoff(
                state_path=state_path,
                state=state,
                manifest=manifest,
            )


def _record_replay_restore(arm: dict[str, Any]) -> None:
    metadata_path = Path(str(arm["run_root"])) / "ablation.json"
    try:
        metadata = _read_json(metadata_path)
    except AblationQueueError:
        return
    restore = metadata.get("replay_restore")
    arm["replay_restore"] = dict(restore) if isinstance(restore, Mapping) else None


def _record_final_replay_backup(
    arm: dict[str, Any],
    *,
    policy: Mapping[str, object] | None,
) -> None:
    if policy is None:
        return
    attempted_ns = time.time_ns()
    run_root = Path(str(arm["run_root"])).expanduser().resolve()
    try:
        destination, evidence = create_backup_with_evidence(
            run_root,
            retain=_positive_integer(
                policy.get("retain"),
                "replay backup retention count",
            ),
            max_total_bytes=_positive_integer(
                policy.get("max_total_bytes"),
                "replay backup maximum bytes",
            ),
        )
        record: dict[str, object] = {
            "schema_version": SCHEMA_VERSION,
            "status": "completed",
            "attempted_ns": attempted_ns,
            "completed_ns": time.time_ns(),
            "path": str(destination),
            "bytes": evidence.get("bytes"),
            "sha256": evidence.get("sha256"),
            "created_ns": evidence.get("created_ns"),
            "error": None,
        }
    except (
        OSError,
        RuntimeError,
        ValueError,
        sqlite3.Error,
        AblationQueueError,
    ) as error:
        record = {
            "schema_version": SCHEMA_VERSION,
            "status": "failed",
            "attempted_ns": attempted_ns,
            "completed_ns": time.time_ns(),
            "path": None,
            "bytes": None,
            "sha256": None,
            "created_ns": None,
            "error": f"{type(error).__name__}: {error}",
        }
    previous = arm.get("replay_backup")
    history = (
        list(previous.get("attempts", []))
        if isinstance(previous, Mapping) and isinstance(previous.get("attempts"), list)
        else []
    )
    history.append(record)
    arm["replay_backup"] = {
        "schema_version": SCHEMA_VERSION,
        "enabled": True,
        "latest": record,
        "attempts": history,
    }


def _ensure_terminal_replay_backups(
    state: dict[str, Any],
    *,
    policy: Mapping[str, object] | None,
) -> None:
    if policy is None:
        return
    for arm in _state_arms(state):
        if arm.get("status") not in {"completed", "failed", "quarantined"}:
            continue
        if arm.get("replay_backup") is not None:
            continue
        _record_replay_restore(arm)
        _record_final_replay_backup(arm, policy=policy)


def _apply_arm_report(
    arm: dict[str, Any],
    report: Mapping[str, object],
    *,
    policy: Mapping[str, object],
) -> str:
    lifecycle = _lifecycle_view(report)
    _record_lifecycle(arm, lifecycle)
    if _is_budget_completion(lifecycle):
        arm["status"] = "completed"
        return "completed"
    if (
        lifecycle.get("status") == "retryable"
        and lifecycle.get("outcome") == TRANSIENT_CRASH
    ):
        arm["transient_failures"] = int(arm.get("transient_failures", 0)) + 1
        if str(lifecycle.get("stop_reason", "")).startswith("signal_"):
            arm["status"] = "pending"
            return "interrupted"
        retries = _nonnegative_integer(
            policy.get("max_transient_retries"),
            "maximum transient retries",
        )
        if int(arm["transient_failures"]) <= retries:
            arm["status"] = "pending"
            return "retry"
        arm["status"] = "failed"
        arm["failure"] = (
            f"transient retry budget exhausted after "
            f"{arm['transient_failures']} failure(s): {arm.get('failure')}"
        )
        return "failed"
    if lifecycle.get("failure_domain") in _ISOLATED_FAILURE_DOMAINS:
        _quarantine_arm(arm, lifecycle)
        return "quarantined"
    arm["status"] = "failed"
    return "failed"


def run_ablation_queue(
    manifest_path: Path,
    *,
    arm_runner: ArmRunner = run_elo_ablation,
    current_source_commit: str | None = None,
    source_tree_clean: bool | None = None,
    sleep: Callable[[float], None] = time.sleep,
    execution_lock_lease: ExecutionLockLease | None = None,
    expected_manifest_sha256: str | None = None,
) -> dict[str, object]:
    """Run pending arms exclusively and resume stale running state safely."""
    resolved_manifest, manifest = _load_deployment_manifest(
        manifest_path,
        expected_sha256=expected_manifest_sha256,
    )
    queue = _queue_config(manifest)
    policy = _mapping(queue.get("policy"), "queue policy")
    replay_backup_policy = _replay_backup_policy(queue)
    state_path = _state_path(manifest)
    execution_lock_path = (
        Path(_string(queue.get("execution_lock_path"), "execution lock path"))
        .expanduser()
        .resolve()
    )
    if execution_lock_lease is not None:
        execution_lock_lease.verify(execution_lock_path)
        execution_lock_context = nullcontext(execution_lock_lease)
    else:
        execution_lock_context = exclusive_execution_lock(execution_lock_path)
    with (
        execution_lock_context,
        exclusive_queue_lock(state_path),
    ):
        state = _load_or_create_state(
            state_path,
            manifest_path=resolved_manifest,
            manifest=manifest,
        )
        try:
            verify_deployment_manifest(
                resolved_manifest,
                current_source_commit=current_source_commit,
                source_tree_clean=source_tree_clean,
                expected_manifest_sha256=expected_manifest_sha256,
            )
            _reconcile_arms(state)
            _ensure_terminal_replay_backups(
                state,
                policy=replay_backup_policy,
            )
            failed = [
                arm for arm in _state_arms(state) if arm.get("status") == "failed"
            ]
            if failed and policy.get("continue_after_fatal") is not True:
                state["queue_status"] = "failed"
                state["queue_error"] = (
                    f"fatal arm {failed[0]['treatment']} blocks queue continuation"
                )
                _save_state(state_path, state)
                return state

            state["queue_status"] = "running"
            state["queue_error"] = None
            _save_state(state_path, state)
            while True:
                pending = next(
                    (
                        arm
                        for arm in _state_arms(state)
                        if arm.get("status") == "pending"
                    ),
                    None,
                )
                if pending is None:
                    break
                verify_deployment_manifest(
                    resolved_manifest,
                    current_source_commit=current_source_commit,
                    source_tree_clean=source_tree_clean,
                    expected_manifest_sha256=expected_manifest_sha256,
                )
                pending["status"] = "running"
                pending["attempts"] = int(pending.get("attempts", 0)) + 1
                pending["last_started_ns"] = time.time_ns()
                pending["failure"] = None
                _save_state(state_path, state)
                try:
                    report = arm_runner(
                        config_path=Path(str(pending["profile"])),
                        orchestrator=_string(
                            queue.get("orchestrator"),
                            "queue orchestrator",
                        ),
                        poll_seconds=float(queue["poll_seconds"]),
                    )
                except Exception as error:
                    pending["status"] = "failed"
                    pending["last_outcome"] = "runner_error"
                    pending["last_stopped_ns"] = time.time_ns()
                    pending["failure"] = f"{type(error).__name__}: {error}"
                    action = "failed"
                else:
                    action = _apply_arm_report(
                        pending,
                        report,
                        policy=policy,
                    )
                _record_replay_restore(pending)
                _record_final_replay_backup(
                    pending,
                    policy=replay_backup_policy,
                )
                _save_state(state_path, state)
                if action == "interrupted":
                    state["queue_status"] = "pending"
                    state["queue_error"] = "queue interrupted; arm remains pending"
                    _save_state(state_path, state)
                    return state
                if action == "retry":
                    delay = float(policy["retry_delay_seconds"])
                    if delay:
                        sleep(delay)
                    continue
                if (
                    action == "failed"
                    and policy.get("continue_after_fatal") is not True
                ):
                    state["queue_status"] = "failed"
                    state["queue_error"] = (
                        f"fatal arm {pending['treatment']} stopped the queue"
                    )
                    _save_state(state_path, state)
                    return state

            arms = _state_arms(state)
            state["queue_status"] = (
                "completed"
                if all(arm.get("status") == "completed" for arm in arms)
                else "failed"
            )
            state["queue_error"] = (
                None
                if state["queue_status"] == "completed"
                else "one or more queue arms failed"
            )
            _save_state(state_path, state)
            return state
        except Exception as error:
            state["queue_status"] = "failed"
            state["queue_error"] = f"{type(error).__name__}: {error}"
            _save_state(state_path, state)
            raise
        finally:
            try:
                _finalize_locked(
                    state_path=state_path,
                    state=state,
                    manifest=manifest,
                )
            except AblationQueueError:
                pass
            try:
                _request_continuity_handoff(
                    state_path=state_path,
                    state=state,
                    manifest=manifest,
                )
            except (OSError, TypeError, ValueError, AblationQueueError) as error:
                handoff = _mapping(
                    state.get("continuity_handoff"),
                    "continuity handoff",
                )
                handoff.update(
                    {
                        "status": "failed",
                        "error": f"{type(error).__name__}: {error}",
                    }
                )
                try:
                    _save_state(state_path, state)
                except OSError:
                    pass


def _error_document(error: Exception) -> dict[str, object]:
    return {
        "schema_version": SCHEMA_VERSION,
        "report": QUEUE_REPORT,
        "status": "error",
        "error": f"{type(error).__name__}: {error}",
    }


def main(argv: list[str] | None = None) -> int:
    arguments = _parser().parse_args(argv)
    try:
        if arguments.command == "manifest":
            report = generate_deployment_manifest(
                plan_path=arguments.plan,
                output_path=arguments.output,
                training_dir=arguments.training_dir,
                queue_unit=arguments.queue_unit,
                finalize_unit=arguments.finalize_unit,
                environment_file=arguments.environment_file,
                state_path=arguments.state,
                comparison_output=arguments.comparison_output,
                continuity_handoff_output=arguments.continuity_handoff_output,
                execution_lock_path=arguments.execution_lock,
                source_commit=arguments.source_commit,
                orchestrator=arguments.orchestrator,
                poll_seconds=arguments.poll_seconds,
                max_transient_retries=arguments.max_transient_retries,
                retry_delay_seconds=arguments.retry_delay_seconds,
                continue_after_fatal=arguments.continue_after_fatal,
                provisioned_gpus=arguments.provisioned_gpus,
                replay_backup_service_unit=arguments.replay_backup_service_unit,
                replay_backup_timer_unit=arguments.replay_backup_timer_unit,
                replay_backup_interval_seconds=(
                    arguments.replay_backup_interval_seconds
                ),
                replay_backup_retain=arguments.replay_backup_retain,
                replay_backup_max_total_bytes=(arguments.replay_backup_max_total_bytes),
            )
            status = 0
        elif arguments.command == "verify":
            report = verify_deployment_manifest(arguments.manifest)
            status = 0
        elif arguments.command == "finalize":
            report = finalize_ablation_queue(arguments.manifest)
            status = 0
        else:
            state = run_ablation_queue(arguments.manifest)
            report = state
            finalization = _mapping(state.get("finalization"), "queue finalization")
            if (
                state.get("queue_status") == "completed"
                and finalization.get("comparison_status") == "complete"
            ):
                status = 0
            elif state.get("queue_status") == "pending":
                status = 75
            else:
                status = 3
    except (
        AblationQueueError,
        FileExistsError,
        FileNotFoundError,
        OSError,
        TypeError,
        ValueError,
    ) as error:
        report = _error_document(error)
        status = 2
    print(json.dumps(report, sort_keys=True, allow_nan=False))
    return status


if __name__ == "__main__":
    raise SystemExit(main())
