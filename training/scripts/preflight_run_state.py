#!/usr/bin/env python3
"""Validate and safely migrate durable training state before GPU startup."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import sqlite3
import sys
import time
from collections.abc import Mapping
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Iterator

from startrain.checkpoint import (
    ModelManifest,
    inspect_checkpoint,
    load_model_manifest,
    load_recovery_pointer,
    load_resume_cutover,
    write_model_pointer,
)
from startrain.config import ExperimentConfig, load_config
from startrain.contracts import FEATURE_SCHEMA_HASH, RULES_HASH_WIRE
from startrain.replay_store import (
    MANIFEST_SCHEMA_VERSION,
    prove_legacy_committed_sample_history,
)
from startrain.runtime import atomic_json, load_run_identity

PREFLIGHT_SCHEMA_VERSION = 1
UTD_SEGMENT_SCHEMA_VERSION = 1


class StatePreflightError(RuntimeError):
    """A fail-closed run-state validation error."""


def _read_json(path: Path, name: str) -> dict[str, Any]:
    if path.is_symlink():
        raise StatePreflightError(f"{name} may not be a symbolic link: {path}")
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise StatePreflightError(f"cannot read {name} {path}: {exc}") from exc
    if not isinstance(payload, dict):
        raise StatePreflightError(f"{name} must be a JSON object")
    return payload


def _nonnegative_int(name: str, value: object) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise StatePreflightError(f"{name} must be a non-negative integer")
    return value


def _positive_int(name: str, value: object) -> int:
    value = _nonnegative_int(name, value)
    if value == 0:
        raise StatePreflightError(f"{name} must be positive")
    return value


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _live_pid(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


def _validate_inactive(run_root: Path) -> str:
    lock = run_root / "coordinator.lock"
    if not lock.exists():
        return "absent"
    payload = _read_json(lock, "coordinator lock")
    pid = _positive_int("coordinator lock pid", payload.get("pid"))
    if _live_pid(pid):
        raise StatePreflightError(f"coordinator lock PID {pid} is live")
    return f"stale-dead-pid:{pid}"


@contextmanager
def state_apply_guard(run_root: Path) -> Iterator[None]:
    """Exclude the coordinator while validated state is being migrated."""

    path = run_root / "coordinator.lock"
    token = f"state-preflight-{os.getpid()}-{time.time_ns()}"
    payload = (
        json.dumps(
            {
                "pid": os.getpid(),
                "created_ns": time.time_ns(),
                "owner": "state-preflight",
                "token": token,
            },
            sort_keys=True,
            separators=(",", ":"),
        )
        + "\n"
    ).encode()
    try:
        descriptor = os.open(
            path,
            os.O_CREAT | os.O_EXCL | os.O_WRONLY | getattr(os, "O_NOFOLLOW", 0),
            0o644,
        )
    except FileExistsError as exc:
        raise StatePreflightError(
            "coordinator lock appeared before migration apply"
        ) from exc
    try:
        os.write(descriptor, payload)
        os.fsync(descriptor)
    finally:
        os.close(descriptor)
    directory = os.open(run_root, os.O_RDONLY)
    try:
        os.fsync(directory)
    finally:
        os.close(directory)
    try:
        yield
    finally:
        try:
            current = _read_json(path, "state preflight lock")
        except StatePreflightError:
            current = {}
        if current.get("token") == token and current.get("pid") == os.getpid():
            path.unlink(missing_ok=True)
            directory = os.open(run_root, os.O_RDONLY)
            try:
                os.fsync(directory)
            finally:
                os.close(directory)


def _validate_profile(
    run_root: Path,
    profile_path: Path,
    *,
    run_id: str,
) -> tuple[ExperimentConfig, dict[str, object]]:
    if profile_path.is_symlink() or not profile_path.is_file():
        raise StatePreflightError(f"profile is missing or unsafe: {profile_path}")
    try:
        experiment = load_config(profile_path)
    except (OSError, ValueError) as exc:
        raise StatePreflightError(f"profile is invalid: {exc}") from exc
    configured_root = (
        Path(experiment.orchestration.directories.root).expanduser().resolve()
    )
    if configured_root != run_root:
        raise StatePreflightError("profile run root does not match requested run root")
    if experiment.orchestration.run_id != run_id:
        raise StatePreflightError("profile run_id does not match run.json")
    digest = _sha256(profile_path)
    checksums = []
    for checksum_path in (
        profile_path.with_suffix(".sha256"),
        run_root / "profile.sha256",
    ):
        if not checksum_path.is_file():
            continue
        try:
            recorded = checksum_path.read_text(encoding="utf-8").strip().split()[0]
        except (OSError, IndexError, UnicodeDecodeError) as exc:
            raise StatePreflightError(
                f"cannot read profile checksum {checksum_path}: {exc}"
            ) from exc
        if recorded != digest:
            raise StatePreflightError(
                f"profile checksum does not match {profile_path.name}"
            )
        checksums.append(str(checksum_path))
    return experiment, {
        "path": str(profile_path),
        "sha256": digest,
        "validated_checksums": checksums,
    }


def _open_replay(path: Path, *, writable: bool) -> sqlite3.Connection:
    if path.is_symlink() or not path.is_file():
        raise StatePreflightError(f"replay manifest is missing or unsafe: {path}")
    uri = f"{path.resolve().as_uri()}?mode={'rw' if writable else 'ro'}"
    try:
        connection = sqlite3.connect(uri, uri=True, timeout=30.0)
    except sqlite3.Error as exc:
        raise StatePreflightError(f"cannot open replay manifest: {exc}") from exc
    connection.row_factory = sqlite3.Row
    return connection


def _validate_replay(
    run_root: Path,
    *,
    run_id: str,
    generation_family: str,
    created_ns: int,
) -> tuple[dict[str, object], bool]:
    path = run_root / "replay" / "manifest.sqlite3"
    try:
        with _open_replay(path, writable=False) as connection:
            integrity = [
                str(row[0]) for row in connection.execute("PRAGMA quick_check")
            ]
            if integrity != ["ok"]:
                raise StatePreflightError(
                    "replay manifest integrity failed: " + "; ".join(integrity)
                )
            metadata = {
                str(row["key"]): str(row["value"])
                for row in connection.execute("SELECT key, value FROM store_metadata")
            }
            expected = {
                "manifest_schema_version": str(MANIFEST_SCHEMA_VERSION),
                "rules_hash": RULES_HASH_WIRE,
                "feature_schema_hash": f"{FEATURE_SCHEMA_HASH:016x}",
            }
            if any(metadata.get(key) != value for key, value in expected.items()):
                raise StatePreflightError("replay manifest metadata is incompatible")
            run = connection.execute(
                """
                SELECT generation_family, created_ns
                FROM runs WHERE run_id = ?
                """,
                (run_id,),
            ).fetchone()
            if (
                run is None
                or str(run["generation_family"]) != generation_family
                or int(run["created_ns"]) != created_ns
            ):
                raise StatePreflightError("replay run registration is incompatible")
            columns = {
                str(row["name"])
                for row in connection.execute("PRAGMA table_info(run_counters)")
            }
            counter_has_history = "history_complete" in columns
            counter = (
                connection.execute(
                    """
                    SELECT committed_samples, history_complete
                    FROM run_counters
                    WHERE run_id = ? AND generation_family = ?
                    """,
                    (run_id, generation_family),
                ).fetchone()
                if counter_has_history
                else None
            )
            legacy_counter = (
                connection.execute(
                    """
                    SELECT committed_samples
                    FROM run_counters
                    WHERE run_id = ? AND generation_family = ?
                    """,
                    (run_id, generation_family),
                ).fetchone()
                if "committed_samples" in columns and not counter_has_history
                else None
            )
            proof = prove_legacy_committed_sample_history(
                connection,
                run_id=run_id,
                generation_family=generation_family,
            )
            ready_samples_by_ring = {
                str(int(row["ring"])): int(row["samples"])
                for row in connection.execute(
                    """
                    SELECT ring, COALESCE(SUM(sample_count), 0) AS samples
                    FROM shards
                    WHERE run_id = ? AND generation_family = ? AND state = 'ready'
                    GROUP BY ring
                    ORDER BY ring
                    """,
                    (run_id, generation_family),
                )
            }
    except sqlite3.Error as exc:
        raise StatePreflightError(f"cannot validate replay manifest: {exc}") from exc
    committed = _nonnegative_int(
        "committed replay samples",
        (
            counter["committed_samples"]
            if counter is not None
            else (
                legacy_counter["committed_samples"]
                if legacy_counter is not None
                else proof.shard_samples
            )
        ),
    )
    history_complete = bool(
        counter is not None and int(counter["history_complete"]) == 1
    )
    reconciliable = (
        not history_complete and proof.complete and committed == proof.shard_samples
    )
    initialized = run_root / "replay" / "initialized.json"
    if initialized.is_file():
        payload = _read_json(initialized, "replay initialization")
        if (
            payload.get("schema_version") != 1
            or payload.get("run_id") != run_id
            or payload.get("generation_family") != generation_family
        ):
            raise StatePreflightError("replay initialization identity is incompatible")
    return (
        {
            "manifest": str(path),
            "committed_samples": committed,
            "ready_samples_by_ring": ready_samples_by_ring,
            "history_complete": history_complete,
            "history_reconciliable": reconciliable,
            "counter_schema_current": counter_has_history,
            "counter_row_present": counter is not None or legacy_counter is not None,
            "proof": {
                "complete": proof.complete,
                "failures": list(proof.failures),
                "shard_count": proof.shard_count,
                "shard_samples": proof.shard_samples,
                "expected_games": proof.expected_games,
                "recorded_games": proof.recorded_games,
                "maximum_shard_id": proof.maximum_shard_id,
                "sqlite_sequence": proof.sqlite_sequence,
            },
        },
        reconciliable,
    )


def _validate_recovery(
    run_root: Path,
    experiment: ExperimentConfig,
    *,
    run_id: str,
    generation_family: str,
) -> tuple[dict[str, object], dict[str, Any]]:
    path = run_root / "learner" / "recovery.json"
    raw = _read_json(path, "recovery pointer")
    try:
        recovery = load_recovery_pointer(
            path,
            expected_run_id=run_id,
            expected_generation_family=generation_family,
        )
        metadata = inspect_checkpoint(
            recovery.checkpoint,
            expected_model_config=experiment.as_dict()["model"],
            expected_game_config=experiment.as_dict()["game"],
            expected_run_id=run_id,
            expected_generation_family=generation_family,
            expected_sha256=recovery.checkpoint_sha256,
            expected_bytes=recovery.checkpoint_bytes,
        )
    except (OSError, RuntimeError, ValueError) as exc:
        raise StatePreflightError(
            f"recovery checkpoint is incompatible: {exc}"
        ) from exc
    examples = _nonnegative_int(
        "recovery examples_consumed", raw.get("examples_consumed")
    )
    recovery_updated_ns = _positive_int("recovery updated_ns", raw.get("updated_ns"))
    extra = metadata.get("extra")
    if (
        metadata["step"] != recovery.step
        or metadata["epoch"] != recovery.epoch
        or not isinstance(extra, Mapping)
        or extra.get("examples_consumed") != examples
        or not all(
            metadata.get(name) is True
            for name in ("has_optimizer", "has_scheduler", "has_ema")
        )
    ):
        raise StatePreflightError(
            "recovery pointer and checkpoint training state disagree"
        )
    active = recovery
    active_metadata = metadata
    active_examples = examples
    active_source = "recovery"
    cutover_report: dict[str, object] | None = None
    cutover_path = run_root / "learner" / "resume-cutover.json"
    if cutover_path.is_file():
        cutover_raw = _read_json(cutover_path, "resume cutover")
        try:
            cutover = load_resume_cutover(
                cutover_path,
                expected_run_id=run_id,
                expected_generation_family=generation_family,
            )
            cutover_metadata = inspect_checkpoint(
                cutover.checkpoint,
                expected_model_config=experiment.as_dict()["model"],
                expected_game_config=experiment.as_dict()["game"],
                expected_run_id=run_id,
                expected_generation_family=generation_family,
                expected_sha256=cutover.checkpoint_sha256,
                expected_bytes=cutover.checkpoint_bytes,
            )
        except (OSError, RuntimeError, ValueError) as exc:
            raise StatePreflightError(f"resume cutover is incompatible: {exc}") from exc
        cutover_created_ns = _positive_int(
            "resume cutover created_ns", cutover_raw.get("created_ns")
        )
        cutover_extra = cutover_metadata.get("extra")
        cutover_examples = (
            cutover_extra.get("examples_consumed")
            if isinstance(cutover_extra, Mapping)
            else None
        )
        cutover_examples = _nonnegative_int(
            "resume cutover examples_consumed", cutover_examples
        )
        if not all(
            cutover_metadata.get(name) is True
            for name in ("has_optimizer", "has_scheduler", "has_ema")
        ):
            raise StatePreflightError("resume cutover training state is incomplete")
        cutover_report = {
            "path": str(cutover_path),
            "checkpoint": str(cutover.checkpoint),
            "checkpoint_sha256": cutover.checkpoint_sha256,
            "step": cutover.step,
            "examples_consumed": cutover_examples,
            "created_ns": cutover_created_ns,
        }
        if (
            cutover.checkpoint_sha256 != recovery.checkpoint_sha256
            and recovery_updated_ns < cutover_created_ns
        ):
            active = cutover
            active_metadata = cutover_metadata
            active_examples = cutover_examples
            active_source = "cutover"
    return (
        {
            "path": str(path),
            "pointer_checkpoint": str(recovery.checkpoint),
            "pointer_checkpoint_sha256": recovery.checkpoint_sha256,
            "pointer_step": recovery.step,
            "pointer_epoch": recovery.epoch,
            "pointer_examples_consumed": examples,
            "pointer_updated_ns": recovery_updated_ns,
            "checkpoint": str(active.checkpoint),
            "checkpoint_sha256": active.checkpoint_sha256,
            "step": active.step,
            "epoch": int(active_metadata["epoch"]),
            "examples_consumed": active_examples,
            "source": active_source,
            "resume_cutover": cutover_report,
        },
        active_metadata,
    )


def _parse_utd_segment(
    payload: object,
    *,
    run_id: str,
    generation_family: str,
    target: float,
    maximum_examples: int,
    maximum_samples: int,
) -> dict[str, object]:
    if not isinstance(payload, dict):
        raise StatePreflightError("UTD segment must be a JSON object")
    required = {
        "schema_version",
        "run_id",
        "generation_family",
        "target_updates_per_new_sample",
        "baseline_examples_consumed",
        "baseline_committed_replay_samples",
    }
    if not required <= set(payload) or set(payload) - (required | {"created_ns"}):
        raise StatePreflightError("UTD segment fields are incompatible")
    raw_target = payload.get("target_updates_per_new_sample")
    if (
        payload.get("schema_version") != UTD_SEGMENT_SCHEMA_VERSION
        or payload.get("run_id") != run_id
        or payload.get("generation_family") != generation_family
        or isinstance(raw_target, bool)
        or not isinstance(raw_target, int | float)
        or not math.isfinite(float(raw_target))
        or float(raw_target) != target
    ):
        raise StatePreflightError("UTD segment identity or target is incompatible")
    examples = _nonnegative_int(
        "UTD baseline examples", payload.get("baseline_examples_consumed")
    )
    samples = _nonnegative_int(
        "UTD baseline committed samples",
        payload.get("baseline_committed_replay_samples"),
    )
    if examples > maximum_examples:
        raise StatePreflightError("learner examples precede the UTD segment baseline")
    if samples > maximum_samples:
        raise StatePreflightError("committed replay precedes the UTD segment baseline")
    normalized = dict(payload)
    normalized["target_updates_per_new_sample"] = target
    return normalized


def _candidate_manifest(
    run_root: Path,
    *,
    run_id: str,
    generation_family: str,
) -> ModelManifest:
    path = run_root / "learner" / "candidate.json"
    if not path.is_file():
        raise StatePreflightError("learner candidate pointer is missing")
    try:
        manifest = load_model_manifest(path)
    except (OSError, ValueError) as exc:
        raise StatePreflightError(f"candidate pointer is invalid: {exc}") from exc
    if manifest.run_id != run_id or manifest.generation_family != generation_family:
        raise StatePreflightError("candidate pointer belongs to another run")
    return manifest


def _manifest_examples(
    manifest: ModelManifest,
    *,
    name: str,
    run_id: str,
    generation_family: str,
) -> int | None:
    try:
        metadata = inspect_checkpoint(
            manifest.checkpoint,
            expected_run_id=run_id,
            expected_generation_family=generation_family,
            expected_sha256=manifest.checkpoint_sha256,
            expected_bytes=manifest.checkpoint_bytes,
        )
    except (OSError, RuntimeError, ValueError) as exc:
        raise StatePreflightError(f"{name} checkpoint is incompatible: {exc}") from exc
    extra = metadata.get("extra")
    if not isinstance(extra, Mapping) or extra.get("examples_consumed") is None:
        return None
    return _nonnegative_int(
        f"{name} examples_consumed",
        extra.get("examples_consumed"),
    )


def _apply_history_reconciliation(
    run_root: Path,
    *,
    run_id: str,
    generation_family: str,
) -> None:
    path = run_root / "replay" / "manifest.sqlite3"
    try:
        with _open_replay(path, writable=True) as connection:
            connection.execute("BEGIN IMMEDIATE")
            columns = {
                str(row["name"])
                for row in connection.execute("PRAGMA table_info(run_counters)")
            }
            if not columns:
                connection.execute(
                    """
                    CREATE TABLE run_counters (
                        run_id TEXT NOT NULL,
                        generation_family TEXT NOT NULL,
                        committed_samples INTEGER NOT NULL
                            CHECK(committed_samples >= 0),
                        updated_ns INTEGER NOT NULL,
                        history_complete INTEGER NOT NULL
                            CHECK(history_complete IN (0, 1)),
                        PRIMARY KEY(run_id, generation_family)
                    )
                    """
                )
            elif "history_complete" not in columns:
                connection.execute(
                    """
                    ALTER TABLE run_counters
                    ADD COLUMN history_complete INTEGER NOT NULL DEFAULT 0
                    CHECK(history_complete IN (0, 1))
                    """
                )
            proof = prove_legacy_committed_sample_history(
                connection,
                run_id=run_id,
                generation_family=generation_family,
            )
            row = connection.execute(
                """
                SELECT committed_samples, history_complete
                FROM run_counters
                WHERE run_id = ? AND generation_family = ?
                """,
                (run_id, generation_family),
            ).fetchone()
            if row is None:
                connection.execute(
                    """
                    INSERT INTO run_counters(
                        run_id, generation_family, committed_samples, updated_ns,
                        history_complete
                    ) VALUES (?, ?, ?, ?, 0)
                    """,
                    (
                        run_id,
                        generation_family,
                        proof.shard_samples,
                        time.time_ns(),
                    ),
                )
                row = connection.execute(
                    """
                    SELECT committed_samples, history_complete
                    FROM run_counters
                    WHERE run_id = ? AND generation_family = ?
                    """,
                    (run_id, generation_family),
                ).fetchone()
            if (
                row is None
                or not proof.complete
                or int(row["committed_samples"]) != proof.shard_samples
            ):
                raise StatePreflightError(
                    "legacy committed-sample history changed before apply"
                )
            connection.execute(
                """
                UPDATE run_counters
                SET history_complete = 1, updated_ns = ?
                WHERE run_id = ? AND generation_family = ?
                """,
                (time.time_ns(), run_id, generation_family),
            )
            connection.execute("COMMIT")
    except sqlite3.Error as exc:
        raise StatePreflightError(
            f"cannot reconcile committed-sample history: {exc}"
        ) from exc


def _plan_cadence(
    run_root: Path,
    experiment: ExperimentConfig,
    *,
    run_id: str,
    generation_family: str,
    examples_consumed: int,
) -> tuple[dict[str, object], ModelManifest, bool]:
    candidate = _candidate_manifest(
        run_root,
        run_id=run_id,
        generation_family=generation_family,
    )
    durable_examples = [examples_consumed]
    candidate_checkpoint_examples = _manifest_examples(
        candidate,
        name="candidate",
        run_id=run_id,
        generation_family=generation_family,
    )
    if candidate_checkpoint_examples is not None:
        durable_examples.append(candidate_checkpoint_examples)
    path = run_root / "learner" / "cadence.json"
    selfplay_enabled = (
        experiment.learner.selfplay_snapshot_interval_examples is not None
    )
    changed = False
    raw: dict[str, Any] = {}
    if path.is_file():
        raw = _read_json(path, "learner cadence")
        if (
            raw.get("schema_version") != 1
            or raw.get("run_id") != run_id
            or raw.get("generation_family") != generation_family
        ):
            raise StatePreflightError("learner cadence identity is incompatible")
        candidate_examples = _nonnegative_int(
            "cadence candidate_examples", raw.get("candidate_examples")
        )
        selfplay_value = raw.get("selfplay_examples")
        selfplay_examples = (
            None
            if selfplay_value is None
            else _nonnegative_int("cadence selfplay_examples", selfplay_value)
        )
    else:
        candidate_examples = examples_consumed
        selfplay_examples = examples_consumed if selfplay_enabled else None
        changed = True
    if selfplay_enabled and selfplay_examples is None:
        selfplay_examples = candidate_examples
        changed = True
    selfplay_pointer = run_root / "learner" / "selfplay" / "candidate.json"
    pointer_changed = False
    if selfplay_enabled:
        if selfplay_pointer.is_file():
            try:
                existing = load_model_manifest(selfplay_pointer)
            except (OSError, ValueError) as exc:
                raise StatePreflightError(
                    f"self-play pointer is invalid: {exc}"
                ) from exc
            if (
                existing.run_id != run_id
                or existing.generation_family != generation_family
            ):
                raise StatePreflightError("self-play pointer belongs to another run")
            selfplay_checkpoint_examples = _manifest_examples(
                existing,
                name="self-play candidate",
                run_id=run_id,
                generation_family=generation_family,
            )
            if selfplay_checkpoint_examples is not None:
                durable_examples.append(selfplay_checkpoint_examples)
        else:
            pointer_changed = True
    durable_example_limit = max(durable_examples)
    if candidate_examples > durable_example_limit or (
        selfplay_examples is not None and selfplay_examples > durable_example_limit
    ):
        raise StatePreflightError(
            "cadence counters are ahead of durable checkpoint examples"
        )
    payload: dict[str, object] = {
        "schema_version": 1,
        "run_id": run_id,
        "generation_family": generation_family,
        "candidate_examples": candidate_examples,
        "selfplay_examples": selfplay_examples,
        "updated_ns": (
            _positive_int("cadence updated_ns", raw.get("updated_ns"))
            if path.is_file() and raw.get("updated_ns") is not None
            else 1
        ),
    }
    return payload, candidate, changed or pointer_changed


def run_state_preflight(
    run_root: str | Path,
    profile: str | Path,
    *,
    apply: bool = False,
) -> dict[str, object]:
    """Validate one stopped run and optionally apply proof-backed migrations."""

    root = Path(run_root).expanduser().resolve()
    profile_path = Path(profile).expanduser().resolve()
    if not root.is_dir() or root.is_symlink():
        raise StatePreflightError(f"run root is missing or unsafe: {root}")
    lock_status = _validate_inactive(root)
    try:
        identity = load_run_identity(root / "run.json")
    except ValueError as exc:
        raise StatePreflightError(f"run identity is invalid: {exc}") from exc
    experiment, profile_report = _validate_profile(
        root, profile_path, run_id=identity.run_id
    )
    replay_report, reconcile_history = _validate_replay(
        root,
        run_id=identity.run_id,
        generation_family=identity.generation_family,
        created_ns=identity.created_ns,
    )
    recovery_report, recovery_metadata = _validate_recovery(
        root,
        experiment,
        run_id=identity.run_id,
        generation_family=identity.generation_family,
    )
    examples = _nonnegative_int(
        "recovery examples_consumed", recovery_report.get("examples_consumed")
    )
    committed = _nonnegative_int(
        "committed replay samples", replay_report.get("committed_samples")
    )
    effective_history_complete = bool(
        replay_report["history_complete"] or reconcile_history
    )
    cadence, candidate, cadence_changed = _plan_cadence(
        root,
        experiment,
        run_id=identity.run_id,
        generation_family=identity.generation_family,
        examples_consumed=examples,
    )

    migrations: list[str] = []
    if reconcile_history:
        migrations.append("reconcile_legacy_committed_sample_history")
    if cadence_changed:
        migrations.append("initialize_selfplay_cadence")

    target_value = experiment.learner.target_updates_per_new_sample
    utd_path = root / "learner" / "utd-segment.json"
    utd_segment: dict[str, object] | None = None
    if target_value is not None:
        target = float(target_value)
        if not effective_history_complete:
            failures = replay_report["proof"]
            raise StatePreflightError(
                f"UTD requires complete committed-sample history; proof={failures}"
            )
        checkpoint_extra = recovery_metadata.get("extra")
        checkpoint_segment = (
            checkpoint_extra.get("utd_segment")
            if isinstance(checkpoint_extra, Mapping)
            else None
        )
        checkpoint_config = recovery_metadata.get("config")
        checkpoint_learner = (
            checkpoint_config.get("learner")
            if isinstance(checkpoint_config, Mapping)
            else None
        )
        checkpoint_target = (
            checkpoint_learner.get("target_updates_per_new_sample")
            if isinstance(checkpoint_learner, Mapping)
            else None
        )
        checkpoint_uses_active_target = (
            isinstance(checkpoint_target, int | float)
            and not isinstance(checkpoint_target, bool)
            and float(checkpoint_target) == target
        )
        if utd_path.is_file():
            utd_segment = _parse_utd_segment(
                _read_json(utd_path, "UTD segment"),
                run_id=identity.run_id,
                generation_family=identity.generation_family,
                target=target,
                maximum_examples=examples,
                maximum_samples=committed,
            )
        elif checkpoint_segment is not None and checkpoint_uses_active_target:
            utd_segment = _parse_utd_segment(
                checkpoint_segment,
                run_id=identity.run_id,
                generation_family=identity.generation_family,
                target=target,
                maximum_examples=examples,
                maximum_samples=committed,
            )
            migrations.append("restore_utd_segment_from_checkpoint")
        else:
            utd_segment = {
                "schema_version": UTD_SEGMENT_SCHEMA_VERSION,
                "run_id": identity.run_id,
                "generation_family": identity.generation_family,
                "target_updates_per_new_sample": target,
                "baseline_examples_consumed": examples,
                "baseline_committed_replay_samples": committed,
            }
            migrations.append("initialize_prospective_utd_segment")
        if checkpoint_segment is not None and checkpoint_uses_active_target:
            parsed_checkpoint_segment = _parse_utd_segment(
                checkpoint_segment,
                run_id=identity.run_id,
                generation_family=identity.generation_family,
                target=target,
                maximum_examples=examples,
                maximum_samples=committed,
            )
            if {
                key: parsed_checkpoint_segment[key]
                for key in (
                    "schema_version",
                    "run_id",
                    "generation_family",
                    "target_updates_per_new_sample",
                    "baseline_examples_consumed",
                    "baseline_committed_replay_samples",
                )
            } != {
                key: utd_segment[key]
                for key in (
                    "schema_version",
                    "run_id",
                    "generation_family",
                    "target_updates_per_new_sample",
                    "baseline_examples_consumed",
                    "baseline_committed_replay_samples",
                )
            }:
                rebase_path = root / "learner" / "state-rebase.json"
                if not rebase_path.is_file():
                    raise StatePreflightError(
                        "persisted UTD segment disagrees with recovery checkpoint"
                    )
                rebase = _read_json(rebase_path, "learner state rebase")
                if (
                    rebase.get("run_id") != identity.run_id
                    or rebase.get("generation_family") != identity.generation_family
                    or rebase.get("to_step") != recovery_report["step"]
                    or rebase.get("to_examples_consumed") != examples
                    or rebase.get("utd_segment") != utd_segment
                ):
                    raise StatePreflightError(
                        "learner state rebase does not authorize UTD divergence"
                    )

    if apply and migrations:
        with state_apply_guard(root):
            if reconcile_history:
                _apply_history_reconciliation(
                    root,
                    run_id=identity.run_id,
                    generation_family=identity.generation_family,
                )
            if utd_segment is not None and (
                "restore_utd_segment_from_checkpoint" in migrations
                or "initialize_prospective_utd_segment" in migrations
            ):
                persisted = dict(utd_segment)
                persisted.setdefault("created_ns", time.time_ns())
                atomic_json(utd_path, persisted)
                utd_segment = persisted
            selfplay_pointer = root / "learner" / "selfplay" / "candidate.json"
            if (
                experiment.learner.selfplay_snapshot_interval_examples is not None
                and not selfplay_pointer.is_file()
            ):
                write_model_pointer(selfplay_pointer, candidate, role="candidate")
            if cadence_changed:
                persisted_cadence = dict(cadence)
                persisted_cadence["updated_ns"] = time.time_ns()
                atomic_json(root / "learner" / "cadence.json", persisted_cadence)
                cadence = persisted_cadence

    return {
        "schema_version": PREFLIGHT_SCHEMA_VERSION,
        "status": "ok",
        "mode": "apply" if apply else "dry-run",
        "run_root": str(root),
        "run_id": identity.run_id,
        "generation_family": identity.generation_family,
        "coordinator_lock": lock_status,
        "profile": profile_report,
        "recovery": recovery_report,
        "cadence": cadence,
        "utd_segment": utd_segment,
        "replay": replay_report,
        "migrations": [
            {"name": name, "status": "applied" if apply else "planned"}
            for name in migrations
        ],
    }


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-root", required=True, type=Path)
    parser.add_argument("--profile", required=True, type=Path)
    parser.add_argument(
        "--if-present",
        action="store_true",
        help="report a skipped first-launch preflight when run.json is absent",
    )
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--dry-run", action="store_true")
    mode.add_argument("--apply", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    mode = "dry-run"
    try:
        arguments = _parser().parse_args(argv)
        mode = "apply" if arguments.apply else "dry-run"
        run_root = arguments.run_root.expanduser().resolve()
        profile = arguments.profile.expanduser().resolve()
        if arguments.if_present and not (run_root / "run.json").exists():
            report = {
                "schema_version": PREFLIGHT_SCHEMA_VERSION,
                "status": "skipped",
                "mode": mode,
                "reason": "first_launch",
                "run_root": str(run_root),
            }
        else:
            report = run_state_preflight(
                run_root,
                profile,
                apply=arguments.apply,
            )
    except Exception as exc:
        print(
            json.dumps(
                {
                    "schema_version": PREFLIGHT_SCHEMA_VERSION,
                    "status": "error",
                    "mode": mode,
                    "error_type": type(exc).__name__,
                    "error": str(exc),
                },
                sort_keys=True,
            ),
            file=sys.stderr,
        )
        return 2
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
