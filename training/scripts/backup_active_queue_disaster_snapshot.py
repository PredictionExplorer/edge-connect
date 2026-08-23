#!/usr/bin/env python3
"""Snapshot only the active arm in a durable Elo-ablation queue."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from collections.abc import Mapping
from pathlib import Path
from typing import Any

SCHEMA_VERSION = 1
REPORT_NAME = "startrain-active-queue-disaster-backup"


class ActiveQueueBackupError(RuntimeError):
    """The queue or pinned backup mapping is unsafe."""


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--queue-state", type=Path, required=True)
    parser.add_argument("--mapping", type=Path, required=True)
    parser.add_argument("--snapshot-script", type=Path, required=True)
    parser.add_argument("--backup-mount", type=Path, required=True)
    return parser


def _read_json(path: Path, *, name: str) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ActiveQueueBackupError(f"cannot read {name} {path}: {error}") from error
    if not isinstance(payload, dict):
        raise ActiveQueueBackupError(f"{name} must contain a JSON object")
    return payload


def _absolute_path(value: object, *, name: str) -> Path:
    if not isinstance(value, str) or not value or not Path(value).is_absolute():
        raise ActiveQueueBackupError(f"{name} must be an absolute path")
    return Path(value)


def _mapping_arms(mapping: Mapping[str, object]) -> Mapping[str, object]:
    if (
        mapping.get("schema_version") != SCHEMA_VERSION
        or mapping.get("report") != "startrain-active-queue-disaster-backup-map"
    ):
        raise ActiveQueueBackupError("backup mapping has an incompatible contract")
    arms = mapping.get("arms")
    if not isinstance(arms, Mapping) or not arms:
        raise ActiveQueueBackupError("backup mapping arms must be a non-empty object")
    return arms


def backup_active_arm(
    queue_state_path: str | Path,
    mapping_path: str | Path,
    *,
    snapshot_script: str | Path,
    backup_mount: str | Path,
    runner: Any = subprocess.run,
) -> dict[str, object]:
    state_path = Path(queue_state_path).expanduser().resolve()
    pinned_mapping_path = Path(mapping_path).expanduser().resolve()
    script = Path(snapshot_script).expanduser().resolve()
    mount = Path(backup_mount).expanduser().resolve()

    mapping = _read_json(pinned_mapping_path, name="backup mapping")
    arms = _mapping_arms(mapping)
    if mapping.get("queue_state") != str(state_path):
        raise ActiveQueueBackupError("backup mapping queue state does not match")
    if mapping.get("backup_mount") != str(mount):
        raise ActiveQueueBackupError("backup mapping mount does not match")
    if not script.is_file() or script.is_symlink():
        raise ActiveQueueBackupError("snapshot script is missing or unsafe")
    if not mount.is_dir() or mount.is_symlink():
        raise ActiveQueueBackupError("backup mount is missing or unsafe")

    if not state_path.is_file():
        return {
            "schema_version": SCHEMA_VERSION,
            "report": REPORT_NAME,
            "status": "no-op",
            "reason": "queue_state_missing",
        }
    state = _read_json(state_path, name="queue state")
    queue_status = state.get("queue_status")
    if queue_status != "running":
        return {
            "schema_version": SCHEMA_VERSION,
            "report": REPORT_NAME,
            "status": "no-op",
            "reason": f"queue_{queue_status or 'unknown'}",
        }
    raw_state_arms = state.get("arms")
    if not isinstance(raw_state_arms, list):
        raise ActiveQueueBackupError("queue state arms must be a list")
    running = [
        arm
        for arm in raw_state_arms
        if isinstance(arm, Mapping) and arm.get("status") == "running"
    ]
    if len(running) != 1:
        raise ActiveQueueBackupError(
            f"running queue must expose exactly one active arm, found {len(running)}"
        )
    state_arm = running[0]
    treatment = state_arm.get("treatment")
    if not isinstance(treatment, str) or treatment not in arms:
        raise ActiveQueueBackupError("active treatment is absent from backup mapping")
    configured = arms[treatment]
    if not isinstance(configured, Mapping):
        raise ActiveQueueBackupError("active treatment backup mapping is invalid")
    run_root = _absolute_path(configured.get("run_root"), name="run root")
    profile = _absolute_path(configured.get("profile"), name="profile")
    backup_root = _absolute_path(configured.get("backup_root"), name="backup root")
    if state_arm.get("run_root") != str(run_root):
        raise ActiveQueueBackupError("queue state run root differs from backup mapping")
    if not run_root.is_dir() or run_root.is_symlink():
        raise ActiveQueueBackupError("active run root is missing or unsafe")
    if not profile.is_file() or profile.is_symlink():
        raise ActiveQueueBackupError("active profile is missing or unsafe")
    if not backup_root.is_dir() or backup_root.is_symlink():
        raise ActiveQueueBackupError("active backup root is missing or unsafe")

    command = [
        sys.executable,
        str(script),
        "snapshot",
        "--run-root",
        str(run_root),
        "--profile",
        str(profile),
        "--backup-root",
        str(backup_root),
        "--expected-backup-mount",
        str(mount),
    ]
    completed = runner(
        command,
        check=False,
        capture_output=True,
        text=True,
    )
    if completed.returncode != 0:
        raise ActiveQueueBackupError(
            "active-arm snapshot failed with status "
            f"{completed.returncode}: {completed.stderr.strip() or completed.stdout.strip()}"
        )
    try:
        snapshot = json.loads(completed.stdout)
    except json.JSONDecodeError as error:
        raise ActiveQueueBackupError(
            f"active-arm snapshot returned invalid JSON: {error}"
        ) from error
    if not isinstance(snapshot, dict) or snapshot.get("status") != "ok":
        raise ActiveQueueBackupError("active-arm snapshot did not report success")
    return {
        "schema_version": SCHEMA_VERSION,
        "report": REPORT_NAME,
        "status": "ok",
        "treatment": treatment,
        "run_root": str(run_root),
        "backup_root": str(backup_root),
        "snapshot": snapshot,
    }


def main(argv: list[str] | None = None) -> int:
    arguments = _parser().parse_args(argv)
    try:
        report = backup_active_arm(
            arguments.queue_state,
            arguments.mapping,
            snapshot_script=arguments.snapshot_script,
            backup_mount=arguments.backup_mount,
        )
    except ActiveQueueBackupError as error:
        print(
            json.dumps(
                {
                    "schema_version": SCHEMA_VERSION,
                    "report": REPORT_NAME,
                    "status": "error",
                    "error": str(error),
                },
                sort_keys=True,
            )
        )
        return 2
    print(json.dumps(report, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
