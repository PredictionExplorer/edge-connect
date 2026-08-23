from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

from scripts.backup_active_queue_disaster_snapshot import (
    ActiveQueueBackupError,
    backup_active_arm,
)


def _write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")


def _fixture(tmp_path: Path) -> tuple[Path, Path, Path, Path, Path]:
    state = tmp_path / "queue.json"
    mapping = tmp_path / "backup-map.json"
    script = tmp_path / "snapshot.py"
    script.write_text("# fixture\n", encoding="utf-8")
    mount = tmp_path / "nfs"
    mount.mkdir()
    run_root = tmp_path / "run"
    run_root.mkdir()
    profile = run_root / "profile.yaml"
    profile.write_text("profile: fixture\n", encoding="utf-8")
    backup_root = mount / "backup"
    backup_root.mkdir()
    _write_json(
        mapping,
        {
            "schema_version": 1,
            "report": "startrain-active-queue-disaster-backup-map",
            "queue_state": str(state),
            "backup_mount": str(mount),
            "arms": {
                "control": {
                    "run_root": str(run_root),
                    "profile": str(profile),
                    "backup_root": str(backup_root),
                }
            },
        },
    )
    return state, mapping, script, mount, run_root


def test_missing_queue_state_is_a_no_op(tmp_path: Path) -> None:
    state, mapping, script, mount, _run_root = _fixture(tmp_path)

    report = backup_active_arm(
        state,
        mapping,
        snapshot_script=script,
        backup_mount=mount,
    )

    assert report["status"] == "no-op"
    assert report["reason"] == "queue_state_missing"


def test_snapshots_exactly_one_pinned_active_arm(tmp_path: Path) -> None:
    state, mapping, script, mount, run_root = _fixture(tmp_path)
    _write_json(
        state,
        {
            "queue_status": "running",
            "arms": [
                {
                    "treatment": "control",
                    "status": "running",
                    "run_root": str(run_root),
                }
            ],
        },
    )
    calls: list[list[str]] = []

    def runner(command, **kwargs):
        calls.append(command)
        assert kwargs == {"check": False, "capture_output": True, "text": True}
        return subprocess.CompletedProcess(
            command,
            0,
            stdout=json.dumps({"status": "ok", "snapshot": "fixture"}),
            stderr="",
        )

    report = backup_active_arm(
        state,
        mapping,
        snapshot_script=script,
        backup_mount=mount,
        runner=runner,
    )

    assert report["status"] == "ok"
    assert report["treatment"] == "control"
    assert calls == [
        [
            sys.executable,
            str(script),
            "snapshot",
            "--run-root",
            str(run_root),
            "--profile",
            str(run_root / "profile.yaml"),
            "--backup-root",
            str(mount / "backup"),
            "--expected-backup-mount",
            str(mount),
        ]
    ]


def test_active_run_root_must_match_pinned_mapping(tmp_path: Path) -> None:
    state, mapping, script, mount, _run_root = _fixture(tmp_path)
    _write_json(
        state,
        {
            "queue_status": "running",
            "arms": [
                {
                    "treatment": "control",
                    "status": "running",
                    "run_root": str(tmp_path / "other-run"),
                }
            ],
        },
    )

    with pytest.raises(ActiveQueueBackupError, match="run root differs"):
        backup_active_arm(
            state,
            mapping,
            snapshot_script=script,
            backup_mount=mount,
        )


def test_multiple_running_arms_fail_closed(tmp_path: Path) -> None:
    state, mapping, script, mount, run_root = _fixture(tmp_path)
    _write_json(
        state,
        {
            "queue_status": "running",
            "arms": [
                {
                    "treatment": "control",
                    "status": "running",
                    "run_root": str(run_root),
                },
                {
                    "treatment": "other",
                    "status": "running",
                    "run_root": str(tmp_path / "other-run"),
                },
            ],
        },
    )

    with pytest.raises(ActiveQueueBackupError, match="exactly one active arm"):
        backup_active_arm(
            state,
            mapping,
            snapshot_script=script,
            backup_mount=mount,
        )
