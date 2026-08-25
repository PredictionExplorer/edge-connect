from __future__ import annotations

import json
import os
import socket
from pathlib import Path

import pytest

from startrain.runtime import (
    CHAMPION_WARM_START_FORMAT,
    CHAMPION_WARM_START_SCHEMA_VERSION,
    CUTOVER_STAGING_FORMAT,
    CUTOVER_STAGING_SCHEMA_VERSION,
    SystemdNotifier,
    require_launch_ready,
)


def test_systemd_notifier_emits_ready_watchdog_and_stopping() -> None:
    socket_path = Path("/tmp") / f"startrain-notify-{os.getpid()}.sock"
    try:
        with socket.socket(socket.AF_UNIX, socket.SOCK_DGRAM) as server:
            server.bind(str(socket_path))
            server.settimeout(1.0)
            notifier = SystemdNotifier(str(socket_path))
            notifier.ready("ready")
            assert server.recv(1024) == b"READY=1\nSTATUS=ready"
            notifier.watchdog("healthy")
            assert server.recv(1024) == b"WATCHDOG=1\nSTATUS=healthy"
            notifier.stopping("stopping")
            assert server.recv(1024) == b"STOPPING=1\nSTATUS=stopping"
    finally:
        socket_path.unlink(missing_ok=True)


def test_systemd_notifier_is_noop_without_socket() -> None:
    notifier = SystemdNotifier("")
    assert notifier.enabled is False
    notifier.ready("ignored")
    notifier.watchdog("ignored")
    notifier.stopping("ignored")


def test_launch_readiness_rejects_prepared_warm_start(tmp_path: Path) -> None:
    learner = tmp_path / "learner"
    learner.mkdir()
    (learner / "champion-warm-start.json").write_text(
        json.dumps(
            {
                "format": CHAMPION_WARM_START_FORMAT,
                "schema_version": CHAMPION_WARM_START_SCHEMA_VERSION,
                "status": "prepared",
            }
        ),
        encoding="utf-8",
    )

    with pytest.raises(RuntimeError, match="prepared but has not been activated"):
        require_launch_ready(learner)


def test_launch_readiness_rejects_pending_cutover_staging(tmp_path: Path) -> None:
    learner = tmp_path / "learner"
    learner.mkdir()
    (learner / "cutover-staging.json").write_text(
        json.dumps(
            {
                "format": CUTOVER_STAGING_FORMAT,
                "schema_version": CUTOVER_STAGING_SCHEMA_VERSION,
                "status": "pending",
            }
        ),
        encoding="utf-8",
    )

    with pytest.raises(RuntimeError, match="pending activation"):
        require_launch_ready(learner)


def test_launch_readiness_accepts_active_warm_start_and_staging(
    tmp_path: Path,
) -> None:
    learner = tmp_path / "learner"
    learner.mkdir()
    (learner / "champion-warm-start.json").write_text(
        json.dumps(
            {
                "format": CHAMPION_WARM_START_FORMAT,
                "schema_version": CHAMPION_WARM_START_SCHEMA_VERSION,
                "status": "active",
                "checkpoint_sha256": "a" * 64,
                "checkpoint_bytes": 1,
                "absolute_model_step": 0,
            }
        ),
        encoding="utf-8",
    )
    (learner / "cutover-staging.json").write_text(
        json.dumps(
            {
                "format": CUTOVER_STAGING_FORMAT,
                "schema_version": CUTOVER_STAGING_SCHEMA_VERSION,
                "status": "active",
                "checkpoint_sha256": "a" * 64,
                "activated_ns": 1,
            }
        ),
        encoding="utf-8",
    )

    require_launch_ready(learner)
