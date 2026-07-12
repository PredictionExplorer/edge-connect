from __future__ import annotations

import os
import socket
from pathlib import Path

from startrain.runtime import SystemdNotifier


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
