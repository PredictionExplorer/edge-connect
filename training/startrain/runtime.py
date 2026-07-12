"""Small process-runtime primitives with no CUDA or PyTorch imports."""

from __future__ import annotations

import json
import os
import re
import signal
import socket
import tempfile
import threading
import time
import uuid
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path

_IDENTIFIER = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$")


@dataclass(frozen=True, slots=True)
class RunIdentity:
    path: Path
    run_id: str
    generation_family: str
    created_ns: int


def validate_identifier(name: str, value: object) -> str:
    if not isinstance(value, str) or _IDENTIFIER.fullmatch(value) is None:
        raise ValueError(
            f"{name} must match {_IDENTIFIER.pattern} and be at most 128 characters"
        )
    return value


def atomic_json(path: str | Path, payload: Mapping[str, object]) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary_name: str | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            prefix=f".{destination.name}.",
            suffix=".tmp",
            dir=destination.parent,
            delete=False,
        ) as temporary:
            temporary_name = temporary.name
            json.dump(payload, temporary, sort_keys=True, separators=(",", ":"))
            temporary.write("\n")
            temporary.flush()
            os.fsync(temporary.fileno())
        os.replace(temporary_name, destination)
        _fsync_directory(destination.parent)
    finally:
        if temporary_name is not None and os.path.exists(temporary_name):
            os.unlink(temporary_name)


def load_run_identity(path: str | Path) -> RunIdentity:
    source = Path(path)
    try:
        with source.open("r", encoding="utf-8") as stream:
            payload = json.load(stream)
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"cannot read run identity {source}: {exc}") from exc
    if not isinstance(payload, dict) or payload.get("schema_version") != 1:
        raise ValueError("unsupported run identity")
    run_id = validate_identifier("run_id", payload.get("run_id"))
    family = validate_identifier("generation_family", payload.get("generation_family"))
    created_ns = payload.get("created_ns")
    if (
        isinstance(created_ns, bool)
        or not isinstance(created_ns, int)
        or created_ns <= 0
    ):
        raise ValueError("run identity created_ns is invalid")
    return RunIdentity(source, run_id, family, created_ns)


def load_or_create_run_identity(
    path: str | Path, *, requested_run_id: str | None = None
) -> RunIdentity:
    destination = Path(path)
    if destination.exists():
        identity = load_run_identity(destination)
        if requested_run_id is not None and identity.run_id != validate_identifier(
            "run_id", requested_run_id
        ):
            raise ValueError("configured run_id does not match durable run identity")
        return identity
    run_id = (
        validate_identifier("run_id", requested_run_id)
        if requested_run_id is not None
        else f"run-{uuid.uuid4().hex}"
    )
    payload = {
        "schema_version": 1,
        "run_id": run_id,
        "generation_family": f"family-{uuid.uuid4().hex}",
        "created_ns": time.time_ns(),
    }
    atomic_json(destination, payload)
    return load_run_identity(destination)


def append_jsonl(
    path: str | Path, payload: Mapping[str, object], *, durable: bool = False
) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    data = (json.dumps(payload, sort_keys=True, separators=(",", ":")) + "\n").encode(
        "utf-8"
    )
    descriptor = os.open(
        destination,
        os.O_APPEND | os.O_CREAT | os.O_WRONLY,
        0o644,
    )
    try:
        written = os.write(descriptor, data)
        if written != len(data):
            raise OSError("short JSONL metrics write")
        if durable:
            os.fsync(descriptor)
    finally:
        os.close(descriptor)


class SignalLatch:
    def __init__(self) -> None:
        self.event = threading.Event()
        self.signal_number: int | None = None

    def install(self) -> None:
        def request_stop(signal_number: int, _frame: object) -> None:
            self.signal_number = signal_number
            self.event.set()

        signal.signal(signal.SIGINT, request_stop)
        signal.signal(signal.SIGTERM, request_stop)

    def is_set(self) -> bool:
        return self.event.is_set()


class SystemdNotifier:
    """Best-effort sd_notify client without an optional systemd dependency."""

    def __init__(self, socket_path: str | None = None) -> None:
        self.socket_path = (
            socket_path if socket_path is not None else os.getenv("NOTIFY_SOCKET")
        )

    @property
    def enabled(self) -> bool:
        return bool(self.socket_path)

    def ready(self, status: str) -> None:
        self._send(f"READY=1\nSTATUS={status}")

    def watchdog(self, status: str) -> None:
        self._send(f"WATCHDOG=1\nSTATUS={status}")

    def stopping(self, status: str) -> None:
        self._send(f"STOPPING=1\nSTATUS={status}")

    def _send(self, message: str) -> None:
        if not self.socket_path:
            return
        address: str | bytes = self.socket_path
        if self.socket_path.startswith("@"):
            address = b"\0" + self.socket_path[1:].encode()
        try:
            with socket.socket(socket.AF_UNIX, socket.SOCK_DGRAM) as client:
                client.sendto(message.encode(), address)
        except OSError:
            return


class HeartbeatReporter:
    def __init__(
        self,
        path: str | Path,
        *,
        worker: str,
        interval_seconds: float,
    ) -> None:
        if interval_seconds <= 0:
            raise ValueError("heartbeat interval must be positive")
        self.path = Path(path)
        self.worker = worker
        self.interval_seconds = interval_seconds
        self._stop = threading.Event()
        self._lock = threading.Lock()
        self._write_lock = threading.Lock()
        self._details: dict[str, object] = {"phase": "starting"}
        self._progress = 0
        self._thread = threading.Thread(
            target=self._run,
            name=f"heartbeat-{worker}",
            daemon=True,
        )

    def start(self) -> None:
        self._write()
        self._thread.start()

    def update(self, **details: object) -> None:
        with self._lock:
            self._details.update(details)
        self._write()

    def advance(self, **details: object) -> None:
        with self._lock:
            self._progress += 1
            self._details.update(details)
            self._details["progress"] = self._progress
            self._details["progress_ns"] = time.time_ns()
        self._write()

    def close(self, *, final_phase: str = "stopped") -> None:
        self.update(phase=final_phase)
        self._stop.set()
        if self._thread.is_alive():
            self._thread.join(timeout=self.interval_seconds + 1.0)

    def _run(self) -> None:
        while not self._stop.wait(self.interval_seconds):
            self._write()

    def _write(self) -> None:
        with self._write_lock:
            with self._lock:
                details = dict(self._details)
            atomic_json(
                self.path,
                {
                    "schema_version": 1,
                    "worker": self.worker,
                    "pid": os.getpid(),
                    "heartbeat_ns": time.time_ns(),
                    **details,
                },
            )


def _fsync_directory(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)
