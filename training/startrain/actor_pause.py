"""Cooperative actor parking outside replay transactions and model operations."""

from __future__ import annotations

import json
import os
import threading
import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .runtime import RunIdentity


@dataclass(frozen=True, slots=True)
class _ActorPauseLease:
    token: str
    owner_pid: int
    requested_ns: int


class ActorPauseGate:
    """Park producers at explicit safe points while the main loop stays alive.

    Only ``poll`` reads coordinator files and publishes readiness. Producers
    retain their complete stack, native searches, and trajectories while
    ``checkpoint`` waits. Request disappearance never releases an adopted
    lease: the coordinator must first prove its GPU owner has released it.
    """

    def __init__(
        self,
        *,
        request_path: str | Path,
        gpu_id: int,
        worker_name: str,
        run_identity: RunIdentity,
        cohort_ids: Sequence[str],
        stop_requested: Callable[[], bool],
        inference_idle: Callable[[], bool],
        synchronize: Callable[[], None],
        stale_seconds: float,
    ) -> None:
        if not cohort_ids or len(set(cohort_ids)) != len(cohort_ids):
            raise ValueError("actor pause requires unique live cohort identifiers")
        if stale_seconds <= 0:
            raise ValueError("actor pause heartbeat timeout must be positive")
        self.request_path = Path(request_path)
        self.ack_path = self.request_path.with_name(
            f"{self.request_path.stem}.ack{self.request_path.suffix}"
        )
        self.gpu_id = gpu_id
        self.worker_name = worker_name
        self.run_identity = run_identity
        self.pid = os.getpid()
        self._stop_requested = stop_requested
        self._inference_idle = inference_idle
        self._synchronize = synchronize
        self._stale_ns = int(stale_seconds * 1_000_000_000)
        self._condition = threading.Condition()
        self._live = set(cohort_ids)
        self._parked: set[str] = set()
        self._lease: _ActorPauseLease | None = None
        self._releasing: _ActorPauseLease | None = None
        self._last_resumed_token: str | None = None
        self._synchronized = False
        self._closed = False

    @staticmethod
    def _read(path: Path) -> dict[str, Any]:
        try:
            value = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            return {}
        return value if isinstance(value, dict) else {}

    def _matching_ack(self, payload: dict[str, Any], lease: _ActorPauseLease) -> bool:
        return (
            payload.get("schema_version") == 1
            and payload.get("protocol") == "coordinator-pause-v1"
            and payload.get("token") == lease.token
            and payload.get("gpu_id") == self.gpu_id
            and payload.get("target_worker") == self.worker_name
            and payload.get("target_pid") == self.pid
            and type(payload.get("ack_ns")) is int
            and payload["ack_ns"] >= lease.requested_ns
        )

    def _requested_lease(
        self, acknowledgement: dict[str, Any]
    ) -> _ActorPauseLease | None:
        payload = self._read(self.request_path)
        token = payload.get("token")
        pid = payload.get("pid")
        requested_ns = payload.get("requested_ns")
        heartbeat_ns = payload.get("heartbeat_ns")
        if not (
            payload.get("schema_version") == 1
            and payload.get("protocol") == "coordinator-pause-v1"
            and payload.get("gpu_id") == self.gpu_id
            and payload.get("state") in ("requested", "active")
            and isinstance(token, str)
            and 8 <= len(token) <= 128
            and type(pid) is int
            and pid > 0
            and type(requested_ns) is int
            and requested_ns > 0
            and type(heartbeat_ns) is int
            and heartbeat_ns >= requested_ns
            and 0 <= time.time_ns() - heartbeat_ns <= self._stale_ns
            and payload.get("run_id", self.run_identity.run_id)
            == self.run_identity.run_id
            and payload.get("generation_family", self.run_identity.generation_family)
            == self.run_identity.generation_family
        ):
            return None
        lease = _ActorPauseLease(token, pid, requested_ns)
        # The coordinator validates the owner against its supervised process.
        # Its token/target-matched waiting acknowledgement authorizes parking.
        if acknowledgement.get("state") != "waiting" or not self._matching_ack(
            acknowledgement, lease
        ):
            return None
        return lease

    def _unadopted_release(self, acknowledgement: dict[str, Any]) -> str | None:
        """Confirm a cancellation that raced ahead of producer adoption."""
        token = acknowledgement.get("token")
        ack_ns = acknowledgement.get("ack_ns")
        if (
            acknowledgement.get("state") not in ("released", "recovered", "draining")
            or not isinstance(token, str)
            or not 8 <= len(token) <= 128
            or type(ack_ns) is not int
            or ack_ns <= 0
            or not 0 <= time.time_ns() - ack_ns <= self._stale_ns
            or acknowledgement.get("run_id", self.run_identity.run_id)
            != self.run_identity.run_id
            or acknowledgement.get(
                "generation_family", self.run_identity.generation_family
            )
            != self.run_identity.generation_family
        ):
            return None
        # No producer adopted the lease, so there is nothing to unpark. The
        # coordinator still needs confirmation before it replaces this ack.
        lease = _ActorPauseLease(token, 0, ack_ns)
        return token if self._matching_ack(acknowledgement, lease) else None

    def checkpoint(self, cohort_id: str) -> None:
        """Pause only: callers must invoke this outside shared operations."""
        with self._condition:
            if cohort_id not in self._live:
                return
            try:
                while (
                    self._lease is not None
                    and not self._closed
                    and not self._stop_requested()
                ):
                    self._parked.add(cohort_id)
                    self._condition.notify_all()
                    self._condition.wait(timeout=0.05)
            finally:
                if cohort_id in self._parked:
                    self._parked.remove(cohort_id)
                    self._synchronized = False
                self._condition.notify_all()

    def finish(self, cohort_id: str) -> None:
        """Deregister only after the producer has completed all cleanup."""
        with self._condition:
            self._live.discard(cohort_id)
            self._parked.discard(cohort_id)
            self._synchronized = False
            self._condition.notify_all()

    def poll(self) -> dict[str, object] | None:
        """Return heartbeat details while quiescing/parked, otherwise None."""
        acknowledgement = self._read(self.ack_path)
        with self._condition:
            if self._closed or self._stop_requested():
                self.close()
                return None
            if self._releasing is not None:
                if self._parked:
                    return self._resuming_details()
                self._last_resumed_token = self._releasing.token
                self._releasing = None
                # Publish the completed release before considering a new token.
                return self._resuming_details()
            if self._lease is None:
                released = self._unadopted_release(acknowledgement)
                if released is not None:
                    self._last_resumed_token = released
                    return self._resuming_details()
                self._lease = self._requested_lease(acknowledgement)
                self._synchronized = False
            lease = self._lease
            if lease is None:
                return (
                    self._resuming_details()
                    if self._last_resumed_token is not None
                    else None
                )
            if self._matching_ack(acknowledgement, lease) and acknowledgement.get(
                "state"
            ) in (
                "released",
                "recovered",
                "draining",
            ):
                self._lease = None
                self._releasing = lease
                self._synchronized = False
                self._condition.notify_all()
                return self._resuming_details()
            all_parked = bool(self._live) and self._parked == self._live
            idle = all_parked and self._inference_idle()
            if idle and not self._synchronized:
                # Every producer is parked; neither refresh nor replay writes
                # can start. The broker's idle proof includes its owned jobs.
                self._synchronize()
                self._synchronized = True
            ready = idle and self._synchronized and not self._stop_requested()
            return {
                "phase": "arena_gpu_pause" if ready else "arena_gpu_quiescing",
                "lease_token": lease.token,
                "lease_owner_pid": lease.owner_pid,
                "lease_requested_ns": lease.requested_ns,
                "actor_quiescent": ready,
                "inference_idle": idle,
                "cuda_synchronized": self._synchronized,
                "parked_cohorts": len(self._parked),
                "live_cohorts": len(self._live),
            }

    def _resuming_details(self) -> dict[str, object]:
        return {
            "phase": "arena_gpu_resuming"
            if self._releasing is not None
            else "shared_cohorts",
            "lease_token": self._releasing.token
            if self._releasing is not None
            else None,
            "last_resumed_lease_token": self._last_resumed_token,
            "actor_quiescent": False,
            "inference_idle": False,
            "cuda_synchronized": False,
            "parked_cohorts": len(self._parked),
            "live_cohorts": len(self._live),
        }

    def close(self) -> None:
        with self._condition:
            self._closed = True
            self._lease = None
            self._releasing = None
            self._synchronized = False
            self._condition.notify_all()
