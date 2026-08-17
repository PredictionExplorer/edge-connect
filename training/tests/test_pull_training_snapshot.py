from __future__ import annotations

import base64
import hashlib
import json
import shutil
import subprocess
from collections.abc import Callable
from pathlib import Path, PurePosixPath

import pytest

from scripts.pull_training_snapshot import (
    LATEST_RELATIVE_PATH,
    OpenSshTransport,
    PullConfig,
    SnapshotPullError,
    SnapshotTransportError,
    SnapshotValidationError,
    pull_snapshot,
)


class LocalTransport:
    def __init__(
        self,
        remote_root: Path,
        *,
        before_fetch: Callable[[PurePosixPath], None] | None = None,
        fail_fetch: PurePosixPath | None = None,
    ) -> None:
        self.remote_root = remote_root
        self.before_fetch = before_fetch
        self.fail_fetch = fail_fetch
        self.read_calls: list[PurePosixPath] = []
        self.fetch_calls: list[PurePosixPath] = []
        self.acknowledgements: list[tuple[PurePosixPath, bytes]] = []
        self.before_ack: Callable[[], None] | None = None

    def read_file(
        self,
        relative_path: PurePosixPath,
        *,
        max_bytes: int,
    ) -> bytes:
        self.read_calls.append(relative_path)
        payload = self.remote_root.joinpath(*relative_path.parts).read_bytes()
        if len(payload) > max_bytes:
            raise SnapshotTransportError("test metadata exceeds limit")
        return payload

    def fetch_file(
        self,
        relative_path: PurePosixPath,
        local_partial_path: Path,
        *,
        expected_bytes: int,
    ) -> None:
        _ = expected_bytes
        self.fetch_calls.append(relative_path)
        if self.before_fetch is not None:
            self.before_fetch(relative_path)
        if relative_path == self.fail_fetch:
            local_partial_path.write_bytes(b"incomplete transfer")
            raise SnapshotTransportError("injected transfer failure")
        shutil.copyfile(
            self.remote_root.joinpath(*relative_path.parts),
            local_partial_path,
        )

    def write_acknowledgement(
        self,
        relative_path: PurePosixPath,
        payload: bytes,
    ) -> None:
        if self.before_ack is not None:
            self.before_ack()
        self.acknowledgements.append((relative_path, payload))


def _json_bytes(document: object) -> bytes:
    return (json.dumps(document, sort_keys=True, separators=(",", ":")) + "\n").encode()


def _write_snapshot(
    remote_root: Path,
    *,
    contents: list[tuple[str, bytes, str]],
    catalog_as_mapping: bool = False,
    report: str = "startrain-disaster-recovery-snapshot",
    source_root: str = "/srv/edgeconnect/run-safe",
    created_ns: int = 123_456_789,
) -> tuple[PurePosixPath, str, dict[str, bytes]]:
    run_id = "run-safe"
    generation_family = "family-safe"
    (remote_root / "namespace.json").parent.mkdir(parents=True, exist_ok=True)
    (remote_root / "namespace.json").write_bytes(
        _json_bytes(
            {
                "report": "startrain-disaster-recovery-namespace",
                "schema_version": 1,
                "run_id": run_id,
                "generation_family": generation_family,
                "source_run_root": source_root,
            }
        )
    )
    entries: list[dict[str, object]] = []
    objects: dict[str, bytes] = {}
    for logical_path, payload, kind in contents:
        digest = hashlib.sha256(payload).hexdigest()
        objects[digest] = payload
        entries.append(
            {
                "logical_path": logical_path,
                "sha256": digest,
                "bytes": len(payload),
                "kind": kind,
            }
        )
        object_path = remote_root / "objects" / "sha256" / digest[:2] / digest
        object_path.parent.mkdir(parents=True, exist_ok=True)
        object_path.write_bytes(payload)
    catalog: object
    if catalog_as_mapping:
        catalog = {
            str(entry["logical_path"]): {
                key: value for key, value in entry.items() if key != "logical_path"
            }
            for entry in entries
        }
    else:
        catalog = entries
    manifest_payload = _json_bytes(
        {
            "schema_version": 1,
            "report": report,
            "run_id": run_id,
            "generation_family": generation_family,
            "created_ns": created_ns,
            "source": (
                {"source_root": source_root}
                if report == "startrain-control-plane-snapshot"
                else {"run_root": source_root}
            ),
            "catalog": catalog,
        }
    )
    manifest_sha256 = hashlib.sha256(manifest_payload).hexdigest()
    manifest_relative = PurePosixPath(
        "snapshots",
        run_id,
        f"{created_ns}-{manifest_sha256}.json",
    )
    manifest_path = remote_root.joinpath(*manifest_relative.parts)
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_bytes(manifest_payload)
    (manifest_path.parent / LATEST_RELATIVE_PATH.name).write_bytes(
        _json_bytes(
            {
                "report": "startrain-disaster-recovery-latest",
                "schema_version": 1,
                "run_id": run_id,
                "generation_family": generation_family,
                "path": manifest_relative.name,
                "sha256": manifest_sha256,
                "bytes": len(manifest_payload),
                "created_ns": created_ns,
            }
        )
    )
    return manifest_relative, manifest_sha256, objects


def _config(
    tmp_path: Path,
    *,
    local_root: Path | None = None,
    ack: bool = False,
) -> PullConfig:
    known_hosts = tmp_path / "known_hosts"
    known_hosts.write_text("test host key\n", encoding="utf-8")
    return PullConfig.from_values(
        host="backup@203.0.113.10",
        remote_backup_root="/srv/edgeconnect-backups",
        local_backup_root=local_root or tmp_path / "local",
        known_hosts_file=known_hosts,
        run_id="run-safe",
        ack_remote_path=(
            "/srv/edgeconnect-backups/acks/mac-mini.json" if ack else None
        ),
    )


def _local_latest(config: PullConfig) -> Path:
    assert config.run_id is not None
    return (
        config.local_backup_root
        / "snapshots"
        / config.run_id
        / LATEST_RELATIVE_PATH.name
    )


def _remote_latest_relative(run_id: str = "run-safe") -> PurePosixPath:
    return PurePosixPath("snapshots", run_id, LATEST_RELATIVE_PATH.name)


def test_configuration_rejects_hostile_transport_arguments(tmp_path: Path) -> None:
    known_hosts = tmp_path / "known_hosts"
    hostile_values = [
        {"host": "-oProxyCommand=touch /tmp/pwned"},
        {"host": "backup@example.test;touch-pwned"},
        {"remote_backup_root": "/srv/backups;touch-pwned"},
        {"run_id": "../other-run"},
        {"ack_remote_path": "/srv/other/ack.json"},
    ]
    defaults: dict[str, object] = {
        "host": "backup@example.test",
        "remote_backup_root": "/srv/backups",
        "local_backup_root": tmp_path / "local",
        "known_hosts_file": known_hosts,
        "run_id": "run-safe",
        "ack_remote_path": None,
    }
    for hostile in hostile_values:
        with pytest.raises(ValueError):
            PullConfig.from_values(**(defaults | hostile))

    with pytest.raises(ValueError, match="absolute"):
        PullConfig.from_values(**(defaults | {"local_backup_root": Path("relative")}))
    with pytest.raises(ValueError, match="absolute"):
        PullConfig.from_values(**(defaults | {"remote_backup_root": "relative"}))


def test_openssh_transport_enforces_pinned_host_keys_and_argument_arrays(
    tmp_path: Path,
) -> None:
    config = _config(
        tmp_path,
        local_root=tmp_path / "local path $(not-a-shell)",
    )
    calls: list[tuple[list[str], dict[str, object]]] = []

    def runner(
        command: list[str],
        **kwargs: object,
    ) -> subprocess.CompletedProcess[bytes]:
        calls.append((command, kwargs))
        if command[0] == "rsync":
            Path(command[-1]).write_bytes(b"artifact")
        stdout = b'{"bytes": 8}' if len(calls) == 2 else b"{}"
        return subprocess.CompletedProcess(command, 0, stdout, b"")

    transport = OpenSshTransport(config, runner=runner)
    assert transport.read_file(PurePosixPath("latest.json"), max_bytes=100) == b"{}"
    partial = tmp_path / "local path $(not-a-shell)" / "artifact.partial"
    partial.parent.mkdir()
    transport.fetch_file(
        PurePosixPath("objects/sha256/aa/" + "a" * 64),
        partial,
        expected_bytes=len(b"artifact"),
    )

    assert {call[0][0] for call in calls} == {"ssh", "rsync"}
    for command, kwargs in calls:
        assert isinstance(command, list)
        assert kwargs["check"] is False
        command_text = " ".join(command)
        assert "StrictHostKeyChecking=yes" in command_text
        assert "GlobalKnownHostsFile=/dev/null" in command_text
        assert "UserKnownHostsFile=" in command_text
        assert "BatchMode=yes" in command_text
    assert calls[2][0][-1] == str(partial)
    assert "--max-size=8" in calls[2][0]


def test_remote_ack_paths_and_payload_are_sent_on_stdin(
    tmp_path: Path,
) -> None:
    config = _config(tmp_path)
    calls: list[tuple[list[str], bytes]] = []

    def runner(
        command: list[str],
        **kwargs: object,
    ) -> subprocess.CompletedProcess[bytes]:
        calls.append((command, kwargs["input"]))  # type: ignore[arg-type]
        return subprocess.CompletedProcess(command, 0, b"", b"")

    transport = OpenSshTransport(config, runner=runner)
    payload = b'{"status":"verified; $(touch /tmp/pwned)"}\n'
    relative = PurePosixPath("acks/mac-mini.json")
    transport.write_acknowledgement(relative, payload)

    command, stdin = calls[0]
    command_text = " ".join(command)
    assert "/srv/edgeconnect-backups" not in command_text
    assert "acks/mac-mini.json" not in command_text
    assert "touch /tmp/pwned" not in command_text
    request = json.loads(stdin)
    assert request["relative_path"] == str(relative)
    assert base64.b64decode(request["payload_base64"]) == payload


@pytest.mark.parametrize("catalog_as_mapping", [False, True])
def test_pull_accepts_list_or_mapping_catalog_and_deduplicates_objects(
    tmp_path: Path,
    catalog_as_mapping: bool,
) -> None:
    remote = tmp_path / "remote"
    manifest_relative, _, objects = _write_snapshot(
        remote,
        contents=[
            ("checkpoints/latest.pt", b"same-content", "checkpoint"),
            ("checkpoints/champion.pt", b"same-content", "checkpoint"),
        ],
        catalog_as_mapping=catalog_as_mapping,
    )
    config = _config(tmp_path)
    digest, payload = next(iter(objects.items()))
    local_object = config.local_backup_root / "objects" / "sha256" / digest[:2] / digest
    local_object.parent.mkdir(parents=True)
    local_object.write_bytes(payload)
    transport = LocalTransport(remote)

    result = pull_snapshot(config, transport=transport)

    assert transport.read_calls == [
        _remote_latest_relative(),
        PurePosixPath("namespace.json"),
        manifest_relative,
    ]
    assert transport.fetch_calls == []
    assert result.object_count == 1
    assert result.transferred_objects == 0
    assert result.snapshot_sha256
    assert _local_latest(config).is_file()
    assert (config.local_backup_root / "namespace.json").is_file()


def test_pull_without_run_id_uses_a_root_latest_pointer(tmp_path: Path) -> None:
    remote = tmp_path / "remote"
    manifest_relative, _, _ = _write_snapshot(
        remote,
        contents=[("run.json", b"identity", "run-identity")],
    )
    remote_latest = remote.joinpath(
        *manifest_relative.parent.parts,
        LATEST_RELATIVE_PATH.name,
    )
    (remote / LATEST_RELATIVE_PATH.name).write_bytes(remote_latest.read_bytes())
    known_hosts = tmp_path / "known_hosts"
    known_hosts.write_text("test host key\n", encoding="utf-8")
    config = PullConfig.from_values(
        host="backup@203.0.113.10",
        remote_backup_root="/srv/edgeconnect-backups",
        local_backup_root=tmp_path / "local",
        known_hosts_file=known_hosts,
    )
    transport = LocalTransport(remote)

    result = pull_snapshot(config, transport=transport)

    assert result.run_id == "run-safe"
    assert transport.read_calls[0] == LATEST_RELATIVE_PATH
    assert (config.local_backup_root / LATEST_RELATIVE_PATH.name).is_file()


def test_pull_accepts_control_plane_snapshot(tmp_path: Path) -> None:
    remote = tmp_path / "remote"
    _write_snapshot(
        remote,
        contents=[("campaign.json", b"{}", "control-plane-file")],
        report="startrain-control-plane-snapshot",
    )
    config = _config(tmp_path)

    result = pull_snapshot(config, transport=LocalTransport(remote))

    assert result.run_id == "run-safe"
    assert result.object_count == 1


def test_namespace_handoff_requires_matching_snapshot_and_is_archived(
    tmp_path: Path,
) -> None:
    remote = tmp_path / "remote"
    first_manifest, _, _ = _write_snapshot(
        remote,
        contents=[("run.json", b"first", "run-identity")],
    )
    config = _config(tmp_path)
    pull_snapshot(config, transport=LocalTransport(remote))
    old_namespace = (config.local_backup_root / "namespace.json").read_bytes()

    remote_namespace = json.loads((remote / "namespace.json").read_text())
    remote_namespace["source_run_root"] = "/srv/edgeconnect/relocated"
    (remote / "namespace.json").write_bytes(_json_bytes(remote_namespace))
    with pytest.raises(
        SnapshotValidationError,
        match="namespace source does not match",
    ):
        pull_snapshot(config, transport=LocalTransport(remote))
    assert (config.local_backup_root / "namespace.json").read_bytes() == old_namespace

    second_manifest, _, _ = _write_snapshot(
        remote,
        contents=[("run.json", b"second", "run-identity")],
        source_root="/srv/edgeconnect/relocated",
        created_ns=223_456_789,
    )
    pull_snapshot(config, transport=LocalTransport(remote))

    assert first_manifest != second_manifest
    local_namespace = json.loads(
        (config.local_backup_root / "namespace.json").read_text()
    )
    assert local_namespace["source_run_root"] == "/srv/edgeconnect/relocated"
    archived = list((config.local_backup_root / "namespace-history").glob("*.json"))
    assert len(archived) == 1
    assert archived[0].read_bytes() == old_namespace


def test_remote_pointer_rollback_does_not_replace_local_latest(tmp_path: Path) -> None:
    remote = tmp_path / "remote"
    first_manifest, _, _ = _write_snapshot(
        remote,
        contents=[("run.json", b"first", "run-identity")],
    )
    first_latest_path = remote.joinpath(
        *first_manifest.parent.parts,
        LATEST_RELATIVE_PATH.name,
    )
    first_latest = first_latest_path.read_bytes()
    config = _config(tmp_path)
    pull_snapshot(config, transport=LocalTransport(remote))
    _write_snapshot(
        remote,
        contents=[("run.json", b"second", "run-identity")],
        created_ns=223_456_789,
    )
    pull_snapshot(config, transport=LocalTransport(remote))
    local_latest = _local_latest(config).read_bytes()
    first_latest_path.write_bytes(first_latest)

    with pytest.raises(SnapshotValidationError, match="older than"):
        pull_snapshot(config, transport=LocalTransport(remote))

    assert _local_latest(config).read_bytes() == local_latest


def test_existing_tampered_object_is_rejected_without_replacement(
    tmp_path: Path,
) -> None:
    remote = tmp_path / "remote"
    manifest_relative, _, objects = _write_snapshot(
        remote,
        contents=[("replay/manifest.sqlite3", b"verified", "replay")],
    )
    config = _config(tmp_path)
    digest = next(iter(objects))
    local_object = config.local_backup_root / "objects" / "sha256" / digest[:2] / digest
    local_object.parent.mkdir(parents=True)
    local_object.write_bytes(b"tampered")
    old_pointer = remote.joinpath(
        *manifest_relative.parent.parts,
        LATEST_RELATIVE_PATH.name,
    ).read_bytes()
    _local_latest(config).parent.mkdir(parents=True, exist_ok=True)
    _local_latest(config).write_bytes(old_pointer)
    transport = LocalTransport(remote)

    with pytest.raises(SnapshotPullError, match="failed SHA-256|bytes"):
        pull_snapshot(config, transport=transport)

    assert local_object.read_bytes() == b"tampered"
    assert _local_latest(config).read_bytes() == old_pointer
    assert all("objects" not in path.parts for path in transport.fetch_calls)


def test_tampered_remote_object_never_publishes_latest_pointer(
    tmp_path: Path,
) -> None:
    remote = tmp_path / "remote"
    manifest_relative, _, objects = _write_snapshot(
        remote,
        contents=[("profile.yaml", b"expected", "profile")],
    )
    digest = next(iter(objects))
    (remote / "objects" / "sha256" / digest[:2] / digest).write_bytes(b"remote tamper")
    config = _config(tmp_path)

    with pytest.raises(SnapshotPullError, match="failed SHA-256|bytes"):
        pull_snapshot(config, transport=LocalTransport(remote))

    assert not _local_latest(config).exists()
    assert not list(config.local_backup_root.rglob("*.partial"))


def test_partial_transfer_failure_retains_verified_objects_but_not_pointer(
    tmp_path: Path,
) -> None:
    remote = tmp_path / "remote"
    manifest_relative, _, objects = _write_snapshot(
        remote,
        contents=[
            ("checkpoints/a.pt", b"first object", "checkpoint"),
            ("checkpoints/b.pt", b"second object", "checkpoint"),
        ],
    )
    ordered_digests = sorted(objects)
    failed_relative = PurePosixPath(
        "objects",
        "sha256",
        ordered_digests[1][:2],
        ordered_digests[1],
    )
    config = _config(tmp_path)
    transport = LocalTransport(remote, fail_fetch=failed_relative)

    with pytest.raises(SnapshotTransportError, match="injected"):
        pull_snapshot(config, transport=transport)

    first_digest = ordered_digests[0]
    assert (
        config.local_backup_root
        / "objects"
        / "sha256"
        / first_digest[:2]
        / first_digest
    ).is_file()
    assert not _local_latest(config).exists()
    assert not config.local_backup_root.joinpath(*manifest_relative.parts).exists()
    assert not list(config.local_backup_root.rglob("*.partial"))


def test_pointer_is_published_after_verification_and_before_remote_ack(
    tmp_path: Path,
) -> None:
    remote = tmp_path / "remote"
    manifest_relative, manifest_sha256, objects = _write_snapshot(
        remote,
        contents=[
            ("run.json", b'{"run_id":"run-safe"}\n', "run-identity"),
            ("profile.yaml", b"training: continuous\n", "profile"),
        ],
    )
    config = _config(tmp_path, ack=True)

    def assert_pointer_not_published(_: PurePosixPath) -> None:
        assert not _local_latest(config).exists()

    transport = LocalTransport(remote, before_fetch=assert_pointer_not_published)

    def assert_complete_before_ack() -> None:
        assert _local_latest(config).is_file()
        assert config.local_backup_root.joinpath(*manifest_relative.parts).is_file()
        for digest in objects:
            assert (
                config.local_backup_root / "objects" / "sha256" / digest[:2] / digest
            ).is_file()

    transport.before_ack = assert_complete_before_ack
    result = pull_snapshot(
        config,
        transport=transport,
        completed_ns=987_654_321,
        mac_hostname="training-mac.example",
    )

    assert result.acknowledged is True
    [(ack_path, ack_payload)] = transport.acknowledgements
    assert ack_path == PurePosixPath("acks/mac-mini.json")
    acknowledgement = json.loads(ack_payload)
    assert acknowledgement == {
        "schema_version": 1,
        "report": "startrain-disaster-recovery-acknowledgement",
        "snapshot_sha256": manifest_sha256,
        "snapshot_path": str(manifest_relative),
        "completed_ns": 987_654_321,
        "mac_hostname": "training-mac.example",
        "local_verification_status": "verified",
    }


def test_ack_failure_retains_fully_verified_local_backup(tmp_path: Path) -> None:
    remote = tmp_path / "remote"
    _write_snapshot(
        remote,
        contents=[("run.json", b"durable", "run-identity")],
    )
    config = _config(tmp_path, ack=True)

    class AckFailureTransport(LocalTransport):
        def write_acknowledgement(
            self,
            relative_path: PurePosixPath,
            payload: bytes,
        ) -> None:
            raise SnapshotTransportError("remote acknowledgement unavailable")

    with pytest.raises(SnapshotTransportError, match="acknowledgement"):
        pull_snapshot(config, transport=AckFailureTransport(remote))

    assert _local_latest(config).is_file()
