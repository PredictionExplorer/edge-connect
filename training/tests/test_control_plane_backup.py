from __future__ import annotations

import json
from pathlib import Path

import pytest

from scripts import control_plane_backup as control


def _source(tmp_path: Path) -> Path:
    source = tmp_path / "state"
    source.mkdir(parents=True)
    (source / "campaign.json").write_text('{"status":"running"}\n')
    (source / "campaign.json.lock").write_text("runtime lock")
    seed = source / "seed17"
    seed.mkdir()
    (seed / "queue.json").write_text('{"queue_status":"running"}\n')
    return source


def test_control_snapshot_verify_restore_and_deduplicate(tmp_path: Path) -> None:
    source = _source(tmp_path)
    backup = tmp_path / "backup"
    first = control.create_control_snapshot(
        source,
        backup,
        namespace_id="confirmation-campaign",
        enforce_separate_filesystem=False,
    )
    second = control.create_control_snapshot(
        source,
        backup,
        namespace_id="confirmation-campaign",
        enforce_separate_filesystem=False,
    )

    report = control.verify_control_snapshot(
        backup / "snapshots" / "confirmation-campaign" / "latest.json",
        backup,
    )
    assert report["status"] == "ok"
    assert control.verify_control_snapshot(backup / "latest.json", backup)[
        "snapshot"
    ] == str(second)
    assert first != second
    objects = list((backup / "objects" / "sha256").glob("*/*"))
    assert len(objects) == 2

    restored = control.restore_control_snapshot(
        second,
        backup,
        tmp_path / "restored",
    )
    assert json.loads((restored / "campaign.json").read_text())["status"] == "running"
    assert (
        json.loads((restored / "seed17" / "queue.json").read_text())["queue_status"]
        == "running"
    )
    assert not (restored / "campaign.json.lock").exists()


def test_control_snapshot_rejects_empty_source_before_publication(
    tmp_path: Path,
) -> None:
    source = _source(tmp_path)
    backup = tmp_path / "backup"
    empty = source / "partial.json"
    empty.write_bytes(b"")

    with pytest.raises(control.DisasterRecoveryError, match="file is empty"):
        control.create_control_snapshot(
            source,
            backup,
            namespace_id="confirmation-campaign",
            enforce_separate_filesystem=False,
        )

    assert not list((backup / "snapshots").glob("**/*.json"))
    empty.unlink()
    snapshot = control.create_control_snapshot(
        source,
        backup,
        namespace_id="confirmation-campaign",
        enforce_separate_filesystem=False,
    )
    assert snapshot.is_file()


def test_control_snapshot_ignores_uncommitted_invalid_document(
    tmp_path: Path,
) -> None:
    source = _source(tmp_path)
    backup = tmp_path / "backup"
    first = control.create_control_snapshot(
        source,
        backup,
        namespace_id="confirmation-campaign",
        enforce_separate_filesystem=False,
    )
    invalid_payload = json.loads(first.read_text())
    next(iter(invalid_payload["catalog"].values()))["bytes"] = 0
    invalid = first.with_name("9999999999999999999-" + "f" * 64 + ".json")
    invalid.write_bytes(control._canonical_json(invalid_payload))

    assert control._is_committed(invalid, backup) is False
    second = control.create_control_snapshot(
        source,
        backup,
        namespace_id="confirmation-campaign",
        enforce_separate_filesystem=False,
    )

    assert second != first
    assert control.verify_control_snapshot(second, backup)["status"] == "ok"


def test_control_latest_pointer_rejects_stale_snapshot(tmp_path: Path) -> None:
    source = _source(tmp_path)
    backup = tmp_path / "backup"
    first = control.create_control_snapshot(
        source,
        backup,
        namespace_id="confirmation-campaign",
        enforce_separate_filesystem=False,
    )
    (source / "campaign.json").write_text('{"status":"advanced"}\n')
    control.create_control_snapshot(
        source,
        backup,
        namespace_id="confirmation-campaign",
        enforce_separate_filesystem=False,
    )
    first_payload = json.loads(first.read_text())
    first_bytes = first.read_bytes()
    stale = {
        "report": control.LATEST_REPORT,
        "schema_version": 1,
        "run_id": "confirmation-campaign",
        "generation_family": control.CONTROL_FAMILY,
        "path": first.name,
        "sha256": control._sha256_bytes(first_bytes),
        "bytes": len(first_bytes),
        "created_ns": first_payload["created_ns"],
    }
    (backup / "snapshots" / "confirmation-campaign" / "latest.json").write_bytes(
        control._canonical_json(stale)
    )

    with pytest.raises(control.DisasterRecoveryError, match="stale"):
        control.verify_control_snapshot(
            backup / "snapshots" / "confirmation-campaign" / "latest.json",
            backup,
        )


def test_control_snapshot_rejects_clock_regression(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source = _source(tmp_path)
    backup = tmp_path / "backup"
    first = control.create_control_snapshot(
        source,
        backup,
        namespace_id="confirmation-campaign",
        enforce_separate_filesystem=False,
    )
    created_ns = int(json.loads(first.read_text())["created_ns"])
    monkeypatch.setattr(control.time, "time_ns", lambda: created_ns - 1)

    with pytest.raises(control.DisasterRecoveryError, match="clock did not advance"):
        control.create_control_snapshot(
            source,
            backup,
            namespace_id="confirmation-campaign",
            enforce_separate_filesystem=False,
        )


def test_control_snapshot_rejects_other_source_root(tmp_path: Path) -> None:
    first = _source(tmp_path / "first")
    second_parent = tmp_path / "second"
    second_parent.mkdir()
    second = _source(second_parent)
    backup = tmp_path / "backup"
    control.create_control_snapshot(
        first,
        backup,
        namespace_id="confirmation-campaign",
        enforce_separate_filesystem=False,
    )

    with pytest.raises(
        control.DisasterRecoveryError,
        match="another source root",
    ):
        control.create_control_snapshot(
            second,
            backup,
            namespace_id="confirmation-campaign",
            enforce_separate_filesystem=False,
        )


def test_control_snapshot_detects_tampered_object(tmp_path: Path) -> None:
    source = _source(tmp_path)
    backup = tmp_path / "backup"
    snapshot = control.create_control_snapshot(
        source,
        backup,
        namespace_id="confirmation-campaign",
        enforce_separate_filesystem=False,
    )
    payload = json.loads(snapshot.read_text())
    entry = next(iter(payload["catalog"].values()))
    path = backup / "objects" / "sha256" / entry["sha256"][:2] / entry["sha256"]
    path.chmod(0o644)
    path.write_text("tampered")

    with pytest.raises(control.DisasterRecoveryError, match="byte length|SHA-256"):
        control.verify_control_snapshot(snapshot, backup)
