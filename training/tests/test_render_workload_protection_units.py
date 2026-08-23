from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest
import yaml

from scripts import render_workload_protection_units as renderer
from startrain import continuity


def _write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, sort_keys=True), encoding="utf-8")


def _run(tmp_path: Path, name: str, *, created_ns: int) -> tuple[Path, Path]:
    root = (tmp_path / name).resolve()
    root.mkdir()
    run_id = f"{name}-run"
    _write_json(
        root / "run.json",
        {
            "schema_version": 1,
            "run_id": run_id,
            "generation_family": f"{name}-family",
            "created_ns": created_ns,
        },
    )
    profile = root / "profile.yaml"
    profile.write_text(
        yaml.safe_dump(
            {
                "orchestration": {
                    "run_id": run_id,
                    "directories": {"root": str(root)},
                }
            }
        ),
        encoding="utf-8",
    )
    return root, profile


def _manifest(tmp_path: Path) -> Path:
    primary_root, primary_profile = _run(
        tmp_path,
        "primary",
        created_ns=100,
    )
    fallback_root, fallback_profile = _run(
        tmp_path,
        "fallback",
        created_ns=90,
    )
    primary_hashes = continuity.workload_fingerprints(
        primary_profile,
        primary_root,
    )
    fallback_hashes = continuity.workload_fingerprints(
        fallback_profile,
        fallback_root,
    )
    state_root = (tmp_path / "state").resolve()
    state_root.mkdir()
    release_root = (tmp_path / "release").resolve()
    training_dir = release_root / "training"
    source_files: dict[str, str] = {}
    for relative in (
        "startrain/orchestration.py",
        "startrain/continuity.py",
        "scripts/reconcile_training_continuity.py",
    ):
        path = training_dir / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(f"# {relative}\n", encoding="utf-8")
        source_files[f"training/{relative}"] = hashlib.sha256(
            path.read_bytes()
        ).hexdigest()
    orchestrator = training_dir / ".venv" / "bin" / "startrain-orchestrate"
    orchestrator.parent.mkdir(parents=True)
    orchestrator.write_text("#!/bin/sh\nexit 0\n", encoding="utf-8")
    orchestrator.chmod(0o755)
    orchestrator_sha256 = hashlib.sha256(orchestrator.read_bytes()).hexdigest()
    runtime_manifest = release_root / "release-manifest.json"
    _write_json(
        runtime_manifest,
        {
            "report": "edgeconnect-immutable-release",
            "schema_version": 1,
            "source_files": source_files,
        },
    )
    runtime_sha256 = hashlib.sha256(runtime_manifest.read_bytes()).hexdigest()
    unit_root = (tmp_path / "units").resolve()
    unit_root.mkdir()
    primary_unit = unit_root / "edgeconnect-primary.service"
    fallback_unit = unit_root / "edgeconnect-fallback.service"
    primary_unit.write_text("[Service]\nExecStart=/bin/true\n", encoding="utf-8")
    fallback_unit.write_text("[Service]\nExecStart=/bin/true\n", encoding="utf-8")

    def workload(
        workload_id: str,
        role: str,
        root: Path,
        profile: Path,
        hashes: dict[str, object],
        unit: Path,
    ) -> dict[str, object]:
        payload: dict[str, object] = {
            "id": workload_id,
            "role": role,
            "unit": unit.name,
            "profile": {
                "path": str(profile),
                "sha256": hashes["profile_sha256"],
            },
            "run_root": {
                "path": str(root),
                "sha256": hashes["run_root_sha256"],
            },
            "runtime": {
                "manifest": str(runtime_manifest),
                "sha256": runtime_sha256,
                "training_dir": str(training_dir),
                "orchestrator": str(orchestrator),
                "orchestrator_sha256": orchestrator_sha256,
                "unit_path": str(unit),
                "unit_sha256": hashlib.sha256(unit.read_bytes()).hexdigest(),
            },
        }
        if role == "fallback":
            payload["last_known_good"] = {"verified_ns": 80}
        return payload

    disaster_mount = (tmp_path / "disaster").resolve()
    disaster_root = (disaster_mount / "primary").resolve()
    disaster_root.mkdir(parents=True)
    primary = workload(
        "primary",
        "primary",
        primary_root,
        primary_profile,
        primary_hashes,
        primary_unit,
    )
    primary["protection"] = {
        "replay_backup_timer": "edgeconnect-startrain-primary-backup.timer",
        "disaster_backup_timer": (
            "edgeconnect-startrain-primary-disaster-backup.timer"
        ),
        "disaster_backup_root": str(disaster_root),
        "disaster_backup_mount": str(disaster_mount),
        "mac_acknowledgement_namespace": str(disaster_root / "acknowledgements"),
        "telemetry_service": "edgeconnect-startrain-primary-monitor.service",
        "telemetry_output": str(primary_root / "status" / "monitor-5s.jsonl"),
    }
    pinned = state_root / "continuity-manifest.json"
    _write_json(
        pinned,
        {
            "format": continuity.MANIFEST_FORMAT,
            "schema_version": 1,
            "state_root": str(state_root),
            "locks": {
                "transition": str(state_root / "transition.lock"),
                "execution": str(tmp_path / "execution.lock"),
            },
            "hardware": {
                "report_path": str(state_root / "hardware.json"),
                "max_age_seconds": 180,
                "probe_workload": "fallback-lkg",
            },
            "primary": "primary",
            "workloads": [
                primary,
                workload(
                    "fallback-lkg",
                    "fallback",
                    fallback_root,
                    fallback_profile,
                    fallback_hashes,
                    fallback_unit,
                ),
            ],
        },
    )
    return pinned


def test_renderer_is_deterministic_and_resolves_all_templates(
    tmp_path: Path,
) -> None:
    manifest = _manifest(tmp_path)

    first = renderer.rendered_workload_protection_units(
        manifest,
        "primary",
        user="ubuntu",
        provisioned_gpus=8,
    )
    second = renderer.rendered_workload_protection_units(
        manifest,
        "primary",
        user="ubuntu",
        provisioned_gpus=8,
    )

    assert first == second
    assert len(first) == 7
    assert all(b"@" not in content for content in first.values())
    assert (
        b"--continuity-manifest " + str(manifest).encode()
        in first["edgeconnect-startrain-primary-monitor.service"]
    )
    assert (
        b"--interval 5 --format jsonl"
        in first["edgeconnect-startrain-primary-monitor.service"]
    )
    assert (
        b"Unit=edgeconnect-startrain-primary-backup.service"
        in first["edgeconnect-startrain-primary-backup.timer"]
    )


def test_renderer_refuses_overwrite_without_safe_replace_flag(
    tmp_path: Path,
) -> None:
    manifest = _manifest(tmp_path)
    output = tmp_path / "rendered"

    files = renderer.render_workload_protection_units(
        manifest,
        "primary",
        output,
        user="ubuntu",
        provisioned_gpus=8,
    )
    monitor = output / "edgeconnect-startrain-primary-monitor.service"
    expected = monitor.read_bytes()
    assert monitor.stat().st_mode & 0o777 == 0o644
    monitor.write_text("drift\n", encoding="utf-8")

    with pytest.raises(renderer.RenderProtectionError, match="refusing to overwrite"):
        renderer.render_workload_protection_units(
            manifest,
            "primary",
            output,
            user="ubuntu",
            provisioned_gpus=8,
        )

    replaced = renderer.render_workload_protection_units(
        manifest,
        "primary",
        output,
        user="ubuntu",
        provisioned_gpus=8,
        replace_existing=True,
    )
    assert len(files) == len(replaced) == 7
    assert monitor.read_bytes() == expected
