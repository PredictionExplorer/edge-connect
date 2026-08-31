#!/usr/bin/env python3
"""Render, but never install, protection units for one continuity workload."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import sys
from pathlib import Path

from startrain.continuity import (
    ContinuityError,
    ContinuityManifest,
    Workload,
    load_continuity_manifest,
    verify_workload,
    workload_protection_commands,
)

_USER = re.compile(r"^[a-z_][a-z0-9_-]{0,63}\$?$")
_PLACEHOLDER = re.compile(r"@[A-Z][A-Z0-9_]*@")
_TEMPLATES = {
    "replay_service": "edgeconnect-startrain-backup.service.example",
    "replay_timer": "edgeconnect-startrain-backup.timer.example",
    "disaster_service": "edgeconnect-startrain-disaster-backup.service.example",
    "disaster_timer": "edgeconnect-startrain-disaster-backup.timer.example",
    "report_service": "edgeconnect-startrain-report.service.example",
    "report_timer": "edgeconnect-startrain-report.timer.example",
}


class RenderProtectionError(RuntimeError):
    """Protection units cannot be rendered safely."""


def _template(path: Path, replacements: dict[str, str]) -> str:
    if path.is_symlink() or not path.is_file():
        raise RenderProtectionError(f"template is missing or unsafe: {path}")
    rendered = path.read_text(encoding="utf-8")
    for name, value in replacements.items():
        rendered = rendered.replace(f"@{name}@", value)
    remaining = sorted(set(_PLACEHOLDER.findall(rendered)))
    if remaining:
        raise RenderProtectionError(
            f"template {path.name} has unresolved placeholders: {remaining}"
        )
    return rendered if rendered.endswith("\n") else rendered + "\n"


def _owner(timer: str, *, suffix: str) -> str:
    prefix = "edgeconnect-startrain-"
    if not timer.startswith(prefix) or not timer.endswith(suffix):
        raise RenderProtectionError(f"unsupported protection timer name: {timer}")
    owner = timer[len(prefix) : -len(suffix)]
    if not owner:
        raise RenderProtectionError(f"protection timer has no owner: {timer}")
    return owner


def _monitor_unit(
    manifest: ContinuityManifest,
    workload: Workload,
    *,
    user: str,
) -> str:
    protection = workload.protection
    assert protection is not None
    commands = workload_protection_commands(manifest, workload)
    command = commands[protection.telemetry_service]
    return (
        "[Unit]\n"
        f"Description=EdgeConnect StarTrain 5s monitor ({workload.workload_id})\n"
        "Wants=network-online.target\n"
        "After=network-online.target\n"
        f"ConditionPathIsDirectory={workload.run_root}\n"
        "\n"
        "[Service]\n"
        "Type=simple\n"
        f"User={user}\n"
        f"WorkingDirectory={workload.runtime_training_dir}\n"
        "Environment=PYTHONUNBUFFERED=1\n"
        "Environment=PYTHONDONTWRITEBYTECODE=1\n"
        f"Environment=PYTHONPATH={workload.runtime_training_dir}\n"
        f"ExecStart={command}\n"
        "Restart=always\n"
        "RestartSec=5\n"
        "TimeoutStopSec=30\n"
        "KillSignal=SIGTERM\n"
        "NoNewPrivileges=true\n"
        "UMask=0027\n"
        "ProtectSystem=strict\n"
        "ProtectHome=read-only\n"
        f"ReadWritePaths={workload.run_root}/status\n"
        "PrivateTmp=true\n"
        "ProtectClock=true\n"
        "ProtectControlGroups=true\n"
        "ProtectHostname=true\n"
        "ProtectKernelLogs=true\n"
        "ProtectKernelModules=true\n"
        "ProtectKernelTunables=true\n"
        "RestrictNamespaces=true\n"
        "RestrictRealtime=true\n"
        "RestrictSUIDSGID=true\n"
        "LockPersonality=true\n"
        "\n"
        "[Install]\n"
        "WantedBy=multi-user.target\n"
    )


def rendered_workload_protection_units(
    manifest_path: str | Path,
    workload_id: str,
    *,
    user: str,
    provisioned_gpus: int,
) -> dict[str, bytes]:
    source = Path(manifest_path).expanduser()
    manifest = load_continuity_manifest(source)
    if manifest.path != manifest.pinned_manifest_path:
        raise RenderProtectionError(
            "renderer requires the atomically pinned continuity manifest at "
            f"{manifest.pinned_manifest_path}"
        )
    if _USER.fullmatch(user) is None:
        raise RenderProtectionError("user must be a safe system account name")
    if (
        isinstance(provisioned_gpus, bool)
        or not isinstance(provisioned_gpus, int)
        or provisioned_gpus <= 0
    ):
        raise RenderProtectionError("provisioned GPUs must be a positive integer")
    verification = verify_workload(manifest, workload_id)
    workload = verification.workload
    protection = workload.protection
    if protection is None:
        raise RenderProtectionError(
            f"workload {workload_id} has no protection metadata"
        )
    template_root = Path(__file__).resolve().parents[1] / "deploy"
    templates = {
        name: template_root / filename for name, filename in _TEMPLATES.items()
    }
    replay_owner = _owner(
        protection.replay_backup_timer,
        suffix="-backup.timer",
    )
    disaster_owner = _owner(
        protection.disaster_backup_timer,
        suffix="-disaster-backup.timer",
    )
    report_service = (
        protection.report_service
        or f"edgeconnect-startrain-{verification.run_id}-report.service"
    )
    report_timer = (
        protection.report_timer
        or f"edgeconnect-startrain-{verification.run_id}-report.timer"
    )
    if (
        protection.report_provisioned_gpus is not None
        and protection.report_provisioned_gpus != provisioned_gpus
    ):
        raise RenderProtectionError(
            "rendered report GPU count differs from the pinned workload protection"
        )
    if protection.service_user is not None and protection.service_user != user:
        raise RenderProtectionError(
            "rendered service user differs from the pinned workload protection"
        )
    report_owner = _owner(report_service, suffix="-report.service")
    rendered = {
        protection.replay_backup_service: _template(
            templates["replay_service"],
            {
                "USER": user,
                "TRAINING_DIR": str(workload.runtime_training_dir),
                "RUN_ROOT": str(workload.run_root),
            },
        ),
        protection.replay_backup_timer: _template(
            templates["replay_timer"],
            {"RUN_ID": replay_owner},
        ),
        protection.disaster_backup_service: _template(
            templates["disaster_service"],
            {
                "USER": user,
                "TRAINING_DIR": str(workload.runtime_training_dir),
                "RUN_ROOT": str(workload.run_root),
                "PROFILE": str(workload.profile_path),
                "BACKUP_ROOT": str(protection.disaster_backup_root),
                "BACKUP_MOUNT": str(protection.disaster_backup_mount),
            },
        ),
        protection.disaster_backup_timer: _template(
            templates["disaster_timer"],
            {"BACKUP_ID": disaster_owner},
        ),
        report_service: _template(
            templates["report_service"],
            {
                "USER": user,
                "TRAINING_DIR": str(workload.runtime_training_dir),
                "RUN_ROOT": str(workload.run_root),
                "PROVISIONED_GPUS": str(provisioned_gpus),
            },
        ),
        report_timer: _template(
            templates["report_timer"],
            {"RUN_ID": report_owner},
        ),
        protection.telemetry_service: _monitor_unit(
            manifest,
            workload,
            user=user,
        ),
    }
    if len(rendered) != 7:
        raise RenderProtectionError("rendered protection unit names collide")
    return {name: content.encode("utf-8") for name, content in sorted(rendered.items())}


def _write_unit(path: Path, data: bytes, *, replace_existing: bool) -> None:
    temporary = path.parent / f".{path.name}.rendering"
    if temporary.exists() or temporary.is_symlink():
        raise RenderProtectionError(f"stale render temporary exists: {temporary}")
    descriptor = os.open(
        temporary,
        os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_NOFOLLOW", 0),
        0o644,
    )
    try:
        os.fchmod(descriptor, 0o644)
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(data)
            stream.flush()
            os.fsync(stream.fileno())
        if replace_existing:
            os.replace(temporary, path)
        else:
            os.link(temporary, path)
            temporary.unlink()
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise


def render_workload_protection_units(
    manifest_path: str | Path,
    workload_id: str,
    output_directory: str | Path,
    *,
    user: str,
    provisioned_gpus: int,
    replace_existing: bool = False,
) -> list[dict[str, object]]:
    rendered = rendered_workload_protection_units(
        manifest_path,
        workload_id,
        user=user,
        provisioned_gpus=provisioned_gpus,
    )
    output = Path(output_directory).expanduser().resolve()
    systemd_root = Path("/etc/systemd").resolve()
    if output == systemd_root or systemd_root in output.parents:
        raise RenderProtectionError(
            "refusing to render directly into the systemd installation tree"
        )
    if output.exists() and (output.is_symlink() or not output.is_dir()):
        raise RenderProtectionError(f"output directory is unsafe: {output}")
    output.mkdir(parents=True, exist_ok=True)
    targets = [output / name for name in rendered]
    for target in targets:
        if target.is_symlink() or (target.exists() and not target.is_file()):
            raise RenderProtectionError(f"output target is unsafe: {target}")
    existing = [target for target in targets if target.exists()]
    if existing and not replace_existing:
        raise RenderProtectionError(
            "refusing to overwrite rendered units without --replace-existing: "
            + ", ".join(path.name for path in existing)
        )
    for target in targets:
        _write_unit(
            target,
            rendered[target.name],
            replace_existing=replace_existing,
        )
    directory = os.open(output, os.O_RDONLY)
    try:
        os.fsync(directory)
    finally:
        os.close(directory)
    return [
        {
            "path": str(target),
            "bytes": len(rendered[target.name]),
            "sha256": hashlib.sha256(rendered[target.name]).hexdigest(),
        }
        for target in targets
    ]


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--workload", required=True)
    parser.add_argument("--output-directory", required=True, type=Path)
    parser.add_argument("--user", required=True)
    parser.add_argument("--provisioned-gpus", required=True, type=int)
    parser.add_argument(
        "--replace-existing",
        action="store_true",
        help="atomically replace existing regular rendered files",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    arguments = _parser().parse_args(argv)
    try:
        files = render_workload_protection_units(
            arguments.manifest,
            arguments.workload,
            arguments.output_directory,
            user=arguments.user,
            provisioned_gpus=arguments.provisioned_gpus,
            replace_existing=arguments.replace_existing,
        )
    except (ContinuityError, OSError, RenderProtectionError) as error:
        print(
            json.dumps(
                {
                    "status": "error",
                    "error_type": type(error).__name__,
                    "error": str(error),
                },
                sort_keys=True,
            ),
            file=sys.stderr,
        )
        return 2
    print(
        json.dumps(
            {"status": "rendered", "files": files},
            sort_keys=True,
            separators=(",", ":"),
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
