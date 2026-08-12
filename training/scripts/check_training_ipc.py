#!/usr/bin/env python3
"""Fail closed when logind may reap a training user's multiprocessing IPC."""

from __future__ import annotations

import argparse
import json
import pwd
import subprocess

SCHEMA_VERSION = 1
REPORT_NAME = "startrain-training-ipc-preflight"


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--user", required=True)
    return parser


def effective_remove_ipc(
    *,
    timeout_seconds: float = 10.0,
) -> bool:
    completed = subprocess.run(
        [
            "busctl",
            "--system",
            "get-property",
            "org.freedesktop.login1",
            "/org/freedesktop/login1",
            "org.freedesktop.login1.Manager",
            "RemoveIPC",
        ],
        check=False,
        capture_output=True,
        text=True,
        timeout=timeout_seconds,
    )
    if completed.returncode != 0:
        detail = completed.stderr.strip() or completed.stdout.strip()
        raise RuntimeError(f"cannot query logind RemoveIPC: {detail}")
    fields = completed.stdout.split()
    if fields == ["b", "true"]:
        return True
    if fields == ["b", "false"]:
        return False
    raise RuntimeError(f"unexpected logind RemoveIPC response: {completed.stdout!r}")


def check_training_ipc(user: str) -> dict[str, object]:
    try:
        account = pwd.getpwnam(user)
    except KeyError as error:
        raise ValueError(f"training user does not exist: {user}") from error
    remove_ipc = effective_remove_ipc()
    if remove_ipc and account.pw_uid >= 1_000:
        raise RuntimeError(
            "systemd-logind RemoveIPC=yes can delete DataLoader shared memory "
            f"when the final {user} login session closes"
        )
    return {
        "schema_version": SCHEMA_VERSION,
        "report": REPORT_NAME,
        "status": "passed",
        "user": user,
        "uid": account.pw_uid,
        "effective_remove_ipc": remove_ipc,
        "system_user_exempt": account.pw_uid < 1_000,
    }


def main(argv: list[str] | None = None) -> int:
    arguments = _parser().parse_args(argv)
    try:
        report = check_training_ipc(arguments.user)
    except (OSError, RuntimeError, ValueError, subprocess.SubprocessError) as error:
        print(
            json.dumps(
                {
                    "schema_version": SCHEMA_VERSION,
                    "report": REPORT_NAME,
                    "status": "failed",
                    "error": f"{type(error).__name__}: {error}",
                },
                sort_keys=True,
            )
        )
        return 2
    print(json.dumps(report, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
