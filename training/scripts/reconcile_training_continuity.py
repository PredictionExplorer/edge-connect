#!/usr/bin/env python3
"""Verify, reconcile, or run one hash-pinned training continuity workload."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from startrain.continuity import (
    ContinuityBusyError,
    ContinuityError,
    ContinuitySplitBrainError,
    load_continuity_manifest,
    reconcile_training_continuity,
    run_locked_workload,
    verify_continuity_manifest,
    workload_fingerprints,
)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)

    fingerprint = commands.add_parser(
        "fingerprint",
        help="print immutable hashes for one initialized run",
    )
    fingerprint.add_argument("--profile", required=True, type=Path)
    fingerprint.add_argument("--run-root", required=True, type=Path)

    verify = commands.add_parser(
        "verify",
        help="fail closed unless the manifest and workload hashes match",
    )
    verify.add_argument("--manifest", required=True, type=Path)
    verification_scope = verify.add_mutually_exclusive_group()
    verification_scope.add_argument("--workload")
    verification_scope.add_argument(
        "--structure-only",
        action="store_true",
        help="validate manifest structure without requiring a failed primary",
    )

    reconcile = commands.add_parser(
        "reconcile",
        help="idempotently select and start the permitted workload",
    )
    reconcile.add_argument("--manifest", required=True, type=Path)
    reconcile.add_argument(
        "--hardware-report",
        type=Path,
        help="override the manifest report path for this reconciliation",
    )

    locked = commands.add_parser(
        "run-locked",
        help="hold the host GPU lock while execing one coordinator",
    )
    locked.add_argument("--manifest", required=True, type=Path)
    locked.add_argument("--workload", required=True)
    locked.add_argument("--orchestrator", required=True, type=Path)
    return parser


def _print(payload: object, *, stream=sys.stdout) -> None:
    print(json.dumps(payload, indent=2, sort_keys=True), file=stream)


def main(argv: list[str] | None = None) -> int:
    arguments = _parser().parse_args(argv)
    try:
        if arguments.command == "fingerprint":
            _print(
                workload_fingerprints(
                    arguments.profile,
                    arguments.run_root,
                )
            )
            return 0
        if arguments.command == "verify":
            manifest = load_continuity_manifest(arguments.manifest)
            if arguments.structure_only:
                _print(
                    {
                        "format": manifest.raw["format"],
                        "schema_version": manifest.raw["schema_version"],
                        "status": "ok",
                        "manifest": str(manifest.path),
                        "manifest_sha256": manifest.sha256,
                        "workload_ids": [
                            workload.workload_id for workload in manifest.workloads
                        ],
                    }
                )
            else:
                _print(
                    verify_continuity_manifest(
                        manifest,
                        workload_id=arguments.workload,
                    )
                )
            return 0
        if arguments.command == "reconcile":
            _print(
                reconcile_training_continuity(
                    arguments.manifest,
                    hardware_report_path=arguments.hardware_report,
                )
            )
            return 0
        if arguments.command == "run-locked":
            run_locked_workload(
                arguments.manifest,
                arguments.workload,
                orchestrator=arguments.orchestrator,
            )
            return 0
        raise AssertionError(f"unhandled command {arguments.command}")
    except ContinuityBusyError as exc:
        _print(
            {
                "status": "busy",
                "error_type": type(exc).__name__,
                "error": str(exc),
            },
            stream=sys.stderr,
        )
        return 75
    except ContinuitySplitBrainError as exc:
        _print(
            {
                "status": "blocked",
                "error_type": type(exc).__name__,
                "error": str(exc),
            },
            stream=sys.stderr,
        )
        return 78
    except ContinuityError as exc:
        _print(
            {
                "status": "error",
                "error_type": type(exc).__name__,
                "error": str(exc),
            },
            stream=sys.stderr,
        )
        return 2
    except Exception as exc:
        _print(
            {
                "status": "error",
                "error_type": type(exc).__name__,
                "error": str(exc),
            },
            stream=sys.stderr,
        )
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
