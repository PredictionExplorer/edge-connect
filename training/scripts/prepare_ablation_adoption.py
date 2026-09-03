#!/usr/bin/env python3
"""Prepare an immutable canary adoption plan from pinned cross-seed evidence."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import re
import stat
import time
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import yaml

from startrain.config import CONFIG_SCHEMA_VERSION

if __package__:
    from .compare_elo_ablation_seeds import (
        REPORT_NAME as CROSS_SEED_REPORT,
        SCHEMA_VERSION,
        AdoptionPolicy,
        CrossSeedComparisonError,
        load_adoption_policy,
        sha256_file,
    )
    from .prepare_elo_ablation import verify_winner_snapshot
else:
    from compare_elo_ablation_seeds import (
        REPORT_NAME as CROSS_SEED_REPORT,
        SCHEMA_VERSION,
        AdoptionPolicy,
        CrossSeedComparisonError,
        load_adoption_policy,
        sha256_file,
    )
    from prepare_elo_ablation import verify_winner_snapshot

PLAN_REPORT = "startrain-elo-ablation-adoption-plan"
INELIGIBLE_REPORT = "startrain-elo-ablation-adoption-ineligible"
_SHA256_PATTERN = re.compile(r"[0-9a-f]{64}")


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--report-sha256", required=True)
    parser.add_argument("--policy", type=Path, required=True)
    parser.add_argument("--policy-sha256", required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser


def _reason(code: str, message: str) -> dict[str, str]:
    return {"code": code, "message": message}


def _read_report(
    path: Path,
    expected_sha256: str,
) -> tuple[Path, dict[str, Any] | None, str | None, list[dict[str, str]]]:
    resolved = path.expanduser().resolve()
    expected = expected_sha256.lower()
    reasons = []
    if not _SHA256_PATTERN.fullmatch(expected):
        return (
            resolved,
            None,
            None,
            [_reason("invalid_report_digest_pin", "cross-seed report pin is invalid")],
        )
    try:
        payload = resolved.read_bytes()
    except OSError as error:
        return (
            resolved,
            None,
            None,
            [_reason("report_unreadable", f"cannot read cross-seed report: {error}")],
        )
    actual = hashlib.sha256(payload).hexdigest()
    if actual != expected:
        reasons.append(
            _reason(
                "report_digest_mismatch",
                f"expected {expected}, observed {actual}",
            )
        )
        return resolved, None, actual, reasons
    try:
        loaded = json.loads(payload)
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        reasons.append(_reason("invalid_report_json", str(error)))
        return resolved, None, actual, reasons
    if not isinstance(loaded, dict):
        reasons.append(
            _reason("invalid_report_schema", "cross-seed report must be an object")
        )
        return resolved, None, actual, reasons
    return resolved, loaded, actual, reasons


def _mapping(
    value: object,
    name: str,
    reasons: list[dict[str, str]],
) -> Mapping[str, object] | None:
    if not isinstance(value, Mapping):
        reasons.append(_reason("invalid_report_schema", f"{name} must be an object"))
        return None
    return value


def _finite(value: object) -> float | None:
    if (
        isinstance(value, bool)
        or not isinstance(value, int | float)
        or not math.isfinite(float(value))
    ):
        return None
    return float(value)


def _paths_overlap(first: Path, second: Path) -> bool:
    return (
        first == second or first.is_relative_to(second) or second.is_relative_to(first)
    )


def _read_verified_profile(
    value: object,
    *,
    name: str,
    reasons: list[dict[str, str]],
) -> tuple[dict[str, str], Mapping[str, object]] | None:
    artifact = _mapping(value, name, reasons)
    if artifact is None:
        return None
    raw_path = artifact.get("path")
    expected = artifact.get("sha256")
    if not isinstance(raw_path, str) or not Path(raw_path).expanduser().is_absolute():
        reasons.append(_reason("invalid_artifact_path", f"{name} path is not absolute"))
        return None
    if not isinstance(expected, str) or not _SHA256_PATTERN.fullmatch(expected):
        reasons.append(_reason("invalid_artifact_digest", f"{name} digest is invalid"))
        return None
    path = Path(raw_path).expanduser().resolve()
    try:
        descriptor = os.open(path, os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0))
    except OSError as error:
        reasons.append(_reason("artifact_unreadable", f"{name}: {error}"))
        return None
    try:
        before = os.fstat(descriptor)
        if not stat.S_ISREG(before.st_mode):
            reasons.append(_reason("artifact_unreadable", f"{name} is not regular"))
            return None
        chunks = []
        while block := os.read(descriptor, 1024 * 1024):
            chunks.append(block)
        after = os.fstat(descriptor)
    finally:
        os.close(descriptor)
    if (
        before.st_dev,
        before.st_ino,
        before.st_size,
        before.st_mtime_ns,
    ) != (
        after.st_dev,
        after.st_ino,
        after.st_size,
        after.st_mtime_ns,
    ):
        reasons.append(_reason("artifact_changed", f"{name} changed while reading"))
        return None
    payload = b"".join(chunks)
    actual = hashlib.sha256(payload).hexdigest()
    if actual != expected:
        reasons.append(
            _reason(
                "artifact_digest_mismatch",
                f"{name}: expected {expected}, observed {actual}",
            )
        )
        return None
    try:
        loaded = yaml.safe_load(payload)
    except yaml.YAMLError as error:
        reasons.append(_reason("invalid_canary_profile", f"{name}: {error}"))
        return None
    if not isinstance(loaded, Mapping):
        reasons.append(_reason("invalid_canary_profile", f"{name} is not an object"))
        return None
    return {"path": str(path), "sha256": expected}, loaded


def _verify_source_comparisons(
    report: Mapping[str, object],
    policy: AdoptionPolicy,
    *,
    report_path: Path,
    reasons: list[dict[str, str]],
) -> list[dict[str, object]]:
    raw_sources = report.get("sources")
    if not isinstance(raw_sources, list):
        reasons.append(
            _reason("invalid_report_schema", "cross-seed sources must be a list")
        )
        return []
    verified = []
    seeds: list[int] = []
    paths: list[Path] = []
    digests: list[str] = []
    for index, raw_source in enumerate(raw_sources):
        source = _mapping(raw_source, f"source {index}", reasons)
        if source is None:
            continue
        seed = source.get("seed")
        raw_path = source.get("path")
        digest = source.get("sha256")
        if type(seed) is not int:
            reasons.append(_reason("invalid_source_seed", f"source {index} seed"))
            continue
        if (
            not isinstance(raw_path, str)
            or not Path(raw_path).expanduser().is_absolute()
        ):
            reasons.append(
                _reason("invalid_source_path", f"seed {seed} source path is invalid")
            )
            continue
        if not isinstance(digest, str) or not _SHA256_PATTERN.fullmatch(digest):
            reasons.append(
                _reason(
                    "invalid_source_digest",
                    f"seed {seed} source digest is invalid",
                )
            )
            continue
        path = Path(raw_path).expanduser().resolve()
        seeds.append(seed)
        paths.append(path)
        digests.append(digest)
        try:
            actual = sha256_file(path)
        except OSError as error:
            reasons.append(
                _reason(
                    "source_comparison_unreadable",
                    f"seed {seed} comparison: {error}",
                )
            )
            continue
        if actual != digest:
            reasons.append(
                _reason(
                    "source_comparison_stale",
                    f"seed {seed}: expected {digest}, observed {actual}",
                )
            )
            continue
        verified.append({"seed": seed, "path": str(path), "sha256": digest})
    if tuple(sorted(seeds)) != policy.seeds or len(set(seeds)) != len(seeds):
        reasons.append(
            _reason(
                "source_seed_set_mismatch",
                f"sources must cover exactly {list(policy.seeds)}",
            )
        )
    if len(set(paths)) != len(paths):
        reasons.append(_reason("duplicate_source_path", "source paths are duplicated"))
    if len(set(digests)) != len(digests):
        reasons.append(
            _reason("duplicate_source_digest", "source digests are duplicated")
        )
    if report_path in paths or policy.path in paths:
        reasons.append(
            _reason(
                "artifact_path_alias",
                "report, policy, and source comparison paths must be distinct",
            )
        )
    if policy.sha256 in digests:
        reasons.append(
            _reason(
                "artifact_digest_alias",
                "policy and source comparison digests must be distinct",
            )
        )

    def source_seed(source: Mapping[str, object]) -> int:
        seed = source.get("seed")
        if type(seed) is not int:
            raise AssertionError("verified source seed must be an integer")
        return seed

    return sorted(verified, key=source_seed)


def _report_snapshot(
    report: Mapping[str, object],
    policy: AdoptionPolicy,
    *,
    reasons: list[dict[str, str]],
) -> tuple[dict[str, object] | None, int | None]:
    selector = _mapping(report.get("selector"), "cross-seed selector", reasons)
    if selector is None:
        return None, None
    if selector.get("status") != "verified":
        reasons.append(
            _reason(
                "unverified_cross_seed_selector",
                "cross-seed selector has no verified winner",
            )
        )
        return None, None
    if (
        selector.get("candidate_label") != policy.candidate_treatment
        or selector.get("control_label") != policy.control_treatment
    ):
        reasons.append(
            _reason(
                "selector_label_mismatch",
                "cross-seed selector labels differ from policy",
            )
        )
    seed = selector.get("source_seed")
    if type(seed) is not int or seed not in policy.seeds:
        reasons.append(
            _reason("invalid_selector_seed", "selector source seed is invalid")
        )
        return None, None
    snapshot = _mapping(
        selector.get("winner_snapshot"),
        "cross-seed winner snapshot",
        reasons,
    )
    if snapshot is None:
        return None, seed
    serialized = json.loads(json.dumps(dict(snapshot), allow_nan=False))
    raw_root = serialized.get("run_root")
    if not isinstance(raw_root, str):
        reasons.append(
            _reason("invalid_winner_snapshot", "winner snapshot run root is invalid")
        )
        return None, seed
    try:
        verify_winner_snapshot(Path(raw_root), serialized)
    except (OSError, TypeError, ValueError) as error:
        reasons.append(_reason("stale_winner_snapshot", str(error)))
        return None, seed
    raw_per_seed = report.get("per_seed")
    matching = (
        [
            record
            for record in raw_per_seed
            if isinstance(record, Mapping) and record.get("seed") == seed
        ]
        if isinstance(raw_per_seed, list)
        else []
    )
    if len(matching) != 1:
        reasons.append(
            _reason(
                "selector_confirmation_missing",
                "selector seed is absent from per-seed evidence",
            )
        )
        return None, seed
    candidate = matching[0].get("candidate")
    candidate_snapshot = (
        candidate.get("verified_winner_snapshot")
        if isinstance(candidate, Mapping)
        else None
    )
    if candidate_snapshot != serialized:
        reasons.append(
            _reason(
                "selector_snapshot_mismatch",
                "selector snapshot differs from its seed confirmation",
            )
        )
        return None, seed
    return serialized, seed


def _targets(
    policy: AdoptionPolicy,
    report: Mapping[str, object],
    *,
    output_path: Path,
    reasons: list[dict[str, str]],
) -> dict[str, object] | None:
    canary = policy.canary
    future = canary.get("future_fresh_winner")
    previous = canary.get("previous_lkg")
    if not isinstance(future, Mapping) or not isinstance(previous, Mapping):
        reasons.append(
            _reason("invalid_canary_targets", "canary targets are not objects")
        )
        return None
    future_root = Path(str(future.get("run_root"))).expanduser().resolve()
    previous_root = Path(str(previous.get("run_root"))).expanduser().resolve()
    if future_root.exists():
        reasons.append(
            _reason(
                "canary_root_not_fresh",
                f"future canary root already exists: {future_root}",
            )
        )
    if not previous_root.is_dir():
        reasons.append(
            _reason(
                "previous_lkg_missing",
                f"previous LKG run root is unavailable: {previous_root}",
            )
        )
    if _paths_overlap(future_root, previous_root):
        reasons.append(
            _reason(
                "canary_lkg_root_alias",
                "future canary and previous LKG roots must not overlap",
            )
        )
    future_verified = _read_verified_profile(
        future.get("profile"),
        name="future canary profile",
        reasons=reasons,
    )
    previous_verified = _read_verified_profile(
        previous.get("profile"),
        name="previous LKG profile",
        reasons=reasons,
    )
    protected_roots = []
    raw_protected = report.get("protected_run_roots")
    if not isinstance(raw_protected, list) or not raw_protected:
        reasons.append(
            _reason(
                "protected_roots_missing",
                "cross-seed report does not enumerate protected treatment roots",
            )
        )
    else:
        for raw_root in raw_protected:
            if (
                not isinstance(raw_root, str)
                or not Path(raw_root).expanduser().is_absolute()
            ):
                reasons.append(
                    _reason(
                        "protected_root_invalid",
                        "cross-seed protected treatment root is invalid",
                    )
                )
                continue
            protected_roots.append(Path(raw_root).expanduser().resolve())
    for treatment_root in protected_roots:
        if _paths_overlap(future_root, treatment_root):
            reasons.append(
                _reason(
                    "canary_treatment_root_overlap",
                    "future canary root overlaps an immutable treatment root",
                )
            )
            break
    protected_outputs = [future_root, previous_root, *protected_roots]
    if any(_paths_overlap(output_path, root) for root in protected_outputs):
        reasons.append(
            _reason(
                "adoption_output_protected_root_overlap",
                "adoption output must be outside canary, LKG, and treatment roots",
            )
        )
    if future_verified is None or previous_verified is None:
        return None
    future_profile, future_document = future_verified
    previous_profile, previous_document = previous_verified
    for name, profile_document, expected_root in (
        ("future canary", future_document, future_root),
        ("previous LKG", previous_document, previous_root),
    ):
        orchestration = _mapping(
            profile_document.get("orchestration"),
            f"{name} orchestration",
            reasons,
        )
        learner = _mapping(
            profile_document.get("learner"),
            f"{name} learner",
            reasons,
        )
        arena = _mapping(
            profile_document.get("arena"),
            f"{name} arena",
            reasons,
        )
        if orchestration is None or learner is None or arena is None:
            continue
        directories = _mapping(
            orchestration.get("directories"),
            f"{name} directories",
            reasons,
        )
        if directories is None:
            continue
        raw_configured_root = directories.get("root")
        if not isinstance(raw_configured_root, str):
            reasons.append(
                _reason(
                    "invalid_canary_profile",
                    f"{name} profile run root is invalid",
                )
            )
            continue
        configured_root = Path(raw_configured_root).expanduser().resolve()
        if configured_root != expected_root:
            reasons.append(
                _reason(
                    "canary_profile_root_mismatch",
                    f"{name} profile targets {configured_root}, expected {expected_root}",
                )
            )
        if (
            profile_document.get("schema_version") != CONFIG_SCHEMA_VERSION
            or orchestration.get("enabled") is not True
            or orchestration.get("training_objective") != "ring10_only"
            or learner.get("unlimited") is not True
            or arena.get("rings") != [10]
        ):
            reasons.append(
                _reason(
                    "canary_profile_objective_mismatch",
                    f"{name} profile is not ring-10-only",
                )
            )
    return {
        "future_fresh_winner": {
            **{
                key: value
                for key, value in future.items()
                if key not in {"run_root", "profile"}
            },
            "run_root": str(future_root),
            "profile": future_profile,
            "root_must_be_fresh": True,
        },
        "previous_lkg": {
            **{
                key: value
                for key, value in previous.items()
                if key not in {"run_root", "profile"}
            },
            "run_root": str(previous_root),
            "profile": previous_profile,
        },
    }


def _validate_report(
    report: Mapping[str, object],
    policy: AdoptionPolicy,
    *,
    report_path: Path,
    output_path: Path,
    reasons: list[dict[str, str]],
) -> tuple[
    list[dict[str, object]],
    dict[str, object] | None,
    int | None,
    dict[str, object] | None,
]:
    if (
        report.get("schema_version") != SCHEMA_VERSION
        or report.get("report") != CROSS_SEED_REPORT
    ):
        reasons.append(
            _reason(
                "invalid_report_schema",
                "cross-seed report schema is unsupported",
            )
        )
    policy_artifact = _mapping(report.get("policy"), "report policy pin", reasons)
    if policy_artifact is not None:
        raw_policy_path = policy_artifact.get("path")
        if (
            not isinstance(raw_policy_path, str)
            or Path(raw_policy_path).expanduser().resolve() != policy.path
            or policy_artifact.get("sha256") != policy.sha256
        ):
            reasons.append(
                _reason(
                    "report_policy_mismatch",
                    "cross-seed report references a different policy",
                )
            )
    if report.get("source_commit") != policy.source_commit:
        reasons.append(
            _reason(
                "source_commit_mismatch",
                "cross-seed report source commit differs from policy",
            )
        )
    if (
        report.get("control_treatment") != policy.control_treatment
        or report.get("candidate_treatment") != policy.candidate_treatment
    ):
        reasons.append(
            _reason(
                "report_treatment_mismatch",
                "cross-seed report treatment labels differ from policy",
            )
        )
    sources = _verify_source_comparisons(
        report,
        policy,
        report_path=report_path,
        reasons=reasons,
    )
    aggregate = _mapping(report.get("aggregate"), "cross-seed aggregate", reasons)
    if aggregate is not None:
        median_improvement = _finite(
            aggregate.get("median_relative_point_elo_per_hour_improvement")
        )
        minimum_lcb = _finite(
            aggregate.get("minimum_per_seed_one_sided_lcb_advantage_elo_per_hour")
        )
        if (
            median_improvement is None
            or median_improvement < policy.minimum_median_improvement
        ):
            reasons.append(
                _reason(
                    "median_improvement_gate_failed",
                    "median point Elo/hour improvement is below policy",
                )
            )
        if minimum_lcb is None or minimum_lcb <= policy.minimum_lcb_advantage:
            reasons.append(
                _reason(
                    "lcb_advantage_gate_failed",
                    "minimum per-seed one-sided LCB advantage is not positive",
                )
            )
    gates = _mapping(report.get("gates"), "cross-seed gates", reasons)
    if gates is not None and any(
        not isinstance(gate, Mapping) or gate.get("passed") is not True
        for gate in gates.values()
    ):
        reasons.append(
            _reason(
                "cross_seed_gate_failed",
                "cross-seed report contains an unsatisfied gate",
            )
        )
    if report.get("status") != "eligible" or report.get("eligible") is not True:
        reasons.append(
            _reason(
                "cross_seed_ineligible",
                "cross-seed report does not authorize adoption planning",
            )
        )
    snapshot, source_seed = _report_snapshot(report, policy, reasons=reasons)
    targets = _targets(
        policy,
        report,
        output_path=output_path,
        reasons=reasons,
    )
    return sources, snapshot, source_seed, targets


def _write_immutable_json(path: Path, document: Mapping[str, object]) -> None:
    output = path.expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    if output.exists():
        raise FileExistsError(f"adoption output already exists: {output}")
    serialized = json.dumps(document, indent=2, sort_keys=True, allow_nan=False) + "\n"
    temporary = output.with_name(f".{output.name}.{os.getpid()}.{time.time_ns()}.tmp")
    descriptor = os.open(temporary, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
            descriptor = -1
            stream.write(serialized)
            stream.flush()
            os.fsync(stream.fileno())
        os.link(temporary, output)
        directory = os.open(output.parent, os.O_RDONLY)
        try:
            os.fsync(directory)
        finally:
            os.close(directory)
    finally:
        if descriptor >= 0:
            os.close(descriptor)
        temporary.unlink(missing_ok=True)


def prepare_ablation_adoption(
    *,
    report_path: Path,
    report_sha256: str,
    policy_path: Path,
    policy_sha256: str,
    output_path: Path,
) -> dict[str, object]:
    """Write exactly one immutable plan or durable ineligible decision."""
    output = output_path.expanduser().resolve()
    if output.exists():
        raise FileExistsError(f"adoption output already exists: {output}")
    report_resolved, report, observed_report_sha256, reasons = _read_report(
        report_path,
        report_sha256,
    )
    policy: AdoptionPolicy | None = None
    try:
        policy = load_adoption_policy(policy_path, policy_sha256)
    except (CrossSeedComparisonError, OSError, TypeError, ValueError) as error:
        reasons.append(_reason("invalid_or_stale_policy", str(error)))

    verified_sources: list[dict[str, object]] = []
    snapshot: dict[str, object] | None = None
    source_seed: int | None = None
    targets: dict[str, object] | None = None
    if report is not None and policy is not None:
        verified_sources, snapshot, source_seed, targets = _validate_report(
            report,
            policy,
            report_path=report_resolved,
            output_path=output,
            reasons=reasons,
        )

    unique_reasons = sorted(
        {json.dumps(reason, sort_keys=True): reason for reason in reasons}.values(),
        key=lambda reason: (reason["code"], reason["message"]),
    )
    if any(
        reason["code"] == "adoption_output_protected_root_overlap"
        for reason in unique_reasons
    ):
        raise ValueError(
            "adoption output must be outside canary, LKG, and treatment roots"
        )
    created_ns = time.time_ns()
    evidence = {
        "source_commit": policy.source_commit if policy is not None else None,
        "policy": {
            "path": str(policy.path if policy is not None else policy_path.resolve()),
            "sha256": policy.sha256 if policy is not None else policy_sha256.lower(),
            "verified": policy is not None,
        },
        "cross_seed_report": {
            "path": str(report_resolved),
            "sha256": (
                observed_report_sha256
                if observed_report_sha256 is not None
                else report_sha256.lower()
            ),
            "expected_sha256": report_sha256.lower(),
            "verified": report is not None,
        },
        "source_comparisons": verified_sources,
    }
    if unique_reasons or policy is None or report is None:
        document: dict[str, object] = {
            "schema_version": SCHEMA_VERSION,
            "report": INELIGIBLE_REPORT,
            "status": "ineligible",
            "eligible": False,
            "created_ns": created_ns,
            "evidence": evidence,
            "reasons": unique_reasons
            or [_reason("incomplete_evidence", "adoption evidence is incomplete")],
            "safety": {
                "adoption_plan_written": False,
                "run_roots_mutated": False,
                "system_mutation_performed": False,
                "automatic_adoption_authorized": False,
            },
        }
    else:
        assert snapshot is not None
        assert source_seed is not None
        assert targets is not None
        canary = policy.canary
        aggregate = report.get("aggregate")
        assert isinstance(aggregate, Mapping)
        document = {
            "schema_version": SCHEMA_VERSION,
            "report": PLAN_REPORT,
            "status": "eligible",
            "eligible": True,
            "immutable": True,
            "created_ns": created_ns,
            "source_commit": policy.source_commit,
            "policy": evidence["policy"],
            "cross_seed_report": evidence["cross_seed_report"],
            "source_comparisons": verified_sources,
            "candidate": {
                "label": policy.candidate_treatment,
                "control_label": policy.control_treatment,
                "winner_snapshot_source_seed": source_seed,
                "verified_winner_snapshot": snapshot,
                "aggregate": dict(aggregate),
            },
            "canary": {
                "status": "planned_not_started",
                "duration_hours": canary["duration_hours"],
                "operator_hold_path": canary["operator_hold_path"],
                "continuity_fallback_required": True,
                "targets": targets,
                "required_gates": [
                    {"name": gate, "required": True, "status": "pending"}
                    for gate in canary["required_gates"]
                ],
                "all_gates_required_before_adoption": True,
            },
            "safety": {
                "fresh_canary_root_required": True,
                "treatment_roots_are_immutable": True,
                "run_roots_mutated": False,
                "system_mutation_performed": False,
                "automatic_adoption_authorized": False,
                "operator_execution_required": True,
            },
        }
    _write_immutable_json(output, document)
    return document


def _error_document(error: Exception) -> dict[str, object]:
    return {
        "schema_version": SCHEMA_VERSION,
        "report": INELIGIBLE_REPORT,
        "status": "error",
        "eligible": False,
        "error": f"{type(error).__name__}: {error}",
    }


def main(argv: list[str] | None = None) -> int:
    arguments = _parser().parse_args(argv)
    try:
        report = prepare_ablation_adoption(
            report_path=arguments.report,
            report_sha256=arguments.report_sha256,
            policy_path=arguments.policy,
            policy_sha256=arguments.policy_sha256,
            output_path=arguments.output,
        )
    except (FileExistsError, OSError, TypeError, ValueError) as error:
        print(json.dumps(_error_document(error), sort_keys=True, allow_nan=False))
        return 2
    print(json.dumps(report, sort_keys=True, allow_nan=False))
    return 0 if report["eligible"] is True else 3


if __name__ == "__main__":
    raise SystemExit(main())
