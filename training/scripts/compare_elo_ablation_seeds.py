#!/usr/bin/env python3
"""Compare three hash-pinned Elo-ablation seed confirmations fail closed."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import re
import statistics
import time
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

if __package__:
    from .compare_elo_ablation import (
        ONE_SIDED_95_NORMAL_QUANTILE,
        REPORT_NAME as PER_SEED_REPORT,
    )
    from .prepare_elo_ablation import verify_winner_snapshot
else:
    from compare_elo_ablation import (
        ONE_SIDED_95_NORMAL_QUANTILE,
        REPORT_NAME as PER_SEED_REPORT,
    )
    from prepare_elo_ablation import verify_winner_snapshot

SCHEMA_VERSION = 1
REPORT_NAME = "startrain-elo-ablation-cross-seed-comparison"
POLICY_REPORT = "startrain-elo-ablation-adoption-policy"
REQUIRED_SEEDS = (17, 18, 19)
RANKING_OBJECTIVE = "ring_10_only"
TRAINING_OBJECTIVE = "ring10_only"
PROMOTION_OBJECTIVE = "ring_10_only"
RANKING_METRIC = "ring10_only_champion_frontier_elo_lcb_per_total_provisioned_wall_hour"
TIME_BASIS = "measurement_started_ns_to_resource_released_ns"
METRIC_SELECTION = "chronological_promotions_only"
LCB_GATE_METHOD = "minimum_per_seed_conservative_difference_lcb"
MINIMUM_POLICY_MEDIAN_IMPROVEMENT = 0.20
REQUIRED_CANARY_HOURS = 24.0
REQUIRED_CANARY_GATES = frozenset(
    {
        "duration_complete",
        "operator_hold_absent",
        "ring10_only_objective_preserved",
        "no_fatal_orchestrator_exit",
        "resource_release_confirmed",
        "integrity_verified",
        "continuity_fallback_ready",
        "previous_lkg_rollback_ready",
    }
)

_LABEL_PATTERN = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]*")
_SHA256_PATTERN = re.compile(r"[0-9a-f]{64}")
_COMMIT_PATTERN = re.compile(r"(?:[0-9a-f]{40}|[0-9a-f]{64})")


class CrossSeedComparisonError(ValueError):
    """Raised when an input artifact or policy cannot be trusted."""


class _ConfirmationError(ValueError):
    """Raised when a pinned per-seed document is semantically ineligible."""


@dataclass(frozen=True, slots=True)
class PinnedComparison:
    """One externally pinned per-seed comparison."""

    seed: int
    path: Path
    sha256: str


@dataclass(frozen=True, slots=True)
class AdoptionPolicy:
    """Validated immutable policy authority."""

    path: Path
    sha256: str
    document: dict[str, Any]
    seeds: tuple[int, ...]
    control_treatment: str
    candidate_treatment: str
    source_commit: str
    common_anchor_identity: str
    common_anchor_step: int
    minimum_median_improvement: float
    minimum_lcb_advantage: float
    canary: dict[str, Any]


@dataclass(frozen=True, slots=True)
class _Metric:
    point_score: float
    lower_bound_score: float
    conservative_standard_error_score: float
    total_provisioned_wall_hours: float
    winner_snapshot: dict[str, object]
    frontier: dict[str, object]


@dataclass(frozen=True, slots=True)
class _SeedConfirmation:
    seed: int
    source_commit: str
    anchor_identity: str
    anchor_step: int
    contract: dict[str, object]
    control: _Metric
    candidate: _Metric
    record: dict[str, object]


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--comparison",
        action="append",
        required=True,
        metavar="SEED=PATH",
        help="labeled per-seed comparison path; pass exactly three",
    )
    parser.add_argument(
        "--comparison-sha256",
        action="append",
        required=True,
        metavar="SEED=SHA256",
        help="external digest pin for each per-seed comparison",
    )
    parser.add_argument("--policy", type=Path, required=True)
    parser.add_argument("--policy-sha256", required=True)
    parser.add_argument("--output", type=Path)
    return parser


def sha256_file(path: Path) -> str:
    """Return a streaming SHA-256 digest."""
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _write_immutable_json(path: Path, document: Mapping[str, object]) -> None:
    output = path.expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    if output.exists():
        raise FileExistsError(f"cross-seed output already exists: {output}")
    temporary = output.with_name(f".{output.name}.{os.getpid()}.{time.time_ns()}.tmp")
    serialized = json.dumps(document, indent=2, sort_keys=True, allow_nan=False) + "\n"
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


def _read_pinned_json(
    path: Path,
    expected_sha256: str,
    *,
    name: str,
) -> tuple[Path, dict[str, Any]]:
    resolved = path.expanduser().resolve()
    expected = expected_sha256.lower()
    if not _SHA256_PATTERN.fullmatch(expected):
        raise CrossSeedComparisonError(f"{name} SHA-256 pin is invalid")
    try:
        payload = resolved.read_bytes()
    except OSError as error:
        raise CrossSeedComparisonError(
            f"cannot read {name} {resolved}: {error}"
        ) from error
    actual = hashlib.sha256(payload).hexdigest()
    if actual != expected:
        raise CrossSeedComparisonError(
            f"{name} digest mismatch: expected {expected}, observed {actual}"
        )
    try:
        loaded = json.loads(payload)
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise CrossSeedComparisonError(
            f"{name} is not valid UTF-8 JSON: {error}"
        ) from error
    if not isinstance(loaded, dict):
        raise CrossSeedComparisonError(f"{name} must contain a JSON object")
    return resolved, loaded


def _read_pinned_yaml(
    path: Path,
    expected_sha256: str,
    *,
    name: str,
) -> tuple[Path, Mapping[str, object]]:
    resolved = path.expanduser().resolve()
    expected = expected_sha256.lower()
    if not _SHA256_PATTERN.fullmatch(expected):
        raise CrossSeedComparisonError(f"{name} SHA-256 pin is invalid")
    try:
        payload = resolved.read_bytes()
    except OSError as error:
        raise CrossSeedComparisonError(
            f"cannot read {name} {resolved}: {error}"
        ) from error
    actual = hashlib.sha256(payload).hexdigest()
    if actual != expected:
        raise CrossSeedComparisonError(
            f"{name} digest mismatch: expected {expected}, observed {actual}"
        )
    try:
        loaded = yaml.safe_load(payload)
    except yaml.YAMLError as error:
        raise CrossSeedComparisonError(f"{name} is not valid YAML: {error}") from error
    if not isinstance(loaded, Mapping):
        raise CrossSeedComparisonError(f"{name} must contain a YAML object")
    return resolved, loaded


def _mapping(value: object, name: str) -> Mapping[str, object]:
    if not isinstance(value, Mapping):
        raise _ConfirmationError(f"{name} must be an object")
    return value


def _list(value: object, name: str) -> list[object]:
    if not isinstance(value, list):
        raise _ConfirmationError(f"{name} must be a list")
    return value


def _string(value: object, name: str) -> str:
    if not isinstance(value, str) or not value:
        raise _ConfirmationError(f"{name} must be a non-empty string")
    return value


def _finite(value: object, name: str) -> float:
    if (
        isinstance(value, bool)
        or not isinstance(value, int | float)
        or not math.isfinite(float(value))
    ):
        raise _ConfirmationError(f"{name} must be finite")
    return float(value)


def _positive(value: object, name: str) -> float:
    parsed = _finite(value, name)
    if parsed <= 0:
        raise _ConfirmationError(f"{name} must be positive")
    return parsed


def _positive_timestamp(value: object, name: str) -> int:
    if type(value) is not int or value <= 0:
        raise _ConfirmationError(f"{name} must be a positive integer timestamp")
    return value


def _artifact_spec(value: object, name: str) -> dict[str, str]:
    artifact = _mapping(value, name)
    path = _string(artifact.get("path"), f"{name} path")
    digest = _string(artifact.get("sha256"), f"{name} SHA-256").lower()
    if not Path(path).expanduser().is_absolute():
        raise CrossSeedComparisonError(f"{name} path must be absolute")
    if not _SHA256_PATTERN.fullmatch(digest):
        raise CrossSeedComparisonError(f"{name} SHA-256 is invalid")
    return {"path": str(Path(path).expanduser().resolve()), "sha256": digest}


def load_adoption_policy(
    policy_path: Path,
    expected_sha256: str,
) -> AdoptionPolicy:
    """Load and strictly validate the hash-pinned cross-seed authority."""
    resolved, document = _read_pinned_json(
        policy_path,
        expected_sha256,
        name="adoption policy",
    )
    try:
        if (
            document.get("schema_version") != SCHEMA_VERSION
            or document.get("report") != POLICY_REPORT
        ):
            raise _ConfirmationError("adoption policy schema is unsupported")
        raw_seeds = _list(document.get("seeds"), "adoption policy seeds")
        parsed_seeds = []
        for seed in raw_seeds:
            if type(seed) is not int:
                raise _ConfirmationError("adoption policy seeds must be integers")
            parsed_seeds.append(seed)
        seeds = tuple(parsed_seeds)
        if len(set(seeds)) != len(seeds):
            raise _ConfirmationError("adoption policy seeds contain duplicates")
        if tuple(sorted(seeds)) != REQUIRED_SEEDS:
            raise _ConfirmationError(
                f"adoption policy seed set must be exactly {list(REQUIRED_SEEDS)}"
            )
        control = _string(
            document.get("control_treatment"),
            "adoption policy control treatment",
        )
        candidate = _string(
            document.get("candidate_treatment"),
            "adoption policy candidate treatment",
        )
        if not _LABEL_PATTERN.fullmatch(control) or not _LABEL_PATTERN.fullmatch(
            candidate
        ):
            raise _ConfirmationError("policy treatment labels are invalid")
        if control == candidate:
            raise _ConfirmationError("control and candidate treatments must differ")
        if document.get("common_objective") != TRAINING_OBJECTIVE:
            raise _ConfirmationError(
                f"adoption policy common objective must be {TRAINING_OBJECTIVE!r}"
            )
        common_anchor = _mapping(
            document.get("common_anchor"),
            "adoption policy common anchor",
        )
        common_anchor_identity = _string(
            common_anchor.get("identity"),
            "adoption policy common anchor identity",
        )
        common_anchor_step = common_anchor.get("step")
        if type(common_anchor_step) is not int or common_anchor_step < 0:
            raise _ConfirmationError(
                "adoption policy common anchor step must be non-negative"
            )
        source_commit = _string(
            document.get("source_commit"),
            "adoption policy source commit",
        ).lower()
        if not _COMMIT_PATTERN.fullmatch(source_commit):
            raise _ConfirmationError(
                "adoption policy source commit must be a full hexadecimal commit"
            )
        minimum_improvement = _finite(
            document.get("minimum_median_point_elo_per_hour_improvement"),
            "minimum median point Elo/hour improvement",
        )
        if minimum_improvement < MINIMUM_POLICY_MEDIAN_IMPROVEMENT:
            raise _ConfirmationError(
                "adoption policy cannot weaken the 20% median improvement floor"
            )
        lcb_gate = _mapping(
            document.get("cross_seed_lcb_gate"),
            "cross-seed LCB gate",
        )
        if lcb_gate.get("method") != LCB_GATE_METHOD:
            raise _ConfirmationError("cross-seed LCB gate method is unsupported")
        if lcb_gate.get("require_strictly_positive") is not True:
            raise _ConfirmationError(
                "cross-seed LCB gate must require a strictly positive advantage"
            )
        minimum_lcb = _finite(
            lcb_gate.get("minimum_advantage_elo_per_hour"),
            "cross-seed minimum LCB advantage",
        )
        if minimum_lcb < 0:
            raise _ConfirmationError(
                "cross-seed minimum LCB advantage cannot be negative"
            )
        raw_eliminations = _list(
            document.get("screening_eliminations", []),
            "adoption policy screening eliminations",
        )
        elimination_labels: set[str] = set()
        for index, raw_elimination in enumerate(raw_eliminations):
            elimination = _mapping(
                raw_elimination,
                f"screening elimination {index}",
            )
            if set(elimination) != {"label", "reason"}:
                raise _ConfirmationError(
                    "screening eliminations require exactly label and reason"
                )
            elimination_label = _string(
                elimination.get("label"),
                f"screening elimination {index} label",
            )
            _string(
                elimination.get("reason"),
                f"screening elimination {index} reason",
            )
            if (
                not _LABEL_PATTERN.fullmatch(elimination_label)
                or elimination_label in {control, candidate}
                or elimination_label in elimination_labels
            ):
                raise _ConfirmationError(
                    "screening elimination labels are invalid or duplicated"
                )
            elimination_labels.add(elimination_label)
        canary_raw = _mapping(document.get("canary"), "adoption policy canary")
        duration = _finite(canary_raw.get("duration_hours"), "canary duration")
        if duration != REQUIRED_CANARY_HOURS:
            raise _ConfirmationError("adoption policy must require a 24h canary")
        if canary_raw.get("continuity_fallback_required") is not True:
            raise _ConfirmationError("adoption policy must require continuity fallback")
        hold_path = _string(
            canary_raw.get("operator_hold_path"),
            "canary operator hold path",
        )
        if not Path(hold_path).expanduser().is_absolute():
            raise _ConfirmationError("canary operator hold path must be absolute")
        raw_gates = _list(canary_raw.get("required_gates"), "canary required gates")
        if any(not isinstance(gate, str) or not gate for gate in raw_gates):
            raise _ConfirmationError("canary required gates must be non-empty strings")
        gates = tuple(str(gate) for gate in raw_gates)
        if len(set(gates)) != len(gates):
            raise _ConfirmationError("canary required gates contain duplicates")
        missing_gates = sorted(REQUIRED_CANARY_GATES - set(gates))
        if missing_gates:
            raise _ConfirmationError(
                f"canary policy is missing required gates: {missing_gates}"
            )
        future = _mapping(
            canary_raw.get("future_fresh_winner"),
            "future fresh winner canary target",
        )
        previous = _mapping(canary_raw.get("previous_lkg"), "previous LKG target")
        for target_name, target in (
            ("future fresh winner", future),
            ("previous LKG", previous),
        ):
            run_root = _string(target.get("run_root"), f"{target_name} run root")
            if not Path(run_root).expanduser().is_absolute():
                raise _ConfirmationError(f"{target_name} run root must be absolute")
            _artifact_spec(target.get("profile"), f"{target_name} profile")
        canary = json.loads(json.dumps(dict(canary_raw), allow_nan=False))
    except _ConfirmationError as error:
        raise CrossSeedComparisonError(str(error)) from error
    return AdoptionPolicy(
        path=resolved,
        sha256=expected_sha256.lower(),
        document=document,
        seeds=tuple(sorted(seeds)),
        control_treatment=control,
        candidate_treatment=candidate,
        source_commit=source_commit,
        common_anchor_identity=common_anchor_identity,
        common_anchor_step=common_anchor_step,
        minimum_median_improvement=minimum_improvement,
        minimum_lcb_advantage=minimum_lcb,
        canary=canary,
    )


def _validate_pins(
    comparisons: Sequence[PinnedComparison],
    policy: AdoptionPolicy,
) -> list[PinnedComparison]:
    if len(comparisons) != len(REQUIRED_SEEDS):
        raise CrossSeedComparisonError(
            f"exactly three per-seed comparisons are required: {list(REQUIRED_SEEDS)}"
        )
    seeds = [comparison.seed for comparison in comparisons]
    if any(type(seed) is not int for seed in seeds):
        raise CrossSeedComparisonError("comparison seeds must be integers")
    if len(set(seeds)) != len(seeds):
        raise CrossSeedComparisonError("duplicate comparison seed")
    if tuple(sorted(seeds)) != policy.seeds:
        raise CrossSeedComparisonError(
            f"comparison seed set must exactly match policy seeds {list(policy.seeds)}"
        )
    normalized = [
        PinnedComparison(
            seed=comparison.seed,
            path=comparison.path.expanduser().resolve(),
            sha256=comparison.sha256.lower(),
        )
        for comparison in comparisons
    ]
    if any(
        not _SHA256_PATTERN.fullmatch(comparison.sha256) for comparison in normalized
    ):
        raise CrossSeedComparisonError("comparison SHA-256 pin is invalid")
    paths = [comparison.path for comparison in normalized]
    digests = [comparison.sha256 for comparison in normalized]
    if len(set(paths)) != len(paths):
        raise CrossSeedComparisonError("duplicate comparison path")
    if len(set(digests)) != len(digests):
        raise CrossSeedComparisonError("duplicate comparison digest")
    if policy.path in paths:
        raise CrossSeedComparisonError(
            "adoption policy and comparison paths must be distinct"
        )
    if policy.sha256 in digests:
        raise CrossSeedComparisonError(
            "adoption policy and comparison digests must be distinct"
        )
    return sorted(normalized, key=lambda comparison: comparison.seed)


def _validate_queue_deployment(
    queue: Mapping[str, object],
    *,
    comparison_path: Path,
    seed: int,
    source_commit: str,
    treatment_roots: Mapping[str, Path],
) -> None:
    raw_manifest = queue.get("manifest")
    manifest_sha256 = queue.get("manifest_sha256")
    if (
        not isinstance(raw_manifest, str)
        or not Path(raw_manifest).expanduser().is_absolute()
        or not isinstance(manifest_sha256, str)
    ):
        raise _ConfirmationError("per-seed queue deployment pin is missing")
    try:
        _manifest_path, manifest = _read_pinned_json(
            Path(raw_manifest),
            manifest_sha256,
            name=f"seed {seed} deployment manifest",
        )
    except CrossSeedComparisonError as error:
        raise _ConfirmationError(str(error)) from error
    if (
        manifest.get("schema_version") != SCHEMA_VERSION
        or manifest.get("report") != "startrain-elo-ablation-deployment"
    ):
        raise _ConfirmationError("per-seed deployment manifest is unsupported")
    manifest_source = _mapping(manifest.get("source"), "deployment source")
    if manifest_source.get("commit") != source_commit:
        raise _ConfirmationError("deployment source commit disagrees")
    manifest_queue = _mapping(manifest.get("queue"), "deployment queue")
    if manifest_queue.get("seed") != seed:
        raise _ConfirmationError("deployment queue seed disagrees")
    raw_output = _string(
        manifest_queue.get("comparison_output"),
        "deployment comparison output",
    )
    if Path(raw_output).expanduser().resolve() != comparison_path:
        raise _ConfirmationError("deployment comparison output path disagrees")
    plan_artifact = _mapping(manifest.get("plan"), "deployment plan artifact")
    plan_path = Path(
        _string(plan_artifact.get("path"), "deployment plan path")
    ).expanduser()
    plan_sha256 = _string(plan_artifact.get("sha256"), "deployment plan digest")
    try:
        _resolved_plan, plan = _read_pinned_json(
            plan_path,
            plan_sha256,
            name=f"seed {seed} ablation plan",
        )
    except CrossSeedComparisonError as error:
        raise _ConfirmationError(str(error)) from error
    if plan.get("seed") != seed:
        raise _ConfirmationError("pinned ablation plan seed disagrees")
    plan_treatments = _list(plan.get("treatments"), "pinned plan treatments")
    plan_entries: dict[str, Mapping[str, object]] = {}
    for index, raw_treatment in enumerate(plan_treatments):
        treatment = _mapping(raw_treatment, f"pinned plan treatment {index}")
        label = _string(
            treatment.get("treatment"),
            f"pinned plan treatment {index} label",
        )
        if label in plan_entries:
            raise _ConfirmationError("pinned plan treatments contain duplicates")
        plan_entries[label] = treatment
    if set(plan_entries) != set(treatment_roots):
        raise _ConfirmationError("pinned plan treatment coverage disagrees")
    profiles = _list(manifest.get("profiles"), "deployment profiles")
    if not profiles:
        raise _ConfirmationError("deployment profiles are missing")
    profile_labels: set[str] = set()
    for index, raw_profile in enumerate(profiles):
        profile = _mapping(raw_profile, f"deployment profile {index}")
        label = _string(
            profile.get("treatment"),
            f"deployment profile {index} treatment",
        )
        if label in profile_labels or label not in treatment_roots:
            raise _ConfirmationError("deployment profile treatment coverage disagrees")
        profile_labels.add(label)
        plan_entry = plan_entries[label]
        plan_run_root = (
            Path(_string(plan_entry.get("run_root"), f"{label} plan run root"))
            .expanduser()
            .resolve()
        )
        if plan_run_root != treatment_roots[label]:
            raise _ConfirmationError(f"{label} plan run root disagrees")
        plan_profile = _mapping(
            profile.get("plan_profile"),
            f"deployment profile {index} plan artifact",
        )
        if plan_profile.get("path") != plan_entry.get("profile") or plan_profile.get(
            "sha256"
        ) != plan_entry.get("profile_sha256"):
            raise _ConfirmationError(f"{label} plan profile artifact disagrees")
        seed_contract = _mapping(
            profile.get("seed_contract"),
            f"deployment profile {index} seed contract",
        )
        if dict(seed_contract) != {
            "train_seed": seed,
            "selfplay_seed": seed,
            "arena_seed": seed,
        }:
            raise _ConfirmationError(
                f"deployment profile {index} seed contract disagrees"
            )
        profile_artifact = _mapping(
            profile.get("profile"),
            f"deployment profile {index} artifact",
        )
        try:
            profile_path, profile_document = _read_pinned_yaml(
                Path(
                    _string(
                        profile_artifact.get("path"),
                        f"deployment profile {index} path",
                    )
                ),
                _string(
                    profile_artifact.get("sha256"),
                    f"deployment profile {index} digest",
                ),
                name=f"seed {seed} treatment {label} profile",
            )
        except CrossSeedComparisonError as error:
            raise _ConfirmationError(str(error)) from error
        orchestration = _mapping(
            profile_document.get("orchestration"),
            f"deployment profile {index} orchestration",
        )
        directories = _mapping(
            orchestration.get("directories"),
            f"deployment profile {index} directories",
        )
        configured_root = (
            Path(
                _string(
                    directories.get("root"),
                    f"deployment profile {index} run root",
                )
            )
            .expanduser()
            .resolve()
        )
        if configured_root != treatment_roots[label]:
            raise _ConfirmationError(f"deployment profile {index} run root disagrees")
        train = _mapping(
            profile_document.get("train"),
            f"deployment profile {index} train",
        )
        arena = _mapping(
            profile_document.get("arena"),
            f"deployment profile {index} arena",
        )
        selfplay = _mapping(
            profile_document.get("selfplay"),
            f"deployment profile {index} selfplay",
        )
        if (
            train.get("seed") != seed
            or selfplay.get("seed") != seed
            or arena.get("seed") != seed
        ):
            raise _ConfirmationError(
                f"deployment profile {index} parsed seed disagrees"
            )
        installed_path = (
            Path(
                _string(
                    profile_artifact.get("path"),
                    f"deployment profile {index} path",
                )
            )
            .expanduser()
            .resolve()
        )
        if installed_path != profile_path:
            raise _ConfirmationError(
                f"deployment profile {index} path resolution disagrees"
            )
        if (
            profile_artifact.get("sha256") != plan_entry.get("profile_sha256")
            or installed_path != treatment_roots[label] / "profile-elo-ablation.yaml"
        ):
            raise _ConfirmationError(
                f"deployment profile {index} installed artifact disagrees"
            )
    if profile_labels != set(treatment_roots):
        raise _ConfirmationError("deployment profile coverage is incomplete")


def _measurement_evidence(
    treatment: Mapping[str, object],
    *,
    label: str,
) -> tuple[float, int, int, int]:
    measurement = _mapping(treatment.get("measurement"), f"{label} measurement")
    if (
        measurement.get("source") != "ablation.json"
        or measurement.get("status") != "complete"
    ):
        raise _ConfirmationError(
            f"{label} measurement is not a complete fixed-budget ablation"
        )
    started_ns = _positive_timestamp(
        measurement.get("started_ns"),
        f"{label} measurement start",
    )
    cutoff_ns = _positive_timestamp(
        measurement.get("cutoff_ns"),
        f"{label} measurement cutoff",
    )
    released_ns = _positive_timestamp(
        measurement.get("resource_released_ns"),
        f"{label} resource release",
    )
    if not started_ns <= cutoff_ns <= released_ns:
        raise _ConfirmationError(f"{label} measurement lifecycle is out of order")
    accounting = _mapping(
        treatment.get("resource_accounting"),
        f"{label} resource accounting",
    )
    if (
        accounting.get("started_ns") != started_ns
        or accounting.get("measurement_cutoff_ns") != cutoff_ns
        or accounting.get("resource_released_ns") != released_ns
    ):
        raise _ConfirmationError(f"{label} resource accounting timestamps disagree")
    hours = _positive(
        accounting.get("total_provisioned_wall_hours"),
        f"{label} total provisioned wall hours",
    )
    observed_hours = (released_ns - started_ns) / 3_600_000_000_000
    if not math.isclose(hours, observed_hours, rel_tol=1e-12, abs_tol=1e-12):
        raise _ConfirmationError(f"{label} total provisioned wall hours disagree")
    return hours, started_ns, cutoff_ns, released_ns


def _validated_snapshot(
    treatment: Mapping[str, object],
    *,
    label: str,
    frontier: Mapping[str, object],
    verify_live: bool,
) -> dict[str, object]:
    snapshot = _mapping(
        treatment.get("verified_winner_snapshot"),
        f"{label} verified winner snapshot",
    )
    if (
        snapshot.get("schema_version") != SCHEMA_VERSION
        or snapshot.get("status") != "verified"
        or snapshot.get("label") != label
    ):
        raise _ConfirmationError(f"{label} winner snapshot is not verified")
    run_root = (
        Path(_string(snapshot.get("run_root"), f"{label} winner run root"))
        .expanduser()
        .resolve()
    )
    if (
        run_root
        != Path(_string(treatment.get("run_root"), f"{label} treatment run root"))
        .expanduser()
        .resolve()
    ):
        raise _ConfirmationError(f"{label} winner snapshot run root disagrees")
    champion = _mapping(snapshot.get("champion"), f"{label} winner champion")
    if champion.get("model_identity") != frontier.get("identity") or champion.get(
        "model_step"
    ) != frontier.get("step"):
        raise _ConfirmationError(
            f"{label} winner snapshot is not the chronological champion frontier"
        )
    serialized = json.loads(json.dumps(dict(snapshot), allow_nan=False))
    if verify_live:
        try:
            verify_winner_snapshot(run_root, serialized)
        except (OSError, TypeError, ValueError) as error:
            raise _ConfirmationError(
                f"{label} winner snapshot verification failed: {error}"
            ) from error
    return serialized


def _treatment_metric(
    treatment: Mapping[str, object],
    *,
    label: str,
    normal_quantile: float,
    verify_snapshot_live: bool,
) -> _Metric:
    if (
        treatment.get("label") != label
        or treatment.get("eligible") is not True
        or treatment.get("status") != "eligible"
    ):
        raise _ConfirmationError(f"{label} is not an eligible source treatment")
    if (
        treatment.get("training_objective") != TRAINING_OBJECTIVE
        or treatment.get("promotion_objective") != PROMOTION_OBJECTIVE
        or treatment.get("ranking_objective") != RANKING_OBJECTIVE
    ):
        raise _ConfirmationError(f"{label} objective contract is incompatible")
    hours, _started_ns, _cutoff_ns, _released_ns = _measurement_evidence(
        treatment,
        label=label,
    )
    metric = _mapping(treatment.get("deployment_metric"), f"{label} deployment metric")
    if (
        metric.get("name") != RANKING_METRIC
        or metric.get("objective") != TRAINING_OBJECTIVE
        or metric.get("time_basis") != TIME_BASIS
        or metric.get("selection") != METRIC_SELECTION
    ):
        raise _ConfirmationError(f"{label} deployment metric contract is incompatible")
    metric_hours = _positive(
        metric.get("total_provisioned_wall_hours"),
        f"{label} metric wall hours",
    )
    if not math.isclose(metric_hours, hours, rel_tol=1e-12, abs_tol=1e-12):
        raise _ConfirmationError(f"{label} metric wall time disagrees")
    point_score = _finite(metric.get("point_value"), f"{label} point score")
    lower_score = _finite(metric.get("value"), f"{label} LCB score")
    gain = _finite(
        metric.get("champion_frontier_ring_10_elo_gained"),
        f"{label} frontier Elo gain",
    )
    standard_error = _finite(
        metric.get("champion_frontier_ring_10_elo_gain_conservative_standard_error"),
        f"{label} frontier conservative standard error",
    )
    if standard_error < 0:
        raise _ConfirmationError(
            f"{label} frontier conservative standard error is negative"
        )
    lower_gain = _finite(
        metric.get("champion_frontier_ring_10_elo_one_sided_95_lower_bound"),
        f"{label} frontier Elo lower bound",
    )
    expected_lower_gain = gain - normal_quantile * standard_error
    for observed, expected, field in (
        (point_score, gain / hours, "point score"),
        (lower_score, lower_gain / hours, "LCB score"),
        (lower_gain, expected_lower_gain, "frontier lower bound"),
    ):
        if not math.isclose(observed, expected, rel_tol=1e-9, abs_tol=1e-9):
            raise _ConfirmationError(f"{label} {field} is internally inconsistent")
    if lower_score > point_score:
        raise _ConfirmationError(f"{label} LCB score exceeds its point score")
    frontier = _mapping(
        treatment.get("champion_frontier"),
        f"{label} champion frontier",
    )
    frontier_identity = _string(
        frontier.get("identity"),
        f"{label} champion frontier identity",
    )
    frontier_step = frontier.get("step")
    if type(frontier_step) is not int or frontier_step < 0:
        raise _ConfirmationError(f"{label} champion frontier step is invalid")
    if not frontier_identity:
        raise _ConfirmationError(f"{label} champion frontier identity is invalid")
    anchor = _mapping(treatment.get("anchor"), f"{label} anchor")
    anchor_rating = _finite(anchor.get("rating_elo"), f"{label} anchor rating")
    anchor_standard_error = _finite(
        anchor.get("standard_error_elo"),
        f"{label} anchor standard error",
    )
    frontier_rating = _finite(
        frontier.get("rating_elo"),
        f"{label} frontier rating",
    )
    frontier_standard_error = _finite(
        frontier.get("standard_error_elo"),
        f"{label} frontier standard error",
    )
    if anchor_standard_error < 0 or frontier_standard_error < 0:
        raise _ConfirmationError(f"{label} anchor/frontier uncertainty is negative")
    recomputed_gain = frontier_rating - anchor_rating
    recomputed_standard_error = anchor_standard_error + frontier_standard_error
    recomputed_lower_gain = (
        recomputed_gain - normal_quantile * recomputed_standard_error
    )
    for observed, expected, field in (
        (gain, recomputed_gain, "frontier gain"),
        (standard_error, recomputed_standard_error, "frontier standard error"),
        (lower_gain, recomputed_lower_gain, "frontier lower bound"),
    ):
        if not math.isclose(observed, expected, rel_tol=1e-9, abs_tol=1e-9):
            raise _ConfirmationError(
                f"{label} {field} disagrees with anchor/frontier evidence"
            )
    snapshot = _validated_snapshot(
        treatment,
        label=label,
        frontier=frontier,
        verify_live=verify_snapshot_live,
    )
    return _Metric(
        point_score=point_score,
        lower_bound_score=lower_score,
        conservative_standard_error_score=standard_error / hours,
        total_provisioned_wall_hours=hours,
        winner_snapshot=snapshot,
        frontier=json.loads(json.dumps(dict(frontier), allow_nan=False)),
    )


def _source_contract(
    document: Mapping[str, object],
    *,
    provisioned_gpus: int,
    normal_quantile: float,
) -> dict[str, object]:
    guardrails = _mapping(
        document.get("guardrail_configuration"),
        "guardrail configuration",
    )
    rings = _list(guardrails.get("rings"), "guardrail rings")
    if rings:
        raise _ConfirmationError(
            "ring10-only confirmation cannot configure guard rings"
        )
    return {
        "ranking_objective": document.get("ranking_objective"),
        "ranking_metric": document.get("ranking_metric"),
        "training_objective": TRAINING_OBJECTIVE,
        "promotion_objective": PROMOTION_OBJECTIVE,
        "guard_rings": [],
        "guard_floor_elo": guardrails.get("floor_elo"),
        "provisioned_gpus": provisioned_gpus,
        "time_basis": TIME_BASIS,
        "metric_selection": METRIC_SELECTION,
        "source_one_sided_normal_quantile": normal_quantile,
    }


def _validate_confirmation(
    *,
    seed: int,
    comparison_path: Path,
    document: Mapping[str, object],
    policy: AdoptionPolicy,
) -> _SeedConfirmation:
    if (
        document.get("schema_version") != SCHEMA_VERSION
        or document.get("report") != PER_SEED_REPORT
    ):
        raise _ConfirmationError("per-seed comparison schema is unsupported")
    if document.get("status") != "complete":
        raise _ConfirmationError("per-seed comparison is not complete")
    if (
        document.get("ranking_objective") != RANKING_OBJECTIVE
        or document.get("ranking_metric") != RANKING_METRIC
    ):
        raise _ConfirmationError("per-seed ranking objective is incompatible")
    errors = _list(document.get("errors"), "per-seed errors")
    if errors:
        raise _ConfirmationError("complete per-seed comparison contains errors")
    confidence = _mapping(document.get("confidence"), "per-seed confidence")
    normal_quantile = _positive(
        confidence.get("normal_quantile"),
        "per-seed one-sided normal quantile",
    )
    if confidence.get("sidedness") != "one-sided-lower" or not math.isclose(
        normal_quantile,
        ONE_SIDED_95_NORMAL_QUANTILE,
        rel_tol=1e-15,
        abs_tol=1e-15,
    ):
        raise _ConfirmationError("per-seed confidence contract is incompatible")
    queue = _mapping(document.get("queue"), "per-seed queue evidence")
    if queue.get("seed") != seed:
        raise _ConfirmationError(
            f"comparison label seed {seed} disagrees with queue seed"
        )
    source_commit = _string(
        queue.get("source_commit"),
        "per-seed queue source commit",
    ).lower()
    if not _COMMIT_PATTERN.fullmatch(source_commit):
        raise _ConfirmationError("per-seed source commit is invalid")
    compute = _mapping(document.get("compute_accounting"), "compute accounting")
    provisioned_gpus = compute.get("provisioned_gpus")
    if type(provisioned_gpus) is not int or provisioned_gpus <= 0:
        raise _ConfirmationError("provisioned GPU count is invalid")
    treatments_raw = _list(document.get("treatments"), "per-seed treatments")
    treatments: dict[str, Mapping[str, object]] = {}
    for index, raw_treatment in enumerate(treatments_raw):
        treatment = _mapping(raw_treatment, f"per-seed treatment {index}")
        label = _string(treatment.get("label"), f"per-seed treatment {index} label")
        if label in treatments:
            raise _ConfirmationError(f"duplicate treatment label {label!r}")
        treatments[label] = treatment
    if document.get("run_count") != len(treatments):
        raise _ConfirmationError("per-seed run count disagrees with treatments")
    if document.get("eligible_count") != len(treatments):
        raise _ConfirmationError("per-seed eligible count is incomplete")
    if not treatments:
        raise _ConfirmationError("per-seed comparison has no treatments")
    treatment_roots = {
        label: Path(_string(treatment.get("run_root"), f"{label} treatment run root"))
        .expanduser()
        .resolve()
        for label, treatment in treatments.items()
    }
    _validate_queue_deployment(
        queue,
        comparison_path=comparison_path,
        seed=seed,
        source_commit=source_commit,
        treatment_roots=treatment_roots,
    )
    common_anchor = _mapping(document.get("common_anchor"), "common anchor")
    if common_anchor.get("status") != "available":
        raise _ConfirmationError("per-seed common anchor is unavailable")
    anchor_identity = _string(
        common_anchor.get("identity"),
        "per-seed common anchor identity",
    )
    if anchor_identity != policy.common_anchor_identity:
        raise _ConfirmationError(
            "per-seed common anchor identity differs from adoption policy"
        )
    by_treatment = _list(
        common_anchor.get("by_treatment"),
        "common anchor treatment evidence",
    )
    anchor_labels: set[str] = set()
    for raw_evidence in by_treatment:
        evidence = _mapping(raw_evidence, "common anchor treatment evidence")
        label = _string(evidence.get("label"), "common anchor evidence label")
        if label in anchor_labels or evidence.get("identity") != anchor_identity:
            raise _ConfirmationError("common anchor treatment evidence is inconsistent")
        anchor_labels.add(label)
    if anchor_labels != set(treatments):
        raise _ConfirmationError("common anchor evidence does not cover all treatments")
    anchor_steps: set[int] = set()
    for label, treatment in treatments.items():
        if (
            treatment.get("eligible") is not True
            or treatment.get("status") != "eligible"
        ):
            raise _ConfirmationError(
                f"source treatment {label!r} is a screening elimination"
            )
        if (
            treatment.get("training_objective") != TRAINING_OBJECTIVE
            or treatment.get("promotion_objective") != PROMOTION_OBJECTIVE
            or treatment.get("ranking_objective") != RANKING_OBJECTIVE
        ):
            raise _ConfirmationError(
                f"source treatment {label!r} has an incompatible objective"
            )
        anchor = _mapping(treatment.get("anchor"), f"{label} anchor")
        if anchor.get("identity") != anchor_identity:
            raise _ConfirmationError(
                f"source treatment {label!r} lacks common anchor evidence"
            )
        anchor_step = anchor.get("step")
        if type(anchor_step) is not int or anchor_step < 0:
            raise _ConfirmationError(
                f"source treatment {label!r} anchor step is invalid"
            )
        anchor_steps.add(anchor_step)
        _measurement_evidence(treatment, label=label)
    if anchor_steps != {policy.common_anchor_step}:
        raise _ConfirmationError(
            "per-seed common anchor step differs from adoption policy"
        )
    try:
        control_treatment = treatments[policy.control_treatment]
        candidate_treatment = treatments[policy.candidate_treatment]
    except KeyError as error:
        raise _ConfirmationError(
            "candidate must be present beside control in every seed confirmation"
        ) from error
    control = _treatment_metric(
        control_treatment,
        label=policy.control_treatment,
        normal_quantile=normal_quantile,
        verify_snapshot_live=False,
    )
    candidate = _treatment_metric(
        candidate_treatment,
        label=policy.candidate_treatment,
        normal_quantile=normal_quantile,
        verify_snapshot_live=True,
    )
    selector = _mapping(document.get("selector"), "per-seed selector")
    if selector.get("status") != "verified":
        raise _ConfirmationError("per-seed selector is not verified")
    selector_snapshot = _mapping(
        selector.get("winner_snapshot"),
        "per-seed selector winner snapshot",
    )
    try:
        verify_winner_snapshot(
            Path(
                _string(
                    selector_snapshot.get("run_root"),
                    "per-seed selector winner root",
                )
            ),
            selector_snapshot,
        )
    except (OSError, TypeError, ValueError) as error:
        raise _ConfirmationError(
            f"per-seed selector winner snapshot verification failed: {error}"
        ) from error
    if control.point_score <= 0:
        raise _ConfirmationError(
            "control point Elo/hour must be positive for relative improvement"
        )
    difference_point = candidate.point_score - control.point_score
    difference_standard_error = (
        candidate.conservative_standard_error_score
        + control.conservative_standard_error_score
    )
    difference_lcb = difference_point - normal_quantile * difference_standard_error
    relative_improvement = difference_point / control.point_score
    contract = _source_contract(
        document,
        provisioned_gpus=provisioned_gpus,
        normal_quantile=normal_quantile,
    )
    record = {
        "seed": seed,
        "status": "confirmed",
        "common_anchor": {
            "status": "available",
            "identity": anchor_identity,
            "step": policy.common_anchor_step,
            "control_label": policy.control_treatment,
            "candidate_label": policy.candidate_treatment,
        },
        "control": {
            "label": policy.control_treatment,
            "primary_ring10_only_champion_frontier_lcb_per_total_provisioned_wall_hour": (
                control.lower_bound_score
            ),
            "point_elo_per_total_provisioned_wall_hour": control.point_score,
            "conservative_standard_error_per_total_provisioned_wall_hour": (
                control.conservative_standard_error_score
            ),
            "total_provisioned_wall_hours": control.total_provisioned_wall_hours,
            "champion_frontier": control.frontier,
        },
        "candidate": {
            "label": policy.candidate_treatment,
            "primary_ring10_only_champion_frontier_lcb_per_total_provisioned_wall_hour": (
                candidate.lower_bound_score
            ),
            "point_elo_per_total_provisioned_wall_hour": candidate.point_score,
            "conservative_standard_error_per_total_provisioned_wall_hour": (
                candidate.conservative_standard_error_score
            ),
            "total_provisioned_wall_hours": candidate.total_provisioned_wall_hours,
            "champion_frontier": candidate.frontier,
            "verified_winner_snapshot": candidate.winner_snapshot,
        },
        "advantage": {
            "point_elo_per_hour": difference_point,
            "conservative_standard_error_per_hour": difference_standard_error,
            "one_sided_lower_bound_elo_per_hour": difference_lcb,
            "relative_point_elo_per_hour_improvement": relative_improvement,
            "method": (
                "candidate_minus_control_point_advantage_less_source_quantile_times_"
                "sum_of_conservative_rate_standard_errors"
            ),
            "source_one_sided_normal_quantile": normal_quantile,
        },
    }
    return _SeedConfirmation(
        seed=seed,
        source_commit=source_commit,
        anchor_identity=anchor_identity,
        anchor_step=policy.common_anchor_step,
        contract=contract,
        control=control,
        candidate=candidate,
        record=record,
    )


def _screening_eliminations(
    documents: Mapping[int, Mapping[str, object]],
    *,
    policy: AdoptionPolicy,
) -> list[dict[str, object]]:
    observations: dict[str, dict[int, bool]] = {}
    for seed, document in documents.items():
        raw_treatments = document.get("treatments")
        if not isinstance(raw_treatments, list):
            continue
        for raw_treatment in raw_treatments:
            if not isinstance(raw_treatment, Mapping):
                continue
            label = raw_treatment.get("label")
            if not isinstance(label, str) or label in {
                policy.control_treatment,
                policy.candidate_treatment,
            }:
                continue
            observations.setdefault(label, {})[seed] = (
                raw_treatment.get("eligible") is True
            )
    eliminations = []
    required = set(policy.seeds)
    for label, by_seed in sorted(observations.items()):
        missing = sorted(required - set(by_seed))
        ineligible = sorted(seed for seed, eligible in by_seed.items() if not eligible)
        if missing or ineligible:
            eliminations.append(
                {
                    "label": label,
                    "recorded": True,
                    "confirmed_seeds": sorted(by_seed),
                    "missing_seeds": missing,
                    "ineligible_seeds": ineligible,
                    "adoption_eligible": False,
                }
            )
    raw_policy_eliminations = policy.document.get("screening_eliminations", [])
    if isinstance(raw_policy_eliminations, list):
        for raw_elimination in raw_policy_eliminations:
            if not isinstance(raw_elimination, Mapping):
                continue
            label = raw_elimination.get("label")
            reason = raw_elimination.get("reason")
            if (
                isinstance(label, str)
                and label not in {policy.control_treatment, policy.candidate_treatment}
                and isinstance(reason, str)
                and reason
            ):
                record = {
                    "label": label,
                    "recorded": True,
                    "reason": reason,
                    "adoption_eligible": False,
                }
                if record not in eliminations:
                    eliminations.append(record)
    return sorted(
        eliminations,
        key=lambda item: (str(item.get("label")), json.dumps(item, sort_keys=True)),
    )


def build_cross_seed_comparison(
    comparisons: Sequence[PinnedComparison],
    *,
    policy_path: Path,
    policy_sha256: str,
) -> dict[str, object]:
    """Build a deterministic three-seed adoption-eligibility report."""
    policy = load_adoption_policy(policy_path, policy_sha256)
    pinned = _validate_pins(comparisons, policy)
    documents: dict[int, Mapping[str, object]] = {}
    sources = []
    protected_run_roots: set[str] = set()
    for comparison in pinned:
        resolved, document = _read_pinned_json(
            comparison.path,
            comparison.sha256,
            name=f"seed {comparison.seed} comparison",
        )
        if sha256_file(resolved) != comparison.sha256:
            raise CrossSeedComparisonError(
                f"seed {comparison.seed} comparison changed while being read"
            )
        documents[comparison.seed] = document
        raw_treatments = document.get("treatments")
        if isinstance(raw_treatments, list):
            for raw_treatment in raw_treatments:
                if not isinstance(raw_treatment, Mapping):
                    continue
                raw_root = raw_treatment.get("run_root")
                if (
                    isinstance(raw_root, str)
                    and Path(raw_root).expanduser().is_absolute()
                ):
                    protected_run_roots.add(str(Path(raw_root).expanduser().resolve()))
        sources.append(
            {
                "seed": comparison.seed,
                "path": str(resolved),
                "sha256": comparison.sha256,
            }
        )

    confirmations: list[_SeedConfirmation] = []
    confirmation_errors = []
    per_seed: list[dict[str, object]] = []
    for comparison in pinned:
        try:
            confirmation = _validate_confirmation(
                seed=comparison.seed,
                comparison_path=comparison.path,
                document=documents[comparison.seed],
                policy=policy,
            )
        except _ConfirmationError as error:
            confirmation_errors.append(
                {
                    "code": "invalid_seed_confirmation",
                    "seed": comparison.seed,
                    "message": str(error),
                }
            )
            per_seed.append(
                {
                    "seed": comparison.seed,
                    "status": "invalid",
                    "error": str(error),
                }
            )
        else:
            confirmations.append(confirmation)
            per_seed.append(confirmation.record)

    consistency_errors: list[dict[str, object]] = []
    if len(confirmations) == len(policy.seeds):
        commits = {confirmation.source_commit for confirmation in confirmations}
        if commits != {policy.source_commit}:
            consistency_errors.append(
                {
                    "code": "inconsistent_source_commit",
                    "message": (
                        "all confirmations must use the policy source commit; "
                        f"observed {sorted(commits)}"
                    ),
                }
            )
        anchors = {
            (confirmation.anchor_identity, confirmation.anchor_step)
            for confirmation in confirmations
        }
        if anchors != {(policy.common_anchor_identity, policy.common_anchor_step)}:
            consistency_errors.append(
                {
                    "code": "inconsistent_common_anchor",
                    "message": (
                        "all confirmations must use the policy-pinned common "
                        f"anchor; observed {sorted(anchors)}"
                    ),
                }
            )
        contracts = {
            json.dumps(confirmation.contract, sort_keys=True, separators=(",", ":"))
            for confirmation in confirmations
        }
        if len(contracts) != 1:
            consistency_errors.append(
                {
                    "code": "inconsistent_topology_contract",
                    "message": (
                        "control/candidate objective and topology contract changed "
                        "across seeds"
                    ),
                }
            )

    errors = [*confirmation_errors, *consistency_errors]
    aggregate: dict[str, object] | None = None
    complete = len(confirmations) == len(policy.seeds) and not errors
    if complete:
        relative_improvements = [
            _finite(
                _mapping(
                    confirmation.record["advantage"],
                    "seed advantage",
                ).get("relative_point_elo_per_hour_improvement"),
                "seed relative point improvement",
            )
            for confirmation in confirmations
        ]
        lcb_advantages = [
            _finite(
                _mapping(
                    confirmation.record["advantage"],
                    "seed advantage",
                ).get("one_sided_lower_bound_elo_per_hour"),
                "seed one-sided LCB advantage",
            )
            for confirmation in confirmations
        ]
        point_advantages = [
            _finite(
                _mapping(
                    confirmation.record["advantage"],
                    "seed advantage",
                ).get("point_elo_per_hour"),
                "seed point advantage",
            )
            for confirmation in confirmations
        ]
        candidate_points = [
            confirmation.candidate.point_score for confirmation in confirmations
        ]
        control_points = [
            confirmation.control.point_score for confirmation in confirmations
        ]
        median_relative = statistics.median(relative_improvements)
        minimum_lcb = min(lcb_advantages)
        aggregate = {
            "method": "deterministic_median_points_and_minimum_seed_lcb",
            "seed_order": list(policy.seeds),
            "control_median_point_elo_per_hour": statistics.median(control_points),
            "candidate_median_point_elo_per_hour": statistics.median(candidate_points),
            "median_point_elo_per_hour_advantage": statistics.median(point_advantages),
            "median_relative_point_elo_per_hour_improvement": median_relative,
            "minimum_per_seed_one_sided_lcb_advantage_elo_per_hour": minimum_lcb,
            "per_seed_relative_improvements": relative_improvements,
            "per_seed_one_sided_lcb_advantages_elo_per_hour": lcb_advantages,
            "cross_seed_confidence": {
                "level": None,
                "method": LCB_GATE_METHOD,
                "pooled_interval": False,
                "note": (
                    "The gate takes the minimum of three separately computed "
                    "conservative candidate-control lower bounds. It does not pool "
                    "or combine per-seed intervals and claims no new aggregate "
                    "confidence level."
                ),
            },
        }
    median_observed = (
        _finite(
            aggregate["median_relative_point_elo_per_hour_improvement"],
            "aggregate median improvement",
        )
        if aggregate is not None
        else None
    )
    lcb_observed = (
        _finite(
            aggregate["minimum_per_seed_one_sided_lcb_advantage_elo_per_hour"],
            "aggregate LCB advantage",
        )
        if aggregate is not None
        else None
    )
    gates = {
        "exact_seed_set": {
            "passed": tuple(sorted(confirmation.seed for confirmation in confirmations))
            == policy.seeds,
            "required": list(policy.seeds),
            "confirmed": sorted(confirmation.seed for confirmation in confirmations),
        },
        "complete_schema_valid_confirmations": {
            "passed": complete,
            "required_count": len(policy.seeds),
            "confirmed_count": len(confirmations),
        },
        "minimum_median_point_improvement": {
            "passed": (
                median_observed is not None
                and median_observed >= policy.minimum_median_improvement
            ),
            "required": policy.minimum_median_improvement,
            "observed": median_observed,
        },
        "positive_one_sided_lcb_advantage": {
            "passed": (
                lcb_observed is not None and lcb_observed > policy.minimum_lcb_advantage
            ),
            "method": LCB_GATE_METHOD,
            "strictly_greater_than": policy.minimum_lcb_advantage,
            "observed": lcb_observed,
        },
    }
    eligible = all(
        isinstance(gate, Mapping) and gate.get("passed") is True
        for gate in gates.values()
    )
    selected_confirmation = None
    if eligible:
        ordered_by_candidate_point = sorted(
            confirmations,
            key=lambda confirmation: (
                confirmation.candidate.point_score,
                confirmation.seed,
            ),
        )
        selected_confirmation = ordered_by_candidate_point[len(policy.seeds) // 2]
    selector = {
        "status": "verified" if selected_confirmation is not None else "unavailable",
        "candidate_label": policy.candidate_treatment,
        "control_label": policy.control_treatment,
        "source_seed": (
            selected_confirmation.seed if selected_confirmation is not None else None
        ),
        "winner_snapshot": (
            selected_confirmation.candidate.winner_snapshot
            if selected_confirmation is not None
            else None
        ),
        "selection": "median_candidate_point_score_seed",
        "latest_terminal_candidates_ranked": False,
        "reason": (
            None
            if selected_confirmation is not None
            else "cross-seed adoption gates are not all satisfied"
        ),
    }
    contract = (
        confirmations[0].contract if confirmations and not consistency_errors else None
    )
    return {
        "schema_version": SCHEMA_VERSION,
        "report": REPORT_NAME,
        "status": "eligible" if eligible else "ineligible",
        "eligible": eligible,
        "source_commit": policy.source_commit,
        "policy": {
            "path": str(policy.path),
            "sha256": policy.sha256,
            "report": POLICY_REPORT,
        },
        "sources": sources,
        "protected_run_roots": sorted(protected_run_roots),
        "control_treatment": policy.control_treatment,
        "candidate_treatment": policy.candidate_treatment,
        "common_objective": TRAINING_OBJECTIVE,
        "comparison_contract": contract,
        "per_seed": per_seed,
        "aggregate": aggregate,
        "gates": gates,
        "selector": selector,
        "screening_eliminations": _screening_eliminations(
            documents,
            policy=policy,
        ),
        "errors": errors,
    }


def _parse_seed_values(
    values: Sequence[str],
    *,
    name: str,
) -> dict[int, str]:
    parsed: dict[int, str] = {}
    for value in values:
        if "=" not in value:
            raise CrossSeedComparisonError(f"{name} must use SEED=VALUE form")
        raw_seed, raw_value = value.split("=", 1)
        try:
            seed = int(raw_seed)
        except ValueError as error:
            raise CrossSeedComparisonError(f"{name} seed must be an integer") from error
        if not raw_value:
            raise CrossSeedComparisonError(f"{name} value must be non-empty")
        if seed in parsed:
            raise CrossSeedComparisonError(f"duplicate {name} seed {seed}")
        parsed[seed] = raw_value
    return parsed


def _comparison_arguments(
    paths: Sequence[str],
    digests: Sequence[str],
) -> list[PinnedComparison]:
    parsed_paths = _parse_seed_values(paths, name="comparison")
    parsed_digests = _parse_seed_values(digests, name="comparison SHA-256")
    if set(parsed_paths) != set(parsed_digests):
        raise CrossSeedComparisonError(
            "comparison paths and SHA-256 pins must cover the same seeds"
        )
    return [
        PinnedComparison(seed, Path(parsed_paths[seed]), parsed_digests[seed])
        for seed in parsed_paths
    ]


def _error_document(error: Exception) -> dict[str, object]:
    return {
        "schema_version": SCHEMA_VERSION,
        "report": REPORT_NAME,
        "status": "error",
        "eligible": False,
        "error": f"{type(error).__name__}: {error}",
    }


def main(argv: list[str] | None = None) -> int:
    arguments = _parser().parse_args(argv)
    try:
        comparisons = _comparison_arguments(
            arguments.comparison,
            arguments.comparison_sha256,
        )
        report = build_cross_seed_comparison(
            comparisons,
            policy_path=arguments.policy,
            policy_sha256=arguments.policy_sha256,
        )
    except (CrossSeedComparisonError, OSError, TypeError, ValueError) as error:
        print(json.dumps(_error_document(error), sort_keys=True, allow_nan=False))
        return 2
    serialized = (
        json.dumps(
            report,
            indent=2,
            sort_keys=True,
            allow_nan=False,
        )
        + "\n"
    )
    if arguments.output is not None:
        try:
            output = arguments.output.expanduser().resolve()
            protected_paths = {
                comparison.path.expanduser().resolve() for comparison in comparisons
            }
            protected_paths.add(arguments.policy.expanduser().resolve())
            if output in protected_paths:
                raise CrossSeedComparisonError(
                    "cross-seed output must not overwrite a source or policy"
                )
            _write_immutable_json(output, report)
        except (CrossSeedComparisonError, FileExistsError, OSError) as error:
            print(json.dumps(_error_document(error), sort_keys=True, allow_nan=False))
            return 2
    print(serialized, end="")
    return 0 if report["eligible"] is True else 3


if __name__ == "__main__":
    raise SystemExit(main())
