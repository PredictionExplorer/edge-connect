"""Read-only evidence for ending teacher transfer before a fixed step deadline."""

from __future__ import annotations

import hashlib
import json
import math
import sqlite3
import time
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from .arena import ArenaPair, summarize_pairs
from .config import ExperimentConfig, load_config
from .contracts import FEATURE_SCHEMA_HASH, RULES_HASH
from .topology import SUPPORTED_RINGS


def recent_teacher_counts(
    connection: sqlite3.Connection,
    *,
    run_id: str,
    generation_family: str,
    step: int,
    lag: int,
    per_ring_quota: int,
    teacher_actor: str,
) -> dict[int, dict[str, int]]:
    """Mirror Stage A's newest eligible position window, within one DB snapshot."""

    if any(type(value) is not int or value < 0 for value in (step, lag)):
        raise ValueError("learner step and replay lag must be non-negative integers")
    if type(per_ring_quota) is not int or per_ring_quota <= 0:
        raise ValueError("replay quota must be positive")
    output = {}
    for ring in SUPPORTED_RINGS:
        remaining = per_ring_quota
        selected = teacher = 0
        rows = connection.execute(
            "SELECT sample_count,actor_id FROM shards WHERE state='ready' "
            "AND run_id=? AND generation_family=? AND ring=? "
            "AND rules_hash=? AND feature_schema_hash=? "
            "AND model_step BETWEEN ? AND ? ORDER BY id DESC",
            (
                run_id,
                generation_family,
                ring,
                f"{RULES_HASH:016x}",
                f"{FEATURE_SCHEMA_HASH:016x}",
                max(0, step - lag),
                step,
            ),
        )
        for count, actor_id in rows:
            amount = min(remaining, int(count))
            if amount <= 0:
                raise ValueError("replay contains an invalid sample count")
            selected += amount
            teacher += amount if actor_id == teacher_actor else 0
            remaining -= amount
            if not remaining:
                break
        output[ring] = {"selected_samples": selected, "teacher_samples": teacher}
    return output


def assess_lineage_result(
    result: Mapping[str, Any],
    *,
    candidate_identity: str,
    teacher_identity: str,
    minimum_pairs_per_ring: int = 15,
    minimum_simulations: int = 1024,
    minimum_elo_lower: float = -15.0,
) -> dict[str, Any]:
    """Recompute paired evidence; stale, partial or mismatched results never pass."""

    if not math.isfinite(minimum_elo_lower):
        raise ValueError("minimum Elo lower bound must be finite")
    for value in (minimum_pairs_per_ring, minimum_simulations):
        if type(value) is not int or value <= 0:
            raise ValueError("lineage evidence budgets must be positive integers")
    if (
        result.get("result_kind") != "lineage_crossplay"
        or result.get("evaluation_mode") != "cross_schema"
        or result.get("candidate") != candidate_identity
        or result.get("baseline") != teacher_identity
        or result.get("interrupted") is not False
    ):
        raise ValueError("lineage result is stale, interrupted or has wrong identities")
    search = result.get("search")
    if not isinstance(search, Mapping) or type(search.get("simulations")) is not int:
        raise ValueError("lineage result lacks a verified search budget")
    if search["simulations"] < minimum_simulations:
        raise ValueError("lineage result search budget is below the readiness budget")
    if search.get("seed_stream_policy") != "independent-cell-pair-seat-move-v2":
        raise ValueError(
            "lineage certification requires independent paired search streams"
        )
    baseline = result.get("baseline_metadata")
    baseline_budget = (
        baseline.get("search_budget") if isinstance(baseline, Mapping) else None
    )
    keys = ("simulations", "max_considered", "c_visit", "c_scale")
    if not isinstance(baseline_budget, Mapping) or any(
        key not in search or baseline_budget.get(key) != search[key] for key in keys
    ):
        raise ValueError(
            "lineage participants must have identical verified search budgets"
        )
    raw_pairs = result.get("pairs")
    if not isinstance(raw_pairs, list) or not raw_pairs:
        raise ValueError("lineage result lacks paired evidence")
    pairs = []
    identities = set()
    counts = {ring: 0 for ring in SUPPORTED_RINGS}
    for raw in raw_pairs:
        if not isinstance(raw, Mapping):
            raise ValueError("invalid lineage pair")
        fields = dict(raw)
        outcomes = fields.get("outcomes")
        if not isinstance(outcomes, (list, tuple)) or len(outcomes) != 2:
            raise ValueError("lineage evidence requires complete role-reversed pairs")
        fields["outcomes"] = tuple(outcomes)
        pair = ArenaPair(**fields)
        key = (pair.ring, pair.pair)
        if (
            pair.ring not in counts
            or pair.variant != "double"
            or pair.segment != "standard"
            or key in identities
        ):
            raise ValueError("lineage evidence has duplicate or nonstandard pairs")
        identities.add(key)
        counts[pair.ring] += 1
        pairs.append(pair)
    if len(set(counts.values())) != 1 or min(counts.values()) < minimum_pairs_per_ring:
        raise ValueError(
            "lineage evidence requires equal sufficient coverage of all rings"
        )
    statistics = summarize_pairs(
        pairs, confidence=0.95, bootstrap_samples=2000, seed=170905
    )
    interval = statistics["anytime_elo_interval"]
    assert isinstance(interval, list)
    return {
        "passed": float(interval[0]) >= minimum_elo_lower,
        "minimum_elo_lower": minimum_elo_lower,
        "pairs_per_ring": counts,
        "search_simulations": search["simulations"],
        "statistics": statistics,
        "criterion": "paired_anytime_95pct_lower_bound",
    }


def transfer_readiness(
    root: Path,
    config: ExperimentConfig,
    *,
    profile_path: Path,
    lineage_result: Path | None = None,
    minimum_elo_lower: float = -15.0,
) -> dict[str, Any]:
    """Produce auditable readiness without changing replay, pointers or profiles."""

    root = root.resolve()
    profile_path = profile_path.resolve()
    profile_bytes = profile_path.read_bytes()
    profile_sha256 = hashlib.sha256(profile_bytes).hexdigest()
    registered = (root / "profile.sha256").read_text().strip().split(maxsplit=1)
    if len(registered) != 2 or registered[0] != profile_sha256:
        raise ValueError("readiness profile does not match the active checksum")
    registered_path = Path(registered[1].lstrip("*"))
    if not registered_path.is_absolute():
        registered_path = root / registered_path
    if (
        registered_path.resolve() != profile_path
        or load_config(profile_path).as_dict() != config.as_dict()
    ):
        raise ValueError("readiness configuration is not the registered active profile")
    if config.selfplay.variants.enabled or config.learner.segment_quotas is not None:
        raise ValueError("transfer readiness expects the Stage A unsegmented profile")
    if config.learner.minimum_replay_shard_id_exclusive is not None:
        raise ValueError(
            "transfer readiness requires an unmodified lineage replay cutoff"
        )
    identity = json.loads((root / "run.json").read_text())
    if (
        config.orchestration.run_id != identity["run_id"]
        or Path(config.orchestration.directories.root).resolve() != root
    ):
        raise ValueError("readiness profile does not identify the active run root")
    transfer_path = root / "lineage-transfer.json"
    transfer_bytes = transfer_path.read_bytes()
    transfer = json.loads(transfer_bytes)
    if (
        transfer.get("report") != "startrain-lineage-transfer"
        or transfer.get("run_id") != identity["run_id"]
        or transfer.get("generation_family") != identity["generation_family"]
        or transfer.get("actor_id") != "lineage-transfer"
    ):
        raise ValueError("transfer provenance does not match the active run")
    heartbeat = json.loads((root / "status/learner.heartbeat.json").read_text())
    step = heartbeat.get("step")
    if type(step) is not int or step < 0:
        raise ValueError("learner step is unavailable")
    database = (root / "replay/manifest.sqlite3").resolve()
    connection = sqlite3.connect(
        f"{database.as_uri()}?mode=ro", uri=True, isolation_level=None
    )
    try:
        connection.execute("PRAGMA query_only=ON")
        connection.execute("BEGIN")
        windows = recent_teacher_counts(
            connection,
            run_id=identity["run_id"],
            generation_family=identity["generation_family"],
            step=step,
            lag=config.learner.max_replay_lag_steps,
            per_ring_quota=config.learner.recent_samples_per_ring,
            teacher_actor=transfer["actor_id"],
        )
        cutoff = connection.execute("SELECT MAX(id) FROM shards").fetchone()[0]
    finally:
        connection.close()
    filled = all(
        v["selected_samples"] == config.learner.recent_samples_per_ring
        for v in windows.values()
    )
    teacher_absent = filled and all(v["teacher_samples"] == 0 for v in windows.values())
    evidence = None
    result_sha256 = None
    if lineage_result is not None:
        result_bytes = lineage_result.read_bytes()
        champion = json.loads((root / "learner/champion.json").read_text())
        evidence = assess_lineage_result(
            json.loads(result_bytes),
            candidate_identity=champion["model_identity"],
            teacher_identity=transfer["teacher"]["identity"],
            minimum_elo_lower=minimum_elo_lower,
        )
        result_sha256 = hashlib.sha256(result_bytes).hexdigest()
    ready = teacher_absent and evidence is not None and evidence["passed"]
    return {
        "schema_version": 1,
        "report": "startrain-transfer-readiness",
        "observed_ns": time.time_ns(),
        "run_id": identity["run_id"],
        "generation_family": identity["generation_family"],
        "learner_step": step,
        "replay_maximum_shard_id": cutoff,
        "windows": windows,
        "teacher_window_empty": teacher_absent,
        "curriculum_ready": teacher_absent,
        "legacy_strength_certified": bool(ready),
        "ready": bool(ready),
        "lineage_evaluation": evidence,
        "lineage_result_sha256": result_sha256,
        "transfer_report_sha256": hashlib.sha256(transfer_bytes).hexdigest(),
        "profile_sha256": profile_sha256,
        "decision": "ready"
        if ready
        else (
            "teacher_window_not_empty"
            if not teacher_absent
            else "evaluation_required"
            if evidence is None
            else "strength_gate_not_met"
        ),
    }
