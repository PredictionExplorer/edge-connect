#!/usr/bin/env python3
"""Summarize optimizer, EMA, replay-source, and arena dynamics for frozen runs."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import statistics
import sys
from collections import Counter, defaultdict
from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import Any

import yaml

SCHEMA_VERSION = 1
REPORT = "startrain-learning-dynamics-diagnosis"
CLINCH_AVAILABILITY = {
    "losses.clinch_policy": "losses.clinch_policy_available",
    "losses.clinch_outcome": "losses.clinch_outcome_available",
    "losses.clinch_score_margin": "losses.clinch_score_margin_available",
    "losses.clinch_ownership": "losses.clinch_ownership_available",
    "losses.clinch_alive": "losses.clinch_alive_available",
    "losses.clinch_soft_policy": "losses.clinch_soft_policy_available",
}
LEARNING_RATE_REDUCTION_EVENTS = frozenset({"plateau_recovery", "plateau_reset"})


class DiagnosisError(RuntimeError):
    """A run cannot provide trustworthy diagnostic evidence."""


def _json(path: Path, name: str) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise DiagnosisError(f"cannot read {name} {path}: {exc}") from exc
    if not isinstance(payload, dict):
        raise DiagnosisError(f"{name} must be a JSON object: {path}")
    return payload


def _jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        return []
    rows = []
    for line_number, line in enumerate(
        path.read_text(encoding="utf-8").splitlines(), 1
    ):
        if not line.strip():
            continue
        try:
            payload = json.loads(line)
        except json.JSONDecodeError as exc:
            raise DiagnosisError(f"invalid JSONL {path}:{line_number}: {exc}") from exc
        if isinstance(payload, dict):
            rows.append(payload)
    return rows


def _number(value: object) -> float | None:
    if isinstance(value, bool):
        return float(value)
    if not isinstance(value, int | float):
        return None
    result = float(value)
    return result if math.isfinite(result) else None


def _summary(values: Iterable[object]) -> dict[str, float | int] | None:
    numbers = [number for value in values if (number := _number(value)) is not None]
    if not numbers:
        return None
    return {
        "count": len(numbers),
        "minimum": min(numbers),
        "mean": statistics.fmean(numbers),
        "median": statistics.median(numbers),
        "maximum": max(numbers),
    }


def _nested(row: Mapping[str, object], path: str) -> object:
    value: object = row
    for part in path.split("."):
        if not isinstance(value, Mapping):
            return None
        value = value.get(part)
    return value


def _diagnostic_metric_values(
    rows: Iterable[Mapping[str, object]],
    path: str,
) -> Iterable[object]:
    availability_path = CLINCH_AVAILABILITY.get(path)
    for row in rows:
        if availability_path is not None:
            availability = _number(_nested(row, availability_path))
            if availability is None or availability <= 0:
                continue
        yield _nested(row, path)


def _finite_number_list(value: object) -> list[float] | None:
    if not isinstance(value, list | tuple):
        return None
    numbers: list[float] = []
    for item in value:
        if isinstance(item, bool):
            return None
        number = _number(item)
        if number is None or number < 0:
            return None
        numbers.append(number)
    return numbers


def _learning_rate_evidence(
    rows: Iterable[Mapping[str, object]],
) -> tuple[list[float] | None, float | None, list[dict[str, object]]]:
    latest_rates = None
    latest_reduction_scale = None
    reduction_events: list[dict[str, object]] = []
    for row in rows:
        rates = _finite_number_list(row.get("learning_rates"))
        if rates is not None:
            latest_rates = rates
        event = row.get("event")
        if event not in LEARNING_RATE_REDUCTION_EVENTS:
            continue
        raw_scale = row.get("learning_rate_scale")
        scale = None if isinstance(raw_scale, bool) else _number(raw_scale)
        if scale is None or not 0 < scale <= 1:
            scale = None
        else:
            latest_reduction_scale = scale
        reduction_events.append(
            {
                "event": event,
                "timestamp_ns": row.get("timestamp_ns"),
                "reason": row.get("reason"),
                "from_step": row.get("from_step"),
                "to_step": row.get("to_step"),
                "candidate_identity": row.get("candidate_identity"),
                "champion_identity": row.get("champion_identity"),
                "learning_rate_scale": scale,
                "learning_rates": rates,
                "optimizer_state_cleared": row.get("optimizer_state_cleared"),
            }
        )
    return latest_rates, latest_reduction_scale, reduction_events


def _profile(root: Path) -> tuple[Path, dict[str, Any]]:
    candidates = (
        root / "profile-elo-ablation.yaml",
        root / "profile.yaml",
        root / "profile-relocated.yaml",
    )
    path = next((candidate for candidate in candidates if candidate.is_file()), None)
    if path is None:
        raise DiagnosisError(f"run has no frozen profile: {root}")
    try:
        payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    except (OSError, yaml.YAMLError) as exc:
        raise DiagnosisError(f"cannot read profile {path}: {exc}") from exc
    if not isinstance(payload, dict):
        raise DiagnosisError(f"profile must be a mapping: {path}")
    return path, payload


def _model_publications(root: Path) -> list[dict[str, Any]]:
    return _jsonl(root / "learner" / "model-history.jsonl")


def _terminal_arena(root: Path) -> list[dict[str, object]]:
    results = []
    for path in sorted(
        (root / "arena").glob("sha256-*.json"),
        key=lambda candidate: candidate.stat().st_mtime_ns,
    ):
        document = _json(path, "arena result")
        if document.get("terminal") is not True:
            continue
        promotion = document.get("promotion")
        promotion = promotion if isinstance(promotion, Mapping) else {}
        results.append(
            {
                "path": str(path),
                "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                "candidate": document.get("candidate"),
                "baseline": document.get("baseline"),
                "completed_ns": document.get("completed_ns"),
                "games": len(document.get("games", [])),
                "decision": promotion.get("decision"),
                "pair_score_rate": promotion.get("pair_score_rate"),
                "confidence_sequence": promotion.get("confidence_sequence"),
            }
        )
    return results


def _diagnose_run(label: str, root: Path) -> dict[str, object]:
    resolved = root.expanduser().resolve()
    if not resolved.is_dir() or resolved.is_symlink():
        raise DiagnosisError(f"run root is missing or unsafe: {resolved}")
    run = _json(resolved / "run.json", "run identity")
    profile_path, profile = _profile(resolved)
    orchestration = profile.get("orchestration")
    orchestration = orchestration if isinstance(orchestration, Mapping) else {}
    learner_config = profile.get("learner")
    learner_config = learner_config if isinstance(learner_config, Mapping) else {}
    train_config = profile.get("train")
    train_config = train_config if isinstance(train_config, Mapping) else {}
    optimizer_config = profile.get("optimizer")
    optimizer_config = optimizer_config if isinstance(optimizer_config, Mapping) else {}

    learner_rows = _jsonl(resolved / "learner" / "metrics.jsonl")
    metric_paths = (
        "examples_per_second",
        "step_seconds",
        "gradient_norm",
        "gradient_pre_clip_norm",
        "gradient_post_clip_norm",
        "gradient_clip_threshold",
        "gradient_clip_coefficient",
        "gradient_clip_severity",
        "gradient_clip_ratio",
        "gradient_clipped",
        "gradient_clipping_frequency",
        "gradient_clip_fraction",
        "nonfinite_loss_count",
        "nonfinite_gradient_count",
        "segment_updates_per_new_sample",
        "updates_per_new_sample",
        "ema_raw_shadow_distance",
        "ema_turnover",
        "scheduler_age_steps",
        "scheduler_segment_position",
        "optimizer_update_norm",
        "optimizer_weight_norm",
        "raw_vs_ema_distance",
        "raw_vs_ema_relative_distance",
        "ema_effective_turnover",
        "losses.total",
        "losses.policy",
        "losses.outcome",
        "losses.score_margin",
        "losses.ownership",
        "losses.alive",
        "losses.clinch_policy",
        "losses.clinch_outcome",
        "losses.clinch_score_margin",
        "losses.clinch_ownership",
        "losses.clinch_alive",
        "losses.clinch_soft_policy",
        "losses.clinch_samples",
        "losses.clinch_policy_available",
        "losses.clinch_outcome_available",
        "losses.clinch_score_margin_available",
        "losses.clinch_ownership_available",
        "losses.clinch_alive_available",
        "losses.clinch_soft_policy_available",
    )
    learner_summary = {
        path: summary
        for path in metric_paths
        if (summary := _summary(_diagnostic_metric_values(learner_rows, path)))
        is not None
    }
    (
        runtime_learning_rates,
        latest_recorded_learning_rate_scale,
        learning_rate_reduction_events,
    ) = _learning_rate_evidence(learner_rows)
    latest_optimizer_groups = next(
        (
            groups
            for row in reversed(learner_rows)
            if isinstance(groups := row.get("optimizer_groups"), list)
        ),
        None,
    )

    actor_rows = []
    for path in sorted((resolved / "metrics").glob("actor*.jsonl")):
        actor_rows.extend(_jsonl(path))
    source_roles = Counter(
        str(row.get("model_role"))
        for row in actor_rows
        if isinstance(row.get("model_role"), str)
    )
    source_samples: defaultdict[str, float] = defaultdict(float)
    for row in actor_rows:
        role = row.get("model_role")
        samples = _number(row.get("samples"))
        if isinstance(role, str) and samples is not None:
            source_samples[role] += samples

    publications = _model_publications(resolved)
    publication_steps = [
        int(row["model_step"])
        for row in publications
        if isinstance(row.get("model_step"), int)
        and not isinstance(row.get("model_step"), bool)
    ]
    publication_examples = [
        int(row["examples_consumed"])
        for row in publications
        if isinstance(row.get("examples_consumed"), int)
        and not isinstance(row.get("examples_consumed"), bool)
    ]
    terminal = _terminal_arena(resolved)
    decisions = Counter(str(row.get("decision")) for row in terminal)

    ablation = (
        _json(resolved / "ablation.json", "ablation metadata")
        if (resolved / "ablation.json").is_file()
        else None
    )
    recovery = (
        _json(resolved / "learner" / "recovery.json", "recovery pointer")
        if (resolved / "learner" / "recovery.json").is_file()
        else None
    )
    champion = (
        _json(resolved / "learner" / "champion.json", "champion pointer")
        if (resolved / "learner" / "champion.json").is_file()
        else None
    )

    return {
        "label": label,
        "run_root": str(resolved),
        "run_id": run.get("run_id"),
        "generation_family": run.get("generation_family"),
        "profile": str(profile_path),
        "profile_sha256": hashlib.sha256(profile_path.read_bytes()).hexdigest(),
        "training_objective": orchestration.get("training_objective", "generalist"),
        "model": profile.get("model"),
        "optimizer": dict(optimizer_config),
        "train": {
            "batch_size": train_config.get("per_rank_batch_size"),
            "ema_decay": train_config.get("ema_decay"),
            "ema_half_life_examples": train_config.get("ema_half_life_examples"),
            "gradient_clip_norm": train_config.get("gradient_clip_norm"),
        },
        "learner_config": {
            key: learner_config.get(key)
            for key in (
                "candidate_interval",
                "candidate_interval_examples",
                "target_updates_per_new_sample",
                "max_replay_lag_steps",
                "minimum_replay_shard_id_exclusive",
            )
        },
        "learner_metric_rows": len(learner_rows),
        "learner_metrics": learner_summary,
        "latest_learner_metric": learner_rows[-1] if learner_rows else None,
        "runtime_learning_rates": runtime_learning_rates,
        "latest_recorded_learning_rate_scale": latest_recorded_learning_rate_scale,
        "learning_rate_reduction_events": learning_rate_reduction_events,
        "latest_optimizer_groups": latest_optimizer_groups,
        "actor_metric_rows": len(actor_rows),
        "selfplay_source_role_rows": dict(sorted(source_roles.items())),
        "selfplay_source_samples": dict(sorted(source_samples.items())),
        "model_publications": len(publications),
        "publication_step_deltas": [
            current - previous
            for previous, current in zip(publication_steps, publication_steps[1:])
        ],
        "publication_example_deltas": [
            current - previous
            for previous, current in zip(
                publication_examples,
                publication_examples[1:],
            )
        ],
        "terminal_evaluations": terminal,
        "terminal_decisions": dict(sorted(decisions.items())),
        "champion": champion,
        "recovery": recovery,
        "ablation": ablation,
    }


def _parse_run(value: str) -> tuple[str, Path]:
    label, separator, raw_path = value.partition("=")
    if not separator or not label or not raw_path:
        raise argparse.ArgumentTypeError("--run must be LABEL=/absolute/run/root")
    if any(
        character not in "abcdefghijklmnopqrstuvwxyz0123456789-_" for character in label
    ):
        raise argparse.ArgumentTypeError("run label is not path-safe lowercase")
    path = Path(raw_path).expanduser()
    if not path.is_absolute():
        raise argparse.ArgumentTypeError("run root must be absolute")
    return label, path


def build_report(runs: list[tuple[str, Path]]) -> dict[str, object]:
    if not runs or len({label for label, _ in runs}) != len(runs):
        raise DiagnosisError("run labels must be non-empty and unique")
    return {
        "schema_version": SCHEMA_VERSION,
        "report": REPORT,
        "runs": [_diagnose_run(label, root) for label, root in runs],
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", action="append", type=_parse_run, required=True)
    parser.add_argument("--output", type=Path)
    arguments = parser.parse_args(argv)
    try:
        report = build_report(arguments.run)
    except (DiagnosisError, OSError, ValueError) as exc:
        print(json.dumps({"status": "error", "error": str(exc)}), file=sys.stderr)
        return 2
    payload = json.dumps(report, indent=2, sort_keys=True, allow_nan=False) + "\n"
    if arguments.output is not None:
        output = arguments.output.expanduser().resolve()
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(payload, encoding="utf-8")
    print(payload, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
