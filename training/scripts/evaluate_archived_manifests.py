#!/usr/bin/env python3
"""Select an archived manifest using isolated, pre-registered ring-10 arenas."""

from __future__ import annotations

import argparse
import json
import multiprocessing
import os
import sys
import time
from collections.abc import Callable, Mapping
from concurrent.futures import ProcessPoolExecutor
from dataclasses import asdict, replace
from pathlib import Path
from typing import Any

from startrain.arena import ArenaPair, ArenaRunner, summarize_arena_pairs
from startrain.checkpoint import load_model_manifest
from startrain.config import ArenaConfig, ExperimentConfig, load_config
from startrain.device import (
    empty_device_cache,
    resolve_device_string,
    synchronize_device,
)
from startrain.manifest_selection import (
    RESULT_KIND,
    ManifestEvidence,
    ManifestSelectionError,
    PersistedSelectionResult,
    SelectionContract,
    SelectionEvidence,
    SelectionPlan,
    build_selection_evidence,
    build_selection_plan,
    candidate_seed,
    freeze_selection_evidence,
    freeze_selection_plan,
    load_persisted_selection_result,
    next_selection_pair_count,
    verify_selection_plan,
    verify_selection_snapshot,
)
from startrain.native import load_star_native
from startrain.promotion import load_manifest_evaluator
from startrain.runtime import atomic_json, load_run_identity

DEFAULT_DISCOVERED_SHORTLIST_SIZE = 8
PLAN_NAME = "selection-plan.json"
SNAPSHOT_NAME = "selection-snapshot.json"
RESULTS_DIRECTORY = "results"


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-run-root", type=Path, required=True)
    parser.add_argument("--profile", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--candidate-manifest",
        type=Path,
        action="append",
        default=[],
        help="explicit archived manifest; repeat to freeze a shortlist",
    )
    parser.add_argument(
        "--shortlist-size",
        type=int,
        help=(
            "keep the newest N eligible manifests; defaults to all explicit "
            f"manifests or {DEFAULT_DISCOVERED_SHORTLIST_SIZE} discovered manifests"
        ),
    )
    parser.add_argument(
        "--device",
        action="append",
        default=[],
        help=(
            "static evaluation device; repeat for bounded parallel assignment "
            "(default: auto)"
        ),
    )
    parser.add_argument(
        "--plan-only",
        action="store_true",
        help="freeze and verify the shortlist without loading native/CUDA evaluators",
    )
    return parser


def _isolated_output(source_run_root: Path, output_directory: Path) -> Path:
    source = source_run_root.expanduser().resolve()
    output = output_directory.expanduser().resolve()
    if output == source or source in output.parents:
        raise ManifestSelectionError(
            "selection output must be outside the parent run root"
        )
    return output


def deterministic_shortlist(
    source_run_root: str | Path,
    *,
    candidate_manifest_paths: tuple[str | Path, ...] = (),
    shortlist_size: int | None = None,
) -> tuple[tuple[Path, ...], str]:
    """Return an immutable-manifest shortlist independent of directory order."""

    root = Path(source_run_root).expanduser().resolve()
    identity = load_run_identity(root / "run.json")
    champion = load_model_manifest(root / "learner" / "champion.json")
    explicit = bool(candidate_manifest_paths)
    paths = (
        [Path(path).expanduser().resolve() for path in candidate_manifest_paths]
        if explicit
        else sorted((root / "learner" / "manifests").glob("manifest-*.json"))
    )
    manifests = {}
    for path in paths:
        manifest = load_model_manifest(path)
        if (
            manifest.run_id != identity.run_id
            or manifest.generation_family != identity.generation_family
        ):
            if explicit:
                raise ManifestSelectionError(
                    "explicit candidate belongs to another run or generation"
                )
            continue
        if manifest.model_identity == champion.model_identity:
            continue
        artifact = (manifest.artifact_manifest or manifest.path).resolve()
        manifests[manifest.model_identity] = (manifest, artifact)
    ordered = sorted(
        manifests.values(),
        key=lambda item: (
            item[0].model_step,
            item[0].model_identity,
            str(item[1]),
        ),
    )
    limit = shortlist_size
    if limit is None and not explicit:
        limit = DEFAULT_DISCOVERED_SHORTLIST_SIZE
    if limit is not None:
        if isinstance(limit, bool) or not isinstance(limit, int) or limit <= 0:
            raise ManifestSelectionError("shortlist size must be a positive integer")
        ordered = ordered[-limit:]
    if not ordered:
        raise ManifestSelectionError("no eligible archived manifests were found")
    method = (
        "explicit-manifests-sorted-by-step-and-identity-v1"
        if explicit and shortlist_size is None
        else "newest-n-by-step-and-identity-v1"
    )
    return tuple(item[1] for item in ordered), method


def selection_contract(
    experiment: ExperimentConfig,
    *,
    shortlist_size: int,
) -> SelectionContract:
    """Materialize exact profile budgets with Bonferroni familywise control."""

    arena = experiment.arena
    if 10 not in arena.rings or 10 not in experiment.game.rings:
        raise ManifestSelectionError("evaluation profile does not support ring 10")
    continuation = arena.continuation_pairs_per_ring or arena.pairs_per_ring
    per_candidate_alpha = arena.alpha / shortlist_size
    per_candidate_beta = arena.beta / shortlist_size
    return SelectionContract(
        initial_pairs=arena.pairs_per_ring,
        continuation_pairs=continuation,
        minimum_pairs=arena.minimum_pairs_per_ring,
        max_pairs=arena.max_pairs_per_ring,
        simulations=arena.simulations,
        max_considered=arena.max_considered,
        c_visit=arena.c_visit,
        c_scale=arena.c_scale,
        pair_chunk_size=arena.pair_chunk_size,
        bootstrap_samples=arena.bootstrap_samples,
        unforced_opening_fraction=arena.unforced_opening_fraction,
        shortlist_size=shortlist_size,
        familywise_alpha=arena.alpha,
        per_candidate_alpha=per_candidate_alpha,
        familywise_beta=arena.beta,
        per_candidate_beta=per_candidate_beta,
        confidence=1.0 - 2.0 * per_candidate_alpha,
    )


def plan_archived_manifest_evaluation(
    *,
    source_run_root: str | Path,
    profile: str | Path,
    candidate_manifest_paths: tuple[str | Path, ...] = (),
    shortlist_size: int | None = None,
) -> SelectionPlan:
    """Build a deterministic selection plan without requiring native code/CUDA."""

    paths, method = deterministic_shortlist(
        source_run_root,
        candidate_manifest_paths=candidate_manifest_paths,
        shortlist_size=shortlist_size,
    )
    experiment = load_config(profile)
    contract = selection_contract(experiment, shortlist_size=len(paths))
    return build_selection_plan(
        source_run_root=source_run_root,
        evaluation_profile=profile,
        candidate_manifest_paths=paths,
        contract=contract,
        shortlist_method=method,
    )


def selection_arena_config(
    experiment: ExperimentConfig,
    plan: SelectionPlan,
    candidate: ManifestEvidence,
) -> ArenaConfig:
    """Create the fixed ring-10 arena config recorded by the frozen plan."""

    contract = plan.contract
    return replace(
        experiment.arena,
        rings=(10,),
        pairs_per_ring=contract.initial_pairs,
        continuation_pairs_per_ring=contract.continuation_pairs,
        simulations=contract.simulations,
        max_considered=contract.max_considered,
        c_visit=contract.c_visit,
        c_scale=contract.c_scale,
        seed=candidate_seed(plan, candidate.model_identity),
        null_elo=contract.improvement_threshold_elo,
        alpha=contract.per_candidate_alpha,
        beta=contract.per_candidate_beta,
        regression_floor_elo=contract.improvement_threshold_elo,
        per_ring_regression_floor_elo={},
        promotion_pair_ratios={},
        required_regression_rings=(),
        weighted_initial_blocks=0,
        weighted_continuation_blocks=0,
        weighted_max_blocks=0,
        confidence=contract.confidence,
        bootstrap_samples=contract.bootstrap_samples,
        unforced_opening_fraction=contract.unforced_opening_fraction,
        minimum_pairs_per_ring=contract.minimum_pairs,
        max_pairs_per_ring=contract.max_pairs,
    )


def _pairs_from_result(result: dict[str, object]) -> list[ArenaPair]:
    raw_pairs = result.get("pairs")
    if not isinstance(raw_pairs, list):
        raise ManifestSelectionError("arena result omitted complete pairs")
    output = []
    for raw in raw_pairs:
        if not isinstance(raw, dict):
            raise ManifestSelectionError("arena result pair is malformed")
        values = dict(raw)
        outcomes = values.get("outcomes")
        if not isinstance(outcomes, list | tuple) or len(outcomes) != 2:
            raise ManifestSelectionError("arena result pair outcomes are malformed")
        values["outcomes"] = (int(outcomes[0]), int(outcomes[1]))
        output.append(ArenaPair(**values))
    return output


def _ring_10_lower_elo(result: dict[str, object]) -> float:
    per_ring = result.get("per_ring")
    if not isinstance(per_ring, dict):
        raise ManifestSelectionError("arena result omitted ring-10 summary")
    summary = per_ring.get("10")
    if not isinstance(summary, dict):
        raise ManifestSelectionError("arena result omitted ring-10 summary")
    interval = summary.get("anytime_elo_interval")
    if (
        not isinstance(interval, list | tuple)
        or len(interval) != 2
        or isinstance(interval[0], bool)
        or not isinstance(interval[0], int | float)
    ):
        raise ManifestSelectionError("arena result ring-10 interval is malformed")
    return float(interval[0])


def _evaluate_candidate(
    *,
    experiment: ExperimentConfig,
    plan: SelectionPlan,
    candidate: ManifestEvidence,
    baseline_evaluator: Any,
    native_module: Any,
    device: str,
    output_path: Path,
    persisted: PersistedSelectionResult | None = None,
) -> Path:
    verified_plan = verify_selection_plan(plan)
    candidate_manifest = candidate.verify()
    arena_config = selection_arena_config(experiment, verified_plan, candidate)
    candidate_evaluator = load_manifest_evaluator(
        experiment,
        candidate_manifest,
        device=device,
    )
    accumulated = list(persisted.pairs) if persisted is not None else []
    persisted_payload = persisted.payload if persisted is not None else {}
    raw_persisted_games = persisted_payload.get("games", [])
    raw_persisted_history = persisted_payload.get("wave_history", [])
    if not isinstance(raw_persisted_games, list) or not isinstance(
        raw_persisted_history,
        list,
    ):
        raise ManifestSelectionError("persisted selection progress is malformed")
    games: list[object] = list(raw_persisted_games)
    wave_history = [dict(item) for item in raw_persisted_history]
    raw_started_ns = persisted_payload.get("evaluation_started_ns")
    evaluation_started_ns = (
        raw_started_ns if isinstance(raw_started_ns, int) else time.time_ns()
    )
    runner: ArenaRunner | None = None
    try:
        runner = ArenaRunner(
            native_module=native_module,
            candidate=candidate_evaluator,
            baseline=baseline_evaluator,
            config=arena_config,
        )
        while len(accumulated) < plan.contract.max_pairs:
            pair_count = next_selection_pair_count(
                plan.contract,
                len(accumulated),
            )
            wave = runner.run(
                pair_starts={10: len(accumulated)},
                pair_counts={10: pair_count},
            )
            completed = _pairs_from_result(wave)
            if len(completed) != pair_count:
                raise RuntimeError(
                    "archived-manifest arena did not complete its frozen pair wave"
                )
            accumulated.extend(completed)
            raw_games = wave.get("games")
            if not isinstance(raw_games, list):
                raise ManifestSelectionError("arena result games are malformed")
            games.extend(raw_games)
            metrics = wave.get("evaluation_metrics")
            wave_history.append(
                {
                    "wave_index": len(wave_history),
                    "pair_start": len(accumulated) - len(completed),
                    "pair_count": len(completed),
                    "completed_pairs": len(accumulated),
                    "evaluation_metrics": metrics if isinstance(metrics, dict) else {},
                }
            )
            result = dict(wave)
            result.update(summarize_arena_pairs(accumulated, arena_config))
            lower_elo = _ring_10_lower_elo(result)
            proven = lower_elo > plan.contract.improvement_threshold_elo
            terminal = (
                proven and len(accumulated) >= plan.contract.minimum_pairs
            ) or len(accumulated) == plan.contract.max_pairs
            result.update(
                {
                    "result_kind": RESULT_KIND,
                    "selection_plan_digest": plan.plan_digest,
                    "selection_contract": plan.contract.as_dict(),
                    "candidate": candidate.model_identity,
                    "baseline": plan.source_champion.manifest.model_identity,
                    "candidate_manifest": candidate.manifest.path,
                    "baseline_manifest": (plan.source_champion.manifest.manifest.path),
                    "arena_seed_block": arena_config.seed,
                    "evaluation_started_ns": evaluation_started_ns,
                    "pairs": [asdict(pair) for pair in accumulated],
                    "games": games,
                    "wave_history": wave_history,
                    "statistically_proven_improvement": proven,
                    "selection_decision": (
                        "proven_improvement" if proven else "not_proven"
                    ),
                    "terminal": terminal,
                }
            )
            atomic_json(output_path, result)
            if terminal:
                os.chmod(output_path, 0o444)
                return output_path
    finally:
        del runner
        del candidate_evaluator
        synchronize_device(device)
        empty_device_cache(device)
    raise RuntimeError("archived-manifest arena exhausted without a terminal result")


def static_device_assignments(
    candidates: tuple[ManifestEvidence, ...],
    devices: tuple[str, ...],
) -> tuple[tuple[str, tuple[ManifestEvidence, ...]], ...]:
    """Assign each candidate to one bounded, sequential per-device lane."""

    if not devices:
        raise ManifestSelectionError("at least one evaluation device is required")
    if len(set(devices)) != len(devices):
        raise ManifestSelectionError("evaluation devices repeat the same physical lane")
    lanes: list[list[ManifestEvidence]] = [[] for _ in devices]
    for index, candidate in enumerate(candidates):
        lanes[index % len(devices)].append(candidate)
    return tuple(
        (device, tuple(lane))
        for device, lane in zip(devices, lanes, strict=True)
        if lane
    )


def canonicalize_device_lanes(
    devices: tuple[str, ...],
    *,
    resolver: Callable[[str], str] = resolve_device_string,
) -> tuple[str, ...]:
    """Resolve aliases and reject duplicate assignments to one physical lane."""

    if not devices:
        raise ManifestSelectionError("at least one evaluation device is required")
    resolved: list[str] = []
    for requested in devices:
        device = resolver(requested)
        device_type, separator, raw_index = device.partition(":")
        if device_type == "cuda":
            index = int(raw_index) if separator else 0
            canonical = f"cuda:{index}"
        elif device_type in {"cpu", "mps"}:
            canonical = device_type
        else:
            raise ManifestSelectionError(
                f"resolved evaluation device is unsupported: {device}"
            )
        if canonical in resolved:
            raise ManifestSelectionError(
                f"evaluation device aliases resolve to duplicate lane {canonical}"
            )
        resolved.append(canonical)
    return tuple(resolved)


def _evaluate_device_lane(
    *,
    experiment: ExperimentConfig,
    plan: SelectionPlan,
    candidates: tuple[ManifestEvidence, ...],
    native_module: Any,
    device: str,
    results_directory: Path,
) -> dict[str, Path]:
    baseline = plan.source_champion.manifest.verify()
    baseline_evaluator = load_manifest_evaluator(experiment, baseline, device=device)
    output: dict[str, Path] = {}
    try:
        for candidate in candidates:
            path = results_directory / f"{candidate.model_identity}.json"
            persisted: PersistedSelectionResult | None = None
            if path.exists() or path.is_symlink():
                persisted = load_persisted_selection_result(plan, candidate, path)
                if persisted.terminal:
                    os.chmod(path, 0o444)
                    output[candidate.model_identity] = path
                    continue
            output[candidate.model_identity] = _evaluate_candidate(
                experiment=experiment,
                plan=plan,
                candidate=candidate,
                baseline_evaluator=baseline_evaluator,
                native_module=native_module,
                device=device,
                output_path=path,
                persisted=persisted,
            )
    finally:
        del baseline_evaluator
        synchronize_device(device)
        empty_device_cache(device)
    return output


def _evaluate_device_lane_process(
    profile: str,
    plan_path: str,
    candidate_identities: tuple[str, ...],
    device: str,
    results_directory: str,
) -> dict[str, Path]:
    """Run one GPU lane in an isolated process so compilers cannot deadlock."""

    plan = verify_selection_plan(plan_path)
    candidates_by_identity = {
        candidate.model_identity: candidate for candidate in plan.candidates
    }
    try:
        candidates = tuple(
            candidates_by_identity[identity] for identity in candidate_identities
        )
    except KeyError as exc:
        raise ManifestSelectionError(
            f"device lane references an unknown candidate: {exc.args[0]}"
        ) from exc
    native = load_star_native(required=True)
    assert native is not None
    return _evaluate_device_lane(
        experiment=load_config(profile),
        plan=plan,
        candidates=candidates,
        native_module=native,
        device=device,
        results_directory=Path(results_directory),
    )


def finalize_archived_manifest_selection(
    *,
    plan_path: str | Path,
    result_paths: Mapping[str, str | Path],
    snapshot_path: str | Path,
) -> SelectionEvidence:
    """Build and freeze final evidence without loading native code or CUDA."""

    evidence = build_selection_evidence(
        plan_path=plan_path,
        result_paths=result_paths,
    )
    try:
        freeze_selection_evidence(snapshot_path, evidence)
    except FileExistsError:
        pass
    verified = verify_selection_snapshot(
        snapshot_path,
        expected_source_run_root=evidence.plan.source_run_root,
    )
    if verified.selected_evidence != evidence.selected:
        raise ManifestSelectionError(
            "existing selection snapshot differs from recomputed evidence"
        )
    return evidence


def evaluate_archived_manifests(
    *,
    source_run_root: str | Path,
    profile: str | Path,
    output_directory: str | Path,
    candidate_manifest_paths: tuple[str | Path, ...] = (),
    shortlist_size: int | None = None,
    devices: tuple[str, ...] = ("auto",),
    plan_only: bool = False,
) -> dict[str, object]:
    """Freeze a plan, run isolated arenas, then freeze selection evidence."""

    source = Path(source_run_root).expanduser().resolve()
    output = _isolated_output(source, Path(output_directory))
    expected_plan = plan_archived_manifest_evaluation(
        source_run_root=source,
        profile=profile,
        candidate_manifest_paths=candidate_manifest_paths,
        shortlist_size=shortlist_size,
    )
    plan_path = output / PLAN_NAME
    plan_reused = False
    if output.is_symlink():
        raise ManifestSelectionError("selection output may not be a symbolic link")
    if output.exists():
        if not output.is_dir():
            raise ManifestSelectionError("selection output is not a directory")
        if not plan_path.is_file() or plan_path.is_symlink():
            raise ManifestSelectionError(
                "existing selection output does not contain a frozen plan"
            )
        plan = verify_selection_plan(
            plan_path,
            expected_source_run_root=source,
        )
        if plan.as_dict() != expected_plan.as_dict():
            raise ManifestSelectionError(
                "existing frozen selection plan differs from this command"
            )
        plan_reused = True
    else:
        output.mkdir(parents=True)
        plan = expected_plan
        freeze_selection_plan(plan_path, plan)

    snapshot_path = output / SNAPSHOT_NAME
    if snapshot_path.exists() or snapshot_path.is_symlink():
        verified = verify_selection_snapshot(
            snapshot_path,
            expected_source_run_root=source,
        )
        if verified.plan.plan_digest != plan.plan_digest:
            raise ManifestSelectionError(
                "existing selection snapshot belongs to another frozen plan"
            )
        return {
            "status": "verified",
            "plan": str(plan_path),
            "plan_digest": plan.plan_digest,
            "snapshot": str(snapshot_path),
            "selected_identity": verified.selected_manifest.model_identity,
            "fallback_to_champion": verified.fallback_used,
            "fallback_reason": verified.fallback_reason,
            "candidate_count": len(plan.candidates),
            "devices": [],
            "plan_reused": plan_reused,
            "snapshot_reused": True,
            "parent_run_mutated": False,
        }
    if plan_only:
        return {
            "status": "planned",
            "plan": str(plan_path),
            "plan_digest": plan.plan_digest,
            "evaluation_seed": plan.evaluation_seed,
            "candidate_count": len(plan.candidates),
            "plan_reused": plan_reused,
            "parent_run_mutated": False,
        }

    results_directory = output / RESULTS_DIRECTORY
    if results_directory.is_symlink():
        raise ManifestSelectionError(
            "selection results directory may not be a symbolic link"
        )
    results_directory.mkdir(exist_ok=True)
    result_paths: dict[str, Path] = {}
    pending: list[ManifestEvidence] = []
    for candidate in plan.candidates:
        result_path = results_directory / f"{candidate.model_identity}.json"
        if result_path.exists() or result_path.is_symlink():
            persisted = load_persisted_selection_result(
                plan,
                candidate,
                result_path,
            )
            if persisted.terminal:
                os.chmod(result_path, 0o444)
                result_paths[candidate.model_identity] = result_path
            else:
                pending.append(candidate)
        else:
            pending.append(candidate)

    resolved_devices: tuple[str, ...] = ()
    if pending:
        resolved_devices = canonicalize_device_lanes(devices)
        assignments = static_device_assignments(tuple(pending), resolved_devices)
        if len(assignments) == 1:
            experiment = load_config(profile)
            native = load_star_native(required=True)
            assert native is not None
            device, candidates = assignments[0]
            result_paths.update(
                _evaluate_device_lane(
                    experiment=experiment,
                    plan=plan,
                    candidates=candidates,
                    native_module=native,
                    device=device,
                    results_directory=results_directory,
                )
            )
        else:
            arguments = [
                (
                    str(Path(profile).expanduser().resolve()),
                    str(plan_path),
                    tuple(candidate.model_identity for candidate in candidates),
                    device,
                    str(results_directory),
                )
                for device, candidates in assignments
            ]
            with ProcessPoolExecutor(
                max_workers=len(assignments),
                mp_context=multiprocessing.get_context("spawn"),
            ) as executor:
                futures = [
                    executor.submit(_evaluate_device_lane_process, *argument)
                    for argument in arguments
                ]
                for future in futures:
                    lane_results = future.result()
                    result_paths.update(lane_results)
    finalize_archived_manifest_selection(
        plan_path=plan_path,
        result_paths=result_paths,
        snapshot_path=snapshot_path,
    )
    verified = verify_selection_snapshot(
        snapshot_path,
        expected_source_run_root=source,
    )
    return {
        "status": "verified",
        "plan": str(plan_path),
        "plan_digest": plan.plan_digest,
        "snapshot": str(snapshot_path),
        "selected_identity": verified.selected_manifest.model_identity,
        "fallback_to_champion": verified.fallback_used,
        "fallback_reason": verified.fallback_reason,
        "candidate_count": len(plan.candidates),
        "devices": list(resolved_devices),
        "plan_reused": plan_reused,
        "snapshot_reused": False,
        "parent_run_mutated": False,
    }


def main(argv: list[str] | None = None) -> int:
    try:
        arguments = _parser().parse_args(argv)
        report = evaluate_archived_manifests(
            source_run_root=arguments.source_run_root,
            profile=arguments.profile,
            output_directory=arguments.output_dir,
            candidate_manifest_paths=tuple(arguments.candidate_manifest),
            shortlist_size=arguments.shortlist_size,
            devices=tuple(arguments.device or ("auto",)),
            plan_only=arguments.plan_only,
        )
    except Exception as exc:
        print(
            json.dumps(
                {
                    "status": "error",
                    "error_type": type(exc).__name__,
                    "error": str(exc),
                },
                sort_keys=True,
            ),
            file=sys.stderr,
        )
        return 2
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
