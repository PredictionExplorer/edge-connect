#!/usr/bin/env python3
"""Frozen-position budget comparisons, with an explicit bounded execution step.

Positions JSON: {"schema_version":1,"positions":[{"id":"opening","rings":6,
"actions":[],"seed":17,"mode":"double","handicap":1,"pie":false,"pda":0}]}.
Actions are authoritative native action codes replayed from the initial state.
Results measure search disagreement and cost, never playing strength or Elo.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass, replace
import hashlib
import json
import math
import os
from pathlib import Path
import signal
import subprocess
import sys
import time
from typing import Any

from startrain.checkpoint import load_model_manifest
from startrain.config import ActorInferenceConfig, load_config
from startrain.contracts import (
    FEATURE_SCHEMA_VERSION,
    RULES_HASH_WIRE,
    SEARCH_ALGORITHM_ID,
)
from startrain.native import load_star_native
from startrain.promotion import load_manifest_evaluator
from startrain.search_options import (
    FullSearchBudgetConfig,
    SearchExecutionConfig,
    normalized_root_entropy,
    require_search_execution,
)
from startrain.selfplay import GameVariant
from startrain.topology import SUPPORTED_RINGS


@dataclass(frozen=True)
class FrozenPosition:
    id: str
    rings: int
    actions: tuple[int, ...]
    seed: int = 17
    mode: str = "double"
    handicap: int = 1
    pie: bool = False
    pda: int = 0

    def __post_init__(self):
        if not isinstance(self.id, str) or not 1 <= len(self.id) <= 128:
            raise ValueError("position id must contain 1..128 characters")
        if type(self.rings) is not int or self.rings not in SUPPORTED_RINGS:
            raise ValueError("position rings must be 4, 6, 8 or 10")
        if type(self.seed) is not int or not 0 <= self.seed < 2**64:
            raise ValueError("position seed must be a uint64")
        if type(self.pda) is not int or not -3 <= self.pda <= 3:
            raise ValueError("position PDA must be in -3..3")
        GameVariant(mode=self.mode, handicap=self.handicap, pie=self.pie)
        nodes = 5 * self.rings * (self.rings + 1) // 2
        if len(self.actions) > nodes + 1 or any(
            type(action) is not int or not 0 <= action <= nodes
            for action in self.actions
        ):
            raise ValueError("position actions exceed the board's bounded action space")


def read_positions(path: Path) -> tuple[tuple[FrozenPosition, ...], str]:
    if path.stat().st_size > 4 * 1024**2:
        raise ValueError("frozen positions file exceeds 4 MiB")
    contents = path.read_bytes()
    raw = json.loads(contents)
    if not isinstance(raw, dict) or set(raw) != {"schema_version", "positions"}:
        raise ValueError("positions require schema_version and positions fields")
    if raw["schema_version"] != 1 or not isinstance(raw["positions"], list):
        raise ValueError("unsupported frozen positions schema")
    if not 1 <= len(raw["positions"]) <= 256:
        raise ValueError("a sweep requires 1..256 frozen positions")
    positions = []
    for row in raw["positions"]:
        if not isinstance(row, dict) or not isinstance(row.get("actions"), list):
            raise ValueError("each frozen position requires an actions list")
        positions.append(FrozenPosition(**{**row, "actions": tuple(row["actions"])}))
    if len({position.id for position in positions}) != len(positions):
        raise ValueError("frozen position ids must be unique")
    return tuple(positions), hashlib.sha256(contents).hexdigest()


def _state(native: Any, position: FrozenPosition):
    states = native.StateBatch(
        position.rings,
        1,
        mode=position.mode,
        handicap=position.handicap,
        pie=position.pie,
    )
    if position.actions:
        states.apply_many([0] * len(position.actions), list(position.actions))
    if states.data().terminal[0]:
        raise ValueError(f"frozen position {position.id!r} is terminal")
    return states


def search_position(
    native: Any,
    evaluator: Any,
    position: FrozenPosition,
    selfplay: Any,
    cap: int,
    budget_policy: FullSearchBudgetConfig,
    *,
    first_visit_batch_size: int = 1,
    deadline: float,
) -> dict[str, Any]:
    execution = SearchExecutionConfig(
        first_visit_batch_size=first_visit_batch_size, full_budget=budget_policy
    )
    require_search_execution(native, execution)
    local = replace(selfplay, rings=position.rings)
    fast = min(cap, local.simulation_budget(full=False))
    local = replace(
        local,
        fast_simulations=fast,
        full_simulations=cap,
        simulation_ring_exponent=0,
        mode=position.mode,
        handicap=position.handicap,
        pie=position.pie,
    )
    evaluator.set_score_utility_weight(local.effective_score_utility_weight())
    states = _state(native, position)
    side = states.data().to_move[0]
    seats = (
        (position.pda, -position.pda) if side == 0 else (-position.pda, position.pda)
    )

    def seat_budget(full_cap):
        high, low = local.playout_budgets(
            simulations=full_cap, pda=abs(position.pda), full_cap=full_cap
        )
        return low if position.pda < 0 else high

    actual = seat_budget(cap)
    options = (
        {"first_visit_batch_size": first_visit_batch_size}
        if first_visit_batch_size > 1
        else {}
    )
    search = native.SearchBatch(
        states,
        simulations=cap,
        simulations_per_root=[actual],
        max_considered=local.considered_actions(),
        c_visit=local.c_visit,
        c_scale=local.c_scale,
        deterministic_seed=position.seed,
        pda_by_seat=[seats],
        **options,
    )
    evaluator.clear_inference_cache()
    before = evaluator.metrics_snapshot()
    started = time.monotonic()
    roots = search.root_requests()
    response = evaluator.evaluate(roots)
    entropy = normalized_root_entropy(response.policy_logits)
    effective_cap = budget_policy.adjusted_cap(
        cap,
        entropy,
        minimum_cap=fast * 2 ** abs(position.pda),
        quantum=2 ** abs(position.pda),
    )
    if budget_policy.mode != "fixed":
        actual = seat_budget(effective_cap)
        search.set_simulations_per_root([actual])
    search.initialize_roots(*response.submit_args())
    while not search.is_done():
        if time.monotonic() >= deadline:
            raise TimeoutError("frozen-position search sweep exceeded its deadline")
        requests = (
            search.next_requests(max_rows=first_visit_batch_size)
            if first_visit_batch_size > 1
            else search.next_requests()
        )
        if len(requests):
            search.submit(*evaluator.evaluate(requests).submit_args())
    result = search.results()
    elapsed = time.monotonic() - started
    metrics = asdict(evaluator.metrics_snapshot().delta(before))
    return {
        "position_id": position.id,
        "base_cap": cap,
        "effective_cap": effective_cap,
        "actual_simulations": actual,
        "root_entropy": entropy,
        "budget_policy": asdict(budget_policy),
        "seconds": elapsed,
        "selected_action": int(result.selected_actions[0]),
        "selected_value": float(result.selected_action_values[0]),
        "root_value": float(result.root_values[0]),
        "actions": list(result.actions),
        "policy_target": list(result.policy_target),
        "q_values": list(result.q_values),
        "visits": list(result.visits),
        "inference": metrics,
    }


def compare_search(
    candidate: dict[str, Any], reference: dict[str, Any]
) -> dict[str, Any]:
    if candidate["actions"] != reference["actions"]:
        raise ValueError("frozen-position arms returned different action spaces")
    index = reference["actions"].index(candidate["selected_action"])
    return {
        "action_matches_reference": candidate["selected_action"]
        == reference["selected_action"],
        "policy_target_l1": math.fsum(
            abs(left - right)
            for left, right in zip(
                candidate["policy_target"], reference["policy_target"], strict=True
            )
        ),
        "selected_value_absolute_difference": abs(
            candidate["selected_value"] - reference["selected_value"]
        ),
        "reference_selected_q_minus_candidate_action_q": reference["selected_value"]
        - reference["q_values"][index],
    }


def _parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument(
        "--checkpoint",
        type=Path,
        required=True,
        help="immutable model manifest or publication pointer",
    )
    parser.add_argument("--positions", type=Path, required=True)
    parser.add_argument("--caps", type=int, nargs="+", default=[128, 256, 384, 640])
    parser.add_argument("--reference-cap", type=int, default=1024)
    parser.add_argument("--entropy-minimum-fraction", type=float, default=0.5)
    parser.add_argument("--entropy-threshold", type=float, default=0.35)
    parser.add_argument("--first-visit-batch-size", type=int, default=1)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--repeats", type=int, default=1)
    parser.add_argument("--timeout-seconds", type=float, default=300)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--pinned-plan", help=argparse.SUPPRESS)
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = _parser()
    arguments = list(sys.argv[1:] if argv is None else argv)
    args = parser.parse_args(arguments)
    if (
        not 1 <= len(args.caps) <= 6
        or len(set(args.caps)) != len(args.caps)
        or any(not 1 <= cap <= 4096 for cap in args.caps)
        or not max(args.caps) < args.reference_cap <= 8192
        or not 1 <= args.repeats <= 5
        or not math.isfinite(args.timeout_seconds)
        or not 0 < args.timeout_seconds <= 600
    ):
        parser.error(
            "require1..6 distinct caps1..4096, deeper reference<=8192, repeats1..5, timeout(0,600]"
        )
    entropy_policy = FullSearchBudgetConfig(
        mode="root-entropy",
        minimum_fraction=args.entropy_minimum_fraction,
        entropy_threshold=args.entropy_threshold,
    )
    SearchExecutionConfig(first_visit_batch_size=args.first_visit_batch_size)
    positions, positions_sha = read_positions(args.positions)
    config_sha = hashlib.sha256(args.config.read_bytes()).hexdigest()
    config = load_config(args.config)
    manifest = load_model_manifest(args.checkpoint)
    plan = {
        "schema_version": 1,
        "search_algorithm": SEARCH_ALGORITHM_ID,
        "rules_hash": RULES_HASH_WIRE,
        "feature_schema_version": FEATURE_SCHEMA_VERSION,
        "positions_sha256": positions_sha,
        "config_sha256": config_sha,
        "model_identity": manifest.model_identity,
        "manifest_sha256": manifest.manifest_sha256,
        "checkpoint_sha256": manifest.checkpoint_sha256,
        "positions": [asdict(position) for position in positions],
        "caps": args.caps,
        "reference_cap": args.reference_cap,
        "entropy_variant_base_cap": max(args.caps),
        "entropy_policy": asdict(entropy_policy),
        "first_visit_batch_size": args.first_visit_batch_size,
        "repeats": args.repeats,
        "device": args.device,
        "runtime": "eager; prediction cache, deduplication and CUDA graphs disabled",
        "configured_precision": config.train.precision,
        "scope": "frozen-position search cost and disagreement; not Elo or training strength",
    }
    plan_sha = hashlib.sha256(json.dumps(plan, sort_keys=True).encode()).hexdigest()
    if args.pinned_plan is not None and plan_sha != args.pinned_plan:
        raise ValueError("frozen search plan inputs changed before execution")
    if not args.execute:
        print(json.dumps({**plan, "plan_sha256": plan_sha}, indent=2))
        return 0
    if args.output is None or args.output.exists():
        parser.error("--execute requires a new --output path")
    if not args.worker:
        command = [
            sys.executable,
            str(Path(__file__).resolve()),
            *arguments,
            "--worker",
            "--pinned-plan",
            plan_sha,
        ]
        process = subprocess.Popen(
            command,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            start_new_session=True,
        )
        try:
            stdout, stderr = process.communicate(timeout=args.timeout_seconds)
        except subprocess.TimeoutExpired as exc:
            try:
                os.killpg(process.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
            process.communicate()
            raise TimeoutError("search sweep exceeded its process deadline") from exc
        except BaseException:
            try:
                os.killpg(process.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
            process.communicate()
            raise
        if process.returncode:
            raise RuntimeError(f"search sweep failed: {stderr[-4000:]}")
        print(stdout, end="")
        return 0
    native = load_star_native(required=True)
    assert native is not None
    benchmark_config = replace(
        config,
        train=replace(config.train, compile=False),
        orchestration=replace(
            config.orchestration,
            model_refresh=replace(
                config.orchestration.model_refresh, inference=ActorInferenceConfig()
            ),
        ),
    )
    evaluator = load_manifest_evaluator(benchmark_config, manifest, device=args.device)
    deadline = time.monotonic() + args.timeout_seconds
    records = []
    try:
        for position in positions:
            # Warm the actual root shape before any measured arm, then clear
            # prediction storage in search_position so every arm starts cold.
            warm = native.SearchBatch(
                _state(native, position), simulations=1, max_considered=1
            )
            evaluator.evaluate(warm.root_requests())
            for repeat in range(args.repeats):
                reference = search_position(
                    native,
                    evaluator,
                    position,
                    config.selfplay,
                    args.reference_cap,
                    FullSearchBudgetConfig(),
                    first_visit_batch_size=args.first_visit_batch_size,
                    deadline=deadline,
                )
                records.append({**reference, "arm": "reference", "repeat": repeat})
                arms = [
                    (f"fixed-{cap}", cap, FullSearchBudgetConfig()) for cap in args.caps
                ]
                arms.append(("root-entropy", max(args.caps), entropy_policy))
                # Rotate arm order across repeats without changing state/seed.
                arms = arms[repeat % len(arms) :] + arms[: repeat % len(arms)]
                for name, cap, policy in arms:
                    result = search_position(
                        native,
                        evaluator,
                        position,
                        config.selfplay,
                        cap,
                        policy,
                        first_visit_batch_size=args.first_visit_batch_size,
                        deadline=deadline,
                    )
                    records.append(
                        {
                            **result,
                            "arm": name,
                            "repeat": repeat,
                            "comparison": compare_search(result, reference),
                        }
                    )
    finally:
        evaluator.close()
    artifact = {
        "plan": plan,
        "plan_sha256": plan_sha,
        "runtime": {
            "device": str(evaluator.device),
            "precision": evaluator.config.precision,
        },
        "results": records,
    }
    with args.output.open("x", encoding="utf-8") as stream:
        json.dump(artifact, stream, sort_keys=True, indent=2, allow_nan=False)
        stream.write("\n")
    print(
        json.dumps(
            {
                "output": str(args.output),
                "records": len(records),
                "plan_sha256": plan_sha,
            }
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
