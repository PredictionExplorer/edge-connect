#!/usr/bin/env python3
"""Generate frozen, one-factor H100 Elo-ablation profiles and manifests."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections.abc import Callable, Mapping, Sequence
from copy import deepcopy
from pathlib import Path
from typing import Any

import yaml

from startrain.config import ExperimentConfig, load_config

SCHEMA_VERSION = 1
REPORT_NAME = "startrain-elo-ablation-plan"
DEFAULT_TREATMENTS = (
    "control",
    "utd-1",
    "plateau-keep",
    "freshness-mix",
    "ring10-70",
    "search-quality",
)
SYSTEM_TREATMENTS = (
    "actor-batch-160",
    "actor-lanes-3",
    "learner-batch-768",
    "learner-batch-1024",
)
CLEAN_TREATMENTS = (
    "lr-quarter",
    "fresh-source",
    "hard-replay",
    "fresh-hard",
)
GUARD_RINGS = (4, 6, 8)

RawConfig = dict[str, Any]
Treatment = Callable[[RawConfig], None]


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-config", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--run-root-parent", type=Path, required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--source-run-root", type=Path, required=True)
    parser.add_argument(
        "--winner-snapshot",
        type=Path,
        help="verified upstream comparator snapshot required for staged preparation",
    )
    parser.add_argument(
        "--futility-policy",
        type=Path,
        help="pre-registered stop-only anytime-valid futility policy",
    )
    parser.add_argument("--prefix", default="ring10-elo")
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--wall-budget-hours", type=float, default=8.0)
    parser.add_argument("--leaf-budget", type=int, default=2_000_000_000)
    parser.add_argument("--guard-floor-elo", type=float, default=-35.0)
    parser.add_argument(
        "--treatment",
        action="append",
        choices=(*DEFAULT_TREATMENTS, *SYSTEM_TREATMENTS, *CLEAN_TREATMENTS),
        dest="treatments",
    )
    return parser


def _mapping(config: RawConfig, name: str) -> RawConfig:
    value = config.get(name)
    if not isinstance(value, dict):
        raise ValueError(f"{name} must be a mapping")
    return value


def _control(_config: RawConfig) -> None:
    return


def _utd_one(config: RawConfig) -> None:
    _mapping(config, "learner")["target_updates_per_new_sample"] = 1.0


def _plateau_keep(config: RawConfig) -> None:
    plateau = _mapping(_mapping(config, "orchestration"), "plateau")
    plateau["action"] = "reduce_lr_keep_weights"
    plateau["clear_optimizer_state_on_recovery"] = True


def _freshness_mix(config: RawConfig) -> None:
    learner = _mapping(config, "learner")
    learner.update(
        {
            "selfplay_snapshot_interval_examples": 3_000_000,
            "selfplay_snapshot_warmup_examples": 20_000_000,
            "selfplay_snapshot_warmup_interval_examples": 1_000_000,
        }
    )
    refresh = _mapping(_mapping(config, "orchestration"), "model_refresh")
    refresh.update(
        {
            "selfplay_source": "candidate_champion_history_mix",
            "candidate_probability": 0.35,
            "history_probability": 0.15,
            "history_pool_size": 8,
        }
    )


def _ring_ten_seventy(config: RawConfig) -> None:
    mixture = _mapping(_mapping(config, "orchestration"), "ring_mixture")
    mixture["step_weights"] = [{"from_step": 0, "weights": [0.10, 0.10, 0.10, 0.70]}]


def _search_quality(config: RawConfig) -> None:
    selfplay = _mapping(config, "selfplay")
    selfplay.update(
        {
            "fast_probability": 0.65,
            "full_probability": 0.35,
            "fast_simulations": 32,
            "full_simulations": 384,
            "max_considered": 32,
            "max_considered_ring_exponent": 1.0,
            "max_considered_cap": 64,
        }
    )


def _actor_batch_160(config: RawConfig) -> None:
    selfplay = _mapping(config, "selfplay")
    selfplay["batch_size"] = 160
    selfplay["games"] = 160
    orchestration = _mapping(config, "orchestration")
    orchestration["actor_games_per_batch"] = 160
    workers = orchestration.get("gpus")
    if not isinstance(workers, list):
        raise ValueError("orchestration.gpus must be a list")
    for worker in workers:
        if isinstance(worker, dict) and worker.get("role") == "actor":
            worker["actor_batch_size"] = 160


def _actor_lanes_three(config: RawConfig) -> None:
    orchestration = _mapping(config, "orchestration")
    promotion_gpu = _mapping(orchestration, "promotion").get("gpu_id")
    workers = orchestration.get("gpus")
    if not isinstance(workers, list):
        raise ValueError("orchestration.gpus must be a list")
    for worker in workers:
        if (
            isinstance(worker, dict)
            and worker.get("role") == "actor"
            and worker.get("gpu_id") != promotion_gpu
        ):
            worker["actor_lanes"] = 3


def _learner_batch(config: RawConfig, size: int) -> None:
    _mapping(config, "train")["per_rank_batch_size"] = size
    _mapping(config, "learner")["target_updates_per_new_sample"] = 1.0


def _lr_quarter(config: RawConfig) -> None:
    optimizer = _mapping(config, "optimizer")
    optimizer["adamw_lr"] = float(optimizer["adamw_lr"]) * 0.5
    optimizer["muon_lr"] = float(optimizer["muon_lr"]) * 0.5


def _fresh_source(config: RawConfig) -> None:
    learner = _mapping(config, "learner")
    learner.update(
        {
            "selfplay_snapshot_interval_examples": 3_000_000,
            "selfplay_snapshot_warmup_examples": 20_000_000,
            "selfplay_snapshot_warmup_interval_examples": 1_000_000,
        }
    )
    refresh = _mapping(_mapping(config, "orchestration"), "model_refresh")
    refresh.update(
        {
            "selfplay_source": "candidate_champion_history_mix",
            "candidate_probability": 0.5,
            "history_probability": 0.15,
            "history_pool_size": 8,
        }
    )


def _hard_replay(config: RawConfig) -> None:
    data = _mapping(config, "data")
    data["shard_cache_size"] = max(16, int(data.get("shard_cache_size", 1)))
    data["shards_per_batch"] = 4
    selfplay = _mapping(config, "selfplay")
    selfplay["policy_surprise_weight"] = 0.5
    selfplay["policy_surprise_max_weight"] = 4.0


def _fresh_hard(config: RawConfig) -> None:
    _fresh_source(config)
    _hard_replay(config)


TREATMENTS: dict[str, Treatment] = {
    "control": _control,
    "utd-1": _utd_one,
    "plateau-keep": _plateau_keep,
    "freshness-mix": _freshness_mix,
    "ring10-70": _ring_ten_seventy,
    "search-quality": _search_quality,
    "actor-batch-160": _actor_batch_160,
    "actor-lanes-3": _actor_lanes_three,
    "learner-batch-768": lambda config: _learner_batch(config, 768),
    "learner-batch-1024": lambda config: _learner_batch(config, 1024),
    "lr-quarter": _lr_quarter,
    "fresh-source": _fresh_source,
    "hard-replay": _hard_replay,
    "fresh-hard": _fresh_hard,
}


def _load_raw(path: Path) -> RawConfig:
    loaded = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(loaded, dict):
        raise ValueError("base config must contain a mapping")
    return loaded


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _read_json(path: Path) -> dict[str, object]:
    loaded = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(loaded, dict):
        raise ValueError(f"{path} must contain a JSON object")
    return loaded


def winner_snapshot_from_document(document: Mapping[str, object]) -> dict[str, object]:
    """Extract a direct snapshot or a comparator selector snapshot."""
    if document.get("status") == "verified" and "champion" in document:
        return dict(document)
    selector = document.get("selector")
    if not isinstance(selector, Mapping) or selector.get("status") != "verified":
        raise ValueError("upstream selector has not emitted a verified winner")
    snapshot = selector.get("winner_snapshot")
    if not isinstance(snapshot, Mapping):
        raise ValueError("verified upstream selector has no winner snapshot")
    return dict(snapshot)


def verify_winner_snapshot(
    source_run_root: Path,
    snapshot: Mapping[str, object],
) -> dict[str, object]:
    """Verify a staged source is still the selector's immutable winner."""
    source = source_run_root.expanduser().resolve()
    if snapshot.get("status") != "verified":
        raise ValueError("winner snapshot status is not verified")
    raw_root = snapshot.get("run_root")
    if not isinstance(raw_root, str) or Path(raw_root).expanduser().resolve() != source:
        raise ValueError("winner snapshot run root does not match staged source")
    champion = snapshot.get("champion")
    if not isinstance(champion, Mapping):
        raise ValueError("winner snapshot champion is missing")
    identity = champion.get("model_identity")
    step = champion.get("model_step")
    if (
        not isinstance(identity, str)
        or not identity
        or type(step) is not int
        or step < 0
    ):
        raise ValueError("winner snapshot champion identity is invalid")
    run_identity = snapshot.get("run_identity")
    if not isinstance(run_identity, Mapping):
        raise ValueError("winner snapshot run identity is missing")
    run_path = source / "run.json"
    champion_path = source / "learner" / "champion.json"
    run_artifact = snapshot.get("run_identity_artifact")
    champion_artifact = snapshot.get("champion_pointer_artifact")
    for name, artifact, expected_path in (
        ("run identity", run_artifact, run_path),
        ("champion pointer", champion_artifact, champion_path),
    ):
        if not isinstance(artifact, Mapping):
            raise ValueError(f"winner snapshot {name} artifact is missing")
        artifact_path = artifact.get("path")
        digest = artifact.get("sha256")
        if (
            not isinstance(artifact_path, str)
            or Path(artifact_path).expanduser().resolve() != expected_path
            or not isinstance(digest, str)
            or digest != _sha256(expected_path)
        ):
            raise ValueError(f"winner snapshot {name} artifact is stale")
    current_run = _read_json(run_path)
    current_champion = _read_json(champion_path)
    if any(
        current_run.get(field) != run_identity.get(field)
        for field in ("run_id", "generation_family", "created_ns")
    ):
        raise ValueError("winner snapshot run identity is stale")
    if (
        current_champion.get("model_identity") != identity
        or current_champion.get("model_step") != step
    ):
        raise ValueError("winner snapshot champion identity is stale")
    return dict(snapshot)


def validate_futility_policy(
    policy: Mapping[str, object],
    *,
    guard_rings: Sequence[int],
    guard_floor_elo: float,
) -> dict[str, object]:
    """Validate a stop-only policy without granting any promotion authority."""
    if (
        policy.get("schema_version") != 1
        or policy.get("report") != "startrain-elo-futility-policy"
    ):
        raise ValueError("unsupported Elo futility policy")
    if type(policy.get("enabled")) is not bool:
        raise ValueError("futility policy enabled must be boolean")
    if policy.get("decision_scope") != "stop_only":
        raise ValueError("futility policy must be stop-only")
    if policy.get("evidence") != "anytime_valid_confidence_sequences":
        raise ValueError("futility policy must require anytime-valid evidence")
    minimum_games = policy.get("minimum_decisive_games")
    if type(minimum_games) is not int or minimum_games <= 0:
        raise ValueError("futility minimum decisive games must be positive")
    guard = policy.get("guard_regression")
    if not isinstance(guard, Mapping):
        raise ValueError("futility guard-regression policy is missing")
    if guard.get("rings") != list(guard_rings):
        raise ValueError("futility guard rings differ from the ablation guardrails")
    if guard.get("floor_elo") != guard_floor_elo:
        raise ValueError("futility guard floor differs from the ablation guardrail")
    if guard.get("arm_upper_field") != "anytime_upper_elo":
        raise ValueError("futility guard regression must use an anytime upper bound")
    control = policy.get("control_comparison")
    if not isinstance(control, Mapping):
        raise ValueError("futility control-comparison policy is missing")
    advantage = control.get("minimum_advantage_elo")
    if isinstance(advantage, bool) or not isinstance(advantage, int | float):
        raise ValueError("futility minimum control advantage must be numeric")
    if (
        control.get("ring") != 10
        or control.get("arm_upper_field") != "anytime_upper_elo"
        or control.get("control_lower_field") != "anytime_lower_elo"
    ):
        raise ValueError("futility control comparison must use ring-10 anytime bounds")
    return json.loads(json.dumps(dict(policy), allow_nan=False))


def _validate_inputs(
    *,
    prefix: str,
    seed: int,
    wall_budget_hours: float,
    leaf_budget: int,
    guard_floor_elo: float,
    treatments: Sequence[str],
) -> None:
    if not prefix or prefix.strip() != prefix or "/" in prefix:
        raise ValueError("prefix must be a non-empty path-safe name")
    if isinstance(seed, bool) or seed < 0:
        raise ValueError("seed must be non-negative")
    if wall_budget_hours <= 0:
        raise ValueError("wall budget must be positive")
    if isinstance(leaf_budget, bool) or leaf_budget <= 0:
        raise ValueError("leaf budget must be positive")
    if guard_floor_elo >= 0:
        raise ValueError("guard floor must be a negative non-inferiority margin")
    if not treatments or len(set(treatments)) != len(treatments):
        raise ValueError("treatments must be non-empty and unique")
    unknown = sorted(set(treatments) - TREATMENTS.keys())
    if unknown:
        raise ValueError(f"unknown treatments: {unknown}")


def _configure_common(
    raw: RawConfig,
    *,
    run_root: Path,
    run_id: str,
    seed: int,
    guard_floor_elo: float,
) -> None:
    _mapping(raw, "train")["seed"] = seed
    _mapping(raw, "selfplay")["seed"] = seed
    orchestration = _mapping(raw, "orchestration")
    orchestration["run_id"] = run_id
    _mapping(orchestration, "directories")["root"] = str(run_root)
    arena = _mapping(raw, "arena")
    floors = arena.get("per_ring_regression_floor_elo")
    if floors is None:
        floors = {}
        arena["per_ring_regression_floor_elo"] = floors
    if not isinstance(floors, dict):
        raise ValueError("arena.per_ring_regression_floor_elo must be a mapping")
    for ring in GUARD_RINGS:
        floors[ring] = guard_floor_elo


def _validate_profile(
    path: Path,
    *,
    expected_root: Path,
    expected_run_id: str,
    expected_seed: int,
    guard_floor_elo: float,
) -> ExperimentConfig:
    loaded = load_config(path)
    if loaded.profile != "continuous" or not loaded.orchestration.enabled:
        raise ValueError("ablation profiles must be continuous orchestrated profiles")
    if loaded.orchestration.directories.root != str(expected_root):
        raise ValueError("generated run root did not round-trip")
    if loaded.orchestration.run_id != expected_run_id:
        raise ValueError("generated run ID did not round-trip")
    if loaded.train.seed != expected_seed or loaded.selfplay.seed != expected_seed:
        raise ValueError("generated treatment seeds did not round-trip")
    for ring in GUARD_RINGS:
        if loaded.arena.per_ring_regression_floor_elo.get(ring) != guard_floor_elo:
            raise ValueError(f"ring {ring} guard floor did not round-trip")
    return loaded


def prepare_elo_ablation(
    *,
    base_config: Path,
    output_dir: Path,
    run_root_parent: Path,
    run_id: str,
    source_run_root: Path,
    prefix: str,
    seed: int,
    wall_budget_hours: float,
    leaf_budget: int,
    guard_floor_elo: float,
    treatments: Sequence[str],
    winner_snapshot: Mapping[str, object] | None = None,
    futility_policy: Mapping[str, object] | None = None,
) -> dict[str, object]:
    _validate_inputs(
        prefix=prefix,
        seed=seed,
        wall_budget_hours=wall_budget_hours,
        leaf_budget=leaf_budget,
        guard_floor_elo=guard_floor_elo,
        treatments=treatments,
    )
    base = base_config.expanduser().resolve()
    source = source_run_root.expanduser().resolve()
    destination = output_dir.expanduser().resolve()
    root_parent = run_root_parent.expanduser().resolve()
    if destination.exists():
        raise FileExistsError(f"output directory already exists: {destination}")
    if not base.is_file():
        raise FileNotFoundError(f"base config does not exist: {base}")
    if not source.is_dir():
        raise FileNotFoundError(f"source run root does not exist: {source}")
    verified_winner = (
        verify_winner_snapshot(source, winner_snapshot)
        if winner_snapshot is not None
        else None
    )
    registered_futility = (
        validate_futility_policy(
            futility_policy,
            guard_rings=GUARD_RINGS,
            guard_floor_elo=guard_floor_elo,
        )
        if futility_policy is not None
        else None
    )
    destination.mkdir(parents=True)
    base_sha256 = _sha256(base)
    raw_base = _load_raw(base)
    generated = []
    for treatment_name in treatments:
        run_root = root_parent / f"{prefix}-{treatment_name}-seed{seed}"
        profile = deepcopy(raw_base)
        _configure_common(
            profile,
            run_root=run_root,
            run_id=run_id,
            seed=seed,
            guard_floor_elo=guard_floor_elo,
        )
        TREATMENTS[treatment_name](profile)
        profile_path = destination / f"{treatment_name}.yaml"
        profile_path.write_text(
            yaml.safe_dump(profile, sort_keys=False),
            encoding="utf-8",
        )
        _validate_profile(
            profile_path,
            expected_root=run_root,
            expected_run_id=run_id,
            expected_seed=seed,
            guard_floor_elo=guard_floor_elo,
        )
        generated.append(
            {
                "treatment": treatment_name,
                "profile": str(profile_path),
                "profile_sha256": _sha256(profile_path),
                "run_root": str(run_root),
            }
        )
    manifest = {
        "schema_version": SCHEMA_VERSION,
        "report": REPORT_NAME,
        "base_config": str(base),
        "base_config_sha256": base_sha256,
        "source_run_root": str(source),
        "source_winner_snapshot": verified_winner,
        "futility_policy": registered_futility,
        "run_id": run_id,
        "prefix": prefix,
        "seed": seed,
        "wall_budget_seconds": wall_budget_hours * 3600.0,
        "leaf_budget": leaf_budget,
        "guard_rings": list(GUARD_RINGS),
        "guard_floor_elo": guard_floor_elo,
        "treatments": generated,
    }
    manifest_path = destination / "ablation-plan.json"
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return manifest


def main(argv: list[str] | None = None) -> int:
    arguments = _parser().parse_args(argv)
    try:
        winner_snapshot = (
            winner_snapshot_from_document(_read_json(arguments.winner_snapshot))
            if arguments.winner_snapshot is not None
            else None
        )
        futility_policy = (
            _read_json(arguments.futility_policy)
            if arguments.futility_policy is not None
            else None
        )
        manifest = prepare_elo_ablation(
            base_config=arguments.base_config,
            output_dir=arguments.output_dir,
            run_root_parent=arguments.run_root_parent,
            run_id=arguments.run_id,
            source_run_root=arguments.source_run_root,
            prefix=arguments.prefix,
            seed=arguments.seed,
            wall_budget_hours=arguments.wall_budget_hours,
            leaf_budget=arguments.leaf_budget,
            guard_floor_elo=arguments.guard_floor_elo,
            treatments=arguments.treatments or DEFAULT_TREATMENTS,
            winner_snapshot=winner_snapshot,
            futility_policy=futility_policy,
        )
    except (FileExistsError, FileNotFoundError, OSError, ValueError) as error:
        print(
            json.dumps(
                {
                    "schema_version": SCHEMA_VERSION,
                    "report": REPORT_NAME,
                    "status": "error",
                    "error": f"{type(error).__name__}: {error}",
                },
                sort_keys=True,
            )
        )
        return 2
    print(json.dumps({"status": "ok", **manifest}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
