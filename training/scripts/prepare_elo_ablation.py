#!/usr/bin/env python3
"""Generate frozen, one-factor H100 Elo-ablation profiles and manifests."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections.abc import Callable, Sequence
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
    parser.add_argument("--prefix", default="ring10-elo")
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--wall-budget-hours", type=float, default=8.0)
    parser.add_argument("--leaf-budget", type=int, default=2_000_000_000)
    parser.add_argument("--guard-floor-elo", type=float, default=-35.0)
    parser.add_argument(
        "--treatment",
        action="append",
        choices=(*DEFAULT_TREATMENTS, *SYSTEM_TREATMENTS),
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
