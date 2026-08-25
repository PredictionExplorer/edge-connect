from __future__ import annotations

import json
from pathlib import Path

from scripts import compare_frozen_replay_optimizer_calibration as comparator
from scripts import run_frozen_replay_optimizer_calibration as runner
from scripts.prepare_elo_ablation import (
    RING10_OPTIMIZER_CALIBRATION_LABELS,
    RING10_OPTIMIZER_CALIBRATION_TREATMENTS,
)


def _config_contract(arm: str) -> dict[str, object]:
    optimizer = {
        "kind": "muon_adamw",
        "adamw_lr": 0.001,
        "muon_lr": 0.01,
        "weight_decay": 0.01,
    }
    train = {
        "per_rank_batch_size": 8,
        "precision": "bf16",
        "compile": True,
        "ema_decay": 0.999,
        "ema_half_life_examples": None,
        "gradient_clip_norm": 1.0,
        "scheduler": {
            "warmup_steps": 0,
            "total_steps": 100,
            "min_lr_ratio": 0.1,
        },
    }
    if arm == "ring10-optimizer-clip-norm-2":
        train["gradient_clip_norm"] = 2.0
    elif arm == "ring10-optimizer-clip-norm-5":
        train["gradient_clip_norm"] = 5.0
    elif arm == runner.FOLLOW_ON_ARM:
        optimizer["adamw_lr"] = 0.0005
        optimizer["muon_lr"] = 0.005
    return {
        "model": {"width": 8},
        "game": {"rings": [4, 6, 8, 10]},
        "loss": {
            "policy": 1.0,
            "outcome": 1.0,
            "score_margin": 0.25,
            "ownership": 0.25,
            "alive": 0.1,
            "soft_policy": 0.25,
        },
        "optimizer": optimizer,
        "train": train,
    }


def _payload(
    arm: str,
    *,
    throughput: float,
    candidate_composites: tuple[float, ...],
    clipping: float,
    finite: bool = True,
) -> dict[str, object]:
    reference = {"policy": 0.6, "value": 0.6, "composite": 1.2}
    observations = [
        {
            "index": index,
            "samples": 10,
            "reference": reference,
            "candidate": {
                "policy": composite / 2,
                "value": composite / 2,
                "composite": composite,
            },
            "composite_improvement": 1.2 - composite,
        }
        for index, composite in enumerate(candidate_composites)
    ]
    payload: dict[str, object] = {
        "format": runner.FORMAT,
        "schema_version": runner.SCHEMA_VERSION,
        "status": "complete",
        "arm": arm,
        "label": RING10_OPTIMIZER_CALIBRATION_LABELS[arm],
        "phase": "follow_on" if arm == runner.FOLLOW_ON_ARM else "primary",
        "run_contract_sha256": "1" * 64,
        "semantic_sha256": "1" * 64,
        "production_promotion_authorized": False,
        "device": {
            "requested": "cuda:0",
            "resolved": "cuda:0",
            "precision": "bf16",
            "compile": True,
        },
        "config_contract": _config_contract(arm),
        "champion": {
            "pointer": {
                "path": "/source/champion.json",
                "sha256": "a" * 64,
                "bytes": 1,
            },
            "manifest": {
                "path": "/source/manifest.json",
                "sha256": "b" * 64,
                "bytes": 1,
            },
            "checkpoint": {
                "path": "/source/checkpoint.pt",
                "sha256": "c" * 64,
                "bytes": 1,
            },
            "model_identity": f"sha256-{'c' * 64}",
            "model_step": 10,
            "run_id": "source-run",
            "generation_family": "source-family",
            "weights": "ema",
        },
        "replay": {
            "root": "/source/replay",
            "manifest_open_mode": "ro",
            "query_only": True,
            "cutoff": 42,
            "cutoff_sha256": "d" * 64,
            "eligible_shards": 1,
            "selected_shards": [
                {
                    "id": 42,
                    "path": "/source/replay/shards/ring10.npz",
                    "sha256": "8" * 64,
                    "bytes": 1,
                }
            ],
        },
        "partition": {
            "method": "bounded-latest-window-hash-order-exact-split-v1",
            "train_samples": 100,
            "holdout_samples": 40,
            "train_sha256": "e" * 64,
            "holdout_sha256": "f" * 64,
            "partition_sha256": "0" * 64,
            "disjoint": True,
        },
        "evaluation": {
            "batch_size": 10,
            "augmentation": False,
            "composite": "policy/value",
            "observation_unit": "deterministic-held-out-batch",
        },
        "optimizer": {
            "fresh_from_champion_ema": True,
            "source_optimizer_loaded": False,
            "source_scheduler_loaded": False,
            "routing": {
                "schema_version": 1,
                "requested_kind": "muon_adamw",
                "implementation": "muon_adamw",
                "fallback_used": False,
                "routing_hash": "sha256-routing",
                "parameter_tensors": 2,
                "parameter_elements": 64,
                "optimizer_config": _config_contract(arm)["optimizer"],
                "groups": [
                    {
                        "name": "muon",
                        "algorithm": "muon",
                        "weight_decay": 0.01,
                        "parameter_tensors": 1,
                        "parameter_elements": 64,
                        "parameter_names": ["trunk.weight"],
                    }
                ],
            },
            "first_step_diagnostics": [],
        },
        "training": {
            "steps": 100,
            "completed_steps": 100,
            "batch_size": 8,
            "examples_consumed": 800,
            "seed": 17,
            "budget_h100_hours": 1.0,
            "h100_count": 1,
            "elapsed_seconds": 10.0,
            "finite": finite,
            "nonfinite_loss_count": 0 if finite else 1,
            "nonfinite_gradient_count": 0,
            "examples_per_second": throughput,
            "measured_training_seconds": 10.0,
            "mean_gradient_norm": 2.0,
            "gradient_clipping_frequency": clipping,
            "mean_losses": {"total": 1.0},
        },
        "heldout": {
            "finite": finite,
            "samples": 10 * len(observations),
            "batches": len(observations),
            "observation_unit": "deterministic-held-out-batch",
            "reference": reference,
            "candidate": {
                "policy": sum(candidate_composites) / len(candidate_composites) / 2,
                "value": sum(candidate_composites) / len(candidate_composites) / 2,
                "composite": sum(candidate_composites) / len(candidate_composites),
            },
            "observations": observations,
        },
        "candidate_checkpoint": {
            "path": f"/output/{arm}.pt",
            "sha256": "9" * 64,
            "bytes": 1,
        },
    }
    payload["result_sha256"] = runner._digest(payload)
    return payload


def _write_suite(
    tmp_path: Path,
    *,
    tied: bool = False,
    reference_drift: bool = False,
) -> list[Path]:
    tmp_path.mkdir(parents=True, exist_ok=True)
    candidates = {
        runner.CONTROL_ARM: (100.0, (1.0, 1.0, 1.0, 1.0), 0.8),
        "ring10-optimizer-clip-norm-2": (
            95.0,
            (0.8, 0.8, 0.8, 0.8),
            0.4,
        ),
        "ring10-optimizer-clip-norm-5": (
            92.0,
            (0.8, 0.8, 0.8, 0.8) if tied else (0.7, 0.7, 0.7, 0.7),
            0.9,
        ),
        runner.FOLLOW_ON_ARM: (89.0, (0.75, 0.75, 0.75, 0.75), 0.5),
    }
    paths = []
    for arm in RING10_OPTIMIZER_CALIBRATION_TREATMENTS:
        throughput, composites, clipping = candidates[arm]
        payload = _payload(
            arm,
            throughput=throughput,
            candidate_composites=composites,
            clipping=clipping,
        )
        if reference_drift and arm == "ring10-optimizer-clip-norm-5":
            payload["heldout"]["observations"][0]["reference"]["composite"] = 1.3
            unsigned = dict(payload)
            unsigned.pop("result_sha256")
            payload["result_sha256"] = runner._digest(unsigned)
        path = tmp_path / f"{arm}.json"
        path.write_text(json.dumps(payload, sort_keys=True), encoding="utf-8")
        paths.append(path)
    return paths


def test_comparator_gates_and_ignores_clip_reduction_for_selection(
    tmp_path: Path,
) -> None:
    result = comparator.compare_results(
        _write_suite(tmp_path),
        bootstrap_samples=100,
    )

    assert result["selection"]["selected_arm"] == "ring10-optimizer-clip-norm-5"
    assert result["selection"]["fallback_to_control"] is False
    selected = result["arms"]["ring10-optimizer-clip-norm-5"]
    assert selected["passes_all_selection_gates"] is True
    assert selected["clip_reduction"]["diagnostic_only"] is True
    assert selected["clip_reduction"]["absolute_reduction"] < 0
    follow_on = result["arms"][runner.FOLLOW_ON_ARM]
    assert follow_on["gates"]["throughput_at_least_90_percent_of_control"] is False


def test_comparator_falls_back_to_control_on_tie_or_reference_drift(
    tmp_path: Path,
) -> None:
    tied = comparator.compare_results(
        _write_suite(tmp_path / "tie", tied=True),
        bootstrap_samples=100,
    )
    assert tied["selection"]["selected_arm"] == runner.CONTROL_ARM
    assert tied["selection"]["fallback_to_control"] is True
    assert "tied" in tied["selection"]["reason"]

    drifted = comparator.compare_results(
        _write_suite(tmp_path / "drift", reference_drift=True),
        bootstrap_samples=100,
    )
    clip_five = drifted["arms"]["ring10-optimizer-clip-norm-5"]
    assert clip_five["gates"]["reference_evaluation_parity"] is False
    assert clip_five["passes_all_selection_gates"] is False


def test_comparator_rejects_execution_setting_drift(
    tmp_path: Path,
) -> None:
    paths = _write_suite(tmp_path)
    treatment_path = next(path for path in paths if "clip-norm-5" in path.name)
    payload = json.loads(treatment_path.read_text(encoding="utf-8"))
    payload["training"]["completed_steps"] = 99
    payload["training"]["examples_consumed"] = 792
    payload["training"]["batch_size"] = 4
    payload["device"]["requested"] = "cuda:1"
    payload["device"]["resolved"] = "cuda:1"
    unsigned = dict(payload)
    unsigned.pop("result_sha256")
    payload["result_sha256"] = runner._digest(unsigned)
    treatment_path.write_text(json.dumps(payload), encoding="utf-8")

    result = comparator.compare_results(paths, bootstrap_samples=100)

    treatment = result["arms"]["ring10-optimizer-clip-norm-5"]
    assert treatment["gates"]["common_source_and_partition_parity"] is False
    assert treatment["passes_all_selection_gates"] is False
    assert result["selection"]["selected_arm"] != "ring10-optimizer-clip-norm-5"


def test_comparator_rejects_actual_progress_drift_in_isolation(
    tmp_path: Path,
) -> None:
    paths = _write_suite(tmp_path)
    treatment_path = next(path for path in paths if "clip-norm-5" in path.name)
    payload = json.loads(treatment_path.read_text(encoding="utf-8"))
    payload["training"]["completed_steps"] = 99
    payload["training"]["examples_consumed"] = 792
    unsigned = dict(payload)
    unsigned.pop("result_sha256")
    payload["result_sha256"] = runner._digest(unsigned)
    treatment_path.write_text(json.dumps(payload), encoding="utf-8")

    result = comparator.compare_results(paths, bootstrap_samples=100)

    treatment = result["arms"]["ring10-optimizer-clip-norm-5"]
    assert treatment["gates"]["common_source_and_partition_parity"] is False
    assert treatment["passes_all_selection_gates"] is False
