from __future__ import annotations

import json
from pathlib import Path

import pytest

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
    cache_root: Path,
    throughput: float,
    candidate_composites: tuple[float, ...],
    clipping: float,
    finite: bool = True,
) -> dict[str, object]:
    cache_root_text = str(cache_root)
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
            "preimport_compile_cache_bootstrap": (
                runner._compile_cache_bootstrap_token(cache_root.parent.parent)
            ),
            "compile_cache": {
                "schema_version": 1,
                "layout": "startrain-isolated-compile-cache-v1",
                "root": cache_root_text,
                "owner_marker": f"{cache_root_text}/cache-owner.json",
                "required_unset_environment": [
                    "TRITON_CACHE_MANAGER",
                    "TRITON_REMOTE_CACHE_BACKEND",
                    "TORCHINDUCTOR_FX_GRAPH_REMOTE_CACHE",
                    "TORCHINDUCTOR_AUTOTUNE_REMOTE_CACHE",
                    "TORCHINDUCTOR_AUTOGRAD_REMOTE_CACHE",
                    "TORCHINDUCTOR_FORCE_DISABLE_CACHES",
                    "TORCHINDUCTOR_BUNDLED_AUTOTUNE_REMOTE_CACHE",
                    "TORCHINDUCTOR_FX_GRAPH_CACHE",
                    "TORCHINDUCTOR_AUTOGRAD_CACHE",
                    "TORCH_DYNAMO_AUTOMATIC_DYNAMIC_LOCAL_PGO",
                    "TORCH_DYNAMO_AUTOMATIC_DYNAMIC_REMOTE_PGO",
                    "TORCH_COMPILE_JOB_ID",
                    "TORCH_COMPILE_FORCE_DISABLE_CACHES",
                    "CUDA_CACHE_DISABLE",
                ],
                "rejected_environment_prefixes": [
                    "TORCHINDUCTOR_",
                    "TORCH_COMPILE_",
                    "TORCH_DYNAMO_",
                    "TRITON_",
                    "CUDA_CACHE_",
                ],
                "environment": {
                    "HOME": f"{cache_root_text}/home",
                    "TORCHINDUCTOR_CACHE_DIR": f"{cache_root_text}/inductor",
                    "TORCHINDUCTOR_PERSISTENT_AUTOTUNE_DIR": (
                        f"{cache_root_text}/inductor-autotune"
                    ),
                    "TRITON_HOME": f"{cache_root_text}/triton-home",
                    "TRITON_CACHE_DIR": f"{cache_root_text}/triton",
                    "TRITON_DUMP_DIR": f"{cache_root_text}/triton-dump",
                    "TRITON_OVERRIDE_DIR": f"{cache_root_text}/triton-override",
                    "XDG_CACHE_HOME": f"{cache_root_text}/xdg",
                    "CUDA_CACHE_PATH": f"{cache_root_text}/cuda",
                },
                "runtime": {
                    "python_version": "3.11.13",
                    "torch_version": "2.13.0+cu130",
                    "cuda_runtime_version": "13.0",
                    "triton_version": "3.6.0",
                },
            },
            "hardware": {
                "logical_index": 0,
                "physical_index": 0,
                "uuid": "GPU-aaaaaaaa-bbbb-cccc-dddd-eeeeeeeeeeee",
                "name": "NVIDIA H100 80GB HBM3",
                "nvidia_smi_name": "NVIDIA H100 80GB HBM3",
                "compute_capability": [9, 0],
                "total_memory_bytes": 80 * 1024**3,
                "driver_version": "580.105.08",
            },
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
        cache_root = (tmp_path / "outputs" / arm / "compile-cache" / "v1").resolve()
        for suffix in (
            "home",
            "inductor",
            "inductor-autotune",
            "triton-home",
            "triton",
            "triton-dump",
            "triton-override",
            "xdg",
            "cuda",
        ):
            (cache_root / suffix).mkdir(parents=True, exist_ok=True)
        payload = _payload(
            arm,
            cache_root=cache_root,
            throughput=throughput,
            candidate_composites=composites,
            clipping=clipping,
        )
        cache = payload["device"]["compile_cache"]
        (cache_root / "cache-owner.json").write_text(
            json.dumps(
                {
                    "format": "startrain.compile-cache-owner",
                    "schema_version": 1,
                    "layout": cache["layout"],
                    "arm": arm,
                    "run_contract_sha256": payload["run_contract_sha256"],
                    "root": cache["root"],
                    "environment": dict(sorted(cache["environment"].items())),
                    "runtime": dict(sorted(cache["runtime"].items())),
                },
                sort_keys=True,
            ),
            encoding="utf-8",
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


def test_comparator_rejects_shared_or_missing_compile_cache(tmp_path: Path) -> None:
    shared_paths = _write_suite(tmp_path / "shared")
    first_payload = json.loads(shared_paths[0].read_text(encoding="utf-8"))
    shared_root = first_payload["device"]["compile_cache"]["root"]
    for path in shared_paths:
        payload = json.loads(path.read_text(encoding="utf-8"))
        cache = payload["device"]["compile_cache"]
        cache["root"] = shared_root
        cache["owner_marker"] = f"{shared_root}/cache-owner.json"
        for name, suffix in {
            "HOME": "home",
            "TORCHINDUCTOR_CACHE_DIR": "inductor",
            "TORCHINDUCTOR_PERSISTENT_AUTOTUNE_DIR": "inductor-autotune",
            "TRITON_HOME": "triton-home",
            "TRITON_CACHE_DIR": "triton",
            "TRITON_DUMP_DIR": "triton-dump",
            "TRITON_OVERRIDE_DIR": "triton-override",
            "XDG_CACHE_HOME": "xdg",
            "CUDA_CACHE_PATH": "cuda",
        }.items():
            cache["environment"][name] = f"{shared_root}/{suffix}"
        unsigned = dict(payload)
        unsigned.pop("result_sha256")
        payload["result_sha256"] = runner._digest(unsigned)
        path.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(ValueError, match="cache"):
        comparator.compare_results(shared_paths, bootstrap_samples=100)

    missing_paths = _write_suite(tmp_path / "missing")
    payload = json.loads(missing_paths[0].read_text(encoding="utf-8"))
    payload["device"]["compile_cache"] = None
    unsigned = dict(payload)
    unsigned.pop("result_sha256")
    payload["result_sha256"] = runner._digest(unsigned)
    missing_paths[0].write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(ValueError, match="cache"):
        comparator.compare_results(missing_paths, bootstrap_samples=100)


def test_comparator_rejects_nested_compile_cache_roots(tmp_path: Path) -> None:
    paths = _write_suite(tmp_path)
    control = json.loads(paths[0].read_text(encoding="utf-8"))
    treatment_path = paths[1]
    treatment = json.loads(treatment_path.read_text(encoding="utf-8"))
    nested_root = (
        Path(control["device"]["compile_cache"]["root"]) / "inductor" / "nested"
    )
    cache = treatment["device"]["compile_cache"]
    cache["root"] = str(nested_root)
    cache["owner_marker"] = str(nested_root / "cache-owner.json")
    suffixes = {
        "HOME": "home",
        "TORCHINDUCTOR_CACHE_DIR": "inductor",
        "TORCHINDUCTOR_PERSISTENT_AUTOTUNE_DIR": "inductor-autotune",
        "TRITON_HOME": "triton-home",
        "TRITON_CACHE_DIR": "triton",
        "TRITON_DUMP_DIR": "triton-dump",
        "TRITON_OVERRIDE_DIR": "triton-override",
        "XDG_CACHE_HOME": "xdg",
        "CUDA_CACHE_PATH": "cuda",
    }
    for name, suffix in suffixes.items():
        path = nested_root / suffix
        path.mkdir(parents=True, exist_ok=True)
        cache["environment"][name] = str(path)
    (nested_root / "cache-owner.json").write_text(
        json.dumps(
            {
                "format": "startrain.compile-cache-owner",
                "schema_version": 1,
                "layout": cache["layout"],
                "arm": treatment["arm"],
                "run_contract_sha256": treatment["run_contract_sha256"],
                "root": cache["root"],
                "environment": dict(sorted(cache["environment"].items())),
                "runtime": dict(sorted(cache["runtime"].items())),
            }
        ),
        encoding="utf-8",
    )
    unsigned = dict(treatment)
    unsigned.pop("result_sha256")
    treatment["result_sha256"] = runner._digest(unsigned)
    treatment_path.write_text(json.dumps(treatment), encoding="utf-8")

    with pytest.raises(ValueError, match="disjoint"):
        comparator.compare_results(paths, bootstrap_samples=100)


def test_suite_hardware_drift_forces_control_fallback(tmp_path: Path) -> None:
    paths = _write_suite(tmp_path)
    treatment_path = paths[1]
    payload = json.loads(treatment_path.read_text(encoding="utf-8"))
    payload["device"]["hardware"]["driver_version"] = "different"
    unsigned = dict(payload)
    unsigned.pop("result_sha256")
    payload["result_sha256"] = runner._digest(unsigned)
    treatment_path.write_text(json.dumps(payload), encoding="utf-8")

    result = comparator.compare_results(paths, bootstrap_samples=100)

    assert result["common_contract"]["suite_common_parity"] is False
    assert result["control_valid"] is False
    assert result["selection"]["selected_arm"] == runner.CONTROL_ARM
    assert result["selection"]["fallback_to_control"] is True


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
