#!/usr/bin/env python3
"""Compare frozen-replay optimizer arms with fail-closed held-out gates."""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
import os
import random
from collections.abc import Mapping, Sequence
from pathlib import Path

from startrain.checkpoint import sha256_file
from startrain.runtime import atomic_json

if __package__:
    from .prepare_elo_ablation import (
        RING10_OPTIMIZER_CALIBRATION_LABELS,
        RING10_OPTIMIZER_CALIBRATION_TREATMENTS,
    )
    from .run_frozen_replay_optimizer_calibration import (
        CONTROL_ARM,
        FOLLOW_ON_ARM,
        FORMAT as RESULT_FORMAT,
        MAX_H100_HOURS_PER_ARM,
        SCHEMA_VERSION as RESULT_SCHEMA_VERSION,
    )
else:
    from prepare_elo_ablation import (  # type: ignore[no-redef]
        RING10_OPTIMIZER_CALIBRATION_LABELS,
        RING10_OPTIMIZER_CALIBRATION_TREATMENTS,
    )
    from run_frozen_replay_optimizer_calibration import (  # type: ignore[no-redef]
        CONTROL_ARM,
        FOLLOW_ON_ARM,
        FORMAT as RESULT_FORMAT,
        MAX_H100_HOURS_PER_ARM,
        SCHEMA_VERSION as RESULT_SCHEMA_VERSION,
    )

FORMAT = "startrain.frozen-replay-optimizer-calibration-comparison"
SCHEMA_VERSION = 1
MINIMUM_CONTROL_THROUGHPUT_FRACTION = 0.9
DEFAULT_CONFIDENCE = 0.95
DEFAULT_BOOTSTRAP_SAMPLES = 10_000
COMPILE_CACHE_ENVIRONMENT = {
    "HOME",
    "TORCHINDUCTOR_CACHE_DIR",
    "TORCHINDUCTOR_PERSISTENT_AUTOTUNE_DIR",
    "TRITON_HOME",
    "TRITON_CACHE_DIR",
    "TRITON_DUMP_DIR",
    "TRITON_OVERRIDE_DIR",
    "XDG_CACHE_HOME",
    "CUDA_CACHE_PATH",
}
UNSAFE_COMPILE_ENVIRONMENT = {
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
}
COMPILE_CONTROL_PREFIXES = {
    "TORCHINDUCTOR_",
    "TORCH_COMPILE_",
    "TORCH_DYNAMO_",
    "TRITON_",
    "CUDA_CACHE_",
}


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--result",
        action="append",
        required=True,
        type=Path,
        help="one completed arm result; provide the complete four-arm suite",
    )
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument(
        "--confidence",
        type=float,
        default=DEFAULT_CONFIDENCE,
    )
    parser.add_argument(
        "--bootstrap-samples",
        type=int,
        default=DEFAULT_BOOTSTRAP_SAMPLES,
    )
    return parser


def _canonical(value: object) -> bytes:
    return json.dumps(
        value,
        allow_nan=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")


def _digest(value: object) -> str:
    return hashlib.sha256(_canonical(value)).hexdigest()


def _mapping(value: object, name: str) -> Mapping[str, object]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{name} must be a JSON object")
    return value


def _finite_number(value: object, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, int | float):
        raise ValueError(f"{name} must be numeric")
    number = float(value)
    if not math.isfinite(number):
        raise ValueError(f"{name} must be finite")
    return number


def _positive_number(value: object, name: str) -> float:
    number = _finite_number(value, name)
    if number <= 0:
        raise ValueError(f"{name} must be positive")
    return number


def _read_result(path: Path) -> tuple[dict[str, object], dict[str, object]]:
    source = path.expanduser().resolve()
    if not source.is_file() or source.is_symlink():
        raise ValueError(f"arm result is not a regular non-symlink file: {source}")
    loaded = json.loads(source.read_text(encoding="utf-8"))
    if not isinstance(loaded, dict):
        raise ValueError(f"arm result must contain an object: {source}")
    if (
        loaded.get("format") != RESULT_FORMAT
        or loaded.get("schema_version") != RESULT_SCHEMA_VERSION
        or loaded.get("status") != "complete"
    ):
        raise ValueError(f"arm result is not a completed calibration: {source}")
    expected_result_hash = loaded.get("result_sha256")
    unsigned = dict(loaded)
    unsigned.pop("result_sha256", None)
    if (
        not isinstance(expected_result_hash, str)
        or _digest(unsigned) != expected_result_hash
    ):
        raise ValueError(f"arm result semantic hash failed: {source}")
    pin = {
        "path": str(source),
        "sha256": sha256_file(source),
        "bytes": source.stat().st_size,
        "result_sha256": expected_result_hash,
    }
    return loaded, pin


def _arm(payload: Mapping[str, object]) -> str:
    value = payload.get("arm")
    if value not in RING10_OPTIMIZER_CALIBRATION_TREATMENTS:
        raise ValueError("calibration result has an unknown arm")
    if payload.get("label") != RING10_OPTIMIZER_CALIBRATION_LABELS[value]:
        raise ValueError(f"{value} has the wrong explicit calibration label")
    expected_phase = "follow_on" if value == FOLLOW_ON_ARM else "primary"
    if payload.get("phase") != expected_phase:
        raise ValueError(f"{value} has the wrong calibration phase")
    return value


def _verify_compile_cache_tree(root: Path) -> None:
    for directory, names, filenames in os.walk(root, followlinks=False):
        parent = Path(directory)
        for name in names:
            path = parent / name
            if path.is_symlink():
                raise ValueError(
                    f"compiled calibration cache contains a symlink: {path}"
                )
        for name in filenames:
            path = parent / name
            if path.is_symlink() or not path.is_file():
                raise ValueError(
                    f"compiled calibration cache contains an unsafe file: {path}"
                )


def _compile_cache(payload: Mapping[str, object]) -> dict[str, object] | None:
    device = _mapping(payload.get("device"), "device")
    raw = device.get("compile_cache")
    if device.get("compile") is not True:
        raise ValueError("optimizer calibration must use CUDA compilation")
    cache = _mapping(raw, "compiled calibration cache")
    if (
        cache.get("schema_version") != 1
        or cache.get("layout") != "startrain-isolated-compile-cache-v1"
    ):
        raise ValueError("compiled calibration cache schema is invalid")
    raw_root = cache.get("root")
    if not isinstance(raw_root, str):
        raise ValueError("compiled calibration cache root is missing")
    try:
        root = Path(raw_root).resolve(strict=True)
    except OSError as error:
        raise ValueError("compiled calibration cache root is unavailable") from error
    if not Path(raw_root).is_absolute() or raw_root != str(root):
        raise ValueError("compiled calibration cache root is not canonical")
    _verify_compile_cache_tree(root)
    owner_marker = cache.get("owner_marker")
    if owner_marker != str(root / "cache-owner.json"):
        raise ValueError("compiled calibration cache owner marker is invalid")
    required_unset = cache.get("required_unset_environment")
    if (
        not isinstance(required_unset, list)
        or set(required_unset) != UNSAFE_COMPILE_ENVIRONMENT
    ):
        raise ValueError("compiled calibration unsafe environment policy changed")
    rejected_prefixes = cache.get("rejected_environment_prefixes")
    if (
        not isinstance(rejected_prefixes, list)
        or set(rejected_prefixes) != COMPILE_CONTROL_PREFIXES
    ):
        raise ValueError("compiled calibration environment prefix policy changed")
    environment = _mapping(
        cache.get("environment"),
        "compiled calibration cache environment",
    )
    if set(environment) != COMPILE_CACHE_ENVIRONMENT:
        raise ValueError("compiled calibration cache environment is incomplete")
    for name, raw_path in environment.items():
        if not isinstance(raw_path, str):
            raise ValueError(f"compiled calibration cache {name} is not a path")
        try:
            path = Path(raw_path).resolve(strict=True)
        except OSError as error:
            raise ValueError(
                f"compiled calibration cache {name} is unavailable"
            ) from error
        if (
            not Path(raw_path).is_absolute()
            or raw_path != str(path)
            or (path != root and root not in path.parents)
        ):
            raise ValueError(f"compiled calibration cache {name} escaped its root")
    runtime = _mapping(cache.get("runtime"), "compiled calibration cache runtime")
    for name in ("python_version", "torch_version", "triton_version"):
        if not isinstance(runtime.get(name), str) or not runtime.get(name):
            raise ValueError(f"compiled calibration cache runtime {name} is missing")
    cuda = runtime.get("cuda_runtime_version")
    if cuda is not None and (not isinstance(cuda, str) or not cuda):
        raise ValueError("compiled calibration CUDA runtime version is invalid")
    marker_path = Path(str(owner_marker))
    if marker_path.is_symlink() or not marker_path.is_file():
        raise ValueError("compiled calibration cache owner marker is unavailable")
    try:
        marker = json.loads(marker_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(
            "compiled calibration cache owner marker is unreadable"
        ) from error
    expected_marker = {
        "format": "startrain.compile-cache-owner",
        "schema_version": 1,
        "layout": "startrain-isolated-compile-cache-v1",
        "arm": payload.get("arm"),
        "run_contract_sha256": payload.get("run_contract_sha256"),
        "root": str(root),
        "environment": dict(sorted(environment.items())),
        "runtime": dict(sorted(runtime.items())),
    }
    if marker != expected_marker:
        raise ValueError("compiled calibration cache owner marker changed")
    return {
        "schema_version": cache["schema_version"],
        "layout": cache["layout"],
        "root": str(root),
        "runtime": dict(runtime),
    }


def _cuda_h100_device(
    payload: Mapping[str, object],
    *,
    compile_cache: Mapping[str, object],
) -> dict[str, object]:
    device = _mapping(payload.get("device"), "device")
    expected_bootstrap = hashlib.sha256(
        str(compile_cache["root"]).encode("utf-8")
    ).hexdigest()
    if (
        device.get("compile") is not True
        or not str(device.get("requested", "")).startswith("cuda")
        or not str(device.get("resolved", "")).startswith("cuda")
        or device.get("preimport_compile_cache_bootstrap") != expected_bootstrap
    ):
        raise ValueError("optimizer calibration must use a compiled CUDA device")
    hardware = _mapping(device.get("hardware"), "CUDA hardware identity")
    capability = hardware.get("compute_capability")
    total_memory = hardware.get("total_memory_bytes")
    if (
        "H100" not in str(hardware.get("name", ""))
        or "H100" not in str(hardware.get("nvidia_smi_name", ""))
        or not isinstance(hardware.get("uuid"), str)
        or not str(hardware["uuid"]).startswith("GPU-")
        or capability != [9, 0]
        or type(hardware.get("logical_index")) is not int
        or type(hardware.get("physical_index")) is not int
        or isinstance(total_memory, bool)
        or not isinstance(total_memory, int)
        or total_memory <= 0
        or not isinstance(hardware.get("driver_version"), str)
        or not hardware["driver_version"]
    ):
        raise ValueError("optimizer calibration requires a complete H100 identity")
    return dict(hardware)


def _common_projection(payload: Mapping[str, object]) -> dict[str, object]:
    champion = _mapping(payload.get("champion"), "champion")
    replay = _mapping(payload.get("replay"), "replay")
    partition = _mapping(payload.get("partition"), "partition")
    training = _mapping(payload.get("training"), "training")
    device = _mapping(payload.get("device"), "device")
    cache = _compile_cache(payload)
    assert cache is not None
    hardware = _cuda_h100_device(payload, compile_cache=cache)
    return {
        "champion": champion,
        "replay": {
            key: replay.get(key)
            for key in (
                "root",
                "cutoff",
                "cutoff_sha256",
                "manifest_open_mode",
                "query_only",
                "eligible_shards",
                "selected_shards",
            )
        },
        "partition": {
            key: partition.get(key)
            for key in (
                "method",
                "train_samples",
                "holdout_samples",
                "train_sha256",
                "holdout_sha256",
                "partition_sha256",
                "disjoint",
            )
        },
        "evaluation": payload.get("evaluation"),
        "execution": {
            "training": {
                key: training.get(key)
                for key in (
                    "steps",
                    "completed_steps",
                    "batch_size",
                    "examples_consumed",
                    "seed",
                    "budget_h100_hours",
                    "h100_count",
                )
            },
            "device": {
                key: device.get(key)
                for key in ("requested", "resolved", "precision", "compile")
            },
            "hardware": hardware,
            "compile_cache_runtime": (
                {
                    "schema_version": cache["schema_version"],
                    "layout": cache["layout"],
                    "runtime": cache["runtime"],
                }
                if cache is not None
                else None
            ),
        },
    }


def _expected_config_contract(
    control: Mapping[str, object],
    arm: str,
) -> dict[str, object]:
    expected = copy.deepcopy(dict(control))
    train = dict(_mapping(expected.get("train"), "config contract train"))
    optimizer = dict(_mapping(expected.get("optimizer"), "config contract optimizer"))
    expected["train"] = train
    expected["optimizer"] = optimizer
    if arm == "ring10-optimizer-clip-norm-2":
        train["gradient_clip_norm"] = 2.0
    elif arm == "ring10-optimizer-clip-norm-5":
        train["gradient_clip_norm"] = 5.0
    elif arm == FOLLOW_ON_ARM:
        optimizer["adamw_lr"] = 0.5 * _positive_number(
            optimizer.get("adamw_lr"),
            "control AdamW learning rate",
        )
        optimizer["muon_lr"] = 0.5 * _positive_number(
            optimizer.get("muon_lr"),
            "control Muon learning rate",
        )
    elif arm != CONTROL_ARM:
        raise ValueError(f"unsupported optimizer calibration arm: {arm}")
    return expected


def _optimizer_structure(payload: Mapping[str, object]) -> dict[str, object]:
    optimizer = _mapping(payload.get("optimizer"), "optimizer")
    if (
        optimizer.get("fresh_from_champion_ema") is not True
        or optimizer.get("source_optimizer_loaded") is not False
        or optimizer.get("source_scheduler_loaded") is not False
    ):
        raise ValueError("arm did not start with fresh optimizer/scheduler state")
    routing = _mapping(optimizer.get("routing"), "optimizer routing")
    if (
        routing.get("requested_kind") != "muon_adamw"
        or routing.get("implementation") != "muon_adamw"
        or routing.get("fallback_used") is not False
    ):
        raise ValueError("arm did not use runtime Muon+AdamW")
    groups = routing.get("groups")
    if not isinstance(groups, list) or not groups:
        raise ValueError("optimizer routing groups are missing")
    return {
        "schema_version": routing.get("schema_version"),
        "requested_kind": routing.get("requested_kind"),
        "implementation": routing.get("implementation"),
        "fallback_used": routing.get("fallback_used"),
        "parameter_tensors": routing.get("parameter_tensors"),
        "parameter_elements": routing.get("parameter_elements"),
        "groups": groups,
    }


def _finite_training(payload: Mapping[str, object]) -> tuple[bool, list[str]]:
    failures: list[str] = []
    training = _mapping(payload.get("training"), "training")
    if training.get("finite") is not True:
        failures.append("training did not declare finite metrics")
    for name in ("nonfinite_loss_count", "nonfinite_gradient_count"):
        if training.get(name) != 0:
            failures.append(f"{name} is nonzero")
    for name in (
        "examples_per_second",
        "measured_training_seconds",
        "mean_gradient_norm",
        "gradient_clipping_frequency",
    ):
        try:
            _finite_number(training.get(name), f"training {name}")
        except ValueError as error:
            failures.append(str(error))
    losses = training.get("mean_losses")
    if not isinstance(losses, Mapping) or not losses:
        failures.append("mean training losses are missing")
    elif any(
        isinstance(value, bool)
        or not isinstance(value, int | float)
        or not math.isfinite(float(value))
        for value in losses.values()
    ):
        failures.append("mean training losses are non-finite")
    heldout = _mapping(payload.get("heldout"), "heldout")
    if heldout.get("finite") is not True:
        failures.append("held-out evaluation is non-finite")
    return not failures, failures


def _budget_gate(payload: Mapping[str, object]) -> bool:
    training = _mapping(payload.get("training"), "training")
    budget = _positive_number(
        training.get("budget_h100_hours"),
        "declared H100-hour budget",
    )
    elapsed = _finite_number(training.get("elapsed_seconds"), "arm elapsed seconds")
    return (
        training.get("h100_count") == 1
        and budget <= MAX_H100_HOURS_PER_ARM
        and elapsed <= budget * 3600.0 + 1e-9
    )


def _reference_observations(
    payload: Mapping[str, object],
) -> tuple[tuple[int, dict[str, float]], ...]:
    heldout = _mapping(payload.get("heldout"), "heldout")
    raw = heldout.get("observations")
    if not isinstance(raw, list) or not raw:
        raise ValueError("held-out observations are missing")
    observations: list[tuple[int, dict[str, float]]] = []
    for index, value in enumerate(raw):
        observation = _mapping(value, f"held-out observation {index}")
        samples = observation.get("samples")
        if type(samples) is not int or samples <= 0:
            raise ValueError("held-out observation sample count is invalid")
        reference = _mapping(observation.get("reference"), "held-out reference")
        observations.append(
            (
                samples,
                {
                    component: _finite_number(
                        reference.get(component),
                        f"reference {component}",
                    )
                    for component in ("policy", "value", "composite")
                },
            )
        )
    return tuple(observations)


def _candidate_composites(
    payload: Mapping[str, object],
) -> tuple[tuple[int, float], ...]:
    heldout = _mapping(payload.get("heldout"), "heldout")
    raw = heldout.get("observations")
    if not isinstance(raw, list) or not raw:
        raise ValueError("held-out observations are missing")
    observations: list[tuple[int, float]] = []
    for index, value in enumerate(raw):
        observation = _mapping(value, f"held-out observation {index}")
        samples = observation.get("samples")
        if type(samples) is not int or samples <= 0:
            raise ValueError("held-out observation sample count is invalid")
        candidate = _mapping(observation.get("candidate"), "held-out candidate")
        observations.append(
            (
                samples,
                _finite_number(candidate.get("composite"), "candidate composite"),
            )
        )
    return tuple(observations)


def _reference_parity(
    control: Mapping[str, object],
    treatment: Mapping[str, object],
) -> tuple[bool, str | None]:
    control_observations = _reference_observations(control)
    treatment_observations = _reference_observations(treatment)
    if len(control_observations) != len(treatment_observations):
        return False, "reference observation counts differ"
    for (control_samples, control_loss), (samples, loss) in zip(
        control_observations,
        treatment_observations,
        strict=True,
    ):
        if control_samples != samples:
            return False, "reference held-out batch shapes differ"
        if any(
            not math.isclose(
                control_loss[component],
                loss[component],
                rel_tol=1e-6,
                abs_tol=1e-7,
            )
            for component in control_loss
        ):
            return False, "reference held-out losses differ"
    return True, None


def _paired_improvements(
    control: Mapping[str, object],
    treatment: Mapping[str, object],
) -> tuple[tuple[int, float], ...]:
    baseline = _candidate_composites(control)
    candidate = _candidate_composites(treatment)
    if len(baseline) != len(candidate):
        raise ValueError("candidate held-out observation counts differ")
    paired = []
    for (baseline_samples, baseline_loss), (samples, candidate_loss) in zip(
        baseline,
        candidate,
        strict=True,
    ):
        if baseline_samples != samples:
            raise ValueError("candidate held-out batch shapes differ")
        paired.append((samples, baseline_loss - candidate_loss))
    return tuple(paired)


def _weighted_mean(values: Sequence[tuple[int, float]]) -> float:
    total = sum(weight for weight, _value in values)
    if total <= 0:
        raise ValueError("paired held-out observations have no samples")
    return sum(weight * value for weight, value in values) / total


def one_sided_bootstrap_lower_bound(
    values: Sequence[tuple[int, float]],
    *,
    confidence: float,
    samples: int,
    seed: int,
) -> float | None:
    """Return a deterministic paired-bootstrap lower bound for loss reduction."""

    if not 0 < confidence < 1:
        raise ValueError("confidence must be in (0, 1)")
    if samples <= 0:
        raise ValueError("bootstrap samples must be positive")
    if len(values) < 2:
        return None
    generator = random.Random(seed)
    count = len(values)
    bootstrap = []
    for _ in range(samples):
        resample = [values[generator.randrange(count)] for _ in range(count)]
        bootstrap.append(_weighted_mean(resample))
    bootstrap.sort()
    index = max(0, math.ceil((1.0 - confidence) * samples) - 1)
    return bootstrap[index]


def _clipping_diagnostic(
    control: Mapping[str, object],
    treatment: Mapping[str, object],
) -> dict[str, object]:
    control_training = _mapping(control.get("training"), "control training")
    treatment_training = _mapping(treatment.get("training"), "treatment training")
    baseline = _finite_number(
        control_training.get("gradient_clipping_frequency"),
        "control clipping frequency",
    )
    measured = _finite_number(
        treatment_training.get("gradient_clipping_frequency"),
        "treatment clipping frequency",
    )
    return {
        "gate": False,
        "diagnostic_only": True,
        "control_frequency": baseline,
        "treatment_frequency": measured,
        "absolute_reduction": baseline - measured,
        "relative_reduction": (baseline - measured) / baseline
        if baseline > 0
        else None,
    }


def compare_results(
    paths: Sequence[Path],
    *,
    confidence: float = DEFAULT_CONFIDENCE,
    bootstrap_samples: int = DEFAULT_BOOTSTRAP_SAMPLES,
) -> dict[str, object]:
    if not 0 < confidence < 1:
        raise ValueError("confidence must be in (0, 1)")
    if bootstrap_samples <= 0:
        raise ValueError("bootstrap samples must be positive")
    loaded = [_read_result(path) for path in paths]
    payloads: dict[str, dict[str, object]] = {}
    pins: dict[str, dict[str, object]] = {}
    for payload, pin in loaded:
        arm = _arm(payload)
        if arm in payloads:
            raise ValueError(f"duplicate calibration result for {arm}")
        payloads[arm] = payload
        pins[arm] = pin
    expected = set(RING10_OPTIMIZER_CALIBRATION_TREATMENTS)
    if set(payloads) != expected:
        raise ValueError(
            "comparison requires the complete frozen optimizer calibration suite"
        )
    compile_caches = {arm: _compile_cache(payload) for arm, payload in payloads.items()}
    cache_roots = [
        Path(str(cache["root"]))
        for cache in compile_caches.values()
        if cache is not None
    ]
    for index, left in enumerate(cache_roots):
        for right in cache_roots[index + 1 :]:
            if left == right or left in right.parents or right in left.parents:
                raise ValueError(
                    "compiled calibration cache roots must be distinct and disjoint"
                )
    control = payloads[CONTROL_ARM]
    treatment_count = len(expected) - 1
    per_treatment_confidence = 1.0 - (1.0 - confidence) / treatment_count
    common = _common_projection(control)
    suite_common_parity = all(
        _common_projection(payload) == common for payload in payloads.values()
    )
    common_sha256 = _digest(common)
    control_contract = _mapping(
        control.get("config_contract"),
        "control config contract",
    )
    control_structure = _optimizer_structure(control)
    control_throughput = _positive_number(
        _mapping(control.get("training"), "control training").get(
            "examples_per_second"
        ),
        "control throughput",
    )
    _, control_failures = _finite_training(control)

    assessments: dict[str, object] = {}
    eligible: list[tuple[str, float]] = []
    for arm in RING10_OPTIMIZER_CALIBRATION_TREATMENTS:
        payload = payloads[arm]
        common_parity = suite_common_parity and _common_projection(payload) == common
        actual_contract = _mapping(payload.get("config_contract"), "config contract")
        one_factor = dict(actual_contract) == _expected_config_contract(
            control_contract,
            arm,
        )
        optimizer_parity = _optimizer_structure(payload) == control_structure
        reference_parity, reference_error = _reference_parity(control, payload)
        finite, finite_failures = _finite_training(payload)
        bounded_budget = _budget_gate(payload)
        throughput = _positive_number(
            _mapping(payload.get("training"), "training").get("examples_per_second"),
            f"{arm} throughput",
        )
        throughput_ratio = throughput / control_throughput
        throughput_gate = (
            throughput_ratio + 1e-12 >= MINIMUM_CONTROL_THROUGHPUT_FRACTION
        )
        clipping = _clipping_diagnostic(control, payload)

        paired = () if arm == CONTROL_ARM else _paired_improvements(control, payload)
        mean_improvement = 0.0 if arm == CONTROL_ARM else _weighted_mean(paired)
        lower = (
            0.0
            if arm == CONTROL_ARM
            else one_sided_bootstrap_lower_bound(
                paired,
                confidence=per_treatment_confidence,
                samples=bootstrap_samples,
                seed=int(
                    hashlib.sha256(f"{common_sha256}:{arm}".encode()).hexdigest()[:16],
                    16,
                ),
            )
        )
        heldout_gate = arm == CONTROL_ARM or (lower is not None and lower > 0.0)
        gates = {
            "common_source_and_partition_parity": common_parity,
            "strict_one_factor_config": one_factor,
            "optimizer_runtime_parity": optimizer_parity,
            "reference_evaluation_parity": reference_parity,
            "finite_training_and_evaluation": finite,
            "bounded_h100_hour_budget": bounded_budget,
            "throughput_at_least_90_percent_of_control": throughput_gate,
            "one_sided_heldout_composite_improvement": heldout_gate,
        }
        passes = all(gates.values())
        if arm != CONTROL_ARM and passes:
            eligible.append((arm, mean_improvement))
        assessments[arm] = {
            "label": RING10_OPTIMIZER_CALIBRATION_LABELS[arm],
            "phase": "follow_on" if arm == FOLLOW_ON_ARM else "primary",
            "gates": gates,
            "passes_all_selection_gates": passes,
            "gate_failures": [name for name, passed in gates.items() if not passed],
            "finite_failures": finite_failures,
            "reference_parity_error": reference_error,
            "throughput": {
                "examples_per_second": throughput,
                "control_fraction": throughput_ratio,
                "minimum_control_fraction": MINIMUM_CONTROL_THROUGHPUT_FRACTION,
            },
            "heldout": {
                "paired_observations": len(paired),
                "mean_composite_improvement": mean_improvement,
                "one_sided_confidence": per_treatment_confidence,
                "familywise_confidence": confidence,
                "familywise_method": "bonferroni",
                "family_size": treatment_count,
                "bootstrap_samples": bootstrap_samples,
                "one_sided_lower_bound": lower,
                "strict_positive_lower_bound_required": True,
            },
            "clip_reduction": clipping,
        }

    control_assessment = _mapping(
        assessments[CONTROL_ARM],
        "control assessment",
    )
    control_valid = control_assessment.get("passes_all_selection_gates") is True
    control_gate_failures = control_assessment.get("gate_failures")
    if not isinstance(control_gate_failures, list):
        raise RuntimeError("control gate failures are invalid")
    fallback = True
    selected = CONTROL_ARM
    reason = "no treatment passed every gate; retained runtime-effective control"
    if not control_valid:
        reason = "control evidence gate failed; fail-closed fallback retained control"
    elif eligible:
        best = max(value for _arm_name, value in eligible)
        winners = [
            arm_name
            for arm_name, value in eligible
            if math.isclose(value, best, rel_tol=0.0, abs_tol=1e-12)
        ]
        if len(winners) == 1:
            selected = winners[0]
            fallback = False
            reason = "selected the unique treatment passing every pre-registered gate"
        else:
            reason = "top treatment evidence tied; retained runtime-effective control"

    comparison: dict[str, object] = {
        "format": FORMAT,
        "schema_version": SCHEMA_VERSION,
        "status": "ok",
        "diagnostic_only": True,
        "production_promotion_authorized": False,
        "input_results": pins,
        "common_contract": {
            "sha256": common_sha256,
            "suite_common_parity": suite_common_parity,
            "source_champion_manifest_sha256": _mapping(
                _mapping(common["champion"], "common champion").get("manifest"),
                "common champion manifest",
            ).get("sha256"),
            "source_champion_checkpoint_sha256": _mapping(
                _mapping(common["champion"], "common champion").get("checkpoint"),
                "common champion checkpoint",
            ).get("sha256"),
            "replay_cutoff_sha256": _mapping(
                common["replay"],
                "common replay",
            ).get("cutoff_sha256"),
            "partition_sha256": _mapping(
                common["partition"],
                "common partition",
            ).get("partition_sha256"),
        },
        "gate_contract": {
            "finite_training_required": True,
            "strict_optimizer_and_reference_parity_required": True,
            "minimum_control_throughput_fraction": (
                MINIMUM_CONTROL_THROUGHPUT_FRACTION
            ),
            "one_sided_confidence": confidence,
            "per_treatment_confidence": per_treatment_confidence,
            "familywise_method": "bonferroni",
            "family_size": treatment_count,
            "bootstrap_samples": bootstrap_samples,
            "strict_positive_heldout_lower_bound_required": True,
            "clip_reduction_is_diagnostic_only": True,
            "isolated_compile_cache_required": True,
            "ties_and_no_winner_fall_back_to_control": True,
        },
        "control_valid": control_valid,
        "control_failures": [*control_failures, *control_gate_failures],
        "arms": assessments,
        "selection": {
            "selected_arm": selected,
            "selected_label": RING10_OPTIMIZER_CALIBRATION_LABELS[selected],
            "fallback_to_control": fallback,
            "reason": reason,
        },
    }
    comparison["semantic_sha256"] = _digest(comparison)
    return comparison


def main(argv: list[str] | None = None) -> int:
    arguments = _parser().parse_args(argv)
    try:
        result = compare_results(
            arguments.result,
            confidence=arguments.confidence,
            bootstrap_samples=arguments.bootstrap_samples,
        )
        output = arguments.output.expanduser().resolve()
        atomic_json(output, result)
    except (OSError, TypeError, ValueError) as error:
        print(
            json.dumps(
                {
                    "format": FORMAT,
                    "schema_version": SCHEMA_VERSION,
                    "status": "error",
                    "error": {
                        "type": type(error).__name__,
                        "message": str(error),
                    },
                },
                sort_keys=True,
            )
        )
        return 2
    print(json.dumps(result, allow_nan=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
