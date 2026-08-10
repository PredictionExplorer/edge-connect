#!/usr/bin/env python3
"""Execute a frozen ring-10 arena occupancy benchmark on one CUDA device."""

from __future__ import annotations

import argparse
import csv
import gc
import hashlib
import io
import json
import math
import os
import platform
import statistics
import subprocess
import sys
import threading
import time
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import torch

from startrain.arena import ArenaRunner
from startrain.checkpoint import sha256_file
from startrain.device import (
    empty_device_cache,
    peak_memory_stats,
    reset_peak_memory_stats,
    synchronize_device,
)
from startrain.native import load_star_native
from startrain.orchestration import CoordinatorLock
from startrain.promotion import load_manifest_evaluator
from startrain.runtime import atomic_json

if __package__:
    from .evaluate_archived_manifests import selection_arena_config
    from .prepare_arena_occupancy_benchmark import (
        CONTROL_ARM,
        REPOSITORY_ROOT,
        REPORT_NAME as PLAN_REPORT_NAME,
        SCHEMA_VERSION,
        TREATMENT_ARM,
        _assert_stopped_source,
        _native_extension_path,
        verify_arena_occupancy_plan,
    )
    from .run_elo_ablation_queue import exclusive_execution_lock
else:
    from evaluate_archived_manifests import selection_arena_config
    from prepare_arena_occupancy_benchmark import (
        CONTROL_ARM,
        REPOSITORY_ROOT,
        REPORT_NAME as PLAN_REPORT_NAME,
        SCHEMA_VERSION,
        TREATMENT_ARM,
        _assert_stopped_source,
        _native_extension_path,
        verify_arena_occupancy_plan,
    )
    from run_elo_ablation_queue import exclusive_execution_lock

REPORT_NAME = "startrain-arena-occupancy-benchmark"
REPORT_FILE = "occupancy-report.json"
PROGRESS_FILE = "benchmark-progress.json"
FAILURE_FILE = "benchmark-failure.json"
GPU_FIELDS = (
    "index",
    "uuid",
    "pci.bus_id",
    "name",
    "driver_version",
    "utilization.gpu",
    "memory.used",
    "power.draw",
    "temperature.gpu",
)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser


def _number(value: str) -> float | None:
    if value in {"N/A", "[N/A]"}:
        return None
    try:
        parsed = float(value)
    except ValueError:
        return None
    return parsed if math.isfinite(parsed) else None


def query_gpu_status(gpu_index: int) -> dict[str, object]:
    try:
        completed = subprocess.run(
            [
                "nvidia-smi",
                f"--query-gpu={','.join(GPU_FIELDS)}",
                "--format=csv,noheader,nounits",
            ],
            check=False,
            capture_output=True,
            text=True,
            timeout=10,
        )
    except (OSError, subprocess.SubprocessError) as error:
        raise RuntimeError(f"nvidia-smi query failed: {error}") from error
    if completed.returncode != 0:
        raise RuntimeError(
            f"nvidia-smi failed with exit {completed.returncode}: "
            f"{completed.stderr.strip()}"
        )
    for row in csv.reader(io.StringIO(completed.stdout)):
        values = [value.strip() for value in row]
        if len(values) != len(GPU_FIELDS):
            continue
        try:
            index = int(values[0])
        except ValueError:
            continue
        if index != gpu_index:
            continue
        return {
            "timestamp_ns": time.time_ns(),
            "index": index,
            "uuid": values[1],
            "pci_bus_id": values[2],
            "name": values[3],
            "driver_version": values[4],
            "utilization_gpu_percent": _number(values[5]),
            "memory_used_mib": _number(values[6]),
            "power_draw_watts": _number(values[7]),
            "temperature_c": _number(values[8]),
        }
    raise RuntimeError(f"nvidia-smi did not report GPU {gpu_index}")


class GPUStatusSampler:
    def __init__(self, gpu_index: int, interval_seconds: float) -> None:
        self.gpu_index = gpu_index
        self.interval_seconds = interval_seconds
        self.samples: list[dict[str, object]] = []
        self._error: BaseException | None = None
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None

    def start(self) -> None:
        self._thread = threading.Thread(
            target=self._sample_until_stopped,
            name=f"arena-gpu-{self.gpu_index}-sampler",
            daemon=True,
        )
        self._thread.start()

    def _sample_until_stopped(self) -> None:
        while not self._stop.is_set():
            try:
                self.samples.append(query_gpu_status(self.gpu_index))
            except BaseException as error:
                self._error = error
                self._stop.set()
                return
            if self._stop.wait(self.interval_seconds):
                return

    def stop(self) -> list[dict[str, object]]:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=max(10.0, self.interval_seconds * 2))
            if self._thread.is_alive():
                raise RuntimeError("GPU sampler did not stop")
        if self._error is not None:
            raise RuntimeError(f"GPU sampling failed: {self._error}") from self._error
        if not self.samples:
            raise RuntimeError("GPU sampler produced no samples")
        return list(self.samples)


def _normalized_uuid(value: object) -> str:
    text = str(value).strip().lower()
    if text.startswith("gpu-"):
        text = text[4:]
    return text


def verify_cuda_device_identity(
    device: str,
    physical_gpu_index: int,
) -> dict[str, object]:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable")
    torch_device = torch.device(device)
    if torch_device.type != "cuda":
        raise RuntimeError("arena occupancy benchmark requires a CUDA device")
    logical_index = (
        torch_device.index
        if torch_device.index is not None
        else torch.cuda.current_device()
    )
    properties = torch.cuda.get_device_properties(logical_index)
    torch_uuid = getattr(properties, "uuid", None)
    if torch_uuid is None:
        raise RuntimeError("PyTorch did not expose the CUDA device UUID")
    physical = query_gpu_status(physical_gpu_index)
    physical_uuid = physical.get("uuid")
    if not isinstance(physical_uuid, str) or _normalized_uuid(
        torch_uuid
    ) != _normalized_uuid(physical_uuid):
        raise RuntimeError(
            "logical CUDA device does not match the requested physical GPU"
        )
    return {
        "logical_device": device,
        "logical_index": logical_index,
        "physical_index": physical_gpu_index,
        "uuid": physical_uuid,
        "pci_bus_id": physical.get("pci_bus_id"),
        "name": physical.get("name"),
        "driver_version": physical.get("driver_version"),
        "total_memory_bytes": properties.total_memory,
        "compute_capability": [properties.major, properties.minor],
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "cuda_device_order": os.environ.get("CUDA_DEVICE_ORDER"),
    }


def assert_physical_gpu_idle(physical_gpu_index: int) -> None:
    physical = query_gpu_status(physical_gpu_index)
    uuid = physical.get("uuid")
    if not isinstance(uuid, str) or not uuid:
        raise RuntimeError("physical GPU UUID is unavailable")
    try:
        completed = subprocess.run(
            [
                "nvidia-smi",
                "--query-compute-apps=gpu_uuid,pid,process_name",
                "--format=csv,noheader,nounits",
            ],
            check=False,
            capture_output=True,
            text=True,
            timeout=10,
        )
    except (OSError, subprocess.SubprocessError) as error:
        raise RuntimeError(f"cannot inspect GPU compute processes: {error}") from error
    if completed.returncode != 0:
        raise RuntimeError(
            "cannot inspect GPU compute processes: "
            f"{completed.stderr.strip() or completed.returncode}"
        )
    active = []
    for row in csv.reader(io.StringIO(completed.stdout)):
        values = [value.strip() for value in row]
        if len(values) >= 2 and _normalized_uuid(values[0]) == _normalized_uuid(uuid):
            active.append(
                {
                    "pid": values[1],
                    "process_name": values[2] if len(values) > 2 else None,
                }
            )
    if active:
        raise RuntimeError(
            f"physical GPU {physical_gpu_index} already has compute processes: {active}"
        )


def runtime_metadata(
    *,
    native_module: Any,
    device_identity: Mapping[str, object],
) -> dict[str, object]:
    native_rules_hash = getattr(native_module, "native_rules_hash", None)
    native_artifact: dict[str, object] | None = None
    if callable(native_rules_hash):
        native_path = _native_extension_path(native_module)
        rules_hash_value = native_rules_hash()
        if native_path.is_file() and type(rules_hash_value) is int:
            native_artifact = {
                "path": str(native_path),
                "bytes": native_path.stat().st_size,
                "sha256": sha256_file(native_path),
                "rules_hash": f"fnv1a64:{rules_hash_value:016x}",
            }
    return {
        "host": {
            "system": platform.system(),
            "release": platform.release(),
            "machine": platform.machine(),
        },
        "python": sys.version,
        "torch": torch.__version__,
        "torch_cuda": torch.version.cuda,
        "cudnn": (
            torch.backends.cudnn.version()
            if torch.backends.cudnn.is_available()
            else None
        ),
        "device": dict(device_identity),
        "native_extension": native_artifact,
    }


def _percentile(values: Sequence[float], quantile: float) -> float:
    ordered = sorted(values)
    if not ordered:
        raise ValueError("cannot compute a percentile of an empty series")
    if len(ordered) == 1:
        return ordered[0]
    position = (len(ordered) - 1) * quantile
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    fraction = position - lower
    return ordered[lower] * (1.0 - fraction) + ordered[upper] * fraction


def _stats(values: Sequence[float]) -> dict[str, float | int]:
    if not values:
        raise ValueError("metric series is empty")
    return {
        "count": len(values),
        "minimum": min(values),
        "mean": statistics.fmean(values),
        "p50": _percentile(values, 0.50),
        "p95": _percentile(values, 0.95),
        "maximum": max(values),
    }


def summarize_gpu_samples(
    samples: Sequence[Mapping[str, object]],
) -> dict[str, object]:
    def series(name: str) -> list[float]:
        output = []
        for sample in samples:
            value = sample.get(name)
            if isinstance(value, bool) or not isinstance(value, int | float):
                continue
            number = float(value)
            if math.isfinite(number):
                output.append(number)
        return output

    utilization = series("utilization_gpu_percent")
    if not utilization:
        raise ValueError("GPU samples omitted utilization")
    summary: dict[str, object] = {
        "samples": len(samples),
        "utilization_gpu_percent": _stats(utilization),
    }
    for name in ("memory_used_mib", "power_draw_watts", "temperature_c"):
        values = series(name)
        if values:
            summary[name] = _stats(values)
    return summary


def _pairs_digest(raw_pairs: object) -> str:
    if not isinstance(raw_pairs, list):
        raise ValueError("arena result omitted pairs")
    encoded = (
        json.dumps(
            raw_pairs,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        )
        + "\n"
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _metric(result: Mapping[str, object], name: str) -> float:
    metrics = result.get("evaluation_metrics")
    if not isinstance(metrics, Mapping):
        raise ValueError("arena result omitted evaluation_metrics")
    value = metrics.get(name)
    if isinstance(value, bool) or not isinstance(value, int | float):
        raise ValueError(f"arena metric {name} is missing")
    number = float(value)
    if not math.isfinite(number):
        raise ValueError(f"arena metric {name} is non-finite")
    return number


def _validated_pairs(
    raw_pairs: object,
    *,
    pair_start: int,
    pair_count: int,
) -> list[dict[str, object]]:
    if not isinstance(raw_pairs, list) or len(raw_pairs) != pair_count:
        raise RuntimeError("arena occupancy wave did not complete its pair budget")
    pairs = []
    for offset, raw_pair in enumerate(raw_pairs):
        if not isinstance(raw_pair, dict):
            raise RuntimeError("arena occupancy wave emitted a malformed pair")
        if raw_pair.get("ring") != 10 or raw_pair.get("pair") != pair_start + offset:
            raise RuntimeError("arena occupancy wave emitted an unexpected pair range")
        pairs.append(raw_pair)
    return pairs


def _validated_search(result: Mapping[str, object]) -> dict[str, object]:
    raw = result.get("search")
    if not isinstance(raw, dict):
        raise RuntimeError("arena occupancy wave omitted its search contract")
    if (
        raw.get("search_workers") != 2
        or raw.get("inference_workers") != 1
        or raw.get("pair_chunk_size") is not None
        or raw.get("effective_pair_chunking") != "full_requested_ring_batch"
    ):
        raise RuntimeError("arena occupancy wave changed its search contract")
    return raw


def _run_wave(
    *,
    runner: ArenaRunner,
    device: str,
    physical_gpu_index: int,
    sample_interval_seconds: float,
    phase: str,
    arm: str,
    repeat: int,
    pair_start: int,
    pair_count: int,
) -> dict[str, object]:
    reset_peak_memory_stats(device)
    sampler = GPUStatusSampler(physical_gpu_index, sample_interval_seconds)
    workload_started_ns = time.time_ns()
    sampler.start()
    result: dict[str, object] | None = None
    workload_error: BaseException | None = None
    try:
        result = runner.run(
            pair_starts={10: pair_start},
            pair_counts={10: pair_count},
        )
    except BaseException as error:
        workload_error = error
    workload_completed_ns = time.time_ns()
    try:
        raw_samples = sampler.stop()
    except BaseException as sampler_error:
        if workload_error is not None:
            workload_error.add_note(f"GPU sampler also failed: {sampler_error}")
            raise workload_error from sampler_error
        raise
    if workload_error is not None:
        raise workload_error
    if result is None:
        raise RuntimeError("arena occupancy wave returned no result")
    synchronize_device(device)
    peak_allocated, peak_reserved = peak_memory_stats(device)
    samples = []
    for sample in raw_samples:
        observed_ns = sample.get("timestamp_ns")
        if (
            type(observed_ns) is int
            and workload_started_ns <= observed_ns <= workload_completed_ns
        ):
            samples.append(sample)
    if samples:
        samples = samples[1:]
    if not samples:
        raise RuntimeError("GPU sampler produced no post-warmup in-window samples")
    pairs = _validated_pairs(
        result.get("pairs"),
        pair_start=pair_start,
        pair_count=pair_count,
    )
    return {
        "phase": phase,
        "arm": arm,
        "repeat": repeat,
        "pair_start": pair_start,
        "pair_count": pair_count,
        "completed_pairs": len(pairs),
        "pairs_sha256": _pairs_digest(pairs),
        "pairs": pairs,
        "search": _validated_search(result),
        "workload_started_ns": workload_started_ns,
        "workload_completed_ns": workload_completed_ns,
        "evaluation_metrics": {
            "wall_seconds": _metric(result, "wall_seconds"),
            "total_evaluator_rows": _metric(result, "total_evaluator_rows"),
            "evaluator_rows_per_second": _metric(
                result,
                "evaluator_rows_per_second",
            ),
            "serialized_inference_seconds": _metric(
                result,
                "serialized_inference_seconds",
            ),
            "inference_queue_wait_seconds": _metric(
                result,
                "inference_queue_wait_seconds",
            ),
            "peak_cuda_allocated_bytes": peak_allocated,
            "peak_cuda_reserved_bytes": peak_reserved,
        },
        "gpu_samples": samples,
        "discarded_initial_gpu_samples": 1,
        "gpu_summary": summarize_gpu_samples(samples),
    }


def _run_arm(
    *,
    runner: ArenaRunner,
    device: str,
    physical_gpu_index: int,
    sample_interval_seconds: float,
    phase: str,
    arm: Mapping[str, object],
    repeat: int,
    pair_start: int,
) -> dict[str, object]:
    arm_name = str(arm["name"])
    total_pair_count = _required_int(arm, "total_pair_count")
    chunk_pair_count = _required_int(arm, "chunk_pair_count")
    chunks_expected = _required_int(arm, "chunks")
    if (
        total_pair_count <= 0
        or chunk_pair_count <= 0
        or total_pair_count != chunk_pair_count * chunks_expected
    ):
        raise ValueError(f"benchmark arm {arm_name} has an invalid chunk contract")
    chunks = []
    for chunk_index in range(chunks_expected):
        chunks.append(
            _run_wave(
                runner=runner,
                device=device,
                physical_gpu_index=physical_gpu_index,
                sample_interval_seconds=sample_interval_seconds,
                phase=phase,
                arm=arm_name,
                repeat=repeat,
                pair_start=pair_start + chunk_index * chunk_pair_count,
                pair_count=chunk_pair_count,
            )
        )
    pairs = []
    samples = []
    for chunk in chunks:
        raw_pairs = chunk.get("pairs")
        raw_samples = chunk.get("gpu_samples")
        assert isinstance(raw_pairs, list) and isinstance(raw_samples, list)
        pairs.extend(raw_pairs)
        samples.extend(raw_samples)
    _validated_pairs(
        pairs,
        pair_start=pair_start,
        pair_count=total_pair_count,
    )
    wall_seconds = sum(
        _nested_metric(chunk, "evaluation_metrics", "wall_seconds") for chunk in chunks
    )
    total_rows = sum(
        _nested_metric(chunk, "evaluation_metrics", "total_evaluator_rows")
        for chunk in chunks
    )
    peak_allocated = [
        _nested_metric(chunk, "evaluation_metrics", "peak_cuda_allocated_bytes")
        for chunk in chunks
        if isinstance(chunk.get("evaluation_metrics"), Mapping)
        and chunk["evaluation_metrics"].get("peak_cuda_allocated_bytes") is not None
    ]
    peak_reserved = [
        _nested_metric(chunk, "evaluation_metrics", "peak_cuda_reserved_bytes")
        for chunk in chunks
        if isinstance(chunk.get("evaluation_metrics"), Mapping)
        and chunk["evaluation_metrics"].get("peak_cuda_reserved_bytes") is not None
    ]
    return {
        "phase": phase,
        "arm": arm_name,
        "repeat": repeat,
        "pair_start": pair_start,
        "total_pair_count": total_pair_count,
        "chunk_pair_count": chunk_pair_count,
        "chunks": chunks,
        "completed_pairs": len(pairs),
        "pairs_sha256": _pairs_digest(pairs),
        "pairs": pairs,
        "evaluation_metrics": {
            "wall_seconds": wall_seconds,
            "total_evaluator_rows": total_rows,
            "evaluator_rows_per_second": (
                total_rows / wall_seconds if wall_seconds > 0 else 0.0
            ),
            "serialized_inference_seconds": sum(
                _nested_metric(
                    chunk,
                    "evaluation_metrics",
                    "serialized_inference_seconds",
                )
                for chunk in chunks
            ),
            "inference_queue_wait_seconds": sum(
                _nested_metric(
                    chunk,
                    "evaluation_metrics",
                    "inference_queue_wait_seconds",
                )
                for chunk in chunks
            ),
            "peak_cuda_allocated_bytes": (
                max(peak_allocated) if peak_allocated else None
            ),
            "peak_cuda_reserved_bytes": max(peak_reserved) if peak_reserved else None,
        },
        "gpu_samples": samples,
        "gpu_summary": summarize_gpu_samples(samples),
    }


def _nested_metric(run: Mapping[str, object], group: str, name: str) -> float:
    values = run.get(group)
    if not isinstance(values, Mapping):
        raise ValueError(f"run omitted {group}")
    value = values.get(name)
    if isinstance(value, bool) or not isinstance(value, int | float):
        raise ValueError(f"run omitted {group}.{name}")
    number = float(value)
    if not math.isfinite(number):
        raise ValueError(f"run metric {group}.{name} is non-finite")
    return number


def _required_int(values: Mapping[str, object], name: str) -> int:
    value = values.get(name)
    if type(value) is not int:
        raise ValueError(f"{name} must be an integer")
    return value


def aggregate_benchmark_runs(
    runs: Sequence[Mapping[str, object]],
) -> dict[str, object]:
    output: dict[str, object] = {}
    for arm in (CONTROL_ARM, TREATMENT_ARM):
        selected = [
            run
            for run in runs
            if run.get("phase") == "measurement" and run.get("arm") == arm
        ]
        if not selected:
            raise ValueError(f"benchmark omitted measurement runs for {arm}")
        gpu_samples = []
        for run in selected:
            raw_samples = run.get("gpu_samples")
            if not isinstance(raw_samples, list):
                raise ValueError("benchmark run omitted gpu_samples")
            gpu_samples.extend(raw_samples)
        peak_allocated = []
        for run in selected:
            metrics = run.get("evaluation_metrics")
            if (
                isinstance(metrics, Mapping)
                and metrics.get("peak_cuda_allocated_bytes") is not None
            ):
                peak_allocated.append(
                    _nested_metric(
                        run,
                        "evaluation_metrics",
                        "peak_cuda_allocated_bytes",
                    )
                )
        output[arm] = {
            "runs": len(selected),
            "total_pair_count": selected[0].get("total_pair_count"),
            "gpu": summarize_gpu_samples(gpu_samples),
            "evaluator_rows_per_second": _stats(
                [
                    _nested_metric(
                        run,
                        "evaluation_metrics",
                        "evaluator_rows_per_second",
                    )
                    for run in selected
                ]
            ),
            "wall_seconds": _stats(
                [
                    _nested_metric(run, "evaluation_metrics", "wall_seconds")
                    for run in selected
                ]
            ),
            "inference_queue_wait_seconds": _stats(
                [
                    _nested_metric(
                        run,
                        "evaluation_metrics",
                        "inference_queue_wait_seconds",
                    )
                    for run in selected
                ]
            ),
            "peak_cuda_allocated_bytes": (
                _stats(peak_allocated) if peak_allocated else None
            ),
        }
    control = output[CONTROL_ARM]
    treatment = output[TREATMENT_ARM]
    assert isinstance(control, Mapping) and isinstance(treatment, Mapping)
    control_gpu = control["gpu"]
    treatment_gpu = treatment["gpu"]
    control_rows = control["evaluator_rows_per_second"]
    treatment_rows = treatment["evaluator_rows_per_second"]
    assert isinstance(control_gpu, Mapping) and isinstance(treatment_gpu, Mapping)
    assert isinstance(control_rows, Mapping) and isinstance(treatment_rows, Mapping)
    control_util = control_gpu["utilization_gpu_percent"]
    treatment_util = treatment_gpu["utilization_gpu_percent"]
    assert isinstance(control_util, Mapping) and isinstance(treatment_util, Mapping)
    paired_utilization_deltas = []
    paired_rows_ratios = []
    repeats = sorted(
        {
            _required_int(run, "repeat")
            for run in runs
            if run.get("phase") == "measurement"
        }
    )
    for repeat in repeats:
        by_arm = {
            str(run.get("arm")): run
            for run in runs
            if run.get("phase") == "measurement"
            and _required_int(run, "repeat") == repeat
        }
        if set(by_arm) != {CONTROL_ARM, TREATMENT_ARM}:
            raise ValueError(f"repeat {repeat} is not a complete paired block")
        if _required_int(
            by_arm[CONTROL_ARM],
            "pair_start",
        ) != _required_int(by_arm[TREATMENT_ARM], "pair_start"):
            raise ValueError(f"repeat {repeat} does not use one matched pair range")
        control_summary = by_arm[CONTROL_ARM].get("gpu_summary")
        treatment_summary = by_arm[TREATMENT_ARM].get("gpu_summary")
        if not isinstance(control_summary, Mapping) or not isinstance(
            treatment_summary,
            Mapping,
        ):
            raise ValueError("paired run omitted gpu_summary")
        control_repeat_util = control_summary.get("utilization_gpu_percent")
        treatment_repeat_util = treatment_summary.get("utilization_gpu_percent")
        if not isinstance(control_repeat_util, Mapping) or not isinstance(
            treatment_repeat_util,
            Mapping,
        ):
            raise ValueError("paired run omitted utilization summary")
        paired_utilization_deltas.append(
            float(treatment_repeat_util["mean"]) - float(control_repeat_util["mean"])
        )
        control_repeat_rows = _nested_metric(
            by_arm[CONTROL_ARM],
            "evaluation_metrics",
            "evaluator_rows_per_second",
        )
        treatment_repeat_rows = _nested_metric(
            by_arm[TREATMENT_ARM],
            "evaluation_metrics",
            "evaluator_rows_per_second",
        )
        if control_repeat_rows <= 0:
            raise ValueError("control evaluator throughput must be positive")
        paired_rows_ratios.append(treatment_repeat_rows / control_repeat_rows)
    output["comparison"] = {
        "utilization_mean_delta_points": (
            float(treatment_util["mean"]) - float(control_util["mean"])
        ),
        "evaluator_rows_per_second_ratio": (
            float(treatment_rows["mean"]) / float(control_rows["mean"])
            if float(control_rows["mean"]) > 0
            else None
        ),
        "paired_utilization_delta_points": _stats(paired_utilization_deltas),
        "paired_evaluator_rows_per_second_ratio": _stats(paired_rows_ratios),
        "deployment_eligible": False,
    }
    return output


def _arm_map(payload: Mapping[str, object]) -> dict[str, dict[str, object]]:
    raw_arms = payload.get("arms")
    if not isinstance(raw_arms, list):
        raise ValueError("benchmark plan omitted arms")
    output = {}
    for raw in raw_arms:
        if not isinstance(raw, dict) or not isinstance(raw.get("name"), str):
            raise ValueError("benchmark arm is malformed")
        output[str(raw["name"])] = raw
    return output


def _write_progress(
    path: Path,
    *,
    plan_digest: str,
    status: str,
    started_ns: int,
    runs: Sequence[Mapping[str, object]],
) -> None:
    atomic_json(
        path,
        {
            "schema_version": SCHEMA_VERSION,
            "report": REPORT_NAME,
            "status": status,
            "plan_digest": plan_digest,
            "started_ns": started_ns,
            "updated_ns": time.time_ns(),
            "completed_runs": len(runs),
            "runs": list(runs),
        },
    )


def run_arena_occupancy_benchmark(
    *,
    plan_path: Path,
    output_dir: Path,
) -> dict[str, object]:
    verified = verify_arena_occupancy_plan(plan_path)
    payload = verified.payload
    source = Path(str(payload["source_run_root"])).expanduser().resolve()
    _assert_stopped_source(source)
    destination = output_dir.expanduser().resolve()
    if destination == source or source in destination.parents:
        raise ValueError("benchmark results must be outside the source run root")
    if destination == REPOSITORY_ROOT or REPOSITORY_ROOT in destination.parents:
        raise ValueError("benchmark results must be outside the Git repository")
    if destination.exists():
        raise FileExistsError(f"output directory already exists: {destination}")
    execution_lock_value = payload.get("execution_lock")
    if not isinstance(execution_lock_value, str):
        raise ValueError("benchmark plan omitted its execution lock")
    execution_lock_context = exclusive_execution_lock(Path(execution_lock_value))
    execution_lock_context.__enter__()
    source_lease = CoordinatorLock(source / "coordinator.lock")
    try:
        source_lease.acquire()
    except BaseException:
        execution_lock_context.__exit__(None, None, None)
        raise
    try:
        destination.mkdir(parents=True)
    except BaseException:
        source_lease.release()
        execution_lock_context.__exit__(None, None, None)
        raise

    plan_digest = str(payload["plan_digest"])
    progress_path = destination / PROGRESS_FILE
    failure_path = destination / FAILURE_FILE
    report_path = destination / REPORT_FILE
    started_ns = time.time_ns()
    completed_runs: list[dict[str, object]] = []
    try:
        _write_progress(
            progress_path,
            plan_digest=plan_digest,
            status="running",
            started_ns=started_ns,
            runs=completed_runs,
        )
    except BaseException:
        source_lease.release()
        execution_lock_context.__exit__(None, None, None)
        raise

    native_module: Any | None = None
    candidate_evaluator: Any | None = None
    baseline_evaluator: Any | None = None
    runner: ArenaRunner | None = None
    cleanup_complete = False
    try:
        device = str(payload["device"])
        physical_gpu_index = _required_int(payload, "physical_gpu_index")
        sample_interval_seconds = float(payload["sample_interval_seconds"])
        _assert_stopped_source(source, allowed_lock_pid=os.getpid())
        assert_physical_gpu_idle(physical_gpu_index)
        device_identity = verify_cuda_device_identity(device, physical_gpu_index)
        selection = verified.selection
        candidate_evidence = selection.candidates[0]
        candidate = candidate_evidence.verify()
        baseline = selection.source_champion.manifest.verify()
        native_module = load_star_native(required=True)
        if native_module is None:
            raise RuntimeError("native module is required")
        observed_runtime = runtime_metadata(
            native_module=native_module,
            device_identity=device_identity,
        )
        if observed_runtime.get("native_extension") != payload.get("native_extension"):
            raise RuntimeError("loaded native extension differs from the frozen plan")
        candidate_evaluator = load_manifest_evaluator(
            verified.experiment,
            candidate,
            device=device,
        )
        baseline_evaluator = load_manifest_evaluator(
            verified.experiment,
            baseline,
            device=device,
        )
        runner = ArenaRunner(
            native_module=native_module,
            candidate=candidate_evaluator,
            baseline=baseline_evaluator,
            config=selection_arena_config(
                verified.experiment,
                selection,
                candidate_evidence,
            ),
            search_workers=2,
        )
        arms = _arm_map(payload)
        warmup_order = payload["warmup_arm_order"]
        assert isinstance(warmup_order, list)
        for arm_name in warmup_order:
            arm = arms[str(arm_name)]
            completed_runs.append(
                _run_arm(
                    runner=runner,
                    device=device,
                    physical_gpu_index=physical_gpu_index,
                    sample_interval_seconds=sample_interval_seconds,
                    phase="warmup",
                    arm=arm,
                    repeat=-1,
                    pair_start=0,
                )
            )
            _write_progress(
                progress_path,
                plan_digest=plan_digest,
                status="running",
                started_ns=started_ns,
                runs=completed_runs,
            )

        schedule = payload["schedule"]
        assert isinstance(schedule, list)
        for raw_entry in schedule:
            assert isinstance(raw_entry, dict)
            repeat = _required_int(raw_entry, "repeat")
            pair_start = _required_int(raw_entry, "pair_start")
            arm_order = raw_entry["arm_order"]
            assert isinstance(arm_order, list)
            for arm_name in arm_order:
                arm = arms[str(arm_name)]
                completed_runs.append(
                    _run_arm(
                        runner=runner,
                        device=device,
                        physical_gpu_index=physical_gpu_index,
                        sample_interval_seconds=sample_interval_seconds,
                        phase="measurement",
                        arm=arm,
                        repeat=repeat,
                        pair_start=pair_start,
                    )
                )
                _write_progress(
                    progress_path,
                    plan_digest=plan_digest,
                    status="running",
                    started_ns=started_ns,
                    runs=completed_runs,
                )

        measurements = [
            run for run in completed_runs if run.get("phase") == "measurement"
        ]
        terminal_plan = verify_arena_occupancy_plan(
            verified.path,
            allowed_lock_pid=os.getpid(),
        )
        if terminal_plan.payload.get("plan_digest") != plan_digest:
            raise RuntimeError("benchmark plan changed during execution")
        aggregate = aggregate_benchmark_runs(measurements)
        runner = None
        candidate_evaluator = None
        baseline_evaluator = None
        native_module = None
        gc.collect()
        synchronize_device(device)
        empty_device_cache(device)
        cleanup_complete = True
        report: dict[str, object] = {
            "schema_version": SCHEMA_VERSION,
            "report": REPORT_NAME,
            "status": "complete",
            "plan_report": PLAN_REPORT_NAME,
            "plan_path": str(verified.path),
            "plan_digest": plan_digest,
            "started_ns": started_ns,
            "completed_ns": time.time_ns(),
            "device": device,
            "physical_gpu_index": physical_gpu_index,
            "runtime": observed_runtime,
            "candidate_identity": candidate.model_identity,
            "champion_identity": baseline.model_identity,
            "runs": completed_runs,
            "aggregate": aggregate,
            "deployment_policy": payload["deployment_policy"],
            "deployment_recommendation": "benchmark_only_no_profile_change",
        }
        atomic_json(report_path, report)
        os.chmod(report_path, 0o444)
        _write_progress(
            progress_path,
            plan_digest=plan_digest,
            status="complete",
            started_ns=started_ns,
            runs=completed_runs,
        )
        return report
    except BaseException as error:
        if report_path.exists() or report_path.is_symlink():
            try:
                os.chmod(report_path, 0o644)
                report_path.unlink()
            except OSError:
                pass
        cleanup_error: str | None = None
        if not cleanup_complete:
            runner = None
            candidate_evaluator = None
            baseline_evaluator = None
            native_module = None
            gc.collect()
            try:
                synchronize_device(str(payload["device"]))
                empty_device_cache(str(payload["device"]))
            except BaseException as cleanup_failure:
                cleanup_error = f"{type(cleanup_failure).__name__}: {cleanup_failure}"
        _write_progress(
            progress_path,
            plan_digest=plan_digest,
            status="error",
            started_ns=started_ns,
            runs=completed_runs,
        )
        atomic_json(
            failure_path,
            {
                "schema_version": SCHEMA_VERSION,
                "report": REPORT_NAME,
                "status": "error",
                "plan_digest": plan_digest,
                "started_ns": started_ns,
                "failed_ns": time.time_ns(),
                "completed_runs": len(completed_runs),
                "error": f"{type(error).__name__}: {error}",
                "cleanup_error": cleanup_error,
            },
        )
        raise
    finally:
        try:
            source_lease.release()
        finally:
            execution_lock_context.__exit__(None, None, None)


def main(argv: list[str] | None = None) -> int:
    arguments = _parser().parse_args(argv)
    try:
        report = run_arena_occupancy_benchmark(
            plan_path=arguments.plan,
            output_dir=arguments.output_dir,
        )
    except (
        FileExistsError,
        FileNotFoundError,
        OSError,
        RuntimeError,
        ValueError,
    ) as error:
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
    print(json.dumps({"status": "ok", **report}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
