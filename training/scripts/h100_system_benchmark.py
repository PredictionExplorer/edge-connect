#!/usr/bin/env python3
"""Run a bounded, reproducible sweep of the H100 inference preflight.

The harness only invokes ``hardware_preflight.py``. It never starts actors,
learners, or the training orchestrator. Results are written to a new output
directory as both a self-contained JSON summary and one-record-per-line JSONL.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import platform
import socket
import statistics
import subprocess
import sys
import time
import uuid
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Protocol

import torch

from startrain.topology import SUPPORTED_RINGS

SCHEMA_VERSION = 1
BENCHMARK_NAME = "h100-system-inference-sweep"
EXIT_OK = 0
EXIT_HARNESS_ERROR = 1
EXIT_BENCHMARK_FAILURE = 2
EXIT_METRICS_FAILURE = 3
EXIT_INTERRUPTED = 130

SCRIPT_PATH = Path(__file__).resolve()
TRAINING_ROOT = SCRIPT_PATH.parents[1]
REPOSITORY_ROOT = SCRIPT_PATH.parents[2]
PREFLIGHT_SCRIPT = SCRIPT_PATH.with_name("hardware_preflight.py")


class BenchmarkHarnessError(RuntimeError):
    """Raised when the harness cannot safely start."""


class ProcessExecutor(Protocol):
    def __call__(
        self,
        command: Sequence[str],
        *,
        cwd: Path,
        timeout_seconds: float,
    ) -> subprocess.CompletedProcess[str]: ...


@dataclass(frozen=True)
class BenchmarkSettings:
    config: Path
    output_directory: Path
    rings: tuple[int, ...] = (6, 10)
    batch_sizes: tuple[int, ...] = (64,)
    repeats: int = 3
    warmup: int = 10
    iterations: int = 50
    device: str = "cuda:0"
    minimum_leaves_per_second: float = 5_000.0
    timeout_seconds: float = 900.0
    metrics_root: Path | None = None
    compile_dynamic: bool | None = None
    compile_mode: str | None = None


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=__doc__,
        epilog=(
            "Exit status: 0 all cases passed; 1 harness/setup error; "
            "2 one or more inference cases failed; 3 metrics JSONL was incomplete."
        ),
    )
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--rings", type=int, nargs="+", default=[6, 10])
    parser.add_argument("--batch-sizes", type=int, nargs="+", default=[64])
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--iterations", type=int, default=50)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument(
        "--compile-dynamic",
        action=argparse.BooleanOptionalAction,
        default=None,
    )
    parser.add_argument(
        "--compile-mode",
        choices=("default", "reduce-overhead", "max-autotune"),
    )
    parser.add_argument(
        "--minimum-leaves-per-second",
        type=float,
        default=5_000.0,
    )
    parser.add_argument(
        "--timeout-seconds",
        type=float,
        default=900.0,
        help="hard timeout for each hardware_preflight.py subprocess",
    )
    parser.add_argument(
        "--metrics-root",
        type=Path,
        help=(
            "optional orchestration run root containing learner/metrics.jsonl "
            "and metrics/*.jsonl"
        ),
    )
    return parser


def _settings(arguments: argparse.Namespace) -> BenchmarkSettings:
    metrics_root = (
        arguments.metrics_root.expanduser().resolve()
        if arguments.metrics_root is not None
        else None
    )
    return BenchmarkSettings(
        config=arguments.config.expanduser().resolve(),
        output_directory=arguments.output_dir.expanduser().resolve(),
        rings=tuple(arguments.rings),
        batch_sizes=tuple(arguments.batch_sizes),
        repeats=arguments.repeats,
        warmup=arguments.warmup,
        iterations=arguments.iterations,
        device=arguments.device,
        minimum_leaves_per_second=arguments.minimum_leaves_per_second,
        timeout_seconds=arguments.timeout_seconds,
        metrics_root=metrics_root,
        compile_dynamic=arguments.compile_dynamic,
        compile_mode=arguments.compile_mode,
    )


def _discover_metric_paths(root: Path) -> tuple[Path, ...]:
    if root.is_file():
        return (root.resolve(),)
    candidates = [root / "learner" / "metrics.jsonl"]
    candidates.extend(sorted((root / "metrics").glob("*.jsonl")))
    if root.name == "metrics":
        candidates.extend(sorted(root.glob("*.jsonl")))
    return tuple(dict.fromkeys(path.resolve() for path in candidates if path.is_file()))


def validate_settings(settings: BenchmarkSettings) -> None:
    if not settings.config.is_file():
        raise BenchmarkHarnessError(f"config is not a file: {settings.config}")
    if not PREFLIGHT_SCRIPT.is_file():
        raise BenchmarkHarnessError(
            f"hardware preflight script is missing: {PREFLIGHT_SCRIPT}"
        )
    if settings.output_directory.exists():
        raise BenchmarkHarnessError(
            f"refusing to overwrite existing output path: {settings.output_directory}"
        )
    if not settings.rings or any(
        type(ring) is not int or ring not in SUPPORTED_RINGS for ring in settings.rings
    ):
        raise BenchmarkHarnessError("rings must be selected from 4, 6, 8, and 10")
    if len(set(settings.rings)) != len(settings.rings):
        raise BenchmarkHarnessError("rings must not contain duplicates")
    if not settings.batch_sizes or any(size <= 0 for size in settings.batch_sizes):
        raise BenchmarkHarnessError("batch sizes must be positive")
    if len(set(settings.batch_sizes)) != len(settings.batch_sizes):
        raise BenchmarkHarnessError("batch sizes must not contain duplicates")
    if settings.repeats <= 0:
        raise BenchmarkHarnessError("repeats must be positive")
    if settings.warmup < 0 or settings.iterations <= 0:
        raise BenchmarkHarnessError(
            "iterations must be positive and warmup must be non-negative"
        )
    if (
        not math.isfinite(settings.minimum_leaves_per_second)
        or settings.minimum_leaves_per_second < 0
    ):
        raise BenchmarkHarnessError(
            "minimum leaves per second must be finite and non-negative"
        )
    if not math.isfinite(settings.timeout_seconds) or settings.timeout_seconds <= 0:
        raise BenchmarkHarnessError("timeout seconds must be finite and positive")
    if not settings.device.strip():
        raise BenchmarkHarnessError("device must not be empty")
    if settings.compile_mode not in (
        None,
        "default",
        "reduce-overhead",
        "max-autotune",
    ):
        raise BenchmarkHarnessError("compile mode is invalid")
    if settings.metrics_root is not None:
        if not settings.metrics_root.exists():
            raise BenchmarkHarnessError(
                f"metrics root does not exist: {settings.metrics_root}"
            )
        if not _discover_metric_paths(settings.metrics_root):
            raise BenchmarkHarnessError(
                f"no metrics JSONL found under: {settings.metrics_root}"
            )


def _utc_now() -> str:
    return datetime.now(UTC).isoformat(timespec="milliseconds").replace("+00:00", "Z")


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while block := stream.read(1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


def _git_output(arguments: Sequence[str]) -> tuple[str | None, str | None]:
    command = ["git", *arguments]
    try:
        completed = subprocess.run(
            command,
            cwd=REPOSITORY_ROOT,
            check=False,
            capture_output=True,
            text=True,
            timeout=10.0,
        )
    except (OSError, subprocess.TimeoutExpired) as error:
        return None, f"{type(error).__name__}: {error}"
    if completed.returncode != 0:
        detail = completed.stderr.strip() or f"exit code {completed.returncode}"
        return None, detail
    return completed.stdout.strip(), None


def _git_metadata() -> dict[str, object]:
    revision, revision_error = _git_output(["rev-parse", "HEAD"])
    status, status_error = _git_output(["status", "--short", "--untracked-files=all"])
    errors = [error for error in (revision_error, status_error) if error is not None]
    return {
        "repository": str(REPOSITORY_ROOT),
        "revision": revision,
        "dirty": bool(status) if status is not None else None,
        "status": status.splitlines() if status else [],
        "errors": errors,
    }


def _nvidia_driver_versions() -> list[str]:
    try:
        completed = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=driver_version",
                "--format=csv,noheader,nounits",
            ],
            check=False,
            capture_output=True,
            text=True,
            timeout=10.0,
        )
    except (OSError, subprocess.TimeoutExpired):
        return []
    if completed.returncode != 0:
        return []
    return sorted(
        {line.strip() for line in completed.stdout.splitlines() if line.strip()}
    )


def _device_metadata(requested_device: str) -> dict[str, object]:
    result: dict[str, object] = {
        "requested": requested_device,
        "cuda_available": torch.cuda.is_available(),
        "visible_cuda_device_count": torch.cuda.device_count(),
    }
    try:
        device = torch.device(requested_device)
        result["type"] = device.type
        result["index"] = device.index
        if device.type != "cuda" or not torch.cuda.is_available():
            return result
        index = (
            device.index if device.index is not None else torch.cuda.current_device()
        )
        properties = torch.cuda.get_device_properties(index)
        result.update(
            {
                "resolved_index": index,
                "name": properties.name,
                "compute_capability": [properties.major, properties.minor],
                "total_memory_bytes": properties.total_memory,
                "multi_processor_count": properties.multi_processor_count,
            }
        )
    except (RuntimeError, ValueError, AssertionError) as error:
        result["metadata_error"] = f"{type(error).__name__}: {error}"
    return result


def collect_run_metadata(
    settings: BenchmarkSettings,
    invocation: Sequence[str],
) -> dict[str, object]:
    host = platform.uname()
    cudnn_version = (
        torch.backends.cudnn.version() if torch.backends.cudnn.is_available() else None
    )
    environment_names = (
        "CUDA_VISIBLE_DEVICES",
        "OMP_NUM_THREADS",
        "MKL_NUM_THREADS",
        "RAYON_NUM_THREADS",
        "PYTORCH_CUDA_ALLOC_CONF",
        "TORCHINDUCTOR_CACHE_DIR",
        "TORCH_LOGS",
    )
    output_directory = settings.output_directory
    return {
        "schema_version": SCHEMA_VERSION,
        "record_type": "run_metadata",
        "benchmark": BENCHMARK_NAME,
        "run_id": uuid.uuid4().hex,
        "started_at_utc": _utc_now(),
        "command": list(invocation),
        "config": {
            "path": str(settings.config),
            "sha256": _sha256_file(settings.config),
            "size_bytes": settings.config.stat().st_size,
        },
        "git": _git_metadata(),
        "host": {
            "hostname": socket.gethostname(),
            "system": host.system,
            "release": host.release,
            "version": host.version,
            "machine": host.machine,
            "processor": host.processor,
            "logical_cpu_count": os.cpu_count(),
        },
        "runtime": {
            "python_version": platform.python_version(),
            "python_executable": sys.executable,
            "torch_version": str(torch.__version__),
            "cuda_runtime_version": torch.version.cuda,
            "cudnn_version": cudnn_version,
            "nvidia_driver_versions": _nvidia_driver_versions(),
        },
        "device": _device_metadata(settings.device),
        "environment": {
            name: os.environ[name] for name in environment_names if name in os.environ
        },
        "scripts": {
            "harness": {
                "path": str(SCRIPT_PATH),
                "sha256": _sha256_file(SCRIPT_PATH),
            },
            "hardware_preflight": {
                "path": str(PREFLIGHT_SCRIPT),
                "sha256": _sha256_file(PREFLIGHT_SCRIPT),
            },
        },
        "parameters": {
            "rings": list(settings.rings),
            "batch_sizes": list(settings.batch_sizes),
            "repeats": settings.repeats,
            "warmup": settings.warmup,
            "iterations": settings.iterations,
            "device": settings.device,
            "minimum_leaves_per_second": settings.minimum_leaves_per_second,
            "timeout_seconds": settings.timeout_seconds,
            "metrics_root": (
                str(settings.metrics_root)
                if settings.metrics_root is not None
                else None
            ),
            "compile_dynamic": settings.compile_dynamic,
            "compile_mode": settings.compile_mode,
        },
        "artifacts": {
            "summary_json": str(output_directory / "summary.json"),
            "cases_jsonl": str(output_directory / "cases.jsonl"),
        },
    }


def build_preflight_command(
    settings: BenchmarkSettings,
    *,
    rings: int,
    batch_size: int,
) -> list[str]:
    command = [
        sys.executable,
        str(PREFLIGHT_SCRIPT),
        "--config",
        str(settings.config),
        "--device",
        settings.device,
        "--rings",
        str(rings),
        "--batch-size",
        str(batch_size),
        "--warmup",
        str(settings.warmup),
        "--iterations",
        str(settings.iterations),
        "--minimum-leaves-per-second",
        str(settings.minimum_leaves_per_second),
    ]
    if settings.compile_dynamic is not None:
        command.append(
            "--compile-dynamic" if settings.compile_dynamic else "--no-compile-dynamic"
        )
    if settings.compile_mode is not None:
        command.extend(("--compile-mode", settings.compile_mode))
    return command


def _execute_preflight(
    command: Sequence[str],
    *,
    cwd: Path,
    timeout_seconds: float,
) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        list(command),
        cwd=cwd,
        check=False,
        capture_output=True,
        text=True,
        timeout=timeout_seconds,
    )


def _output_text(output: str | bytes | None) -> str:
    if output is None:
        return ""
    if isinstance(output, bytes):
        return output.decode("utf-8", errors="replace")
    return output


def _preflight_payload(stdout: str) -> tuple[dict[str, object] | None, str | None]:
    if not stdout.strip():
        return None, "hardware preflight emitted no stdout"
    for line in reversed(stdout.splitlines()):
        try:
            payload = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(payload, dict):
            return payload, None
    return None, "hardware preflight did not emit a JSON object"


def _number(value: object) -> float | None:
    if isinstance(value, bool) or not isinstance(value, int | float):
        return None
    result = float(value)
    return result if math.isfinite(result) else None


def _payload_validation_error(
    payload: dict[str, object],
    *,
    rings: int,
    batch_size: int,
) -> str | None:
    if payload.get("benchmark") != "native-feature-model-inference-boundary":
        return "unexpected benchmark payload"
    if payload.get("rings") != rings or payload.get("batch_size") != batch_size:
        return "preflight payload dimensions do not match the requested case"
    required_numbers = (
        "leaf_evaluations_per_second",
        "mean_batch_ms",
        "p95_batch_ms",
        "peak_allocated_bytes",
    )
    missing = [name for name in required_numbers if _number(payload.get(name)) is None]
    if missing:
        return f"preflight payload is missing numeric fields: {', '.join(missing)}"
    if not isinstance(payload.get("passed"), bool):
        return "preflight payload is missing a boolean passed field"
    return None


def _measurement(payload: dict[str, object] | None) -> dict[str, object] | None:
    if payload is None:
        return None
    return {
        "throughput": {
            "leaf_evaluations_per_second": payload.get("leaf_evaluations_per_second"),
            "minimum_leaf_evaluations_per_second": payload.get(
                "minimum_leaf_evaluations_per_second"
            ),
        },
        "latency_ms": {
            "mean_batch": payload.get("mean_batch_ms"),
            "p95_batch": payload.get("p95_batch_ms"),
        },
        "memory_bytes": {
            "peak_allocated": payload.get("peak_allocated_bytes"),
        },
        "model_parameters": payload.get("model_parameters"),
        "feature_path": payload.get("feature_path"),
        "feature_path_counts": payload.get("feature_path_counts"),
    }


def run_case(
    *,
    settings: BenchmarkSettings,
    metadata: dict[str, object],
    rings: int,
    batch_size: int,
    repeat: int,
    executor: ProcessExecutor = _execute_preflight,
) -> dict[str, object]:
    command = build_preflight_command(
        settings,
        rings=rings,
        batch_size=batch_size,
    )
    started_at = _utc_now()
    started = time.perf_counter()
    stdout = ""
    stderr = ""
    return_code: int | None = None
    payload: dict[str, object] | None = None
    failure: dict[str, object] | None = None
    try:
        completed = executor(
            command,
            cwd=TRAINING_ROOT,
            timeout_seconds=settings.timeout_seconds,
        )
        return_code = completed.returncode
        stdout = completed.stdout
        stderr = completed.stderr
        payload, parse_error = _preflight_payload(stdout)
        if parse_error is not None:
            failure = {"kind": "invalid_output", "message": parse_error}
        elif payload is not None:
            validation_error = _payload_validation_error(
                payload,
                rings=rings,
                batch_size=batch_size,
            )
            if validation_error is not None:
                failure = {
                    "kind": "invalid_output",
                    "message": validation_error,
                }
            elif return_code != 0:
                failure = {
                    "kind": (
                        "performance_gate"
                        if payload.get("passed") is False
                        else "child_exit"
                    ),
                    "message": f"hardware preflight exited {return_code}",
                }
            elif payload.get("passed") is not True:
                failure = {
                    "kind": "performance_gate",
                    "message": "hardware preflight did not pass its performance gate",
                }
    except subprocess.TimeoutExpired as error:
        stdout = _output_text(error.stdout)
        stderr = _output_text(error.stderr)
        failure = {
            "kind": "timeout",
            "message": (
                f"hardware preflight exceeded {settings.timeout_seconds:g} seconds"
            ),
        }
    except OSError as error:
        failure = {
            "kind": "spawn_error",
            "message": f"{type(error).__name__}: {error}",
        }

    context_names = ("config", "git", "host", "runtime", "device", "environment")
    return {
        "schema_version": SCHEMA_VERSION,
        "record_type": "inference_case",
        "benchmark": BENCHMARK_NAME,
        "run_id": metadata["run_id"],
        "case_id": f"rings-{rings}-batch-{batch_size}-repeat-{repeat}",
        "started_at_utc": started_at,
        "wall_seconds": time.perf_counter() - started,
        "case": {
            "rings": rings,
            "batch_size": batch_size,
            "repeat": repeat,
            "warmup": settings.warmup,
            "iterations": settings.iterations,
        },
        "command": command,
        "context": {name: metadata[name] for name in context_names},
        "status": "passed" if failure is None else "failed",
        "process": {
            "return_code": return_code,
            "stdout": stdout,
            "stderr": stderr,
        },
        "measurement": _measurement(payload),
        "raw_preflight": payload,
        "failure": failure,
    }


def _stats(values: Sequence[float]) -> dict[str, object] | None:
    if not values:
        return None
    ordered = sorted(values)
    p95_index = max(0, math.ceil(0.95 * len(ordered)) - 1)
    return {
        "count": len(ordered),
        "minimum": ordered[0],
        "maximum": ordered[-1],
        "mean": statistics.fmean(ordered),
        "median": statistics.median(ordered),
        "p95": ordered[p95_index],
        "sample_standard_deviation": (
            statistics.stdev(ordered) if len(ordered) > 1 else 0.0
        ),
    }


def _measurement_number(
    record: dict[str, object],
    section: str,
    name: str,
) -> float | None:
    measurement = record.get("measurement")
    if not isinstance(measurement, dict):
        return None
    values = measurement.get(section)
    if not isinstance(values, dict):
        return None
    return _number(values.get(name))


def summarize_cases(records: Sequence[dict[str, object]]) -> list[dict[str, object]]:
    groups: dict[tuple[int, int], list[dict[str, object]]] = {}
    for record in records:
        case = record["case"]
        assert isinstance(case, dict)
        key = (int(case["rings"]), int(case["batch_size"]))
        groups.setdefault(key, []).append(record)

    summaries = []
    for (rings, batch_size), group in sorted(groups.items()):
        passed = [record for record in group if record["status"] == "passed"]
        throughput = [
            value
            for record in group
            if (
                value := _measurement_number(
                    record,
                    "throughput",
                    "leaf_evaluations_per_second",
                )
            )
            is not None
        ]
        mean_latency = [
            value
            for record in group
            if (value := _measurement_number(record, "latency_ms", "mean_batch"))
            is not None
        ]
        p95_latency = [
            value
            for record in group
            if (value := _measurement_number(record, "latency_ms", "p95_batch"))
            is not None
        ]
        memory = [
            value
            for record in group
            if (
                value := _measurement_number(
                    record,
                    "memory_bytes",
                    "peak_allocated",
                )
            )
            is not None
        ]
        summaries.append(
            {
                "rings": rings,
                "batch_size": batch_size,
                "requested_repeats": len(group),
                "passed_repeats": len(passed),
                "failed_repeats": len(group) - len(passed),
                "passed": len(passed) == len(group),
                "leaf_evaluations_per_second": _stats(throughput),
                "mean_batch_latency_ms": _stats(mean_latency),
                "p95_batch_latency_ms": _stats(p95_latency),
                "peak_allocated_bytes": _stats(memory),
            }
        )
    return summaries


def _metric_number(record: dict[str, object], *names: str) -> float | None:
    for name in names:
        value = _number(record.get(name))
        if value is not None:
            return value
    return None


def _timestamp_ns(record: dict[str, object]) -> int | None:
    value = record.get("timestamp_ns")
    if isinstance(value, bool) or not isinstance(value, int):
        return None
    return value


def _is_replay_wait(record: dict[str, object]) -> bool:
    for name in ("phase", "event", "state", "status"):
        value = record.get(name)
        if not isinstance(value, str):
            continue
        normalized = value.lower().replace("-", "_").replace(" ", "_")
        if "replay_wait" in normalized:
            return True
    return False


_ACTOR_SERIES_NAMES = (
    "games",
    "samples",
    "batches",
    "evaluator_calls",
    "evaluator_rows",
    "evaluator_seconds",
    "evaluator_rows_per_second",
    "completed_decisions",
    "attempted_decisions",
    "full_decisions",
    "fast_decisions",
    "game_lengths",
    "policy_entropy_count",
    "policy_entropy_sum",
    "policy_entropy_mean",
    "interrupted_cohorts",
    "dropped_games",
    "dropped_decisions",
    "model_refresh_latency_seconds",
    "replay_append_calls",
    "replay_append_bytes",
    "replay_append_seconds",
    "peak_cuda_memory_bytes",
    "peak_cuda_memory_reserved_bytes",
)


def _actor_series() -> dict[str, list[float]]:
    return {name: [] for name in _ACTOR_SERIES_NAMES}


def _append_actor_record(
    series: dict[str, list[float]],
    record: dict[str, object],
    *,
    games_per_second: float | None,
    samples_per_second: float | None,
    batch_seconds: float | None,
) -> None:
    for name, value in (
        ("games", games_per_second),
        ("samples", samples_per_second),
        ("batches", batch_seconds),
    ):
        if value is not None:
            series[name].append(value)

    aliases = {
        "evaluator_calls": ("evaluator_calls",),
        "evaluator_rows": ("evaluator_rows",),
        "evaluator_rows_per_second": ("evaluator_rows_per_second",),
        "completed_decisions": ("completed_decisions",),
        "attempted_decisions": ("attempted_decisions",),
        "full_decisions": ("full_decisions", "full_search_decisions"),
        "fast_decisions": ("fast_decisions", "fast_search_decisions"),
        "policy_entropy_count": (
            "policy_entropy_count",
            "policy_target_count",
        ),
        "policy_entropy_sum": ("policy_entropy_sum",),
        "policy_entropy_mean": (
            "policy_entropy_mean",
            "mean_policy_entropy",
        ),
        "interrupted_cohorts": ("interrupted_cohorts",),
        "dropped_games": (
            "dropped_games",
            "interrupted_cohort_dropped_games",
        ),
        "dropped_decisions": (
            "dropped_decisions",
            "interrupted_cohort_dropped_decisions",
        ),
        "model_refresh_latency_seconds": (
            "model_refresh_latency_seconds",
            "model_refresh_seconds",
        ),
        "replay_append_calls": ("replay_append_calls",),
        "replay_append_bytes": ("replay_append_bytes",),
        "replay_append_seconds": (
            "replay_append_seconds",
            "replay_append_time_seconds",
        ),
        "peak_cuda_memory_bytes": (
            "peak_cuda_memory_bytes",
            "peak_cuda_memory_allocated_bytes",
        ),
        "peak_cuda_memory_reserved_bytes": (
            "peak_cuda_memory_reserved_bytes",
            "peak_cuda_reserved_memory_bytes",
        ),
    }
    for destination, names in aliases.items():
        value = _metric_number(record, *names)
        if value is not None:
            series[destination].append(value)
    if (
        _metric_number(record, "evaluator_rows") is not None
        and batch_seconds is not None
    ):
        series["evaluator_seconds"].append(batch_seconds)
    if _metric_number(record, "attempted_decisions") is None:
        full = _metric_number(record, "full_decisions", "full_search_decisions")
        fast = _metric_number(record, "fast_decisions", "fast_search_decisions")
        if full is not None and fast is not None:
            series["attempted_decisions"].append(full + fast)
    if _metric_number(record, "policy_entropy_sum") is None:
        entropy_count = _metric_number(
            record,
            "policy_entropy_count",
            "policy_target_count",
        )
        entropy_mean = _metric_number(
            record,
            "policy_entropy_mean",
            "mean_policy_entropy",
        )
        if entropy_count is not None and entropy_mean is not None:
            series["policy_entropy_sum"].append(entropy_count * entropy_mean)

    raw_lengths = record.get("game_lengths")
    if isinstance(raw_lengths, list):
        for raw_length in raw_lengths:
            length = _number(raw_length)
            if length is not None and length >= 0 and length.is_integer():
                series["game_lengths"].append(length)
        return
    distribution = record.get("game_length_distribution")
    if not isinstance(distribution, dict):
        return
    for raw_length, raw_count in distribution.items():
        try:
            length = float(raw_length)
        except (TypeError, ValueError):
            continue
        count = _number(raw_count)
        if (
            math.isfinite(length)
            and length >= 0
            and length.is_integer()
            and count is not None
            and count >= 0
            and count.is_integer()
        ):
            series["game_lengths"].extend([length] * int(count))


def _counter_summary(values: Sequence[float]) -> dict[str, object] | None:
    if not values:
        return None
    total: int | float = sum(values)
    if all(value.is_integer() for value in values):
        total = int(total)
    return {
        "total": total,
        "per_batch": _stats(values),
    }


def _actor_series_summary(series: dict[str, list[float]]) -> dict[str, object]:
    game_length_stats = _stats(series["game_lengths"])
    game_length_distribution: dict[str, int] = {}
    for length in sorted(int(value) for value in series["game_lengths"]):
        key = str(length)
        game_length_distribution[key] = game_length_distribution.get(key, 0) + 1

    entropy_count = sum(series["policy_entropy_count"])
    entropy_sum = sum(series["policy_entropy_sum"])
    replay_bytes = sum(series["replay_append_bytes"])
    replay_seconds = sum(series["replay_append_seconds"])
    evaluator_rows = sum(series["evaluator_rows"])
    evaluator_seconds = sum(series["evaluator_seconds"])
    return {
        "games_per_second": _stats(series["games"]),
        "samples_per_second": _stats(series["samples"]),
        "batch_seconds": _stats(series["batches"]),
        "evaluator": {
            "calls": _counter_summary(series["evaluator_calls"]),
            "rows": _counter_summary(series["evaluator_rows"]),
            "rows_per_second": _stats(series["evaluator_rows_per_second"]),
            "aggregate_rows_per_second": (
                evaluator_rows / evaluator_seconds if evaluator_seconds else None
            ),
        },
        "decisions": {
            "completed": _counter_summary(series["completed_decisions"]),
            "attempted": _counter_summary(series["attempted_decisions"]),
            "full": _counter_summary(series["full_decisions"]),
            "fast": _counter_summary(series["fast_decisions"]),
        },
        "game_length": {
            "mean": (
                game_length_stats["mean"] if game_length_stats is not None else None
            ),
            "distribution": game_length_distribution,
            "statistics": game_length_stats,
        },
        "policy_entropy": {
            "unit": "nats",
            "target_count": (
                int(entropy_count) if series["policy_entropy_count"] else None
            ),
            "sum": entropy_sum if series["policy_entropy_sum"] else None,
            "mean": entropy_sum / entropy_count if entropy_count else None,
            "per_batch_mean": _stats(series["policy_entropy_mean"]),
        },
        "interrupted_cohorts": {
            "cohorts": _counter_summary(series["interrupted_cohorts"]),
            "dropped_games": _counter_summary(series["dropped_games"]),
            "dropped_decisions": _counter_summary(series["dropped_decisions"]),
        },
        "model_refresh_latency_seconds": _stats(
            series["model_refresh_latency_seconds"]
        ),
        "replay_append": {
            "calls": _counter_summary(series["replay_append_calls"]),
            "bytes": _counter_summary(series["replay_append_bytes"]),
            "seconds": _counter_summary(series["replay_append_seconds"]),
            "bytes_per_second": (
                replay_bytes / replay_seconds if replay_seconds else None
            ),
        },
        "peak_cuda_memory_bytes": _stats(series["peak_cuda_memory_bytes"]),
        "peak_cuda_memory_reserved_bytes": _stats(
            series["peak_cuda_memory_reserved_bytes"]
        ),
    }


def summarize_orchestration_metrics(root: Path) -> dict[str, object]:
    paths = _discover_metric_paths(root)
    if not paths:
        raise BenchmarkHarnessError(f"no metrics JSONL found under: {root}")

    learner_examples: list[float] = []
    learner_batches: list[float] = []
    learner_device_examples: list[float] = []
    learner_device_batches: list[float] = []
    learner_data_waits: list[float] = []
    learner_h2d: list[float] = []
    learner_window_setup: list[float] = []
    actor_metrics = _actor_series()
    actor_by_worker: dict[str, dict[str, list[float]]] = {}
    replay_wait_durations: list[float] = []
    active_replay_waits: dict[str, int] = {}
    replay_wait_events = 0
    learner_records = 0
    actor_records = 0
    record_count = 0
    parse_failures: list[dict[str, object]] = []
    file_summaries: list[dict[str, object]] = []

    for path in paths:
        digest = hashlib.sha256()
        file_bytes = 0
        file_records = 0
        with path.open("rb") as stream:
            for line_number, raw_line in enumerate(stream, start=1):
                digest.update(raw_line)
                file_bytes += len(raw_line)
                if not raw_line.strip():
                    continue
                try:
                    decoded = raw_line.decode("utf-8")
                    loaded = json.loads(decoded)
                except (UnicodeDecodeError, json.JSONDecodeError) as error:
                    parse_failures.append(
                        {
                            "path": str(path),
                            "line": line_number,
                            "error": f"{type(error).__name__}: {error}",
                        }
                    )
                    continue
                if not isinstance(loaded, dict):
                    parse_failures.append(
                        {
                            "path": str(path),
                            "line": line_number,
                            "error": "JSONL record is not an object",
                        }
                    )
                    continue
                record: dict[str, object] = loaded
                record_count += 1
                file_records += 1

                examples = _metric_number(record, "examples_per_second")
                step_seconds = _metric_number(
                    record,
                    "step_seconds",
                    "batch_time_seconds",
                )
                if examples is not None or step_seconds is not None:
                    learner_records += 1
                    if examples is not None:
                        learner_examples.append(examples)
                    if step_seconds is not None:
                        learner_batches.append(step_seconds)
                    for destination, names in (
                        (
                            learner_device_examples,
                            ("device_examples_per_second",),
                        ),
                        (learner_device_batches, ("device_step_seconds",)),
                        (learner_data_waits, ("data_wait_seconds",)),
                        (learner_h2d, ("h2d_seconds",)),
                        (learner_window_setup, ("window_setup_seconds",)),
                    ):
                        value = _metric_number(record, *names)
                        if value is not None:
                            destination.append(value)

                games = _metric_number(record, "games_per_second")
                samples = _metric_number(record, "samples_per_second")
                is_actor = (
                    games is not None
                    or samples is not None
                    or "games" in record
                    or "search_simulations_per_second" in record
                )
                if is_actor:
                    actor_records += 1
                    batch_seconds = _metric_number(
                        record,
                        "batch_time_seconds",
                        "elapsed_seconds",
                    )
                    worker = str(record.get("worker") or path.stem)
                    worker_values = actor_by_worker.setdefault(
                        worker,
                        _actor_series(),
                    )
                    _append_actor_record(
                        actor_metrics,
                        record,
                        games_per_second=games,
                        samples_per_second=samples,
                        batch_seconds=batch_seconds,
                    )
                    _append_actor_record(
                        worker_values,
                        record,
                        games_per_second=games,
                        samples_per_second=samples,
                        batch_seconds=batch_seconds,
                    )

                wait_marker = _is_replay_wait(record)
                explicit_wait = _metric_number(
                    record,
                    "replay_wait_seconds",
                    "replay_wait_duration_seconds",
                )
                if explicit_wait is None and wait_marker:
                    explicit_wait = _metric_number(
                        record,
                        "wait_seconds",
                        "duration_seconds",
                        "elapsed_seconds",
                    )
                worker_key = f"{path}:{record.get('worker', path.stem)}"
                timestamp_ns = _timestamp_ns(record)
                if explicit_wait is not None:
                    replay_wait_events += 1
                    replay_wait_durations.append(explicit_wait)
                    active_replay_waits.pop(worker_key, None)
                elif wait_marker:
                    if worker_key not in active_replay_waits:
                        replay_wait_events += 1
                        if timestamp_ns is not None:
                            active_replay_waits[worker_key] = timestamp_ns
                elif worker_key in active_replay_waits and timestamp_ns is not None:
                    started_ns = active_replay_waits.pop(worker_key)
                    if timestamp_ns >= started_ns:
                        replay_wait_durations.append(
                            (timestamp_ns - started_ns) / 1_000_000_000.0
                        )

        file_summaries.append(
            {
                "path": str(path),
                "sha256": digest.hexdigest(),
                "size_bytes": file_bytes,
                "records": file_records,
            }
        )

    workers = {
        worker: _actor_series_summary(values)
        for worker, values in sorted(actor_by_worker.items())
    }
    actors = _actor_series_summary(actor_metrics)
    actors.update(
        {
            "records": actor_records,
            "by_worker": workers,
        }
    )
    return {
        "status": "complete" if not parse_failures else "incomplete",
        "summarized_at_utc": _utc_now(),
        "root": str(root.resolve()),
        "files": file_summaries,
        "records": record_count,
        "parse_failure_count": len(parse_failures),
        "parse_failures": parse_failures,
        "learner": {
            "records": learner_records,
            "examples_per_second": _stats(learner_examples),
            "batch_seconds": _stats(learner_batches),
            "device_examples_per_second": _stats(learner_device_examples),
            "device_batch_seconds": _stats(learner_device_batches),
            "data_wait_seconds": _stats(learner_data_waits),
            "h2d_seconds": _stats(learner_h2d),
            "window_setup_seconds": _stats(learner_window_setup),
        },
        "actors": actors,
        "replay_waits": {
            "availability": (
                "observed" if replay_wait_events else "not_recorded_in_jsonl"
            ),
            "events": replay_wait_events,
            "completed_intervals": len(replay_wait_durations),
            "open_intervals": len(active_replay_waits),
            "seconds": _stats(replay_wait_durations),
        },
    }


def _write_jsonl(stream, payload: dict[str, object]) -> None:
    stream.write(json.dumps(payload, sort_keys=True, separators=(",", ":")) + "\n")
    stream.flush()


def run_harness(
    settings: BenchmarkSettings,
    metadata: dict[str, object],
    *,
    executor: ProcessExecutor = _execute_preflight,
) -> tuple[int, dict[str, object]]:
    validate_settings(settings)
    settings.output_directory.mkdir(parents=True, exist_ok=False)
    cases_path = settings.output_directory / "cases.jsonl"
    summary_path = settings.output_directory / "summary.json"
    records: list[dict[str, object]] = []

    with cases_path.open("x", encoding="utf-8") as stream:
        _write_jsonl(stream, metadata)
        for rings in settings.rings:
            for batch_size in settings.batch_sizes:
                for repeat in range(1, settings.repeats + 1):
                    record = run_case(
                        settings=settings,
                        metadata=metadata,
                        rings=rings,
                        batch_size=batch_size,
                        repeat=repeat,
                        executor=executor,
                    )
                    records.append(record)
                    _write_jsonl(stream, record)

    metrics: dict[str, object] | None = None
    if settings.metrics_root is not None:
        try:
            metrics = summarize_orchestration_metrics(settings.metrics_root)
        except OSError as error:
            metrics = {
                "status": "incomplete",
                "root": str(settings.metrics_root),
                "parse_failure_count": 1,
                "parse_failures": [
                    {
                        "error": f"{type(error).__name__}: {error}",
                    }
                ],
            }

    failures = [
        {
            "case_id": record["case_id"],
            "failure": record["failure"],
        }
        for record in records
        if record["status"] == "failed"
    ]
    metrics_failed = metrics is not None and (
        metrics.get("status") != "complete" or metrics.get("parse_failure_count") != 0
    )
    if failures:
        exit_code = EXIT_BENCHMARK_FAILURE
        status = "benchmark_failed"
    elif metrics_failed:
        exit_code = EXIT_METRICS_FAILURE
        status = "metrics_incomplete"
    else:
        exit_code = EXIT_OK
        status = "passed"

    summary = {
        "schema_version": SCHEMA_VERSION,
        "benchmark": BENCHMARK_NAME,
        "status": status,
        "exit_code": exit_code,
        "completed_at_utc": _utc_now(),
        "run": metadata,
        "case_count": len(records),
        "passed_case_count": len(records) - len(failures),
        "failed_case_count": len(failures),
        "aggregates": summarize_cases(records),
        "failures": failures,
        "cases": records,
        "orchestration_metrics": metrics,
    }
    with summary_path.open("x", encoding="utf-8") as stream:
        json.dump(summary, stream, indent=2, sort_keys=True)
        stream.write("\n")
    return exit_code, summary


def main(argv: list[str] | None = None) -> int:
    raw_arguments = list(sys.argv[1:] if argv is None else argv)
    settings = _settings(_parser().parse_args(raw_arguments))
    invocation = [sys.executable, str(SCRIPT_PATH), *raw_arguments]
    try:
        validate_settings(settings)
        metadata = collect_run_metadata(settings, invocation)
        exit_code, summary = run_harness(settings, metadata)
    except (BenchmarkHarnessError, OSError) as error:
        print(
            json.dumps(
                {
                    "schema_version": SCHEMA_VERSION,
                    "benchmark": BENCHMARK_NAME,
                    "status": "harness_error",
                    "exit_code": EXIT_HARNESS_ERROR,
                    "error": f"{type(error).__name__}: {error}",
                },
                sort_keys=True,
            ),
            file=sys.stderr,
        )
        return EXIT_HARNESS_ERROR
    except KeyboardInterrupt:
        print(
            json.dumps(
                {
                    "schema_version": SCHEMA_VERSION,
                    "benchmark": BENCHMARK_NAME,
                    "status": "interrupted",
                    "exit_code": EXIT_INTERRUPTED,
                },
                sort_keys=True,
            ),
            file=sys.stderr,
        )
        return EXIT_INTERRUPTED

    print(
        json.dumps(
            {
                "schema_version": SCHEMA_VERSION,
                "benchmark": BENCHMARK_NAME,
                "status": summary["status"],
                "exit_code": exit_code,
                "case_count": summary["case_count"],
                "failed_case_count": summary["failed_case_count"],
                "summary_json": str(settings.output_directory / "summary.json"),
                "cases_jsonl": str(settings.output_directory / "cases.jsonl"),
            },
            sort_keys=True,
        )
    )
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
