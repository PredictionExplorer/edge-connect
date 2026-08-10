from __future__ import annotations

import json
import subprocess
import time
from pathlib import Path
from types import SimpleNamespace
from typing import Any, cast

import pytest

import scripts.benchmark_arena_occupancy as benchmark_module
from scripts.benchmark_arena_occupancy import (
    CONTROL_ARM,
    TREATMENT_ARM,
    aggregate_benchmark_runs,
    query_gpu_status,
    run_arena_occupancy_benchmark,
    summarize_gpu_samples,
)


def _samples(
    utilization: float,
    *,
    timestamp_ns: int | None = None,
) -> list[dict[str, object]]:
    observed_ns = time.time_ns() if timestamp_ns is None else timestamp_ns
    return [
        {
            "timestamp_ns": observed_ns,
            "index": 7,
            "utilization_gpu_percent": utilization,
            "memory_used_mib": 1000.0,
            "power_draw_watts": 300.0,
            "temperature_c": 40.0,
        },
        {
            "timestamp_ns": observed_ns + 1,
            "index": 7,
            "utilization_gpu_percent": utilization + 10.0,
            "memory_used_mib": 1100.0,
            "power_draw_watts": 320.0,
            "temperature_c": 42.0,
        },
    ]


def _run(arm: str, utilization: float, rows_per_second: float) -> dict[str, object]:
    return {
        "phase": "measurement",
        "arm": arm,
        "repeat": 0,
        "pair_start": 50,
        "total_pair_count": 50,
        "gpu_samples": _samples(utilization),
        "gpu_summary": summarize_gpu_samples(_samples(utilization)),
        "evaluation_metrics": {
            "wall_seconds": 10.0,
            "evaluator_rows_per_second": rows_per_second,
            "inference_queue_wait_seconds": 1.0,
            "peak_cuda_allocated_bytes": 100.0,
        },
    }


def test_gpu_and_run_aggregation_reports_utilization_uplift() -> None:
    summary = summarize_gpu_samples(_samples(50.0))
    utilization = summary["utilization_gpu_percent"]
    assert isinstance(utilization, dict)
    assert utilization["mean"] == 55.0
    assert utilization["p50"] == 55.0

    aggregate = aggregate_benchmark_runs(
        [
            _run(CONTROL_ARM, 50.0, 1000.0),
            _run(TREATMENT_ARM, 70.0, 1250.0),
        ]
    )
    comparison = aggregate["comparison"]
    assert isinstance(comparison, dict)
    assert comparison["utilization_mean_delta_points"] == 20.0
    assert comparison["evaluator_rows_per_second_ratio"] == 1.25
    assert comparison["deployment_eligible"] is False


def test_gpu_query_normalizes_subprocess_timeout(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def timeout(*_args: object, **_kwargs: object) -> None:
        raise subprocess.TimeoutExpired("nvidia-smi", 10)

    monkeypatch.setattr(benchmark_module.subprocess, "run", timeout)
    with pytest.raises(RuntimeError, match="nvidia-smi query failed"):
        query_gpu_status(7)


def test_wave_preserves_workload_error_when_sampler_teardown_also_fails(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class FailingRunner:
        def run(self, **_kwargs: object) -> dict[str, object]:
            raise ValueError("arena failed")

    class FailingSampler:
        def __init__(self, _gpu_index: int, _interval: float) -> None:
            pass

        def start(self) -> None:
            pass

        def stop(self) -> list[dict[str, object]]:
            raise RuntimeError("sampler failed")

    monkeypatch.setattr(benchmark_module, "GPUStatusSampler", FailingSampler)
    monkeypatch.setattr(
        benchmark_module,
        "reset_peak_memory_stats",
        lambda _device: None,
    )

    with pytest.raises(ValueError, match="arena failed") as raised:
        benchmark_module._run_wave(
            runner=cast(Any, FailingRunner()),
            device="cuda:7",
            physical_gpu_index=7,
            sample_interval_seconds=1.0,
            phase="measurement",
            arm=CONTROL_ARM,
            repeat=0,
            pair_start=50,
            pair_count=25,
        )
    assert any("sampler also failed" in note for note in raised.value.__notes__)


def _verified_plan(tmp_path: Path) -> SimpleNamespace:
    source = tmp_path / "source"
    source.mkdir()
    candidate_manifest = SimpleNamespace(model_identity="candidate")
    champion_manifest = SimpleNamespace(model_identity="champion")
    candidate = SimpleNamespace(verify=lambda: candidate_manifest)
    champion = SimpleNamespace(
        manifest=SimpleNamespace(verify=lambda: champion_manifest)
    )
    selection = SimpleNamespace(
        candidates=(candidate,),
        source_champion=champion,
    )
    payload = {
        "source_run_root": str(source),
        "plan_digest": "b" * 64,
        "device": "cuda:7",
        "physical_gpu_index": 7,
        "execution_lock": str(tmp_path / "host-execution.lock"),
        "sample_interval_seconds": 1.0,
        "warmup_arm_order": [CONTROL_ARM, TREATMENT_ARM],
        "arms": [
            {
                "name": CONTROL_ARM,
                "total_pair_count": 50,
                "chunk_pair_count": 25,
                "chunks": 2,
            },
            {
                "name": TREATMENT_ARM,
                "total_pair_count": 50,
                "chunk_pair_count": 50,
                "chunks": 1,
            },
        ],
        "schedule": [
            {
                "repeat": 0,
                "pair_start": 50,
                "arm_order": [CONTROL_ARM, TREATMENT_ARM],
            }
        ],
        "deployment_policy": {"treatment_deployable": False},
    }
    return SimpleNamespace(
        path=tmp_path / "plan.json",
        payload=payload,
        experiment=SimpleNamespace(),
        selection=selection,
    )


def _patch_runtime(
    monkeypatch: pytest.MonkeyPatch,
    verified: SimpleNamespace,
    *,
    fail_after: int | None = None,
    lease_observations: list[bool] | None = None,
) -> list[int]:
    seen_pair_counts: list[int] = []
    latest = {"pair_count": 25}
    monkeypatch.setattr(
        benchmark_module,
        "verify_arena_occupancy_plan",
        lambda _path, **_kwargs: verified,
    )
    monkeypatch.setattr(
        benchmark_module, "load_star_native", lambda **_kwargs: object()
    )
    monkeypatch.setattr(
        benchmark_module,
        "load_manifest_evaluator",
        lambda *_args, **_kwargs: SimpleNamespace(),
    )
    monkeypatch.setattr(
        benchmark_module,
        "selection_arena_config",
        lambda *_args, **_kwargs: SimpleNamespace(),
    )
    monkeypatch.setattr(
        benchmark_module,
        "assert_physical_gpu_idle",
        lambda _index: None,
    )
    monkeypatch.setattr(
        benchmark_module,
        "verify_cuda_device_identity",
        lambda _device, index: {"physical_index": index},
    )
    monkeypatch.setattr(
        benchmark_module,
        "runtime_metadata",
        lambda **_kwargs: {"runtime": "test"},
    )
    monkeypatch.setattr(
        benchmark_module, "reset_peak_memory_stats", lambda _device: None
    )
    monkeypatch.setattr(
        benchmark_module,
        "peak_memory_stats",
        lambda _device: (100, 200),
    )
    monkeypatch.setattr(benchmark_module, "synchronize_device", lambda _device: None)
    monkeypatch.setattr(benchmark_module, "empty_device_cache", lambda _device: None)

    class FakeRunner:
        def __init__(self, **_kwargs: object) -> None:
            pass

        def run(
            self,
            *,
            pair_starts: dict[int, int],
            pair_counts: dict[int, int],
        ) -> dict[str, object]:
            pair_start = pair_starts[10]
            pair_count = pair_counts[10]
            if lease_observations is not None and not lease_observations:
                source = Path(str(verified.payload["source_run_root"]))
                contender = benchmark_module.CoordinatorLock(
                    source / "coordinator.lock"
                )
                try:
                    contender.acquire()
                except RuntimeError:
                    lease_observations.append(True)
                else:
                    lease_observations.append(False)
                    contender.release()
                try:
                    with benchmark_module.exclusive_execution_lock(
                        Path(str(verified.payload["execution_lock"]))
                    ):
                        lease_observations.append(False)
                except RuntimeError:
                    lease_observations.append(True)
            assert len(pair_counts) == 1
            seen_pair_counts.append(pair_count)
            latest["pair_count"] = pair_count
            if fail_after is not None and len(seen_pair_counts) > fail_after:
                raise RuntimeError("planned benchmark failure")
            return {
                "evaluation_metrics": {
                    "wall_seconds": float(pair_count),
                    "total_evaluator_rows": float(pair_count * 100),
                    "evaluator_rows_per_second": float(pair_count * 10),
                    "serialized_inference_seconds": 1.0,
                    "inference_queue_wait_seconds": 0.5,
                },
                "search": {
                    "search_workers": 2,
                    "inference_workers": 1,
                    "pair_chunk_size": None,
                    "effective_pair_chunking": "full_requested_ring_batch",
                },
                "pairs": [
                    {"ring": 10, "pair": pair_start + index}
                    for index in range(pair_count)
                ],
            }

    class FakeSampler:
        def __init__(self, _gpu_index: int, _interval: float) -> None:
            self.timestamp_ns = 0

        def start(self) -> None:
            self.timestamp_ns = time.time_ns()

        def stop(self) -> list[dict[str, object]]:
            utilization = 50.0 if latest["pair_count"] == 25 else 70.0
            return _samples(utilization, timestamp_ns=self.timestamp_ns)

    monkeypatch.setattr(benchmark_module, "ArenaRunner", FakeRunner)
    monkeypatch.setattr(benchmark_module, "GPUStatusSampler", FakeSampler)
    return seen_pair_counts


def test_benchmark_executes_warmups_then_interleaved_measurements(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    verified = _verified_plan(tmp_path)
    lease_observations: list[bool] = []
    seen = _patch_runtime(
        monkeypatch,
        verified,
        lease_observations=lease_observations,
    )
    output = tmp_path / "results"

    report = run_arena_occupancy_benchmark(
        plan_path=tmp_path / "plan.json",
        output_dir=output,
    )

    assert seen == [25, 25, 50, 25, 25, 50]
    assert lease_observations == [True, True]
    assert not (tmp_path / "source" / "coordinator.lock").exists()
    assert report["status"] == "complete"
    assert report["deployment_recommendation"] == "benchmark_only_no_profile_change"
    assert (output / "occupancy-report.json").is_file()
    assert not (output / "benchmark-failure.json").exists()
    persisted = json.loads((output / "occupancy-report.json").read_text())
    assert persisted["aggregate"]["comparison"]["deployment_eligible"] is False


def test_benchmark_failure_keeps_partial_evidence_without_final_report(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    verified = _verified_plan(tmp_path)
    _patch_runtime(monkeypatch, verified, fail_after=2)
    output = tmp_path / "failed-results"

    with pytest.raises(RuntimeError, match="planned benchmark failure"):
        run_arena_occupancy_benchmark(
            plan_path=tmp_path / "plan.json",
            output_dir=output,
        )

    assert (output / "benchmark-progress.json").is_file()
    assert (output / "benchmark-failure.json").is_file()
    assert not (output / "occupancy-report.json").exists()
    progress = json.loads((output / "benchmark-progress.json").read_text())
    assert progress["status"] == "error"


def test_benchmark_refuses_source_that_became_active(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    verified = _verified_plan(tmp_path)
    source = Path(verified.payload["source_run_root"])
    (source / "coordinator.lock").write_text("active\n", encoding="utf-8")
    monkeypatch.setattr(
        benchmark_module,
        "verify_arena_occupancy_plan",
        lambda _path, **_kwargs: verified,
    )
    output = tmp_path / "results"

    with pytest.raises(ValueError, match="coordinator lock"):
        run_arena_occupancy_benchmark(
            plan_path=tmp_path / "plan.json",
            output_dir=output,
        )
    assert not output.exists()
