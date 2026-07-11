from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

from scripts import h100_system_benchmark as benchmark


def _settings(
    tmp_path: Path,
    *,
    repeats: int = 2,
    metrics_root: Path | None = None,
) -> benchmark.BenchmarkSettings:
    config = tmp_path / "profile.yaml"
    config.write_text("schema_version: 2\n", encoding="utf-8")
    return benchmark.BenchmarkSettings(
        config=config,
        output_directory=tmp_path / "benchmark-output",
        rings=(6,),
        batch_sizes=(32,),
        repeats=repeats,
        warmup=2,
        iterations=4,
        device="cuda:0",
        minimum_leaves_per_second=5_000.0,
        timeout_seconds=30.0,
        metrics_root=metrics_root,
    )


def _metadata() -> dict[str, object]:
    return {
        "schema_version": 1,
        "record_type": "run_metadata",
        "benchmark": benchmark.BENCHMARK_NAME,
        "run_id": "test-run",
        "config": {"path": "profile.yaml", "sha256": "abc"},
        "git": {"revision": "deadbeef", "dirty": False},
        "host": {"hostname": "h100-host"},
        "runtime": {"torch_version": "test", "cuda_runtime_version": "test"},
        "device": {"requested": "cuda:0", "name": "NVIDIA H100"},
        "environment": {},
    }


def _payload(*, passed: bool = True, throughput: float = 8_000.0) -> dict[str, object]:
    return {
        "schema_version": 1,
        "benchmark": "native-feature-model-inference-boundary",
        "device": "NVIDIA H100",
        "rings": 6,
        "batch_size": 32,
        "iterations": 4,
        "mean_batch_ms": 4.0,
        "p95_batch_ms": 5.0,
        "leaf_evaluations_per_second": throughput,
        "minimum_leaf_evaluations_per_second": 5_000.0,
        "peak_allocated_bytes": 123_456,
        "model_parameters": 10,
        "feature_path": "rust",
        "feature_path_counts": {"rust": 4},
        "passed": passed,
    }


def test_preflight_command_is_explicit_and_bounded(tmp_path: Path) -> None:
    settings = _settings(tmp_path)

    command = benchmark.build_preflight_command(
        settings,
        rings=6,
        batch_size=32,
    )

    assert command[:2] == [sys.executable, str(benchmark.PREFLIGHT_SCRIPT)]
    assert command[command.index("--config") + 1] == str(settings.config)
    assert command[command.index("--warmup") + 1] == "2"
    assert command[command.index("--iterations") + 1] == "4"
    assert "startrain-orchestrate" not in command
    assert "startrain-train" not in command


def test_run_case_captures_measurements_and_context(tmp_path: Path) -> None:
    settings = _settings(tmp_path)
    observed: dict[str, object] = {}

    def executor(
        command,
        *,
        cwd: Path,
        timeout_seconds: float,
    ) -> subprocess.CompletedProcess[str]:
        observed.update(
            command=list(command),
            cwd=cwd,
            timeout_seconds=timeout_seconds,
        )
        return subprocess.CompletedProcess(
            command,
            0,
            stdout=json.dumps(_payload()) + "\n",
            stderr="",
        )

    record = benchmark.run_case(
        settings=settings,
        metadata=_metadata(),
        rings=6,
        batch_size=32,
        repeat=1,
        executor=executor,
    )

    assert record["status"] == "passed"
    assert record["failure"] is None
    assert observed["cwd"] == benchmark.TRAINING_ROOT
    assert observed["timeout_seconds"] == 30.0
    measurement = record["measurement"]
    assert isinstance(measurement, dict)
    assert measurement["throughput"] == {
        "leaf_evaluations_per_second": 8_000.0,
        "minimum_leaf_evaluations_per_second": 5_000.0,
    }
    assert measurement["latency_ms"] == {
        "mean_batch": 4.0,
        "p95_batch": 5.0,
    }
    assert measurement["memory_bytes"] == {"peak_allocated": 123_456}
    context = record["context"]
    assert isinstance(context, dict)
    assert context["git"] == {"revision": "deadbeef", "dirty": False}
    assert context["device"] == {
        "requested": "cuda:0",
        "name": "NVIDIA H100",
    }


def test_run_case_records_timeout_as_structured_failure(tmp_path: Path) -> None:
    settings = _settings(tmp_path)

    def executor(
        command,
        *,
        cwd: Path,
        timeout_seconds: float,
    ) -> subprocess.CompletedProcess[str]:
        del cwd, timeout_seconds
        raise subprocess.TimeoutExpired(command, 30.0, output="partial output")

    record = benchmark.run_case(
        settings=settings,
        metadata=_metadata(),
        rings=6,
        batch_size=32,
        repeat=1,
        executor=executor,
    )

    assert record["status"] == "failed"
    assert record["failure"] == {
        "kind": "timeout",
        "message": "hardware preflight exceeded 30 seconds",
    }
    process = record["process"]
    assert isinstance(process, dict)
    assert process["return_code"] is None
    assert process["stdout"] == "partial output"


def test_harness_repeats_cases_and_returns_failure_status(tmp_path: Path) -> None:
    settings = _settings(tmp_path, repeats=2)
    calls = 0

    def executor(
        command,
        *,
        cwd: Path,
        timeout_seconds: float,
    ) -> subprocess.CompletedProcess[str]:
        nonlocal calls
        del cwd, timeout_seconds
        calls += 1
        passed = calls == 1
        payload = _payload(
            passed=passed,
            throughput=8_000.0 if passed else 4_000.0,
        )
        return subprocess.CompletedProcess(
            command,
            0 if passed else 2,
            stdout=json.dumps(payload) + "\n",
            stderr="below gate" if not passed else "",
        )

    exit_code, summary = benchmark.run_harness(
        settings,
        _metadata(),
        executor=executor,
    )

    assert calls == 2
    assert exit_code == benchmark.EXIT_BENCHMARK_FAILURE
    assert summary["status"] == "benchmark_failed"
    assert summary["case_count"] == 2
    assert summary["passed_case_count"] == 1
    assert summary["failed_case_count"] == 1
    assert summary["aggregates"][0]["leaf_evaluations_per_second"]["count"] == 2
    lines = (
        (settings.output_directory / "cases.jsonl")
        .read_text(encoding="utf-8")
        .splitlines()
    )
    assert len(lines) == 3
    assert json.loads(lines[0])["record_type"] == "run_metadata"
    assert [json.loads(line)["case"]["repeat"] for line in lines[1:]] == [1, 2]
    written_summary = json.loads(
        (settings.output_directory / "summary.json").read_text(encoding="utf-8")
    )
    assert written_summary["exit_code"] == benchmark.EXIT_BENCHMARK_FAILURE
    assert written_summary["failures"][0]["failure"]["kind"] == "performance_gate"


def test_orchestration_metrics_summary_covers_throughput_and_replay_waits(
    tmp_path: Path,
) -> None:
    run_root = tmp_path / "run"
    learner_path = run_root / "learner" / "metrics.jsonl"
    actor_path = run_root / "metrics" / "actor-gpu-1.jsonl"
    learner_path.parent.mkdir(parents=True)
    actor_path.parent.mkdir(parents=True)
    learner_records = [
        {
            "worker": "learner",
            "timestamp_ns": 1_000_000_000,
            "examples_per_second": 100.0,
            "step_seconds": 1.0,
        },
        {
            "worker": "learner",
            "timestamp_ns": 2_000_000_000,
            "examples_per_second": 200.0,
            "step_seconds": 0.5,
        },
        {
            "worker": "learner",
            "timestamp_ns": 3_000_000_000,
            "phase": "replay_wait",
        },
        {
            "worker": "learner",
            "timestamp_ns": 5_500_000_000,
            "phase": "training",
        },
    ]
    actor_records = [
        {
            "worker": "actor-gpu-1",
            "games": 8,
            "games_per_second": 2.0,
            "samples_per_second": 20.0,
            "elapsed_seconds": 4.0,
        },
        {
            "worker": "actor-gpu-1",
            "games": 9,
            "games_per_second": 4.0,
            "samples_per_second": 40.0,
            "elapsed_seconds": 3.0,
        },
    ]
    learner_path.write_text(
        "".join(json.dumps(record) + "\n" for record in learner_records),
        encoding="utf-8",
    )
    actor_path.write_text(
        "".join(json.dumps(record) + "\n" for record in actor_records),
        encoding="utf-8",
    )

    summary = benchmark.summarize_orchestration_metrics(run_root)

    assert summary["status"] == "complete"
    learner = summary["learner"]
    assert isinstance(learner, dict)
    assert learner["examples_per_second"]["mean"] == 150.0
    assert learner["batch_seconds"]["median"] == 0.75
    actors = summary["actors"]
    assert isinstance(actors, dict)
    assert actors["games_per_second"]["mean"] == 3.0
    assert actors["batch_seconds"]["maximum"] == 4.0
    assert actors["by_worker"]["actor-gpu-1"]["samples_per_second"]["p95"] == 40.0
    replay_waits = summary["replay_waits"]
    assert isinstance(replay_waits, dict)
    assert replay_waits["availability"] == "observed"
    assert replay_waits["events"] == 1
    assert replay_waits["completed_intervals"] == 1
    assert replay_waits["seconds"]["mean"] == 2.5
