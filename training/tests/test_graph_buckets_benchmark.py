from dataclasses import replace
import json
import os
import signal
import subprocess
from types import SimpleNamespace
from pathlib import Path

import pytest
import torch

from scripts import benchmark_graph_buckets as benchmark
from startrain.inference import (
    DetailedInferenceResponse,
    GraphInferenceAdapter,
    InferenceConfig,
    InferenceMetrics,
    InferenceResponse,
)


def output(rows=3):
    return DetailedInferenceResponse(
        response=InferenceResponse(
            list(range(rows)), [0.25] * rows, list(range(rows + 1)), [0.5] * rows
        ),
        outcome_probabilities=[[0.25, 0.25, 0.5]] * rows,
        outcome_values=[0.25] * rows,
        score_expectations=[1.0] * rows,
        score_probabilities=[[0.1, 0.2, 0.7]] * rows,
    )


def snapshot(*, pid=None, start=5, verified=True):
    return {
        "verified": verified,
        "owners": [{"pid": os.getpid() if pid is None else pid, "start_ticks": start}],
    }


class Clock:
    now = 0.0

    def __call__(self):
        return self.now


class Adapter:
    def __init__(self, clock, *, seconds, physical, fallback=False):
        self.clock, self.seconds, self.physical = clock, seconds, physical
        self.fallback = fallback
        self.metrics = InferenceMetrics()

    def prepare_requests(self, request):
        return request

    def _inference_batch_rows(self, rows):
        assert rows == 3
        return self.physical

    def evaluate_prepared(self, requests, *, include_details):
        assert len(requests) == 1 and len(requests[0]) == 3
        self.clock.now += self.seconds
        self.metrics = replace(
            self.metrics,
            evaluator_rows=self.metrics.evaluator_rows + 3,
            evaluator_calls=self.metrics.evaluator_calls + 1,
            neural_rows=self.metrics.neural_rows + self.physical,
            neural_calls=self.metrics.neural_calls + 1,
            graph_captures=1,
            graph_replays=self.metrics.graph_replays
            + int(self.metrics.graph_captures > 0),
            graph_fallbacks=self.metrics.graph_fallbacks + int(self.fallback),
        )
        details = output()
        return [(details.response, details if include_details[0] else None)]

    def metrics_snapshot(self):
        return self.metrics


def run_case(*, context="isolated", fallback=False):
    clock = Clock()
    return benchmark._case(
        [0, 1, 2],
        [
            Adapter(clock, seconds=2, physical=4),
            Adapter(clock, seconds=1, physical=3, fallback=fallback),
        ],
        repeats=4,
        iterations=2,
        warmups=2,
        sync=lambda: None,
        snapshot=snapshot,
        load_context=context,
        clock=clock,
    )


def test_paired_timings_exclude_warmup_charge_padding_and_counterbalance_exactly():
    case = run_case()
    assert case["physical_rows"] == [4, 3]
    assert case["warmup_capture_compile_seconds"] == [4, 2]
    assert case["latency_seconds"] == [[2] * 4, [1] * 4]
    assert case["paired_speedup_median"] == 2
    assert case["comparison_adoptable"] and case["graph_replay_only"]
    assert [row["arm"] for row in case["counterbalanced_sequence"]] == [
        benchmark.ARMS[index] for index in (0, 1, 1, 0, 0, 1, 1, 0)
    ]
    assert [row["neural_rows"] for row in case["measured_metrics"]] == [32, 24]
    assert len(case["gpu_observations"]) == 9


@pytest.mark.parametrize(
    "context,fallback", [("shared", False), ("unverified", False), ("isolated", True)]
)
def test_shared_unverified_or_fallback_runs_never_make_adoptable_speed_claim(
    context, fallback
):
    case = run_case(context=context, fallback=fallback)
    assert not case["comparison_adoptable"]
    assert case["paired_speedup_median"] is None
    assert case["raw_paired_latency_ratios"] == [2] * 4


@pytest.mark.parametrize(
    "later", [snapshot(pid=1234), snapshot(start=6), snapshot(verified=False)]
)
def test_changed_owner_reused_pid_or_missing_telemetry_invalidates_isolation(later):
    result = benchmark._load_assessment(
        [snapshot(), later], declared="isolated", worker_pid=os.getpid()
    )
    assert not result["comparison_adoptable"]
    assert not result["observed_pid_stationarity"]


@pytest.mark.parametrize("wrong_gpu", [False, True])
@pytest.mark.parametrize("torch_uuid", ["test", "GPU-test"])
def test_gpu_telemetry_validates_physical_identity_and_records_pid_start_time(
    monkeypatch, wrong_gpu, torch_uuid
):
    def query(command, **kwargs):
        assert kwargs["timeout"] == 3
        assert command[command.index("--id") + 1] == "GPU-test"
        if "--query-gpu=" in " ".join(command):
            return SimpleNamespace(stdout="GPU-test, 20, 100, 80000, 1500, 200, 50\n")
        uuid = "GPU-wrong" if wrong_gpu else "GPU-test"
        return SimpleNamespace(stdout=f"{uuid}, 123, 100\n")

    def read(path):
        assert str(path) == "/proc/123/stat"
        return "123 (benchmark worker) " + " ".join(["S"] + ["0"] * 18 + ["99"])

    monkeypatch.setattr(benchmark.subprocess, "run", query)
    monkeypatch.setattr(Path, "read_text", read)
    row = benchmark._gpu_snapshot(torch_uuid)
    assert row["gpu_uuid"] == "GPU-test"
    assert row["verified"] is (not wrong_gpu)
    if not wrong_gpu:
        assert row["owners"] == [{"pid": 123, "start_ticks": 99, "memory_mib": "100"}]
        assert row["load"]["utilization_percent"] == "20"


@pytest.mark.parametrize(
    "corruption", ["tokens", "padding", "nonfinite", "policy", "value"]
)
def test_output_parity_rejects_routing_padding_nonfinite_and_numeric_errors(corruption):
    expected, actual = output(), output()
    if corruption == "tokens":
        actual.response.tokens.reverse()
    elif corruption == "padding":
        actual.score_expectations.append(1.0)
    elif corruption == "nonfinite":
        actual.outcome_values[1] = float("nan")
    elif corruption == "policy":
        actual.response.policy_logits.append(0.5)
    else:
        actual.response.values[2] = 0.8
    with pytest.raises(ValueError):
        benchmark._compare_outputs(expected, actual, 3)


def test_requested_cases_match_actual_production_padding_including_65_row_control():
    adapter = object.__new__(GraphInferenceAdapter)
    adapter.device = torch.device("cuda:0")
    result = []
    for small in (False, True):
        adapter.config = InferenceConfig(
            cuda_graphs=True, small_batch_graph_buckets=small
        )
        result.append(
            [adapter._inference_batch_rows(n) for n in benchmark.DEFAULT_ROWS]
        )
    assert result == [[4, 8, 16, 32, 64, 96], [3, 6, 12, 24, 48, 96]]


@pytest.mark.native
def test_real_native_positions_route_all_valid_predictions_through_tiny_network():
    native = pytest.importorskip("star_native")
    from startrain.model import GraphResTNet, ModelConfig

    requests = benchmark._native_requests(native, 4, 3)
    assert len(set(requests.inference_keys())) == 3
    model = GraphResTNet(
        ModelConfig(width=8, rrt_groups=1, attention_heads=2, kv_heads=1)
    ).eval()
    adapters = [
        GraphInferenceAdapter(
            model,
            config=InferenceConfig(cuda_graphs=True, small_batch_graph_buckets=small),
        )
        for small in (False, True)
    ]
    try:
        expected, actual = [adapter.evaluate_detailed(requests) for adapter in adapters]
        parity = benchmark._compare_outputs(expected, actual, 3)
        assert parity["values"]["shape"] == [3]
        assert all(row["max_absolute_difference"] == 0 for row in parity.values())
    finally:
        for adapter in adapters:
            adapter.close()


def arguments():
    return ["--config", "config.yaml", "--checkpoint", "frozen.json"]


@pytest.mark.parametrize(
    "extra",
    [
        ["--repeats", "3"],
        ["--batch-sizes", "7"],
        ["--batch-sizes", "3", "3"],
        ["--rings", "12"],
        ["--device", "cpu"],
        ["--timeout-seconds", "nan"],
        ["--memory-fraction", "0.5"],
        ["--warmups", "1"],
        ["--graph-cache-bytes", "0"],
    ],
)
def test_unbounded_or_non_counterbalanced_plans_are_rejected_before_loading(
    extra, monkeypatch
):
    monkeypatch.setattr(
        benchmark, "_plan", lambda _: pytest.fail("loaded invalid plan")
    )
    with pytest.raises(SystemExit):
        benchmark.main([*arguments(), *extra])


def test_default_plan_does_not_execute_or_write(monkeypatch, capsys):
    monkeypatch.setattr(
        benchmark, "_plan", lambda _: ({"frozen": "identity"}, None, None)
    )
    monkeypatch.setattr(benchmark, "_worker", lambda *_: pytest.fail("executed worker"))
    monkeypatch.setattr(
        benchmark.subprocess,
        "Popen",
        lambda *_args, **_kwargs: pytest.fail("spawned worker"),
    )
    assert benchmark.main(arguments()) == 0
    assert json.loads(capsys.readouterr().out)["frozen"] == "identity"


def test_changed_pinned_plan_is_rejected_before_execution(monkeypatch):
    monkeypatch.setattr(benchmark, "_plan", lambda _: ({"frozen": "new"}, None, None))
    with pytest.raises(ValueError, match="changed after planning"):
        benchmark.main([*arguments(), "--pinned-plan", "old"])


def test_worker_deadline_kills_entire_process_group(tmp_path, monkeypatch):
    monkeypatch.setattr(
        benchmark, "_plan", lambda _: ({"frozen": "identity"}, None, None)
    )
    calls = []

    class Child:
        pid = 12345
        returncode = 0

        def communicate(self, timeout=None):
            calls.append(timeout)
            if timeout is not None:
                raise subprocess.TimeoutExpired("worker", timeout)
            return "", ""

    def spawn(cmd, **kwargs):
        assert "--worker" in cmd and "--pinned-plan" in cmd
        assert kwargs["start_new_session"] is True
        assert kwargs["env"]["OMP_NUM_THREADS"] == "2"
        return Child()

    killed = []
    monkeypatch.setattr(benchmark.subprocess, "Popen", spawn)
    monkeypatch.setattr(benchmark.os, "killpg", lambda *args: killed.append(args))
    with pytest.raises(subprocess.TimeoutExpired):
        benchmark.main(
            [
                *arguments(),
                "--execute",
                "--output",
                str(tmp_path / "report.json"),
                "--timeout-seconds",
                "5",
            ]
        )
    assert killed == [(12345, signal.SIGKILL)]
    assert calls == [5, None]
    assert not (tmp_path / "report.json").exists()
