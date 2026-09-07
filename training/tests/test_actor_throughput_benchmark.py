from __future__ import annotations

import argparse
from dataclasses import replace
import json
import os
from pathlib import Path
import signal
from types import SimpleNamespace
import subprocess
import sys
import threading

import pytest

from scripts import benchmark_actor_throughput as benchmark
from startrain.config import load_config
from startrain.inference import InferenceResponse


def _arguments(tmp_path: Path) -> list[str]:
    return [
        "--config",
        "configs/small.yaml",
        "--checkpoint",
        "immutable.json",
        "--device",
        "cuda:7",
        "--cpu-affinity",
        "0-7",
        "--arms",
        "2:4:256",
        "4:8:512",
        "--output-dir",
        str(tmp_path / "report"),
    ]


@pytest.mark.parametrize(
    "value", ["2:4", "2:4:256:3", "a:4:256", "0:4:256", "2:0:256", "2:4:0"]
)
def test_arm_rejects_invalid_resource_budgets(value):
    with pytest.raises(argparse.ArgumentTypeError):
        benchmark.parse_arm(value)


def test_plan_does_not_load_checkpoint_or_create_output(tmp_path, monkeypatch, capsys):
    import startrain.checkpoint as checkpoint

    monkeypatch.setattr(
        checkpoint, "load_model_manifest", lambda _: pytest.fail("loaded model")
    )
    assert benchmark.main(_arguments(tmp_path)) == 0
    report = json.loads(capsys.readouterr().out)
    assert report["arms"] == ["2:4:256", "4:8:512"]
    assert "completed-game canary required" in report["scope"]
    assert not (tmp_path / "report").exists()


@pytest.mark.parametrize(
    "option,value",
    [
        ("--plies", "17"),
        ("--tasks", "2"),
        ("--blas-threads", "9"),
        ("--timeout-seconds", "nan"),
    ],
)
def test_harness_rejects_unbounded_or_incompatible_cases(tmp_path, option, value):
    with pytest.raises(SystemExit) as error:
        benchmark.main([*_arguments(tmp_path), option, value])
    assert error.value.code == 2


def _report():
    return {
        "status": "measured",
        "model_identity": "model",
        "manifest_sha256": "manifest",
        "selfplay_config": {"full_simulations": 384},
        "completed_search_decisions": 4096,
        "search_simulations": 40000,
        "requested_neural_rows": 41000,
        "tasks": [{"action_visit_sha256": "a", "replay_trace_sha256": "b"}],
        "measured_seconds": 20,
    }


@pytest.mark.parametrize(
    "field",
    [
        "tasks",
        "search_simulations",
        "requested_neural_rows",
        "completed_search_decisions",
        "selfplay_config",
        "model_identity",
        "manifest_sha256",
    ],
)
def test_any_work_or_trace_difference_prevents_speed_claim(field):
    before = _report()
    after = before | {"measured_seconds": 10}
    assert benchmark.compare(before, after)["relative_search_speed"] == 2
    after[field] = None
    result = benchmark.compare(before, after)
    assert result["exact_work_and_trace_parity"] is False
    assert result["relative_search_speed"] is None
    assert result["parity_mismatches"] == [field]


@pytest.mark.parametrize(
    "context,verified,adoptable",
    [
        ("unverified", True, False),
        ("shared", True, False),
        ("isolated", False, False),
        ("isolated", True, True),
    ],
)
def test_only_verified_isolated_exact_comparison_is_adoptable(
    context, verified, adoptable
):
    record = _report() | {"load_context": context, "isolation_verified": verified}
    assert benchmark.compare(record, record)["adoptable_comparison"] is adoptable


def test_ready_marker_is_atomic_and_cannot_replace_another_run(tmp_path):
    marker = tmp_path / "ready.json"
    benchmark._write_ready(marker, {"status": "ready", "token": "first"})
    with pytest.raises(FileExistsError):
        benchmark._write_ready(marker, {"token": "second"})
    assert json.loads(marker.read_text())["token"] == "first"
    assert list(tmp_path.iterdir()) == [marker]


def test_independent_reports_rank_only_exact_isolated_cases_against_first_baseline(
    tmp_path,
):
    base = _report() | {
        "load_context": "isolated",
        "isolation_verified": True,
        "arm": "2:4:256",
    }
    records = [
        base,
        base | {"measured_seconds": 10, "preserve_broadcast_topology": True},
        base | {"measured_seconds": 5, "load_context": "shared"},
        base | {"measured_seconds": 6, "search_simulations": 20000},
    ]
    paths = []
    for index, record in enumerate(records):
        path = tmp_path / f"report-{index}.json"
        path.write_text(
            json.dumps({"benchmark": "fixed-selfplay-search-prefix", "cases": [record]})
        )
        paths.append(path)
    combined = benchmark.combine_reports(paths)
    assert [item["measured_seconds"] for item in combined["ranked_adoptable"]] == [
        10,
        20,
    ]
    assert combined["ranked_adoptable"][0]["relative_search_speed"] == 2
    assert len(combined["rejected"]) == 2
    assert combined["baseline_report"] == str(paths[0])
    with pytest.raises(ValueError, match="unique"):
        benchmark.combine_reports([paths[0], paths[0]])


@pytest.mark.native
def test_warmup_exercises_all_physical_buckets_with_unique_positions_and_cold_cache():
    native = pytest.importorskip("star_native")
    from startrain.native import positions_from_native

    class Adapter:
        rows = 0
        clears = 0
        seen = set()

        def clear_inference_cache(self):
            self.clears += 1
            self.seen = set()

        def metrics_snapshot(self):
            return SimpleNamespace(neural_rows=self.rows)

        def evaluate(self, requests):
            assert not self.seen
            positions = positions_from_native(requests.states)
            keys = {position.stones.numpy().tobytes() for position in positions}
            assert len(keys) == len(requests)
            self.seen = keys
            self.rows += len(keys)

    adapter = Adapter()
    warmed = benchmark._warmup_inference(
        native, adapter, ring=4, max_rows=512, variants=["double", "handicap-9-double"]
    )
    assert [item["physical_rows"] for item in warmed] == [
        1,
        2,
        4,
        8,
        16,
        32,
        64,
        128,
        256,
        512,
    ] * 2
    assert adapter.clears == len(warmed) + 1 and not adapter.seen


def test_barrier_releases_only_its_own_ready_token(tmp_path, monkeypatch):
    marker = tmp_path / "ready.json"
    barrier = tmp_path / "release.json"
    args = SimpleNamespace(start_barrier=barrier, ready_marker=marker)
    sleeps = []

    def sleep(_seconds):
        sleeps.append(_seconds)
        ready = json.loads(marker.read_text())
        assert ready["pid"] == benchmark.os.getpid()
        tokens = ["another-run"] if len(sleeps) == 1 else [ready["token"]]
        barrier.write_text(json.dumps({"schema_version": 1, "ready_tokens": tokens}))

    monkeypatch.setattr(benchmark.time, "sleep", sleep)
    waited = benchmark._await_start(
        args, {"arm": "2:4:256"}, deadline=benchmark.time.monotonic() + 2
    )
    assert waited >= 0 and len(sleeps) == 2


def test_barrier_deadline_is_bounded(tmp_path):
    args = SimpleNamespace(
        start_barrier=tmp_path / "barrier", ready_marker=tmp_path / "ready"
    )
    with pytest.raises(TimeoutError, match="deadline"):
        benchmark._await_start(args, {}, deadline=benchmark.time.monotonic() - 1)


@pytest.mark.parametrize("other_pid", [None, 12345])
def test_gpu_ownership_gate_requires_current_process_only(monkeypatch, other_pid):
    lines = [f"GPU-example, {benchmark.os.getpid()}"]
    if other_pid is not None:
        lines.append(f"GPU-example, {other_pid}")
    monkeypatch.setattr(
        benchmark.subprocess,
        "run",
        lambda *args, **kwargs: SimpleNamespace(
            returncode=0, stdout="\n".join(lines), stderr=""
        ),
    )
    evidence = benchmark._gpu_ownership("GPU-example")
    assert evidence["verified"] is (other_pid is None)


def test_controller_passes_barrier_and_total_timeout_to_child(tmp_path, monkeypatch):
    import startrain.checkpoint as checkpoints

    manifest = SimpleNamespace(
        artifact_manifest=tmp_path / "model",
        path=tmp_path / "model",
        manifest_sha256="a" * 64,
    )
    monkeypatch.setattr(checkpoints, "load_model_manifest", lambda _: manifest)

    def run(command, **kwargs):
        for option, expected in (
            ("--start-barrier", str(tmp_path / "barrier")),
            ("--ready-marker", str(tmp_path / "ready")),
            ("--load-context", "isolated"),
            ("--timeout-seconds", "300.0"),
        ):
            assert command[command.index(option) + 1] == expected
        assert kwargs["timeout"] == 300
        return SimpleNamespace(returncode=0, stdout=json.dumps(_report()), stderr="")

    monkeypatch.setattr(benchmark, "_run_owned", run)
    assert (
        benchmark.main(
            [
                *_arguments(tmp_path),
                "--arms",
                "2:4:256",
                "--execute",
                "--start-barrier",
                str(tmp_path / "barrier"),
                "--ready-marker",
                str(tmp_path / "ready"),
                "--timeout-seconds",
                "300",
                "--load-context",
                "isolated",
            ]
        )
        == 0
    )


@pytest.mark.parametrize("timed_out", [False, True])
def test_parent_pins_checkpoint_starts_fresh_process_and_keeps_timeout_diagnostic(
    tmp_path, monkeypatch, timed_out
):
    import startrain.checkpoint as checkpoints

    pinned = SimpleNamespace(
        artifact_manifest=tmp_path / "pinned.json",
        path=tmp_path / "pointer.json",
        manifest_sha256="a" * 64,
    )
    monkeypatch.setattr(checkpoints, "load_model_manifest", lambda _: pinned)
    calls = []

    def run(command, **kwargs):
        calls.append(command)
        assert command[command.index("--checkpoint") + 1] == str(
            pinned.artifact_manifest
        )
        assert command[command.index("--manifest-sha256") + 1] == pinned.manifest_sha256
        arm = command[command.index("--arms") + 1]
        assert kwargs["env"]["RAYON_NUM_THREADS"] == arm.split(":")[1]
        assert kwargs["timeout"] == 900
        if timed_out:
            raise subprocess.TimeoutExpired(
                command, 900, output=b"partial", stderr=b"diagnostic"
            )
        return SimpleNamespace(returncode=0, stdout=json.dumps(_report()), stderr="")

    monkeypatch.setattr(benchmark, "_run_owned", run)
    assert benchmark.main([*_arguments(tmp_path), "--execute"]) == int(timed_out)
    report = json.loads((tmp_path / "report/report.json").read_text())
    assert len(calls) == 2
    if timed_out:
        assert all(case["status"] == "timeout" for case in report["cases"])
        assert report["cases"][1]["relative_search_speed"] is None
        assert (tmp_path / "report/arm-0.stdout.log").read_text() == "partial"
    else:
        assert report["cases"][1]["exact_work_and_trace_parity"] is True
    assert not list(tmp_path.rglob("*.pt")) and not list(tmp_path.rglob("*replay*"))


def test_controller_interruption_records_logs_and_does_not_start_more_arms(
    tmp_path, monkeypatch
):
    import startrain.checkpoint as checkpoints

    manifest = SimpleNamespace(
        artifact_manifest=tmp_path / "model",
        path=tmp_path / "model",
        manifest_sha256="a" * 64,
    )
    monkeypatch.setattr(checkpoints, "load_model_manifest", lambda _: manifest)
    calls = []

    def run(command, **_kwargs):
        calls.append(command)
        raise benchmark.BenchmarkInterrupted(
            signal.SIGTERM, stdout="partial", stderr="cancelled"
        )

    monkeypatch.setattr(benchmark, "_run_owned", run)
    assert benchmark.main([*_arguments(tmp_path), "--execute"]) == 128 + signal.SIGTERM
    report = json.loads((tmp_path / "report/report.json").read_text())
    assert len(calls) == len(report["cases"]) == 1
    assert report["cases"][0]["status"] == "interrupted"
    assert (tmp_path / "report/arm-0.stdout.log").read_text() == "partial"
    assert (tmp_path / "report/arm-0.stderr.log").read_text() == "cancelled"


def _sleeping_process_tree(pid_path: Path) -> list[str]:
    source = """
import json, os, signal, subprocess, sys, time
signal.signal(signal.SIGTERM, signal.SIG_IGN)
grandchild = subprocess.Popen([sys.executable, '-c', 'import time; time.sleep(60)'])
with open(sys.argv[1], 'w') as stream:
    json.dump([os.getpid(), grandchild.pid], stream)
print('ready', flush=True)
print('compiler running', file=sys.stderr, flush=True)
time.sleep(60)
"""
    return [sys.executable, "-c", source, str(pid_path)]


def _assert_tree_stopped(pid_path: Path) -> None:
    pids = json.loads(pid_path.read_text())
    for pid in pids:
        state = subprocess.run(
            ["ps", "-o", "stat=", "-p", str(pid)],
            capture_output=True,
            text=True,
            check=False,
        ).stdout.strip()
        # A grandchild may briefly be a reparented zombie before init reaps it;
        # it is no longer running and cannot consume CPU or retain GPU resources.
        assert not state or state.startswith("Z"), (pid, state)


@pytest.mark.skipif(os.name != "posix", reason="requires owned POSIX sessions")
@pytest.mark.parametrize("interrupted", [False, True])
def test_real_timeout_and_sigterm_kill_owned_child_and_grandchild(
    tmp_path, interrupted
):
    pid_path = tmp_path / "pids.json"
    timer = None
    try:
        if interrupted:
            timer = threading.Timer(0.5, lambda: os.kill(os.getpid(), signal.SIGTERM))
            timer.start()
        with benchmark._controller_signals():
            expected = (
                benchmark.BenchmarkInterrupted
                if interrupted
                else subprocess.TimeoutExpired
            )
            with pytest.raises(expected) as captured:
                benchmark._run_owned(
                    _sleeping_process_tree(pid_path),
                    env=dict(os.environ),
                    timeout=5 if interrupted else 0.5,
                    grace_seconds=0.05,
                )
        assert "ready" in captured.value.stdout
        assert "compiler running" in captured.value.stderr
        _assert_tree_stopped(pid_path)
    finally:
        if timer is not None:
            timer.cancel()
        if pid_path.exists():
            parent_pid = json.loads(pid_path.read_text())[0]
            try:
                os.killpg(parent_pid, signal.SIGKILL)
            except ProcessLookupError:
                pass


@pytest.mark.native
@pytest.mark.parametrize(
    "variant",
    [
        "double",
        "classic",
        "pie-double",
        "pie-classic",
        "handicap-9-double",
        "handicap-9-classic",
    ],
)
def test_native_prefix_finishes_searches_preserves_task_seeds_and_never_emits_samples(
    variant,
):
    native = pytest.importorskip("star_native")

    class UniformEvaluator:
        model_version = model_identity = "uniform"
        model_step = 0

        def evaluate(self, requests):
            return InferenceResponse(
                tokens=list(requests.tokens),
                values=[-0.1] * len(requests),
                policy_offsets=list(requests.legal_offsets),
                policy_logits=[0.0] * len(requests.legal_actions),
            )

    class Broker:
        def cohort_adapter(self, base, **_kwargs):
            return base

    config = load_config(Path(__file__).parents[1] / "configs/small.yaml")
    config = replace(
        config,
        selfplay=replace(
            config.selfplay,
            full_simulations=8,
            fast_simulations=2,
            full_probability=0.5,
            fast_probability=0.5,
            simulation_reference_rings=4,
            max_considered=2,
            record_fast_policy_targets=True,
        ),
    )
    options = dict(
        broker=Broker(), tasks=4, batch_size=2, plies=3, ring=4, variants=[variant]
    )
    first = benchmark.run_tasks(
        native, UniformEvaluator(), config, cohorts=1, **options
    )
    second = benchmark.run_tasks(
        native, UniformEvaluator(), config, cohorts=4, **options
    )
    assert first == second
    assert len({task["action_visit_sha256"] for task in first}) == 4
    for task in first:
        assert task["completed_search_waves"] == 3
        assert task["completed_search_decisions"] == 6
        assert task["search_simulations"] >= 12
        assert task["completed_games"] == task["persisted_positions"] == 0
