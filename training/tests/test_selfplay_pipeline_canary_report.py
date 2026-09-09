import json
from pathlib import Path

import pytest
import yaml

from scripts import report_selfplay_pipeline_canary as canary


NOW = 100_000_000_000
SINCE = 10_000_000_000
START = 20_000_000_000


def write(path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload))


def rows(path, records):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(row) + "\n" for row in records))


@pytest.fixture
def fixture(tmp_path):
    root = tmp_path / "run"
    profile = tmp_path / "profile.yaml"
    source = Path(__file__).parents[1] / "configs/h100-8gpu-largest-board-priority.yaml"
    raw = yaml.safe_load(source.read_text())
    raw["orchestration"]["directories"]["root"] = str(root)
    raw["orchestration"]["model_refresh"]["inference"]["cuda_graph_max_bytes"] = (
        16 * 1024**3
    )
    raw["orchestration"]["gpus"][7].update(
        actor_cohorts=4,
        actor_batch_size=64,
        actor_pipeline={
            "compatible_work": True,
            "stream_completed_games": True,
            "rolling_game_slots": True,
            "seed_contract": "game-v1",
            "games_per_task": 128,
            "cuda_graphs": True,
        },
    )
    profile.write_text(yaml.safe_dump(raw))
    write(
        root / "run.json", {"run_id": "variant-network", "generation_family": "family"}
    )
    workers = {}
    for gpu, pid in ((6, 600), (7, 700)):
        name = f"actor-{gpu}"
        path = root / "status" / f"{name}.heartbeat.json"
        workers[name] = {
            "role": "actor",
            "gpu_ids": [gpu],
            "pid": pid,
            "state": "running",
            "restart_count": 0,
            "heartbeat": str(path),
        }
        write(
            path,
            {
                "worker": name,
                "pid": pid,
                "heartbeat_ns": NOW,
                "phase": "shared_cohorts",
                "effective_coordinated_work": gpu == 7,
                "compatible_work": {
                    "leases_issued": 4,
                    "rings": {"10": 4},
                    "mode_categories": {"double-standard": 2, "classic-pie": 2},
                },
                "inference": {
                    "worker_failures": 0,
                    "failed_requests": 0,
                    "neural_batches": 200,
                    "pending_requests": 2,
                    "physical_inference": {
                        "neural_calls": 200,
                        "neural_rows": 51_200,
                        "neural_padding_rows": 100,
                        "cache_hits": 2000,
                        "cache_misses": 30_000,
                        "graph_captures": 2,
                        "graph_replays": 150,
                        "graph_validation_failures": 0,
                        "graph_fallbacks": 5,
                        "graph_retained_bytes": 3 * 1024**3,
                    },
                },
            },
        )
    write(
        root / "status/coordinator.json",
        {
            "coordinator_pid": 500,
            "timestamp_ns": NOW,
            "state": "running",
            "draining": False,
            "workers": workers,
        },
    )
    child = {
        "worker": "actor-7-cohort-0",
        "pid": 700,
        "heartbeat_ns": NOW,
        "phase": "selfplay_refill",
        "generation": 9,
        "process_started_ns": START,
        "cumulative_games": 20,
        "cumulative_samples": 5000,
        "cumulative_batch_wall_seconds": 50.0,
        "completed_games": 20,
        "started_games": 80,
        "refilled_games": 16,
    }
    write(root / "status/actor-7.heartbeat-cohort-0.json", child)
    base = {
        "worker": child["worker"],
        "gpu_id": 7,
        "run_id": "variant-network",
        "generation_family": "family",
        "process_started_ns": START,
        "timestamp_ns": NOW - 100,
        "generation": 9,
        "ring": 10,
        "variant": "double",
        "model_identity": "model-a",
        "model_role": "candidate",
        "requested_model_role": "candidate",
        "work_bundle": 0,
        "work_lease": 0,
    }
    publication = base | {
        "record_kind": "publication",
        "published_task_games": 20,
        "published_task_samples": 5000,
        "cumulative_games": 20,
        "cumulative_samples": 5000,
        "cumulative_batch_wall_seconds": 50.0,
    }
    path = root / "metrics/actor-7-cohort-0.jsonl"
    rows(path, [publication, publication])
    (root / "logs").mkdir()
    (root / "logs/actor-7.log").write_text(
        "CUDA graph inference fallback: unsupported stride\n"
    )
    return root, profile, child, base, path


def report(fixture):
    root, profile, *_ = fixture
    return canary.build_report(root, profile, since_ns=SINCE, now_ns=NOW)


def test_active_publications_are_durable_without_waiting_for_task_completion(fixture):
    result = report(fixture)
    assert result["gate"] == "pass"
    assert result["canary"]["durable_games"] == 20
    assert result["canary"]["durable_samples"] == 5000
    assert result["canary"]["processes"][0]["completed_tasks"] == 0
    assert result["canary"]["refilled_games_lower_bound"] == 16
    assert result["canary"]["physical_inference"]["graph_replays"] == 150
    assert result["canary"]["settings"]["graph_max_bytes_aggregate"] == 16 * 1024**3
    assert (
        result["canary"]["settings"]["graph_max_bytes_per_model"] == 16 * 1024**3 // 6
    )
    assert result["canary"]["fallback_logs"]["count"] == 1
    assert result["expected"]["model_parameters"] == 17_402_775
    assert "6" in result["baseline_gpus"]


def test_publication_and_final_records_use_process_maxima_not_sums(fixture):
    root, _, child, base, path = fixture
    final = base | {
        "games": 80,
        "started_games": 80,
        "cumulative_games": 80,
        "cumulative_samples": 20_000,
        "dropped_games": 0,
        "dropped_decisions": 0,
    }
    with path.open("a") as stream:
        stream.write(json.dumps(final) + "\n" + json.dumps(final) + "\n")
    child.update(
        phase="cohort_complete", cumulative_games=80, cumulative_samples=20_000
    )
    write(root / "status/actor-7.heartbeat-cohort-0.json", child)
    result = report(fixture)
    assert result["canary"]["durable_games"] == 80
    assert result["canary"]["durable_samples"] == 20_000
    assert result["canary"]["refilled_games_lower_bound"] == 16
    assert result["canary"]["observed_task_mix"]["variant"] == {"double": 1}


def test_old_process_wrong_identity_and_partial_tail_are_excluded(fixture):
    _, _, _, base, path = fixture
    fake = base | {
        "record_kind": "publication",
        "cumulative_games": 100_000,
        "cumulative_samples": 1_000_000,
    }
    with path.open("a") as stream:
        for record in (
            fake | {"process_started_ns": 1},
            fake | {"process_started_ns": START - 1},
            fake | {"generation_family": "other"},
            fake | {"worker": "actor-70-cohort-0"},
        ):
            stream.write(json.dumps(record) + "\n")
        stream.write('{broken-json}\n{"unfinished":')
    result = report(fixture)
    assert result["canary"]["durable_games"] == 20
    assert result["read_evidence"]["partial_jsonl_lines"] == 1
    assert result["read_evidence"]["malformed_jsonl_lines"] == 1


def test_missing_process_start_is_anchored_to_current_heartbeat_generation(fixture):
    root, _, child, *_ = fixture
    del child["process_started_ns"]
    write(root / "status/actor-7.heartbeat-cohort-0.json", child)
    assert report(fixture)["gate"] == "pass"


@pytest.mark.parametrize("change", ["wrong_pid", "stale"])
def test_untrusted_heartbeat_cannot_supply_old_success_counters(fixture, change):
    root, _, child, *_ = fixture
    child["pid" if change == "wrong_pid" else "heartbeat_ns"] = 3
    write(root / "status/actor-7.heartbeat-cohort-0.json", child)
    result = report(fixture)
    assert result["gate"] == "pending"
    assert result["canary"]["durable_games"] == 0


@pytest.mark.parametrize(
    "failure", ["restart", "inference", "validation", "drop", "history"]
)
def test_operational_failures_block_the_gate(fixture, failure):
    root, _, _, base, path = fixture
    if failure == "restart":
        coordinator = canary._json(root / "status/coordinator.json")
        coordinator["workers"]["actor-7"]["restart_count"] = 1
        write(root / "status/coordinator.json", coordinator)
    elif failure in ("inference", "validation"):
        heartbeat = canary._json(root / "status/actor-7.heartbeat.json")
        if failure == "inference":
            heartbeat["inference"]["failed_requests"] = 1
        else:
            heartbeat["inference"]["physical_inference"][
                "graph_validation_failures"
            ] = 1
        write(root / "status/actor-7.heartbeat.json", heartbeat)
    elif failure == "drop":
        with path.open("a") as stream:
            stream.write(
                json.dumps(
                    base
                    | {
                        "games": 20,
                        "started_games": 21,
                        "dropped_games": 1,
                        "dropped_decisions": 3,
                        "cumulative_games": 20,
                        "cumulative_samples": 5000,
                    }
                )
                + "\n"
            )
    else:
        rows(
            root / "metrics/coordinator.jsonl",
            [
                {
                    "timestamp_ns": NOW - 10,
                    "worker": "actor-7",
                    "event": "worker_exited",
                    "pid": 699,
                    "restart_count": 1,
                }
            ],
        )
    assert report(fixture)["gate"] == "fail"


def test_carried_heartbeat_refill_counters_are_not_counted_twice(fixture):
    root, _, child, base, path = fixture
    final = base | {
        "games": 80,
        "started_games": 80,
        "cumulative_games": 80,
        "cumulative_samples": 20_000,
    }
    publication = base | {
        "generation": 10,
        "record_kind": "publication",
        "published_task_games": 1,
        "cumulative_games": 81,
        "cumulative_samples": 20_250,
    }
    rows(path, [final, publication])
    child.update(
        generation=10,
        phase="selfplay_cohort",
        cumulative_games=81,
        cumulative_samples=20_250,
        started_games=80,
        refilled_games=16,
    )
    write(root / "status/actor-7.heartbeat-cohort-0.json", child)
    result = report(fixture)
    assert result["canary"]["refilled_games_lower_bound"] == 16
    assert result["canary"]["started_games_lower_bound"] == 81


def test_report_only_writes_its_own_external_artifact(fixture, tmp_path):
    root, profile, *_ = fixture
    before = {
        str(path): path.read_bytes() for path in root.rglob("*") if path.is_file()
    }
    result = report(fixture)
    output = tmp_path / "reports/canary.json"
    canary.write_report(output, result)
    canary.write_report(output, result)
    assert json.loads(output.read_text())["gate"] == "pass"
    assert {
        str(path): path.read_bytes() for path in root.rglob("*") if path.is_file()
    } == before
    for protected in (root / "status/new-control.json", profile):
        with pytest.raises(ValueError, match="protected"):
            canary.write_report(protected, result)
    foreign = tmp_path / "foreign.json"
    foreign.write_text("{}")
    with pytest.raises(ValueError, match="not this canary"):
        canary.write_report(foreign, result)


def test_cli_produces_a_read_only_report_with_configurable_minimums(
    fixture, monkeypatch, capsys
):
    root, profile, *_ = fixture
    monkeypatch.setattr(canary.time, "time_ns", lambda: NOW)
    assert (
        canary.main(
            [
                "--run-root",
                str(root),
                "--profile",
                str(profile),
                "--gpu7",
                "--since-ns",
                str(SINCE),
                "--min-games",
                "21",
            ]
        )
        == 0
    )
    result = json.loads(capsys.readouterr().out)
    assert result["gate"] == "pending"
    assert result["minimums"]["durable_games"] == {"observed": 20, "required": 21}
