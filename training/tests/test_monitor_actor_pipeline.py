import json

from scripts import monitor_run as monitor


def records(tmp_path, rows):
    root = tmp_path / "metrics"
    root.mkdir()
    (root / "actor.jsonl").write_text("".join(json.dumps(row) + "\n" for row in rows))
    return root


def publication(time, games, samples, *, started=1_000_000_000):
    return {
        "worker": "actor-gpu-7-cohort-0",
        "gpu_id": 7,
        "process_started_ns": started,
        "batch": 0,
        "record_kind": "publication",
        "task_started_ns": 2_000_000_000,
        "timestamp_ns": time,
        "cumulative_games": games,
        "cumulative_samples": samples,
        "cumulative_evaluator_rows": samples * 10,
    }


def test_publications_are_visible_before_task_completion_and_never_double_count(
    tmp_path,
):
    first = publication(5_000_000_000, 2, 100)
    second = publication(8_000_000_000, 4, 200)
    root = records(tmp_path, [first, second])
    before = monitor._actor_throughput_window(root, now_ns=9_000_000_000)
    assert before["fleet"]["games"] == 4
    assert before["fleet"]["samples"] == 200
    final = {
        **second,
        "record_kind": "batch",
        "batch_started_ns": 2_000_000_000,
        "batch_completed_ns": 9_000_000_000,
        "games": 4,
        "samples": 200,
    }
    with (root / "actor.jsonl").open("a") as stream:
        stream.write(json.dumps(final) + "\n")
    after = monitor._actor_throughput_window(root, now_ns=10_000_000_000)
    assert after["fleet"]["games"] == 4
    assert after["fleet"]["samples"] == 200


def test_truncated_publication_history_uses_only_known_counter_difference(tmp_path):
    root = records(
        tmp_path,
        [publication(110_000_000_000, 10, 100), publication(120_000_000_000, 13, 130)],
    )
    result = monitor._actor_throughput_window(
        root,
        now_ns=120_000_000_000,
        window_seconds=30,
    )
    assert result["fleet"]["samples"] == 30
    assert result["partial_processes"] == ["actor-gpu-7-cohort-0"]


def test_physical_batches_are_distinct_from_dispatches_and_require_current_pid(
    tmp_path,
):
    heartbeat = tmp_path / "actor.json"
    raw = {
        "pid": 7,
        "heartbeat_ns": 10_000_000_000,
        "inference": {
            "neural_batches": 100,
            "requested_rows": 400,
            "physical_inference": {
                "neural_calls": 2,
                "neural_rows": 256,
                "neural_padding_rows": 56,
                "cache_hits": 20,
                "cache_misses": 80,
            },
        },
    }
    heartbeat.write_text(json.dumps(raw))
    workers = {
        "actor-gpu-7": {
            "role": "actor",
            "gpu_ids": [7],
            "pid": 7,
            "state": "running",
            "heartbeat": str(heartbeat),
        }
    }
    result = monitor._actor_physical_inference(workers, now_ns=11_000_000_000)
    gpu = result["by_gpu"]["7"]
    assert gpu["broker_dispatches"] == 100
    assert gpu["mean_neural_batch_rows"] == 128
    assert gpu["padding_fraction"] == 56 / 256
    assert gpu["prediction_cache_hit_fraction"] == 0.2
    workers["actor-gpu-7"]["pid"] = 8
    assert not monitor._actor_physical_inference(workers, now_ns=11_000_000_000)[
        "available"
    ]
