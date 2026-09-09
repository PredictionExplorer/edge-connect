from startrain import actor_publication as publication


def test_publications_are_rate_limited_and_final_flush_preserves_all_counters(
    monkeypatch,
):
    clock = {"now": 10.0, "rows": 20}
    records = []
    heartbeats = []
    monkeypatch.setattr(publication.time, "monotonic", lambda: clock["now"])
    callback = publication.PublicationProgress(
        metadata={"worker": "cohort", "process_started_ns": 1},
        base_games=5,
        base_samples=50,
        base_evaluator_rows=100,
        base_wall_seconds=20,
        task_started=10,
        evaluator_rows=lambda: clock["rows"],
        heartbeat=lambda **fields: heartbeats.append(fields),
        emit=records.append,
    )
    callback.progress(
        phase="selfplay_completed", completed_games=1, persisted_decisions=10
    )
    clock.update(now=10.1, rows=40)
    callback.progress(
        phase="selfplay_completed", completed_games=2, persisted_decisions=30
    )
    callback.progress(
        phase="selfplay_refill",
        started_games=4,
        completed_games=2,
        persisted_decisions=30,
    )
    assert len(records) == 1
    assert (
        heartbeats[-1]["cumulative_games"] == 7
        and heartbeats[-1]["cumulative_samples"] == 80
    )
    callback.finish()
    callback.finish()
    assert len(records) == 2
    assert [r["published_games"] for r in records] == [1, 1]
    assert [r["published_samples"] for r in records] == [10, 20]
    assert records[-1]["cumulative_evaluator_rows"] == 140
    assert records[-1]["cumulative_batch_wall_seconds"] == 20.1
    assert all("games" not in row and "samples" not in row for row in records)


def test_no_completed_games_means_no_publication_record():
    records = []
    callback = publication.PublicationProgress(
        metadata={},
        base_games=0,
        base_samples=0,
        base_evaluator_rows=0,
        base_wall_seconds=0,
        task_started=0,
        evaluator_rows=lambda: 7,
        heartbeat=lambda **_: None,
        emit=records.append,
    )
    callback.progress(
        phase="selfplay_refill",
        started_games=4,
        completed_games=0,
        persisted_decisions=0,
    )
    callback.finish()
    assert records == []
