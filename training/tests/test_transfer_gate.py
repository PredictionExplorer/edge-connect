from dataclasses import asdict
import sqlite3

import pytest

from startrain.arena import ArenaPair
from startrain.contracts import FEATURE_SCHEMA_HASH, RULES_HASH
from startrain.transfer_gate import assess_lineage_result, recent_teacher_counts


def test_teacher_exclusion_tracks_recency_and_full_eligibility_not_step_zero():
    database = sqlite3.connect(":memory:")
    database.execute(
        "CREATE TABLE shards(id INTEGER PRIMARY KEY, sample_count INTEGER, actor_id TEXT, state TEXT, run_id TEXT, generation_family TEXT, ring INTEGER, rules_hash TEXT, feature_schema_hash TEXT, model_step INTEGER)"
    )

    def insert(ring, count, actor, step=0, run="run"):
        database.execute(
            "INSERT INTO shards(sample_count,actor_id,state,run_id,generation_family,ring,rules_hash,feature_schema_hash,model_step) VALUES(?,?,?,?,?,?,?,?,?)",
            (
                count,
                actor,
                "ready",
                run,
                "family",
                ring,
                f"{RULES_HASH:016x}",
                f"{FEATURE_SCHEMA_HASH:016x}",
                step,
            ),
        )

    for ring in (4, 6, 8, 10):
        insert(ring, 100, "lineage-transfer")
        insert(ring, 30, "actor", 0)
        insert(ring, 100, "actor", 150)  # Future model must not make a gate pass.
        insert(ring, 100, "actor", 10, "foreign")

    def inspect():
        return recent_teacher_counts(
            database,
            run_id="run",
            generation_family="family",
            step=100,
            lag=120,
            per_ring_quota=50,
            teacher_actor="lineage-transfer",
        )

    assert all(
        row == {"selected_samples": 50, "teacher_samples": 20}
        for row in inspect().values()
    )
    for ring in (4, 6, 8, 10):
        insert(
            ring, 20, "actor", 0
        )  # Newly generated step-zero history is not a teacher.
    assert all(
        row == {"selected_samples": 50, "teacher_samples": 0}
        for row in inspect().values()
    )
    database.close()


def result(outcomes=(1, 1)):
    return {
        "result_kind": "lineage_crossplay",
        "evaluation_mode": "cross_schema",
        "candidate": "candidate",
        "baseline": "teacher",
        "interrupted": False,
        "search": {
            "simulations": 1024,
            "seed_stream_policy": "independent-cell-pair-seat-move-v2",
            "max_considered": 32,
            "c_visit": 50.0,
            "c_scale": 1.0,
        },
        "baseline_metadata": {
            "search_budget": {
                "simulations": 1024,
                "max_considered": 32,
                "c_visit": 50.0,
                "c_scale": 1.0,
            }
        },
        "pairs": [
            asdict(
                ArenaPair(
                    ring=ring,
                    pair=index,
                    opening_seed=index,
                    opening_action=None,
                    forced_opening=False,
                    outcomes=outcomes,
                )
            )
            for ring in (4, 6, 8, 10)
            for index in range(15)
        ],
    }


def assess(value):
    return assess_lineage_result(
        value, candidate_identity="candidate", teacher_identity="teacher"
    )


def test_readiness_recomputes_actual_paired_outcomes():
    assert assess(result())["passed"] is True
    losses = result((-1, -1))
    losses["aggregate"] = {
        "elo_difference": 9000
    }  # Untrusted summaries cannot override games.
    assert assess(losses)["passed"] is False


@pytest.mark.parametrize(
    "mutation",
    [
        "duplicate",
        "partial",
        "wrong_model",
        "interrupted",
        "wrong_budget",
        "baseline_budget",
    ],
)
def test_readiness_rejects_incompatible_or_partial_evidence(mutation):
    evidence = result()
    if mutation == "duplicate":
        evidence["pairs"].append(evidence["pairs"][0])
    elif mutation == "partial":
        evidence["pairs"].pop()
    elif mutation == "wrong_model":
        evidence["candidate"] = "stale"
    elif mutation == "interrupted":
        evidence["interrupted"] = True
    elif mutation == "baseline_budget":
        evidence["baseline_metadata"]["search_budget"]["max_considered"] = 16
    else:
        evidence["search"]["simulations"] = 128
    with pytest.raises(ValueError):
        assess(evidence)
