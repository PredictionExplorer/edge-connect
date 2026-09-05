import json

from scripts.monitor_run import _strength_efficiency_status


def test_balanced_monitor_never_falls_back_to_rejected_or_standard_only_elo(tmp_path):
    identity = {"run_id": "variant", "generation_family": "family", "created_ns": 1}
    (tmp_path / "run.json").write_text(json.dumps(identity))
    report = {
        "schema_version": 1,
        "report": "startrain-strength-efficiency",
        "status": "complete",
        "run_id": "variant",
        "generation_family": "family",
        "run_root": str(tmp_path),
        "started_ns": 1,
        "observed_until_ns": 10,
        "autonomous_elo": {
            "headline_elo": 999,
            "headline": {"source": "aggregate", "rating": 999},
        },
        "balanced_strength": {"status": "missing", "available": False, "rating": None},
    }
    path = tmp_path / "strength-efficiency.json"
    path.write_text(json.dumps(report))
    balanced = _strength_efficiency_status(tmp_path, now_ns=10, balanced=True)
    assert balanced["available"] is True  # Report is valid, measurement is missing.
    assert balanced["headline_elo"] is None
    assert balanced["headline_source"] == "balanced_champion_frontier"
    legacy = _strength_efficiency_status(tmp_path, now_ns=10)
    assert legacy["headline_elo"] == 999
    report["balanced_strength"] = {
        "status": "measured",
        "available": True,
        "rating": 25,
        "confidence_interval": [5, 45],
    }
    path.write_text(json.dumps(report))
    measured = _strength_efficiency_status(tmp_path, now_ns=10, balanced=True)
    assert measured["headline_elo"] == 25
    assert measured["headline_confidence_interval"] == [5, 45]
