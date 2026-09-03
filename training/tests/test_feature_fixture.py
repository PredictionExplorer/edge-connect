"""The checked-in v4 feature fixture must match the Python encoder exactly."""

from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

import export_feature_fixture  # noqa: E402


def test_checked_in_feature_fixture_matches_the_encoder() -> None:
    fixture = export_feature_fixture.build_fixture()
    assert fixture["featureSchemaVersion"] == 4
    assert len(fixture["positions"]) >= 30
    games = {position["game"] for position in fixture["positions"]}
    assert "rings-4-pie-double-swap-board-full" in games
    assert "rings-6-handicap-9-classic-board-full" in games
    checked_in = json.loads(
        export_feature_fixture.FIXTURE_PATH.read_text(encoding="utf-8")
    )
    assert checked_in == json.loads(export_feature_fixture.serialize(fixture))
    for position in fixture["positions"]:
        node_count = len(position["legalActionMask"])
        assert len(position["nodeFeatures"]) == node_count * 19
        assert len(position["globalFeatures"]) == 25
