#!/usr/bin/env python
"""Export schema-v4 feature vectors for positions of the conformance fixture.

The browser encoder (``src/lib/star/ai/features.ts``) must reproduce
``startrain.features.encode_position`` bit for bit, otherwise the published
ONNX model sees inputs it was never trained on. This script replays the
rules-v3 conformance games in Python, encodes a fixed set of positions, and
writes ``testdata/star/features-v4.json``; ``tests/test_feature_fixture.py``
pins the checked-in file and ``features.test.ts`` compares against it.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "training"))

from startrain.contracts import (  # noqa: E402
    EXTERNAL_FEATURE_SCHEMA_ID,
    FEATURE_SCHEMA_HASH,
    FEATURE_SCHEMA_VERSION,
    RULES_HASH_WIRE,
)
from startrain.features import (  # noqa: E402
    GLOBAL_FEATURE_NAMES,
    NODE_FEATURE_NAMES,
    DoubleStarPosition,
    encode_position,
)

CONFORMANCE_PATH = ROOT / "testdata" / "star" / "conformance-v3.json"
FIXTURE_PATH = ROOT / "testdata" / "star" / "features-v4.json"
FIXTURE_SCHEMA = "edgeconnect.star.model-features.fixture.v4"
# State indices sampled from every conformance game: the opening, an
# early mid-game position, and a late position (clamped to the trace).
SAMPLE_INDICES = (0, 1, 3, 9)


def _mask(nodes: list[int], count: int) -> torch.Tensor:
    mask = torch.zeros(count, dtype=torch.bool)
    for node in nodes:
        mask[node] = True
    return mask


def position_from_fixture(
    config: dict[str, Any], state: dict[str, Any]
) -> DoubleStarPosition:
    stones = torch.tensor(state["stones"], dtype=torch.int8)
    count = stones.numel()
    return DoubleStarPosition(
        rings=int(config["rings"]),
        stones=stones,
        to_move=int(state["toMove"]),
        moves_left=int(state["movesLeft"]),
        opening=bool(state["opening"]),
        terminal=bool(state["over"]),
        mode=str(config["mode"]),
        handicap=int(config.get("handicap", 1)),
        pie=bool(config["pieRule"]),
        swap_available=bool(state["canSwap"]),
        swapped=bool(state["swapped"]),
        current_turn=_mask(state["currentTurnMoves"], count),
        previous_turn=_mask(state["previousTurnMoves"], count),
        own_previous_turn=_mask(state["ownPreviousTurnMoves"], count),
        handicap_stones=_mask(state["handicapStones"], count),
        history_known=True,
        pda=0,
    )


def build_fixture() -> dict[str, Any]:
    conformance = json.loads(CONFORMANCE_PATH.read_text(encoding="utf-8"))
    positions: list[dict[str, Any]] = []
    for game in conformance["games"]:
        states = game["states"]
        chosen = sorted({min(index, len(states) - 1) for index in SAMPLE_INDICES})
        for index in chosen:
            state = states[index]
            position = position_from_fixture(game["config"], state)
            encoded = encode_position(position)
            positions.append(
                {
                    "game": game["id"],
                    "stateIndex": index,
                    "rings": position.rings,
                    "nodeFeatures": [
                        round(value, 7)
                        for value in encoded.node_features.flatten().tolist()
                    ],
                    "globalFeatures": [
                        round(value, 7) for value in encoded.global_features.tolist()
                    ],
                    "legalActionMask": [
                        int(value) for value in encoded.legal_node_mask.tolist()
                    ],
                }
            )
    return {
        "schema": FIXTURE_SCHEMA,
        "featureSchema": EXTERNAL_FEATURE_SCHEMA_ID,
        "featureSchemaVersion": FEATURE_SCHEMA_VERSION,
        "featureSchemaHash": f"{FEATURE_SCHEMA_HASH:016x}",
        "rulesHash": RULES_HASH_WIRE,
        "conformance": conformance["schema"],
        "nodeFeatureNames": list(NODE_FEATURE_NAMES),
        "globalFeatureNames": list(GLOBAL_FEATURE_NAMES),
        "context": {"historyKnown": True, "pda": 0},
        "positions": positions,
    }


def serialize(fixture: dict[str, Any]) -> str:
    return json.dumps(fixture, separators=(",", ":"), sort_keys=True) + "\n"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--check", action="store_true", help="fail if the file is stale"
    )
    args = parser.parse_args(argv)
    payload = serialize(build_fixture())
    if args.check:
        if (
            not FIXTURE_PATH.exists()
            or FIXTURE_PATH.read_text(encoding="utf-8") != payload
        ):
            print(f"{FIXTURE_PATH} is stale; rerun without --check", file=sys.stderr)
            return 1
        return 0
    FIXTURE_PATH.write_text(payload, encoding="utf-8")
    print(f"wrote {FIXTURE_PATH}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
