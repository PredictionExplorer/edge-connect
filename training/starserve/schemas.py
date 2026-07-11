"""Versioned wire schemas for full-strength Double *Star analysis."""

from __future__ import annotations

import re
import math
from typing import Annotated, Literal

import torch
from pydantic import BaseModel, ConfigDict, Field, model_validator

from startrain.contracts import RULES_HASH_WIRE, SCORE_MARGIN_MAX, SCORE_MARGIN_MIN
from startrain.features import DoubleStarPosition
from startrain.topology import get_topology

StrictRing = Annotated[int, Field(strict=True)]
StrictPlayer = Annotated[int, Field(strict=True, ge=0, le=1)]
StrictMoves = Annotated[int, Field(strict=True, ge=0, le=2)]
StrictNode = Annotated[int, Field(strict=True, ge=0)]
StrictSeed = Annotated[int, Field(strict=True, ge=0, le=(1 << 64) - 1)]
PositiveInt = Annotated[int, Field(strict=True, gt=0)]
NonnegativeInt = Annotated[int, Field(strict=True, ge=0)]
Probability = Annotated[float, Field(strict=True, ge=0.0, le=1.0)]
UnitValue = Annotated[float, Field(strict=True, ge=-1.0, le=1.0)]
NonnegativeFinite = Annotated[float, Field(strict=True, ge=0.0)]
ScoreMarginExpectation = Annotated[
    float,
    Field(strict=True, ge=SCORE_MARGIN_MIN, le=SCORE_MARGIN_MAX),
]
RulesHash = Annotated[
    str,
    Field(strict=True, pattern=f"^{re.escape(RULES_HASH_WIRE)}$"),
]
ScoreSupportMin = Annotated[
    int,
    Field(strict=True, ge=SCORE_MARGIN_MIN, le=SCORE_MARGIN_MIN),
]
ScoreSupportMax = Annotated[
    int,
    Field(strict=True, ge=SCORE_MARGIN_MAX, le=SCORE_MARGIN_MAX),
]


class SearchBudget(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)

    simulations: PositiveInt
    max_considered: PositiveInt
    seed: StrictSeed


class AnalyzeRequest(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)

    schema_version: Literal[2]
    rules_hash: RulesHash
    rings: StrictRing
    stones: list[Literal[-1, 0, 1]]
    to_move: StrictPlayer
    moves_left: StrictMoves
    opening: bool
    terminal: Literal[False]
    search: SearchBudget

    @model_validator(mode="after")
    def validate_semantic_state(self) -> "AnalyzeRequest":
        if any(type(stone) is not int for stone in self.stones):
            raise ValueError("stones must contain strict integers")
        nodes = get_topology(self.rings).n
        if len(self.stones) != nodes:
            raise ValueError(f"stones must contain exactly {nodes} entries")
        try:
            DoubleStarPosition(
                rings=self.rings,
                stones=torch.tensor(self.stones, dtype=torch.int8),
                to_move=self.to_move,
                moves_left=self.moves_left,
                opening=self.opening,
                terminal=self.terminal,
            )
        except (TypeError, ValueError) as exc:
            raise ValueError(f"invalid no-pie Double *Star state: {exc}") from exc
        return self


class AtomicAction(BaseModel):
    model_config = ConfigDict(extra="forbid")

    code: StrictNode
    kind: Literal["place"]
    node: StrictNode

    @model_validator(mode="after")
    def validate_node_code(self) -> "AtomicAction":
        if self.code != self.node:
            raise ValueError("placement code and node must match")
        return self


class OutcomeBelief(BaseModel):
    model_config = ConfigDict(extra="forbid")

    loss: Probability
    win: Probability

    @model_validator(mode="after")
    def validate_probability_mass(self) -> "OutcomeBelief":
        if not math.isclose(self.loss + self.win, 1.0, rel_tol=1e-6, abs_tol=1e-6):
            raise ValueError("binary outcome probabilities must sum to one")
        return self


class ScoreBelief(BaseModel):
    model_config = ConfigDict(extra="forbid")

    support_min: ScoreSupportMin
    support_max: ScoreSupportMax
    expected_margin: ScoreMarginExpectation
    probabilities: list[Probability]

    @model_validator(mode="after")
    def validate_score_belief(self) -> "ScoreBelief":
        expected_bins = SCORE_MARGIN_MAX - SCORE_MARGIN_MIN + 1
        if len(self.probabilities) != expected_bins:
            raise ValueError(f"score probabilities must contain {expected_bins} bins")
        if not math.isfinite(self.expected_margin) or not (
            SCORE_MARGIN_MIN <= self.expected_margin <= SCORE_MARGIN_MAX
        ):
            raise ValueError("expected score margin is outside its support")
        if not math.isclose(sum(self.probabilities), 1.0, rel_tol=1e-6, abs_tol=1e-6):
            raise ValueError("score probabilities must sum to one")
        return self


class Timing(BaseModel):
    model_config = ConfigDict(extra="forbid")

    queue: NonnegativeFinite
    model_reload: NonnegativeFinite
    inference_search: NonnegativeFinite
    total: NonnegativeFinite


class AnalyzeResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")

    schema_version: Literal[2]
    request_id: str
    action: AtomicAction
    root_actions: list[AtomicAction]
    root_policy: list[Probability]
    root_q: list[UnitValue]
    root_visits: list[NonnegativeInt]
    outcome: OutcomeBelief
    value: UnitValue
    search_value: UnitValue
    score_belief: ScoreBelief
    model_version: str
    model_step: NonnegativeInt
    timing_ms: Timing

    @model_validator(mode="after")
    def validate_response_shapes(self) -> "AnalyzeResponse":
        width = len(self.root_actions)
        if width == 0 or not (
            len(self.root_policy) == len(self.root_q) == len(self.root_visits) == width
        ):
            raise ValueError("root action statistics have inconsistent shapes")
        if self.action not in self.root_actions:
            raise ValueError("selected action must appear in root_actions")
        if any(visit < 0 for visit in self.root_visits):
            raise ValueError("root visits must be non-negative")
        if not math.isclose(sum(self.root_policy), 1.0, rel_tol=1e-5, abs_tol=1e-5):
            raise ValueError("root policy must sum to one")
        expected_value = self.outcome.win - self.outcome.loss
        if not math.isclose(self.value, expected_value, rel_tol=1e-6, abs_tol=1e-6):
            raise ValueError("value must equal P(win)-P(loss)")
        return self
