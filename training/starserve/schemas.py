"""Versioned wire schemas for full-strength Double *Star analysis."""

from __future__ import annotations

from typing import Annotated, Literal

import torch
from pydantic import BaseModel, ConfigDict, Field, model_validator

from startrain.contracts import RULES_HASH_WIRE, SCORE_MARGIN_MAX, SCORE_MARGIN_MIN
from startrain.features import DoubleStarPosition
from startrain.topology import get_topology

StrictRing = Annotated[int, Field(strict=True, ge=3, le=12)]
StrictPlayer = Annotated[int, Field(strict=True, ge=0, le=1)]
StrictMoves = Annotated[int, Field(strict=True, ge=0, le=2)]
StrictSeed = Annotated[int, Field(strict=True, ge=0, le=(1 << 64) - 1)]
PositiveInt = Annotated[int, Field(strict=True, gt=0)]


class SearchBudget(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)

    simulations: PositiveInt
    max_considered: PositiveInt
    seed: StrictSeed


class AnalyzeRequest(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)

    schema_version: Literal[1]
    rules_hash: Literal[RULES_HASH_WIRE]
    rings: StrictRing
    stones: list[Literal[-1, 0, 1]]
    to_move: StrictPlayer
    moves_left: StrictMoves
    opening: bool
    pass_streak: StrictMoves
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
                pass_streak=self.pass_streak,
                terminal=self.terminal,
            )
        except (TypeError, ValueError) as exc:
            raise ValueError(f"invalid no-pie Double *Star state: {exc}") from exc
        return self


class AtomicAction(BaseModel):
    model_config = ConfigDict(extra="forbid")

    code: int
    kind: Literal["place", "pass"]
    node: int | None


class WDLBelief(BaseModel):
    model_config = ConfigDict(extra="forbid")

    loss: float
    draw: float
    win: float


class ScoreBelief(BaseModel):
    model_config = ConfigDict(extra="forbid")

    support_min: Literal[SCORE_MARGIN_MIN]
    support_max: Literal[SCORE_MARGIN_MAX]
    expected_margin: float
    probabilities: list[float]


class Timing(BaseModel):
    model_config = ConfigDict(extra="forbid")

    queue: float
    model_reload: float
    inference_search: float
    total: float


class AnalyzeResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")

    schema_version: Literal[1]
    request_id: str
    action: AtomicAction
    root_actions: list[AtomicAction]
    root_policy: list[float]
    root_q: list[float]
    root_visits: list[int]
    wdl: WDLBelief
    value: float
    search_value: float
    score_belief: ScoreBelief
    model_version: str
    model_step: int
    timing_ms: Timing
