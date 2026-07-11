"""Frozen deterministic non-neural opponents for internal Elo anchoring."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any, Final, Literal

import numpy as np

from .arena import ArenaSearchBudget
from .inference import InferenceResponse, NativeEvalBatchProtocol
from .native import BITBOARD_WORDS, NativeStateDataProtocol
from .topology import StarTopology, get_topology

_GREEDY_LOGIT_GAP: Final = 32.0
_EVALUATOR_VERSION: Final = "native-static-score-greedy-v1"


@dataclass(frozen=True, slots=True)
class FrozenBaselineDefinition:
    """Versioned identity and search contract for one frozen opponent."""

    name: str
    identity: str
    evaluator: Literal["uniform", "greedy"]
    algorithm: str
    search_budget: ArenaSearchBudget

    def metadata(self) -> dict[str, object]:
        return {
            "schema_version": 1,
            "kind": "frozen_non_human",
            "name": self.name,
            "identity": self.identity,
            "frozen": True,
            "non_human": True,
            "algorithm": self.algorithm,
            "evaluator": (
                "uniform-zero-v1" if self.evaluator == "uniform" else _EVALUATOR_VERSION
            ),
            "search_budget": self.search_budget.metadata(),
        }


_UNIFORM = FrozenBaselineDefinition(
    name="uniform",
    identity="frozen-uniform-random-v1-s1-k1-cv50-cs1",
    evaluator="uniform",
    algorithm="deterministic-seeded-native-search-over-uniform-policy",
    search_budget=ArenaSearchBudget(1, 1, 50.0, 1.0),
)
_GREEDY = FrozenBaselineDefinition(
    name="greedy",
    identity="frozen-greedy-native-score-v1-s1-k1-cv50-cs1",
    evaluator="greedy",
    algorithm="native-one-ply-static-score-greedy",
    search_budget=ArenaSearchBudget(1, 1, 50.0, 1.0),
)
_SHALLOW_SEARCH = FrozenBaselineDefinition(
    name="shallow-search",
    identity="frozen-shallow-native-score-v1-s64-k16-cv50-cs1",
    evaluator="greedy",
    algorithm="native-gumbel-mcts-with-one-ply-static-score-heuristic",
    search_budget=ArenaSearchBudget(64, 16, 50.0, 1.0),
)
_DEFINITIONS: Final = {
    definition.name: definition for definition in (_UNIFORM, _GREEDY, _SHALLOW_SEARCH)
}
_ALIASES: Final = {
    "random": "uniform",
    "shallow": "shallow-search",
}
FROZEN_BASELINE_CHOICES: Final = tuple((*_DEFINITIONS, *_ALIASES))


class UniformRandomBaseline:
    """Zero-value, uniform-policy evaluator; seeded native Gumbel makes it random-like."""

    def __init__(self, definition: FrozenBaselineDefinition = _UNIFORM) -> None:
        if definition.evaluator != "uniform":
            raise ValueError("uniform baseline requires a uniform definition")
        self.definition = definition
        self.model_version = definition.identity
        self.model_identity = definition.identity
        self.search_budget = definition.search_budget
        self.evaluator_calls = 0
        self.evaluator_rows = 0

    def result_metadata(self) -> dict[str, object]:
        return self.definition.metadata()

    def evaluate(self, requests: NativeEvalBatchProtocol) -> InferenceResponse:
        tokens, offsets, actions = _request_layout(requests)
        self.evaluator_calls += 1
        self.evaluator_rows += len(tokens)
        return InferenceResponse(
            tokens=tokens,
            values=[0.0] * len(tokens),
            policy_offsets=offsets,
            policy_logits=[0.0] * len(actions),
        )


class NativeGreedyBaseline:
    """Uses native transitions and scoring to rank every legal one-ply child."""

    def __init__(
        self,
        native_module: Any,
        definition: FrozenBaselineDefinition = _GREEDY,
    ) -> None:
        if definition.evaluator != "greedy":
            raise ValueError("greedy baseline requires a greedy definition")
        _require_native_baseline_api(native_module)
        self.native = native_module
        self.definition = definition
        self.model_version = definition.identity
        self.model_identity = definition.identity
        self.search_budget = definition.search_budget
        self.evaluator_calls = 0
        self.evaluator_rows = 0

    def result_metadata(self) -> dict[str, object]:
        return self.definition.metadata()

    def evaluate(self, requests: NativeEvalBatchProtocol) -> InferenceResponse:
        tokens, offsets, actions = _request_layout(requests)
        self.evaluator_calls += 1
        self.evaluator_rows += len(tokens)
        rows = len(tokens)
        if rows == 0:
            return InferenceResponse([], [], [0], [])

        states = requests.states
        to_move = _state_integers("to_move", states.to_move, rows)
        root_components = _request_score_components(
            self.native,
            requests,
            rows,
        )
        values = [
            _static_value(root_components[row], to_move[row]) for row in range(rows)
        ]

        parent_rows = [
            row for row in range(rows) for _ in range(offsets[row], offsets[row + 1])
        ]
        child_batch = _semantic_batch(self.native, states, parent_rows)
        child_batch.apply_many(list(range(len(actions))), actions)
        child_components = _component_rows(
            child_batch.score_data().components,
            len(actions),
        )
        child_data = child_batch.data()
        terminal = _state_booleans("terminal", child_data.terminal, len(actions))
        topology = get_topology(int(states.rings))

        logits = [0.0] * len(actions)
        for row in range(rows):
            start, end = offsets[row], offsets[row + 1]
            ranked = sorted(
                range(start, end),
                key=lambda index: _greedy_action_key(
                    components=child_components[index],
                    terminal=terminal[index],
                    actor=to_move[row],
                    action=actions[index],
                    topology=topology,
                    states=states,
                    parent_row=row,
                ),
            )
            for rank, index in enumerate(ranked):
                # Native deterministic Gumbels have a range below this gap, so a
                # one-candidate search is exact greedy rather than merely likely.
                logits[index] = rank * _GREEDY_LOGIT_GAP

        return InferenceResponse(
            tokens=tokens,
            values=values,
            policy_offsets=offsets,
            policy_logits=logits,
        )


def create_frozen_baseline(
    name: str,
    *,
    native_module: Any,
) -> UniformRandomBaseline | NativeGreedyBaseline:
    """Build a frozen baseline by canonical name or stable CLI alias."""

    canonical = _ALIASES.get(name, name)
    try:
        definition = _DEFINITIONS[canonical]
    except KeyError:
        choices = ", ".join(FROZEN_BASELINE_CHOICES)
        raise ValueError(
            f"unknown frozen baseline {name!r}; choose one of: {choices}"
        ) from None
    if definition.evaluator == "uniform":
        return UniformRandomBaseline(definition)
    return NativeGreedyBaseline(native_module, definition)


def _request_layout(
    requests: NativeEvalBatchProtocol,
) -> tuple[list[int], list[int], list[int]]:
    tokens = [int(value) for value in requests.tokens]
    offsets = [int(value) for value in requests.legal_offsets]
    actions = [int(value) for value in requests.legal_actions]
    rows = len(tokens)
    if len(requests) != rows:
        raise ValueError("baseline request length and token count disagree")
    if (
        len(offsets) != rows + 1
        or not offsets
        or offsets[0] != 0
        or offsets[-1] != len(actions)
        or any(left >= right for left, right in zip(offsets[:-1], offsets[1:]))
    ):
        if rows == 0 and offsets == [0] and not actions:
            return tokens, offsets, actions
        raise ValueError("baseline legal action CSR offsets are invalid")
    return tokens, offsets, actions


def _require_native_baseline_api(native_module: Any) -> None:
    state_batch = getattr(native_module, "StateBatch", None)
    required = ("from_semantic", "apply_many", "data", "score_data")
    missing = [
        name for name in required if not callable(getattr(state_batch, name, None))
    ]
    if missing:
        methods = ", ".join(f"StateBatch.{name}()" for name in missing)
        raise RuntimeError(f"native greedy baselines require {methods}")


def _state_integers(name: str, values: Sequence[int], rows: int) -> list[int]:
    output = [int(value) for value in values]
    if len(output) != rows:
        raise ValueError(f"baseline state {name} row count is invalid")
    return output


def _state_booleans(name: str, values: Sequence[bool], rows: int) -> list[bool]:
    output = [bool(value) for value in values]
    if len(output) != rows:
        raise ValueError(f"baseline state {name} row count is invalid")
    return output


def _semantic_batch(
    native_module: Any,
    states: NativeStateDataProtocol,
    parent_rows: Sequence[int],
) -> Any:
    rows = int(states.batch_size)

    def duplicate_words(name: str) -> list[int]:
        source = [int(value) for value in getattr(states, name)]
        if len(source) != rows * BITBOARD_WORDS:
            raise ValueError(f"baseline state {name} buffer is invalid")
        output = []
        for row in parent_rows:
            start = row * BITBOARD_WORDS
            output.extend(source[start : start + BITBOARD_WORDS])
        return output

    to_move = _state_integers("to_move", states.to_move, rows)
    moves_left = _state_integers("moves_left", states.moves_left, rows)
    opening = _state_booleans("opening", states.opening, rows)
    pass_streak = _state_integers("pass_streak", states.pass_streak, rows)
    return native_module.StateBatch.from_semantic(
        int(states.rings),
        duplicate_words("zero_bits"),
        duplicate_words("one_bits"),
        [to_move[row] for row in parent_rows],
        [moves_left[row] for row in parent_rows],
        [opening[row] for row in parent_rows],
        [pass_streak[row] for row in parent_rows],
    )


def _request_score_components(
    native_module: Any,
    requests: NativeEvalBatchProtocol,
    rows: int,
) -> list[tuple[int, ...]]:
    features = getattr(requests, "features", None)
    raw_components = getattr(features, "score_components", None)
    if raw_components is not None:
        return _component_rows(raw_components, rows)
    root_batch = _semantic_batch(native_module, requests.states, list(range(rows)))
    return _component_rows(root_batch.score_data().components, rows)


def _component_rows(values: Any, rows: int) -> list[tuple[int, ...]]:
    if isinstance(values, (bytes, bytearray, memoryview)):
        array = np.frombuffer(values, dtype=np.int32)
    else:
        array = np.asarray(values, dtype=np.int32)
    if array.size != rows * 14:
        raise ValueError("baseline native score component buffer is invalid")
    matrix = array.reshape(rows, 14)
    return [tuple(int(value) for value in matrix[row]) for row in range(rows)]


def _static_value(components: Sequence[int], actor: int) -> float:
    opponent = 1 - actor
    total_margin = components[actor * 6 + 5] - components[opponent * 6 + 5]
    quark_margin = components[actor * 6 + 1] - components[opponent * 6 + 1]
    value = (4.0 * total_margin + quark_margin) / 64.0
    return max(-1.0, min(1.0, value))


def _greedy_action_key(
    *,
    components: Sequence[int],
    terminal: bool,
    actor: int,
    action: int,
    topology: StarTopology,
    states: NativeStateDataProtocol,
    parent_row: int,
) -> tuple[int, ...]:
    opponent = 1 - actor
    leader = int(components[13])
    decisive = (
        2
        if terminal and leader == actor
        else 0
        if terminal and leader == opponent
        else 1
    )
    total_margin = components[actor * 6 + 5] - components[opponent * 6 + 5]
    quark_margin = components[actor * 6 + 1] - components[opponent * 6 + 1]
    peri_margin = components[actor * 6] - components[opponent * 6]
    if action == -1:
        return (
            decisive,
            total_margin,
            quark_margin,
            peri_margin,
            0,
            0,
            0,
            0,
            0,
            0,
        )
    if not 0 <= action < topology.n:
        raise ValueError("baseline request contains an invalid native action")

    offsets = topology.adjacency_offsets
    adjacency = topology.adjacency
    friendly = 0
    enemy = 0
    for edge in range(int(offsets[action]), int(offsets[action + 1])):
        neighbor = int(adjacency[edge])
        friendly += int(_stone_present(states, actor, parent_row, neighbor))
        enemy += int(_stone_present(states, opponent, parent_row, neighbor))
    return (
        decisive,
        total_margin,
        quark_margin,
        peri_margin,
        friendly,
        enemy,
        int(topology.is_quark[action]),
        int(topology.is_peri[action]),
        int(offsets[action + 1] - offsets[action]),
        -action,
    )


def _stone_present(
    states: NativeStateDataProtocol,
    player: int,
    row: int,
    node: int,
) -> bool:
    words = states.zero_bits if player == 0 else states.one_bits
    word = int(words[row * BITBOARD_WORDS + node // 64])
    return bool(word & (1 << (node % 64)))
