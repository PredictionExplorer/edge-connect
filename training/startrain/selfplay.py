"""Deterministic single-machine ring- and variant-homogeneous native self-play.

Every cohort plays one board size and one rule variant (mode, handicap, pie).
Inside a cohort each game may carry a playout-doubling advantage for one seat:
that seat searches with more simulations and both networks see the advantage
as an input, so the network learns to evaluate positions under a strength
asymmetry (the KataGo remedy for lopsided handicap games). In pie games the
responder swaps exactly when its root value is below a small dead zone.
"""

from __future__ import annotations

import hashlib
import math
import time
from collections.abc import Callable
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any, Literal, Protocol, Sequence

import numpy as np

from .contracts import (
    MAX_HANDICAP,
    MAX_PLAYOUT_DOUBLING_ADVANTAGE,
    MODES,
    OUTCOME_LOSS,
    OUTCOME_WIN,
    SEGMENT_CLASSIC,
    SEGMENT_HANDICAP,
    SEGMENT_PIE,
    SEGMENT_STANDARD,
    SEGMENTS,
)
from .features import variant_label, variant_segment
from .inference import InferenceResponse, NativeEvalBatchProtocol
from .native import (
    positions_from_native,
    score_results_from_native,
    trajectory_rows_from_native,
)
from .replay import ReplaySample
from .runtime import validate_identifier
from .topology import SUPPORTED_RINGS


class EvaluatorProtocol(Protocol):
    model_version: str
    model_step: int
    model_identity: str

    def evaluate(self, requests: NativeEvalBatchProtocol) -> InferenceResponse: ...


class ReplaySinkProtocol(Protocol):
    def append(
        self,
        samples: Sequence[ReplaySample],
        *,
        phase_min: int,
        phase_max: int,
        model_version: str,
        model_step: int,
        model_identity: str,
        run_id: str,
        generation_family: str,
        actor_id: str,
        generation: int,
    ) -> object: ...


@dataclass(frozen=True, slots=True)
class GameVariant:
    """One drawn rule variant for a self-play cohort or arena pair."""

    mode: str = "double"
    handicap: int = 1
    pie: bool = False

    def __post_init__(self) -> None:
        if type(self.mode) is not str or self.mode not in MODES:
            raise ValueError("variant mode must be classic or double")
        if (
            isinstance(self.handicap, bool)
            or not isinstance(self.handicap, int)
            or not 1 <= self.handicap <= MAX_HANDICAP
        ):
            raise ValueError(f"variant handicap must be in 1..{MAX_HANDICAP}")
        if type(self.pie) is not bool:
            raise ValueError("variant pie must be boolean")
        if self.pie and self.handicap != 1:
            raise ValueError("handicap games cannot use the pie rule")

    @property
    def label(self) -> str:
        return variant_label(self.mode, self.handicap, self.pie)

    @property
    def segment(self) -> str:
        return variant_segment(self.mode, self.handicap, self.pie)

    @property
    def is_standard(self) -> bool:
        return self.mode == "double" and self.handicap == 1 and not self.pie

    @classmethod
    def parse(cls, label: str) -> "GameVariant":
        """Inverse of :attr:`label`."""

        if label in MODES:
            return cls(mode=label)
        if label.startswith("pie-"):
            return cls(mode=label[len("pie-") :], pie=True)
        if label.startswith("handicap-"):
            _, count, mode = label.split("-", 2)
            return cls(mode=mode, handicap=int(count))
        raise ValueError(f"unknown variant label {label!r}")


STANDARD_VARIANT = GameVariant()


@dataclass(frozen=True, slots=True)
class VariantMixtureConfig:
    """Per-batch variant draw and the playout-doubling-advantage pairing.

    Fractions are the share of self-play batches in each mixture segment.
    Handicap games pair the handicap count with a playout-doubling advantage
    for the second player (``handicap_pda[k - 2]``) so the disadvantaged side
    still produces informative outcomes. ``asymmetric_pda_fraction`` of the
    standard and classic games give one random seat a random advantage from
    ``pda_magnitudes`` so the network learns the input everywhere.
    """

    enabled: bool = False
    standard: float = 0.45
    classic: float = 0.25
    handicap: float = 0.20
    pie: float = 0.10
    pie_classic_share: float = 0.3
    handicap_min: int = 2
    handicap_max: int = MAX_HANDICAP
    handicap_pda: tuple[int, ...] = (1, 1, 2, 2, 2, 3, 3, 3)
    asymmetric_pda_fraction: float = 0.2
    pda_magnitudes: tuple[int, ...] = (1, 2)
    swap_dead_zone: float = 0.02
    score_utility_weight_by_segment: dict[str, float] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if type(self.enabled) is not bool:
            raise ValueError("variants.enabled must be boolean")
        fractions = (self.standard, self.classic, self.handicap, self.pie)
        if any(
            isinstance(value, bool) or not isinstance(value, int | float)
            for value in fractions
        ) or any(not math.isfinite(value) or value < 0 for value in fractions):
            raise ValueError("variant fractions must be finite and non-negative")
        if self.enabled and not np.isclose(sum(fractions), 1.0):
            raise ValueError("variant fractions must sum to one")
        if not 0 <= self.pie_classic_share <= 1:
            raise ValueError("pie_classic_share must be in [0, 1]")
        for name in ("handicap_min", "handicap_max"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int):
                raise ValueError(f"variants.{name} must be an integer")
        if not 2 <= self.handicap_min <= self.handicap_max <= MAX_HANDICAP:
            raise ValueError(
                f"variant handicap range must satisfy 2 <= min <= max <= {MAX_HANDICAP}"
            )
        pdas = tuple(self.handicap_pda)
        if len(pdas) != MAX_HANDICAP - 1 or any(
            isinstance(value, bool)
            or not isinstance(value, int)
            or not 0 <= value <= MAX_PLAYOUT_DOUBLING_ADVANTAGE
            for value in pdas
        ):
            raise ValueError(
                "handicap_pda must give one advantage in 0..3 for each handicap 2..9"
            )
        object.__setattr__(self, "handicap_pda", pdas)
        if not 0 <= self.asymmetric_pda_fraction <= 1:
            raise ValueError("asymmetric_pda_fraction must be in [0, 1]")
        magnitudes = tuple(self.pda_magnitudes)
        if not magnitudes or any(
            isinstance(value, bool)
            or not isinstance(value, int)
            or not 1 <= value <= MAX_PLAYOUT_DOUBLING_ADVANTAGE
            for value in magnitudes
        ):
            raise ValueError("pda_magnitudes must be non-empty values in 1..3")
        object.__setattr__(self, "pda_magnitudes", magnitudes)
        if not 0 <= self.swap_dead_zone < 1:
            raise ValueError("swap_dead_zone must be in [0, 1)")
        weights = dict(self.score_utility_weight_by_segment)
        for segment, weight in weights.items():
            if segment not in SEGMENTS:
                raise ValueError(f"unknown score utility segment {segment!r}")
            if (
                isinstance(weight, bool)
                or not isinstance(weight, int | float)
                or not 0 <= float(weight) <= 1
            ):
                raise ValueError("score utility weights must be in [0, 1]")
        object.__setattr__(
            self,
            "score_utility_weight_by_segment",
            {segment: float(weight) for segment, weight in weights.items()},
        )

    @property
    def segment_fractions(self) -> dict[str, float]:
        if not self.enabled:
            return {SEGMENT_STANDARD: 1.0}
        return {
            SEGMENT_STANDARD: self.standard,
            SEGMENT_CLASSIC: self.classic,
            SEGMENT_HANDICAP: self.handicap,
            SEGMENT_PIE: self.pie,
        }

    def pda_for_handicap(self, handicap: int) -> int:
        if handicap < 2:
            return 0
        return self.handicap_pda[handicap - 2]

    def score_utility_weight(self, segment: str, default: float) -> float:
        return self.score_utility_weight_by_segment.get(segment, default)

    def draw(self, seed: int) -> GameVariant:
        """Deterministically draw one variant for a batch from a 64-bit seed."""

        if not self.enabled:
            return STANDARD_VARIANT
        unit = (seed & 0xFFFFFFFF) / float(1 << 32)
        secondary = ((seed >> 32) & 0xFFFFFFFF) / float(1 << 32)
        cumulative = 0.0
        for segment, fraction in self.segment_fractions.items():
            cumulative += fraction
            if unit < cumulative or segment == SEGMENT_PIE:
                break
        else:  # pragma: no cover - fractions sum to one
            segment = SEGMENT_STANDARD
        if segment == SEGMENT_STANDARD:
            return STANDARD_VARIANT
        if segment == SEGMENT_CLASSIC:
            return GameVariant(mode="classic")
        if segment == SEGMENT_HANDICAP:
            span = self.handicap_max - self.handicap_min + 1
            handicap = self.handicap_min + min(span - 1, int(secondary * span))
            return GameVariant(mode="double", handicap=handicap)
        mode = "classic" if secondary < self.pie_classic_share else "double"
        return GameVariant(mode=mode, pie=True)


@dataclass(frozen=True, slots=True)
class SelfPlayConfig:
    rings: int = 4
    batch_size: int = 1
    games: int = 1
    # The variant played by this cohort; the actor replaces these per batch
    # from ``variants.draw`` exactly like ``rings``.
    mode: str = "double"
    handicap: int = 1
    pie: bool = False
    variants: VariantMixtureConfig = VariantMixtureConfig()
    fast_probability: float = 0.75
    full_probability: float = 0.25
    fast_simulations: int = 8
    full_simulations: int = 64
    simulation_reference_rings: int = 6
    simulation_ring_exponent: float = 1.0
    max_considered: int = 16
    max_considered_ring_exponent: float = 0.0
    max_considered_cap: int = 64
    record_fast_policy_targets: bool = False
    fast_policy_weight: float = 0.25
    policy_surprise_weight: float = 0.0
    policy_surprise_max_weight: float = 4.0
    c_visit: float = 50.0
    c_scale: float = 1.0
    score_utility_weight: float = 0.0
    clinch_finalization: Literal["disabled", "loser-fill"] = "disabled"
    clinch_auxiliary_targets: Literal["synthetic", "outcome_only"] = "synthetic"
    shard_size: int = 512
    seed: int = 17

    def __post_init__(self) -> None:
        if type(self.rings) is not int or self.rings not in SUPPORTED_RINGS:
            raise ValueError("self-play rings must be one of (4, 6, 8, 10)")
        if self.batch_size <= 0 or self.games <= 0:
            raise ValueError("batch_size and games must be positive")
        if min(self.fast_probability, self.full_probability) < 0 or not np.isclose(
            self.fast_probability + self.full_probability, 1.0
        ):
            raise ValueError(
                "fast/full probabilities must be non-negative and sum to one"
            )
        if min(self.fast_simulations, self.full_simulations) <= 0:
            raise ValueError("playout caps must be positive")
        if (
            type(self.simulation_reference_rings) is not int
            or self.simulation_reference_rings not in SUPPORTED_RINGS
            or self.simulation_ring_exponent < 0
        ):
            raise ValueError("ring-count simulation scaling is invalid")
        if (
            self.max_considered <= 0
            or self.max_considered_cap < self.max_considered
            or self.max_considered_ring_exponent < 0
            or self.shard_size <= 0
        ):
            raise ValueError("candidate scaling and shard_size are invalid")
        if type(self.record_fast_policy_targets) is not bool:
            raise ValueError("record_fast_policy_targets must be boolean")
        if not 0 <= self.fast_policy_weight <= 1:
            raise ValueError("fast_policy_weight must be in [0, 1]")
        if (
            not 0 <= self.policy_surprise_weight <= 1
            or not math.isfinite(self.policy_surprise_max_weight)
            or self.policy_surprise_max_weight < 1
        ):
            raise ValueError("policy-surprise weighting settings are invalid")
        if not 0 <= self.score_utility_weight <= 1:
            raise ValueError("score utility weight must be in [0, 1]")
        if self.clinch_finalization not in ("disabled", "loser-fill"):
            raise ValueError("clinch_finalization must be disabled or loser-fill")
        if self.clinch_auxiliary_targets not in ("synthetic", "outcome_only"):
            raise ValueError(
                "clinch_auxiliary_targets must be synthetic or outcome_only"
            )
        GameVariant(mode=self.mode, handicap=self.handicap, pie=self.pie)
        if not isinstance(self.variants, VariantMixtureConfig):
            raise ValueError("variants must be a VariantMixtureConfig")

    @property
    def variant(self) -> GameVariant:
        return GameVariant(mode=self.mode, handicap=self.handicap, pie=self.pie)

    def with_variant(self, variant: GameVariant) -> "SelfPlayConfig":
        return replace(
            self, mode=variant.mode, handicap=variant.handicap, pie=variant.pie
        )

    def effective_score_utility_weight(self) -> float:
        return self.variants.score_utility_weight(
            self.variant.segment, self.score_utility_weight
        )

    def playout_budgets(self, *, simulations: int, pda: int) -> tuple[int, int]:
        """Return (advantaged, disadvantaged) budgets for a doubling advantage.

        The ratio is exactly ``2 ** pda`` while both budgets stay inside the
        configured fast/full playout caps, so a fast wave raises the advantaged
        side and a full wave lowers the disadvantaged side.
        """

        low = self.simulation_budget(full=False)
        high = self.simulation_budget(full=True)
        if pda <= 0:
            return simulations, simulations
        factor = 2**pda
        advantaged = int(min(high, max(low, simulations * factor)))
        disadvantaged = int(min(high, max(low, advantaged // factor)))
        return max(1, advantaged), max(1, disadvantaged)

    @classmethod
    def cpu_smoke(cls, *, seed: int = 17) -> "SelfPlayConfig":
        return cls(
            rings=4,
            batch_size=1,
            games=1,
            fast_probability=0.0,
            full_probability=1.0,
            fast_simulations=1,
            full_simulations=2,
            max_considered=2,
            shard_size=64,
            seed=seed,
        )

    def simulation_budget(self, *, full: bool) -> int:
        base = self.full_simulations if full else self.fast_simulations
        scale = (
            self.rings / self.simulation_reference_rings
        ) ** self.simulation_ring_exponent
        return max(1, int(round(base * scale)))

    def considered_actions(self) -> int:
        scale = max(
            1.0,
            (self.rings / self.simulation_reference_rings)
            ** self.max_considered_ring_exponent,
        )
        return min(
            self.max_considered_cap,
            max(1, int(round(self.max_considered * scale))),
        )


@dataclass(frozen=True, slots=True)
class GameSummary:
    row: int
    samples: int
    policy_samples: int
    search_simulations: int
    winner: int
    terminal_value: float
    score_margin: int
    turn_count: int
    last_move: int
    model_version: str
    model_identity: str
    game_id: str
    generation: int
    finish_reason: Literal["board-full", "clinch"]
    empty_nodes_saved: int
    variant: str = "double"
    swapped: bool = False
    pda_seat0: int = 0
    pda_seat1: int = 0


@dataclass(frozen=True, slots=True)
class SelfPlayMetrics:
    """Monotonic counters for one self-play actor lifetime.

    Decision-mode and entropy counters are recorded when a decision is made.
    ``completed_decisions`` and ``dropped_decisions`` partition attempts.
    """

    completed_decisions: int = 0
    full_decisions: int = 0
    fast_decisions: int = 0
    policy_entropy_count: int = 0
    policy_entropy_sum: float = 0.0
    policy_weight_sum: float = 0.0
    policy_surprise_count: int = 0
    policy_surprise_sum: float = 0.0
    sample_weight_sum: float = 0.0
    interrupted_cohorts: int = 0
    dropped_games: int = 0
    dropped_decisions: int = 0
    replay_append_calls: int = 0
    replay_append_bytes: int = 0
    replay_append_seconds: float = 0.0
    clinched_games: int = 0
    clinch_empty_nodes: int = 0
    source_champion_games: int = 0
    source_candidate_games: int = 0
    source_history_games: int = 0
    source_unattributed_games: int = 0
    source_champion_samples: int = 0
    source_candidate_samples: int = 0
    source_history_samples: int = 0
    source_unattributed_samples: int = 0
    pie_decisions: int = 0
    pie_swaps: int = 0
    asymmetric_games: int = 0

    def delta(self, previous: "SelfPlayMetrics") -> "SelfPlayMetrics":
        values = {
            field: getattr(self, field) - getattr(previous, field)
            for field in self.__dataclass_fields__
        }
        if any(value < 0 for value in values.values()):
            raise ValueError("self-play metrics counters must be monotonic")
        return SelfPlayMetrics(**values)


@dataclass(frozen=True, slots=True)
class SelfPlayIdentity:
    run_id: str
    generation_family: str
    actor_id: str
    generation: int

    def __post_init__(self) -> None:
        validate_identifier("run_id", self.run_id)
        validate_identifier("generation_family", self.generation_family)
        validate_identifier("actor_id", self.actor_id)
        if (
            isinstance(self.generation, bool)
            or not isinstance(self.generation, int)
            or self.generation < 0
        ):
            raise ValueError("generation must be a non-negative integer")


@dataclass(slots=True)
class _Decision:
    position: Any
    policy: np.ndarray | None
    full_search: bool
    simulations: int
    phase: int
    search_seed: int
    ply: int
    policy_weight: float
    policy_surprise: float
    swapped: bool = False


@dataclass(frozen=True, slots=True)
class _ClinchFinalization:
    winner: int
    empty_nodes: int
    last_move: int
    turn_count: int


class SelfPlayActor:
    """Drives PyO3 ``StateBatch``/``SearchBatch`` and emits replay shards."""

    def __init__(
        self,
        native_module: Any,
        evaluator: EvaluatorProtocol,
        replay_sink: ReplaySinkProtocol,
        config: SelfPlayConfig,
        identity: SelfPlayIdentity | None = None,
        *,
        source_role: Literal[
            "champion", "candidate", "history", "unattributed"
        ] = "unattributed",
    ) -> None:
        self.native = native_module
        self.evaluator = evaluator
        self.sink = replay_sink
        self.config = config
        self.identity = identity or SelfPlayIdentity("manual", "manual", "manual", 0)
        if source_role not in ("champion", "candidate", "history", "unattributed"):
            raise ValueError("self-play source_role is invalid")
        self.source_role = source_role
        self.model_identity = str(
            getattr(evaluator, "model_identity", evaluator.model_version)
        )
        validate_identifier("model_identity", self.model_identity)
        self.pending_samples: list[ReplaySample] = []
        self.pending_phases: list[int] = []
        self.persisted_decisions = 0
        self.completed_decisions = 0
        self.full_decisions = 0
        self.fast_decisions = 0
        self.policy_entropy_count = 0
        self.policy_entropy_sum = 0.0
        self.policy_weight_sum = 0.0
        self.policy_surprise_count = 0
        self.policy_surprise_sum = 0.0
        self.sample_weight_sum = 0.0
        self.interrupted_cohorts = 0
        self.dropped_games = 0
        self.dropped_decisions = 0
        self.replay_append_calls = 0
        self.replay_append_bytes = 0
        self.replay_append_seconds = 0.0
        self.clinched_games = 0
        self.clinch_empty_nodes = 0
        self.source_champion_games = 0
        self.source_candidate_games = 0
        self.source_history_games = 0
        self.source_unattributed_games = 0
        self.source_champion_samples = 0
        self.source_candidate_samples = 0
        self.source_history_samples = 0
        self.source_unattributed_samples = 0
        self.pie_decisions = 0
        self.pie_swaps = 0
        self.asymmetric_games = 0

    def metrics_snapshot(self) -> SelfPlayMetrics:
        return SelfPlayMetrics(
            completed_decisions=self.completed_decisions,
            full_decisions=self.full_decisions,
            fast_decisions=self.fast_decisions,
            policy_entropy_count=self.policy_entropy_count,
            policy_entropy_sum=self.policy_entropy_sum,
            policy_weight_sum=self.policy_weight_sum,
            policy_surprise_count=self.policy_surprise_count,
            policy_surprise_sum=self.policy_surprise_sum,
            sample_weight_sum=self.sample_weight_sum,
            interrupted_cohorts=self.interrupted_cohorts,
            dropped_games=self.dropped_games,
            dropped_decisions=self.dropped_decisions,
            replay_append_calls=self.replay_append_calls,
            replay_append_bytes=self.replay_append_bytes,
            replay_append_seconds=self.replay_append_seconds,
            clinched_games=self.clinched_games,
            clinch_empty_nodes=self.clinch_empty_nodes,
            source_champion_games=self.source_champion_games,
            source_candidate_games=self.source_candidate_games,
            source_history_games=self.source_history_games,
            source_unattributed_games=self.source_unattributed_games,
            source_champion_samples=self.source_champion_samples,
            source_candidate_samples=self.source_candidate_samples,
            source_history_samples=self.source_history_samples,
            source_unattributed_samples=self.source_unattributed_samples,
            pie_decisions=self.pie_decisions,
            pie_swaps=self.pie_swaps,
            asymmetric_games=self.asymmetric_games,
        )

    def run(
        self,
        *,
        stop_requested: Callable[[], bool] = lambda: False,
        progress: Callable[..., None] | None = None,
    ) -> list[GameSummary]:
        summaries: list[GameSummary] = []
        cohort = 0
        while len(summaries) < self.config.games:
            if stop_requested():
                break
            cohort_size = min(
                self.config.batch_size, self.config.games - len(summaries)
            )
            summaries.extend(
                self._run_cohort(
                    cohort_size,
                    cohort=cohort,
                    first_game=len(summaries),
                    stop_requested=stop_requested,
                    progress=progress,
                )
            )
            cohort += 1
            if progress is not None:
                progress(
                    phase="selfplay",
                    cohort=cohort,
                    completed_games=len(summaries),
                    requested_games=self.config.games,
                    persisted_decisions=self.persisted_decisions,
                )
        self._flush()
        completed_decisions = sum(summary.samples for summary in summaries)
        if len({summary.game_id for summary in summaries}) != len(summaries):
            raise RuntimeError("self-play generated duplicate game identifiers")
        if (
            self.pending_samples
            or self.persisted_decisions != completed_decisions
            or self.completed_decisions != completed_decisions
            or self.full_decisions + self.fast_decisions
            != completed_decisions + self.dropped_decisions
        ):
            raise RuntimeError(
                "completed-game and persisted-decision accounting disagree"
            )
        return summaries

    def _draw_pda_seats(self, cohort: int, cohort_size: int) -> list[tuple[int, int]]:
        """Playout-doubling advantages ``(seat 0, seat 1)`` for every game.

        Handicap games hand the second player the configured advantage; a
        configured fraction of other games hands a random seat a random
        magnitude. The disadvantaged seat sees the negated value.
        """

        variant = self.config.variant
        mixture = self.config.variants
        seats: list[tuple[int, int]] = []
        for row in range(cohort_size):
            if variant.handicap >= 2:
                advantage = mixture.pda_for_handicap(variant.handicap)
                seats.append((-advantage, advantage))
                continue
            if not mixture.enabled or mixture.asymmetric_pda_fraction <= 0:
                seats.append((0, 0))
                continue
            roll = self._seed("pda", cohort, row)
            unit = (roll & 0xFFFFFFFF) / float(1 << 32)
            if unit >= mixture.asymmetric_pda_fraction:
                seats.append((0, 0))
                continue
            choice = (roll >> 32) & 0xFFFFFFFF
            magnitude = mixture.pda_magnitudes[choice % len(mixture.pda_magnitudes)]
            if (choice >> 16) & 1:
                seats.append((magnitude, -magnitude))
            else:
                seats.append((-magnitude, magnitude))
        return seats

    def _root_budgets(
        self,
        state_data: Any,
        pda_seats: Sequence[tuple[int, int]],
        *,
        simulations: int,
    ) -> list[int]:
        budgets: list[int] = []
        to_move = [int(value) for value in state_data.to_move]
        for row, seats in enumerate(pda_seats):
            advantage = seats[to_move[row]]
            if advantage == 0:
                budgets.append(simulations)
                continue
            advantaged, disadvantaged = self.config.playout_budgets(
                simulations=simulations, pda=abs(advantage)
            )
            budgets.append(advantaged if advantage > 0 else disadvantaged)
        return budgets

    def _run_cohort(
        self,
        cohort_size: int,
        *,
        cohort: int,
        first_game: int,
        stop_requested: Callable[[], bool],
        progress: Callable[..., None] | None,
    ) -> list[GameSummary]:
        variant = self.config.variant
        states = self.native.StateBatch(
            self.config.rings,
            cohort_size,
            mode=variant.mode,
            handicap=variant.handicap,
            pie=variant.pie,
        )
        node_count = int(states.node_count)
        trajectories: list[list[_Decision]] = [[] for _ in range(cohort_size)]
        clinch_finalizations: list[_ClinchFinalization | None] = [
            None for _ in range(cohort_size)
        ]
        game_ids = [self._game_id(first_game + row) for row in range(cohort_size)]
        pda_seats = self._draw_pda_seats(cohort, cohort_size)
        swapped_rows = [False] * cohort_size
        pinned_versions = [
            (
                self.evaluator.model_version,
                self.evaluator.model_step,
                self.model_identity,
            )
            for _ in range(cohort_size)
        ]
        iteration = 0
        while True:
            if self.config.clinch_finalization == "loser-fill":
                self._complete_clinches(states, clinch_finalizations)
            state_data = states.data()
            if all(bool(terminal) for terminal in state_data.terminal):
                break
            if stop_requested():
                dropped_decisions = sum(len(row) for row in trajectories)
                self.interrupted_cohorts += 1
                self.dropped_games += cohort_size
                self.dropped_decisions += dropped_decisions
                if progress is not None:
                    progress(
                        phase="selfplay_abort",
                        cohort=cohort,
                        ply_wave=iteration,
                        dropped_games=cohort_size,
                        dropped_decisions=dropped_decisions,
                    )
                return []
            current_pin = (
                self.evaluator.model_version,
                self.evaluator.model_step,
                self.model_identity,
            )
            if any(pin != current_pin for pin in pinned_versions):
                raise RuntimeError(
                    "model changed while an exact game cohort was active"
                )
            to_move = [int(value) for value in state_data.to_move]
            row_pdas = [pda_seats[row][to_move[row]] for row in range(cohort_size)]
            positions = positions_from_native(state_data, pda=row_pdas)
            mode_seed = self._seed("mode", cohort, iteration, *game_ids)
            full_search = mode_seed / float(1 << 64) < self.config.full_probability
            simulations = self.config.simulation_budget(full=full_search)
            budgets = self._root_budgets(state_data, pda_seats, simulations=simulations)
            search_seed = self._seed("search", cohort, iteration, *game_ids)
            search = self.native.SearchBatch(
                states,
                simulations=simulations,
                max_considered=self.config.considered_actions(),
                c_visit=self.config.c_visit,
                c_scale=self.config.c_scale,
                deterministic_seed=search_seed,
                simulations_per_root=budgets,
                pda_by_seat=pda_seats,
            )
            roots = search.root_requests()
            root_response = self.evaluator.evaluate(roots)
            search.initialize_roots(*root_response.submit_args())
            guard = 0
            guard_limit = max(budgets) * self.config.batch_size * 4 + 16
            while not search.is_done():
                guard += 1
                if guard > guard_limit:
                    raise RuntimeError("native search failed to make progress")
                requests = search.next_requests()
                if len(requests) == 0:
                    continue
                response = self.evaluator.evaluate(requests)
                search.submit(*response.submit_args())
            results = search.results()
            swap_available = [bool(value) for value in state_data.swap_available]
            root_values = [float(value) for value in results.root_values]
            swaps = [
                swap_available[row]
                and root_values[row] < -self.config.variants.swap_dead_zone
                for row in range(cohort_size)
            ]
            self._record_decisions(
                trajectories,
                positions,
                state_data,
                results,
                full_search=full_search,
                simulations=simulations,
                search_seed=search_seed,
                budgets=budgets,
                swaps=swaps,
            )
            selected = [int(action) for action in results.selected_actions]
            active_rows = [
                row for row, terminal in enumerate(results.terminal) if not terminal
            ]
            if any(
                selected[row] < 0 or selected[row] >= positions[row].stones.numel()
                for row in active_rows
            ):
                raise RuntimeError("active search row returned an invalid placement")
            actions: list[int] = []
            for row in active_rows:
                if swap_available[row]:
                    self.pie_decisions += 1
                if swaps[row]:
                    self.pie_swaps += 1
                    swapped_rows[row] = True
                    actions.append(node_count)
                else:
                    actions.append(selected[row])
            states.apply_many(active_rows, actions)
            iteration += 1
            if progress is not None and iteration % 32 == 0:
                progress(
                    phase="selfplay_cohort",
                    cohort=cohort,
                    ply_wave=iteration,
                    active_games=len(active_rows),
                )
        return self._finalize_rows(
            states,
            states.data(),
            trajectories,
            pinned_versions,
            list(range(cohort_size)),
            game_ids,
            clinch_finalizations,
            pda_seats=pda_seats,
            swapped_rows=swapped_rows,
        )

    def _complete_clinches(
        self,
        states: Any,
        finalizations: list[_ClinchFinalization | None],
    ) -> None:
        data = states.complete_clinches()
        batch_size = int(data.batch_size)
        clinched = [bool(value) for value in data.clinched]
        winners = [int(value) for value in data.winner]
        empty_nodes = [int(value) for value in data.empty_nodes]
        last_moves = [int(value) for value in data.last_move]
        turn_counts = [int(value) for value in data.turn_count]
        if batch_size != len(finalizations) or any(
            len(values) != batch_size
            for values in (
                clinched,
                winners,
                empty_nodes,
                last_moves,
                turn_counts,
            )
        ):
            raise RuntimeError("native clinch completion buffers are invalid")
        for row, was_clinched in enumerate(clinched):
            if not was_clinched:
                continue
            if (
                finalizations[row] is not None
                or winners[row] not in (0, 1)
                or empty_nodes[row] <= 0
                or last_moves[row] < 0
                or turn_counts[row] <= 0
            ):
                raise RuntimeError("native clinch completion metadata is invalid")
            finalizations[row] = _ClinchFinalization(
                winner=winners[row],
                empty_nodes=empty_nodes[row],
                last_move=last_moves[row],
                turn_count=turn_counts[row],
            )

    def _record_decisions(
        self,
        trajectories: list[list[_Decision]],
        positions: Sequence[Any],
        state_data: Any,
        results: Any,
        *,
        full_search: bool,
        simulations: int,
        search_seed: int,
        budgets: Sequence[int] | None = None,
        swaps: Sequence[bool] | None = None,
    ) -> None:
        offsets = [int(value) for value in results.action_offsets]
        actions = [int(value) for value in results.actions]
        probabilities = np.asarray(results.policy_target, dtype=np.float32)
        raw_priors = getattr(results, "priors", None)
        priors = (
            np.asarray(raw_priors, dtype=np.float32)
            if raw_priors is not None
            else np.empty(0, dtype=np.float32)
        )
        selected_actions = [int(value) for value in results.selected_actions]
        if len(offsets) != len(positions) + 1 or offsets[-1] != len(actions):
            raise RuntimeError("native search result CSR is invalid")
        if len(selected_actions) != len(positions):
            raise RuntimeError("native search selected-action rows are invalid")
        stones_placed = list(getattr(state_data, "stones_placed"))
        terminal = [bool(value) for value in results.terminal]
        for row, position in enumerate(positions):
            if terminal[row]:
                continue
            policy = None
            policy_entropy = None
            policy_surprise = 0.0
            if full_search or self.config.record_fast_policy_targets:
                start, end = offsets[row], offsets[row + 1]
                if end <= start or end > probabilities.size:
                    raise RuntimeError("full-search policy target is missing")
                policy = np.zeros(position.stones.numel(), dtype=np.float32)
                for action, probability in zip(
                    actions[start:end], probabilities[start:end], strict=True
                ):
                    if action < 0 or action >= position.stones.numel():
                        raise RuntimeError("search policy action is invalid")
                    policy[action] = probability
                mass = float(policy.sum())
                if mass <= 0:
                    raise RuntimeError("completed-Q policy has no mass")
                policy /= mass
                positive = policy[policy > 0]
                policy_entropy = max(
                    0.0,
                    -float(
                        np.sum(
                            positive * np.log(positive),
                            dtype=np.float64,
                        )
                    ),
                )
                if not math.isfinite(policy_entropy):
                    raise RuntimeError("completed-Q policy entropy is not finite")
                if priors.size:
                    if priors.size != probabilities.size:
                        raise RuntimeError(
                            "root priors and policy target sizes disagree"
                        )
                    prior = np.zeros(position.stones.numel(), dtype=np.float32)
                    for action, probability in zip(
                        actions[start:end], priors[start:end], strict=True
                    ):
                        prior[action] = probability
                    prior_mass = float(prior.sum())
                    if prior_mass <= 0:
                        raise RuntimeError("root policy prior has no mass")
                    prior /= prior_mass
                    positive = policy > 0
                    policy_surprise = float(
                        np.sum(
                            policy[positive]
                            * (
                                np.log(policy[positive])
                                - np.log(np.maximum(prior[positive], 1e-12))
                            ),
                            dtype=np.float64,
                        )
                    )
                    if not math.isfinite(policy_surprise) or policy_surprise < -1e-6:
                        raise RuntimeError("policy surprise is invalid")
                    policy_surprise = max(0.0, policy_surprise)
                elif self.config.policy_surprise_weight:
                    raise RuntimeError(
                        "policy-surprise weighting requires native root priors"
                    )
            trajectories[row].append(
                _Decision(
                    position=position,
                    policy=policy,
                    full_search=full_search,
                    simulations=(
                        int(budgets[row]) if budgets is not None else simulations
                    ),
                    phase=int(stones_placed[row]),
                    search_seed=search_seed,
                    ply=len(trajectories[row]),
                    policy_weight=(
                        1.0 if full_search else self.config.fast_policy_weight
                    ),
                    policy_surprise=policy_surprise,
                    swapped=bool(swaps[row]) if swaps is not None else False,
                )
            )
            if full_search:
                self.full_decisions += 1
            else:
                self.fast_decisions += 1
            if policy_entropy is not None:
                self.policy_entropy_count += 1
                self.policy_entropy_sum += policy_entropy
                self.policy_weight_sum += (
                    1.0 if full_search else self.config.fast_policy_weight
                )
                self.policy_surprise_count += 1
                self.policy_surprise_sum += policy_surprise

    def _finalize_rows(
        self,
        states: Any,
        state_data: Any,
        trajectories: list[list[_Decision]],
        pinned_versions: list[tuple[str, int, str]],
        rows: Sequence[int],
        game_ids: Sequence[str],
        clinch_finalizations: Sequence[_ClinchFinalization | None],
        pda_seats: Sequence[tuple[int, int]] | None = None,
        swapped_rows: Sequence[bool] | None = None,
    ) -> list[GameSummary]:
        variant = self.config.variant
        scores_data = states.score_data()
        trajectory_data = states.trajectory_data()
        scores = score_results_from_native(scores_data)
        final_positions = positions_from_native(state_data)
        trajectory_rows = trajectory_rows_from_native(trajectory_data)
        terminal_value = list(scores_data.terminal_value)
        outcome_class = list(scores_data.outcome_class)
        score_margin = list(scores_data.score_margin)
        winner = list(scores_data.winner)
        summaries: list[GameSummary] = []
        for row in rows:
            final_position = final_positions[row]
            final_score = scores[row]
            clinch = clinch_finalizations[row]
            if not final_position.terminal or final_score.leader not in (0, 1):
                raise RuntimeError("self-play final state must be full and decisive")
            if int(winner[row]) != final_score.leader:
                raise RuntimeError("native final winner disagrees with final score")
            if clinch is not None and clinch.winner != final_score.leader:
                raise RuntimeError("clinch winner disagrees with proof-board score")
            expected_outcome = (
                OUTCOME_WIN
                if final_score.leader == final_position.to_move
                else OUTCOME_LOSS
            )
            expected_value = 1.0 if expected_outcome == OUTCOME_WIN else -1.0
            expected_margin = (
                final_score.players[final_position.to_move].total
                - final_score.players[1 - final_position.to_move].total
            )
            if (
                int(outcome_class[row]) != expected_outcome
                or float(terminal_value[row]) != expected_value
                or int(score_margin[row]) != expected_margin
            ):
                raise RuntimeError("native binary terminal targets are inconsistent")
            decisions = trajectories[row]
            if not decisions:
                raise RuntimeError("terminal game has no recorded decisions")
            version, model_step, model_identity = pinned_versions[row]
            game_id = game_ids[row]
            seats = pda_seats[row] if pda_seats is not None else (0, 0)
            game_swapped = (
                bool(swapped_rows[row]) if swapped_rows is not None else False
            )
            sample_weights = self._policy_surprise_sample_weights(decisions)
            for decision, sample_weight in zip(decisions, sample_weights, strict=True):
                mode = "full" if decision.full_search else "fast"
                self.pending_samples.append(
                    ReplaySample.from_position(
                        decision.position,
                        policy=decision.policy,
                        final_score=scores[row],
                        search_provenance=(
                            f"gumbel-completed-q:{mode}:"
                            f"simulations={decision.simulations}:"
                            f"seed={decision.search_seed}:model={model_identity}:"
                            f"game={game_id}:ply={decision.ply}:"
                            f"final={'clinch-loser-fill' if clinch else 'board-full'}:"
                            f"variant={variant.label}:pda={decision.position.pda}:"
                            f"swap={'taken' if decision.swapped else 'no'}"
                        ),
                        policy_provenance=(
                            (
                                "completed-q-full"
                                if decision.full_search
                                else "completed-q-fast"
                            )
                            if decision.policy is not None
                            else "none"
                        ),
                        clinch_auxiliary_targets=(
                            self.config.clinch_auxiliary_targets
                            if clinch is not None
                            else "synthetic"
                        ),
                        run_id=self.identity.run_id,
                        generation_family=self.identity.generation_family,
                        actor_id=self.identity.actor_id,
                        generation=self.identity.generation,
                        game_id=game_id,
                        ply=decision.ply,
                        model_identity=model_identity,
                        weight=sample_weight,
                        policy_weight=decision.policy_weight,
                    )
                )
                self.sample_weight_sum += sample_weight
                self.pending_phases.append(decision.phase)
                self.completed_decisions += 1
            metadata = trajectory_rows[row]
            finish_reason: Literal["board-full", "clinch"] = (
                "clinch" if clinch is not None else "board-full"
            )
            empty_nodes_saved = clinch.empty_nodes if clinch is not None else 0
            if clinch is not None:
                self.clinched_games += 1
                self.clinch_empty_nodes += clinch.empty_nodes
            if seats != (0, 0):
                self.asymmetric_games += 1
            self._record_source_role(samples=len(decisions))
            summaries.append(
                GameSummary(
                    row=row,
                    samples=len(decisions),
                    policy_samples=sum(
                        decision.policy is not None for decision in decisions
                    ),
                    search_simulations=sum(
                        decision.simulations for decision in decisions
                    ),
                    winner=int(winner[row]),
                    terminal_value=float(terminal_value[row]),
                    score_margin=int(score_margin[row]),
                    turn_count=(
                        clinch.turn_count if clinch is not None else metadata.turn_count
                    ),
                    last_move=(
                        clinch.last_move if clinch is not None else metadata.last_move
                    ),
                    model_version=version,
                    model_identity=model_identity,
                    game_id=game_id,
                    generation=self.identity.generation,
                    finish_reason=finish_reason,
                    empty_nodes_saved=empty_nodes_saved,
                    variant=variant.label,
                    swapped=game_swapped,
                    pda_seat0=seats[0],
                    pda_seat1=seats[1],
                )
            )
            if len(self.pending_samples) >= self.config.shard_size:
                self._flush(model_version=version, model_step=model_step)
        return summaries

    def _record_source_role(self, *, samples: int) -> None:
        if isinstance(samples, bool) or not isinstance(samples, int) or samples <= 0:
            raise ValueError("source-role sample count must be a positive integer")
        games_field = f"source_{self.source_role}_games"
        samples_field = f"source_{self.source_role}_samples"
        setattr(self, games_field, int(getattr(self, games_field)) + 1)
        setattr(
            self,
            samples_field,
            int(getattr(self, samples_field)) + samples,
        )

    def _policy_surprise_sample_weights(
        self,
        decisions: Sequence[_Decision],
    ) -> list[float]:
        mix = self.config.policy_surprise_weight
        if not mix:
            return [1.0] * len(decisions)
        surprises = np.asarray(
            [
                decision.policy_surprise if decision.policy is not None else 0.0
                for decision in decisions
            ],
            dtype=np.float64,
        )
        eligible = surprises > 0
        if not bool(eligible.any()):
            return [1.0] * len(decisions)
        normalized = np.zeros_like(surprises)
        normalized[eligible] = (
            int(eligible.sum()) * surprises[eligible] / surprises[eligible].sum()
        )
        weights = np.ones_like(surprises)
        weights[eligible] = (1.0 - mix) + mix * normalized[eligible]
        maximum = self.config.policy_surprise_max_weight
        weights = np.minimum(weights, maximum)
        deficit = len(weights) - float(weights.sum())
        if deficit > 1e-12:
            available = weights < maximum
            headroom = maximum - weights[available]
            if headroom.size and float(headroom.sum()) >= deficit:
                weights[available] += deficit * headroom / headroom.sum()
        return [float(value) for value in weights]

    def _flush(
        self,
        *,
        model_version: str | None = None,
        model_step: int | None = None,
    ) -> None:
        if not self.pending_samples:
            return
        if model_version is None:
            model_version = self.evaluator.model_version
        if model_step is None:
            model_step = self.evaluator.model_step
        expected = len(self.pending_samples)
        append_started = time.perf_counter()
        record = self.sink.append(
            self.pending_samples,
            phase_min=min(self.pending_phases),
            phase_max=max(self.pending_phases),
            model_version=model_version,
            model_step=model_step,
            model_identity=self.model_identity,
            run_id=self.identity.run_id,
            generation_family=self.identity.generation_family,
            actor_id=self.identity.actor_id,
            generation=self.identity.generation,
        )
        append_seconds = time.perf_counter() - append_started
        persisted = int(getattr(record, "sample_count", expected))
        if persisted != expected:
            raise RuntimeError("replay sink persisted an unexpected decision count")
        append_bytes = 0
        record_path = getattr(record, "path", None)
        if record_path is not None:
            try:
                append_bytes = Path(record_path).stat().st_size
            except (OSError, TypeError, ValueError):
                pass
        self.replay_append_calls += 1
        self.replay_append_bytes += append_bytes
        self.replay_append_seconds += append_seconds
        self.persisted_decisions += persisted
        self.pending_samples = []
        self.pending_phases = []

    def _game_id(self, game: int) -> str:
        digest = hashlib.sha256(
            (
                f"{self.identity.run_id}\0{self.identity.generation_family}\0"
                f"{self.identity.actor_id}\0{self.identity.generation}\0{game}"
            ).encode("utf-8")
        ).hexdigest()
        return f"game-{digest}"

    def _seed(self, purpose: str, *parts: object) -> int:
        encoded = "\0".join(
            (
                str(self.config.seed),
                self.identity.run_id,
                self.identity.generation_family,
                self.identity.actor_id,
                str(self.identity.generation),
                self.model_identity,
                purpose,
                *(str(part) for part in parts),
            )
        ).encode("utf-8")
        return int.from_bytes(hashlib.sha256(encoded).digest()[:8], "big")
