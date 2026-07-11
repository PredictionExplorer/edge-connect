"""Deterministic single-machine ring-homogeneous native self-play."""

from __future__ import annotations

import hashlib
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any, Protocol, Sequence

import numpy as np

from .inference import InferenceResponse, NativeEvalBatchProtocol
from .native import (
    positions_from_native,
    score_results_from_native,
    trajectory_rows_from_native,
)
from .replay import ReplaySample
from .runtime import validate_identifier


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
class SelfPlayConfig:
    rings: int = 3
    batch_size: int = 1
    games: int = 1
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
    c_visit: float = 50.0
    c_scale: float = 1.0
    initial_pass_logit_penalty: float = 1.5
    score_utility_weight: float = 0.0
    shard_size: int = 512
    seed: int = 17

    def __post_init__(self) -> None:
        if not 3 <= self.rings <= 12:
            raise ValueError("self-play rings must be in 3..12")
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
        if self.simulation_reference_rings <= 0 or self.simulation_ring_exponent < 0:
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
        if self.initial_pass_logit_penalty < 0:
            raise ValueError("initial pass logit penalty must be non-negative")
        if not 0 <= self.score_utility_weight <= 1:
            raise ValueError("score utility weight must be in [0, 1]")

    @classmethod
    def cpu_smoke(cls, *, seed: int = 17) -> "SelfPlayConfig":
        return cls(
            rings=3,
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
        scale = (
            self.rings / self.simulation_reference_rings
        ) ** self.max_considered_ring_exponent
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
    terminal_reason: int
    turn_count: int
    last_move: int
    model_version: str
    model_identity: str
    game_id: str
    generation: int


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


class SelfPlayActor:
    """Drives PyO3 ``StateBatch``/``SearchBatch`` and emits replay shards."""

    def __init__(
        self,
        native_module: Any,
        evaluator: EvaluatorProtocol,
        replay_sink: ReplaySinkProtocol,
        config: SelfPlayConfig,
        identity: SelfPlayIdentity | None = None,
    ) -> None:
        self.native = native_module
        self.evaluator = evaluator
        self.sink = replay_sink
        self.config = config
        self.identity = identity or SelfPlayIdentity("manual", "manual", "manual", 0)
        self.model_identity = str(
            getattr(evaluator, "model_identity", evaluator.model_version)
        )
        validate_identifier("model_identity", self.model_identity)
        self.pending_samples: list[ReplaySample] = []
        self.pending_phases: list[int] = []
        self.persisted_decisions = 0

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
        if self.pending_samples or self.persisted_decisions != completed_decisions:
            raise RuntimeError(
                "completed-game and persisted-decision accounting disagree"
            )
        return summaries

    def _run_cohort(
        self,
        cohort_size: int,
        *,
        cohort: int,
        first_game: int,
        stop_requested: Callable[[], bool],
        progress: Callable[..., None] | None,
    ) -> list[GameSummary]:
        states = self.native.StateBatch(self.config.rings, cohort_size)
        trajectories: list[list[_Decision]] = [[] for _ in range(cohort_size)]
        game_ids = [self._game_id(first_game + row) for row in range(cohort_size)]
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
            state_data = states.data()
            if all(bool(terminal) for terminal in state_data.terminal):
                break
            if stop_requested():
                if progress is not None:
                    progress(
                        phase="selfplay_abort",
                        cohort=cohort,
                        ply_wave=iteration,
                        dropped_games=cohort_size,
                        dropped_decisions=sum(len(row) for row in trajectories),
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
            positions = positions_from_native(state_data)
            mode_seed = self._seed("mode", cohort, iteration, *game_ids)
            full_search = mode_seed / float(1 << 64) < self.config.full_probability
            simulations = self.config.simulation_budget(full=full_search)
            search_seed = self._seed("search", cohort, iteration, *game_ids)
            search = self.native.SearchBatch(
                states,
                simulations=simulations,
                max_considered=self.config.considered_actions(),
                c_visit=self.config.c_visit,
                c_scale=self.config.c_scale,
                deterministic_seed=search_seed,
            )
            roots = search.root_requests()
            root_response = self.evaluator.evaluate(roots)
            search.initialize_roots(*root_response.submit_args())
            guard = 0
            while not search.is_done():
                guard += 1
                if guard > simulations * self.config.batch_size * 4 + 16:
                    raise RuntimeError("native search failed to make progress")
                requests = search.next_requests()
                if len(requests) == 0:
                    continue
                response = self.evaluator.evaluate(requests)
                search.submit(*response.submit_args())
            results = search.results()
            self._record_decisions(
                trajectories,
                positions,
                state_data,
                results,
                full_search=full_search,
                simulations=simulations,
                search_seed=search_seed,
            )
            selected = [int(action) for action in results.selected_actions]
            active_rows = [
                row for row, terminal in enumerate(results.terminal) if not terminal
            ]
            if any(selected[row] == -2 for row in active_rows):
                raise RuntimeError("active search row returned terminal sentinel")
            states.apply_many(active_rows, [selected[row] for row in active_rows])
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
    ) -> None:
        offsets = [int(value) for value in results.action_offsets]
        actions = [int(value) for value in results.actions]
        probabilities = np.asarray(results.policy_target, dtype=np.float32)
        if len(offsets) != len(positions) + 1 or offsets[-1] != len(actions):
            raise RuntimeError("native search result CSR is invalid")
        stones_placed = list(getattr(state_data, "stones_placed"))
        terminal = [bool(value) for value in results.terminal]
        for row, position in enumerate(positions):
            if terminal[row]:
                continue
            policy = None
            if full_search or self.config.record_fast_policy_targets:
                start, end = offsets[row], offsets[row + 1]
                if end <= start or end > probabilities.size:
                    raise RuntimeError("full-search policy target is missing")
                policy = np.zeros(position.stones.numel() + 1, dtype=np.float32)
                for action, probability in zip(
                    actions[start:end], probabilities[start:end], strict=True
                ):
                    index = position.stones.numel() if action == -1 else action
                    if index < 0 or index > position.stones.numel():
                        raise RuntimeError("search policy action is invalid")
                    policy[index] = probability
                mass = float(policy.sum())
                if mass <= 0:
                    raise RuntimeError("completed-Q policy has no mass")
                policy /= mass
            trajectories[row].append(
                _Decision(
                    position=position,
                    policy=policy,
                    full_search=full_search,
                    simulations=simulations,
                    phase=int(stones_placed[row]),
                    search_seed=search_seed,
                    ply=len(trajectories[row]),
                )
            )

    def _finalize_rows(
        self,
        states: Any,
        state_data: Any,
        trajectories: list[list[_Decision]],
        pinned_versions: list[tuple[str, int, str]],
        rows: Sequence[int],
        game_ids: Sequence[str],
    ) -> list[GameSummary]:
        scores_data = states.score_data()
        trajectory_data = states.trajectory_data()
        scores = score_results_from_native(scores_data)
        trajectory_rows = trajectory_rows_from_native(trajectory_data)
        terminal_value = list(scores_data.terminal_value)
        score_margin = list(scores_data.score_margin)
        terminal_reason = list(scores_data.terminal_reason)
        winner = list(scores_data.winner)
        summaries: list[GameSummary] = []
        for row in rows:
            decisions = trajectories[row]
            if not decisions:
                raise RuntimeError("terminal game has no recorded decisions")
            version, model_step, model_identity = pinned_versions[row]
            game_id = game_ids[row]
            for decision in decisions:
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
                            f"game={game_id}:ply={decision.ply}"
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
                        run_id=self.identity.run_id,
                        generation_family=self.identity.generation_family,
                        actor_id=self.identity.actor_id,
                        generation=self.identity.generation,
                        game_id=game_id,
                        ply=decision.ply,
                        model_identity=model_identity,
                    )
                )
                self.pending_phases.append(decision.phase)
            metadata = trajectory_rows[row]
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
                    terminal_reason=int(terminal_reason[row]),
                    turn_count=metadata.turn_count,
                    last_move=metadata.last_move,
                    model_version=version,
                    model_identity=model_identity,
                    game_id=game_id,
                    generation=self.identity.generation,
                )
            )
            if len(self.pending_samples) >= self.config.shard_size:
                self._flush(model_version=version, model_step=model_step)
        return summaries

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
        persisted = int(getattr(record, "sample_count", expected))
        if persisted != expected:
            raise RuntimeError("replay sink persisted an unexpected decision count")
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
