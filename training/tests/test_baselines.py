from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from startrain.arena import ArenaRunner
from startrain.baselines import create_frozen_baseline
from startrain.cli import arena_main
from startrain.config import ArenaConfig
from startrain.inference import InferenceResponse


class GreedyStateBatch:
    def __init__(self, to_move: list[int]) -> None:
        self.to_move = to_move
        self.actions: list[int | None] = [None] * len(to_move)

    @classmethod
    def from_semantic(
        cls,
        _rings: int,
        zero_bits: list[int],
        _one_bits: list[int],
        to_move: list[int],
        _moves_left: list[int],
        _opening: list[bool],
        _pass_streak: list[int],
    ) -> "GreedyStateBatch":
        assert len(zero_bits) == len(to_move) * 7
        return cls(to_move)

    def apply_many(self, indices: list[int], actions: list[int]) -> None:
        for index, action in zip(indices, actions, strict=True):
            self.actions[index] = action

    def data(self) -> object:
        return SimpleNamespace(terminal=[False] * len(self.actions))

    def score_data(self) -> object:
        rows = []
        for actor, action in zip(self.to_move, self.actions, strict=True):
            components = [0] * 14
            if action is not None and action >= 0:
                components[actor * 6 + 5] = action + 1
                components[actor * 6 + 1] = action
                components[13] = actor
            else:
                components[13] = -1
            rows.extend(components)
        return SimpleNamespace(components=rows)


class GreedyNative:
    StateBatch = GreedyStateBatch


class GreedyRequests:
    def __init__(self) -> None:
        self.tokens = [101]
        self.legal_offsets = [0, 3]
        self.legal_actions = [0, 1, -1]
        self.states = SimpleNamespace(
            rings=3,
            batch_size=1,
            zero_bits=[0] * 7,
            one_bits=[0] * 7,
            to_move=[0],
            moves_left=[2],
            opening=[False],
            pass_streak=[0],
        )

    def __len__(self) -> int:
        return 1


def test_frozen_evaluators_are_deterministic_and_versioned() -> None:
    greedy = create_frozen_baseline("greedy", native_module=GreedyNative)
    first = greedy.evaluate(GreedyRequests())
    second = greedy.evaluate(GreedyRequests())
    assert first == second
    assert greedy.evaluator_calls == 2
    assert greedy.evaluator_rows == 2
    assert first.policy_logits[1] == max(first.policy_logits)
    assert greedy.model_version == "frozen-greedy-native-score-v1-s1-k1-cv50-cs1"

    shallow = create_frozen_baseline("shallow-search", native_module=GreedyNative)
    assert shallow.search_budget.simulations == 64
    assert shallow.search_budget.max_considered == 16
    assert shallow.result_metadata()["algorithm"] == (
        "native-gumbel-mcts-with-one-ply-static-score-heuristic"
    )

    uniform = create_frozen_baseline("uniform", native_module=GreedyNative)
    random_alias = create_frozen_baseline("random", native_module=GreedyNative)
    assert random_alias.model_version == uniform.model_version
    assert uniform.evaluate(GreedyRequests()) == uniform.evaluate(GreedyRequests())
    assert uniform.evaluator_calls == 2
    assert uniform.evaluator_rows == 2


@pytest.mark.native
def test_frozen_baselines_are_deterministic_with_native_search() -> None:
    native = pytest.importorskip("star_native")
    for name in ("uniform", "greedy", "shallow-search"):
        baseline = create_frozen_baseline(name, native_module=native)
        selected = []
        for _ in range(2):
            states = native.StateBatch(3, 1)
            budget = baseline.search_budget
            search = native.SearchBatch(
                states,
                simulations=budget.simulations,
                max_considered=budget.max_considered,
                c_visit=budget.c_visit,
                c_scale=budget.c_scale,
                deterministic_seed=123,
            )
            roots = search.root_requests()
            search.initialize_roots(*baseline.evaluate(roots).submit_args())
            while not search.is_done():
                requests = search.next_requests()
                if len(requests):
                    search.submit(*baseline.evaluate(requests).submit_args())
            selected.append(int(search.results().selected_actions[0]))
        assert selected[0] == selected[1]


class ArenaStateBatch:
    def __init__(self, rings: int, batch_size: int) -> None:
        assert rings == 3
        assert batch_size == 1
        self.to_move = 0
        self.search_started = False
        self.searched_moves = 0
        self.terminal = False
        self.winner = -1

    def apply_many(self, indices: list[int], actions: list[int]) -> None:
        assert indices == [0]
        if not self.search_started:
            self.to_move = 1
            return
        self.searched_moves += 1
        if self.searched_moves == 1:
            self.to_move = 1 - self.to_move
        else:
            self.winner = actions[0]
            self.terminal = True

    def data(self) -> object:
        return SimpleNamespace(
            terminal=[self.terminal],
            to_move=[self.to_move],
        )

    def score_data(self) -> object:
        return SimpleNamespace(winner=[self.winner])


class ArenaRequests:
    def __init__(
        self,
        states: ArenaStateBatch,
        *,
        simulations: int,
        max_considered: int,
    ) -> None:
        self.tokens = [1]
        self.states = states.data()
        self.legal_offsets = [0, 2]
        self.legal_actions = [0, 1]
        self.simulations = simulations
        self.max_considered = max_considered

    def __len__(self) -> int:
        return 1


class ArenaSearchBatch:
    def __init__(
        self,
        states: ArenaStateBatch,
        *,
        simulations: int,
        max_considered: int,
        **_options: object,
    ) -> None:
        self.states = states
        self.states.search_started = True
        self.simulations = simulations
        self.max_considered = max_considered
        self.selected = -2

    def root_requests(self) -> ArenaRequests:
        return ArenaRequests(
            self.states,
            simulations=self.simulations,
            max_considered=self.max_considered,
        )

    def initialize_roots(
        self,
        _tokens: list[int],
        _values: list[float],
        _offsets: list[int],
        logits: list[float],
    ) -> None:
        self.selected = max(range(len(logits)), key=logits.__getitem__)

    def is_done(self) -> bool:
        return self.selected != -2

    def next_requests(self) -> object:
        raise AssertionError("fake search completes at the root")

    def submit(self, *_buffers: object) -> None:
        raise AssertionError("fake search has no pending leaves")

    def results(self) -> object:
        return SimpleNamespace(
            terminal=[False],
            selected_actions=[self.selected],
        )


class ArenaNative:
    StateBatch = ArenaStateBatch
    SearchBatch = ArenaSearchBatch


class RecordingEvaluator:
    def __init__(self, model_version: str, selected_action: int) -> None:
        self.model_version = model_version
        self.selected_action = selected_action
        self.calls: list[tuple[int, int, int]] = []

    def evaluate(self, requests: ArenaRequests) -> InferenceResponse:
        self.calls.append(
            (
                int(requests.states.to_move[0]),
                requests.simulations,
                requests.max_considered,
            )
        )
        logits = [0.0, 0.0]
        logits[self.selected_action] = 1.0
        return InferenceResponse(
            tokens=[1],
            values=[0.0],
            policy_offsets=[0, 2],
            policy_logits=logits,
        )


def _run_paired_uniform_arena() -> tuple[
    dict[str, object],
    list[tuple[int, int, int]],
    list[tuple[int, int, int]],
]:
    frozen = create_frozen_baseline("uniform", native_module=ArenaNative)
    candidate = RecordingEvaluator("candidate", selected_action=1)
    baseline = RecordingEvaluator(frozen.model_version, selected_action=0)
    result = ArenaRunner(
        native_module=ArenaNative,
        candidate=candidate,
        baseline=baseline,
        config=ArenaConfig(
            rings=(3,),
            pairs_per_ring=2,
            simulations=7,
            max_considered=3,
            bootstrap_samples=200,
            unforced_opening_fraction=0.5,
            minimum_pairs_per_ring=2,
            regression_floor_elo=-2_500.0,
        ),
        baseline_search=frozen.search_budget,
        baseline_metadata=frozen.result_metadata(),
    ).run()
    return result, candidate.calls, baseline.calls


def test_frozen_baseline_preserves_paired_role_reversal_and_result_metadata() -> None:
    first, candidate_calls, baseline_calls = _run_paired_uniform_arena()
    second, _, _ = _run_paired_uniform_arena()

    assert first["games"] == second["games"]
    assert first["pairs"] == second["pairs"]
    assert sorted(candidate_calls) == [(0, 7, 3), (0, 7, 3), (1, 7, 3), (1, 7, 3)]
    assert sorted(baseline_calls) == [(0, 1, 1), (0, 1, 1), (1, 1, 1), (1, 1, 1)]
    games = first["games"]
    assert [game["candidate_player"] for game in games] == [0, 1, 0, 1]
    assert games[0]["opening_seed"] == games[1]["opening_seed"]

    metadata = first["baseline_metadata"]
    assert metadata["kind"] == "frozen_non_human"
    assert metadata["identity"] == first["baseline"]
    assert metadata["frozen"] is True
    assert metadata["search_budget"] == {
        "simulations": 1,
        "max_considered": 1,
        "c_visit": 50.0,
        "c_scale": 1.0,
    }
    assert first["search"]["simulations"] == 7


class CapturingArenaRunner:
    calls: list[dict[str, object]] = []

    def __init__(self, **options: object) -> None:
        self.options = options
        self.calls.append(options)

    def run(self) -> dict[str, object]:
        baseline = self.options["baseline"]
        metadata = dict(self.options.get("baseline_metadata", {"kind": "checkpoint"}))
        metadata["identity"] = baseline.model_version
        metadata.setdefault(
            "search_budget",
            {
                "simulations": 9,
                "max_considered": 4,
                "c_visit": 50.0,
                "c_scale": 1.0,
            },
        )
        return {
            "baseline": baseline.model_version,
            "baseline_metadata": metadata,
            "promotion": {"decision": "continue"},
            "aggregate": {"games": 2},
            "per_ring": {
                str(ring): {
                    "anytime_elo_interval": [450.0, 700.0],
                    "pairs": 50,
                }
                for ring in (4, 6, 8, 10)
            },
        }


def test_arena_cli_preserves_checkpoint_baseline_and_selects_frozen_baseline(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    tmp_path: Path,
) -> None:
    manifests: list[str] = []
    writes: list[tuple[str, dict[str, object]]] = []
    CapturingArenaRunner.calls.clear()

    monkeypatch.setattr(
        "startrain.cli.load_config",
        lambda _path: SimpleNamespace(arena=SimpleNamespace(rings=(4, 6, 8, 10))),
    )

    def load_manifest(path: str) -> str:
        manifests.append(path)
        return f"manifest:{path}"

    monkeypatch.setattr("startrain.cli.load_model_manifest", load_manifest)
    monkeypatch.setattr(
        "startrain.cli.load_manifest_evaluator",
        lambda _experiment, manifest, *, device: SimpleNamespace(
            model_version=f"{manifest}@{device}"
        ),
    )
    monkeypatch.setattr(
        "startrain.cli.load_star_native",
        lambda *, required: GreedyNative,
    )
    monkeypatch.setattr("startrain.cli.ArenaRunner", CapturingArenaRunner)
    monkeypatch.setattr(
        "startrain.cli.atomic_json",
        lambda path, payload: writes.append((path, payload)),
    )

    checkpoint_output = tmp_path / "checkpoint.json"
    arena_main(
        [
            "--config",
            "config.yaml",
            "--candidate",
            "candidate.json",
            "--baseline",
            "baseline.json",
            "--output",
            str(checkpoint_output),
            "--device",
            "cpu",
        ]
    )
    checkpoint_summary = json.loads(capsys.readouterr().out)
    assert manifests == ["candidate.json", "baseline.json"]
    assert "baseline_search" not in CapturingArenaRunner.calls[0]
    assert checkpoint_summary["baseline"]["kind"] == "checkpoint"

    frozen_output = tmp_path / "frozen.json"
    arena_main(
        [
            "--config",
            "config.yaml",
            "--candidate",
            "candidate.json",
            "--baseline-kind",
            "shallow-search",
            "--output",
            str(frozen_output),
            "--device",
            "cpu",
            "--target-elo-lcb",
            "400",
            "--target-rings",
            "4",
            "6",
            "8",
            "10",
        ]
    )
    frozen_summary = json.loads(capsys.readouterr().out)
    assert manifests == ["candidate.json", "baseline.json", "candidate.json"]
    frozen_options = CapturingArenaRunner.calls[1]
    assert frozen_options["baseline_search"].simulations == 64
    assert frozen_options["baseline_metadata"]["name"] == "shallow-search"
    assert frozen_summary["baseline"]["identity"].startswith("frozen-shallow-")
    assert frozen_summary["internal_elo_target"]["passed"] is True
    assert writes[-1][1]["internal_elo_target"]["target_elo"] == 400.0
    assert [path for path, _payload in writes] == [
        str(checkpoint_output),
        str(frozen_output),
    ]

    with pytest.raises(SystemExit):
        arena_main(
            [
                "--config",
                "config.yaml",
                "--candidate",
                "candidate.json",
                "--output",
                str(tmp_path / "missing.json"),
            ]
        )
