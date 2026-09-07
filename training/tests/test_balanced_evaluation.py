from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import pytest

from startrain.arena import ArenaPair, ArenaRunner
from startrain.balanced_evaluation import (
    BALANCED_CATEGORIES,
    BALANCED_OBSERVATION_MODEL,
    balanced_cells,
    balanced_observation_model,
    balanced_opening_seed,
    cell_variant,
    completed_counts_by_ring,
    cycle_log_e_value,
    cycle_confidence_sequence,
    evaluation_contract,
    pair_key,
    summarize_balanced_pairs,
)
from startrain.config import ArenaConfig, load_config
from startrain.promotion import PromotionSupervisor


def config(**changes) -> ArenaConfig:
    return ArenaConfig(
        balanced_cells=True,
        pairs_per_ring=4,
        minimum_pairs_per_ring=4,
        max_pairs_per_ring=400,
        simulations=256,
        **changes,
    )


def pairs_for(config: ArenaConfig, count: int, outcomes=(1, -1)) -> list[ArenaPair]:
    pairs = []
    for ring in config.rings:
        for name in BALANCED_CATEGORIES:
            for index in range(count):
                variant = cell_variant(name, index, config)
                selected = (
                    outcomes(ring, name, index) if callable(outcomes) else outcomes
                )
                pairs.append(
                    ArenaPair(
                        ring=ring,
                        pair=index,
                        opening_seed=balanced_opening_seed(
                            config.seed, ring, variant, index
                        ),
                        opening_action=None,
                        forced_opening=False,
                        outcomes=selected,
                        variant=variant.label,
                        segment=variant.segment,
                    )
                )
    return pairs


def test_balanced_objective_has_equal_cells_severities_and_paired_cycles() -> None:
    cfg = config()
    pairs = pairs_for(cfg, 8)
    result = summarize_balanced_pairs(pairs, cfg)
    aggregate = result["balanced_aggregate"]
    assert len(result["per_cell"]) == 24
    assert aggregate["cell_weights"] == {key: 1 / 24 for key in balanced_cells(cfg)}
    assert aggregate["pairs"] == 192 and aggregate["games"] == 384
    assert aggregate["complete_cycles"] == 2
    assert aggregate["cycle_scores"] == [0.5, 0.5]
    assert aggregate["observation_model"] == BALANCED_OBSERVATION_MODEL
    assert result["promotion"]["decision"] == "continue"
    assert summarize_balanced_pairs(list(reversed(pairs)), cfg) == result


def test_missing_or_oversampled_cells_do_not_change_fixed_weights() -> None:
    cfg = config()
    original = pairs_for(cfg, 8)
    extra = [
        pair
        for pair in pairs_for(cfg, 16, (1, 1))
        if pair.ring == 4 and pair.variant == "classic" and pair.pair >= 8
    ]
    before = summarize_balanced_pairs(original, cfg)
    after = summarize_balanced_pairs(original + extra, cfg)
    assert (
        after["balanced_aggregate"]["score_rate"]
        == before["balanced_aggregate"]["score_rate"]
    )
    assert after["balanced_aggregate"]["cycle_scores"] == [0.5, 0.5]
    missing = [
        pair for pair in original if not (pair.ring == 10 and pair.variant == "classic")
    ]
    result = summarize_balanced_pairs(missing, cfg)
    assert result["balanced_aggregate"]["status"] == "incomplete"
    assert result["balanced_aggregate"]["score_rate"] is None
    assert result["balanced_aggregate"]["missing_cells"] == ["r10/classic-standard"]
    assert result["promotion"]["decision"] == "continue"


def test_sparse_interrupted_cells_resume_without_using_incomplete_cycles() -> None:
    cfg = config()
    original = pairs_for(cfg, 8)
    partial = [
        pair for pair in original if pair_key(pair) != (4, "classic-standard", 2)
    ]
    assert completed_counts_by_ring(partial, cfg)[4] == 2
    assert (
        summarize_balanced_pairs(partial, cfg)["balanced_aggregate"]["complete_cycles"]
        == 0
    )
    restored = partial + [
        pair for pair in original if pair_key(pair) == (4, "classic-standard", 2)
    ]
    assert summarize_balanced_pairs(restored, cfg) == summarize_balanced_pairs(
        original, cfg
    )
    with pytest.raises(ValueError, match="unique"):
        summarize_balanced_pairs(original + [original[0]], cfg)


def test_promotion_depends_on_all_cells_and_retains_proven_cell_vetoes() -> None:
    cfg = config(cell_regression_floor_elo=-50)
    win = summarize_balanced_pairs(pairs_for(cfg, 80, (1, 1)), cfg)
    assert win["promotion"]["decision"] == "promote"
    assert win["balanced_aggregate"]["status"] == "saturated"
    assert win["balanced_aggregate"]["elo_difference"] is None
    standard_only = summarize_balanced_pairs(
        pairs_for(
            cfg,
            80,
            lambda ring, name, index: (1, 1) if name == "double-standard" else (-1, -1),
        ),
        cfg,
    )
    assert standard_only["promotion"]["decision"] != "promote"
    collapsed = summarize_balanced_pairs(
        pairs_for(
            cfg,
            160,
            lambda ring, name, index: (
                (-1, -1) if (ring, name) == (10, "classic-handicap") else (1, 1)
            ),
        ),
        cfg,
    )
    assert collapsed["promotion"]["decision"] == "reject_ring_regression"
    assert collapsed["promotion"]["regression_source"] == "cell"
    assert collapsed["promotion"]["cell_vetoes"] == ["r10/classic-handicap"]


def test_contract_separates_search_budgets_and_rejects_severity_drift() -> None:
    cfg = config()
    assert evaluation_contract(cfg) != evaluation_contract(
        replace(cfg, simulations=1024)
    )
    assert evaluation_contract(cfg) == evaluation_contract(replace(cfg, seed=999))
    pairs = pairs_for(cfg, 4)
    wrong = replace(
        next(pair for pair in pairs if pair.variant == "handicap-2-classic"),
        variant="handicap-9-classic",
    )
    with pytest.raises(ValueError, match="handicap schedule"):
        summarize_balanced_pairs([wrong], cfg)


def test_largest_board_objective_keeps_equal_modes_without_small_board_vetoes() -> None:
    all_boards = config(cell_regression_floor_elo=-50)
    evidence = pairs_for(
        all_boards,
        160,
        lambda ring, name, index: (1, 1) if ring == 10 else (-1, -1),
    )
    previous = summarize_balanced_pairs(evidence, all_boards)
    assert previous["promotion"]["decision"] == "reject_ring_regression"

    largest = replace(all_boards, rings=(10,))
    result = summarize_balanced_pairs(
        [pair for pair in evidence if pair.ring == 10], largest
    )
    assert result["promotion"]["decision"] == "promote"
    assert result["promotion"]["cell_vetoes"] == []
    assert set(result["per_ring"]) == {"10"}
    aggregate = result["balanced_aggregate"]
    assert aggregate["cell_weights"] == {
        f"r10/{name}": 1 / 6 for name in BALANCED_CATEGORIES
    }
    assert aggregate["pairs_per_cycle"] == 24
    assert aggregate["observation_model"] == balanced_observation_model(largest)
    assert "24-cells" not in aggregate["observation_model"]
    with pytest.raises(ValueError, match="unconfigured cell"):
        summarize_balanced_pairs(evidence, largest)


def test_largest_board_still_vetoes_a_proven_regression_in_one_of_six_modes() -> None:
    largest = config(rings=(10,), cell_regression_floor_elo=-50)
    evidence = pairs_for(
        largest,
        160,
        lambda ring, name, index: (-1, -1) if name == "classic-handicap" else (1, 1),
    )
    result = summarize_balanced_pairs(evidence, largest)
    assert result["promotion"]["decision"] == "reject_ring_regression"
    assert result["promotion"]["cell_vetoes"] == ["r10/classic-handicap"]


def test_subset_contracts_are_distinct_and_legacy_contract_hash_is_unchanged() -> None:
    legacy = ArenaConfig(balanced_cells=True)
    assert evaluation_contract(legacy)["identity"] == (
        "sha256-b7814ca5e96b14ce9392d26791870a2dd4aeed83bea6230a0ed68ebbe0848a20"
    )
    contracts = [
        evaluation_contract(replace(legacy, rings=rings))
        for rings in ((4,), (10,), (6, 10), (4, 6, 8, 10))
    ]
    assert len({contract["identity"] for contract in contracts}) == 4
    assert len({contract["objective"] for contract in contracts}) == 4
    assert contracts[1]["cell_weight"] == 1 / 6
    assert contracts[1]["cells"] == [f"r10/{name}" for name in BALANCED_CATEGORIES]


def test_paired_cycle_test_controls_sequential_null_error_and_has_screen_power() -> (
    None
):
    import math
    import numpy as np

    rng = np.random.default_rng(7341)
    threshold = math.log(20)
    trials = 2_000
    rates = {}
    # Each Bernoulli is a maximally correlated role-reversed pair (both games
    # win or both lose), never two independent game observations.
    for probability in (0.5, 0.6):
        trajectories = rng.binomial(96, probability, size=(trials, 10)) / 96
        crosses = sum(
            any(
                cycle_log_e_value(
                    scores[:stop],
                    pairs_per_cycle=96,
                    null_mean=0.5,
                    direction="greater",
                )
                >= threshold
                for stop in range(1, 11)
            )
            for scores in trajectories
        )
        rates[probability] = crosses / trials
    assert rates[0.5] <= 0.065
    assert rates[0.6] >= 0.9
    lower, upper = cycle_confidence_sequence(
        [0.6] * 10, pairs_per_cycle=96, error_probability=0.05
    )
    assert 0.5 < lower < 0.6 < upper


def test_balanced_cell_seed_streams_are_distinct_and_repeatable() -> None:
    cfg = config()
    pairs = pairs_for(cfg, 8)
    assert len({pair.opening_seed for pair in pairs}) == len(pairs)
    assert pairs_for(cfg, 8) == pairs
    corrupted = list(pairs)
    corrupted[1] = replace(corrupted[1], opening_seed=corrupted[0].opening_seed)
    with pytest.raises(ValueError, match="distinct seed"):
        summarize_balanced_pairs(corrupted, cfg)


def test_balanced_promotion_never_resumes_legacy_or_other_budget_evidence(
    tmp_path,
) -> None:
    base = load_config(Path(__file__).parents[1] / "configs" / "small.yaml")
    supervisor = object.__new__(PromotionSupervisor)
    supervisor.experiment = replace(base, arena=config())
    supervisor.results_directory = tmp_path
    candidate = SimpleNamespace(model_identity="sha256-" + "c" * 64)
    champion = SimpleNamespace(model_identity="sha256-" + "b" * 64)
    payload = {
        "schema_version": 4,
        "candidate": candidate.model_identity,
        "baseline": champion.model_identity,
        "promotion": {},
        "terminal": True,
    }
    legacy_path = (
        tmp_path / f"{candidate.model_identity}-vs-{champion.model_identity}.json"
    )
    legacy_path.write_text(json.dumps(payload))
    assert supervisor._read_result(candidate, champion) is None
    path = supervisor._result_path(candidate, champion)
    path.write_text(json.dumps(payload))
    with pytest.raises(ValueError, match="contract changed"):
        supervisor._read_result(candidate, champion)
    payload["evaluation_contract"] = evaluation_contract(supervisor.experiment.arena)
    path.write_text(json.dumps(payload))
    assert supervisor._read_result(candidate, champion) is not None
    supervisor.experiment = replace(
        supervisor.experiment,
        arena=replace(supervisor.experiment.arena, simulations=1024),
    )
    assert supervisor._read_result(candidate, champion) is None
    assert legacy_path.exists() and path.exists()


def test_balanced_promotion_persists_and_recovers_all_cell_pairs(
    tmp_path, monkeypatch
) -> None:
    from dataclasses import asdict
    import startrain.promotion as promotion_module
    from test_promotion import _promotion_wave_case

    case = _promotion_wave_case(tmp_path, monkeypatch)
    cfg = replace(config(), max_pairs_per_ring=8)
    case.supervisor.experiment = replace(case.experiment, arena=cfg)
    calls = []

    class BalancedWave:
        def __init__(self, **options):
            self.config = options["config"]

        def run(self, *, pair_starts, pair_counts, previous_pairs, **options):
            existing = {pair_key(pair) for pair in previous_pairs}
            available = pairs_for(self.config, 8)
            selected = [
                pair
                for pair in available
                if pair_starts[pair.ring]
                <= pair.pair
                < pair_starts[pair.ring] + pair_counts[pair.ring]
                and pair_key(pair) not in existing
            ]
            if not calls:
                selected = [
                    pair
                    for pair in selected
                    if pair_key(pair) != (4, "classic-standard", 2)
                ]
            calls.append((dict(pair_starts), len(selected)))
            return {
                "schema_version": 4,
                "candidate": case.candidate.model_identity,
                "baseline": case.champion.model_identity,
                "pairs": [asdict(pair) for pair in selected],
                "games": [],
                **summarize_balanced_pairs(
                    list(previous_pairs) + selected, self.config
                ),
            }

    monkeypatch.setattr(promotion_module, "ArenaRunner", BalancedWave)
    path = case.supervisor._result_path(case.candidate, case.champion)
    for turn in range(3):
        case.supervisor.run(stop_requested=lambda: False, once=True)
        saved = json.loads(path.read_text())
        assert len(saved["pairs"]) == len(
            {(p["ring"], p["variant"], p["pair"]) for p in saved["pairs"]}
        )
        if turn == 0:
            assert not saved["terminal"]
            assert saved["balanced_aggregate"]["complete_cycles"] == 0
    assert len(saved["pairs"]) == 24 * 8
    assert saved["terminal"] and saved["promotion"]["decision"] == "reject_max_pairs"
    assert saved["balanced_aggregate"]["complete_cycles"] == 2
    assert calls[1][0][4] == 2


@pytest.mark.native
@pytest.mark.parametrize("rings", [(4, 6, 8, 10), (10,)])
def test_native_balanced_runner_plays_configured_cells_with_role_reversal(
    rings,
) -> None:
    from startrain.inference import GraphInferenceAdapter, InferenceConfig
    from startrain.model import GraphResTNet, ModelConfig

    native = pytest.importorskip("star_native")
    evaluator = GraphInferenceAdapter(
        GraphResTNet(ModelConfig(width=8, rrt_groups=1, attention_heads=2, kv_heads=1)),
        config=InferenceConfig(precision="fp32"),
        model_version="sha256-" + "c" * 64,
        model_step=0,
        model_identity="sha256-" + "c" * 64,
    )
    cfg = replace(config(rings=rings), simulations=2, max_considered=2)
    result = ArenaRunner(
        native_module=native, candidate=evaluator, baseline=evaluator, config=cfg
    ).run(pair_counts={ring: 1 for ring in cfg.rings})
    cells = len(rings) * len(BALANCED_CATEGORIES)
    assert len(result["pairs"]) == cells and len(result["games"]) == 2 * cells
    assert len({(game["ring"], game["variant"]) for game in result["games"]}) == cells
    assert result["evaluation_metrics"]["requested_pairs"] == cells
    assert result["search"]["pie_rule"] is True
    assert set(result["search"]["segments"]) == {
        "standard",
        "classic",
        "pie",
        "handicap",
    }
    assert result["search"]["segment_handicap_classic_share"] == 0.5
    assert all(
        {
            game["candidate_player"]
            for game in result["games"]
            if game["ring"] == pair["ring"] and game["variant"] == pair["variant"]
        }
        == {0, 1}
        for pair in result["pairs"]
    )
    assert result["balanced_aggregate"]["status"] == "incomplete"
    assert result["promotion"]["decision"] == "continue"


@pytest.mark.native
@pytest.mark.parametrize(
    "name,balanced",
    [(name, True) for name in BALANCED_CATEGORIES] + [("double-standard", False)],
)
def test_balanced_pair_search_is_stable_across_reordered_and_partial_batches(
    name, balanced
):
    from concurrent.futures import ThreadPoolExecutor
    from startrain.inference import InferenceResponse

    native = pytest.importorskip("star_native")

    class StableEvaluator:
        model_version = "sha256-" + "c" * 64

        def evaluate(self, requests):
            return InferenceResponse(
                list(requests.tokens),
                [0.0] * len(requests),
                list(requests.legal_offsets),
                [0.0] * len(requests.legal_actions),
            )

    cfg = replace(config(), simulations=2, max_considered=2)
    if not balanced:
        cfg = replace(cfg, balanced_cells=False, rings=(4,))
    evaluator = StableEvaluator()
    runner = ArenaRunner(
        native_module=native,
        candidate=evaluator,
        baseline=evaluator,
        config=cfg,
        stable_pair_seeds=not balanced,
    )
    variant = cell_variant(name, 0, cfg)
    specifications = runner._pair_specifications(4, [0, 4], variant)
    with ThreadPoolExecutor(max_workers=1) as inference:

        def play(specs):
            games = runner._play_ring_batch(
                4,
                specs,
                variant=variant,
                progress=None,
                inference_executor=inference,
                stop_requested=lambda: False,
            )
            return {(game.pair, game.candidate_player): game for game in games}

        complete = play(specifications)
        assert play(specifications[2:] + specifications[:2]) == complete
        assert play(specifications[:2]) == {
            key: game for key, game in complete.items() if key[0] == 0
        }
    if not balanced:
        result = runner.run(pair_counts={4: 1})
        assert (
            result["search"]["seed_stream_policy"]
            == "independent-cell-pair-seat-move-v2"
        )
        legacy = ArenaRunner(
            native_module=native, candidate=evaluator, baseline=evaluator, config=cfg
        )
        assert legacy.stable_pair_seeds is False
