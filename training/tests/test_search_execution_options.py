from copy import deepcopy
from dataclasses import asdict, replace
import math
from pathlib import Path
from types import SimpleNamespace

import pytest
import yaml

from startrain.balanced_evaluation import evaluation_contract
from startrain.config import ArenaConfig, ConfigError, load_config
from startrain.config_compatibility import (
    compatible_config_epoch_payloads,
    without_search_execution_defaults,
)
from startrain.inference import InferenceResponse
from startrain.search_options import (
    FullSearchBudgetConfig,
    SearchExecutionConfig,
    normalized_root_entropy,
    require_search_execution,
    root_policy_entropies,
    search_batch_row_limit,
    search_model_context,
)
from startrain.selfplay import SelfPlayActor, SelfPlayConfig


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("first_visit_batch_size", 0),
        ("first_visit_batch_size", 65),
        ("first_visit_batch_size", True),
        ("first_visit_batch_size", 2.0),
        ("subtree_reuse", 1),
        ("subtree_reuse_max_nodes", 0),
        ("subtree_reuse_max_nodes", 1_000_001),
        ("full_budget", {}),
    ],
)
def test_search_execution_requires_typed_bounded_options(field, value):
    with pytest.raises(ValueError):
        SearchExecutionConfig(**{field: value})


@pytest.mark.parametrize(
    "options",
    [
        {"mode": "confidence"},
        {"minimum_fraction": 0},
        {"minimum_fraction": 1.1},
        {"minimum_fraction": True},
        {"minimum_fraction": float("nan")},
        {"entropy_threshold": float("inf")},
        {"entropy_threshold": -0.1},
        {"entropy_threshold": 1.1},
    ],
)
def test_full_budget_policy_rejects_invalid_controls(options):
    with pytest.raises(ValueError):
        FullSearchBudgetConfig(**options)


def test_optional_search_settings_round_trip_and_reject_unknown_keys(tmp_path):
    raw = yaml.safe_load(Path("configs/small.yaml").read_text())
    settings = SearchExecutionConfig(
        first_visit_batch_size=4,
        subtree_reuse=True,
        full_budget=FullSearchBudgetConfig(mode="root-entropy"),
    )
    for section in ("selfplay", "arena"):
        raw.setdefault(section, {})["search_execution"] = asdict(
            settings
            if section == "selfplay"
            else replace(settings, full_budget=FullSearchBudgetConfig())
        )
    path = tmp_path / "experiment.yaml"
    path.write_text(yaml.safe_dump(raw))
    parsed = load_config(path)
    assert parsed.selfplay.search_execution == settings
    assert parsed.arena.search_execution == replace(
        settings, full_budget=FullSearchBudgetConfig()
    )
    path.write_text(yaml.safe_dump(parsed.as_dict()))
    assert load_config(path) == parsed
    raw["selfplay"]["search_execution"]["unknown"] = True
    path.write_text(yaml.safe_dump(raw))
    with pytest.raises(ConfigError, match="search_execution"):
        load_config(path)


def test_adaptive_full_budgets_are_rejected_for_fixed_effort_arena_evaluation():
    with pytest.raises(ConfigError, match="arena must measure fixed search effort"):
        ArenaConfig(
            search_execution=SearchExecutionConfig(
                full_budget=FullSearchBudgetConfig(mode="root-entropy")
            )
        )


def test_default_omission_preserves_legacy_config_and_evaluation_authority():
    current = load_config("configs/small.yaml").as_dict()
    original = deepcopy(current)
    previous = without_search_execution_defaults(current)
    assert current == original
    assert "search_execution" not in previous["selfplay"]
    assert "search_execution" not in previous["arena"]
    assert previous in compatible_config_epoch_payloads(current)
    assert SearchExecutionConfig().contract() is None
    assert SearchExecutionConfig().provenance() == ""
    default = evaluation_contract(ArenaConfig())
    assert "search_execution" not in default
    changed = SearchExecutionConfig(first_visit_batch_size=4)
    alternative = evaluation_contract(ArenaConfig(search_execution=changed))
    assert alternative["identity"] != default["identity"]
    assert alternative["search_execution"] == asdict(changed)
    current["selfplay"]["search_execution"] = asdict(changed)
    assert previous not in compatible_config_epoch_payloads(current)
    current["selfplay"]["search_execution"] = asdict(SearchExecutionConfig())
    current["selfplay"]["search_execution"]["first_visit_batch_size"] = True
    assert "search_execution" in without_search_execution_defaults(current)["selfplay"]


def test_only_enabled_experiments_require_the_native_capability():
    require_search_execution(SimpleNamespace(), SearchExecutionConfig())
    enabled = SearchExecutionConfig(subtree_reuse=True)
    for native in (
        SimpleNamespace(),
        SimpleNamespace(native_search_execution_version=lambda: 2),
    ):
        with pytest.raises(ValueError, match="native_search_execution_version 1"):
            require_search_execution(native, enabled)
    require_search_execution(
        SimpleNamespace(native_search_execution_version=lambda: 1), enabled
    )


def test_normalized_entropy_is_shift_invariant_and_handles_degenerate_legal_sets():
    assert normalized_root_entropy([]) == normalized_root_entropy([123]) == 0
    assert normalized_root_entropy([1, 1, 1, 1]) == pytest.approx(1)
    assert normalized_root_entropy([0, -10000]) == 0
    expected = -(0.25 * math.log(0.25) + 0.75 * math.log(0.75)) / math.log(2)
    assert normalized_root_entropy([0, math.log(3)]) == pytest.approx(expected)
    assert normalized_root_entropy([100, 100 + math.log(3)]) == pytest.approx(expected)
    for invalid in (float("nan"), float("inf"), -float("inf")):
        with pytest.raises(ValueError, match="finite"):
            normalized_root_entropy([0, invalid])


def test_entropy_rows_match_by_token_instead_of_response_order():
    requests = SimpleNamespace(
        tokens=[10, 20], tree_indices=[2, 0], legal_offsets=[0, 2, 4]
    )
    response = InferenceResponse([20, 10], [0, 0], [0, 2, 4], [0, 0, 0, -10000])
    assert root_policy_entropies(requests, response) == {0: 1, 2: 0}
    with pytest.raises(ValueError, match="matching token"):
        root_policy_entropies(requests, replace(response, tokens=[10, 10]))
    with pytest.raises(ValueError, match="legal action count"):
        root_policy_entropies(requests, replace(response, policy_offsets=[0, 1, 4]))


def test_entropy_adjusts_full_cap_before_deriving_exact_pda_budgets():
    configuration = SelfPlayConfig(
        rings=10,
        fast_simulations=32,
        full_simulations=384,
        search_execution=SearchExecutionConfig(
            full_budget=FullSearchBudgetConfig(
                mode="root-entropy", minimum_fraction=0.53
            )
        ),
    )
    evaluator = SimpleNamespace(model_identity="fixed", model_version="fixed")
    worker = SelfPlayActor(
        SimpleNamespace(native_search_execution_version=lambda: 1),
        evaluator,
        None,
        configuration,
    )
    roots = SimpleNamespace(tokens=[1, 2], tree_indices=[0, 1], legal_offsets=[0, 2, 4])
    response = InferenceResponse([1, 2], [0, 0], [0, 2, 4], [0, -10000, 0, -10000])
    state = SimpleNamespace(to_move=[1, 0], terminal=[False, False])
    budgets, evidence = worker._entropy_budgets(
        roots, response, state, [(2, -2), (0, 0)], [640, 53], [True, False]
    )
    assert budgets == [84, 53]  # 339 rounds down to a 336/84 paired cap.
    state.to_move[0] = 0
    advantage, _ = worker._entropy_budgets(
        roots, response, state, [(2, -2), (0, 0)], [640, 53], [True, False]
    )
    assert advantage[0] == 4 * budgets[0]
    assert ":planned_base=640:effective_base=336:full_cap=336" in evidence[0]
    assert ":planned_base=53:effective_base=53:full_cap=640" in evidence[1]
    policy = configuration.search_execution.full_budget
    assert policy.adjusted_cap(640, 0, minimum_cap=53 * 8, quantum=8) == 424
    assert policy.adjusted_cap(100, 0, minimum_cap=53 * 8, quantum=8) == 100
    assert policy.adjusted_cap(640, 0.35, minimum_cap=53) == 640
    assert FullSearchBudgetConfig().adjusted_cap(639, 0, quantum=4) == 639


def test_reuse_context_changes_with_model_identity_object_and_value_semantics():
    base = SimpleNamespace(
        model_identity="immutable",
        model_version="v1",
        model_step=1,
        model=object(),
        config=SimpleNamespace(precision="fp32", legacy_features=False),
    )
    wrapped = SimpleNamespace(
        base=base, config=SimpleNamespace(score_utility_weight=0.1)
    )
    context = search_model_context(wrapped)
    assert context == search_model_context(wrapped)
    wrapped.config.score_utility_weight = 0.2
    assert context != search_model_context(wrapped)
    wrapped.config.score_utility_weight = 0.1
    base.config.score_utility_weight = 0.3
    assert context != search_model_context(wrapped)
    del base.config.score_utility_weight
    wrapped.score_utility_weight = 0.4
    assert context != search_model_context(wrapped)
    del wrapped.score_utility_weight
    base.config.feature_schema_version = 3
    assert context != search_model_context(wrapped)
    del base.config.feature_schema_version
    base.config.precision = "bf16"
    assert context != search_model_context(wrapped)
    base.config.precision = "fp32"
    base.model_version = "v2"
    assert context != search_model_context(wrapped)
    base.model_version = "v1"
    wrapped.base = SimpleNamespace(**vars(base))
    assert context != search_model_context(wrapped)
    assert search_batch_row_limit(wrapped, 4) == 4
    wrapped.broker = SimpleNamespace(max_batch_rows=2)
    assert search_batch_row_limit(wrapped, 4) == 2
    with pytest.raises(ValueError, match="immutable model identity"):
        search_model_context(SimpleNamespace())
