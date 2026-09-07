from copy import deepcopy
from dataclasses import replace
from pathlib import Path

import pytest

from scripts.validate_continuous_profile import validate_continuous_config
from startrain.config import (
    ConfigError,
    HistoricalEvaluationConfig,
    PromotionConfig,
    load_config,
)
from startrain.config_compatibility import (
    compatible_config_epoch_payloads,
    without_evaluation_session_defaults,
    without_pause_strategy_default,
)


@pytest.mark.parametrize("config_type", [PromotionConfig, HistoricalEvaluationConfig])
@pytest.mark.parametrize(
    "value", [True, False, None, "300", 0, -1, float("nan"), float("inf")]
)
def test_evaluation_sessions_require_finite_positive_durations(config_type, value):
    with pytest.raises(ConfigError, match="session_seconds"):
        config_type(session_seconds=value)


@pytest.mark.parametrize(
    "value", [True, False, None, "1800", -1, float("nan"), float("inf")]
)
def test_historical_cooldown_requires_finite_nonnegative_duration(value):
    with pytest.raises(ConfigError, match="cooldown_seconds"):
        HistoricalEvaluationConfig(cooldown_seconds=value)


def test_typed_scheduling_defaults_and_zero_cooldown():
    assert PromotionConfig().session_seconds == 300.0
    assert PromotionConfig().pause_strategy == "terminate"
    history = HistoricalEvaluationConfig()
    assert history.session_seconds == 300.0
    assert history.cooldown_seconds == 1800.0
    assert replace(history, cooldown_seconds=0).cooldown_seconds == 0


def _efficiency_profile(stage="b"):
    return load_config(
        Path(__file__).parents[1]
        / f"configs/h100-8gpu-variant-efficiency-stage-{stage}.yaml"
    )


@pytest.mark.parametrize("stage", ["a", "b"])
def test_efficiency_profiles_bound_evaluation_sessions(stage):
    config = _efficiency_profile(stage)
    validate_continuous_config(config)
    assert config.orchestration.promotion.session_seconds == 300.0
    assert config.orchestration.promotion.pause_strategy == (
        "suspend" if stage == "b" else "terminate"
    )
    history = config.orchestration.historical_evaluation
    assert (history.session_seconds, history.cooldown_seconds) == (300.0, 1800.0)


@pytest.mark.parametrize(
    ("section", "field", "value", "message"),
    [
        ("promotion", "session_seconds", 901, "bounded promotion"),
        ("historical_evaluation", "session_seconds", 901, "15-minute"),
        ("historical_evaluation", "cooldown_seconds", 299, "catch-up"),
    ],
)
def test_continuous_profiles_reject_evaluation_starvation(
    section, field, value, message
):
    config = _efficiency_profile()
    selected = replace(getattr(config.orchestration, section), **{field: value})
    changed = replace(
        config,
        orchestration=replace(config.orchestration, **{section: selected}),
    )
    with pytest.raises(ValueError, match=message):
        validate_continuous_config(changed)


def test_session_epoch_preserves_evaluation_and_efficiency_contracts():
    payload = _efficiency_profile().as_dict()
    untouched = deepcopy(payload)
    before = without_evaluation_session_defaults(payload)
    expected = deepcopy(payload)
    del expected["orchestration"]["promotion"]["session_seconds"]
    del expected["orchestration"]["historical_evaluation"]["session_seconds"]
    del expected["orchestration"]["historical_evaluation"]["cooldown_seconds"]
    assert before == expected
    assert payload == untouched
    assert before["arena"] == payload["arena"]
    assert before["model"] == payload["model"]
    assert before["optimizer"] == payload["optimizer"]
    assert (
        before["orchestration"]["model_refresh"]
        == payload["orchestration"]["model_refresh"]
    )
    assert payload in compatible_config_epoch_payloads(payload)
    assert before in compatible_config_epoch_payloads(payload)


@pytest.mark.parametrize(
    ("section", "field", "value"),
    [
        ("promotion", "session_seconds", 120.0),
        ("historical_evaluation", "session_seconds", 120.0),
        ("historical_evaluation", "cooldown_seconds", 3600.0),
        ("promotion", "session_seconds", 300),
        ("historical_evaluation", "cooldown_seconds", "1800.0"),
    ],
)
def test_epoch_compatibility_never_erases_nondefault_scheduling(section, field, value):
    payload = _efficiency_profile().as_dict()
    payload["orchestration"][section][field] = value
    for epoch in compatible_config_epoch_payloads(payload):
        assert epoch["orchestration"][section][field] == value
        assert type(epoch["orchestration"][section][field]) is type(value)


@pytest.mark.parametrize("value", [None, True, False, "kill", "SIGSTOP", 0, []])
def test_pause_strategy_rejects_unknown_values(value):
    with pytest.raises(ConfigError, match="pause_strategy"):
        PromotionConfig(pause_strategy=value)


def test_suspend_strategy_requires_actor_pause_sharing():
    config = _efficiency_profile()
    promotion = config.orchestration.promotion
    with pytest.raises(ConfigError, match="actor pause-sharing"):
        replace(
            config.orchestration,
            promotion=replace(promotion, gpu_id=0),
        )
    with pytest.raises(ConfigError, match="pause-sharing"):
        replace(
            config.orchestration,
            promotion=replace(promotion, pause_sharing_mode=False),
        )


@pytest.mark.parametrize(
    ("target_cohorts", "shared_batching"), [(1, True), (1, False), (2, False)]
)
def test_cooperative_suspend_requires_shared_actor_cohorts(
    target_cohorts, shared_batching
):
    config = _efficiency_profile()
    orchestration = config.orchestration
    inference = replace(
        orchestration.model_refresh.inference, shared_batching=shared_batching
    )
    gpus = tuple(
        replace(
            gpu,
            actor_cohorts=(
                target_cohorts if gpu.gpu_id == orchestration.promotion.gpu_id else 1
            ),
        )
        if gpu.role == "actor"
        else gpu
        for gpu in orchestration.gpus
    )
    with pytest.raises(ConfigError, match="shared inference batching"):
        replace(
            orchestration,
            gpus=gpus,
            model_refresh=replace(orchestration.model_refresh, inference=inference),
        )


def test_terminate_strategy_keeps_legacy_single_actor_support():
    config = _efficiency_profile()
    orchestration = config.orchestration
    gpus = tuple(replace(gpu, actor_cohorts=1) for gpu in orchestration.gpus)
    changed = replace(
        orchestration,
        gpus=gpus,
        promotion=replace(orchestration.promotion, pause_strategy="terminate"),
        model_refresh=replace(
            orchestration.model_refresh,
            inference=replace(
                orchestration.model_refresh.inference, shared_batching=False
            ),
        ),
    )
    validate_continuous_config(replace(config, orchestration=changed))


def test_pause_epoch_keeps_session_settings_and_preserves_suspend_authority():
    payload = _efficiency_profile().as_dict()
    payload["orchestration"]["promotion"].update(
        pause_strategy="terminate", session_seconds=900.0
    )
    expected = deepcopy(payload)
    del expected["orchestration"]["promotion"]["pause_strategy"]
    assert without_pause_strategy_default(payload) == expected
    assert expected in compatible_config_epoch_payloads(payload)
    assert payload["orchestration"]["promotion"]["pause_strategy"] == "terminate"
    payload["orchestration"]["promotion"]["pause_strategy"] = "suspend"
    assert without_pause_strategy_default(payload) == payload
    for epoch in compatible_config_epoch_payloads(payload):
        assert epoch["orchestration"]["promotion"]["pause_strategy"] == "suspend"
        assert epoch["orchestration"]["promotion"]["session_seconds"] == 900.0
