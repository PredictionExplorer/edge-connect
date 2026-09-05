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
    assert len(compatible_config_epoch_payloads(payload)) == 4


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
