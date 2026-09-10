from copy import deepcopy
from dataclasses import asdict
from pathlib import Path

import pytest
import yaml

from startrain.config import ActorInferenceConfig, ConfigError, load_config
from startrain.config_compatibility import (
    compatible_config_epoch_payloads,
    without_efficiency_defaults,
    without_inference_execution_defaults,
)


FIELDS = ("compact_inference_gather", "small_batch_graph_buckets")


def nested_inference(payload):
    return payload["orchestration"]["model_refresh"]["inference"]


def test_inference_execution_defaults_preserve_the_complete_prior_payload():
    config = load_config(Path(__file__).parents[1] / "configs/small.yaml")
    current = config.as_dict()
    original = deepcopy(current)
    previous = deepcopy(current)
    for name in FIELDS:
        assert getattr(ActorInferenceConfig(), name) is False
        assert nested_inference(current)[name] is False
        del nested_inference(previous)[name]
    assert without_inference_execution_defaults(current) == previous
    assert previous in compatible_config_epoch_payloads(current)
    assert current == original
    assert without_inference_execution_defaults(previous) == previous
    assert (
        "inference"
        not in without_efficiency_defaults(current)["orchestration"]["model_refresh"]
    )


@pytest.mark.parametrize("name", FIELDS)
@pytest.mark.parametrize("value", [True, 0, 1, None, "false", 0.0])
def test_compatibility_never_erases_enabled_or_untyped_execution_flags(name, value):
    inference = asdict(ActorInferenceConfig())
    inference[name] = value
    inference["unknown_future_option"] = "authoritative"
    payload = {"orchestration": {"model_refresh": {"inference": inference}}}
    for representation in (
        without_efficiency_defaults(payload),
        *compatible_config_epoch_payloads(payload),
    ):
        retained = nested_inference(representation)
        assert retained[name] == value and type(retained[name]) is type(value)
        assert retained["unknown_future_option"] == "authoritative"
    # Unknown keys are not needed to protect an enabled feature from omission
    # of the entire otherwise-disabled inference service.
    del inference["unknown_future_option"]
    for representation in (
        without_efficiency_defaults(payload),
        *compatible_config_epoch_payloads(payload),
    ):
        retained = nested_inference(representation)
        assert retained[name] == value and type(retained[name]) is type(value)


@pytest.mark.parametrize("name", FIELDS)
@pytest.mark.parametrize("value", [0, 1, None, "false", 0.0])
def test_execution_flags_require_actual_booleans_in_profiles(tmp_path, name, value):
    with pytest.raises(ConfigError, match=f"inference.{name} must be boolean"):
        ActorInferenceConfig(**{name: value})
    raw = yaml.safe_load((Path(__file__).parents[1] / "configs/small.yaml").read_text())
    raw.setdefault("orchestration", {}).setdefault("model_refresh", {}).setdefault(
        "inference", {}
    )[name] = value
    path = tmp_path / "invalid-inference.yaml"
    path.write_text(yaml.safe_dump(raw))
    with pytest.raises(ConfigError, match=f"inference.{name} must be boolean"):
        load_config(path)


@pytest.mark.parametrize("name", FIELDS)
def test_enabled_execution_flag_is_preserved_by_profile_loading(tmp_path, name):
    raw = yaml.safe_load((Path(__file__).parents[1] / "configs/small.yaml").read_text())
    raw.setdefault("orchestration", {}).setdefault("model_refresh", {}).setdefault(
        "inference", {}
    )[name] = True
    path = tmp_path / "enabled-inference.yaml"
    path.write_text(yaml.safe_dump(raw))
    config = load_config(path)
    assert getattr(config.orchestration.model_refresh.inference, name) is True
    assert all(
        nested_inference(representation)[name] is True
        for representation in compatible_config_epoch_payloads(config.as_dict())
    )
