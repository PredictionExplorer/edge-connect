from copy import deepcopy
from dataclasses import replace
from pathlib import Path

import pytest

from startrain.config import ActorInferenceConfig, ConfigError, load_config
from startrain.config_compatibility import (
    compatible_config_epoch_payloads,
    without_broadcast_topology_default,
    without_efficiency_defaults,
    without_evaluation_session_defaults,
    without_pause_strategy_default,
)


def _priority_profile():
    return load_config(
        Path(__file__).parents[1] / "configs/h100-8gpu-largest-board-priority.yaml"
    )


def test_broadcast_topology_is_explicitly_opted_in_only_for_production():
    assert ActorInferenceConfig().preserve_broadcast_topology is False
    for path in (Path(__file__).parents[1] / "configs").glob("h100*.yaml"):
        profile = load_config(path)
        assert (
            profile.orchestration.model_refresh.inference.preserve_broadcast_topology
            is (path.name == "h100-8gpu-largest-board-priority.yaml")
        )


@pytest.mark.parametrize("value", [None, 0, 1, "false", "true", [], {}])
def test_broadcast_topology_requires_a_boolean(value):
    with pytest.raises(ConfigError, match="preserve_broadcast_topology.*boolean"):
        ActorInferenceConfig(preserve_broadcast_topology=value)


def test_broadcast_topology_epoch_preserves_all_previous_release_representations():
    profile = _priority_profile()
    control = profile.orchestration
    profile = replace(
        profile,
        orchestration=replace(
            control,
            model_refresh=replace(
                control.model_refresh,
                inference=replace(
                    control.model_refresh.inference, preserve_broadcast_topology=False
                ),
            ),
        ),
    )
    payload = profile.as_dict()
    untouched = deepcopy(payload)
    previous_release = deepcopy(payload)
    del previous_release["orchestration"]["model_refresh"]["inference"][
        "preserve_broadcast_topology"
    ]
    assert without_broadcast_topology_default(payload) == previous_release
    pre_session = without_evaluation_session_defaults(previous_release)
    previous_epochs = (
        previous_release,
        without_efficiency_defaults(previous_release),
        pre_session,
        without_efficiency_defaults(pre_session),
    )
    for epoch in previous_epochs:
        assert epoch in compatible_config_epoch_payloads(payload)
    assert payload == untouched
    # Existing enabled inference services survive the additive omission.
    assert (
        previous_release["orchestration"]["model_refresh"]["inference"][
            "shared_batching"
        ]
        is True
    )
    assert (
        previous_release["orchestration"]["model_refresh"]["inference"]["deduplicate"]
        is True
    )
    payload["orchestration"]["promotion"]["pause_strategy"] = "terminate"
    for epoch in compatible_config_epoch_payloads(payload):
        assert without_pause_strategy_default(
            epoch
        ) in compatible_config_epoch_payloads(payload)


@pytest.mark.parametrize("value", [True, 0, 1, None, "false"])
def test_epoch_never_omits_enabled_or_untyped_broadcast_authority(value):
    payload = _priority_profile().as_dict()
    payload["orchestration"]["model_refresh"]["inference"][
        "preserve_broadcast_topology"
    ] = value
    assert without_broadcast_topology_default(payload) == payload
    for epoch in compatible_config_epoch_payloads(payload):
        actual = epoch["orchestration"]["model_refresh"]["inference"][
            "preserve_broadcast_topology"
        ]
        assert actual == value and type(actual) is type(value)
