from copy import deepcopy
from dataclasses import replace
import json
from pathlib import Path

import pytest
import yaml

from startrain import learner as learner_module
from startrain.config import (
    ConfigError,
    LearnerConfig,
    SchedulerConfig,
    TrainConfig,
    load_config,
)
from startrain.config_compatibility import (
    compatible_config_epoch_payloads,
    without_training_execution_defaults,
)
from startrain.replay_store import ReplayStore
from test_pipeline_core import (
    append_replay,
    make_replay_sample,
    make_test_learner,
    run_identity,
)


FIELD = "share_homogeneous_geometry"


@pytest.mark.parametrize("value", [0, 1, 0.0, None, "false", [], {}])
def test_geometry_sharing_requires_actual_boolean_in_config_and_yaml(tmp_path, value):
    assert TrainConfig().share_homogeneous_geometry is False
    with pytest.raises(ConfigError, match=f"{FIELD} must be boolean"):
        TrainConfig(**{FIELD: value})
    profile = Path(__file__).parents[1] / "configs/small.yaml"
    raw = yaml.safe_load(profile.read_text())
    raw["train"][FIELD] = value
    path = tmp_path / "invalid-geometry.yaml"
    path.write_text(yaml.safe_dump(raw))
    with pytest.raises(ConfigError, match=f"{FIELD} must be boolean"):
        load_config(path)


@pytest.mark.parametrize(
    "preserve,geometry",
    [
        (False, False),
        (True, False),
        (False, True),
        (True, True),
        (0, False),
        (False, 0),
        ("false", True),
        (True, None),
    ],
)
def test_shared_release_epoch_omits_only_exact_false_and_does_not_expand_subsets(
    preserve, geometry
):
    payload = {
        "selfplay": {"preserve_interrupted_policy": preserve},
        "train": {FIELD: geometry},
    }
    original = deepcopy(payload)
    previous = without_training_execution_defaults(payload)
    representations = compatible_config_epoch_payloads(payload)
    distinct = {json.dumps(row, sort_keys=True) for row in representations}
    assert distinct == {
        json.dumps(payload, sort_keys=True),
        json.dumps(previous, sort_keys=True),
    }
    assert len(distinct) <= 2
    for row in representations:
        for section, field, value in (
            ("selfplay", "preserve_interrupted_policy", preserve),
            ("train", FIELD, geometry),
        ):
            if value is not False:
                assert row[section][field] == value
                assert type(row[section][field]) is type(value)
    assert payload == original
    if preserve is False and geometry is False:
        assert previous == {"selfplay": {}, "train": {}}


def test_geometry_yaml_optin_preserves_all_other_training_authority(tmp_path):
    original = load_config(Path(__file__).parents[1] / "configs/small.yaml")
    enabled = replace(
        original, train=replace(original.train, share_homogeneous_geometry=True)
    )
    path = tmp_path / "geometry.yaml"
    path.write_text(yaml.safe_dump(enabled.as_dict()))
    restored = load_config(path)
    assert restored == enabled
    assert replace(restored.train, share_homogeneous_geometry=False) == original.train
    assert all(
        row["train"][FIELD] is True
        for row in compatible_config_epoch_payloads(restored.as_dict())
    )


@pytest.mark.parametrize("enabled", [False, True])
def test_real_learner_forwards_geometry_option_to_training_step(
    tmp_path, monkeypatch, enabled
):
    identity = run_identity(tmp_path)
    observed = []
    original = learner_module.train_step

    def step(*args, **kwargs):
        observed.append(kwargs[FIELD])
        return original(*args, **kwargs)

    monkeypatch.setattr(learner_module, "train_step", step)
    with ReplayStore(tmp_path / "replay") as store:
        generation = store.lease_generation(identity, "actor-test")
        append_replay(
            store,
            [
                make_replay_sample(
                    identity=identity,
                    generation=generation,
                    game_id=f"geometry-{index}",
                )
                for index in range(2)
            ],
            identity,
            model_step=0,
            generation=generation,
        )
        learner = make_test_learner(
            store,
            identity,
            tmp_path / "learner",
            learner_config=LearnerConfig(
                steps=1,
                recent_samples_per_ring=8,
                max_replay_lag_steps=10,
                steps_per_window=1,
                candidate_interval=1,
                metrics_interval=1,
                device="cpu",
            ),
            train_config=TrainConfig(
                per_rank_batch_size=2,
                share_homogeneous_geometry=enabled,
                scheduler=SchedulerConfig(warmup_steps=0, total_steps=4),
            ),
        )
        assert learner.run(steps=1) == 1
        assert observed == [enabled]
        assert learner.examples_consumed == 2
