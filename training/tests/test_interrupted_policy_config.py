from copy import deepcopy
from dataclasses import replace
import json
from pathlib import Path

import pytest
import yaml

from scripts import migrate_continuous_profile as migration
from startrain.balanced_evaluation import evaluation_contract
from startrain.config import load_config
from startrain.config_compatibility import (
    compatible_config_epoch_payloads,
    without_training_execution_defaults,
)
from startrain.selfplay import SelfPlayConfig
from test_continuous_profile_migration import _fixture, _write_json


FIELD = "preserve_interrupted_policy"


def test_disabled_interrupted_policy_preserves_prior_source_epoch_without_mutation():
    profile = load_config(Path(__file__).parents[1] / "configs/small.yaml")
    payload = profile.as_dict()
    original = deepcopy(payload)
    prior = deepcopy(payload)
    assert prior["selfplay"].pop(FIELD) is False
    assert prior["train"].pop("share_homogeneous_geometry") is False
    assert without_training_execution_defaults(payload) == prior
    assert prior in compatible_config_epoch_payloads(payload)
    assert payload == original
    assert without_training_execution_defaults(prior) == prior


@pytest.mark.parametrize("value", [True, 0, 1, "false", None, 0.0])
def test_policy_preservation_optin_and_untyped_values_never_disappear(value):
    payload = {"selfplay": {FIELD: value, "unknown_future_option": "kept"}}
    for representation in (
        without_training_execution_defaults(payload),
        *compatible_config_epoch_payloads(payload),
    ):
        assert representation["selfplay"][FIELD] == value
        assert type(representation["selfplay"][FIELD]) is type(value)
        assert representation["selfplay"]["unknown_future_option"] == "kept"


@pytest.mark.parametrize("value", [0, 1, "false", None, 0.0])
def test_policy_preservation_rejects_nonboolean_profiles(tmp_path, value):
    with pytest.raises(ValueError, match=f"{FIELD} must be boolean"):
        SelfPlayConfig(**{FIELD: value})
    profile = Path(__file__).parents[1] / "configs/small.yaml"
    raw = yaml.safe_load(profile.read_text())
    raw["selfplay"][FIELD] = value
    path = tmp_path / "invalid-policy.yaml"
    path.write_text(yaml.safe_dump(raw))
    with pytest.raises(ValueError, match=f"{FIELD} must be boolean"):
        load_config(path)


@pytest.mark.parametrize(
    "section,field", [("selfplay", FIELD), ("train", "share_homogeneous_geometry")]
)
def test_training_execution_cutover_is_reversible_and_changes_no_training_controls(
    tmp_path,
    section,
    field,
):
    fixture = _fixture(tmp_path, "h100-8gpu-largest-board-priority.yaml")
    original = load_config(fixture.old_profile)
    pending = {
        "terminal": False,
        "candidate_step": 90,
        "evaluation_contract": evaluation_contract(original.arena),
        "pairs": [{"ring": 10, "pair": 0}],
    }
    _write_json(fixture.root / "arena/promotion-status.json", pending)
    protected = [
        fixture.root / name
        for name in (
            "run.json",
            "learner/recovery.json",
            "learner/champion.json",
            "arena/promotion-status.json",
        )
    ] + [fixture.checkpoint]
    utd = fixture.root / "learner/utd-segment.json"
    _write_json(
        utd,
        {
            "schema_version": 1,
            "run_id": "continuous-test-run",
            "generation_family": "family-continuous-test",
            "target_updates_per_new_sample": original.learner.target_updates_per_new_sample,
            "baseline_examples_consumed": 1_024,
            "baseline_committed_replay_samples": 2_048,
            "created_ns": 5,
        },
    )
    protected.append(utd)
    before = {path: path.read_bytes() for path in protected}
    source = fixture.old_profile
    commit = fixture.request.from_source_commit
    for index, enabled in enumerate((True, False)):
        raw = yaml.safe_load(source.read_text())
        raw[section][field] = enabled
        fixture.candidate_profile.write_text(yaml.safe_dump(raw))
        request = replace(
            fixture.request,
            old_profile=source,
            target_profile_name=f"profile-{field}-{index}.yaml",
            from_source_commit=commit,
            to_source_commit=str(index + 2) * 40,
        )
        plan = migration.plan_migration(request)
        migration.apply_migration(plan)
        changed = load_config(plan.target_profile)
        assert getattr(getattr(changed, section), field) is enabled
        assert replace(getattr(changed, section), **{field: False}) == getattr(
            original, section
        )
        for other_section in (
            "game",
            "model",
            "loss",
            "optimizer",
            "train",
            "data",
            "learner",
            "arena",
            "orchestration",
            "selfplay",
        ):
            if other_section != section:
                assert getattr(changed, other_section) == getattr(
                    original, other_section
                )
        assert evaluation_contract(changed.arena) == evaluation_contract(original.arena)
        assert {path: path.read_bytes() for path in protected} == before
        record = json.loads(
            (fixture.root / "continuous-migrations.jsonl").read_text().splitlines()[-1]
        )
        assert record["changes"] == [
            {"path": f"{section}.{field}", "from": not enabled, "to": enabled}
        ]
        assert "utd_segment" not in record
        assert "evaluation_contract_transition" not in record
        source, commit = plan.target_profile, request.to_source_commit
