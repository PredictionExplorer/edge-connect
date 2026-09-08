from copy import deepcopy
from dataclasses import asdict, replace
import hashlib
import json
from pathlib import Path

import pytest
import yaml

from scripts import migrate_continuous_profile as migration
from startrain.balanced_evaluation import evaluation_contract
from startrain.config import ConfigError, TrainConfig, load_config
from startrain.config_compatibility import (
    compatible_config_epoch_payloads,
    without_gradient_clipping_defaults,
)
from startrain.gradient_clipping import GradientClippingConfig
from startrain.orchestration import _compatible_autonomous_config_sha256s
from test_continuous_profile_migration import (
    _fixture,
    _with_update_to_data,
    _write_json,
)


def profile():
    return load_config(
        Path(__file__).parents[1] / "configs/h100-8gpu-largest-board-priority.yaml"
    )


def test_train_config_requires_typed_clipping_and_boolean_diagnostics():
    assert TrainConfig().gradient_clipping == GradientClippingConfig()
    assert TrainConfig().gradient_diagnostics is False
    for value in (None, 0, 1, "false", [], {}):
        with pytest.raises(ConfigError, match="gradient_diagnostics.*boolean"):
            TrainConfig(gradient_diagnostics=value)
    with pytest.raises(ConfigError, match="GradientClippingConfig"):
        TrainConfig(gradient_clipping={"mode": "global"})
    selected = GradientClippingConfig(
        mode="adagc", beta=0.95, multiplier=1.2, warmup_steps=12
    )
    assert (
        TrainConfig(
            gradient_clipping=selected, gradient_diagnostics=True
        ).gradient_clipping
        is selected
    )


@pytest.mark.parametrize("mode", ("global", "adagc"))
def test_nested_clipping_yaml_roundtrip_preserves_all_training_authority(
    tmp_path, mode
):
    original = profile()
    selected = replace(
        original,
        train=replace(
            original.train,
            gradient_clipping=GradientClippingConfig(
                mode=mode, beta=0.9, multiplier=1.1, warmup_steps=5
            ),
            gradient_diagnostics=True,
        ),
    )
    path = tmp_path / "profile.yaml"
    path.write_text(yaml.safe_dump(selected.as_dict(), sort_keys=False))
    restored = load_config(path)
    assert restored == selected
    assert isinstance(restored.train.gradient_clipping, GradientClippingConfig)
    assert restored.model == original.model and restored.optimizer == original.optimizer
    assert restored.train.scheduler == original.train.scheduler


def test_yaml_rejects_unknown_or_malformed_clipping_settings(tmp_path):
    baseline = profile().as_dict()
    for settings in (
        None,
        "adagc",
        {"mode": "unknown"},
        {"unknown": 1},
        {"warmup_steps": True},
        {"beta": float("nan")},
    ):
        changed = deepcopy(baseline)
        changed["train"]["gradient_clipping"] = settings
        path = tmp_path / "bad.yaml"
        path.write_text(yaml.safe_dump(changed))
        with pytest.raises(ConfigError, match="GradientClippingConfig"):
            load_config(path)


def test_only_exact_typed_defaults_can_be_omitted_without_mutating_source():
    current = profile().as_dict()
    current["train"]["gradient_clipping"] = asdict(GradientClippingConfig())
    current["train"]["gradient_diagnostics"] = False
    before = deepcopy(current)
    expected = deepcopy(current)
    del expected["train"]["gradient_clipping"]
    del expected["train"]["gradient_diagnostics"]
    assert without_gradient_clipping_defaults(current) == expected
    assert current == before
    assert current in compatible_config_epoch_payloads(current)
    assert expected in compatible_config_epoch_payloads(current)


def test_unknown_partial_or_untyped_lookalikes_keep_hash_authority():
    defaults = asdict(GradientClippingConfig())
    variants = (
        {**defaults, "warmup_steps": 100.0},
        {**defaults, "beta": False},
        {**defaults, "unknown": 0},
        {"mode": "global"},
        None,
        "global",
    )
    for value in variants:
        payload = {"train": {"gradient_clipping": value, "gradient_diagnostics": 0}}
        assert without_gradient_clipping_defaults(payload) == payload
        for candidate in compatible_config_epoch_payloads(payload):
            assert candidate["train"] == payload["train"]
    for value in (None, 0, "false", [], {}):
        payload = {"train": {"gradient_diagnostics": value}}
        assert without_gradient_clipping_defaults(payload) == payload


def test_nondefault_adaptive_settings_and_enabled_diagnostics_are_never_omitted():
    current = profile().as_dict()
    current["train"]["gradient_clipping"] = asdict(GradientClippingConfig(mode="adagc"))
    current["train"]["gradient_diagnostics"] = True
    for candidate in compatible_config_epoch_payloads(current):
        assert (
            candidate["train"]["gradient_clipping"]
            == current["train"]["gradient_clipping"]
        )
        assert candidate["train"]["gradient_diagnostics"] is True
    for field, value in (("beta", 0.9), ("multiplier", 1.2), ("warmup_steps", 10)):
        control = profile().as_dict()
        control["train"]["gradient_clipping"][field] = value
        assert (
            without_gradient_clipping_defaults(control)["train"]["gradient_clipping"]
            == control["train"]["gradient_clipping"]
        )


def test_pre_clipping_baseline_hash_remains_accepted_but_adaptive_changes_do_not():
    current = profile()
    legacy = current.as_dict()
    del legacy["train"]["gradient_clipping"]
    del legacy["train"]["gradient_diagnostics"]
    legacy_hash = hashlib.sha256(
        json.dumps(legacy, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    assert legacy_hash in _compatible_autonomous_config_sha256s(current)
    changed = replace(current, train=replace(current.train, gradient_diagnostics=True))
    assert legacy_hash not in _compatible_autonomous_config_sha256s(changed)
    changed = replace(
        current,
        train=replace(
            current.train, gradient_clipping=GradientClippingConfig(mode="adagc")
        ),
    )
    assert legacy_hash not in _compatible_autonomous_config_sha256s(changed)


def migration_fixture(tmp_path):
    fixture = _fixture(tmp_path, "h100-8gpu-largest-board-priority.yaml")
    segment = {
        "schema_version": 1,
        "run_id": "continuous-test-run",
        "generation_family": "family-continuous-test",
        "target_updates_per_new_sample": 1.5,
        "baseline_examples_consumed": 1024,
        "baseline_committed_replay_samples": 2048,
        "created_ns": 5,
    }
    _with_update_to_data(
        fixture,
        old_target=1.5,
        new_target=1.5,
        old_intervals=(3_000_000, 1_500_000),
        new_intervals=(3_000_000, 1_500_000),
        segment=segment,
    )
    fixture.candidate_profile.chmod(0o644)
    old = load_config(fixture.old_profile)
    evidence = {
        "arena/promotion-status.json": {"terminal": False, "candidate_step": 100},
        "arena/pending-balanced.json": {
            "terminal": False,
            "result_kind": "promotion",
            "evaluation_contract": evaluation_contract(old.arena),
            "pairs": [{"ring": 10, "pair": 0}],
        },
        "arena/pending-balanced.resume.json": {
            "arena_state": {"game_states": [{"actions": [1, 2, 3]}]}
        },
        "arena/control.json": {"candidate_identity": "pending", "next_wave": 2},
        "arena/.historical-cooldown.json": {"not_before_ns": 100000},
        "arena/.promotion-cooldown.json": {"not_before_ns": 200000},
    }
    for name, payload in evidence.items():
        _write_json(fixture.root / name, payload)
    paths = [
        fixture.root / name
        for name in (
            *evidence,
            "learner/utd-segment.json",
            "learner/recovery.json",
            "learner/champion.json",
            "run.json",
        )
    ]
    paths.append(fixture.checkpoint)
    return fixture, old, {path: path.read_bytes() for path in paths}


@pytest.mark.parametrize("adaptive", (False, True))
def test_clipping_migration_preserves_pending_arena_utd_and_optimizer_state(
    tmp_path, adaptive
):
    fixture, old, preserved = migration_fixture(tmp_path)
    target = yaml.safe_load(fixture.old_profile.read_text())
    target["train"]["gradient_diagnostics"] = True
    if adaptive:
        target["train"]["gradient_clipping"] = asdict(
            GradientClippingConfig(
                mode="adagc", beta=0.9, multiplier=1.1, warmup_steps=10
            )
        )
    fixture.candidate_profile.write_text(yaml.safe_dump(target))
    result = migration.migrate_continuous_profile(fixture.request, apply=True)
    active = load_config(fixture.root / fixture.target_name)
    expected_paths = {"train.gradient_diagnostics"}
    if adaptive:
        expected_paths.update(
            f"train.gradient_clipping.{name}"
            for name in ("mode", "beta", "multiplier", "warmup_steps")
        )
    assert {change["path"] for change in result["changes"]} == expected_paths
    assert result["utd_segment"] is None
    for section in (
        "model",
        "game",
        "loss",
        "optimizer",
        "learner",
        "selfplay",
        "arena",
    ):
        assert getattr(active, section) == getattr(old, section)
    assert active.train.scheduler == old.train.scheduler
    assert active.train.gradient_clip_norm == old.train.gradient_clip_norm
    assert active.train.per_rank_batch_size == old.train.per_rank_batch_size
    assert active.orchestration == old.orchestration
    assert {path: path.read_bytes() for path in preserved} == preserved
    record = json.loads((fixture.root / "continuous-migrations.jsonl").read_text())
    assert (
        "evaluation_contract_transition" not in record and "utd_segment" not in record
    )


def test_clipping_allowlist_does_not_allow_model_optimizer_or_scheduler_changes(
    tmp_path,
):
    fixture, _, preserved = migration_fixture(tmp_path)
    baseline = yaml.safe_load(fixture.old_profile.read_text())
    for section, key, value in (
        ("model", "dropout", 0.1),
        ("optimizer", "adamw_lr", 0.0006),
        ("scheduler", "warmup_steps", 4000),
    ):
        target = deepcopy(baseline)
        target["train"]["gradient_clipping"] = asdict(
            GradientClippingConfig(mode="adagc")
        )
        parent = (
            target["train"]["scheduler"] if section == "scheduler" else target[section]
        )
        parent[key] = value
        fixture.candidate_profile.write_text(yaml.safe_dump(target))
        with pytest.raises(migration.MigrationError):
            migration.migrate_continuous_profile(fixture.request, apply=True)
        assert {path: path.read_bytes() for path in preserved} == preserved
