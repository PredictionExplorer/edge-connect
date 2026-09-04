from __future__ import annotations

import json
import os
import sqlite3
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
import yaml

import scripts.prepare_champion_warm_start as warm_start_module
from scripts.prepare_champion_warm_start import prepare_champion_warm_start
from scripts.preflight_run_state import (
    StatePreflightError,
    main as preflight_main,
    run_state_preflight,
)
from startrain.checkpoint import (
    ExponentialMovingAverage,
    inspect_checkpoint,
    load_model_manifest,
    load_resume_cutover,
    write_model_pointer,
    write_recovery_checkpoint,
    write_resume_cutover,
)
from startrain.config import load_config
from startrain.learner import ImmutableModelPublisher
from startrain.model import GraphResTNet
from startrain.optim import build_optimizer
from startrain.replay_store import ReplayStore
from startrain.runtime import RunIdentity, atomic_json, require_launch_ready
from startrain.training import build_scheduler

CONFIGS = Path(__file__).parents[1] / "configs"


def _write_json(path: Path, payload: dict[str, object]) -> None:
    atomic_json(path, payload)


def _profile(root: Path, tmp_path: Path) -> Path:
    raw = yaml.safe_load(
        (CONFIGS / "h100-8gpu-autonomous.yaml").read_text(encoding="utf-8")
    )
    raw["model"].update(
        {
            "width": 8,
            "rrt_groups": 1,
            "attention_heads": 2,
            "kv_heads": 1,
        }
    )
    raw["train"].update(
        {
            "per_rank_batch_size": 1,
            "precision": "fp32",
            "compile": False,
        }
    )
    raw["data"].update({"workers": 0, "pin_memory": False})
    raw["learner"].update(
        {
            "device": "cpu",
            "target_updates_per_new_sample": 1.0,
            "selfplay_snapshot_interval_examples": 10,
            "selfplay_snapshot_warmup_examples": 0,
            "selfplay_snapshot_warmup_interval_examples": None,
        }
    )
    raw["orchestration"]["run_id"] = "run-preflight"
    raw["orchestration"]["directories"]["root"] = str(root)
    profile = tmp_path / "profile.yaml"
    profile.write_text(yaml.safe_dump(raw, sort_keys=False), encoding="utf-8")
    return profile


def _fixture(tmp_path: Path) -> SimpleNamespace:
    root = tmp_path / "run"
    root.mkdir()
    profile = _profile(root, tmp_path)
    experiment = load_config(profile)
    identity = RunIdentity(
        root / "run.json",
        "run-preflight",
        "family-preflight",
        1,
    )
    _write_json(
        identity.path,
        {
            "schema_version": 1,
            "run_id": identity.run_id,
            "generation_family": identity.generation_family,
            "created_ns": identity.created_ns,
        },
    )
    with ReplayStore(root / "replay") as store:
        store.register_run(identity)

    model = GraphResTNet(experiment.model)
    optimizer = build_optimizer(model, experiment.optimizer)
    scheduler = build_scheduler(optimizer, experiment.train.scheduler)
    ema = ExponentialMovingAverage(model, decay=experiment.train.ema_decay)
    with torch.no_grad():
        for parameter in model.parameters():
            parameter.add_(0.25)
    ema.update(model)
    publisher = ImmutableModelPublisher(root / "learner", identity)
    candidate = publisher.publish(
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
        ema=ema,
        step=10,
        epoch=0,
        config=experiment.as_dict(),
        examples_consumed=100,
        global_batch_size=1,
    )
    write_model_pointer(
        root / "learner" / "champion.json",
        candidate,
        role="champion",
        promotion_result="bootstrap",
    )
    write_recovery_checkpoint(
        root / "learner",
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
        ema=ema,
        step=10,
        epoch=0,
        config=experiment.as_dict(),
        run_id=identity.run_id,
        generation_family=identity.generation_family,
        examples_consumed=100,
        global_batch_size=1,
    )
    _write_json(
        root / "learner" / "cadence.json",
        {
            "schema_version": 1,
            "run_id": identity.run_id,
            "generation_family": identity.generation_family,
            "candidate_examples": 100,
            "selfplay_examples": None,
            "updated_ns": 1,
        },
    )
    return SimpleNamespace(
        root=root,
        profile=profile,
        identity=identity,
        experiment=experiment,
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
        ema=ema,
        publisher=publisher,
    )


def test_preflight_dry_run_and_apply_are_safe_and_idempotent(tmp_path: Path) -> None:
    fixture = _fixture(tmp_path)
    cadence_path = fixture.root / "learner" / "cadence.json"
    cadence_before = cadence_path.read_bytes()

    dry_run = run_state_preflight(fixture.root, fixture.profile)

    assert [item["name"] for item in dry_run["migrations"]] == [
        "initialize_selfplay_cadence",
        "initialize_prospective_utd_segment",
    ]
    assert cadence_path.read_bytes() == cadence_before
    assert not (fixture.root / "learner" / "utd-segment.json").exists()
    assert not (fixture.root / "learner" / "selfplay" / "candidate.json").exists()

    applied = run_state_preflight(fixture.root, fixture.profile, apply=True)

    assert all(item["status"] == "applied" for item in applied["migrations"])
    cadence = json.loads(cadence_path.read_text(encoding="utf-8"))
    assert cadence["candidate_examples"] == cadence["selfplay_examples"] == 100
    segment = json.loads(
        (fixture.root / "learner" / "utd-segment.json").read_text(encoding="utf-8")
    )
    assert segment["baseline_examples_consumed"] == 100
    assert segment["baseline_committed_replay_samples"] == 0
    assert (fixture.root / "learner" / "selfplay" / "candidate.json").is_file()

    cadence_applied = cadence_path.read_bytes()
    segment_applied = (fixture.root / "learner" / "utd-segment.json").read_bytes()
    repeated = run_state_preflight(fixture.root, fixture.profile, apply=True)
    assert repeated["migrations"] == []
    assert cadence_path.read_bytes() == cadence_applied
    assert (
        fixture.root / "learner" / "utd-segment.json"
    ).read_bytes() == segment_applied


def test_preflight_accepts_cadence_backed_by_newer_candidate(
    tmp_path: Path,
) -> None:
    fixture = _fixture(tmp_path)
    fixture.publisher.publish(
        model=fixture.model,
        optimizer=fixture.optimizer,
        scheduler=fixture.scheduler,
        ema=fixture.ema,
        step=11,
        epoch=0,
        config=fixture.experiment.as_dict(),
        examples_consumed=120,
        global_batch_size=1,
    )
    cadence_path = fixture.root / "learner" / "cadence.json"
    cadence = json.loads(cadence_path.read_text(encoding="utf-8"))
    cadence["candidate_examples"] = 120
    _write_json(cadence_path, cadence)

    report = run_state_preflight(fixture.root, fixture.profile)

    assert report["status"] == "ok"
    assert report["cadence"]["candidate_examples"] == 120
    assert report["recovery"]["examples_consumed"] == 100


def test_preflight_cli_can_skip_a_genuine_first_launch(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    root = tmp_path / "new-run"
    root.mkdir()
    profile = _profile(root, tmp_path)

    assert (
        preflight_main(
            [
                "--run-root",
                str(root),
                "--profile",
                str(profile),
                "--if-present",
                "--apply",
            ]
        )
        == 0
    )
    report = json.loads(capsys.readouterr().out)
    assert report["status"] == "skipped"
    assert report["reason"] == "first_launch"


def test_preflight_accepts_a_seeded_run_root_without_learner_state(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """A lineage-transferred root has an identity and replay but no learner yet."""

    root = tmp_path / "seeded"
    root.mkdir()
    profile = _profile(root, tmp_path)
    identity = RunIdentity(root / "run.json", "run-preflight", "family-seeded", 1)
    _write_json(
        identity.path,
        {
            "schema_version": 1,
            "run_id": identity.run_id,
            "generation_family": identity.generation_family,
            "created_ns": identity.created_ns,
        },
    )
    with ReplayStore(root / "replay") as store:
        store.register_run(identity)

    for apply in (False, True):
        assert (
            preflight_main(
                [
                    "--run-root",
                    str(root),
                    "--profile",
                    str(profile),
                    "--if-present",
                    *(["--apply"] if apply else []),
                ]
            )
            == 0
        )
        report = json.loads(capsys.readouterr().out)
        assert report["status"] == "ok"
        assert report["reason"] == "seeded_first_launch"
        assert report["mode"] == ("apply" if apply else "dry-run")
        assert report["recovery"] is None and report["migrations"] == []
        assert report["replay"]["committed_samples"] == 0
    assert not (root / "learner").exists()

    # Learner pointers without a recovery pointer are corruption, not a seed.
    (root / "learner").mkdir()
    _write_json(root / "learner" / "champion.json", {"schema_version": 1})
    assert preflight_main(["--run-root", str(root), "--profile", str(profile)]) == 2
    assert "without a recovery pointer" in capsys.readouterr().err


@pytest.mark.parametrize("missing_counter_table", [False, True])
def test_preflight_reconciles_history_only_when_manifest_proves_it(
    tmp_path: Path,
    missing_counter_table: bool,
) -> None:
    fixture = _fixture(tmp_path)
    manifest = fixture.root / "replay" / "manifest.sqlite3"
    with sqlite3.connect(manifest) as connection:
        if missing_counter_table:
            connection.execute("DROP TABLE run_counters")
        else:
            connection.execute("UPDATE run_counters SET history_complete = 0")

    dry_run = run_state_preflight(fixture.root, fixture.profile)
    assert dry_run["replay"]["history_reconciliable"] is True
    assert dry_run["migrations"][0]["name"] == (
        "reconcile_legacy_committed_sample_history"
    )
    if not missing_counter_table:
        with sqlite3.connect(manifest) as connection:
            assert (
                connection.execute(
                    "SELECT history_complete FROM run_counters"
                ).fetchone()[0]
                == 0
            )

    run_state_preflight(fixture.root, fixture.profile, apply=True)

    with sqlite3.connect(manifest) as connection:
        assert (
            connection.execute("SELECT history_complete FROM run_counters").fetchone()[
                0
            ]
            == 1
        )


@pytest.mark.parametrize("state", ["cadence", "utd"])
def test_preflight_rejects_watermarks_ahead_of_recovery(
    tmp_path: Path,
    state: str,
) -> None:
    fixture = _fixture(tmp_path)
    if state == "cadence":
        cadence_path = fixture.root / "learner" / "cadence.json"
        cadence = json.loads(cadence_path.read_text(encoding="utf-8"))
        cadence["candidate_examples"] = 101
        _write_json(cadence_path, cadence)
        expected = "cadence counters are ahead"
    else:
        _write_json(
            fixture.root / "learner" / "utd-segment.json",
            {
                "schema_version": 1,
                "run_id": fixture.identity.run_id,
                "generation_family": fixture.identity.generation_family,
                "target_updates_per_new_sample": 1.0,
                "baseline_examples_consumed": 101,
                "baseline_committed_replay_samples": 0,
            },
        )
        expected = "examples precede"

    with pytest.raises(StatePreflightError, match=expected):
        run_state_preflight(fixture.root, fixture.profile, apply=True)


@pytest.mark.parametrize(
    ("component", "expected"),
    [
        ("profile", "profile run root"),
        ("recovery", "recovery checkpoint is incompatible"),
        ("cutover", "resume cutover is incompatible"),
        ("replay", "replay initialization identity"),
        ("lock", "coordinator lock PID"),
    ],
)
def test_preflight_rejects_identity_recovery_and_live_lock_drift(
    tmp_path: Path,
    component: str,
    expected: str,
) -> None:
    fixture = _fixture(tmp_path)
    if component == "profile":
        raw = yaml.safe_load(fixture.profile.read_text(encoding="utf-8"))
        raw["orchestration"]["directories"]["root"] = str(tmp_path / "other")
        fixture.profile.write_text(
            yaml.safe_dump(raw, sort_keys=False),
            encoding="utf-8",
        )
    elif component == "recovery":
        path = fixture.root / "learner" / "recovery.json"
        payload = json.loads(path.read_text(encoding="utf-8"))
        payload["run_id"] = "run-other"
        _write_json(path, payload)
    elif component == "cutover":
        champion = load_model_manifest(fixture.root / "learner" / "champion.json")
        write_resume_cutover(
            fixture.root / "learner",
            manifest=champion,
            run_id=fixture.identity.run_id,
            generation_family=fixture.identity.generation_family,
        )
        path = fixture.root / "learner" / "resume-cutover.json"
        payload = json.loads(path.read_text(encoding="utf-8"))
        payload["generation_family"] = "family-other"
        _write_json(path, payload)
    elif component == "replay":
        path = fixture.root / "replay" / "initialized.json"
        payload = json.loads(path.read_text(encoding="utf-8"))
        payload["generation_family"] = "family-other"
        _write_json(path, payload)
    else:
        _write_json(
            fixture.root / "coordinator.lock",
            {"pid": os.getpid(), "created_ns": 1},
        )

    with pytest.raises(StatePreflightError, match=expected):
        run_state_preflight(fixture.root, fixture.profile)


def test_champion_warm_start_uses_ema_with_fresh_train_state_and_cutover(
    tmp_path: Path,
) -> None:
    fixture = _fixture(tmp_path)
    old_champion = load_model_manifest(fixture.root / "learner" / "champion.json")
    old_cutover = write_resume_cutover(
        fixture.root / "learner",
        manifest=old_champion,
        run_id=fixture.identity.run_id,
        generation_family=fixture.identity.generation_family,
    )

    planned = prepare_champion_warm_start(fixture.root, fixture.profile)

    assert planned["mode"] == "dry-run"
    assert planned["absolute_model_step"] == 10
    assert planned["examples_consumed"] == 100
    assert not (fixture.root / "learner" / "champion-warm-start.json").exists()
    assert (
        load_resume_cutover(
            fixture.root / "learner" / "resume-cutover.json",
            expected_run_id=fixture.identity.run_id,
            expected_generation_family=fixture.identity.generation_family,
        ).checkpoint_sha256
        == old_cutover.checkpoint_sha256
    )

    applied = prepare_champion_warm_start(
        fixture.root,
        fixture.profile,
        apply=True,
    )

    marker_path = fixture.root / "learner" / "champion-warm-start.json"
    marker = json.loads(marker_path.read_text(encoding="utf-8"))
    assert marker["status"] == "active"
    assert marker["absolute_model_step"] == 10
    assert marker["training_segment"]["segment_step"] == 0
    assert marker["training_segment"]["optimizer_state"] == "fresh"
    checkpoint = fixture.root / "learner" / marker["checkpoint"]
    payload = torch.load(checkpoint, map_location="cpu", weights_only=True)
    champion = json.loads(
        (fixture.root / "learner" / "champion.json").read_text(encoding="utf-8")
    )
    champion_manifest = json.loads(
        (fixture.root / "learner" / champion["manifest"]).read_text(encoding="utf-8")
    )
    champion_checkpoint = (
        fixture.root / "learner" / "manifests" / champion_manifest["checkpoint"]
    ).resolve()
    champion_payload = torch.load(
        champion_checkpoint,
        map_location="cpu",
        weights_only=True,
    )
    for name, expected in champion_payload["ema"]["shadow"].items():
        torch.testing.assert_close(payload["model"][name], expected)
        torch.testing.assert_close(payload["ema"]["shadow"][name], expected)
    assert payload["optimizer"]["state"] == {}
    assert payload["ema"]["num_updates"] == 0
    assert payload["step"] == 10
    assert payload["extra"]["examples_consumed"] == 100
    assert payload["extra"]["training_segment"]["segment_step"] == 0
    assert payload["extra"]["utd_segment"]["baseline_examples_consumed"] == 100
    metadata = inspect_checkpoint(
        checkpoint,
        expected_run_id=fixture.identity.run_id,
        expected_generation_family=fixture.identity.generation_family,
        expected_sha256=marker["checkpoint_sha256"],
        expected_bytes=marker["checkpoint_bytes"],
    )
    assert metadata["has_optimizer"] is True
    cutover = load_resume_cutover(
        fixture.root / "learner" / "resume-cutover.json",
        expected_run_id=fixture.identity.run_id,
        expected_generation_family=fixture.identity.generation_family,
    )
    assert cutover.checkpoint == checkpoint.resolve()
    assert cutover.step == 10
    assert cutover.checkpoint_sha256 != old_cutover.checkpoint_sha256
    assert (
        applied["resume_cutover"]["checkpoint_sha256"] == (marker["checkpoint_sha256"])
    )
    cadence = json.loads(
        (fixture.root / "learner" / "cadence.json").read_text(encoding="utf-8")
    )
    assert cadence["candidate_examples"] == cadence["selfplay_examples"] == 100

    interrupted = dict(marker)
    interrupted["status"] = "prepared"
    interrupted.pop("activated_ns")
    interrupted.pop("cutover_created_ns")
    _write_json(marker_path, interrupted)
    resumed = prepare_champion_warm_start(
        fixture.root,
        fixture.profile,
        apply=True,
    )
    assert resumed["mode"] == "resumed-apply"
    assert json.loads(marker_path.read_text(encoding="utf-8"))["status"] == "active"

    marker_bytes = marker_path.read_bytes()
    repeated = prepare_champion_warm_start(
        fixture.root,
        fixture.profile,
        apply=True,
    )
    assert repeated["mode"] == "already-active"
    assert marker_path.read_bytes() == marker_bytes


def test_champion_warm_start_prepare_only_stages_without_repointing(
    tmp_path: Path,
) -> None:
    fixture = _fixture(tmp_path)
    learner = fixture.root / "learner"
    champion = load_model_manifest(learner / "champion.json")
    old_cutover = write_resume_cutover(
        learner,
        manifest=champion,
        run_id=fixture.identity.run_id,
        generation_family=fixture.identity.generation_family,
    )
    active_paths = (
        learner / "champion.json",
        learner / "candidate.json",
        learner / "recovery.json",
        learner / "resume-cutover.json",
        learner / "cadence.json",
    )
    active_before = {path: path.read_bytes() for path in active_paths}

    prepared = prepare_champion_warm_start(
        fixture.root,
        fixture.profile,
        prepare_only=True,
    )

    assert prepared["mode"] == "prepare-only"
    marker_path = learner / "champion-warm-start.json"
    marker = json.loads(marker_path.read_text(encoding="utf-8"))
    staging_path = learner / "cutover-staging.json"
    staging = json.loads(staging_path.read_text(encoding="utf-8"))
    assert marker["status"] == "prepared"
    assert staging["status"] == "pending"
    assert staging["checkpoint_sha256"] == marker["checkpoint_sha256"]
    assert (
        json.loads((learner / staging["cadence"]["path"]).read_text(encoding="utf-8"))
        == marker["cadence"]
    )
    assert (
        json.loads(
            (learner / staging["utd_segment"]["path"]).read_text(encoding="utf-8")
        )
        == marker["utd_segment"]
    )
    prepared_marker_artifact = learner / staging["prepared_marker"]["path"]
    prepared_marker_bytes = prepared_marker_artifact.read_bytes()
    assert json.loads(prepared_marker_bytes)["status"] == "prepared"
    inspect_checkpoint(
        learner / marker["checkpoint"],
        expected_run_id=fixture.identity.run_id,
        expected_generation_family=fixture.identity.generation_family,
        expected_sha256=marker["checkpoint_sha256"],
        expected_bytes=marker["checkpoint_bytes"],
    )
    assert {path: path.read_bytes() for path in active_paths} == active_before
    assert not (learner / "utd-segment.json").exists()
    assert not (learner / "selfplay" / "candidate.json").exists()
    assert (
        load_resume_cutover(
            learner / "resume-cutover.json",
            expected_run_id=fixture.identity.run_id,
            expected_generation_family=fixture.identity.generation_family,
        ).checkpoint_sha256
        == old_cutover.checkpoint_sha256
    )

    marker_before = marker_path.read_bytes()
    staging_before = staging_path.read_bytes()
    repeated = prepare_champion_warm_start(
        fixture.root,
        fixture.profile,
        prepare_only=True,
    )
    assert repeated["mode"] == "already-prepared"
    assert marker_path.read_bytes() == marker_before
    assert staging_path.read_bytes() == staging_before
    assert {path: path.read_bytes() for path in active_paths} == active_before

    activated = prepare_champion_warm_start(
        fixture.root,
        fixture.profile,
        apply=True,
    )
    assert activated["mode"] == "resumed-apply"
    assert json.loads(marker_path.read_text(encoding="utf-8"))["status"] == "active"
    active_staging = json.loads(staging_path.read_text(encoding="utf-8"))
    assert active_staging["status"] == "active"
    assert prepared_marker_artifact.read_bytes() == prepared_marker_bytes
    assert active_staging["prepared_marker"] == staging["prepared_marker"]
    assert (learner / "utd-segment.json").is_file()
    assert (learner / "selfplay" / "candidate.json").is_file()


def test_direct_apply_crash_leaves_launch_blocked_before_pointer_changes(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fixture = _fixture(tmp_path)
    learner = fixture.root / "learner"
    active_paths = (
        learner / "champion.json",
        learner / "candidate.json",
        learner / "recovery.json",
    )
    active_before = {path: path.read_bytes() for path in active_paths}

    def interrupt(*_args, **_kwargs):
        raise RuntimeError("synthetic activation interruption")

    monkeypatch.setattr(
        warm_start_module,
        "_activate_prepared_warm_start",
        interrupt,
    )
    with pytest.raises(RuntimeError, match="synthetic activation interruption"):
        prepare_champion_warm_start(
            fixture.root,
            fixture.profile,
            apply=True,
        )

    marker = json.loads(
        (learner / "champion-warm-start.json").read_text(encoding="utf-8")
    )
    staging = json.loads((learner / "cutover-staging.json").read_text(encoding="utf-8"))
    assert marker["status"] == "prepared"
    assert staging["status"] == "pending"
    assert {path: path.read_bytes() for path in active_paths} == active_before
    with pytest.raises(RuntimeError, match="prepared but has not been activated"):
        require_launch_ready(learner)


def test_champion_warm_start_uses_champion_examples_not_later_recovery(
    tmp_path: Path,
) -> None:
    fixture = _fixture(tmp_path)
    model = GraphResTNet(fixture.experiment.model)
    optimizer = build_optimizer(model, fixture.experiment.optimizer)
    scheduler = build_scheduler(optimizer, fixture.experiment.train.scheduler)
    ema = ExponentialMovingAverage(model, decay=fixture.experiment.train.ema_decay)
    write_recovery_checkpoint(
        fixture.root / "learner",
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
        ema=ema,
        step=20,
        epoch=1,
        config=fixture.experiment.as_dict(),
        run_id=fixture.identity.run_id,
        generation_family=fixture.identity.generation_family,
        examples_consumed=200,
        global_batch_size=1,
    )

    planned = prepare_champion_warm_start(fixture.root, fixture.profile)

    assert planned["source_model_step"] == 10
    assert planned["examples_consumed"] == 100
    assert planned["initial_replay_credit"] == 0


def test_champion_warm_start_replaces_stale_active_marker_after_valid_preflight(
    tmp_path: Path,
) -> None:
    fixture = _fixture(tmp_path)
    prepare_champion_warm_start(
        fixture.root,
        fixture.profile,
        apply=True,
    )
    marker_path = fixture.root / "learner" / "champion-warm-start.json"
    stale = json.loads(marker_path.read_text(encoding="utf-8"))
    stale["source_model_identity"] = "stale-prior-champion"
    _write_json(marker_path, stale)

    replaced = prepare_champion_warm_start(
        fixture.root,
        fixture.profile,
        apply=True,
        replace_existing=True,
    )
    current = json.loads(marker_path.read_text(encoding="utf-8"))
    champion = load_model_manifest(fixture.root / "learner" / "champion.json")
    history = list(
        (fixture.root / "learner" / "champion-warm-start-history").glob("*.json")
    )

    assert replaced["mode"] == "apply"
    assert current["status"] == "active"
    assert current["source_model_identity"] == champion.model_identity
    assert len(history) == 1
    assert json.loads(history[0].read_text(encoding="utf-8")) == stale
