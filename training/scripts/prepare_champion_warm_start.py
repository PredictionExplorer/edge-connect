#!/usr/bin/env python3
"""Prepare a weights-only champion cutover with fresh training state."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path
from typing import Any

from scripts.preflight_run_state import run_state_preflight, state_apply_guard
from startrain.checkpoint import (
    ExponentialMovingAverage,
    ModelManifest,
    ResumeCheckpoint,
    create_recovery_checkpoint_artifact,
    inspect_checkpoint,
    load_ema_weights_for_warm_start,
    load_model_manifest,
    load_resume_cutover,
    publish_recovery_checkpoint,
    verify_file,
    write_model_pointer,
    write_resume_cutover,
)
from startrain.config import load_config
from startrain.manifest_selection import (
    ManifestEvidence,
    ManifestSelectionError,
    selected_manifest_in_copy,
    verify_selection_snapshot,
)
from startrain.model import GraphResTNet
from startrain.optim import build_optimizer
from startrain.runtime import (
    SELECTION_CUTOVER_FORMAT,
    SELECTION_CUTOVER_SCHEMA_VERSION,
    atomic_json,
    load_run_identity,
)
from startrain.training import build_scheduler

WARM_START_FORMAT = "startrain.champion-warm-start"
WARM_START_SCHEMA_VERSION = 1


class WarmStartError(RuntimeError):
    """A strict warm-start preparation failure."""


def _read_json(path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise WarmStartError(f"cannot read {path}: {exc}") from exc
    if not isinstance(payload, dict):
        raise WarmStartError(f"{path} must contain a JSON object")
    return payload


def _existing_warm_start(
    marker_path: Path,
    *,
    run_id: str,
    generation_family: str,
    champion_identity: str,
    profile_sha256: str,
    source_manifest_sha256: str | None = None,
) -> dict[str, Any] | None:
    if not marker_path.is_file():
        return None
    payload = _read_json(marker_path)
    expected = {
        "format": WARM_START_FORMAT,
        "schema_version": WARM_START_SCHEMA_VERSION,
        "run_id": run_id,
        "generation_family": generation_family,
        "source_model_identity": champion_identity,
        "profile_sha256": profile_sha256,
    }
    if source_manifest_sha256 is not None:
        expected["source_manifest_sha256"] = source_manifest_sha256
    if any(payload.get(key) != value for key, value in expected.items()):
        raise WarmStartError(
            "existing champion warm-start marker is incomplete or incompatible"
        )
    if payload.get("status") not in ("prepared", "active"):
        raise WarmStartError("champion warm-start marker status is invalid")
    if payload.get("status") == "prepared":
        return payload
    cutover = load_resume_cutover(
        marker_path.parent / "resume-cutover.json",
        expected_run_id=run_id,
        expected_generation_family=generation_family,
    )
    if (
        payload.get("checkpoint_sha256") != cutover.checkpoint_sha256
        or payload.get("absolute_model_step") != cutover.step
    ):
        raise WarmStartError("warm-start marker disagrees with resume cutover")
    return payload


def _warm_start_source(
    root: Path,
    *,
    run_id: str,
    generation_family: str,
    selection_snapshot: str | Path | None,
    source_manifest: str | Path | None,
) -> tuple[ModelManifest, dict[str, object], bool]:
    """Resolve a strictly verified source without changing any model pointer."""

    champion = load_model_manifest(root / "learner" / "champion.json")
    verified_selection = (
        verify_selection_snapshot(selection_snapshot)
        if selection_snapshot is not None
        else None
    )
    selected_from_snapshot: ModelManifest | None = None
    if verified_selection is not None:
        if (
            verified_selection.plan.source_run_id != run_id
            or verified_selection.plan.source_generation_family != generation_family
        ):
            raise WarmStartError(
                "selection snapshot belongs to another run or generation"
            )
        try:
            selected_from_snapshot = selected_manifest_in_copy(
                verified_selection,
                root,
            )
        except ManifestSelectionError:
            # A direct source manifest can be used when the caller did not make
            # a complete filesystem fork, but it still has to match the exact
            # selected artifact digests below.
            if source_manifest is None:
                selected_from_snapshot = verified_selection.selected_manifest
        ablation_path = root / "ablation.json"
        if ablation_path.is_file():
            ablation = _read_json(ablation_path)
            pinned = ablation.get("source_manifest_selection")
            anchor = ablation.get("anchor")
            if not isinstance(pinned, dict) or not isinstance(anchor, dict):
                raise WarmStartError(
                    "fork metadata does not pin the selection snapshot and anchor"
                )
            expected_snapshot = {
                "source_snapshot_sha256": (verified_selection.snapshot_artifact.sha256),
                "source_snapshot_bytes": verified_selection.snapshot_artifact.bytes,
                "plan_digest": verified_selection.plan.plan_digest,
            }
            if any(
                pinned.get(key) != value for key, value in expected_snapshot.items()
            ) or any(
                anchor.get(key) != value
                for key, value in (
                    (
                        "model_identity",
                        verified_selection.selected_evidence.model_identity,
                    ),
                    (
                        "model_step",
                        verified_selection.selected_evidence.model_step,
                    ),
                )
            ):
                raise WarmStartError(
                    "selection snapshot disagrees with the fork's verified anchor"
                )
            if (
                champion.model_identity
                != verified_selection.selected_evidence.model_identity
                or champion.model_step
                != verified_selection.selected_evidence.model_step
                or champion.manifest_sha256
                != verified_selection.selected_evidence.manifest.sha256
                or champion.checkpoint_sha256
                != verified_selection.selected_evidence.checkpoint.sha256
            ):
                raise WarmStartError(
                    "fork champion no longer matches its verified selection anchor"
                )

    try:
        explicit = (
            load_model_manifest(Path(source_manifest).expanduser().resolve())
            if source_manifest is not None
            else None
        )
    except (OSError, RuntimeError, ValueError) as exc:
        raise WarmStartError(f"source manifest is incompatible: {exc}") from exc
    if explicit is not None and verified_selection is not None:
        selected_evidence = verified_selection.selected_evidence
        explicit_evidence = ManifestEvidence.from_manifest(explicit)
        if (
            explicit_evidence.model_identity != selected_evidence.model_identity
            or explicit_evidence.model_step != selected_evidence.model_step
            or explicit_evidence.manifest.sha256 != selected_evidence.manifest.sha256
            or explicit_evidence.manifest.bytes != selected_evidence.manifest.bytes
            or explicit_evidence.checkpoint.sha256
            != selected_evidence.checkpoint.sha256
            or explicit_evidence.checkpoint.bytes != selected_evidence.checkpoint.bytes
        ):
            raise WarmStartError(
                "source manifest does not match the verified selection snapshot"
            )
    selected = selected_from_snapshot or explicit or champion
    if selected.run_id != run_id or selected.generation_family != generation_family:
        raise WarmStartError("warm-start source belongs to another run or generation")
    evidence = ManifestEvidence.from_manifest(selected)
    source_details: dict[str, object] = {
        "kind": (
            "verified-selection"
            if verified_selection is not None
            else ("explicit-immutable-manifest" if explicit is not None else "champion")
        ),
        "manifest": evidence.as_dict(),
        "selection_snapshot": (
            {
                "path": str(verified_selection.snapshot_path),
                "sha256": verified_selection.snapshot_artifact.sha256,
                "bytes": verified_selection.snapshot_artifact.bytes,
                "plan_digest": verified_selection.plan.plan_digest,
                "source_run_root": verified_selection.plan.source_run_root,
                "fallback_to_champion": {
                    "used": verified_selection.fallback_used,
                    "reason": verified_selection.fallback_reason,
                },
            }
            if verified_selection is not None
            else None
        ),
    }
    return (
        selected,
        source_details,
        (selection_snapshot is not None or source_manifest is not None),
    )


def _prepared_checkpoint(
    learner_root: Path,
    marker: dict[str, Any],
    *,
    run_id: str,
    generation_family: str,
) -> ResumeCheckpoint:
    checkpoint_value = marker.get("checkpoint")
    if not isinstance(checkpoint_value, str) or not checkpoint_value:
        raise WarmStartError("prepared checkpoint path is invalid")
    checkpoint = (learner_root / checkpoint_value).resolve()
    if checkpoint.parent != (learner_root / "recovery").resolve():
        raise WarmStartError("prepared checkpoint escaped the recovery directory")
    digest = marker.get("checkpoint_sha256")
    size = marker.get("checkpoint_bytes")
    step = marker.get("absolute_model_step")
    if (
        not isinstance(digest, str)
        or len(digest) != 64
        or isinstance(size, bool)
        or not isinstance(size, int)
        or size <= 0
        or isinstance(step, bool)
        or not isinstance(step, int)
        or step < 0
    ):
        raise WarmStartError("prepared checkpoint metadata is invalid")
    verify_file(
        checkpoint,
        expected_sha256=digest,
        expected_bytes=size,
    )
    metadata = inspect_checkpoint(
        checkpoint,
        expected_run_id=run_id,
        expected_generation_family=generation_family,
        expected_sha256=digest,
        expected_bytes=size,
    )
    extra = metadata.get("extra")
    if (
        metadata.get("step") != step
        or metadata.get("epoch") != 0
        or not isinstance(extra, dict)
        or extra.get("examples_consumed") != marker.get("examples_consumed")
        or extra.get("training_segment") != marker.get("training_segment")
        or not all(
            metadata.get(name) is True
            for name in ("has_optimizer", "has_scheduler", "has_ema")
        )
    ):
        raise WarmStartError("prepared checkpoint payload is incompatible")
    return ResumeCheckpoint(
        checkpoint=checkpoint,
        checkpoint_sha256=digest,
        checkpoint_bytes=size,
        step=step,
        epoch=0,
        run_id=run_id,
        generation_family=generation_family,
        source="prepared-champion-warm-start",
    )


def _activate_selection_cutover(
    learner_root: Path,
    *,
    champion: ModelManifest,
    recovery: ResumeCheckpoint,
) -> None:
    marker_path = learner_root / "selection-cutover.json"
    if not marker_path.is_file():
        return
    marker = _read_json(marker_path)
    if (
        marker.get("format") != SELECTION_CUTOVER_FORMAT
        or marker.get("schema_version") != SELECTION_CUTOVER_SCHEMA_VERSION
        or marker.get("status") not in ("pending", "active")
        or marker.get("selected_model_identity") != champion.model_identity
        or marker.get("selected_model_step") != champion.model_step
    ):
        raise WarmStartError("archived selection cutover marker is incompatible")
    active = {
        **marker,
        "status": "active",
        "warm_start_checkpoint_sha256": recovery.checkpoint_sha256,
        "warm_start_checkpoint_bytes": recovery.checkpoint_bytes,
        "activated_ns": time.time_ns(),
    }
    atomic_json(marker_path, active)


def _activate_prepared_warm_start(
    learner_root: Path,
    marker_path: Path,
    marker: dict[str, Any],
    *,
    champion: ModelManifest,
    selfplay_enabled: bool,
    run_id: str,
    generation_family: str,
) -> tuple[dict[str, Any], ResumeCheckpoint]:
    prepared = _prepared_checkpoint(
        learner_root,
        marker,
        run_id=run_id,
        generation_family=generation_family,
    )
    examples = marker.get("examples_consumed")
    if isinstance(examples, bool) or not isinstance(examples, int) or examples < 0:
        raise WarmStartError("prepared examples_consumed is invalid")
    cadence = marker.get("cadence")
    utd_segment = marker.get("utd_segment")
    if not isinstance(cadence, dict) or (
        utd_segment is not None and not isinstance(utd_segment, dict)
    ):
        raise WarmStartError("prepared cadence or UTD segment is invalid")
    if utd_segment is not None:
        atomic_json(learner_root / "utd-segment.json", utd_segment)
    atomic_json(learner_root / "cadence.json", cadence)
    write_model_pointer(learner_root / "candidate.json", champion, role="candidate")
    if selfplay_enabled:
        write_model_pointer(
            learner_root / "selfplay" / "candidate.json",
            champion,
            role="candidate",
        )

    cutover_path = learner_root / "resume-cutover.json"
    if cutover_path.is_file():
        existing_cutover = load_resume_cutover(
            cutover_path,
            expected_run_id=run_id,
            expected_generation_family=generation_family,
        )
        if (
            existing_cutover.checkpoint_sha256 == prepared.checkpoint_sha256
            and existing_cutover.step == prepared.step
        ):
            cutover = existing_cutover
            cutover_payload = _read_json(cutover_path)
            cutover_created_ns = cutover_payload.get("created_ns")
            if (
                isinstance(cutover_created_ns, bool)
                or not isinstance(cutover_created_ns, int)
                or cutover_created_ns <= 0
            ):
                raise WarmStartError("existing resume cutover timestamp is invalid")
        else:
            old_payload = _read_json(cutover_path)
            old_created_ns = old_payload.get("created_ns")
            prepared_ns = marker.get("prepared_ns")
            if (
                isinstance(old_created_ns, bool)
                or not isinstance(old_created_ns, int)
                or isinstance(prepared_ns, bool)
                or not isinstance(prepared_ns, int)
                or old_created_ns >= prepared_ns
            ):
                raise WarmStartError(
                    "newer resume cutover conflicts with prepared warm start"
                )
            cutover_created_ns = max(time.time_ns(), prepared_ns + 1)
            cutover = write_resume_cutover(
                learner_root,
                manifest=prepared,
                run_id=run_id,
                generation_family=generation_family,
                created_ns=cutover_created_ns,
            )
    else:
        cutover_created_ns = time.time_ns()
        cutover = write_resume_cutover(
            learner_root,
            manifest=prepared,
            run_id=run_id,
            generation_family=generation_family,
            created_ns=cutover_created_ns,
        )
    publish_recovery_checkpoint(
        learner_root,
        recovery=prepared,
        examples_consumed=examples,
        updated_ns=cutover_created_ns,
    )
    active = dict(marker)
    active.update(
        {
            "status": "active",
            "cutover_created_ns": cutover_created_ns,
            "activated_ns": time.time_ns(),
        }
    )
    atomic_json(marker_path, active)
    _activate_selection_cutover(
        learner_root,
        champion=champion,
        recovery=prepared,
    )
    return active, cutover


def prepare_champion_warm_start(
    run_root: str | Path,
    profile: str | Path,
    *,
    apply: bool = False,
    initial_replay_credit: int | None = None,
    replace_existing: bool = False,
    selection_snapshot: str | Path | None = None,
    source_manifest: str | Path | None = None,
) -> dict[str, object]:
    """Validate and optionally activate a fresh optimizer segment."""

    root = Path(run_root).expanduser().resolve()
    profile_path = Path(profile).expanduser().resolve()
    experiment = load_config(profile_path)
    identity = load_run_identity(root / "run.json")
    learner_root = root / "learner"
    warm_source, source_details, strict_source_pin = _warm_start_source(
        root,
        run_id=identity.run_id,
        generation_family=identity.generation_family,
        selection_snapshot=selection_snapshot,
        source_manifest=source_manifest,
    )
    if apply and strict_source_pin and not (root / "ablation.json").is_file():
        raise WarmStartError(
            "verified archived-manifest warm starts may only be applied to a fork"
        )
    if apply and strict_source_pin:
        source_artifact = (warm_source.artifact_manifest or warm_source.path).resolve()
        if source_artifact.parent != (root / "learner" / "manifests").resolve():
            raise WarmStartError(
                "warm-start source manifest must be isolated inside the fork"
            )
    snapshot_details = source_details.get("selection_snapshot")
    if (
        apply
        and isinstance(snapshot_details, dict)
        and Path(str(snapshot_details.get("source_run_root"))).resolve() == root
    ):
        raise WarmStartError(
            "selection snapshot source cannot be mutated as its own warm-start fork"
        )
    preflight = run_state_preflight(root, profile_path, apply=apply)
    source_metadata = inspect_checkpoint(
        warm_source.checkpoint,
        expected_model_config=experiment.as_dict()["model"],
        expected_game_config=experiment.as_dict()["game"],
        expected_run_id=identity.run_id,
        expected_generation_family=identity.generation_family,
        expected_sha256=warm_source.checkpoint_sha256,
        expected_bytes=warm_source.checkpoint_bytes,
    )
    source_extra = source_metadata.get("extra")
    source_examples = (
        source_extra.get("examples_consumed")
        if isinstance(source_extra, dict)
        else None
    )
    if (
        source_metadata.get("step") != warm_source.model_step
        or source_metadata.get("has_ema") is not True
        or isinstance(source_examples, bool)
        or not isinstance(source_examples, int)
        or source_examples < 0
    ):
        raise WarmStartError(
            "warm-start source lacks a compatible examples/EMA boundary"
        )
    profile_report = preflight.get("profile")
    if not isinstance(profile_report, dict) or not isinstance(
        profile_report.get("sha256"), str
    ):
        raise WarmStartError("preflight returned invalid profile identity")
    profile_sha256 = str(profile_report["sha256"])
    marker_path = learner_root / "champion-warm-start.json"
    replaced_marker: dict[str, Any] | None = None
    try:
        existing = _existing_warm_start(
            marker_path,
            run_id=identity.run_id,
            generation_family=identity.generation_family,
            champion_identity=warm_source.model_identity,
            profile_sha256=profile_sha256,
            source_manifest_sha256=(
                warm_source.manifest_sha256 if strict_source_pin else None
            ),
        )
    except WarmStartError:
        if not replace_existing or not marker_path.is_file():
            raise
        candidate = _read_json(marker_path)
        old_identity = candidate.get("source_model_identity")
        old_profile_sha256 = candidate.get("profile_sha256")
        if (
            candidate.get("format") != WARM_START_FORMAT
            or candidate.get("schema_version") != WARM_START_SCHEMA_VERSION
            or candidate.get("run_id") != identity.run_id
            or candidate.get("generation_family") != identity.generation_family
            or candidate.get("status") != "active"
            or not isinstance(old_identity, str)
            or not isinstance(old_profile_sha256, str)
        ):
            raise WarmStartError(
                "existing champion warm-start marker cannot be safely replaced"
            ) from None
        # A prior active marker can legitimately disagree with the current
        # cutover after later promotions or a plateau reset. The run-state
        # preflight above already verified the current recovery/cutover chain;
        # archive the historical marker and build a new segment from the
        # current champion.
        replaced_marker = candidate
        existing = None
    if existing is not None:
        if existing.get("status") == "prepared" and apply:
            selfplay_enabled = (
                experiment.learner.selfplay_snapshot_interval_examples is not None
            )
            with state_apply_guard(root):
                active, cutover = _activate_prepared_warm_start(
                    learner_root,
                    marker_path,
                    existing,
                    champion=warm_source,
                    selfplay_enabled=selfplay_enabled,
                    run_id=identity.run_id,
                    generation_family=identity.generation_family,
                )
            return {
                "status": "ok",
                "mode": "resumed-apply",
                "run_root": str(root),
                "warm_start": active,
                "resume_cutover": {
                    "checkpoint": str(cutover.checkpoint),
                    "checkpoint_sha256": cutover.checkpoint_sha256,
                    "step": cutover.step,
                },
                "preflight": preflight,
            }
        if existing.get("status") == "active" and apply:
            prepared = _prepared_checkpoint(
                learner_root,
                existing,
                run_id=identity.run_id,
                generation_family=identity.generation_family,
            )
            with state_apply_guard(root):
                _activate_selection_cutover(
                    learner_root,
                    champion=warm_source,
                    recovery=prepared,
                )
        return {
            "status": "ok",
            "mode": (
                "already-active"
                if apply
                else (
                    "prepared-needs-apply"
                    if existing.get("status") == "prepared"
                    else "dry-run"
                )
            ),
            "run_root": str(root),
            "warm_start": existing,
            "preflight": preflight,
        }

    recovery = preflight["recovery"]
    replay = preflight["replay"]
    if not isinstance(recovery, dict) or not isinstance(replay, dict):
        raise WarmStartError("preflight returned invalid durable boundaries")
    committed_value = replay.get("committed_samples")
    if (
        isinstance(committed_value, bool)
        or not isinstance(committed_value, int)
        or committed_value < 0
    ):
        raise WarmStartError("preflight returned invalid numeric boundaries")
    examples_consumed = source_examples
    committed_samples = committed_value
    if initial_replay_credit is None:
        mixture = experiment.orchestration.ring_mixture
        active_weights = mixture.weights_for_step(warm_source.model_step)
        active_rings = (
            tuple(
                ring
                for ring, weight in zip(
                    mixture.rings,
                    active_weights,
                    strict=True,
                )
                if weight > 0
            )
            if active_weights is not None
            else mixture.rings
        )
        ready_samples = replay.get("ready_samples_by_ring")
        if not isinstance(ready_samples, dict):
            raise WarmStartError("preflight omitted ready replay samples by ring")
        active_ready_samples = sum(
            int(ready_samples.get(str(ring), 0)) for ring in active_rings
        )
        replay_credit = min(
            committed_samples,
            active_ready_samples,
            experiment.learner.recent_samples_per_ring * len(active_rings),
        )
    else:
        if (
            isinstance(initial_replay_credit, bool)
            or not isinstance(initial_replay_credit, int)
            or initial_replay_credit < 0
            or initial_replay_credit > committed_samples
        ):
            raise WarmStartError(
                "initial replay credit must be in [0, committed replay samples]"
            )
        replay_credit = initial_replay_credit
    replay_baseline = committed_samples - replay_credit
    target = experiment.learner.target_updates_per_new_sample
    utd_segment: dict[str, object] | None = None
    if target is not None:
        migration_records = preflight.get("migrations")
        if not isinstance(migration_records, list):
            raise WarmStartError("preflight returned invalid migration records")
        if replay.get("history_complete") is not True and not any(
            item.get("name") == "reconcile_legacy_committed_sample_history"
            and item.get("status") == "applied"
            for item in migration_records
            if isinstance(item, dict)
        ):
            raise WarmStartError(
                "warm start requires complete committed-sample history"
            )
        utd_segment = {
            "schema_version": 1,
            "run_id": identity.run_id,
            "generation_family": identity.generation_family,
            "target_updates_per_new_sample": float(target),
            "baseline_examples_consumed": examples_consumed,
            "baseline_committed_replay_samples": replay_baseline,
            "created_ns": time.time_ns(),
        }
    selfplay_enabled = (
        experiment.learner.selfplay_snapshot_interval_examples is not None
    )
    cadence: dict[str, object] = {
        "schema_version": 1,
        "run_id": identity.run_id,
        "generation_family": identity.generation_family,
        "candidate_examples": examples_consumed,
        "selfplay_examples": examples_consumed if selfplay_enabled else None,
        "updated_ns": time.time_ns(),
    }
    training_segment: dict[str, object] = {
        "schema_version": 1,
        "kind": "weights-only-champion-warm-start",
        "run_id": identity.run_id,
        "generation_family": identity.generation_family,
        "source_model_identity": warm_source.model_identity,
        "source_model_step": warm_source.model_step,
        "absolute_model_step": warm_source.model_step,
        "segment_step": 0,
        "baseline_examples_consumed": examples_consumed,
        "baseline_committed_replay_samples": committed_samples,
        "initial_replay_credit": replay_credit,
        "optimizer_state": "fresh",
        "scheduler_state": "fresh",
        "ema_state": "fresh-from-champion-ema",
        "source_selection": source_details,
        "created_ns": time.time_ns(),
    }

    model = GraphResTNet(experiment.model)
    ema = ExponentialMovingAverage(model, decay=experiment.train.ema_decay)
    load_ema_weights_for_warm_start(
        warm_source.checkpoint,
        model=model,
        ema=ema,
        expected_model_config=experiment.as_dict()["model"],
        expected_game_config=experiment.as_dict()["game"],
        expected_run_id=identity.run_id,
        expected_generation_family=identity.generation_family,
        expected_sha256=warm_source.checkpoint_sha256,
        expected_bytes=warm_source.checkpoint_bytes,
    )
    optimizer = build_optimizer(model, experiment.optimizer)
    scheduler = build_scheduler(optimizer, experiment.train.scheduler)
    if optimizer.state or ema.num_updates != 0:
        raise WarmStartError("fresh optimizer/EMA state was not initialized")

    plan: dict[str, object] = {
        "status": "ok",
        "mode": "apply" if apply else "dry-run",
        "run_root": str(root),
        "profile": str(profile_path),
        "profile_sha256": profile_sha256,
        "run_id": identity.run_id,
        "generation_family": identity.generation_family,
        "source_model_identity": warm_source.model_identity,
        "source_model_step": warm_source.model_step,
        "absolute_model_step": warm_source.model_step,
        "source_selection": source_details,
        "examples_consumed": examples_consumed,
        "committed_replay_samples": committed_samples,
        "initial_replay_credit": replay_credit,
        "utd_segment": utd_segment,
        "cadence": cadence,
        "training_segment": training_segment,
        "preflight": preflight,
        "replaces_warm_start": (
            {
                "source_model_identity": replaced_marker.get("source_model_identity"),
                "source_model_step": replaced_marker.get("source_model_step"),
                "profile_sha256": replaced_marker.get("profile_sha256"),
                "status": replaced_marker.get("status"),
            }
            if replaced_marker is not None
            else None
        ),
    }
    if not apply:
        return plan

    with state_apply_guard(root):
        if replaced_marker is not None:
            encoded = json.dumps(
                replaced_marker,
                sort_keys=True,
                separators=(",", ":"),
            ).encode("utf-8")
            history = learner_root / "champion-warm-start-history"
            history.mkdir(parents=True, exist_ok=True)
            archive = history / (
                f"marker-{replaced_marker.get('source_model_step', 'unknown')}-"
                f"{hashlib.sha256(encoded).hexdigest()[:16]}.json"
            )
            if archive.exists() and _read_json(archive) != replaced_marker:
                raise WarmStartError(
                    "existing champion warm-start history artifact is incompatible"
                )
            atomic_json(archive, replaced_marker)
        world_size = max(1, len(experiment.orchestration.learner_gpus))
        prepared = create_recovery_checkpoint_artifact(
            learner_root,
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            ema=ema,
            step=warm_source.model_step,
            epoch=0,
            config=experiment.as_dict(),
            run_id=identity.run_id,
            generation_family=identity.generation_family,
            examples_consumed=examples_consumed,
            global_batch_size=experiment.train.global_batch_size(world_size),
            utd_segment=utd_segment,
            extra={"training_segment": training_segment},
        )
        marker: dict[str, object] = {
            "format": WARM_START_FORMAT,
            "schema_version": WARM_START_SCHEMA_VERSION,
            "status": "prepared",
            "run_id": identity.run_id,
            "generation_family": identity.generation_family,
            "profile": str(profile_path),
            "profile_sha256": profile_sha256,
            "source_model_identity": warm_source.model_identity,
            "source_model_step": warm_source.model_step,
            "source_manifest": str(
                (warm_source.artifact_manifest or warm_source.path).resolve()
            ),
            "source_manifest_sha256": warm_source.manifest_sha256,
            "source_manifest_bytes": warm_source.manifest_bytes,
            "source_checkpoint_sha256": warm_source.checkpoint_sha256,
            "source_checkpoint_bytes": warm_source.checkpoint_bytes,
            "source_selection": source_details,
            "absolute_model_step": prepared.step,
            "examples_consumed": examples_consumed,
            "committed_replay_samples": committed_samples,
            "checkpoint": str(prepared.checkpoint.relative_to(learner_root)),
            "checkpoint_sha256": prepared.checkpoint_sha256,
            "checkpoint_bytes": prepared.checkpoint_bytes,
            "training_segment": training_segment,
            "utd_segment": utd_segment,
            "cadence": cadence,
            "prepared_ns": time.time_ns(),
        }
        if utd_segment is not None:
            atomic_json(learner_root / "utd-segment.json", utd_segment)
        atomic_json(learner_root / "cadence.json", cadence)
        write_model_pointer(
            learner_root / "candidate.json",
            warm_source,
            role="candidate",
        )
        if selfplay_enabled:
            write_model_pointer(
                learner_root / "selfplay" / "candidate.json",
                warm_source,
                role="candidate",
            )
        atomic_json(marker_path, marker)

        marker, cutover = _activate_prepared_warm_start(
            learner_root,
            marker_path,
            marker,
            champion=warm_source,
            selfplay_enabled=selfplay_enabled,
            run_id=identity.run_id,
            generation_family=identity.generation_family,
        )
    plan["warm_start"] = marker
    plan["resume_cutover"] = {
        "checkpoint": str(cutover.checkpoint),
        "checkpoint_sha256": cutover.checkpoint_sha256,
        "step": cutover.step,
    }
    return plan


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-root", required=True, type=Path)
    parser.add_argument("--profile", required=True, type=Path)
    parser.add_argument("--initial-replay-credit", type=int)
    parser.add_argument(
        "--selection-snapshot",
        type=Path,
        help="verified archived-manifest selection snapshot",
    )
    parser.add_argument(
        "--source-manifest",
        type=Path,
        help="immutable source manifest (must match the snapshot when both are set)",
    )
    parser.add_argument(
        "--replace-existing",
        action="store_true",
        help="archive and replace a valid prior champion warm-start marker",
    )
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--dry-run", action="store_true")
    mode.add_argument("--apply", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    mode = "dry-run"
    try:
        arguments = _parser().parse_args(argv)
        mode = "apply" if arguments.apply else "dry-run"
        report = prepare_champion_warm_start(
            arguments.run_root,
            arguments.profile,
            apply=arguments.apply,
            initial_replay_credit=arguments.initial_replay_credit,
            replace_existing=arguments.replace_existing,
            selection_snapshot=arguments.selection_snapshot,
            source_manifest=arguments.source_manifest,
        )
    except Exception as exc:
        print(
            json.dumps(
                {
                    "status": "error",
                    "mode": mode,
                    "error_type": type(exc).__name__,
                    "error": str(exc),
                },
                sort_keys=True,
            ),
            file=sys.stderr,
        )
        return 2
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
