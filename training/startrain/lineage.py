"""Lineage transfer: seed a rules-v3 run from the previous lineage's evidence.

The previous lineage (rules v2, feature schema v3, model schema v2) cannot be
warm-started into the variant-capable architecture: the input planes, the
relational bias, and the rule conditioning have no counterpart in its weights.
What transfers instead is its *knowledge*. This module

1. loads the legacy champion with the frozen v3 encoder as a teacher,
2. walks the legacy replay store read-only, upgrading every schema-v4 shard to
   schema v5 (standard variant, ``history_known = False``),
3. attaches the teacher's soft targets (policy over legal actions, outcome and
   score-margin distributions) to every position, and
4. commits the result to the new run's replay store under the new identity.

The learner then distils the teacher through the ``loss.teacher_*`` weights
while its own self-play data (which carries real history) phases the
transferred shards out of the replay window.
"""

from __future__ import annotations

import hashlib
import json
import sqlite3
import time
from collections.abc import Callable, Iterator, Sequence
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

import numpy as np
import torch

from .checkpoint import (
    LEGACY_MODEL_SCHEMA_VERSION,
    legacy_model_config,
    load_legacy_ema_checkpoint,
    sha256_file,
    verify_file,
)
from .contracts import (
    LEGACY_FEATURE_SCHEMA_HASH,
    LEGACY_RULES_HASH,
    LEGACY_RULES_HASH_WIRE,
    TARGET_TEACHER,
)
from .features_v3 import encode_legacy_batch
from .model import GraphResTNet, ModelConfig
from .replay import ReplaySample, read_replay_shard
from .replay_store import ReplayStore
from .runtime import RunIdentity, atomic_json, validate_identifier

LINEAGE_TRANSFER_SCHEMA_VERSION = 1
LINEAGE_TRANSFER_REPORT = "startrain-lineage-transfer"
LEGACY_MANIFEST_SCHEMA_VERSION = 4
TRANSFER_ACTOR_ID = "lineage-transfer"


class LineageTransferError(RuntimeError):
    """A fail-closed lineage transfer error."""


@dataclass(frozen=True, slots=True)
class LegacyShard:
    shard_id: int
    path: Path
    sample_count: int
    ring: int
    phase_min: int
    phase_max: int
    model_step: int
    model_identity: str
    game_count: int
    checksum_sha256: str


@dataclass(frozen=True, slots=True)
class LegacyTeacher:
    model: GraphResTNet
    config: ModelConfig
    identity: str
    checkpoint: Path
    checkpoint_sha256: str
    checkpoint_bytes: int
    step: int
    device: torch.device


def _read_json(path: Path, name: str) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise LineageTransferError(f"cannot read {name} {path}: {error}") from error
    if not isinstance(payload, dict):
        raise LineageTransferError(f"{name} must be a JSON object")
    return payload


def resolve_legacy_champion(pointer: str | Path) -> Path:
    """Resolve a previous-lineage ``champion.json`` pointer to its EMA checkpoint.

    The current manifest parser rejects rules-v2 manifests by design; this
    walks the pointer and immutable manifest with the legacy contract instead
    and verifies the checkpoint digest and size before returning its path.
    """

    source = Path(pointer).resolve()
    payload = _read_json(source, "legacy model pointer")
    if payload.get("format") != "startrain.model-pointer":
        raise LineageTransferError("legacy champion must be a startrain model pointer")
    if payload.get("role") != "champion":
        raise LineageTransferError("lineage transfer requires the champion pointer")
    manifest_value = payload.get("manifest")
    if not isinstance(manifest_value, str) or not manifest_value:
        raise LineageTransferError("legacy pointer manifest path is invalid")
    artifact = Path(manifest_value)
    if not artifact.is_absolute():
        artifact = source.parent / artifact
    artifact = artifact.resolve()
    manifest_sha256 = payload.get("manifest_sha256")
    manifest_bytes = payload.get("manifest_bytes")
    if not isinstance(manifest_sha256, str) or not isinstance(manifest_bytes, int):
        raise LineageTransferError("legacy pointer manifest pin is invalid")
    try:
        verify_file(
            artifact, expected_sha256=manifest_sha256, expected_bytes=manifest_bytes
        )
    except ValueError as error:
        raise LineageTransferError(str(error)) from error
    manifest = _read_json(artifact, "legacy model manifest")
    if (
        manifest.get("format") != "startrain.model-manifest"
        or manifest.get("rules_hash") != LEGACY_RULES_HASH_WIRE
        or manifest.get("feature_schema_hash") != f"{LEGACY_FEATURE_SCHEMA_HASH:016x}"
        or manifest.get("model_schema_version") != LEGACY_MODEL_SCHEMA_VERSION
        or manifest.get("weights") != "ema"
    ):
        raise LineageTransferError(
            "legacy champion manifest is not a previous-lineage EMA publication"
        )
    checkpoint_value = manifest.get("checkpoint")
    checkpoint_sha256 = manifest.get("checkpoint_sha256")
    checkpoint_bytes = manifest.get("checkpoint_bytes")
    if (
        not isinstance(checkpoint_value, str)
        or not checkpoint_value
        or not isinstance(checkpoint_sha256, str)
        or not isinstance(checkpoint_bytes, int)
        or manifest.get("model_identity") != f"sha256-{checkpoint_sha256}"
        or payload.get("model_identity") != manifest.get("model_identity")
    ):
        raise LineageTransferError("legacy champion checkpoint pin is invalid")
    checkpoint = Path(checkpoint_value)
    if not checkpoint.is_absolute():
        checkpoint = artifact.parent / checkpoint
    checkpoint = checkpoint.resolve()
    try:
        verify_file(
            checkpoint,
            expected_sha256=checkpoint_sha256,
            expected_bytes=checkpoint_bytes,
        )
    except ValueError as error:
        raise LineageTransferError(str(error)) from error
    return checkpoint


def load_legacy_teacher(
    checkpoint: str | Path,
    *,
    device: torch.device | str = "cpu",
) -> LegacyTeacher:
    """Load the previous lineage's EMA champion as a frozen teacher."""

    path = Path(checkpoint)
    payload = torch.load(path, map_location="cpu", weights_only=True)
    if not isinstance(payload, dict) or not isinstance(payload.get("config"), dict):
        raise LineageTransferError("legacy checkpoint configuration is missing")
    raw_model = payload["config"].get("model")
    if not isinstance(raw_model, dict):
        raise LineageTransferError("legacy checkpoint model configuration is missing")
    try:
        config = legacy_model_config(raw_model)
    except ValueError as error:
        raise LineageTransferError(str(error)) from error
    target = torch.device(device)
    model = GraphResTNet(config)
    try:
        metadata = load_legacy_ema_checkpoint(
            path,
            model=model,
            expected_model_config=config,
            map_location="cpu",
        )
    except ValueError as error:
        raise LineageTransferError(str(error)) from error
    model.to(target).eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    digest = sha256_file(path)
    return LegacyTeacher(
        model=model,
        config=config,
        identity=f"sha256-{digest}",
        checkpoint=path.resolve(),
        checkpoint_sha256=digest,
        checkpoint_bytes=path.stat().st_size,
        step=int(metadata["step"]),
        device=target,
    )


def list_legacy_shards(
    replay_root: str | Path,
    *,
    rings: Sequence[int] | None = None,
) -> list[LegacyShard]:
    """List the ready previous-lineage shards, oldest first, read-only."""

    root = Path(replay_root).resolve()
    manifest = root / "manifest.sqlite3"
    if not manifest.is_file():
        raise LineageTransferError(f"legacy replay manifest is missing: {manifest}")
    uri = f"file:{manifest}?mode=ro"
    try:
        connection = sqlite3.connect(uri, uri=True)
    except sqlite3.Error as error:
        raise LineageTransferError(f"cannot open legacy manifest: {error}") from error
    try:
        connection.row_factory = sqlite3.Row
        metadata = {
            str(row["key"]): str(row["value"])
            for row in connection.execute("SELECT key, value FROM store_metadata")
        }
        expected = {
            "manifest_schema_version": str(LEGACY_MANIFEST_SCHEMA_VERSION),
            "rules_hash": LEGACY_RULES_HASH_WIRE,
            "feature_schema_hash": f"{LEGACY_FEATURE_SCHEMA_HASH:016x}",
        }
        for key, value in expected.items():
            if metadata.get(key) != value:
                raise LineageTransferError(
                    f"legacy replay store {key} is not the previous lineage"
                )
        clauses = [
            "state = 'ready'",
            "rules_hash = ?",
            "feature_schema_hash = ?",
        ]
        parameters: list[object] = [
            f"{LEGACY_RULES_HASH:016x}",
            f"{LEGACY_FEATURE_SCHEMA_HASH:016x}",
        ]
        if rings is not None:
            requested = tuple(int(ring) for ring in rings)
            if not requested:
                raise LineageTransferError("rings filter must be non-empty")
            clauses.append(f"ring IN ({','.join('?' for _ in requested)})")
            parameters.extend(requested)
        rows = connection.execute(
            f"""
            SELECT id, relative_path, sample_count, ring, phase_min, phase_max,
                   model_step, model_identity, game_count, checksum_sha256
            FROM shards
            WHERE {" AND ".join(clauses)}
            ORDER BY id ASC
            """,
            parameters,
        ).fetchall()
    except sqlite3.Error as error:
        raise LineageTransferError(f"cannot read legacy manifest: {error}") from error
    finally:
        connection.close()
    shards: list[LegacyShard] = []
    for row in rows:
        relative = Path(str(row["relative_path"]))
        if relative.is_absolute():
            raise LineageTransferError("legacy manifest contains an absolute path")
        path = (root / relative).resolve()
        try:
            path.relative_to(root)
        except ValueError as error:
            raise LineageTransferError(
                "legacy shard escapes the replay root"
            ) from error
        shards.append(
            LegacyShard(
                shard_id=int(row["id"]),
                path=path,
                sample_count=int(row["sample_count"]),
                ring=int(row["ring"]),
                phase_min=int(row["phase_min"]),
                phase_max=int(row["phase_max"]),
                model_step=int(row["model_step"]),
                model_identity=str(row["model_identity"]),
                game_count=int(row["game_count"]),
                checksum_sha256=str(row["checksum_sha256"]),
            )
        )
    return shards


def select_recent_legacy_shards(
    shards: Sequence[LegacyShard], *, max_samples: int | None
) -> list[LegacyShard]:
    """Keep the most recent shards up to ``max_samples``, oldest first."""

    if max_samples is None:
        return list(shards)
    if max_samples <= 0:
        raise LineageTransferError("max_samples must be positive")
    selected: list[LegacyShard] = []
    total = 0
    for shard in reversed(list(shards)):
        if total >= max_samples:
            break
        selected.append(shard)
        total += shard.sample_count
    selected.reverse()
    return selected


@torch.no_grad()
def teacher_targets(
    teacher: LegacyTeacher,
    samples: Sequence[ReplaySample],
    *,
    batch_size: int = 512,
) -> list[tuple[np.ndarray, np.ndarray, np.ndarray]]:
    """Soft targets ``(policy, outcome, score_margin)`` for every sample.

    Policies are normalized over the legal actions and stored on the node axis
    (zero on illegal nodes); terminal positions get an all-zero policy since
    the replay contract forbids policies there.
    """

    output: list[tuple[np.ndarray, np.ndarray, np.ndarray]] = []
    if batch_size <= 0:
        raise LineageTransferError("teacher batch size must be positive")
    for start in range(0, len(samples), batch_size):
        chunk = samples[start : start + batch_size]
        positions = [sample.to_position() for sample in chunk]
        encoded = encode_legacy_batch(positions).to(teacher.device)
        result = teacher.model(*encoded.model_args())
        policy_logits = result.policy_logits.float()
        legal = encoded.legal_action_mask
        masked = policy_logits.masked_fill(~legal, float("-inf"))
        policy = torch.softmax(masked, dim=-1)
        policy = torch.where(legal, policy, torch.zeros_like(policy))
        outcome = torch.softmax(result.outcome_logits.float(), dim=-1)
        score_margin = torch.softmax(result.score_margin_logits.float(), dim=-1)
        policy_rows = policy.cpu().numpy()
        outcome_rows = outcome.cpu().numpy()
        margin_rows = score_margin.cpu().numpy()
        for index, sample in enumerate(chunk):
            nodes = sample.stones.shape[0]
            if sample.terminal:
                node_policy = np.zeros(nodes, dtype=np.float32)
            else:
                node_policy = policy_rows[index, :nodes].astype(np.float32)
                mass = float(node_policy.sum())
                if not np.isfinite(mass) or mass <= 0:
                    raise LineageTransferError("teacher policy has no legal mass")
                node_policy /= mass
            output.append(
                (
                    node_policy,
                    outcome_rows[index].astype(np.float32),
                    margin_rows[index].astype(np.float32),
                )
            )
    return output


def label_samples(
    teacher: LegacyTeacher,
    samples: Sequence[ReplaySample],
    *,
    identity: RunIdentity,
    generation: int,
    batch_size: int = 512,
) -> list[ReplaySample]:
    """Attach teacher targets and re-home the samples under the new identity."""

    targets = teacher_targets(teacher, samples, batch_size=batch_size)
    labelled: list[ReplaySample] = []
    for sample, (policy, outcome, margin) in zip(samples, targets, strict=True):
        labelled.append(
            replace(
                sample,
                run_id=identity.run_id,
                generation_family=identity.generation_family,
                actor_id=TRANSFER_ACTOR_ID,
                generation=generation,
                model_identity=teacher.identity,
                search_provenance=(
                    f"lineage-transfer:teacher={teacher.identity}:"
                    f"source={sample.model_identity}:{sample.search_provenance}"
                ),
                target_mask=sample.target_mask | TARGET_TEACHER,
                teacher_policy=policy,
                teacher_outcome=outcome,
                teacher_score_margin=margin,
            )
        )
    return labelled


def iter_legacy_samples(
    shards: Sequence[LegacyShard],
) -> Iterator[tuple[LegacyShard, list[ReplaySample]]]:
    for shard in shards:
        if not shard.path.is_file():
            raise LineageTransferError(f"legacy shard is missing: {shard.path}")
        if sha256_file(shard.path) != shard.checksum_sha256:
            raise LineageTransferError(f"legacy shard checksum mismatch: {shard.path}")
        samples = read_replay_shard(shard.path, allow_legacy=True)
        if len(samples) != shard.sample_count:
            raise LineageTransferError(
                f"legacy shard sample count mismatch: {shard.path}"
            )
        if any(sample.history_known for sample in samples):
            raise LineageTransferError("upgraded legacy samples must not claim history")
        yield shard, samples


def transfer_lineage(
    *,
    teacher: LegacyTeacher,
    legacy_shards: Sequence[LegacyShard],
    replay_root: str | Path,
    identity: RunIdentity,
    batch_size: int = 512,
    progress: Callable[..., None] | None = None,
) -> dict[str, Any]:
    """Commit teacher-labelled copies of the legacy shards to the new store."""

    started_ns = time.time_ns()
    root = Path(replay_root)
    committed_shards = 0
    committed_samples = 0
    committed_games = 0
    samples_by_ring: dict[str, int] = {}
    with ReplayStore(root) as store:
        store.register_run(identity)
        generation = store.lease_generation(identity, TRANSFER_ACTOR_ID)
        for shard, samples in iter_legacy_samples(legacy_shards):
            labelled = label_samples(
                teacher,
                samples,
                identity=identity,
                generation=generation,
                batch_size=batch_size,
            )
            record = store.append(
                labelled,
                phase_min=shard.phase_min,
                phase_max=shard.phase_max,
                model_version=teacher.identity,
                model_step=0,
                model_identity=teacher.identity,
                run_id=identity.run_id,
                generation_family=identity.generation_family,
                actor_id=TRANSFER_ACTOR_ID,
                generation=generation,
            )
            committed_shards += 1
            committed_samples += record.sample_count
            committed_games += record.game_count
            samples_by_ring[str(shard.ring)] = (
                samples_by_ring.get(str(shard.ring), 0) + record.sample_count
            )
            if progress is not None:
                progress(
                    phase="lineage_transfer",
                    legacy_shard=shard.shard_id,
                    committed_shards=committed_shards,
                    committed_samples=committed_samples,
                )
    return {
        "schema_version": LINEAGE_TRANSFER_SCHEMA_VERSION,
        "report": LINEAGE_TRANSFER_REPORT,
        "started_ns": started_ns,
        "completed_ns": time.time_ns(),
        "run_id": identity.run_id,
        "generation_family": identity.generation_family,
        "actor_id": TRANSFER_ACTOR_ID,
        "generation": generation,
        "teacher": {
            "identity": teacher.identity,
            "checkpoint": str(teacher.checkpoint),
            "checkpoint_sha256": teacher.checkpoint_sha256,
            "checkpoint_bytes": teacher.checkpoint_bytes,
            "step": teacher.step,
            "model_config": {
                name: getattr(teacher.config, name)
                for name in teacher.config.__dataclass_fields__
            },
            "rules_hash": LEGACY_RULES_HASH_WIRE,
            "feature_schema_hash": f"{LEGACY_FEATURE_SCHEMA_HASH:016x}",
        },
        "legacy_shards": [
            {
                "shard_id": shard.shard_id,
                "path": str(shard.path),
                "sample_count": shard.sample_count,
                "ring": shard.ring,
                "model_step": shard.model_step,
                "checksum_sha256": shard.checksum_sha256,
            }
            for shard in legacy_shards
        ],
        "committed_shards": committed_shards,
        "committed_samples": committed_samples,
        "committed_games": committed_games,
        "samples_by_ring": samples_by_ring,
        "replay_root": str(root.resolve()),
    }


def transfer_digest(report: dict[str, Any]) -> str:
    """Stable digest of a transfer report's evidence for downstream pins."""

    parts = [
        str(report["teacher"]["identity"]),
        str(report["run_id"]),
        str(report["generation_family"]),
        str(report["committed_samples"]),
        *(str(shard["checksum_sha256"]) for shard in report["legacy_shards"]),
    ]
    return hashlib.sha256("\0".join(parts).encode("utf-8")).hexdigest()


def write_transfer_report(path: str | Path, report: dict[str, Any]) -> Path:
    destination = Path(path)
    payload = dict(report)
    payload["digest"] = transfer_digest(report)
    atomic_json(destination, payload)
    return destination


def new_run_identity(
    root: str | Path, *, run_id: str, generation_family: str
) -> RunIdentity:
    """Create the new run's durable identity; refuse to overwrite one."""

    destination = Path(root) / "run.json"
    if destination.exists():
        raise LineageTransferError(f"run identity already exists: {destination}")
    payload = {
        "schema_version": 1,
        "run_id": validate_identifier("run_id", run_id),
        "generation_family": validate_identifier(
            "generation_family", generation_family
        ),
        "created_ns": time.time_ns(),
    }
    atomic_json(destination, payload)
    return RunIdentity(
        destination,
        payload["run_id"],
        payload["generation_family"],
        payload["created_ns"],
    )
