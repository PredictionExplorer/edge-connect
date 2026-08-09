"""Immutable evidence for selecting archived manifests on ring 10.

The selection plan is content-addressed before any arena work begins.  A final
snapshot embeds that plan, pins every independent result artifact, and can be
re-verified before an isolated fork changes any model pointer.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import tempfile
import time
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

from .arena import ArenaPair, _forced_opening, _opening_seed, summarize_pairs
from .checkpoint import (
    ModelManifest,
    load_model_manifest,
    sha256_file,
    verify_file,
)
from .runtime import load_run_identity
from .topology import get_topology

SELECTION_PLAN_FORMAT = "startrain.archived-manifest-selection-plan"
SELECTION_EVIDENCE_FORMAT = "startrain.archived-manifest-selection"
SELECTION_SCHEMA_VERSION = 1
SELECTION_RING = 10
MULTIPLICITY_METHOD = "bonferroni-fixed-shortlist-v1"
ANYTIME_METHOD = "pair-level-mixture-betting-confidence-sequence-v1"
RANKING_METRIC = "ring-10-anytime-lower-elo"
SEED_SCHEDULE = "sha256-plan-digest-and-candidate-v1"
RESULT_KIND = "archived_manifest_selection"
ARENA_SEED_SCHEDULE = "arena-runner-v2-pair-chunks"


class ManifestSelectionError(ValueError):
    """A frozen plan, result, or selection snapshot failed verification."""


def _canonical_bytes(payload: Mapping[str, object]) -> bytes:
    return (
        json.dumps(payload, sort_keys=True, separators=(",", ":"), allow_nan=False)
        + "\n"
    ).encode("utf-8")


def _read_bytes_once(path: str | Path, name: str) -> tuple[Path, bytes]:
    requested = Path(path).expanduser()
    if requested.is_symlink():
        raise ManifestSelectionError(f"{name} may not be a symbolic link: {requested}")
    source = requested.resolve()
    flags = os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0)
    try:
        descriptor = os.open(source, flags)
    except OSError as exc:
        raise ManifestSelectionError(f"cannot open {name} {source}: {exc}") from exc
    try:
        before = os.fstat(descriptor)
        chunks: list[bytes] = []
        while True:
            chunk = os.read(descriptor, 1024 * 1024)
            if not chunk:
                break
            chunks.append(chunk)
        after = os.fstat(descriptor)
    except OSError as exc:
        raise ManifestSelectionError(f"cannot read {name} {source}: {exc}") from exc
    finally:
        os.close(descriptor)
    if (
        before.st_dev != after.st_dev
        or before.st_ino != after.st_ino
        or before.st_size != after.st_size
        or before.st_mtime_ns != after.st_mtime_ns
    ):
        raise ManifestSelectionError(f"{name} changed while it was being read")
    data = b"".join(chunks)
    if len(data) != before.st_size:
        raise ManifestSelectionError(f"{name} byte length changed while being read")
    return source, data


def _artifact_from_bytes(path: Path, data: bytes) -> "ArtifactEvidence":
    return ArtifactEvidence(
        path=str(path),
        bytes=len(data),
        sha256=hashlib.sha256(data).hexdigest(),
    )


def _read_json_artifact(
    path: str | Path,
    name: str,
    *,
    expected: "ArtifactEvidence | None" = None,
) -> tuple[dict[str, Any], "ArtifactEvidence"]:
    source, data = _read_bytes_once(path, name)
    artifact = _artifact_from_bytes(source, data)
    if expected is not None and (
        artifact.bytes != expected.bytes or artifact.sha256 != expected.sha256
    ):
        raise ManifestSelectionError(f"{name} failed its frozen artifact digest")
    try:
        payload = json.loads(data)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ManifestSelectionError(f"cannot parse {name} {source}: {exc}") from exc
    if not isinstance(payload, dict):
        raise ManifestSelectionError(f"{name} must contain a JSON object")
    return payload, artifact


def _read_json(path: Path, name: str) -> dict[str, Any]:
    return _read_json_artifact(path, name)[0]


def _mapping(value: object, name: str) -> Mapping[str, object]:
    if not isinstance(value, Mapping):
        raise ManifestSelectionError(f"{name} must be an object")
    return value


def _sequence(value: object, name: str) -> Sequence[object]:
    if isinstance(value, str | bytes) or not isinstance(value, Sequence):
        raise ManifestSelectionError(f"{name} must be an array")
    return value


def _text(value: object, name: str) -> str:
    if not isinstance(value, str) or not value:
        raise ManifestSelectionError(f"{name} must be a non-empty string")
    return value


def _sha256_text(value: object, name: str) -> str:
    text = _text(value, name)
    if len(text) != 64 or any(
        character not in "0123456789abcdef" for character in text
    ):
        raise ManifestSelectionError(
            f"{name} must be 64 lowercase hexadecimal characters"
        )
    return text


def _positive_int(value: object, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ManifestSelectionError(f"{name} must be a positive integer")
    return value


def _nonnegative_int(value: object, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ManifestSelectionError(f"{name} must be a non-negative integer")
    return value


def _finite_number(value: object, name: str) -> float:
    if (
        isinstance(value, bool)
        or not isinstance(value, int | float)
        or not math.isfinite(float(value))
    ):
        raise ManifestSelectionError(f"{name} must be finite")
    return float(value)


@dataclass(frozen=True, slots=True)
class ArtifactEvidence:
    """Path, byte length, and digest for one immutable input."""

    path: str
    bytes: int
    sha256: str

    @classmethod
    def from_path(cls, path: str | Path) -> "ArtifactEvidence":
        source = Path(path).expanduser()
        if source.is_symlink():
            raise ManifestSelectionError(
                f"selection artifact may not be a symbolic link: {source}"
            )
        source = source.resolve()
        if not source.is_file():
            raise ManifestSelectionError(f"selection artifact does not exist: {source}")
        return cls(
            path=str(source),
            bytes=source.stat().st_size,
            sha256=sha256_file(source),
        )

    @classmethod
    def from_dict(
        cls, payload: Mapping[str, object], *, name: str
    ) -> "ArtifactEvidence":
        return cls(
            path=_text(payload.get("path"), f"{name} path"),
            bytes=_positive_int(payload.get("bytes"), f"{name} bytes"),
            sha256=_sha256_text(payload.get("sha256"), f"{name} SHA-256"),
        )

    def as_dict(self) -> dict[str, object]:
        return {"path": self.path, "bytes": self.bytes, "sha256": self.sha256}

    def verify(self, path: str | Path | None = None) -> Path:
        source = Path(self.path if path is None else path).expanduser()
        if source.is_symlink():
            raise ManifestSelectionError(
                f"selection artifact may not be a symbolic link: {source}"
            )
        source = source.resolve()
        try:
            verify_file(
                source,
                expected_sha256=self.sha256,
                expected_bytes=self.bytes,
            )
        except ValueError as exc:
            raise ManifestSelectionError(str(exc)) from exc
        return source


@dataclass(frozen=True, slots=True)
class ManifestEvidence:
    """Exact immutable model manifest and checkpoint identity."""

    manifest: ArtifactEvidence
    checkpoint: ArtifactEvidence
    model_identity: str
    model_step: int
    published_ns: int
    run_id: str
    generation_family: str

    @classmethod
    def from_manifest(cls, manifest: ModelManifest) -> "ManifestEvidence":
        artifact = (manifest.artifact_manifest or manifest.path).resolve()
        return cls(
            manifest=ArtifactEvidence.from_path(artifact),
            checkpoint=ArtifactEvidence.from_path(manifest.checkpoint),
            model_identity=manifest.model_identity,
            model_step=manifest.model_step,
            published_ns=manifest.published_ns,
            run_id=manifest.run_id,
            generation_family=manifest.generation_family,
        )

    @classmethod
    def from_dict(
        cls, payload: Mapping[str, object], *, name: str
    ) -> "ManifestEvidence":
        return cls(
            manifest=ArtifactEvidence.from_dict(
                _mapping(payload.get("manifest"), f"{name} manifest"),
                name=f"{name} manifest",
            ),
            checkpoint=ArtifactEvidence.from_dict(
                _mapping(payload.get("checkpoint"), f"{name} checkpoint"),
                name=f"{name} checkpoint",
            ),
            model_identity=_text(
                payload.get("model_identity"), f"{name} model identity"
            ),
            model_step=_nonnegative_int(
                payload.get("model_step"), f"{name} model step"
            ),
            published_ns=_positive_int(
                payload.get("published_ns"), f"{name} published_ns"
            ),
            run_id=_text(payload.get("run_id"), f"{name} run_id"),
            generation_family=_text(
                payload.get("generation_family"), f"{name} generation family"
            ),
        )

    def as_dict(self) -> dict[str, object]:
        return {
            "manifest": self.manifest.as_dict(),
            "checkpoint": self.checkpoint.as_dict(),
            "model_identity": self.model_identity,
            "model_step": self.model_step,
            "published_ns": self.published_ns,
            "run_id": self.run_id,
            "generation_family": self.generation_family,
        }

    def verify(self, manifest_path: str | Path | None = None) -> ModelManifest:
        artifact = self.manifest.verify(manifest_path)
        try:
            manifest = load_model_manifest(artifact)
        except (OSError, RuntimeError, ValueError) as exc:
            raise ManifestSelectionError(
                f"selected model manifest is incompatible: {exc}"
            ) from exc
        actual = ManifestEvidence.from_manifest(manifest)
        expected_identity = (
            self.model_identity,
            self.model_step,
            self.published_ns,
            self.run_id,
            self.generation_family,
            self.manifest.bytes,
            self.manifest.sha256,
            self.checkpoint.bytes,
            self.checkpoint.sha256,
        )
        actual_identity = (
            actual.model_identity,
            actual.model_step,
            actual.published_ns,
            actual.run_id,
            actual.generation_family,
            actual.manifest.bytes,
            actual.manifest.sha256,
            actual.checkpoint.bytes,
            actual.checkpoint.sha256,
        )
        if actual_identity != expected_identity:
            raise ManifestSelectionError(
                "model manifest identity or artifact digests changed"
            )
        self.checkpoint.verify(manifest.checkpoint)
        return manifest


@dataclass(frozen=True, slots=True)
class ChampionEvidence:
    """The source champion pointer and the immutable manifest it resolved to."""

    pointer: ArtifactEvidence
    manifest: ManifestEvidence

    @classmethod
    def from_pointer(cls, pointer_path: str | Path) -> "ChampionEvidence":
        pointer = ArtifactEvidence.from_path(pointer_path)
        try:
            manifest = load_model_manifest(pointer.path)
        except (OSError, RuntimeError, ValueError) as exc:
            raise ManifestSelectionError(
                f"source champion pointer is incompatible: {exc}"
            ) from exc
        return cls(pointer=pointer, manifest=ManifestEvidence.from_manifest(manifest))

    @classmethod
    def from_dict(
        cls, payload: Mapping[str, object], *, name: str
    ) -> "ChampionEvidence":
        return cls(
            pointer=ArtifactEvidence.from_dict(
                _mapping(payload.get("pointer"), f"{name} pointer"),
                name=f"{name} pointer",
            ),
            manifest=ManifestEvidence.from_dict(
                _mapping(payload.get("manifest"), f"{name} manifest"),
                name=f"{name} manifest",
            ),
        )

    def as_dict(self) -> dict[str, object]:
        return {
            "pointer": self.pointer.as_dict(),
            "manifest": self.manifest.as_dict(),
        }

    def verify(self) -> ModelManifest:
        pointer = self.pointer.verify()
        try:
            resolved = load_model_manifest(pointer)
        except (OSError, RuntimeError, ValueError) as exc:
            raise ManifestSelectionError(
                f"source champion pointer is incompatible: {exc}"
            ) from exc
        expected = self.manifest.verify()
        if (
            resolved.model_identity != expected.model_identity
            or resolved.model_step != expected.model_step
            or resolved.manifest_sha256 != expected.manifest_sha256
            or resolved.checkpoint_sha256 != expected.checkpoint_sha256
        ):
            raise ManifestSelectionError(
                "source champion pointer no longer resolves to the frozen baseline"
            )
        return resolved


@dataclass(frozen=True, slots=True)
class SelectionContract:
    """Pre-registered ring-10 budget and familywise statistical contract."""

    initial_pairs: int
    continuation_pairs: int
    minimum_pairs: int
    max_pairs: int
    simulations: int
    max_considered: int
    c_visit: float
    c_scale: float
    pair_chunk_size: int | None
    bootstrap_samples: int
    unforced_opening_fraction: float
    shortlist_size: int
    familywise_alpha: float
    per_candidate_alpha: float
    familywise_beta: float
    per_candidate_beta: float
    confidence: float
    improvement_threshold_elo: float = 0.0
    ring: int = SELECTION_RING
    multiplicity_method: str = MULTIPLICITY_METHOD
    anytime_method: str = ANYTIME_METHOD
    ranking_metric: str = RANKING_METRIC
    seed_schedule: str = SEED_SCHEDULE

    def __post_init__(self) -> None:
        if self.ring != SELECTION_RING:
            raise ManifestSelectionError("archived selection is fixed to ring 10")
        if any(
            isinstance(value, bool) or not isinstance(value, int) or value < 2
            for value in (self.initial_pairs, self.continuation_pairs)
        ):
            raise ManifestSelectionError(
                "selection initial/continuation pair budgets must be at least two"
            )
        if (
            isinstance(self.max_pairs, bool)
            or not isinstance(self.max_pairs, int)
            or isinstance(self.minimum_pairs, bool)
            or not isinstance(self.minimum_pairs, int)
            or self.minimum_pairs < self.initial_pairs
            or self.max_pairs < self.minimum_pairs
            or self.max_pairs < self.continuation_pairs
        ):
            raise ManifestSelectionError("selection maximum pair budget is invalid")
        if (
            isinstance(self.simulations, bool)
            or not isinstance(self.simulations, int)
            or self.simulations <= 0
            or isinstance(self.max_considered, bool)
            or not isinstance(self.max_considered, int)
            or self.max_considered <= 0
            or self.c_visit <= 0
            or self.c_scale <= 0
            or (
                self.pair_chunk_size is not None
                and (
                    isinstance(self.pair_chunk_size, bool)
                    or not isinstance(self.pair_chunk_size, int)
                    or self.pair_chunk_size <= 0
                )
            )
        ):
            raise ManifestSelectionError("selection search contract is invalid")
        if (
            isinstance(self.bootstrap_samples, bool)
            or not isinstance(self.bootstrap_samples, int)
            or self.bootstrap_samples < 200
            or not 0 < self.unforced_opening_fraction < 1
        ):
            raise ManifestSelectionError("selection summary contract is invalid")
        if (
            isinstance(self.shortlist_size, bool)
            or not isinstance(self.shortlist_size, int)
            or self.shortlist_size <= 0
        ):
            raise ManifestSelectionError("selection shortlist must be non-empty")
        if not 0 < self.familywise_alpha < 1 or not 0 < self.familywise_beta < 1:
            raise ManifestSelectionError("selection familywise errors are invalid")
        expected_alpha = self.familywise_alpha / self.shortlist_size
        expected_beta = self.familywise_beta / self.shortlist_size
        if not math.isclose(
            self.per_candidate_alpha, expected_alpha
        ) or not math.isclose(self.per_candidate_beta, expected_beta):
            raise ManifestSelectionError(
                "selection per-candidate errors do not implement Bonferroni allocation"
            )
        if not math.isclose(self.confidence, 1.0 - 2.0 * expected_alpha):
            raise ManifestSelectionError(
                "selection confidence does not match its one-sided alpha allocation"
            )
        if (
            self.improvement_threshold_elo != 0.0
            or self.multiplicity_method != MULTIPLICITY_METHOD
            or self.anytime_method != ANYTIME_METHOD
            or self.ranking_metric != RANKING_METRIC
            or self.seed_schedule != SEED_SCHEDULE
        ):
            raise ManifestSelectionError("selection decision contract is unsupported")

    @classmethod
    def from_dict(cls, payload: Mapping[str, object]) -> "SelectionContract":
        rings = _sequence(payload.get("rings"), "selection rings")
        if list(rings) != [SELECTION_RING]:
            raise ManifestSelectionError("selection rings must be exactly [10]")
        pairs = _mapping(payload.get("pair_budgets"), "selection pair budgets")
        search = _mapping(payload.get("search"), "selection search")
        allocation = _mapping(
            payload.get("alpha_allocation"), "selection alpha allocation"
        )
        beta_allocation = _mapping(
            payload.get("beta_allocation"), "selection beta allocation"
        )
        if (
            search.get("deterministic") is not True
            or search.get("pie_rule") is not False
            or search.get("search_workers") != 2
            or search.get("inference_workers") != 1
        ):
            raise ManifestSelectionError(
                "selection deterministic search settings are unsupported"
            )
        return cls(
            initial_pairs=_positive_int(pairs.get("initial"), "initial pair budget"),
            continuation_pairs=_positive_int(
                pairs.get("continuation"), "continuation pair budget"
            ),
            minimum_pairs=_positive_int(
                pairs.get("minimum", pairs.get("initial")),
                "minimum pair budget",
            ),
            max_pairs=_positive_int(pairs.get("maximum"), "maximum pair budget"),
            simulations=_positive_int(
                search.get("simulations"), "selection simulations"
            ),
            max_considered=_positive_int(
                search.get("max_considered"), "selection max_considered"
            ),
            c_visit=_finite_number(search.get("c_visit"), "selection c_visit"),
            c_scale=_finite_number(search.get("c_scale"), "selection c_scale"),
            pair_chunk_size=(
                None
                if search.get("pair_chunk_size") is None
                else _positive_int(
                    search.get("pair_chunk_size"),
                    "selection pair chunk size",
                )
            ),
            bootstrap_samples=_positive_int(
                payload.get("bootstrap_samples"), "selection bootstrap samples"
            ),
            unforced_opening_fraction=_finite_number(
                payload.get("unforced_opening_fraction"),
                "selection unforced opening fraction",
            ),
            shortlist_size=_positive_int(
                allocation.get("hypotheses"), "selection hypotheses"
            ),
            familywise_alpha=_finite_number(
                allocation.get("familywise_alpha"), "selection familywise alpha"
            ),
            per_candidate_alpha=_finite_number(
                allocation.get("per_candidate_alpha"),
                "selection per-candidate alpha",
            ),
            familywise_beta=_finite_number(
                beta_allocation.get("familywise_beta"), "selection familywise beta"
            ),
            per_candidate_beta=_finite_number(
                beta_allocation.get("per_candidate_beta"),
                "selection per-candidate beta",
            ),
            confidence=_finite_number(
                payload.get("confidence"), "selection confidence"
            ),
            improvement_threshold_elo=_finite_number(
                payload.get("improvement_threshold_elo"),
                "selection improvement threshold",
            ),
            ring=SELECTION_RING,
            multiplicity_method=_text(
                payload.get("multiplicity_method"), "selection multiplicity method"
            ),
            anytime_method=_text(
                payload.get("anytime_method"), "selection anytime method"
            ),
            ranking_metric=_text(
                payload.get("ranking_metric"), "selection ranking metric"
            ),
            seed_schedule=_text(
                payload.get("seed_schedule"), "selection seed schedule"
            ),
        )

    def as_dict(self) -> dict[str, object]:
        return {
            "rings": [self.ring],
            "pair_budgets": {
                "initial": self.initial_pairs,
                "continuation": self.continuation_pairs,
                "minimum": self.minimum_pairs,
                "maximum": self.max_pairs,
            },
            "search": {
                "simulations": self.simulations,
                "max_considered": self.max_considered,
                "c_visit": self.c_visit,
                "c_scale": self.c_scale,
                "pair_chunk_size": self.pair_chunk_size,
                "deterministic": True,
                "pie_rule": False,
                "search_workers": 2,
                "inference_workers": 1,
            },
            "bootstrap_samples": self.bootstrap_samples,
            "unforced_opening_fraction": self.unforced_opening_fraction,
            "confidence": self.confidence,
            "improvement_threshold_elo": self.improvement_threshold_elo,
            "ranking_metric": self.ranking_metric,
            "anytime_method": self.anytime_method,
            "multiplicity_method": self.multiplicity_method,
            "alpha_allocation": {
                "method": self.multiplicity_method,
                "hypotheses": self.shortlist_size,
                "familywise_alpha": self.familywise_alpha,
                "per_candidate_alpha": self.per_candidate_alpha,
            },
            "beta_allocation": {
                "method": self.multiplicity_method,
                "hypotheses": self.shortlist_size,
                "familywise_beta": self.familywise_beta,
                "per_candidate_beta": self.per_candidate_beta,
            },
            "seed_schedule": self.seed_schedule,
        }


@dataclass(frozen=True, slots=True)
class SelectionPlan:
    source_run_root: str
    source_run_id: str
    source_generation_family: str
    source_created_ns: int
    run_identity_artifact: ArtifactEvidence
    source_commit: str | None
    source_commit_artifact: ArtifactEvidence | None
    source_champion: ChampionEvidence
    candidates: tuple[ManifestEvidence, ...]
    evaluation_profile: ArtifactEvidence
    contract: SelectionContract
    shortlist_method: str
    plan_digest: str
    evaluation_seed: int

    def unsigned_dict(self) -> dict[str, object]:
        return {
            "format": SELECTION_PLAN_FORMAT,
            "schema_version": SELECTION_SCHEMA_VERSION,
            "status": "frozen",
            "source_run_root": self.source_run_root,
            "source_run_id": self.source_run_id,
            "source_generation_family": self.source_generation_family,
            "source_created_ns": self.source_created_ns,
            "run_identity_artifact": self.run_identity_artifact.as_dict(),
            "source_commit": self.source_commit,
            "source_commit_artifact": (
                self.source_commit_artifact.as_dict()
                if self.source_commit_artifact is not None
                else None
            ),
            "source_champion": self.source_champion.as_dict(),
            "common_baseline": {
                "kind": "source_champion",
                **self.source_champion.manifest.as_dict(),
            },
            "candidate_manifests": [
                candidate.as_dict() for candidate in self.candidates
            ],
            "evaluation_profile": self.evaluation_profile.as_dict(),
            "evaluation_contract": self.contract.as_dict(),
            "shortlist": {
                "method": self.shortlist_method,
                "count": len(self.candidates),
                "frozen_before_evaluation": True,
            },
        }

    def as_dict(self) -> dict[str, object]:
        return {
            **self.unsigned_dict(),
            "plan_digest": self.plan_digest,
            "evaluation_seed": self.evaluation_seed,
        }


def plan_digest(payload: Mapping[str, object] | SelectionPlan) -> str:
    """Return the deterministic digest over the seedless frozen plan."""

    unsigned = (
        payload.unsigned_dict() if isinstance(payload, SelectionPlan) else dict(payload)
    )
    unsigned.pop("plan_digest", None)
    unsigned.pop("evaluation_seed", None)
    return hashlib.sha256(_canonical_bytes(unsigned)).hexdigest()


def seed_from_plan_digest(digest: str) -> int:
    """Derive a fresh deterministic base seed from a frozen plan digest."""

    validated = _sha256_text(digest, "selection plan digest")
    return int.from_bytes(bytes.fromhex(validated)[:8], "big")


def candidate_seed(plan: SelectionPlan, model_identity: str) -> int:
    material = f"{SEED_SCHEDULE}\0{plan.plan_digest}\0{model_identity}".encode("utf-8")
    return int.from_bytes(hashlib.sha256(material).digest()[:8], "big")


def selection_opening(
    plan: SelectionPlan,
    candidate: ManifestEvidence,
    pair: int,
) -> tuple[int, bool, int | None]:
    """Recompute the exact frozen opening for one candidate pair."""

    pair_index = _nonnegative_int(pair, "selection pair index")
    opening_seed = _opening_seed(
        candidate_seed(plan, candidate.model_identity),
        SELECTION_RING,
        pair_index,
    )
    forced = _forced_opening(
        opening_seed,
        plan.contract.unforced_opening_fraction,
    )
    opening_action = opening_seed % get_topology(SELECTION_RING).n if forced else None
    return opening_seed, forced, opening_action


def next_selection_pair_count(contract: SelectionContract, completed: int) -> int:
    """Return the next pre-registered wave size from a persisted boundary."""

    if isinstance(completed, bool) or not isinstance(completed, int) or completed < 0:
        raise ManifestSelectionError("completed selection pair count is invalid")
    if completed >= contract.max_pairs:
        return 0
    if completed == 0:
        requested = contract.initial_pairs
    elif completed < contract.minimum_pairs:
        requested = min(
            contract.initial_pairs,
            contract.minimum_pairs - completed,
        )
    else:
        requested = contract.continuation_pairs
    return min(requested, contract.max_pairs - completed)


def _is_selection_wave_boundary(contract: SelectionContract, completed: int) -> bool:
    total = 0
    while total < contract.max_pairs:
        total += next_selection_pair_count(contract, total)
        if total == completed:
            return True
        if total > completed:
            return False
    return False


def _source_commit(root: Path) -> tuple[str | None, ArtifactEvidence | None]:
    path = root / "source-commit.txt"
    if not path.exists():
        return None, None
    artifact = ArtifactEvidence.from_path(path)
    try:
        value = path.read_text(encoding="utf-8").strip().split()[0]
    except (OSError, UnicodeDecodeError, IndexError) as exc:
        raise ManifestSelectionError(f"cannot parse source commit: {exc}") from exc
    if len(value) not in (40, 64) or any(
        character not in "0123456789abcdef" for character in value
    ):
        raise ManifestSelectionError("source commit is not a hexadecimal Git object ID")
    return value, artifact


def build_selection_plan(
    *,
    source_run_root: str | Path,
    evaluation_profile: str | Path,
    candidate_manifest_paths: Sequence[str | Path],
    contract: SelectionContract,
    shortlist_method: str = "explicit-manifests-sorted-by-step-and-identity-v1",
) -> SelectionPlan:
    """Verify and deterministically freeze a shortlist before arena evaluation."""

    root = Path(source_run_root).expanduser().resolve()
    if not root.is_dir():
        raise ManifestSelectionError(f"source run root does not exist: {root}")
    identity = load_run_identity(root / "run.json")
    run_artifact = ArtifactEvidence.from_path(identity.path)
    champion = ChampionEvidence.from_pointer(root / "learner" / "champion.json")
    champion_manifest = champion.manifest
    if (
        champion_manifest.run_id != identity.run_id
        or champion_manifest.generation_family != identity.generation_family
    ):
        raise ManifestSelectionError("source champion belongs to another run")

    manifests: dict[str, ManifestEvidence] = {}
    manifest_root = (root / "learner" / "manifests").resolve()
    for raw_path in candidate_manifest_paths:
        try:
            loaded = load_model_manifest(Path(raw_path).expanduser().resolve())
        except (OSError, RuntimeError, ValueError) as exc:
            raise ManifestSelectionError(
                f"candidate manifest is incompatible: {raw_path}: {exc}"
            ) from exc
        evidence = ManifestEvidence.from_manifest(loaded)
        artifact = Path(evidence.manifest.path)
        if artifact.parent != manifest_root:
            raise ManifestSelectionError(
                "candidate manifest is not in the source run manifest archive"
            )
        if (
            evidence.run_id != identity.run_id
            or evidence.generation_family != identity.generation_family
        ):
            raise ManifestSelectionError(
                "candidate manifest belongs to another run or generation"
            )
        if evidence.model_identity == champion_manifest.model_identity:
            continue
        previous = manifests.get(evidence.model_identity)
        if previous is not None and previous != evidence:
            raise ManifestSelectionError(
                "shortlist repeats a model identity with different evidence"
            )
        manifests[evidence.model_identity] = evidence
    candidates = tuple(
        sorted(
            manifests.values(),
            key=lambda item: (
                item.model_step,
                item.model_identity,
                item.manifest.path,
            ),
        )
    )
    if not candidates:
        raise ManifestSelectionError("archived manifest shortlist is empty")
    if contract.shortlist_size != len(candidates):
        raise ManifestSelectionError(
            "selection contract hypothesis count does not match the shortlist"
        )
    profile = ArtifactEvidence.from_path(evaluation_profile)
    commit, commit_artifact = _source_commit(root)
    provisional = SelectionPlan(
        source_run_root=str(root),
        source_run_id=identity.run_id,
        source_generation_family=identity.generation_family,
        source_created_ns=identity.created_ns,
        run_identity_artifact=run_artifact,
        source_commit=commit,
        source_commit_artifact=commit_artifact,
        source_champion=champion,
        candidates=candidates,
        evaluation_profile=profile,
        contract=contract,
        shortlist_method=_text(shortlist_method, "shortlist method"),
        plan_digest="0" * 64,
        evaluation_seed=0,
    )
    digest = plan_digest(provisional)
    return replace(
        provisional,
        plan_digest=digest,
        evaluation_seed=seed_from_plan_digest(digest),
    )


def _parse_plan(payload: Mapping[str, object]) -> SelectionPlan:
    if (
        payload.get("format") != SELECTION_PLAN_FORMAT
        or payload.get("schema_version") != SELECTION_SCHEMA_VERSION
        or payload.get("status") != "frozen"
    ):
        raise ManifestSelectionError("unsupported archived-manifest selection plan")
    source_champion = ChampionEvidence.from_dict(
        _mapping(payload.get("source_champion"), "source champion"),
        name="source champion",
    )
    common = _mapping(payload.get("common_baseline"), "common baseline")
    if common.get("kind") != "source_champion":
        raise ManifestSelectionError("selection common baseline is not the champion")
    common_manifest = ManifestEvidence.from_dict(common, name="common baseline")
    if common_manifest != source_champion.manifest:
        raise ManifestSelectionError(
            "selection common baseline differs from the source champion"
        )
    candidates = tuple(
        ManifestEvidence.from_dict(
            _mapping(item, f"candidate manifest {index}"),
            name=f"candidate manifest {index}",
        )
        for index, item in enumerate(
            _sequence(payload.get("candidate_manifests"), "candidate manifests")
        )
    )
    shortlist = _mapping(payload.get("shortlist"), "selection shortlist")
    if shortlist.get("frozen_before_evaluation") is not True or _positive_int(
        shortlist.get("count"), "shortlist count"
    ) != len(candidates):
        raise ManifestSelectionError("selection shortlist was not frozen correctly")
    commit_value = payload.get("source_commit")
    if commit_value is not None:
        commit_value = _text(commit_value, "source commit")
    commit_artifact_value = payload.get("source_commit_artifact")
    commit_artifact = (
        ArtifactEvidence.from_dict(
            _mapping(commit_artifact_value, "source commit artifact"),
            name="source commit artifact",
        )
        if commit_artifact_value is not None
        else None
    )
    if (commit_value is None) != (commit_artifact is None):
        raise ManifestSelectionError(
            "source commit value and artifact are inconsistent"
        )
    return SelectionPlan(
        source_run_root=_text(payload.get("source_run_root"), "source run root"),
        source_run_id=_text(payload.get("source_run_id"), "source run_id"),
        source_generation_family=_text(
            payload.get("source_generation_family"), "source generation family"
        ),
        source_created_ns=_positive_int(
            payload.get("source_created_ns"), "source created_ns"
        ),
        run_identity_artifact=ArtifactEvidence.from_dict(
            _mapping(payload.get("run_identity_artifact"), "run identity artifact"),
            name="run identity artifact",
        ),
        source_commit=commit_value,
        source_commit_artifact=commit_artifact,
        source_champion=source_champion,
        candidates=candidates,
        evaluation_profile=ArtifactEvidence.from_dict(
            _mapping(payload.get("evaluation_profile"), "evaluation profile"),
            name="evaluation profile",
        ),
        contract=SelectionContract.from_dict(
            _mapping(payload.get("evaluation_contract"), "evaluation contract")
        ),
        shortlist_method=_text(shortlist.get("method"), "shortlist method"),
        plan_digest=_sha256_text(payload.get("plan_digest"), "selection plan digest"),
        evaluation_seed=_nonnegative_int(
            payload.get("evaluation_seed"), "selection evaluation seed"
        ),
    )


def verify_selection_plan(
    plan: str | Path | Mapping[str, object] | SelectionPlan,
    *,
    expected_source_run_root: str | Path | None = None,
) -> SelectionPlan:
    """Re-verify a frozen plan and all source artifacts without writing."""

    if isinstance(plan, SelectionPlan):
        parsed = plan
    elif isinstance(plan, Mapping):
        parsed = _parse_plan(plan)
    else:
        payload, _artifact = _read_json_artifact(plan, "selection plan")
        parsed = _parse_plan(payload)
    expected_digest = plan_digest(parsed)
    if parsed.plan_digest != expected_digest:
        raise ManifestSelectionError("selection plan digest mismatch")
    if parsed.evaluation_seed != seed_from_plan_digest(expected_digest):
        raise ManifestSelectionError("selection evaluation seed is not plan-derived")
    root = Path(parsed.source_run_root).expanduser().resolve()
    if (
        expected_source_run_root is not None
        and root != Path(expected_source_run_root).expanduser().resolve()
    ):
        raise ManifestSelectionError("selection source run root mismatch")
    identity_path = parsed.run_identity_artifact.verify()
    identity = load_run_identity(identity_path)
    if (
        identity.path.resolve() != (root / "run.json").resolve()
        or identity.run_id != parsed.source_run_id
        or identity.generation_family != parsed.source_generation_family
        or identity.created_ns != parsed.source_created_ns
    ):
        raise ManifestSelectionError("selection source run identity changed")
    champion = parsed.source_champion.verify()
    if (
        champion.run_id != identity.run_id
        or champion.generation_family != identity.generation_family
    ):
        raise ManifestSelectionError("selection champion run identity changed")
    seen: set[str] = set()
    previous_key: tuple[int, str, str] | None = None
    manifest_root = (root / "learner" / "manifests").resolve()
    for candidate in parsed.candidates:
        manifest = candidate.verify()
        key = (candidate.model_step, candidate.model_identity, candidate.manifest.path)
        if previous_key is not None and key <= previous_key:
            raise ManifestSelectionError(
                "selection shortlist is not in deterministic manifest order"
            )
        previous_key = key
        if (
            Path(candidate.manifest.path).resolve().parent != manifest_root
            or manifest.run_id != identity.run_id
            or manifest.generation_family != identity.generation_family
            or manifest.model_identity == champion.model_identity
            or manifest.model_identity in seen
        ):
            raise ManifestSelectionError(
                "selection shortlist run identity or membership is invalid"
            )
        seen.add(manifest.model_identity)
    if len(parsed.candidates) != parsed.contract.shortlist_size:
        raise ManifestSelectionError(
            "selection shortlist no longer matches multiplicity allocation"
        )
    parsed.evaluation_profile.verify()
    if parsed.source_commit_artifact is not None:
        commit_path = parsed.source_commit_artifact.verify()
        try:
            current_commit = commit_path.read_text(encoding="utf-8").strip().split()[0]
        except (OSError, UnicodeDecodeError, IndexError) as exc:
            raise ManifestSelectionError(
                f"cannot re-read source commit: {exc}"
            ) from exc
        if current_commit != parsed.source_commit:
            raise ManifestSelectionError("selection source commit changed")
    return parsed


def write_frozen_json(
    path: str | Path, payload: Mapping[str, object], *, mode: int = 0o444
) -> ArtifactEvidence:
    """Atomically create one read-only JSON evidence artifact."""

    destination = Path(path).expanduser().resolve()
    if destination.exists() or destination.is_symlink():
        raise FileExistsError(f"frozen evidence already exists: {destination}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    data = _canonical_bytes(payload)
    temporary: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="wb",
            prefix=f".{destination.name}.",
            dir=destination.parent,
            delete=False,
        ) as handle:
            temporary = Path(handle.name)
            handle.write(data)
            handle.flush()
            os.fsync(handle.fileno())
        os.chmod(temporary, mode)
        try:
            os.link(temporary, destination)
        except FileExistsError as exc:
            raise FileExistsError(
                f"frozen evidence already exists: {destination}"
            ) from exc
        directory_fd = os.open(destination.parent, os.O_RDONLY)
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)
    payload_read, artifact = _read_json_artifact(destination, "frozen evidence")
    if payload_read != dict(payload):
        raise ManifestSelectionError("frozen evidence bytes changed after publication")
    return artifact


def freeze_selection_plan(path: str | Path, plan: SelectionPlan) -> ArtifactEvidence:
    verified = verify_selection_plan(plan)
    return write_frozen_json(path, verified.as_dict())


@dataclass(frozen=True, slots=True)
class ResultEvidence:
    candidate: ManifestEvidence
    artifact: ArtifactEvidence
    completed_pairs: int
    anytime_lower_elo: float
    anytime_upper_elo: float
    score_rate: float
    statistically_proven_improvement: bool
    rank: int = 0

    def as_dict(self) -> dict[str, object]:
        return {
            "candidate": self.candidate.as_dict(),
            "result_artifact": self.artifact.as_dict(),
            "completed_pairs": self.completed_pairs,
            "ring_10_anytime_lower_elo": self.anytime_lower_elo,
            "ring_10_anytime_upper_elo": self.anytime_upper_elo,
            "ring_10_score_rate": self.score_rate,
            "statistically_proven_improvement": (self.statistically_proven_improvement),
            "status": (
                "proven_improvement"
                if self.statistically_proven_improvement
                else "not_proven"
            ),
            "rank": self.rank,
        }


@dataclass(frozen=True, slots=True)
class SelectionEvidence:
    plan: SelectionPlan
    plan_artifact: ArtifactEvidence
    results: tuple[ResultEvidence, ...]
    ranking: tuple[str, ...]
    selected: ManifestEvidence
    fallback_used: bool
    fallback_reason: str | None
    generated_ns: int

    def as_dict(self) -> dict[str, object]:
        return {
            "format": SELECTION_EVIDENCE_FORMAT,
            "schema_version": SELECTION_SCHEMA_VERSION,
            "status": "verified",
            "plan": self.plan.as_dict(),
            "plan_artifact": self.plan_artifact.as_dict(),
            "result_artifacts": [result.as_dict() for result in self.results],
            "ranking": {
                "metric": RANKING_METRIC,
                "model_identities": list(self.ranking),
            },
            "selected_identity": self.selected.model_identity,
            "selected_manifest": self.selected.as_dict(),
            "fallback_to_champion": {
                "used": self.fallback_used,
                "reason": self.fallback_reason,
            },
            "generated_ns": self.generated_ns,
        }


@dataclass(frozen=True, slots=True)
class PersistedSelectionResult:
    payload: Mapping[str, object]
    artifact: ArtifactEvidence
    pairs: tuple[ArenaPair, ...]
    terminal: bool
    statistically_proven_improvement: bool
    anytime_lower_elo: float
    anytime_upper_elo: float
    score_rate: float


def _verify_search_metadata(
    plan: SelectionPlan,
    payload: Mapping[str, object],
) -> None:
    contract = plan.contract
    budget = {
        "simulations": contract.simulations,
        "max_considered": contract.max_considered,
        "c_visit": contract.c_visit,
        "c_scale": contract.c_scale,
    }
    expected_search = {
        "deterministic": True,
        **budget,
        "pie_rule": False,
        "search_workers": 2,
        "inference_workers": 1,
        "pair_chunk_size": contract.pair_chunk_size,
        "effective_pair_chunking": (
            "configured"
            if contract.pair_chunk_size is not None
            else "full_requested_ring_batch"
        ),
    }
    search = _mapping(payload.get("search"), "selection candidate search metadata")
    if dict(search) != expected_search:
        raise ManifestSelectionError(
            "selection candidate search metadata or budget changed"
        )
    baseline = _mapping(
        payload.get("baseline_metadata"),
        "selection baseline search metadata",
    )
    expected_baseline = {
        "kind": "checkpoint",
        "identity": plan.source_champion.manifest.model_identity,
        "search_budget": budget,
        "deterministic": True,
        "seed_schedule": ARENA_SEED_SCHEDULE,
    }
    if dict(baseline) != expected_baseline:
        raise ManifestSelectionError(
            "selection baseline search metadata or budget changed"
        )


def load_persisted_selection_result(
    plan: SelectionPlan,
    candidate: ManifestEvidence,
    result_path: str | Path,
) -> PersistedSelectionResult:
    """Verify a terminal result or resumable partial wave from coherent bytes."""

    path = Path(result_path).expanduser().resolve()
    payload, artifact = _read_json_artifact(
        path,
        "archived-manifest arena result",
    )
    if (
        payload.get("schema_version") != 3
        or payload.get("result_kind") != RESULT_KIND
        or payload.get("selection_plan_digest") != plan.plan_digest
        or payload.get("candidate") != candidate.model_identity
        or payload.get("baseline") != plan.source_champion.manifest.model_identity
        or payload.get("arena_seed_block")
        != candidate_seed(plan, candidate.model_identity)
        or payload.get("selection_contract") != plan.contract.as_dict()
    ):
        raise ManifestSelectionError(
            f"selection result identity or frozen contract mismatch: {path}"
        )
    if payload.get("candidate_manifest") != candidate.manifest.path:
        raise ManifestSelectionError("selection result candidate manifest changed")
    if payload.get("baseline_manifest") != plan.source_champion.manifest.manifest.path:
        raise ManifestSelectionError("selection result common baseline changed")
    _verify_search_metadata(plan, payload)
    pairs = _sequence(payload.get("pairs"), "selection result pairs")
    verified_pairs: list[ArenaPair] = []
    for item in pairs:
        pair = _mapping(item, "selection result pair")
        expected_pair_fields = {
            "ring",
            "pair",
            "opening_seed",
            "opening_action",
            "forced_opening",
            "outcomes",
        }
        if set(pair) != expected_pair_fields:
            raise ManifestSelectionError("selection result pair fields are invalid")
        outcomes = _sequence(pair.get("outcomes"), "selection pair outcomes")
        if len(outcomes) != 2:
            raise ManifestSelectionError("selection pair outcomes are invalid")
        first_outcome = outcomes[0]
        second_outcome = outcomes[1]
        if type(first_outcome) is not int or type(second_outcome) is not int:
            raise ManifestSelectionError("selection pair outcomes are invalid")
        opening_action_value = pair.get("opening_action")
        opening_action = (
            None
            if opening_action_value is None
            else _nonnegative_int(
                opening_action_value,
                "selection pair opening action",
            )
        )
        forced_opening = pair.get("forced_opening")
        if type(forced_opening) is not bool:
            raise ManifestSelectionError(
                "selection pair forced-opening flag is invalid"
            )
        try:
            verified_pair = ArenaPair(
                ring=_nonnegative_int(pair.get("ring"), "selection pair ring"),
                pair=_nonnegative_int(pair.get("pair"), "selection pair index"),
                opening_seed=_nonnegative_int(
                    pair.get("opening_seed"), "selection pair opening seed"
                ),
                opening_action=opening_action,
                forced_opening=forced_opening,
                outcomes=(first_outcome, second_outcome),
            )
        except (TypeError, ValueError) as exc:
            raise ManifestSelectionError(
                f"selection result pair is invalid: {exc}"
            ) from exc
        if verified_pair.ring != SELECTION_RING:
            raise ManifestSelectionError(
                "selection result contains a non-ring-10 observation"
            )
        expected_opening = selection_opening(
            plan,
            candidate,
            verified_pair.pair,
        )
        actual_opening = (
            verified_pair.opening_seed,
            verified_pair.forced_opening,
            verified_pair.opening_action,
        )
        if actual_opening != expected_opening:
            raise ManifestSelectionError(
                "selection result pair opening is not candidate-seed-derived"
            )
        verified_pairs.append(verified_pair)
    pair_indices = [pair.pair for pair in verified_pairs]
    if sorted(pair_indices) != list(range(len(pair_indices))):
        raise ManifestSelectionError(
            "selection pair indices must be unique and contiguous from zero"
        )
    completed = len(pairs)
    if (
        completed <= 0
        or completed > plan.contract.max_pairs
        or not _is_selection_wave_boundary(plan.contract, completed)
    ):
        raise ManifestSelectionError(
            "selection result pair count is not a frozen wave boundary"
        )
    per_ring = _mapping(payload.get("per_ring"), "selection per-ring summary")
    if set(per_ring) != {str(SELECTION_RING)}:
        raise ManifestSelectionError("selection result is not ring-10-only")
    summary = _mapping(per_ring[str(SELECTION_RING)], "ring-10 selection summary")
    recomputed_summary = summarize_pairs(
        verified_pairs,
        confidence=plan.contract.confidence,
        bootstrap_samples=plan.contract.bootstrap_samples,
        seed=candidate_seed(plan, candidate.model_identity)
        + SELECTION_RING * 1_000_003,
    )
    if any(summary.get(key) != value for key, value in recomputed_summary.items()):
        raise ManifestSelectionError(
            "ring-10 selection summary does not match its complete pairs"
        )
    interval = _sequence(
        summary.get("anytime_elo_interval"), "ring-10 anytime Elo interval"
    )
    if len(interval) != 2:
        raise ManifestSelectionError("ring-10 anytime Elo interval is malformed")
    lower = _finite_number(interval[0], "ring-10 anytime lower Elo")
    upper = _finite_number(interval[1], "ring-10 anytime upper Elo")
    score_rate = _finite_number(summary.get("score_rate"), "ring-10 score rate")
    if not 0 <= score_rate <= 1 or lower > upper:
        raise ManifestSelectionError("ring-10 selection summary is invalid")
    error_probability = _finite_number(
        summary.get("anytime_error_probability_per_side"),
        "ring-10 anytime error probability",
    )
    if not math.isclose(error_probability, plan.contract.per_candidate_alpha):
        raise ManifestSelectionError(
            "selection result does not apply the frozen familywise alpha allocation"
        )
    proven = lower > plan.contract.improvement_threshold_elo
    if payload.get("statistically_proven_improvement") is not proven:
        raise ManifestSelectionError("selection result improvement decision changed")
    expected_decision = "proven_improvement" if proven else "not_proven"
    if payload.get("selection_decision") != expected_decision:
        raise ManifestSelectionError("selection result decision label changed")
    terminal = (proven and completed >= plan.contract.minimum_pairs) or (
        completed == plan.contract.max_pairs
    )
    if payload.get("terminal") is not terminal:
        raise ManifestSelectionError(
            "selection result terminal state violates the frozen stopping rule"
        )
    evaluation_started_ns = payload.get("evaluation_started_ns")
    if (
        isinstance(evaluation_started_ns, bool)
        or not isinstance(evaluation_started_ns, int)
        or evaluation_started_ns <= 0
    ):
        raise ManifestSelectionError("selection evaluation start timestamp is invalid")
    games = _sequence(payload.get("games"), "selection result games")
    if len(games) != completed * 2:
        raise ManifestSelectionError("selection result game count is incomplete")
    history = _sequence(payload.get("wave_history"), "selection wave history")
    if not history:
        raise ManifestSelectionError("selection wave history is empty")
    history_total = 0
    for wave_index, raw_wave in enumerate(history):
        wave = _mapping(raw_wave, f"selection wave history {wave_index}")
        pair_count = _positive_int(
            wave.get("pair_count"),
            f"selection wave {wave_index} pair count",
        )
        if (
            set(wave)
            != {
                "wave_index",
                "pair_start",
                "pair_count",
                "completed_pairs",
                "evaluation_metrics",
            }
            or wave.get("wave_index") != wave_index
            or wave.get("pair_start") != history_total
            or pair_count != next_selection_pair_count(plan.contract, history_total)
        ):
            raise ManifestSelectionError("selection wave history changed")
        history_total += pair_count
        if wave.get("completed_pairs") != history_total or not isinstance(
            wave.get("evaluation_metrics"),
            Mapping,
        ):
            raise ManifestSelectionError("selection wave history is inconsistent")
    if history_total != completed:
        raise ManifestSelectionError("selection wave history is incomplete")
    return PersistedSelectionResult(
        payload=payload,
        artifact=artifact,
        pairs=tuple(verified_pairs),
        terminal=terminal,
        statistically_proven_improvement=proven,
        anytime_lower_elo=lower,
        anytime_upper_elo=upper,
        score_rate=score_rate,
    )


def _result_evidence(
    plan: SelectionPlan,
    candidate: ManifestEvidence,
    result_path: Path,
) -> ResultEvidence:
    persisted = load_persisted_selection_result(plan, candidate, result_path)
    if not persisted.terminal:
        raise ManifestSelectionError(
            "selection result is a valid partial wave, not a terminal result"
        )
    return ResultEvidence(
        candidate=candidate,
        artifact=persisted.artifact,
        completed_pairs=len(persisted.pairs),
        anytime_lower_elo=persisted.anytime_lower_elo,
        anytime_upper_elo=persisted.anytime_upper_elo,
        score_rate=persisted.score_rate,
        statistically_proven_improvement=(persisted.statistically_proven_improvement),
    )


def build_selection_evidence(
    *,
    plan_path: str | Path,
    result_paths: Mapping[str, str | Path],
    generated_ns: int | None = None,
) -> SelectionEvidence:
    """Rank verified independent results and conservatively choose an anchor."""

    plan_file = Path(plan_path).expanduser().resolve()
    plan_payload, plan_artifact = _read_json_artifact(plan_file, "selection plan")
    plan = verify_selection_plan(plan_payload)
    if set(result_paths) != {candidate.model_identity for candidate in plan.candidates}:
        raise ManifestSelectionError(
            "selection results do not cover the entire frozen shortlist"
        )
    raw_results = tuple(
        _result_evidence(
            plan,
            candidate,
            Path(result_paths[candidate.model_identity]).expanduser().resolve(),
        )
        for candidate in plan.candidates
    )
    ranked = sorted(
        raw_results,
        key=lambda result: (
            -result.anytime_lower_elo,
            result.candidate.model_identity,
        ),
    )
    ranked = [
        replace(result, rank=index) for index, result in enumerate(ranked, start=1)
    ]
    proven = [result for result in ranked if result.statistically_proven_improvement]
    if proven:
        selected = proven[0].candidate
        fallback_used = False
        fallback_reason = None
    else:
        selected = plan.source_champion.manifest
        fallback_used = True
        fallback_reason = (
            "no_shortlisted_candidate_proved_positive_ring10_elo_under_"
            "familywise_error_control"
        )
    timestamp = time.time_ns() if generated_ns is None else generated_ns
    if isinstance(timestamp, bool) or not isinstance(timestamp, int) or timestamp <= 0:
        raise ManifestSelectionError("selection evidence timestamp is invalid")
    return SelectionEvidence(
        plan=plan,
        plan_artifact=plan_artifact,
        results=tuple(ranked),
        ranking=tuple(result.candidate.model_identity for result in ranked),
        selected=selected,
        fallback_used=fallback_used,
        fallback_reason=fallback_reason,
        generated_ns=timestamp,
    )


def freeze_selection_evidence(
    path: str | Path, evidence: SelectionEvidence
) -> ArtifactEvidence:
    return write_frozen_json(path, evidence.as_dict())


@dataclass(frozen=True, slots=True)
class VerifiedSelection:
    snapshot_path: Path
    snapshot_artifact: ArtifactEvidence
    plan: SelectionPlan
    selected_evidence: ManifestEvidence
    selected_manifest: ModelManifest
    source_champion_manifest: ModelManifest
    fallback_used: bool
    fallback_reason: str | None


def verify_selection_snapshot(
    snapshot_path: str | Path,
    *,
    expected_source_run_root: str | Path | None = None,
) -> VerifiedSelection:
    """Verify a final snapshot, every result digest, and the selected manifest."""

    path = Path(snapshot_path).expanduser().resolve()
    payload, snapshot_artifact = _read_json_artifact(
        path,
        "manifest selection snapshot",
    )
    if (
        payload.get("format") != SELECTION_EVIDENCE_FORMAT
        or payload.get("schema_version") != SELECTION_SCHEMA_VERSION
        or payload.get("status") != "verified"
    ):
        raise ManifestSelectionError("unsupported manifest selection snapshot")
    embedded_plan = _mapping(payload.get("plan"), "embedded selection plan")
    plan = verify_selection_plan(
        embedded_plan,
        expected_source_run_root=expected_source_run_root,
    )
    plan_artifact = ArtifactEvidence.from_dict(
        _mapping(payload.get("plan_artifact"), "selection plan artifact"),
        name="selection plan artifact",
    )
    plan_payload, actual_plan_artifact = _read_json_artifact(
        plan_artifact.path,
        "selection plan artifact",
        expected=plan_artifact,
    )
    plan_path = Path(actual_plan_artifact.path)
    if plan_payload != dict(embedded_plan):
        raise ManifestSelectionError(
            "embedded selection plan differs from its pinned artifact"
        )
    raw_results = _sequence(payload.get("result_artifacts"), "selection results")
    result_paths: dict[str, str] = {}
    for index, raw_result in enumerate(raw_results):
        result = _mapping(raw_result, f"selection result evidence {index}")
        candidate = ManifestEvidence.from_dict(
            _mapping(result.get("candidate"), f"selection result candidate {index}"),
            name=f"selection result candidate {index}",
        )
        artifact = ArtifactEvidence.from_dict(
            _mapping(
                result.get("result_artifact"),
                f"selection result artifact {index}",
            ),
            name=f"selection result artifact {index}",
        )
        if candidate.model_identity in result_paths:
            raise ManifestSelectionError("selection snapshot repeats a result identity")
        result_paths[candidate.model_identity] = artifact.path
    generated_ns = _positive_int(payload.get("generated_ns"), "selection generated_ns")
    recomputed = build_selection_evidence(
        plan_path=plan_path,
        result_paths=result_paths,
        generated_ns=generated_ns,
    )
    expected_payload = recomputed.as_dict()
    for key in (
        "plan",
        "plan_artifact",
        "result_artifacts",
        "ranking",
        "selected_identity",
        "selected_manifest",
        "fallback_to_champion",
        "generated_ns",
    ):
        if payload.get(key) != expected_payload[key]:
            raise ManifestSelectionError(
                f"selection snapshot {key} failed deterministic verification"
            )
    selected = recomputed.selected.verify()
    champion = recomputed.plan.source_champion.verify()
    return VerifiedSelection(
        snapshot_path=path,
        snapshot_artifact=snapshot_artifact,
        plan=plan,
        selected_evidence=recomputed.selected,
        selected_manifest=selected,
        source_champion_manifest=champion,
        fallback_used=recomputed.fallback_used,
        fallback_reason=recomputed.fallback_reason,
    )


def selected_manifest_in_copy(
    selection: VerifiedSelection,
    copied_run_root: str | Path,
) -> ModelManifest:
    """Resolve and verify the selected archive manifest inside a copied run."""

    source_root = Path(selection.plan.source_run_root).resolve()
    selected_path = Path(selection.selected_evidence.manifest.path).resolve()
    try:
        relative = selected_path.relative_to(source_root)
    except ValueError as exc:
        raise ManifestSelectionError(
            "selected manifest is outside the frozen source run"
        ) from exc
    copied_path = Path(copied_run_root).expanduser().resolve() / relative
    manifest = selection.selected_evidence.verify(copied_path)
    if (
        manifest.run_id != selection.plan.source_run_id
        or manifest.generation_family != selection.plan.source_generation_family
    ):
        raise ManifestSelectionError(
            "copied selected manifest belongs to another run or generation"
        )
    return manifest
