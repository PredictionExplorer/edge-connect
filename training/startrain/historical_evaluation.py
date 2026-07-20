"""Plan bounded non-adjacent checkpoint evaluations between promotions."""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path

from .checkpoint import ModelManifest, load_model_manifest
from .config import HistoricalEvaluationConfig
from .runtime import RunIdentity

HISTORICAL_CROSSPLAY_RESULT_KIND = "historical_crossplay"
PROMOTION_RESULT_KIND = "promotion"


@dataclass(frozen=True, slots=True)
class HistoricalEvaluationPlan:
    candidate: ModelManifest
    baseline: ModelManifest
    result_path: Path
    previous: dict[str, object] | None


def arena_result_kind(result: Mapping[str, object]) -> str:
    """Classify legacy arena files as promotion results."""

    value = result.get("result_kind")
    return value if isinstance(value, str) else PROMOTION_RESULT_KIND


def load_historical_manifests(
    directory: str | Path,
    *,
    run_identity: RunIdentity,
) -> dict[str, ModelManifest]:
    manifests: dict[str, ModelManifest] = {}
    for path in Path(directory).glob("manifest-*.json"):
        manifest = load_model_manifest(path)
        if (
            manifest.run_id == run_identity.run_id
            and manifest.generation_family == run_identity.generation_family
        ):
            manifests[manifest.model_identity] = manifest
    return manifests


def load_arena_results(directory: str | Path) -> list[tuple[Path, dict[str, object]]]:
    results = []
    for path in Path(directory).glob("*.json"):
        if path.name == "promotion-status.json":
            continue
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        if isinstance(payload, dict):
            results.append((path, payload))
    return results


def select_historical_evaluation(
    *,
    config: HistoricalEvaluationConfig,
    champion: ModelManifest,
    manifests: Mapping[str, ModelManifest],
    arena_results: Sequence[tuple[Path, Mapping[str, object]]],
    results_directory: str | Path,
) -> HistoricalEvaluationPlan | None:
    """Choose one resumable, non-adjacent evaluation without mutating pointers."""

    if not config.enabled:
        return None
    promoted: list[tuple[int, int, str, str]] = []
    direct_predecessor: str | None = None
    existing_crossplay: dict[tuple[str, str], tuple[Path, dict[str, object]]] = {}
    for path, result in arena_results:
        candidate = result.get("candidate")
        baseline = result.get("baseline")
        if not isinstance(candidate, str) or not isinstance(baseline, str):
            continue
        kind = arena_result_kind(result)
        if kind == HISTORICAL_CROSSPLAY_RESULT_KIND:
            existing_crossplay[(candidate, baseline)] = (path, dict(result))
            continue
        promotion = result.get("promotion")
        decision = promotion.get("decision") if isinstance(promotion, Mapping) else None
        completed_ns = result.get("completed_ns")
        if (
            decision == "promote"
            and isinstance(completed_ns, int)
            and candidate in manifests
            and baseline in manifests
        ):
            promoted.append(
                (
                    completed_ns,
                    manifests[candidate].model_step,
                    candidate,
                    baseline,
                )
            )
            if candidate == champion.model_identity:
                direct_predecessor = baseline
    promoted.sort()
    promoted_identities = {baseline for _, _, _, baseline in promoted}
    promoted_identities.update(candidate for _, _, candidate, _ in promoted)
    promotion_count = len({candidate for _, _, candidate, _ in promoted})
    if not promotion_count or promotion_count % config.every_promotions:
        return None

    eligible = [
        manifest
        for identity, manifest in manifests.items()
        if identity in promoted_identities
        and identity not in (champion.model_identity, direct_predecessor)
        and manifest.model_step < champion.model_step
    ]
    eligible.sort(key=lambda item: (item.model_step, item.model_identity), reverse=True)
    for baseline in eligible[: config.anchors_per_evaluation]:
        key = (champion.model_identity, baseline.model_identity)
        prior = existing_crossplay.get(key)
        if prior is not None:
            path, payload = prior
            if bool(payload.get("terminal")):
                continue
            return HistoricalEvaluationPlan(
                candidate=champion,
                baseline=baseline,
                result_path=path,
                previous=payload,
            )
        name = f"crossplay-{champion.model_identity}-vs-{baseline.model_identity}.json"
        return HistoricalEvaluationPlan(
            candidate=champion,
            baseline=baseline,
            result_path=Path(results_directory) / name,
            previous=None,
        )
    return None
