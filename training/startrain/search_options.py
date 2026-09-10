"""Explicit, opt-in search experiments shared by self-play and evaluation."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import json
import math
from typing import Any, Literal, Mapping, Sequence

from .contracts import FEATURE_SCHEMA_VERSION


@dataclass(frozen=True, slots=True)
class FullSearchBudgetConfig:
    mode: Literal["fixed", "root-entropy"] = "fixed"
    minimum_fraction: float = 0.5
    entropy_threshold: float = 0.35

    def __post_init__(self) -> None:
        if self.mode not in ("fixed", "root-entropy"):
            raise ValueError("full budget mode must be fixed or root-entropy")
        for name in ("minimum_fraction", "entropy_threshold"):
            value = getattr(self, name)
            if (
                isinstance(value, bool)
                or not isinstance(value, int | float)
                or not math.isfinite(value)
            ):
                raise ValueError(f"{name} must be finite")
        if not 0 < self.minimum_fraction <= 1:
            raise ValueError("minimum_fraction must be in (0, 1]")
        if not 0 <= self.entropy_threshold <= 1:
            raise ValueError("entropy_threshold must be in [0, 1]")

    def adjusted_cap(
        self,
        base_cap: int,
        entropy: float,
        *,
        minimum_cap: int = 1,
        quantum: int = 1,
    ) -> int:
        """Reduce confident roots, retaining a caller's paired-budget floor."""
        if type(base_cap) is not int or base_cap <= 0:
            raise ValueError("base_cap must be a positive integer")
        if type(minimum_cap) is not int or minimum_cap <= 0:
            raise ValueError("minimum_cap must be a positive integer")
        if type(quantum) is not int or quantum <= 0:
            raise ValueError("quantum must be a positive integer")
        if not math.isfinite(entropy) or not 0 <= entropy <= 1:
            raise ValueError("normalized entropy must be in [0, 1]")
        if (
            self.mode == "fixed"
            or entropy >= self.entropy_threshold
            or minimum_cap > base_cap
        ):
            return base_cap
        reduced = (round(base_cap * self.minimum_fraction) // quantum) * quantum
        return min(base_cap, max(minimum_cap, reduced))


@dataclass(frozen=True, slots=True)
class SearchExecutionConfig:
    first_visit_batch_size: int = 1
    subtree_reuse: bool = False
    subtree_reuse_max_nodes: int = 4_096
    full_budget: FullSearchBudgetConfig = FullSearchBudgetConfig()

    def __post_init__(self) -> None:
        if (
            type(self.first_visit_batch_size) is not int
            or not 1 <= self.first_visit_batch_size <= 64
        ):
            raise ValueError("first_visit_batch_size must be an integer in 1..64")
        if type(self.subtree_reuse) is not bool:
            raise ValueError("subtree_reuse must be boolean")
        if (
            type(self.subtree_reuse_max_nodes) is not int
            or not 1 <= self.subtree_reuse_max_nodes <= 65_536
        ):
            raise ValueError("subtree_reuse_max_nodes must be in 1..65536")
        if not isinstance(self.full_budget, FullSearchBudgetConfig):
            raise ValueError("full_budget requires typed settings")

    @property
    def enabled(self) -> bool:
        return (
            self.first_visit_batch_size > 1
            or self.subtree_reuse
            or self.full_budget.mode != "fixed"
        )

    def contract(self) -> dict[str, Any] | None:
        """Default settings do not alter an existing evaluation identity."""
        return None if self == SearchExecutionConfig() else asdict(self)

    def provenance(self) -> str:
        contract = self.contract()
        if contract is None:
            return ""
        encoded = json.dumps(contract, sort_keys=True, separators=(",", ":")).encode()
        return (
            f":execution=sha256-{hashlib.sha256(encoded).hexdigest()}"
            f":first_visit_batch_size={self.first_visit_batch_size}"
            f":subtree_reuse={str(self.subtree_reuse).lower()}"
            f":full_budget={self.full_budget.mode}-v1"
        )


def parse_search_execution(value: object) -> SearchExecutionConfig:
    if not isinstance(value, Mapping):
        raise ValueError("search_execution must be a mapping")
    values = dict(value)
    if "full_budget" in values:
        nested = values["full_budget"]
        if not isinstance(nested, Mapping):
            raise ValueError("search_execution.full_budget must be a mapping")
        values["full_budget"] = FullSearchBudgetConfig(**dict(nested))
    return SearchExecutionConfig(**values)


def require_search_execution(native: Any, execution: SearchExecutionConfig) -> None:
    if not execution.enabled:
        return
    version = getattr(native, "native_search_execution_version", None)
    actual = version() if callable(version) else None
    if type(actual) is not int or actual != 1:
        raise ValueError("search experiments require native_search_execution_version 1")


def normalized_root_entropy(logits: Sequence[float]) -> float:
    """Entropy of legal raw logits divided by log(number of legal actions)."""
    values = [float(value) for value in logits]
    if any(not math.isfinite(value) for value in values):
        raise ValueError("root policy logits must be finite")
    if len(values) <= 1:
        return 0.0
    maximum = max(values)
    relative = [value - maximum for value in values]
    mass = math.fsum(math.exp(value) for value in relative)
    log_mass = math.log(mass)
    entropy = math.fsum(
        -(math.exp(value) / mass) * (value - log_mass)
        for value in relative
        if math.exp(value) > 0
    )
    return min(1.0, max(0.0, entropy / math.log(len(values))))


def root_policy_entropies(requests: Any, response: Any) -> dict[int, float]:
    """Match response tokens to physical tree rows, including omitted terminals."""
    tokens = list(requests.tokens)
    rows = list(requests.tree_indices)
    offsets = list(requests.legal_offsets)
    returned = list(response.tokens)
    policy_offsets = list(response.policy_offsets)
    if (
        len(rows) != len(tokens)
        or len(set(rows)) != len(rows)
        or len(set(tokens)) != len(tokens)
        or len(offsets) != len(tokens) + 1
        or len(returned) != len(tokens)
        or len(set(returned)) != len(returned)
        or set(returned) != set(tokens)
        or len(policy_offsets) != len(returned) + 1
        or not policy_offsets
        or policy_offsets[0] != 0
        or policy_offsets[-1] != len(response.policy_logits)
    ):
        raise ValueError("root entropy requires matching token-addressed policy rows")
    expected = {
        token: (row, offsets[index + 1] - offsets[index])
        for index, (token, row) in enumerate(zip(tokens, rows, strict=True))
    }
    result = {}
    for index, token in enumerate(returned):
        start, end = policy_offsets[index : index + 2]
        row, count = expected[token]
        if start < 0 or end < start or end - start != count:
            raise ValueError(
                "root entropy policy row has an invalid legal action count"
            )
        result[row] = normalized_root_entropy(response.policy_logits[start:end])
    return result


def search_model_context(
    evaluator: Any, score_utility_weight: float | None = None
) -> str:
    """Process-local immutable adapter identity plus search-value semantics."""
    base = getattr(evaluator, "base", evaluator)
    identity = getattr(base, "model_identity", None)
    if not identity or identity == "unversioned":
        raise ValueError("subtree reuse requires an immutable model identity")
    config = getattr(evaluator, "config", None)
    base_config = getattr(base, "config", None)
    model = getattr(base, "model", None)
    raw_model = getattr(model, "_orig_mod", model)
    feature_version = getattr(
        base_config,
        "feature_schema_version",
        3 if getattr(base_config, "legacy_features", False) else FEATURE_SCHEMA_VERSION,
    )
    material = (
        id(base),
        id(model),
        identity,
        getattr(base, "model_version", None),
        getattr(base, "model_step", None),
        getattr(base, "namespace", None),
        getattr(base_config, "precision", None),
        feature_version,
        getattr(config, "feature_schema_version", feature_version),
        getattr(getattr(raw_model, "config", None), "feature_schema_version", None),
        getattr(base_config, "score_utility_weight", None),
        getattr(config, "score_utility_weight", None),
        getattr(evaluator, "score_utility_weight", None),
        score_utility_weight,
    )
    return "sha256-" + hashlib.sha256(repr(material).encode()).hexdigest()


def search_batch_row_limit(evaluator: Any, fallback_rows: int) -> int:
    broker = getattr(evaluator, "broker", None)
    maximum = getattr(broker, "max_batch_rows", fallback_rows)
    if type(maximum) is not int or maximum <= 0:
        raise ValueError("search inference row limit must be a positive integer")
    return maximum
