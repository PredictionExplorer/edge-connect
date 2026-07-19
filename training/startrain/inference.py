"""Token-addressed neural inference for native search request batches."""

from __future__ import annotations

import dataclasses
import numbers
from dataclasses import dataclass
from typing import Protocol, Sequence, runtime_checkable

import torch
from torch import nn

from .contracts import SCORE_MARGIN_MAX, SCORE_MARGIN_MIN
from .device import resolve_precision
from .features import EncodedBatch
from .native import (
    NativeStateDataProtocol,
    encode_native_feature_data,
    encode_native_state_data,
)


@runtime_checkable
class NativeEvalBatchProtocol(Protocol):
    tokens: Sequence[int]
    states: NativeStateDataProtocol
    legal_offsets: Sequence[int]
    legal_actions: Sequence[int]

    def __len__(self) -> int: ...


@dataclass(frozen=True, slots=True)
class InferenceResponse:
    tokens: list[int]
    values: list[float]
    policy_offsets: list[int]
    policy_logits: list[float]

    def submit_args(self) -> tuple[list[int], list[float], list[int], list[float]]:
        return self.tokens, self.values, self.policy_offsets, self.policy_logits


@dataclass(frozen=True, slots=True)
class DetailedInferenceResponse:
    response: InferenceResponse
    outcome_probabilities: list[list[float]]
    outcome_values: list[float]
    score_expectations: list[float]
    score_probabilities: list[list[float]]


@dataclass(frozen=True, slots=True)
class InferenceConfig:
    precision: str = "fp32"
    score_utility_weight: float = 0.0

    def __post_init__(self) -> None:
        if self.precision not in ("fp32", "bf16", "auto"):
            raise ValueError("inference precision must be fp32, bf16, or auto")
        if not 0 <= self.score_utility_weight <= 1:
            raise ValueError("score_utility_weight must be in [0, 1]")


@dataclass(frozen=True, slots=True)
class InferenceMetrics:
    """Monotonic evaluator counters suitable for batch-boundary deltas."""

    evaluator_calls: int = 0
    evaluator_rows: int = 0

    def delta(self, previous: "InferenceMetrics") -> "InferenceMetrics":
        calls = self.evaluator_calls - previous.evaluator_calls
        rows = self.evaluator_rows - previous.evaluator_rows
        if calls < 0 or rows < 0:
            raise ValueError("inference metrics counters must be monotonic")
        return InferenceMetrics(evaluator_calls=calls, evaluator_rows=rows)


def _integer_list(name: str, values: Sequence[int]) -> list[int]:
    output: list[int] = []
    for value in values:
        if isinstance(value, bool) or not isinstance(value, numbers.Integral):
            raise ValueError(f"{name} must contain integers")
        output.append(int(value))
    return output


class GraphInferenceAdapter:
    """Runs ``GraphResTNet`` and emits buffers accepted by ``SearchBatch``."""

    def __init__(
        self,
        model: nn.Module,
        *,
        device: torch.device | str = "cpu",
        config: InferenceConfig = InferenceConfig(),
        model_version: str = "unversioned",
        model_step: int = 0,
        model_identity: str | None = None,
    ) -> None:
        self.model = model.to(device)
        self.device = torch.device(device)
        resolved_precision = resolve_precision(config.precision, self.device)
        if resolved_precision != config.precision:
            config = dataclasses.replace(config, precision=resolved_precision)
        self.config = config
        self.model_version = model_version
        self.model_step = int(model_step)
        self.model_identity = model_identity or model_version
        self.last_feature_path: str | None = None
        self.feature_path_counts = {"rust": 0, "python": 0}
        self._evaluator_calls = 0
        self._evaluator_rows = 0
        self._topology_cache: dict[
            tuple[int, int, int, int],
            tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor],
        ] = {}
        self._score_support: torch.Tensor | None = None

    @property
    def evaluator_calls(self) -> int:
        return self._evaluator_calls

    @property
    def evaluator_rows(self) -> int:
        return self._evaluator_rows

    def metrics_snapshot(self) -> InferenceMetrics:
        return InferenceMetrics(
            evaluator_calls=self._evaluator_calls,
            evaluator_rows=self._evaluator_rows,
        )

    def _to_device(self, encoded: EncodedBatch) -> EncodedBatch:
        ring_values = encoded.rings.tolist()
        if (
            self.device.type == "cpu"
            or not ring_values
            or any(ring != ring_values[0] for ring in ring_values[1:])
        ):
            return encoded.to(self.device)

        batch_size = encoded.batch_size
        key = (
            int(ring_values[0]),
            batch_size,
            encoded.max_nodes,
            int(encoded.neighbor_index.shape[-1]),
        )
        topology = self._topology_cache.get(key)
        if topology is None:
            topology = (
                encoded.neighbor_index[0]
                .to(self.device, non_blocking=True)
                .unsqueeze(0)
                .expand(batch_size, -1, -1)
                .contiguous(),
                encoded.neighbor_mask[0]
                .to(self.device, non_blocking=True)
                .unsqueeze(0)
                .expand(batch_size, -1, -1)
                .contiguous(),
                encoded.neighbor_edge_type[0]
                .to(self.device, non_blocking=True)
                .unsqueeze(0)
                .expand(batch_size, -1, -1)
                .contiguous(),
                encoded.node_mask[0]
                .to(self.device, non_blocking=True)
                .unsqueeze(0)
                .expand(batch_size, -1)
                .contiguous(),
            )
            self._topology_cache[key] = topology
        neighbor_index, neighbor_mask, neighbor_edge_type, node_mask = topology
        return EncodedBatch(
            node_features=encoded.node_features.to(self.device, non_blocking=True),
            global_features=encoded.global_features.to(self.device, non_blocking=True),
            neighbor_index=neighbor_index,
            neighbor_mask=neighbor_mask,
            neighbor_edge_type=neighbor_edge_type,
            node_mask=node_mask,
            legal_action_mask=encoded.legal_action_mask.to(
                self.device, non_blocking=True
            ),
            rings=encoded.rings.to(self.device, non_blocking=True),
        )

    def evaluate(self, requests: NativeEvalBatchProtocol) -> InferenceResponse:
        response, _ = self._evaluate(requests, include_details=False)
        return response

    def evaluate_detailed(
        self, requests: NativeEvalBatchProtocol
    ) -> DetailedInferenceResponse:
        response, details = self._evaluate(requests, include_details=True)
        assert details is not None
        return details

    def _evaluate(
        self,
        requests: NativeEvalBatchProtocol,
        *,
        include_details: bool,
    ) -> tuple[InferenceResponse, DetailedInferenceResponse | None]:
        native_features = getattr(requests, "features", None)
        if (
            native_features is not None
            and isinstance(requests.tokens, list)
            and isinstance(requests.legal_offsets, list)
            and isinstance(requests.legal_actions, list)
        ):
            # Validated PyO3 requests expose exact ``list[int]`` buffers. Their
            # legality is checked against the independently encoded mask below,
            # so avoid converting and type-checking tens of thousands of legal
            # actions in Python for every search wave.
            tokens = requests.tokens.copy()
            legal_offsets = requests.legal_offsets.copy()
            legal_actions = requests.legal_actions.copy()
        else:
            tokens = _integer_list("tokens", requests.tokens)
            legal_offsets = _integer_list("legal_offsets", requests.legal_offsets)
            legal_actions = _integer_list("legal_actions", requests.legal_actions)
        rows = len(tokens)
        if len(requests) != rows:
            raise ValueError("request length and token count disagree")
        if rows == 0:
            if legal_offsets != [0] or legal_actions:
                raise ValueError("empty request batches require offsets [0]")
            self._evaluator_calls += 1
            response = InferenceResponse([], [], [0], [])
            details = (
                DetailedInferenceResponse(response, [], [], [], [])
                if include_details
                else None
            )
            return response, details
        if (
            len(legal_offsets) != rows + 1
            or legal_offsets[0] != 0
            or legal_offsets[-1] != len(legal_actions)
            or any(
                left > right
                for left, right in zip(
                    legal_offsets[:-1], legal_offsets[1:], strict=True
                )
            )
        ):
            raise ValueError("legal action CSR offsets are invalid")

        if native_features is not None:
            host_encoded = encode_native_feature_data(
                native_features, source="native_request"
            )
            feature_path = "rust"
        else:
            has_state_export = callable(getattr(requests.states, "feature_data", None))
            host_encoded = encode_native_state_data(requests.states)
            feature_path = "rust" if has_state_export else "python"
        encoded = self._to_device(host_encoded)
        self.last_feature_path = feature_path
        self.feature_path_counts[feature_path] += 1
        if encoded.batch_size != rows:
            raise ValueError("state row count and tokens disagree")
        legal_counts = host_encoded.legal_action_mask.sum(dim=1, dtype=torch.int64)
        expected_offsets = [0, *legal_counts.cumsum(dim=0).tolist()]
        expected_indices = torch.nonzero(
            host_encoded.legal_action_mask, as_tuple=False
        )[:, 1]
        expected_actions = expected_indices.tolist()
        if legal_offsets != expected_offsets or legal_actions != expected_actions:
            raise ValueError("native legal action order is not ascending node-only")
        self._evaluator_calls += 1
        self._evaluator_rows += rows
        was_training = self.model.training
        if was_training:
            self.model.eval()
        autocast = self.config.precision == "bf16"
        if autocast and self.device.type not in ("cpu", "cuda"):
            raise ValueError(f"BF16 inference is unsupported on {self.device.type}")
        try:
            with (
                torch.inference_mode(),
                torch.autocast(
                    device_type=self.device.type,
                    dtype=torch.bfloat16,
                    enabled=autocast,
                ),
            ):
                output = self.model(*encoded.model_args())
                expected_margin_bins = SCORE_MARGIN_MAX - SCORE_MARGIN_MIN + 1
                if (
                    output.policy_logits.shape != encoded.legal_action_mask.shape
                    or output.outcome_logits.shape != (rows, 2)
                    or output.score_margin_logits.shape != (rows, expected_margin_bins)
                    or output.ownership_logits.shape != (rows, encoded.max_nodes, 3)
                    or output.alive_logits.shape != (rows, encoded.max_nodes)
                    or output.soft_policy_logits.shape
                    != encoded.legal_action_mask.shape
                ):
                    raise ValueError("model output shapes violate schema v2")
                outcome = torch.softmax(output.outcome_logits.float(), dim=-1)
                outcome_values = outcome[:, 1] - outcome[:, 0]
                values = outcome_values
                score_probability = None
                score_belief = None
                if self.config.score_utility_weight or include_details:
                    if self._score_support is None:
                        self._score_support = torch.arange(
                            SCORE_MARGIN_MIN,
                            SCORE_MARGIN_MAX + 1,
                            device=self.device,
                            dtype=torch.float32,
                        )
                    score_probability = torch.softmax(
                        output.score_margin_logits.float(), dim=-1
                    )
                    score_belief = (score_probability * self._score_support).sum(dim=-1)
                    score_belief = score_belief / max(
                        abs(SCORE_MARGIN_MIN), SCORE_MARGIN_MAX
                    )
                if self.config.score_utility_weight:
                    assert score_belief is not None
                    values = (
                        values + self.config.score_utility_weight * score_belief
                    ).clamp(-1, 1)
                legal_logits = output.policy_logits.float().masked_select(
                    encoded.legal_action_mask
                )
        finally:
            if was_training:
                self.model.train()

        if include_details:
            flattened = legal_logits.cpu().tolist()
            host_values = values.cpu().tolist()
        else:
            packed = torch.cat((values.float(), legal_logits))
            host = packed.cpu().tolist()
            host_values = host[:rows]
            flattened = host[rows:]
        response = InferenceResponse(
            tokens=tokens,
            values=host_values,
            policy_offsets=legal_offsets,
            policy_logits=flattened,
        )
        if not include_details:
            return response, None
        assert score_probability is not None and score_belief is not None
        details = DetailedInferenceResponse(
            response=response,
            outcome_probabilities=outcome.cpu().tolist(),
            outcome_values=outcome_values.cpu().tolist(),
            score_expectations=(
                score_belief * max(abs(SCORE_MARGIN_MIN), SCORE_MARGIN_MAX)
            )
            .cpu()
            .tolist(),
            score_probabilities=score_probability.cpu().tolist(),
        )
        return response, details
