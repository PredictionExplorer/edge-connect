"""Token-addressed neural inference for native search request batches."""

from __future__ import annotations

import numbers
from dataclasses import dataclass
from typing import Protocol, Sequence, runtime_checkable

import torch
from torch import nn

from .contracts import SCORE_MARGIN_MAX, SCORE_MARGIN_MIN
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
    wdl_probabilities: list[list[float]]
    wdl_values: list[float]
    score_expectations: list[float]
    score_probabilities: list[list[float]]


@dataclass(frozen=True, slots=True)
class InferenceConfig:
    precision: str = "fp32"
    score_utility_weight: float = 0.0
    initial_pass_logit_penalty: float = 1.5

    def __post_init__(self) -> None:
        if self.precision not in ("fp32", "bf16"):
            raise ValueError("inference precision must be fp32 or bf16")
        if not 0 <= self.score_utility_weight <= 1:
            raise ValueError("score_utility_weight must be in [0, 1]")
        if self.initial_pass_logit_penalty < 0:
            raise ValueError("initial_pass_logit_penalty must be non-negative")


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
        self.config = config
        self.model_version = model_version
        self.model_step = int(model_step)
        self.model_identity = model_identity or model_version
        self.last_feature_path: str | None = None
        self.feature_path_counts = {"rust": 0, "python": 0}
        self._topology_cache: dict[
            tuple[int, int, int, int],
            tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor],
        ] = {}

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
                .to(self.device)
                .unsqueeze(0)
                .expand(batch_size, -1, -1)
                .contiguous(),
                encoded.neighbor_mask[0]
                .to(self.device)
                .unsqueeze(0)
                .expand(batch_size, -1, -1)
                .contiguous(),
                encoded.neighbor_edge_type[0]
                .to(self.device)
                .unsqueeze(0)
                .expand(batch_size, -1, -1)
                .contiguous(),
                encoded.node_mask[0]
                .to(self.device)
                .unsqueeze(0)
                .expand(batch_size, -1)
                .contiguous(),
            )
            self._topology_cache[key] = topology
        neighbor_index, neighbor_mask, neighbor_edge_type, node_mask = topology
        return EncodedBatch(
            node_features=encoded.node_features.to(self.device),
            global_features=encoded.global_features.to(self.device),
            neighbor_index=neighbor_index,
            neighbor_mask=neighbor_mask,
            neighbor_edge_type=neighbor_edge_type,
            node_mask=node_mask,
            legal_action_mask=encoded.legal_action_mask.to(self.device),
            rings=encoded.rings.to(self.device),
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
        node_count = encoded.max_nodes
        legal_counts = host_encoded.legal_action_mask.sum(dim=1, dtype=torch.int64)
        expected_offsets = [0, *legal_counts.cumsum(dim=0).tolist()]
        expected_indices = torch.nonzero(
            host_encoded.legal_action_mask, as_tuple=False
        )[:, 1]
        expected_actions = expected_indices.masked_fill(
            expected_indices == node_count, -1
        ).tolist()
        if legal_offsets != expected_offsets or legal_actions != expected_actions:
            raise ValueError(
                "native legal action order does not match nodes-then-pass"
            )
        was_training = self.model.training
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
                wdl = torch.softmax(output.wdl_logits.float(), dim=-1)
                wdl_values = wdl[:, 2] - wdl[:, 0]
                values = wdl_values
                score_probability = None
                score_belief = None
                if self.config.score_utility_weight or include_details:
                    support = torch.arange(
                        SCORE_MARGIN_MIN,
                        SCORE_MARGIN_MAX + 1,
                        device=self.device,
                        dtype=torch.float32,
                    )
                    score_probability = torch.softmax(
                        output.score_margin_logits.float(), dim=-1
                    )
                    score_belief = (score_probability * support).sum(dim=-1)
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
            self.model.train(was_training)

        flattened = legal_logits.cpu().tolist()
        openings = list(requests.states.opening)
        if len(openings) != rows:
            raise ValueError("opening metadata row count is invalid")
        if self.config.initial_pass_logit_penalty:
            # Native legal actions are nodes-then-pass, so a legal pass is the
            # final flattened logit for its row. Apply the opening prior after
            # the single device-to-host copy instead of synchronizing per row.
            penalty = self.config.initial_pass_logit_penalty
            for row, opening in enumerate(openings):
                if opening:
                    end = legal_offsets[row + 1]
                    if end > legal_offsets[row] and legal_actions[end - 1] == -1:
                        flattened[end - 1] -= penalty
        response = InferenceResponse(
            tokens=tokens,
            values=values.cpu().tolist(),
            policy_offsets=legal_offsets,
            policy_logits=flattened,
        )
        if not include_details:
            return response, None
        assert score_probability is not None and score_belief is not None
        details = DetailedInferenceResponse(
            response=response,
            wdl_probabilities=wdl.cpu().tolist(),
            wdl_values=wdl_values.cpu().tolist(),
            score_expectations=(
                score_belief * max(abs(SCORE_MARGIN_MIN), SCORE_MARGIN_MAX)
            )
            .cpu()
            .tolist(),
            score_probabilities=score_probability.cpu().tolist(),
        )
        return response, details
