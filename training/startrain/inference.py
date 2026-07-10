"""Token-addressed neural inference for native search request batches."""

from __future__ import annotations

import numbers
from dataclasses import dataclass
from typing import Protocol, Sequence, runtime_checkable

import torch
from torch import nn

from .contracts import SCORE_MARGIN_MAX, SCORE_MARGIN_MIN
from .native import NativeStateDataProtocol, encode_native_state_data


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

        encoded = encode_native_state_data(requests.states).to(self.device)
        if encoded.batch_size != rows:
            raise ValueError("state row count and tokens disagree")
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
                dense_logits = output.policy_logits.float().clone()
        finally:
            self.model.train(was_training)

        dense_logits = dense_logits.cpu().clone()
        values = values.cpu()
        openings = list(requests.states.opening)
        node_count = encoded.max_nodes
        if len(openings) != rows:
            raise ValueError("opening metadata row count is invalid")
        for row, opening in enumerate(openings):
            if opening and self.config.initial_pass_logit_penalty:
                # Pass remains legal and represented; only its initial prior is reduced.
                dense_logits[row, node_count] -= self.config.initial_pass_logit_penalty

        flattened: list[float] = []
        for row in range(rows):
            start, end = legal_offsets[row], legal_offsets[row + 1]
            actions = legal_actions[start:end]
            expected = (
                torch.nonzero(
                    encoded.legal_action_mask[row, :node_count], as_tuple=False
                )
                .flatten()
                .tolist()
            )
            if bool(encoded.legal_action_mask[row, node_count]):
                expected.append(-1)
            if actions != expected:
                raise ValueError(
                    "native legal action order does not match nodes-then-pass"
                )
            for action in actions:
                index = node_count if action == -1 else action
                if index < 0 or index > node_count:
                    raise ValueError("native legal action code is invalid")
                flattened.append(float(dense_logits[row, index]))
        response = InferenceResponse(
            tokens=tokens,
            values=[float(value) for value in values],
            policy_offsets=legal_offsets,
            policy_logits=flattened,
        )
        if not include_details:
            return response, None
        assert score_probability is not None and score_belief is not None
        details = DetailedInferenceResponse(
            response=response,
            wdl_probabilities=wdl.cpu().tolist(),
            wdl_values=[float(value) for value in wdl_values.cpu()],
            score_expectations=[
                float(value * max(abs(SCORE_MARGIN_MIN), SCORE_MARGIN_MAX))
                for value in score_belief.cpu()
            ],
            score_probabilities=score_probability.cpu().tolist(),
        )
        return response, details
