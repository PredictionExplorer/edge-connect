"""Token-addressed neural inference for native search request batches."""

from __future__ import annotations

import dataclasses
from collections import OrderedDict
from contextlib import nullcontext
import numbers
import struct
import threading
from dataclasses import dataclass
from typing import Protocol, Sequence, runtime_checkable

import numpy as np
import torch
from torch import nn

from .contracts import (
    FEATURE_SCHEMA_VERSION,
    LEGACY_FEATURE_SCHEMA_VERSION,
    RULES_HASH,
    SCORE_MARGIN_MAX,
    SCORE_MARGIN_MIN,
)
from .device import resolve_precision
from .features import EncodedBatch
from .features_v3 import encode_legacy_batch
from .inference_cache import BoundedPredictionCache, PinnedTransferPool, RawPrediction
from .native import (
    NativeStateDataProtocol,
    encode_native_feature_data,
    encode_native_state_data,
    positions_from_native,
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
    # The previous lineage's models (feature schema v3) evaluate through the
    # frozen Python encoder so cross-schema arenas can measure a transfer.
    feature_schema_version: int = FEATURE_SCHEMA_VERSION
    cache_max_entries: int = 0
    cache_max_bytes: int = 0
    deduplicate: bool = False
    pinned_transfers: bool = False
    pinned_buffer_slots: int = 2

    def __post_init__(self) -> None:
        if self.precision not in ("fp32", "bf16", "auto"):
            raise ValueError("inference precision must be fp32, bf16, or auto")
        if not 0 <= self.score_utility_weight <= 1:
            raise ValueError("score_utility_weight must be in [0, 1]")
        if self.feature_schema_version not in (
            FEATURE_SCHEMA_VERSION,
            LEGACY_FEATURE_SCHEMA_VERSION,
        ):
            raise ValueError(
                "inference feature_schema_version must be "
                f"{FEATURE_SCHEMA_VERSION} or {LEGACY_FEATURE_SCHEMA_VERSION}"
            )
        for name in ("cache_max_entries", "cache_max_bytes"):
            value = getattr(self, name)
            if type(value) is not int or value < 0:
                raise ValueError(f"{name} must be a non-negative integer")
        if bool(self.cache_max_entries) != bool(self.cache_max_bytes):
            raise ValueError(
                "cache entry and byte limits must both be positive or zero"
            )
        for name in ("deduplicate", "pinned_transfers"):
            if type(getattr(self, name)) is not bool:
                raise ValueError(f"{name} must be boolean")
        if (
            type(self.pinned_buffer_slots) is not int
            or not 1 <= self.pinned_buffer_slots <= 8
        ):
            raise ValueError("pinned_buffer_slots must be in [1, 8]")

    @property
    def legacy_features(self) -> bool:
        return self.feature_schema_version == LEGACY_FEATURE_SCHEMA_VERSION


InferenceNamespace = tuple[str, str, int, int, str, int]


@dataclass(frozen=True, slots=True)
class PreparedInferenceRequest:
    """Owned host snapshot, prepared independently of the GPU worker."""

    encoded: EncodedBatch
    tokens: tuple[int, ...]
    legal_offsets: tuple[int, ...]
    legal_actions: tuple[int, ...]
    namespace: InferenceNamespace
    score_utility_weight: float
    prepared_keys: tuple[bytes, ...] | None = None
    key_tensor_stamps: tuple[tuple[int, int, int], ...] | None = None

    @property
    def rows(self) -> int:
        return len(self.tokens)

    @property
    def ring(self) -> int | None:
        rings = self.encoded.rings.tolist()
        return (
            int(rings[0]) if rings and all(ring == rings[0] for ring in rings) else None
        )


@dataclass(frozen=True, slots=True)
class InferenceMetrics:
    """Monotonic evaluator counters suitable for batch-boundary deltas."""

    evaluator_calls: int = 0
    evaluator_rows: int = 0
    neural_calls: int = 0
    neural_rows: int = 0
    neural_padding_rows: int = 0
    cache_hits: int = 0
    cache_misses: int = 0
    cache_evictions: int = 0
    deduplicated_rows: int = 0

    def delta(self, previous: "InferenceMetrics") -> "InferenceMetrics":
        differences = {
            field.name: getattr(self, field.name) - getattr(previous, field.name)
            for field in dataclasses.fields(self)
        }
        if any(value < 0 for value in differences.values()):
            raise ValueError("inference metrics counters must be monotonic")
        return InferenceMetrics(**differences)


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
        homogeneous_relational_bias: bool = False,
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
        self.homogeneous_relational_bias = homogeneous_relational_bias
        self.last_feature_path: str | None = None
        self.feature_path_counts = {"rust": 0, "python": 0, "python-legacy": 0}
        self._evaluator_calls = 0
        self._evaluator_rows = 0
        self._neural_calls = 0
        self._neural_rows = 0
        self._neural_padding_rows = 0
        self._cache_hits = 0
        self._cache_misses = 0
        self._deduplicated_rows = 0
        self._evaluation_lock = threading.RLock()
        self._stats_lock = threading.Lock()
        self._prediction_cache = BoundedPredictionCache(
            max_entries=config.cache_max_entries, max_bytes=config.cache_max_bytes
        )
        self._cache_namespace: InferenceNamespace | None = None
        self._pinned_pool = (
            PinnedTransferPool(slots=config.pinned_buffer_slots, device=self.device)
            if config.pinned_transfers and self.device.type == "cuda"
            else None
        )
        self._topology_cache: OrderedDict[
            tuple[int, int, int, int],
            tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor],
        ] = OrderedDict()
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
            neural_calls=self._neural_calls,
            neural_rows=self._neural_rows,
            neural_padding_rows=self._neural_padding_rows,
            cache_hits=self._cache_hits,
            cache_misses=self._cache_misses,
            cache_evictions=self._prediction_cache.evictions,
            deduplicated_rows=self._deduplicated_rows,
        )

    @property
    def namespace(self) -> InferenceNamespace:
        return (
            self.model_identity,
            self.model_version,
            self.model_step,
            self.config.feature_schema_version,
            self.config.precision,
            RULES_HASH,
        )

    def efficiency_snapshot(self) -> dict[str, int]:
        """Monotonic counters plus current resident-cache/staging gauges."""

        metrics = dataclasses.asdict(self.metrics_snapshot())
        metrics.update(
            cache_entries=len(self._prediction_cache),
            cache_bytes=self._prediction_cache.bytes,
        )
        if self._pinned_pool is not None:
            metrics.update(
                pinned_allocated_bytes=self._pinned_pool.allocated_bytes,
                pinned_allocations=self._pinned_pool.allocations,
                pinned_reuses=self._pinned_pool.reuses,
                pinned_waits=self._pinned_pool.waits,
                transferred_bytes=self._pinned_pool.transferred_bytes,
            )
        return metrics

    def clear_inference_cache(self) -> None:
        with self._evaluation_lock:
            self._prediction_cache.clear()
            self._cache_namespace = None
            raw_model = getattr(self.model, "_orig_mod", self.model)
            clear = getattr(raw_model, "clear_inference_caches", None)
            if callable(clear):
                clear()

    def close(self) -> None:
        with self._evaluation_lock:
            self.clear_inference_cache()
            self._topology_cache.clear()
            if self._pinned_pool is not None:
                self._pinned_pool.close()

    def _to_device(self, encoded: EncodedBatch) -> EncodedBatch:
        ring_values = encoded.rings.tolist()
        if (
            self.device.type == "cpu"
            or not ring_values
            or any(ring != ring_values[0] for ring in ring_values[1:])
        ):
            if self._pinned_pool is not None:
                return EncodedBatch(
                    **self._pinned_pool.transfer(
                        {
                            field.name: getattr(encoded, field.name)
                            for field in dataclasses.fields(encoded)
                        }
                    )
                )
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
            while len(self._topology_cache) > 32:
                self._topology_cache.popitem(last=False)
        else:
            self._topology_cache.move_to_end(key)
        neighbor_index, neighbor_mask, neighbor_edge_type, node_mask = topology
        dynamic = {
            "node_features": encoded.node_features,
            "global_features": encoded.global_features,
            "legal_action_mask": encoded.legal_action_mask,
            "rings": encoded.rings,
        }
        uploaded = (
            self._pinned_pool.transfer(dynamic)
            if self._pinned_pool is not None
            else {
                name: tensor.to(self.device, non_blocking=True)
                for name, tensor in dynamic.items()
            }
        )
        return EncodedBatch(
            node_features=uploaded["node_features"],
            global_features=uploaded["global_features"],
            neighbor_index=neighbor_index,
            neighbor_mask=neighbor_mask,
            neighbor_edge_type=neighbor_edge_type,
            node_mask=node_mask,
            legal_action_mask=uploaded["legal_action_mask"],
            rings=uploaded["rings"],
        )

    def set_score_utility_weight(self, weight: float) -> None:
        """Retarget the search utility between batches, never mid-cohort."""

        self.config = dataclasses.replace(self.config, score_utility_weight=weight)

    def evaluate(self, requests: NativeEvalBatchProtocol) -> InferenceResponse:
        with self._evaluation_lock:
            response, _ = self._evaluate(requests, include_details=False)
        return response

    def evaluate_detailed(
        self, requests: NativeEvalBatchProtocol
    ) -> DetailedInferenceResponse:
        with self._evaluation_lock:
            response, details = self._evaluate(requests, include_details=True)
        assert details is not None
        return details

    def prepare_requests(
        self,
        requests: NativeEvalBatchProtocol,
        *,
        score_utility_weight: float | None = None,
    ) -> PreparedInferenceRequest:
        """Validate and snapshot inputs on a CPU producer, before GPU submission.

        An immutable model identity must remain pinned until the result arrives.
        No GPU operations or prediction-cache accesses happen in this method.
        """

        namespace = self.namespace
        weight = (
            self.config.score_utility_weight
            if score_utility_weight is None
            else score_utility_weight
        )
        if not 0 <= weight <= 1:
            raise ValueError("score utility weight must be in [0, 1]")
        native_features = getattr(requests, "features", None)
        if native_features is not None and all(
            isinstance(value, list)
            for value in (
                requests.tokens,
                requests.legal_offsets,
                requests.legal_actions,
            )
        ):
            tokens = tuple(requests.tokens)
            offsets = tuple(requests.legal_offsets)
            actions = tuple(requests.legal_actions)
        else:
            tokens = tuple(_integer_list("tokens", requests.tokens))
            offsets = tuple(_integer_list("legal_offsets", requests.legal_offsets))
            actions = tuple(_integer_list("legal_actions", requests.legal_actions))
        rows = len(tokens)
        if rows <= 0 or len(requests) != rows:
            raise ValueError(
                "prepared requests must contain a nonempty matching token batch"
            )
        if (
            len(offsets) != rows + 1
            or offsets[0] != 0
            or offsets[-1] != len(actions)
            or any(
                left > right
                for left, right in zip(offsets[:-1], offsets[1:], strict=True)
            )
        ):
            raise ValueError("legal action CSR offsets are invalid")
        if self.config.legacy_features:
            host = encode_legacy_batch(positions_from_native(requests.states))
            feature_path = "python-legacy"
        elif native_features is not None:
            host = encode_native_feature_data(native_features, source="native_request")
            feature_path = "rust"
        else:
            native_encoder = callable(getattr(requests.states, "feature_data", None))
            host = encode_native_state_data(requests.states)
            feature_path = "rust" if native_encoder else "python"
        if host.batch_size != rows:
            raise ValueError("state row count and tokens disagree")
        counts = host.legal_action_mask.sum(dim=1, dtype=torch.int64)
        expected_offsets = (0, *counts.cumsum(dim=0).tolist())
        expected_actions = tuple(
            torch.nonzero(host.legal_action_mask, as_tuple=False)[:, 1].tolist()
        )
        if offsets != expected_offsets or actions != expected_actions:
            raise ValueError("native legal action order is not ascending node-only")
        # Rust buffers are writable exports. Own the queued snapshot so reuse or
        # mutation by the producer cannot change an accepted request in flight.
        # Preserve version counters even when a producer uses inference_mode.
        # Direct callers' later PyTorch mutations then invalidate prepared keys.
        with torch.inference_mode(False):
            owned = EncodedBatch(
                **{
                    field.name: getattr(host, field.name).detach().clone()
                    for field in dataclasses.fields(host)
                }
            )
        keyed = self._prediction_cache.enabled or self.config.deduplicate
        keys = tuple(self._row_keys(owned, namespace)) if keyed else None
        stamps = self._tensor_stamps(owned) if keyed else None
        with self._stats_lock:
            self.last_feature_path = feature_path
            self.feature_path_counts[feature_path] += 1
        return PreparedInferenceRequest(
            owned, tokens, offsets, actions, namespace, float(weight), keys, stamps
        )

    @staticmethod
    def _tensor_stamps(
        encoded: EncodedBatch,
    ) -> tuple[tuple[int, int, int], ...] | None:
        """Detect tensor replacement, storage replacement and PyTorch mutations."""

        try:
            return tuple(
                (id(tensor), tensor.data_ptr(), tensor._version)
                for tensor in encoded.model_args()
            )
        except RuntimeError:
            # Direct callers can supply tensors lacking version counters;
            # those requests must rebuild keys on the inference owner.
            return None

    @staticmethod
    def _row_keys(encoded: EncodedBatch, namespace: InferenceNamespace) -> list[bytes]:
        """Complete bytes, not a lossy state hash: includes every model input."""

        prefix = repr(namespace).encode("utf-8") + b"\0"
        serialized: list[tuple[bytes, bytes, int]] = []
        for field in dataclasses.fields(encoded):
            tensor = getattr(encoded, field.name).detach().contiguous()
            row_shape = tuple(tensor.shape[1:])
            header = f"{field.name}:{tensor.dtype}:{row_shape}".encode("ascii")
            row_bytes = tensor.element_size()
            for dimension in row_shape:
                row_bytes *= dimension
            payload = tensor.reshape(-1).view(torch.uint8).numpy().tobytes()
            framing = (
                struct.pack("<I", len(header)) + header + struct.pack("<I", row_bytes)
            )
            serialized.append((framing, payload, row_bytes))
        keys: list[bytes] = []
        for row in range(encoded.batch_size):
            parts = [prefix]
            for framing, payload, row_bytes in serialized:
                offset = row * row_bytes
                parts.extend((framing, payload[offset : offset + row_bytes]))
            keys.append(b"".join(parts))
        return keys

    def _run_raw_predictions(
        self, host: EncodedBatch, *, ring: int | None, logical_rows: int | None = None
    ) -> list[RawPrediction]:
        requested_rows = host.batch_size if logical_rows is None else logical_rows
        if not 0 < requested_rows <= host.batch_size:
            raise ValueError("logical inference rows must fit the physical batch")
        encoded = self._to_device(host)
        was_training = self.model.training
        self.model.eval()
        kwargs: dict[str, object] = {}
        if (
            self.homogeneous_relational_bias
            and ring is not None
            and torch.equal(
                host.node_mask, host.node_mask[:1].expand_as(host.node_mask)
            )
        ):
            kwargs["homogeneous_ring"] = ring
            raw_model = getattr(self.model, "_orig_mod", self.model)
            prepare_bias = getattr(raw_model, "prepare_inference_relational_bias", None)
            if callable(prepare_bias):
                kwargs["inference_relation_bias"] = prepare_bias(
                    ring,
                    dtype=torch.bfloat16
                    if self.config.precision == "bf16"
                    else encoded.node_features.dtype,
                )
        try:
            with (
                torch.inference_mode(),
                torch.autocast(
                    device_type=self.device.type,
                    dtype=torch.bfloat16,
                    enabled=self.config.precision == "bf16",
                ),
            ):
                output = self.model(*encoded.model_args(), **kwargs)
                bins = SCORE_MARGIN_MAX - SCORE_MARGIN_MIN + 1
                rows = host.batch_size
                if (
                    output.policy_logits.shape != (rows, host.max_nodes)
                    or output.outcome_logits.shape != (rows, 2)
                    or output.score_margin_logits.shape != (rows, bins)
                    or output.ownership_logits.shape != (rows, host.max_nodes, 3)
                    or output.alive_logits.shape != (rows, host.max_nodes)
                    or output.soft_policy_logits.shape != (rows, host.max_nodes)
                ):
                    raise ValueError("model output shapes violate inference schema")
                packed = torch.cat(
                    (
                        output.policy_logits.float(),
                        output.outcome_logits.float(),
                        output.score_margin_logits.float(),
                    ),
                    dim=1,
                ).cpu()
                if not torch.isfinite(packed[:, host.max_nodes :]).all():
                    raise ValueError("non-finite neural value predictions")
                if not torch.isfinite(
                    packed[:, : host.max_nodes].masked_select(host.legal_action_mask)
                ).all():
                    raise ValueError("non-finite legal neural policy predictions")
                self._neural_calls += 1
                self._neural_rows += rows
                self._neural_padding_rows += rows - requested_rows
        finally:
            if was_training:
                self.model.train()
        return [
            RawPrediction(row.numpy().tobytes(), host.max_nodes)
            for row in packed[:requested_rows]
        ]

    def _inference_batch_rows(self, rows: int) -> int:
        """Reuse a small set of CUDA backend plans as cache-miss counts vary.

        A new arbitrary CUDA batch shape has a measurable first-use planning
        cost. Repeating the last valid row up to a power of two avoids it;
        outputs of the repeated rows never become search results or cache data.
        CPU inference retains its exact row count.
        """

        return 1 << (rows - 1).bit_length() if self.device.type == "cuda" else rows

    def evaluate_prepared(
        self,
        requests: Sequence[PreparedInferenceRequest],
        *,
        include_details: Sequence[bool] | None = None,
    ) -> list[tuple[InferenceResponse, DetailedInferenceResponse | None]]:
        """Run compatible requests together on the single inference owner.

        Different cohort score weights are applied after raw neural prediction,
        so sharing/cache hits never inherit another game's utility setting.
        """

        if not requests:
            return []
        details_flags = (
            tuple(include_details)
            if include_details is not None
            else (False,) * len(requests)
        )
        if len(details_flags) != len(requests):
            raise ValueError("detail flags must match prepared requests")
        with (
            self._evaluation_lock,
            (
                torch.cuda.device(self.device)
                if self.device.type == "cuda"
                else nullcontext()
            ),
        ):
            namespace = self.namespace
            first = requests[0]
            if any(request.namespace != namespace for request in requests):
                raise RuntimeError(
                    "model identity changed while inference requests were queued"
                )
            ring = first.ring
            if len(requests) > 1 and (
                ring is None or any(request.ring != ring for request in requests)
            ):
                raise ValueError("batched requests must have one common ring")
            if any(
                request.encoded.max_nodes != first.encoded.max_nodes
                for request in requests
            ):
                raise ValueError("batched inference shapes are incompatible")
            if self._cache_namespace != namespace:
                self._prediction_cache.clear()
                self._cache_namespace = namespace
            if self._prediction_cache.enabled and self.model_identity == "unversioned":
                raise ValueError(
                    "prediction caching requires an immutable model identity"
                )
            host = (
                first.encoded
                if len(requests) == 1
                else EncodedBatch(
                    **{
                        field.name: torch.cat(
                            [
                                getattr(request.encoded, field.name)
                                for request in requests
                            ],
                            dim=0,
                        )
                        for field in dataclasses.fields(first.encoded)
                    }
                )
            )
            rows = host.batch_size
            keyed = self._prediction_cache.enabled or self.config.deduplicate
            keys: list[bytes] = []
            if keyed:
                for request in requests:
                    stamps = self._tensor_stamps(request.encoded)
                    if (
                        request.prepared_keys is not None
                        and len(request.prepared_keys) == request.rows
                        and stamps is not None
                        and stamps == request.key_tensor_stamps
                    ):
                        keys.extend(request.prepared_keys)
                    else:
                        counts = request.encoded.legal_action_mask.sum(
                            dim=1, dtype=torch.int64
                        )
                        offsets = (0, *counts.cumsum(dim=0).tolist())
                        actions = tuple(
                            torch.nonzero(
                                request.encoded.legal_action_mask, as_tuple=False
                            )[:, 1].tolist()
                        )
                        if (
                            offsets != request.legal_offsets
                            or actions != request.legal_actions
                        ):
                            raise ValueError(
                                "prepared legal-action metadata changed after validation"
                            )
                        keys.extend(self._row_keys(request.encoded, namespace))
            predictions: list[RawPrediction | None] = [None] * rows
            misses: list[int] = []
            pending_keys: dict[bytes, int] = {}
            duplicate_of: dict[int, int] = {}
            for row in range(rows):
                prediction = (
                    self._prediction_cache.get(keys[row])
                    if self._prediction_cache.enabled
                    else None
                )
                if prediction is not None:
                    self._cache_hits += 1
                    predictions[row] = prediction
                    continue
                self._cache_misses += 1
                if self.config.deduplicate and keys[row] in pending_keys:
                    duplicate_of[row] = pending_keys[keys[row]]
                    self._deduplicated_rows += 1
                    continue
                misses.append(row)
                if self.config.deduplicate:
                    pending_keys[keys[row]] = row
            if misses:
                physical_rows = self._inference_batch_rows(len(misses))
                indices = torch.tensor(
                    misses + [misses[-1]] * (physical_rows - len(misses)),
                    dtype=torch.long,
                )
                selected = (
                    host
                    if len(misses) == rows and physical_rows == rows
                    else EncodedBatch(
                        **{
                            field.name: getattr(host, field.name).index_select(
                                0, indices
                            )
                            for field in dataclasses.fields(host)
                        }
                    )
                )
                fresh = self._run_raw_predictions(
                    selected, ring=ring, logical_rows=len(misses)
                )
                for row, prediction in zip(misses, fresh, strict=True):
                    predictions[row] = prediction
                    if self._prediction_cache.enabled:
                        self._prediction_cache.put(keys[row], prediction)
            for row, source in duplicate_of.items():
                predictions[row] = predictions[source]
            if self.namespace != namespace:
                self._prediction_cache.clear()
                raise RuntimeError("model identity changed during neural inference")
            self._evaluator_calls += len(requests)
            self._evaluator_rows += rows
            if any(prediction is None for prediction in predictions):
                raise RuntimeError("incomplete neural prediction batch")
            # Cache records own compact bytes. Stacking creates writable CPU
            # storage and never exposes a mutable view of cache contents.
            raw = torch.from_numpy(
                np.stack(
                    [
                        np.frombuffer(prediction.packed, dtype=np.float32)
                        for prediction in predictions
                        if prediction is not None
                    ]
                )
            )
            nodes = host.max_nodes
            outcome = torch.softmax(raw[:, nodes : nodes + 2], dim=-1)
            outcome_values = outcome[:, 1] - outcome[:, 0]
            score_probability = torch.softmax(raw[:, nodes + 2 :], dim=-1)
            support = torch.arange(
                SCORE_MARGIN_MIN, SCORE_MARGIN_MAX + 1, dtype=torch.float32
            )
            expectations = (score_probability * support).sum(dim=-1)
            scale = max(abs(SCORE_MARGIN_MIN), SCORE_MARGIN_MAX)
            results: list[
                tuple[InferenceResponse, DetailedInferenceResponse | None]
            ] = []
            start = 0
            for request, detailed in zip(requests, details_flags, strict=True):
                end = start + request.rows
                values = outcome_values[start:end]
                if request.score_utility_weight:
                    values = (
                        values
                        + request.score_utility_weight * expectations[start:end] / scale
                    ).clamp(-1, 1)
                logits = raw[start:end, :nodes].masked_select(
                    request.encoded.legal_action_mask
                )
                response = InferenceResponse(
                    list(request.tokens),
                    values.tolist(),
                    list(request.legal_offsets),
                    logits.tolist(),
                )
                detail = (
                    DetailedInferenceResponse(
                        response,
                        outcome[start:end].tolist(),
                        outcome_values[start:end].tolist(),
                        expectations[start:end].tolist(),
                        score_probability[start:end].tolist(),
                    )
                    if detailed
                    else None
                )
                results.append((response, detail))
                start = end
            return results

    def _evaluate(
        self,
        requests: NativeEvalBatchProtocol,
        *,
        include_details: bool,
    ) -> tuple[InferenceResponse, DetailedInferenceResponse | None]:
        if len(requests) and (
            self._prediction_cache.enabled
            or self.config.deduplicate
            or self.homogeneous_relational_bias
        ):
            prepared = self.prepare_requests(requests)
            return self.evaluate_prepared(
                (prepared,), include_details=(include_details,)
            )[0]
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

        if self.config.legacy_features:
            # Search always packs the production schema; the previous lineage
            # re-encodes the semantic states with its frozen v3 encoder.
            host_encoded = encode_legacy_batch(positions_from_native(requests.states))
            feature_path = "python-legacy"
        elif native_features is not None:
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
                self._neural_calls += 1
                self._neural_rows += rows
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
