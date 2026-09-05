from __future__ import annotations

from dataclasses import dataclass, fields, replace
from types import SimpleNamespace
import struct
import threading

import pytest
import torch
from torch import nn

from startrain.features import DoubleStarPosition, EncodedBatch, encode_batch
from startrain.inference import GraphInferenceAdapter, InferenceConfig
from startrain.inference_cache import (
    BoundedPredictionCache,
    PinnedTransferPool,
    RawPrediction,
)
from startrain.model import StarModelOutput
from startrain.topology import get_topology


@dataclass
class EncodedRequests:
    features: SimpleNamespace
    tokens: list[int]
    legal_offsets: list[int]
    legal_actions: list[int]
    states: object = None

    def __len__(self) -> int:
        return len(self.tokens)


def encoded_requests(batch: EncodedBatch, *, token_start: int = 1) -> EncodedRequests:
    counts = batch.legal_action_mask.sum(dim=1, dtype=torch.int64)
    return EncodedRequests(
        SimpleNamespace(encoded=batch),
        list(range(token_start, token_start + batch.batch_size)),
        [0, *counts.cumsum(dim=0).tolist()],
        torch.nonzero(batch.legal_action_mask, as_tuple=False)[:, 1].tolist(),
    )


def position(ring: int = 4, *, pda: int = 0) -> DoubleStarPosition:
    return DoubleStarPosition(
        rings=ring,
        stones=torch.full((get_topology(ring).n,), -1, dtype=torch.int8),
        to_move=0,
        moves_left=1,
        opening=True,
        terminal=False,
        pda=pda,
    )


class ObservedNetwork(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.bias = nn.Parameter(torch.tensor(0.1))
        self.rows: list[int] = []
        self.threads: list[str] = []
        self.started: threading.Event | None = None
        self.release: threading.Event | None = None
        self.fail = False

    def forward(self, node_features, global_features, *args, **kwargs):
        self.rows.append(node_features.shape[0])
        self.threads.append(threading.current_thread().name)
        if self.started is not None:
            self.started.set()
        if self.release is not None and not self.release.wait(5):
            raise TimeoutError("test inference release was not signalled")
        if self.fail:
            raise ValueError("test neural failure")
        batch, nodes, _ = node_features.shape
        summary = global_features.sum(dim=1) / 10 + self.bias
        policy = node_features.sum(dim=-1) + self.bias
        outcome = torch.stack((torch.zeros_like(summary), summary), dim=-1)
        margin = torch.full((batch, 303), -8.0, device=node_features.device)
        margin[:, 210] = summary + 8
        return StarModelOutput(
            policy,
            outcome,
            margin,
            torch.zeros(batch, nodes, 3, device=node_features.device),
            torch.zeros(batch, nodes, device=node_features.device),
            policy,
        )


@pytest.fixture
def feature_requests(monkeypatch):
    monkeypatch.setattr(
        "startrain.inference.encode_native_feature_data",
        lambda data, **_: data.encoded,
    )
    return encoded_requests


def cached_adapter(**overrides) -> GraphInferenceAdapter:
    return GraphInferenceAdapter(
        ObservedNetwork(),
        model_version="checkpoint-a",
        model_identity="sha256-a",
        config=InferenceConfig(
            cache_max_entries=16,
            cache_max_bytes=2_000_000,
            deduplicate=True,
            **overrides,
        ),
    )


def test_exact_cache_deduplicates_and_retokens_without_reusing_score_utility(
    feature_requests,
):
    batch = encode_batch([position(), position(), position(pda=2)])
    requests = feature_requests(batch)
    adapter = cached_adapter()
    initial = adapter.evaluate_detailed(requests)
    assert adapter.model.rows == [2]
    second = adapter.evaluate(feature_requests(batch, token_start=30))
    assert second.tokens == [30, 31, 32]
    assert second.policy_logits == initial.response.policy_logits
    assert second.values == initial.response.values
    assert adapter.model.rows == [2]
    adapter.set_score_utility_weight(0.5)
    weighted = adapter.evaluate(requests)
    expected = [
        max(-1.0, min(1.0, value + 0.5 * margin / 151))
        for value, margin in zip(
            initial.outcome_values, initial.score_expectations, strict=True
        )
    ]
    assert weighted.values == pytest.approx(expected, abs=1e-7)
    metrics = adapter.metrics_snapshot()
    assert (metrics.evaluator_calls, metrics.evaluator_rows) == (3, 9)
    assert (metrics.neural_calls, metrics.neural_rows) == (1, 2)
    assert (metrics.cache_hits, metrics.cache_misses, metrics.deduplicated_rows) == (
        6,
        3,
        1,
    )


@pytest.mark.parametrize("field_name", [field.name for field in fields(EncodedBatch)])
def test_cache_key_covers_every_model_input_and_metadata(field_name):
    batch = encode_batch([position()])
    changed = replace(batch, **{field_name: getattr(batch, field_name).clone()})
    tensor = getattr(changed, field_name)
    if tensor.dtype == torch.bool:
        tensor.reshape(-1)[0] = ~tensor.reshape(-1)[0]
    else:
        tensor.reshape(-1)[0] += 1
    namespace = cached_adapter().namespace
    assert GraphInferenceAdapter._row_keys(
        batch, namespace
    ) != GraphInferenceAdapter._row_keys(changed, namespace)
    changed_namespace = ("different-model", *namespace[1:])
    assert GraphInferenceAdapter._row_keys(
        batch, namespace
    ) != GraphInferenceAdapter._row_keys(batch, changed_namespace)


@pytest.mark.parametrize("dtype", (torch.float32, torch.float64, torch.bfloat16))
def test_batched_key_serialization_preserves_the_exact_original_byte_format(dtype):
    batch = encode_batch([position(4), position(10, pda=2)]).to(
        "cpu", feature_dtype=dtype
    )
    namespace = cached_adapter().namespace
    expected = []
    prefix = repr(namespace).encode("utf-8") + b"\0"
    # Frozen original row-at-a-time encoding, independent of the vectorized
    # implementation. Mixed/padded topology, booleans and scalar rings count.
    for row in range(batch.batch_size):
        parts = [prefix]
        for field in fields(batch):
            value = getattr(batch, field.name).detach().contiguous()[row].contiguous()
            header = f"{field.name}:{value.dtype}:{tuple(value.shape)}".encode("ascii")
            payload = value.reshape(-1).view(torch.uint8).numpy().tobytes()
            parts.extend(
                (
                    struct.pack("<I", len(header)),
                    header,
                    struct.pack("<I", len(payload)),
                    payload,
                )
            )
        expected.append(b"".join(parts))
    assert GraphInferenceAdapter._row_keys(batch, namespace) == expected


def test_owned_preparation_and_identity_change_prevent_stale_predictions(
    feature_requests,
):
    batch = encode_batch([position()])
    adapter = cached_adapter()
    prepared = adapter.prepare_requests(feature_requests(batch))
    expected = prepared.encoded.node_features.clone()
    batch.node_features.fill_(99)
    torch.testing.assert_close(prepared.encoded.node_features, expected)
    adapter.evaluate_prepared([prepared])
    adapter.model_identity = "sha256-b"
    with pytest.raises(RuntimeError, match="identity changed"):
        adapter.evaluate_prepared([prepared])
    fresh = adapter.evaluate(feature_requests(encode_batch([position()])))
    assert fresh.tokens == [1]
    assert adapter.model.rows == [1, 1]
    assert adapter.efficiency_snapshot()["cache_entries"] == 1


def test_prepared_keys_recomputed_after_direct_tensor_mutation_or_replacement(
    feature_requests,
):
    adapter = cached_adapter()
    prepared = adapter.prepare_requests(feature_requests(encode_batch([position()])))
    assert prepared.prepared_keys is not None and prepared.key_tensor_stamps is not None
    before = adapter.evaluate_prepared([prepared])[0][0]
    prepared.encoded.node_features.add_(2)
    after = adapter.evaluate_prepared([prepared])[0][0]
    assert after.policy_logits != before.policy_logits
    assert adapter.model.rows == [1, 1]
    replaced = replace(
        prepared,
        encoded=replace(
            prepared.encoded, node_features=prepared.encoded.node_features + 3
        ),
    )
    newest = adapter.evaluate_prepared([replaced])[0][0]
    assert newest.policy_logits != after.policy_logits
    assert adapter.model.rows == [1, 1, 1]


def test_preparation_inside_inference_mode_still_tracks_snapshot_mutations(
    feature_requests,
):
    adapter = cached_adapter()
    request = feature_requests(encode_batch([position()]))
    with torch.inference_mode():
        prepared = adapter.prepare_requests(request)
    assert prepared.key_tensor_stamps is not None
    original_stamps = prepared.key_tensor_stamps
    prepared.encoded.global_features.add_(0.1)
    assert adapter._tensor_stamps(prepared.encoded) != original_stamps
    assert adapter.evaluate_prepared([prepared])[0][0].tokens == [1]


def test_mutated_prepared_legality_is_rejected_before_returning_invalid_csr(
    feature_requests,
):
    adapter = cached_adapter()
    prepared = adapter.prepare_requests(feature_requests(encode_batch([position()])))
    prepared.encoded.legal_action_mask[0, 0] = False
    with pytest.raises(ValueError, match="legal-action metadata changed"):
        adapter.evaluate_prepared([prepared])
    assert adapter.model.rows == []


def test_cache_bounds_charge_keys_payload_and_evict_lru():
    cache = BoundedPredictionCache(max_entries=2, max_bytes=2000)
    prediction = RawPrediction(b"\0" * 80, 1)
    cache.put(b"a", prediction)
    cache.put(b"b", prediction)
    assert cache.get(b"a") is prediction
    cache.put(b"c", prediction)
    assert cache.get(b"b") is None
    assert cache.evictions == 1
    assert len(cache) == 2 and cache.bytes <= cache.max_bytes
    cache.put(b"large" * 1000, prediction)
    assert len(cache) == 2 and cache.bytes <= cache.max_bytes
    cache.clear()
    assert len(cache) == cache.bytes == 0


def test_configuration_and_unversioned_cache_fail_closed(feature_requests):
    for kwargs in (
        {"cache_max_entries": 1},
        {"cache_max_bytes": 1},
        {"deduplicate": 1},
        {"pinned_buffer_slots": 0},
        {"cache_max_entries": -1},
    ):
        with pytest.raises(ValueError):
            InferenceConfig(**kwargs)
    adapter = GraphInferenceAdapter(
        ObservedNetwork(),
        config=InferenceConfig(cache_max_entries=1, cache_max_bytes=100000),
    )
    with pytest.raises(ValueError, match="immutable model"):
        adapter.evaluate(feature_requests(encode_batch([position()])))


@pytest.mark.native
def test_native_variant_requests_share_only_exact_features():
    native = pytest.importorskip("star_native")
    adapter = cached_adapter()
    for mode, handicap, pie in (
        ("double", 1, False),
        ("classic", 1, True),
        ("classic", 4, False),
    ):
        states = native.StateBatch(4, 3, mode=mode, handicap=handicap, pie=pie)
        search = native.SearchBatch(
            states,
            simulations=1,
            max_considered=2,
            pda_by_seat=[(0, 0), (0, 0), (2, -2)],
        )
        roots = search.root_requests()
        response = adapter.evaluate(roots)
        search.initialize_roots(*response.submit_args())
        assert response.tokens == roots.tokens
    assert adapter.model.rows == [2, 2, 2]
    assert adapter.last_feature_path == "rust"
    assert adapter.metrics_snapshot().deduplicated_rows == 3


@pytest.mark.cuda
def test_pinned_transfer_slots_reuse_without_overwriting_inflight_data():
    device = torch.device("cuda:0")
    pool = PinnedTransferPool(slots=1, device=device)
    first = pool.transfer({"values": torch.arange(65536, dtype=torch.float32)})[
        "values"
    ]
    second = pool.transfer({"values": torch.full((32768,), 7.0)})["values"]
    torch.testing.assert_close(first.cpu(), torch.arange(65536, dtype=torch.float32))
    torch.testing.assert_close(second.cpu(), torch.full((32768,), 7.0))
    assert pool.allocations == 1 and pool.reuses == 1
    assert pool.allocated_bytes == 65536 * 4
    pool.close()
    assert pool.allocated_bytes == 0
