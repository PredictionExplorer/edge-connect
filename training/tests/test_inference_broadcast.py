from dataclasses import fields, replace

import pytest
import torch

from startrain.features import EncodedBatch, encode_batch
from startrain.inference import (
    GraphInferenceAdapter,
    InferenceConfig,
    _merge_batch_tensors,
    _select_batch_tensor,
)
from test_inference_efficiency import ObservedNetwork, encoded_requests, position


TOPOLOGY_FIELDS = ("neighbor_index", "neighbor_mask", "neighbor_edge_type")


def broadcast_topology(batch: EncodedBatch) -> EncodedBatch:
    return replace(
        batch,
        **{
            name: getattr(batch, name)[:1].clone().expand_as(getattr(batch, name))
            for name in TOPOLOGY_FIELDS
        },
    )


@pytest.fixture
def feature_requests(monkeypatch):
    monkeypatch.setattr(
        "startrain.inference.encode_native_feature_data", lambda data, **_: data.encoded
    )
    return encoded_requests


class CapturingNetwork(ObservedNetwork):
    def __init__(self):
        super().__init__()
        self.inputs = []
        self.strides = []

    def forward(self, *args, **kwargs):
        self.inputs.append(tuple(tensor.clone() for tensor in args))
        self.strides.append(tuple(tensor.stride(0) for tensor in args))
        return super().forward(*args, **kwargs)


def adapter(*, preserve_broadcast_topology: bool = True) -> GraphInferenceAdapter:
    return GraphInferenceAdapter(
        CapturingNetwork(),
        model_identity="checkpoint-a",
        config=InferenceConfig(
            cache_max_entries=32,
            cache_max_bytes=4_000_000,
            deduplicate=True,
            preserve_broadcast_topology=preserve_broadcast_topology,
        ),
    )


def test_broadcast_preservation_is_opt_in_and_does_not_change_cache_identity(
    feature_requests,
):
    assert InferenceConfig().preserve_broadcast_topology is False
    with pytest.raises(ValueError, match="preserve_broadcast_topology must be boolean"):
        InferenceConfig(preserve_broadcast_topology=1)
    batch = broadcast_topology(encode_batch([position(), position(pda=1)]))
    dense_adapter = adapter(preserve_broadcast_topology=False)
    shared_adapter = adapter()
    dense = dense_adapter.prepare_requests(feature_requests(batch))
    shared = shared_adapter.prepare_requests(feature_requests(batch))
    assert dense.namespace == shared.namespace
    assert dense.prepared_keys == shared.prepared_keys
    for name in TOPOLOGY_FIELDS:
        assert getattr(dense.encoded, name).stride(0) != 0
        assert getattr(shared.encoded, name).stride(0) == 0
    assert dense_adapter.evaluate_prepared([dense], include_details=(True,)) == (
        shared_adapter.evaluate_prepared([shared], include_details=(True,))
    )


@pytest.mark.parametrize("ring", (4, 6, 8, 10))
@pytest.mark.parametrize("dtype", (torch.float32, torch.float64, torch.bfloat16))
def test_broadcast_key_bytes_equal_dense_key_bytes(ring, dtype):
    dense = encode_batch([position(ring, pda=value) for value in (0, 1, -1)]).to(
        "cpu", feature_dtype=dtype
    )
    namespace = adapter().namespace
    assert GraphInferenceAdapter._row_keys(
        broadcast_topology(dense), namespace
    ) == GraphInferenceAdapter._row_keys(dense, namespace)


@pytest.mark.parametrize("field_name", TOPOLOGY_FIELDS)
def test_broadcast_snapshot_owns_storage_and_tracks_mutations(
    field_name, feature_requests
):
    source = broadcast_topology(encode_batch([position(), position(pda=1)]))
    inference = adapter()
    prepared = inference.prepare_requests(feature_requests(source))
    original = getattr(source, field_name)
    owned = getattr(prepared.encoded, field_name)
    assert owned.stride(0) == 0
    assert owned.untyped_storage().nbytes() == owned[0].numel() * owned.element_size()
    expected = owned.clone()
    if original.dtype == torch.bool:
        original[0, 0, 0] = ~original[0, 0, 0]
    else:
        original[0, 0, 0] += 1
    torch.testing.assert_close(owned, expected)

    inference.evaluate_prepared([prepared])
    before_stamps = inference._tensor_stamps(prepared.encoded)
    if owned.dtype == torch.bool:
        owned[0, 0, 0] = ~owned[0, 0, 0]
    else:
        owned[0, 0, 0] += 1
    assert inference._tensor_stamps(prepared.encoded) != before_stamps
    inference.evaluate_prepared([prepared])
    assert inference.model.rows == [2, 2]
    assert inference.efficiency_snapshot()["cache_entries"] == 4


def test_broadcast_merge_and_padding_preserve_every_model_input(
    feature_requests, monkeypatch
):
    dense = adapter()
    shared = adapter()
    for inference in (dense, shared):
        monkeypatch.setattr(
            inference,
            "_inference_batch_rows",
            lambda rows: 1 << (rows - 1).bit_length(),
        )
    batches = [
        encode_batch([position(10, pda=value) for value in values])
        for values in ((-3, -2, -1), (0, 1, 2))
    ]
    expected = dense.evaluate_prepared(
        [
            dense.prepare_requests(feature_requests(batch, token_start=10 * index))
            for index, batch in enumerate(batches)
        ],
        include_details=(True, True),
    )
    prepared = [
        shared.prepare_requests(
            feature_requests(broadcast_topology(batch), token_start=10 * index)
        )
        for index, batch in enumerate(batches)
    ]
    actual = shared.evaluate_prepared(prepared, include_details=(True, True))
    assert actual == expected
    for left, right in zip(dense.model.inputs[0], shared.model.inputs[0], strict=True):
        torch.testing.assert_close(left, right, rtol=0, atol=0)
    assert shared.model.rows == [8]
    assert shared.model.strides[0][2:5] == (0, 0, 0)

    def unexpected_merge(*args, **kwargs):
        raise AssertionError("cache hits must not merge model inputs")

    monkeypatch.setattr("startrain.inference._merge_batch_tensors", unexpected_merge)
    assert shared.evaluate_prepared(prepared, include_details=(True, True)) == expected
    assert shared.model.rows == [8]


@pytest.mark.parametrize("dtype", (torch.bool, torch.int64, torch.float32))
def test_nonidentical_topology_rows_keep_exact_merge_and_selection(dtype):
    first = torch.zeros((1, 3, 2), dtype=dtype).expand(2, -1, -1)
    second = torch.ones((1, 3, 2), dtype=dtype).expand(3, -1, -1)
    merged = _merge_batch_tensors((first, second))
    expected = torch.cat((first, second))
    torch.testing.assert_close(merged, expected, rtol=0, atol=0)
    indices = torch.tensor([4, 0, 3, 2, 0])
    torch.testing.assert_close(
        _select_batch_tensor(merged, indices), expected.index_select(0, indices)
    )


def test_merge_does_not_confuse_signed_zero_or_broadcast_shape():
    positive = torch.tensor([[0.0, 1.0]]).expand(2, -1)
    negative = torch.tensor([[-0.0, 1.0]]).expand(3, -1)
    merged = _merge_batch_tensors((positive, negative))
    assert torch.signbit(merged[:, 0]).tolist() == [False, False, True, True, True]
    assert merged.stride(0) != 0
    equal = _merge_batch_tensors((positive, positive[:1]))
    assert equal.shape == (3, 2) and equal.stride(0) == 0


@pytest.mark.native
@pytest.mark.parametrize("ring", (4, 6, 8, 10))
@pytest.mark.parametrize(
    "mode,handicap,pie",
    (
        ("classic", 1, False),
        ("double", 1, False),
        ("classic", 1, True),
        ("double", 1, True),
        ("classic", 9, False),
        ("double", 9, False),
    ),
)
def test_native_broadcast_features_match_dense_inference(ring, mode, handicap, pie):
    native = pytest.importorskip("star_native")
    states = native.StateBatch(ring, 3, mode=mode, handicap=handicap, pie=pie)
    search = native.SearchBatch(
        states,
        simulations=1,
        max_considered=2,
        pda_by_seat=[(0, 0), (1, -1), (2, -2)],
    )
    inference = adapter()
    prepared = inference.prepare_requests(search.root_requests())
    dense = replace(
        prepared,
        encoded=EncodedBatch(
            **{
                field.name: getattr(prepared.encoded, field.name).clone()
                for field in fields(EncodedBatch)
            }
        ),
        prepared_keys=None,
        key_tensor_stamps=None,
    )
    assert inference.evaluate_prepared([prepared], include_details=(True,)) == (
        adapter().evaluate_prepared([dense], include_details=(True,))
    )
