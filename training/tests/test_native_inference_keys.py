"""Exact native cache keys, lazy feature preparation, and mutation boundaries."""

from dataclasses import replace
from types import SimpleNamespace

import pytest
import torch

from startrain.inference import GraphInferenceAdapter, InferenceConfig
from startrain.inference_cache import BoundedPredictionCache, RawPrediction
from startrain.native import encode_native_feature_data
from test_inference_efficiency import ObservedNetwork


@pytest.fixture
def native():
    module = pytest.importorskip("star_native")
    assert module.native_inference_key_version() == 1, "rebuild native inference keys"
    return module


def adapter(**options):
    return GraphInferenceAdapter(
        ObservedNetwork(), model_identity="immutable-a",
        config=InferenceConfig(
            cache_max_entries=options.pop("cache_max_entries", 16),
            cache_max_bytes=2_000_000, deduplicate=True, **options,
        ),
    )


def roots(native, pda=(0,), rings=4):
    states = native.StateBatch(rings, len(pda))
    search = native.SearchBatch(states, simulations=1, pda_by_seat=[(x, -x) for x in pda])
    return states, search.root_requests()


def general(request):
    return SimpleNamespace(
        features=request.features, tokens=request.tokens, states=request.states,
        legal_offsets=request.legal_offsets, legal_actions=request.legal_actions,
        # A proxy cannot obtain trust merely by advertising the native methods.
        inference_keys=request.inference_keys, selected_features=request.selected_features,
    )


class GeneralRequest:
    def __init__(self, request):
        self.__dict__.update(vars(general(request)))

    def __len__(self):
        return len(self.tokens)


def test_compact_keys_deduplicate_before_scoring_and_warm_hits_need_no_features(native):
    _, request = roots(native, (0, 0, 2), rings=10)
    keys = request.inference_keys()
    assert keys[0] == keys[1] and keys[0] != keys[2]
    assert all(isinstance(key, bytes) and len(key) < 512 for key in keys)
    assert request.encoded_feature_rows == 0
    inference = adapter()
    prepared = inference.prepare_requests(request)
    assert prepared.native_lazy and request.encoded_feature_rows == 2
    assert prepared.ring == 10 and prepared.max_nodes == 275
    first = inference.evaluate_prepared([prepared])[0][0]
    assert request.encoded_feature_rows == 2
    assert inference.model.rows == [2]
    _, warm = roots(native, (0, 0, 2), rings=10)
    second = inference.evaluate(warm)
    assert warm.encoded_feature_rows == 0
    assert inference.model.rows == [2]
    assert first.values == second.values and first.policy_logits == second.policy_logits
    assert second.tokens == warm.tokens


@pytest.mark.parametrize("rings", [4, 6, 8, 10])
def test_native_and_general_inputs_match_predictions_and_keep_separate_trust(native, rings):
    _, request = roots(native, (0, 1, -2), rings)
    fast = adapter()
    slow = adapter()
    fast_prepared = fast.prepare_requests(request)
    general_prepared = slow.prepare_requests(GeneralRequest(request))
    assert fast_prepared.native_lazy
    assert general_prepared.native_snapshot is None
    assert max(map(len, fast_prepared.prepared_keys)) < 1024
    assert min(map(len, general_prepared.prepared_keys)) > 10_000
    assert fast.evaluate_prepared([fast_prepared], include_details=[True]) == (
        slow.evaluate_prepared([general_prepared], include_details=[True])
    )


def test_partial_hits_and_eviction_after_producer_peek_are_safe(native):
    inference = adapter(cache_max_entries=1)
    _, initial = roots(native)
    expected = inference.evaluate(initial)
    _, warm = roots(native)
    prepared = inference.prepare_requests(warm)
    assert warm.encoded_feature_rows == 0
    _, replacement = roots(native, (1,))
    inference.evaluate(replacement)
    restored = inference.evaluate_prepared([prepared])[0][0]
    assert warm.encoded_feature_rows == 1
    assert restored.values == expected.values and restored.policy_logits == expected.policy_logits
    inference = adapter()
    inference.evaluate(initial)
    _, mixed = roots(native, (0, 1, 1))
    prepared = inference.prepare_requests(mixed)
    assert mixed.encoded_feature_rows == 1
    inference.evaluate_prepared([prepared])
    assert mixed.encoded_feature_rows == 1
    assert inference.model.rows == [1, 1]


def test_native_snapshot_survives_state_and_export_mutations(native):
    states, request = roots(native)
    keys = request.inference_keys()
    inference = adapter()
    prepared = inference.prepare_requests(request)
    states.apply_many([0], [0])
    exported = request.features.node_features
    exported[:] = b"\xff" * len(exported)
    actions = request.legal_actions
    actions.reverse()
    assert request.inference_keys() == keys
    actual = inference.evaluate_prepared([prepared])[0][0]
    _, original = roots(native)
    expected = adapter().evaluate(original)
    assert actual.values == expected.values and actual.policy_logits == expected.policy_logits


def test_lazy_rows_match_existing_full_native_encoder_and_invalid_selection_is_atomic(native):
    for rings in [4, 10]:
        for mode, handicap, pie in [("classic", 1, False), ("double", 4, False), ("double", 1, True)]:
            states = native.StateBatch(rings, 2, mode=mode, handicap=handicap, pie=pie)
            states.apply_many([0, 1], [0, 1])
            request = native.SearchBatch(states, simulations=1, pda_by_seat=[(2, -2), (-1, 1)]).root_requests()
            assert request.encoded_feature_rows == 0
            with pytest.raises(ValueError, match="out of range"):
                request.prefetch_features([0, 2])
            assert request.encoded_feature_rows == 0
            expected = request.states.feature_data(pda=request.pda)
            actual = request.selected_features([0, 1])
            for name in ["rings", "node_features", "global_features", "node_mask", "legal_action_mask", "score_components", "node_owner", "alive_stones"]:
                assert bytes(getattr(actual, name)) == bytes(getattr(expected, name))
            assert request.encoded_feature_rows == 2
            request.prefetch_features([0, 1, 0])
            assert request.encoded_feature_rows == 2


def test_explicit_prepared_tensor_mutation_switches_to_full_input_validation(native):
    _, request = roots(native)
    inference = adapter()
    prepared = inference.prepare_requests(request)
    before = inference.evaluate_prepared([prepared])[0][0]
    prepared.encoded.node_features.add_(2)
    assert not prepared.native_lazy
    after = inference.evaluate_prepared([prepared])[0][0]
    assert after.policy_logits != before.policy_logits
    changed = replace(prepared, encoded=replace(prepared.encoded, global_features=prepared.encoded.global_features + 1))
    assert inference.evaluate_prepared([changed])[0][0].values != after.values
    prepared.encoded.legal_action_mask[0, 0] = False
    with pytest.raises(ValueError, match="legal-action metadata"):
        inference.evaluate_prepared([prepared])
    _, unchanged = roots(native)
    assert inference.evaluate(unchanged).policy_logits == before.policy_logits


def test_each_history_plane_and_pda_are_present_without_a_lossy_hash(native):
    words = len(native.StateBatch(4, 1).data().zero_bits)
    def bits(nodes):
        return [sum(1 << x for x in nodes), *([0] * (words - 1))]
    base = dict(
        rings=4, zero_bits=bits([0, 2, 4]), one_bits=bits([1, 3]),
        to_move=[0], moves_left=[1], opening=[False], mode=[1], handicap=[1],
        pie=[False], swap_available=[False], swapped=[False],
        current_turn_bits=bits([4]), previous_turn_bits=bits([1]),
        own_previous_turn_bits=bits([0]), handicap_bits=bits([0]),
    )
    requests = []
    for updates in [{}, {"current_turn_bits": bits([2])}, {"previous_turn_bits": bits([3])},
                    {"own_previous_turn_bits": bits([2])}, {"handicap_bits": bits([2])}]:
        state = native.StateBatch.from_semantic(**(base | updates))
        requests.append(native.SearchBatch(state, simulations=1).root_requests())
    assert len({r.inference_keys()[0] for r in requests}) == 5
    assert len({(tuple(r.states.zero_bits), tuple(r.states.one_bits)) for r in requests}) == 1
    assert all(len(r.inference_keys()[0]) > 6 * words * 8 for r in requests)
    flipped = native.StateBatch.from_semantic(**(base | {
        "zero_bits": base["one_bits"], "one_bits": base["zero_bits"], "to_move": [1],
    }))
    flipped_request = native.SearchBatch(flipped, simulations=1).root_requests()
    assert flipped_request.inference_keys() == requests[0].inference_keys()
    left = encode_native_feature_data(requests[0].features)
    right = encode_native_feature_data(flipped_request.features)
    for a, b in zip(left.model_args(), right.model_args(), strict=True):
        torch.testing.assert_close(a, b, rtol=0, atol=0)


def test_cache_peek_is_nonmutating_and_small_graph_buckets_are_opt_in():
    cache = BoundedPredictionCache(max_entries=2, max_bytes=4096)
    value = RawPrediction(b"\0" * 16, 1)
    cache.put(b"a", value)
    cache.put(b"b", value)
    assert cache.peek_many([b"a", b"missing"]) == (value, None)
    cache.put(b"c", value)
    assert cache.get(b"a") is None
    default = adapter(cuda_graphs=True)
    small = adapter(cuda_graphs=True, small_batch_graph_buckets=True)
    # Exercise shape arithmetic only; no CUDA allocation or inference.
    default.device = small.device = torch.device("cuda")
    assert default._inference_batch_rows(33) == 64
    assert small._inference_batch_rows(33) == 48
    assert {small._inference_batch_rows(n) for n in range(1, 257)} == {
        1, 2, 3, 4, 6, 8, 12, 16, 24, 32, 48, 64, 96, 128, 192, 256,
    }


def test_runtime_gather_signature_invalidates_prediction_namespace():
    class Network(ObservedNetwork):
        def set_compact_inference_gather(self, enabled):
            self.enabled = enabled

        @property
        def inference_execution_signature(self):
            return "graph-inference-v1", self.enabled

    model = Network()
    inference = GraphInferenceAdapter(model, config=InferenceConfig(compact_inference_gather=True))
    assert model.enabled and inference.namespace[-1] == ("graph-inference-v1", True)
    model.set_compact_inference_gather(False)
    assert len(inference.namespace) == 6
