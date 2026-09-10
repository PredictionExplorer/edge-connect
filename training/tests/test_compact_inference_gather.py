from copy import deepcopy

import pytest
import torch

import startrain.model as model_module
from startrain.features import encode_batch
from startrain.model import GraphResTNet, ModelConfig
from test_model import position, randomize_v3_parameters


@pytest.mark.parametrize("rings", [4, 6, 8, 10])
@pytest.mark.parametrize("autocast", [False, True])
def test_compact_gather_preserves_all_heads_and_weights(rings, autocast, monkeypatch):
    torch.manual_seed(47)
    baseline = GraphResTNet(
        ModelConfig(width=16, rrt_groups=2, attention_heads=4, kv_heads=1)
    ).eval()
    randomize_v3_parameters(baseline)
    compact = deepcopy(baseline)
    before = {name: value.clone() for name, value in compact.state_dict().items()}
    compact.set_compact_inference_gather(True)
    batch = encode_batch([position(rings), position(rings)])
    gathered_dtypes = []
    gather = model_module._gather_neighbors

    def observe(inputs, indices):
        gathered_dtypes.append(inputs.dtype)
        return gather(inputs, indices)

    with (
        torch.inference_mode(),
        torch.autocast("cpu", dtype=torch.bfloat16, enabled=autocast),
    ):
        expected = baseline(*batch.model_args())
        monkeypatch.setattr(model_module, "_gather_neighbors", observe)
        actual = compact(*batch.model_args())
    for field in expected._fields:
        torch.testing.assert_close(
            getattr(actual, field), getattr(expected, field), rtol=0, atol=0
        )
    assert set(gathered_dtypes) == {torch.bfloat16 if autocast else torch.float32}
    assert compact.inference_execution_signature == ("graph-inference-v1", True)
    assert baseline.inference_execution_signature == ("graph-inference-v1", False)
    for name, value in compact.state_dict().items():
        assert torch.equal(value, before[name])


def test_compact_gather_cannot_silently_change_training():
    model = GraphResTNet(
        ModelConfig(width=16, rrt_groups=1, attention_heads=4, kv_heads=1)
    )
    with pytest.raises(ValueError, match="eval model"):
        model.set_compact_inference_gather(True)
    model.eval()
    with pytest.raises(ValueError, match="boolean"):
        model.set_compact_inference_gather(1)
    model.set_compact_inference_gather(True)
    batch = encode_batch([position(4)])
    with pytest.raises(ValueError, match="requires inference"):
        model(*batch.model_args())
    model.set_compact_inference_gather(False)
    model.train()
    loss = model(*batch.model_args()).outcome_logits.square().mean()
    loss.backward()
    assert any(parameter.grad is not None for parameter in model.parameters())


def test_compact_gather_compiles_with_bf16_and_matches_eager():
    model = GraphResTNet(
        ModelConfig(width=16, rrt_groups=1, attention_heads=4, kv_heads=1)
    ).eval()
    model.set_compact_inference_gather(True)
    batch = encode_batch([position(4)])
    compiled = torch.compile(model, backend="aot_eager", fullgraph=True)
    with torch.inference_mode(), torch.autocast("cpu", dtype=torch.bfloat16):
        expected = model(*batch.model_args())
        actual = compiled(*batch.model_args())
    for field in expected._fields:
        torch.testing.assert_close(
            getattr(actual, field), getattr(expected, field), rtol=0, atol=0
        )
