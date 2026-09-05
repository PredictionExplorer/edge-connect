from dataclasses import asdict
import hashlib

import torch

from scripts.run_lineage_arena import load_candidate
from startrain.checkpoint import ExponentialMovingAverage, save_checkpoint
from startrain.config import GameConfig
from startrain.model import GraphResTNet, ModelConfig


def test_diagnostic_defaults_override_large_production_continuation_budget(
    tmp_path, monkeypatch
):
    from pathlib import Path
    from scripts import compare_checkpoint_averaging as diagnostic

    def measured(*args, config, **kwargs):
        assert (
            config.pairs_per_ring
            == config.minimum_pairs_per_ring
            == config.max_pairs_per_ring
            == 4
        )
        assert config.continuation_pairs_per_ring is None
        return {"candidate": "raw", "baseline": "ema"}

    monkeypatch.setattr(diagnostic, "compare", measured)
    assert (
        diagnostic.main(
            [
                "--checkpoint",
                str(tmp_path / "unused.pt"),
                "--profile",
                str(
                    Path(__file__).parents[1] / "configs/h100-8gpu-variant-stage-a.yaml"
                ),
                "--output",
                str(tmp_path / "comparison.json"),
            ]
        )
        == 0
    )


def test_raw_and_ema_diagnostic_load_distinct_weights_without_mutating_checkpoint(
    tmp_path,
):
    config = ModelConfig(width=8, rrt_groups=1, attention_heads=2, kv_heads=1)
    model = GraphResTNet(config)
    ema = ExponentialMovingAverage(model)
    original = next(model.parameters()).detach().clone()
    with torch.no_grad():
        next(model.parameters()).add_(1)
    path = tmp_path / "checkpoint.pt"
    save_checkpoint(
        path,
        model=model,
        step=10,
        ema=ema,
        config={"model": asdict(config), "game": asdict(GameConfig())},
    )
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    raw, raw_meta = load_candidate(path, device=torch.device("cpu"), weights="raw")
    averaged, ema_meta = load_candidate(path, device=torch.device("cpu"), weights="ema")
    torch.testing.assert_close(next(raw.model.parameters()), original + 1)
    torch.testing.assert_close(next(averaged.model.parameters()), original)
    assert raw_meta["identity"] != ema_meta["identity"]
    assert raw_meta["checkpoint_sha256"] == ema_meta["checkpoint_sha256"] == digest
    assert raw_meta["weights"] == "raw" and ema_meta["weights"] == "ema"
    assert hashlib.sha256(path.read_bytes()).hexdigest() == digest
