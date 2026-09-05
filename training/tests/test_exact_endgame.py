from dataclasses import replace

import pytest

from startrain.replay_store import ReplayStore
from startrain.runtime import RunIdentity
from startrain.selfplay import (
    GameVariant,
    SelfPlayActor,
    SelfPlayConfig,
    SelfPlayIdentity,
    VariantMixtureConfig,
)
from test_variant_selfplay import evaluator


@pytest.mark.native
@pytest.mark.parametrize(
    "mode,handicap,pie",
    [
        ("classic", 1, False),
        ("double", 1, False),
        ("classic", 4, False),
        ("double", 4, False),
        ("classic", 1, True),
        ("double", 1, True),
    ],
)
def test_exact_tail_completion_preserves_variant_pda_and_real_score_targets(
    tmp_path, mode, handicap, pie
):
    native = pytest.importorskip("star_native")
    config = replace(
        SelfPlayConfig.cpu_smoke(seed=31),
        games=1,
        batch_size=1,
        exact_endgame_max_empty=4,
        exact_endgame_max_nodes=1000,
        variants=VariantMixtureConfig(enabled=True),
    ).with_variant(GameVariant(mode=mode, handicap=handicap, pie=pie))
    identity = RunIdentity(tmp_path / "run.json", "run-exact", "family-exact", 1)
    with ReplayStore(tmp_path / "replay") as store:
        generation = store.lease_generation(identity, "actor-exact")
        actor = SelfPlayActor(
            native,
            evaluator(),
            store,
            config,
            SelfPlayIdentity(
                identity.run_id, identity.generation_family, "actor-exact", generation
            ),
        )
        summaries = actor.run()
        assert len(summaries) == 1 and summaries[0].finish_reason == "exact-endgame"
        assert summaries[0].empty_nodes_saved == 4
        assert summaries[0].variant == config.variant.label
        metrics = actor.metrics_snapshot()
        assert (
            metrics.exact_endgame_solved == 1 and metrics.exact_endgame_exhausted == 0
        )
        assert metrics.exact_endgame_nodes <= 1000
        samples = store.load_recent_samples(
            sample_window=4096,
            run_id=identity.run_id,
            generation_family=identity.generation_family,
            current_model_step=0,
            max_model_lag_steps=0,
        )
        assert samples and all(
            "final=exact-endgame:" in sample.search_provenance for sample in samples
        )
        if handicap > 1:
            assert {sample.pda for sample in samples} == {-2, 2}


@pytest.mark.parametrize(
    "values",
    [
        {"exact_endgame_max_empty": 9},
        {"exact_endgame_max_empty": True},
        {"exact_endgame_max_nodes": 0},
        {"exact_endgame_max_nodes": 1000001},
    ],
)
def test_solver_limits_reject_unbounded_work(values):
    with pytest.raises(ValueError, match="exact_endgame"):
        SelfPlayConfig(**values)
