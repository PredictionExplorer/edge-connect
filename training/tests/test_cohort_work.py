from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace
import json
from pathlib import Path
from types import SimpleNamespace
import threading

import pytest

from startrain import actor as actors
from startrain.actor import (
    ActorSupervisor,
    SharedModelRegistry,
    resolve_actor_experiment,
)
from startrain.cohort_work import (
    CompatibleWorkCoordinator,
    WeightedFairChoice,
    WorkBundle,
)
from startrain.config import ActorPipelineConfig, GPUWorkerConfig, load_config
from startrain.runtime import RunIdentity
from startrain.selfplay import SelfPlayMetrics, VariantMixtureConfig


def metadata(role="candidate", ring=10, mode="double-standard", games=128):
    return {
        "requested_model_role": role,
        "model_role": role,
        "ring": ring,
        "mode_category": mode,
        "games": games,
        "model_identity": "immutable",
    }


@pytest.mark.parametrize("producers", [2, 3, 4])
def test_joint_weighted_schedule_preserves_role_ring_and_six_mode_marginals(producers):
    coordinator = CompatibleWorkCoordinator(cohort_count=producers, seed=17)
    roles = {"candidate": 0.5, "champion": 0.25, "history": 0.25}
    rings = {4: 0.05, 6: 0.05, 8: 0.05, 10: 0.85}
    modes = [
        "classic-standard",
        "double-standard",
        "classic-pie",
        "double-pie",
        "classic-handicap",
        "double-handicap",
    ]
    weights = {
        (role, ring, mode): rw * bw / 6
        for role, rw in roles.items()
        for ring, bw in rings.items()
        for mode in modes
    }

    def factory(owner):
        role, ring = owner.choice.choose(
            {
                (role, ring): rw * bw
                for role, rw in roles.items()
                for ring, bw in rings.items()
            }
        )
        return WorkBundle(
            metadata(role, ring),
            lambda: object(),
            lambda: None,
            tuple(
                {
                    "mode_category": owner.choose_mode(
                        (role, ring), dict.fromkeys(modes, 1.0)
                    )
                }
                for _ in range(producers)
            ),
        )

    counts = Counter()
    for _ in range(480 * producers):
        lease = coordinator.acquire(factory)
        counts[
            (
                lease.metadata["requested_model_role"],
                lease.metadata["ring"],
                lease.metadata["mode_category"],
            )
        ] += 1
        coordinator.record_outcome(
            lease,
            requested=128,
            started=128,
            completed=128,
            dropped=0,
            cancelling=False,
        )
    assert counts == {
        key: round(value * 480 * producers) for key, value in weights.items()
    }
    assert coordinator.metrics_snapshot()["outstanding_promised_games"] == 0
    coordinator.close()


def test_weight_changes_drop_obsolete_deficits_and_zero_weight_support():
    chooser = WeightedFairChoice(4)
    for _ in range(7):
        chooser.choose({"old": 0.8, "other": 0.2})
    assert {chooser.choose({"old": 0, "new": 1}) for _ in range(10)} == {"new"}
    for invalid in ({}, {"a": 0}, {"a": float("nan")}, {"a": -1}):
        with pytest.raises(ValueError):
            chooser.choose(invalid)


def test_bundle_acquires_each_pin_before_releasing_reservation_and_never_barriers():
    events = []
    coordinator = CompatibleWorkCoordinator(cohort_count=4, seed=17)

    def factory(_):
        events.append("reserve")
        return WorkBundle(
            metadata(),
            lambda: events.append("pin") or object(),
            lambda: events.append("unreserve"),
        )

    # A single fast producer can consume a full bundle; no missing producer
    # causes a barrier or timeout, and every lease remains separately owned.
    leases = [coordinator.acquire(factory) for _ in range(4)]
    assert events == ["reserve", "pin", "pin", "pin", "pin", "unreserve"]
    assert len({id(x.resource) for x in leases}) == 4
    assert {x.bundle_id for x in leases} == {1}
    coordinator.acquire(factory)
    coordinator.close()
    assert events[-1] == "unreserve"
    with pytest.raises(RuntimeError, match="closed"):
        coordinator.acquire(factory)


def test_concurrent_claims_have_unique_bounded_bundle_slots():
    coordinator = CompatibleWorkCoordinator(cohort_count=3, seed=17)

    def factory(_):
        return WorkBundle(metadata(), lambda: object(), lambda: None)

    with ThreadPoolExecutor(max_workers=4) as pool:
        leases = list(pool.map(lambda _: coordinator.acquire(factory), range(48)))
    assert len({(x.bundle_id, x.index) for x in leases}) == 48
    assert Counter(x.bundle_id for x in leases) == dict.fromkeys(range(1, 17), 3)
    coordinator.close()


def test_invalid_per_lease_count_releases_bundle_reservation():
    coordinator = CompatibleWorkCoordinator(cohort_count=3, seed=17)
    released = []
    with pytest.raises(ValueError, match="metadata count"):
        coordinator.acquire(
            lambda _: WorkBundle(
                metadata(),
                lambda: pytest.fail("must not acquire"),
                lambda: released.append(True),
                ({"mode_category": "classic-standard"},),
            )
        )
    assert released == [True]
    coordinator.close()


def test_early_drain_retains_exact_promised_quota_and_cancellation_is_explicit():
    coordinator = CompatibleWorkCoordinator(cohort_count=2, seed=17)
    lease = coordinator.acquire(
        lambda _: WorkBundle(metadata(games=128), lambda: object(), lambda: None)
    )
    assert (
        coordinator.record_outcome(
            lease, requested=128, started=96, completed=96, dropped=0, cancelling=False
        )
        == 32
    )
    assert (
        coordinator.record_outcome(
            lease, requested=32, started=32, completed=32, dropped=0, cancelling=False
        )
        == 0
    )
    other = coordinator.acquire(lambda _: pytest.fail("same bundle"))
    assert (
        coordinator.record_outcome(
            other, requested=128, started=64, completed=32, dropped=32, cancelling=True
        )
        == 0
    )
    snapshot = coordinator.metrics_snapshot()
    assert snapshot["planned_games_by_requested_role"] == {"candidate": 256}
    assert snapshot["completed_games_by_requested_role"] == {"candidate": 160}
    assert snapshot["started_games"] == 192 and snapshot["dropped_games"] == 32
    assert snapshot["cancelled_unstarted_games"] == 64
    assert snapshot["outstanding_promised_games"] == 0
    with pytest.raises(ValueError, match="outstanding"):
        coordinator.record_outcome(
            lease, requested=32, started=32, completed=32, dropped=0, cancelling=False
        )
    coordinator.close()


def config():
    base = load_config(Path(__file__).parents[1] / "configs/small.yaml")
    return replace(
        base,
        orchestration=replace(
            base.orchestration,
            model_refresh=replace(
                base.orchestration.model_refresh,
                selfplay_source="candidate",
                inference=replace(
                    base.orchestration.model_refresh.inference, shared_batching=True
                ),
            ),
        ),
    )


def test_per_gpu_resolver_changes_only_opt_in_pipeline_and_quota(tmp_path):
    base = config()
    pipeline = ActorPipelineConfig(
        compatible_work=True,
        stream_completed_games=True,
        rolling_game_slots=True,
        seed_contract="game-v1",
        games_per_task=128,
        cuda_graphs=True,
    )
    gpu = GPUWorkerConfig(
        gpu_id=7,
        role="actor",
        cpu_threads=4,
        actor_batch_size=64,
        actor_cohorts=3,
        actor_pipeline=pipeline,
    )
    changed = resolve_actor_experiment(base, gpu)
    assert (
        changed.model == base.model
        and changed.game == base.game
        and changed.optimizer == base.optimizer
    )
    assert changed.selfplay.full_simulations == base.selfplay.full_simulations
    assert changed.selfplay.variants == base.selfplay.variants
    assert changed.orchestration.model_refresh.compatible_cohort_work
    assert changed.orchestration.model_refresh.inference.cuda_graphs
    assert not base.orchestration.model_refresh.compatible_cohort_work
    assert not base.selfplay.rolling_game_slots
    assert resolve_actor_experiment(base, replace(gpu, actor_pipeline=None)) is base
    actor = ActorSupervisor(
        native_module=object(),
        experiment=base,
        gpu=gpu,
        replay_directory=tmp_path / "replay",
        manifest_path=tmp_path / "champion",
        candidate_manifest_path=tmp_path / "candidate",
        run_identity=RunIdentity(tmp_path / "run", "run", "family", 1),
        heartbeat_path=tmp_path / "hb",
        metrics_path=tmp_path / "metrics",
        device="cpu",
    )
    assert actor.games_per_batch == 128


def fake_actor(tmp_path, monkeypatch):
    base = config()
    identity = RunIdentity(tmp_path / "run", "run", "family", 1)
    manifests = {}
    for name, step in (("champion", 1), ("candidate", 10), ("new", 11)):
        path = tmp_path / name
        direct = tmp_path / (name + "-immutable")
        path.write_text("pointer")
        direct.write_text("manifest")
        manifests[path] = SimpleNamespace(
            model_identity=name,
            model_step=step,
            run_id="run",
            generation_family="family",
            path=path,
            artifact_manifest=direct,
            role="candidate" if name != "champion" else "champion",
        )
        manifests[direct] = SimpleNamespace(
            **(
                vars(manifests[path])
                | {"path": direct, "artifact_manifest": None, "role": "direct"}
            )
        )
    monkeypatch.setattr(
        actors, "load_model_manifest", lambda path: manifests[Path(path)]
    )
    loaded = []

    class Provider:
        def __init__(
            self, experiment, path, *, registry=None, pinned_manifest=None, **kwargs
        ):
            self.config = experiment
            self.manifest_path = Path(path)
            self.registry = registry
            self.pinned = pinned_manifest
            self.device = "cpu"
            self.run_identity = identity
            self.evaluator = None
            self.manifest = None

        def refresh(self):
            manifest = self.pinned or manifests[self.manifest_path]
            self.manifest = manifest
            if self.registry is not None:
                self.evaluator = self.registry.acquire(self, manifest)
            else:
                loaded.append(manifest.model_identity)
                self.evaluator = SimpleNamespace(
                    model_identity=manifest.model_identity,
                    model_version=manifest.model_identity,
                    model_step=manifest.model_step,
                    close=lambda: None,
                )
            return self.evaluator

        def wait_for_initial(self, **_kwargs):
            return self.refresh()

        def release(self):
            if self.registry is not None and self.evaluator is not None:
                self.registry.release(self.manifest.model_identity)
                self.evaluator = None

    monkeypatch.setattr(actors, "ManifestModelProvider", Provider)
    broker = SimpleNamespace(
        cohort_adapter=lambda base: SimpleNamespace(
            model_identity=base.model_identity,
            model_step=base.model_step,
            model_version=base.model_identity,
            evaluator_rows=0,
            evaluator_calls=0,
        ),
        device_lock=threading.RLock(),
    )
    registry = SharedModelRegistry(broker, max_entries=4)
    coordinator = CompatibleWorkCoordinator(cohort_count=2, seed=17)
    actor = ActorSupervisor(
        native_module=object(),
        experiment=base,
        gpu=GPUWorkerConfig(gpu_id=1, role="actor", cpu_threads=4, actor_batch_size=1),
        registry=registry,
        work_coordinator=coordinator,
        replay_directory=tmp_path / "replay",
        manifest_path=tmp_path / "champion",
        candidate_manifest_path=tmp_path / "candidate",
        run_identity=identity,
        heartbeat_path=tmp_path / "hb",
        metrics_path=tmp_path / "metrics",
        device="cpu",
    )
    monkeypatch.setattr(
        actor, "_read_candidate", lambda: manifests[tmp_path / "candidate"]
    )
    monkeypatch.setattr(
        actor, "_read_learner_scheduling_step", lambda **kwargs: (10, "test")
    )
    store = SimpleNamespace(
        sample_counts_by_ring=lambda *args, **kwargs: dict.fromkeys((4, 6, 8, 10), 100)
    )
    return actor, store, manifests, loaded


def test_bundle_pins_immutable_weights_across_pointer_refresh_and_continuation_refreshes(
    tmp_path, monkeypatch
):
    actor, store, manifests, loaded = fake_actor(tmp_path, monkeypatch)
    first = actor._acquire_cohort_work(store)
    manifests[tmp_path / "candidate"] = manifests[tmp_path / "new"]
    second = actor._acquire_cohort_work(store)
    assert (
        first.resource[1].model_identity
        == second.resource[1].model_identity
        == "candidate"
    )
    assert (
        first.metadata["ring"] == second.metadata["ring"]
        and first.metadata["variant"] == second.metadata["variant"]
    )
    first.resource[0].release()
    second.resource[0].release()
    actor._continuation = (replace(first, resource=None), 1)
    continued = actor._acquire_cohort_work(store)
    assert continued.resource[1].model_identity == "new"
    assert continued.metadata["variant"] == first.metadata["variant"]
    assert continued.metadata["games"] == 1 and continued.bundle_id == first.bundle_id
    continued.resource[0].release()
    actor.work_coordinator.close()
    actor.registry.close()


def test_actor_bundle_shares_model_and_board_while_varying_modes_per_lease(
    tmp_path, monkeypatch
):
    actor, store, _, _ = fake_actor(tmp_path, monkeypatch)
    actor.work_coordinator.close()
    actor.work_coordinator = CompatibleWorkCoordinator(cohort_count=4, seed=17)
    actor.experiment = replace(
        actor.experiment,
        selfplay=replace(
            actor.experiment.selfplay,
            variants=VariantMixtureConfig(
                enabled=True,
                standard=1 / 6,
                classic=1 / 6,
                pie=1 / 3,
                handicap=1 / 3,
                pie_classic_share=0.5,
                handicap_classic_share=0.5,
            ),
        ),
    )
    leases = [actor._acquire_cohort_work(store) for _ in range(4)]
    assert len({lease.resource[1].model_identity for lease in leases}) == 1
    assert len({lease.metadata["ring"] for lease in leases}) == 1
    assert len({lease.metadata["mode_category"] for lease in leases}) == 4
    assert len({lease.metadata["variant"] for lease in leases}) == 4
    for lease in leases:
        lease.resource[0].release()
    actor.work_coordinator.close()
    actor.registry.close()


def test_refill_gate_waits_for_a_new_snapshot_not_merely_a_historical_model_age(
    tmp_path, monkeypatch
):
    actor, _, manifests, _ = fake_actor(tmp_path, monkeypatch)
    clock = {"now": 0.0}
    monkeypatch.setattr(actors.time, "monotonic", lambda: clock["now"])
    gate = actor._stop_refill_gate(
        {
            "model_identity": "old-history",
            "candidate_identity_at_start": "candidate",
            "champion_identity_at_start": "champion",
        }
    )
    assert gate() is False
    clock["now"] = 10
    manifests[tmp_path / "candidate"] = manifests[tmp_path / "new"]
    assert gate() is True
    manifests[tmp_path / "candidate"] = manifests[tmp_path / "candidate-immutable"]
    assert gate() is True
    second = actor._stop_refill_gate(
        {
            "candidate_identity_at_start": "candidate",
            "champion_identity_at_start": "champion",
        }
    )
    clock["now"] = 3610
    assert second() is True
    actor.work_coordinator.close()
    actor.registry.close()


def test_actor_streams_early_counts_and_continues_quota_with_new_model_and_generation(
    tmp_path, monkeypatch
):
    actor, _, manifests, _ = fake_actor(tmp_path, monkeypatch)
    actor.games_per_batch = 4
    actor.experiment = replace(
        actor.experiment,
        selfplay=replace(
            actor.experiment.selfplay,
            stream_completed_games=True,
            rolling_game_slots=True,
            seed_contract="game-v1",
        ),
    )
    stopped = {"value": False}
    tasks = []

    class SelfPlay:
        def __init__(self, _native, evaluator, _store, config, identity):
            self.evaluator = evaluator
            self.config = config
            self.identity = identity
            tasks.append(
                (
                    config.games,
                    evaluator.model_identity,
                    identity.generation,
                    config.variant,
                )
            )

        def run(self, *, progress, stop_refill_requested, **_kwargs):
            self.evaluator.evaluator_rows += 50
            self.evaluator.evaluator_calls += 4
            progress(
                phase="selfplay_completed", completed_games=2, persisted_decisions=6
            )
            if len(tasks) == 1:
                manifests[tmp_path / "candidate"] = manifests[tmp_path / "new"]
                assert stop_refill_requested()
            else:
                assert not stop_refill_requested()
                stopped["value"] = True
            return [
                SimpleNamespace(
                    winner=0,
                    samples=3,
                    policy_samples=1,
                    search_simulations=9,
                    model_version=self.evaluator.model_version,
                    model_identity=self.evaluator.model_identity,
                )
            ] * 2

        def metrics_snapshot(self):
            return SelfPlayMetrics(
                started_games=2,
                completed_games=2,
                completed_decisions=6,
                full_decisions=2,
                fast_decisions=4,
            )

    monkeypatch.setattr(actors, "SelfPlayActor", SelfPlay)
    assert actor.run(stop_requested=lambda: stopped["value"]) == 2
    assert [(games, model, generation) for games, model, generation, _ in tasks] == [
        (4, "candidate", 0),
        (2, "new", 1),
    ]
    assert tasks[0][3] == tasks[1][3]
    rows = [json.loads(line) for line in actor.metrics_path.read_text().splitlines()]
    publications = [row for row in rows if row.get("record_kind") == "publication"]
    final = [row for row in rows if row.get("record_kind") != "publication"]
    assert len(publications) == len(final) == 2
    assert all("games" not in row and "samples" not in row for row in publications)
    assert [row["cumulative_games"] for row in rows] == [2, 2, 4, 4]
    assert sum(row.get("games", 0) for row in rows) == 4
    assert sum(row["published_games"] for row in publications) == 4
    assert actor.work_coordinator.metrics_snapshot()[
        "planned_games_by_requested_role"
    ] == {"candidate": 4}
    assert actor.work_coordinator.metrics_snapshot()[
        "completed_games_by_requested_role"
    ] == {"candidate": 4}
    actor.work_coordinator.close()
    actor.registry.close()


def test_task_setup_failure_releases_individual_model_pin(tmp_path, monkeypatch):
    actor, _, _, _ = fake_actor(tmp_path, monkeypatch)

    def fail(*_args, **_kwargs):
        raise RuntimeError("task setup failed")

    monkeypatch.setattr(actors, "SelfPlayActor", fail)
    with pytest.raises(RuntimeError, match="task setup failed"):
        actor.run(stop_requested=lambda: False)
    assert actor._active_work_provider is None
    actor.work_coordinator.close()
    actor.registry.close()


def test_champion_poll_cache_avoids_reverification_but_verifies_pointer_replacements(
    tmp_path, monkeypatch
):
    actor, _, manifests, _ = fake_actor(tmp_path, monkeypatch)
    clock = {"now": 0.0}
    checks = []
    monkeypatch.setattr(actors.time, "monotonic", lambda: clock["now"])

    def verified(path):
        checks.append(Path(path))
        return manifests[Path(path)]

    monkeypatch.setattr(actors, "load_model_manifest", verified)
    gate = actor._stop_refill_gate(
        {
            "candidate_identity_at_start": "candidate",
            "champion_identity_at_start": "champion",
        }
    )
    for instant in (0, 3, 6, 9, 12):
        clock["now"] = instant
        assert gate() is False
    assert checks == [tmp_path / "champion"]
    replacement = tmp_path / "replacement"
    replacement.write_text("new verified pointer")
    replacement.replace(tmp_path / "champion")
    manifests[tmp_path / "champion"] = SimpleNamespace(
        **(vars(manifests[tmp_path / "champion"]) | {"model_identity": "champion-new"})
    )
    clock["now"] = 15
    assert gate() is True
    assert checks == [tmp_path / "champion", tmp_path / "champion"]
    assert actor._read_champion().model_identity == "champion-new"
    assert len(checks) == 2
    actor.work_coordinator.close()
    actor.registry.close()


def test_failed_champion_verification_is_never_cached(tmp_path, monkeypatch):
    actor, _, _, _ = fake_actor(tmp_path, monkeypatch)
    failures = []

    def invalid(_):
        failures.append(True)
        raise ValueError("checkpoint checksum mismatch")

    monkeypatch.setattr(actors, "load_model_manifest", invalid)
    for _ in range(2):
        with pytest.raises(ValueError, match="checksum"):
            actor._read_champion()
    assert len(failures) == 2 and actor._champion_manifest is None
    actor.work_coordinator.close()
    actor.registry.close()


def test_noncoordinated_rolling_actor_receives_max_pin_refill_gate(
    tmp_path, monkeypatch
):
    actor, _, _, _ = fake_actor(tmp_path, monkeypatch)
    actor.work_coordinator.close()
    actor.work_coordinator = None
    actor.games_per_batch = 1
    actor.experiment = replace(
        actor.experiment,
        selfplay=replace(
            actor.experiment.selfplay,
            stream_completed_games=True,
            rolling_game_slots=True,
            seed_contract="game-v1",
        ),
    )
    clock = {"now": 0.0}
    stopped = {"value": False}
    monkeypatch.setattr(actors.time, "monotonic", lambda: clock["now"])

    class SelfPlay:
        def __init__(self, _native, evaluator, _store, _config, _identity):
            self.evaluator = evaluator

        def run(self, *, stop_refill_requested, **_kwargs):
            assert not stop_refill_requested()
            clock["now"] = 3601.0
            assert stop_refill_requested()
            stopped["value"] = True
            return [
                SimpleNamespace(
                    winner=0,
                    samples=1,
                    policy_samples=1,
                    search_simulations=1,
                    model_version=self.evaluator.model_version,
                    model_identity=self.evaluator.model_identity,
                )
            ]

        def metrics_snapshot(self):
            return SelfPlayMetrics(
                started_games=1,
                completed_games=1,
                completed_decisions=1,
                full_decisions=1,
            )

    monkeypatch.setattr(actors, "SelfPlayActor", SelfPlay)
    assert actor.run(stop_requested=lambda: stopped["value"]) == 1
    actor.registry.close()
