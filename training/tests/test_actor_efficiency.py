from __future__ import annotations

import random
import threading
from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import pytest


from startrain import actor as actor_module
from startrain.actor import ActorSupervisor, HistoricalModelPool, SharedModelRegistry
from startrain.config import CPUActorConfig, GPUWorkerConfig, load_config
from startrain.orchestration import RunDirectories, build_worker_specs
from startrain.runtime import RunIdentity


def test_history_eligibility_drops_stale_and_future_models_before_sampling(
    tmp_path, monkeypatch
):
    config = load_config(Path(__file__).parents[1] / "configs/small.yaml")
    identity = RunIdentity(tmp_path / "run.json", "run", "family", 1)
    manifests = {}
    for step in (10, 20, 30, 40):
        path = tmp_path / f"manifest-{step}.json"
        path.write_text("{}")
        manifests[path] = SimpleNamespace(
            role="direct",
            run_id="run",
            generation_family="family",
            published_ns=1,
            model_identity=f"model-{step}",
            model_step=step,
            artifact_manifest=None,
            path=path,
        )
    monkeypatch.setattr(
        actor_module, "load_model_manifest", lambda path: manifests[Path(path)]
    )
    monkeypatch.setattr(
        actor_module,
        "ManifestModelProvider",
        lambda _config, path, **kwargs: SimpleNamespace(path=path),
    )
    pool = HistoricalModelPool(
        config, tmp_path, device="cpu", run_identity=identity, pool_size=8
    )
    observed = {
        pool.select(
            random_source=random.Random(seed),
            exclude=set(),
            minimum_model_step=20,
            maximum_model_step=30,
        ).path
        for seed in range(50)
    }
    assert observed == {tmp_path / "manifest-20.json", tmp_path / "manifest-30.json"}
    assert pool.last_selection_metrics["history_ineligible_models"] == 2


def test_cpu_worker_has_reserved_affinity_and_independent_native_blas_budgets(tmp_path):
    config = load_config(Path(__file__).parents[1] / "configs/small.yaml")
    gpu = GPUWorkerConfig(
        gpu_id=1,
        role="actor",
        actor_batch_size=8,
        cpu_threads=8,
        native_threads=6,
        blas_threads=2,
        cpu_affinity="0-7",
    )
    cpu = CPUActorConfig(actor_id="actor-cpu-small", cpu_affinity="8-15")
    config = replace(
        config,
        orchestration=replace(
            config.orchestration,
            enabled=True,
            gpus=(
                GPUWorkerConfig(
                    gpu_id=0, role="learner", cpu_threads=4, cpu_affinity="0-7"
                ),
                gpu,
            ),
            cpu_actors=(cpu,),
            promotion=replace(
                config.orchestration.promotion,
                enabled=True,
                gpu_id=1,
                pause_sharing_mode=True,
            ),
            directories=replace(config.orchestration.directories, root=str(tmp_path)),
        ),
    )
    specs = build_worker_specs(
        config,
        config_path=tmp_path / "config.yaml",
        directories=RunDirectories.from_experiment(config),
        python_executable="python",
        base_environment={},
    )
    gpu_spec = next(spec for spec in specs if spec.name == "actor-gpu-1")
    cpu_spec = next(spec for spec in specs if spec.name == cpu.actor_id)
    assert gpu_spec.environment["RAYON_NUM_THREADS"] == "6"
    assert gpu_spec.environment["OMP_NUM_THREADS"] == "2"
    assert cpu_spec.gpu_ids == () and cpu_spec.cpu_affinity == tuple(range(8, 16))
    assert cpu_spec.environment["CUDA_VISIBLE_DEVICES"] == ""
    assert "--cpu-actor-index" in cpu_spec.command and cpu_spec.command[-1] == "cpu"


def test_registry_shares_immutable_model_and_does_not_evict_a_pinned_cohort(
    monkeypatch,
):
    loaded, closed = [], []
    broker = SimpleNamespace(cohort_adapter=lambda adapter: adapter)
    registry = SharedModelRegistry(broker, max_entries=2)

    def manifest(path):
        return SimpleNamespace(
            model_identity=path, artifact_manifest=None, path=path, role="direct"
        )

    monkeypatch.setattr(actor_module, "load_model_manifest", manifest)

    class Loader:
        def __init__(self, _config, path, **_kwargs):
            self.path = path

        def refresh(self):
            loaded.append(self.path)
            return SimpleNamespace(
                model_identity=self.path, close=lambda: closed.append(self.path)
            )

    monkeypatch.setattr(actor_module, "ManifestModelProvider", Loader)
    provider = SimpleNamespace(
        config=load_config(Path(__file__).parents[1] / "configs/small.yaml"),
        device="cpu",
        run_identity=None,
    )
    a = registry.acquire(provider, manifest("a"))
    assert registry.acquire(provider, manifest("a")) is a
    registry.release("a")
    registry.acquire(provider, manifest("b"))
    registry.release("b")
    registry.acquire(provider, manifest("c"))
    assert loaded == ["a", "b", "c"] and closed == ["b"]
    registry.release("a")
    registry.release("c")
    registry.close()


@pytest.mark.parametrize("fail_one", [False, True])
def test_shared_cohorts_drain_before_closing_broker(tmp_path, monkeypatch, fail_one):
    import startrain.inference_batching as batching

    events = []

    class Broker:
        def __init__(self, **_kwargs):
            pass

        def metrics_snapshot(self):
            return {}

        def shutdown(self, **_kwargs):
            events.append("broker-closed")

    monkeypatch.setattr(batching, "BoundedInferenceBroker", Broker)
    config = load_config(Path(__file__).parents[1] / "configs/small.yaml")
    config = replace(
        config,
        orchestration=replace(
            config.orchestration,
            model_refresh=replace(
                config.orchestration.model_refresh,
                inference=replace(
                    config.orchestration.model_refresh.inference, shared_batching=True
                ),
            ),
        ),
    )
    supervisor = ActorSupervisor(
        native_module=object(),
        experiment=config,
        gpu=GPUWorkerConfig(
            gpu_id=1, role="actor", cpu_threads=4, actor_batch_size=1, actor_cohorts=2
        ),
        replay_directory=tmp_path / "replay",
        manifest_path=tmp_path / "champion.json",
        candidate_manifest_path=tmp_path / "candidate.json",
        run_identity=RunIdentity(tmp_path / "run.json", "run", "family", 1),
        heartbeat_path=tmp_path / "actor.heartbeat.json",
        metrics_path=tmp_path / "actor.jsonl",
        device="cpu",
    )
    barrier = threading.Barrier(2)

    def child_run(child, **kwargs):
        barrier.wait(timeout=5)
        if fail_one and child.actor_id.endswith("cohort-0"):
            raise RuntimeError("cohort failure")
        if fail_one:
            import time

            deadline = time.monotonic() + 5
            while not kwargs["stop_requested"]():
                assert time.monotonic() < deadline
                time.sleep(0.001)
        events.append(child.actor_id)
        return 1

    monkeypatch.setattr(ActorSupervisor, "run", child_run)
    if fail_one:
        with pytest.raises(RuntimeError, match="cohort failure"):
            supervisor._run_cohorts(stop_requested=lambda: False)
        assert len(events) == 2 and events[-1] == "broker-closed"
    else:
        assert supervisor._run_cohorts(stop_requested=lambda: False) == 2
        assert len(set(events[:2])) == 2 and events[-1] == "broker-closed"


@pytest.mark.native
@pytest.mark.parametrize("handicap,pie", [(1, False), (4, False), (1, True)])
def test_native_independent_cohorts_share_one_inference_owner(tmp_path, handicap, pie):
    from startrain.inference_batching import BoundedInferenceBroker
    from startrain.replay_store import ReplayStore
    from startrain.selfplay import (
        GameVariant,
        SelfPlayActor,
        SelfPlayConfig,
        SelfPlayIdentity,
        VariantMixtureConfig,
    )
    from test_variant_selfplay import evaluator

    native = pytest.importorskip("star_native")
    base = evaluator()
    identity = RunIdentity(tmp_path / "run.json", "shared-native", "shared-native", 1)
    # Initialize the ledger before threads start; each cohort owns a connection.
    with ReplayStore(tmp_path / "replay") as store:
        generations = [
            store.lease_generation(identity, f"actor-{mode}")
            for mode in ("classic", "double")
        ]
    owners = set()
    original = base.evaluate_prepared

    def measured(*args, **kwargs):
        owners.add(threading.get_ident())
        return original(*args, **kwargs)

    base.evaluate_prepared = measured
    with BoundedInferenceBroker(
        max_batch_rows=8, max_pending_requests=4, max_wait_seconds=0.001
    ) as broker:

        def cohort(mode, generation):
            with ReplayStore(tmp_path / "replay") as store:
                actor = SelfPlayActor(
                    native,
                    broker.cohort_adapter(base),
                    store,
                    replace(
                        SelfPlayConfig.cpu_smoke(seed=generation + 41),
                        games=1,
                        batch_size=1,
                        exact_endgame_max_empty=3,
                        variants=VariantMixtureConfig(enabled=True),
                    ).with_variant(GameVariant(mode=mode, handicap=handicap, pie=pie)),
                    SelfPlayIdentity(
                        identity.run_id,
                        identity.generation_family,
                        f"actor-{mode}",
                        generation,
                    ),
                )
                return actor.run()

        with ThreadPoolExecutor(max_workers=2) as pool:
            futures = [
                pool.submit(cohort, mode, generation)
                for mode, generation in zip(
                    ("classic", "double"), generations, strict=True
                )
            ]
            results = [future.result(timeout=30) for future in futures]
        assert all(len(result) == 1 for result in results)
        assert len(owners) == 1 and threading.get_ident() not in owners
