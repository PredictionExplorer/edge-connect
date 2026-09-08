from collections import Counter
from dataclasses import replace
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch
import yaml

from scripts import run_gradient_clipping_trial as trial
from startrain.checkpoint import (
    ExponentialMovingAverage,
    save_checkpoint,
    sha256_file,
    write_recovery_checkpoint,
)
from startrain.config import load_config
from startrain.model import GraphResTNet, ModelConfig
from startrain.optim import build_optimizer
from startrain.training import build_scheduler


def tiny_config():
    source = load_config(Path(__file__).parents[1] / "configs/small.yaml")
    return replace(
        source,
        model=ModelConfig(
            width=8,
            rrt_groups=1,
            attention_heads=1,
            kv_heads=1,
            local_blocks_per_group=1,
            ff_multiplier=1.0,
        ),
        optimizer=replace(source.optimizer, kind="adamw"),
        train=replace(
            source.train,
            per_rank_batch_size=1,
            precision="fp32",
            compile=False,
            ema_decay=0.9,
            ema_half_life_examples=None,
        ),
    )


def state(config):
    torch.manual_seed(4)
    model = GraphResTNet(config.model)
    optimizer = build_optimizer(model, config.optimizer)
    scheduler = build_scheduler(optimizer, config.train.scheduler)
    ema = ExponentialMovingAverage(model, decay=config.train.resolved_ema_decay(1))
    for p in model.parameters():
        p.grad = torch.ones_like(p) * 0.01
    optimizer.step()
    scheduler.step()
    ema.update(model)
    with torch.no_grad():
        for p in model.parameters():
            p.add_(0.02)
    return model, optimizer, scheduler, ema


def test_schedule_realizes_exact_objective_and_identical_arm_batches():
    first = trial.cell_schedule(seed=17, steps=600)
    assert first == trial.cell_schedule(seed=17, steps=600)
    assert first != trial.cell_schedule(seed=18, steps=600)
    counts = Counter(first)
    assert counts == {
        cell: 85 if cell.startswith("r10/") else 5 for cell in trial.CELLS
    }
    assert len(trial.cell_schedule(seed=17, steps=7)) == 7


def test_full_recovery_restores_raw_weights_optimizer_scheduler_and_ema_for_both_arms(
    tmp_path,
):
    config = tiny_config()
    model, opt, scheduler, ema = state(config)
    checkpoint = tmp_path / "recovery.pt"
    save_checkpoint(
        checkpoint,
        model=model,
        optimizer=opt,
        scheduler=scheduler,
        ema=ema,
        step=731,
        epoch=4,
        config=config.as_dict(),
        extra={"run_id": "trial-run", "generation_family": "trial-family"},
    )
    document = {
        "source_step": 731,
        "run_id": "trial-run",
        "generation_family": "trial-family",
        "recovery": {
            "path": str(checkpoint),
            "sha256": sha256_file(checkpoint),
            "bytes": checkpoint.stat().st_size,
        },
    }
    recovered = []
    for arm in ("global", "adagc"):
        restored = trial.restore_training_state(
            config, document, device=torch.device("cpu"), arm=arm
        )
        actual, actual_opt, actual_sched, actual_ema, clipper, metadata, initial = (
            restored
        )
        for name, parameter in actual.named_parameters():
            assert torch.equal(parameter, dict(model.named_parameters())[name])
            assert not torch.equal(parameter, ema.shadow[name])
        for key, expected in (
            ("model", model.state_dict()),
            ("optimizer", opt.state_dict()),
            ("scheduler", scheduler.state_dict()),
            ("ema", ema.state_dict()),
        ):
            assert initial[key] == trial.state_fingerprint(expected)
        assert metadata["step"] == 731 and actual_ema.num_updates == ema.num_updates
        assert actual_opt.param_groups[0]["lr"] == opt.param_groups[0]["lr"]
        if clipper is not None:
            assert clipper.steps == 0
        recovered.append(initial)
    assert recovered[0] == recovered[1]


def test_restore_rejects_changed_optimizer_contract(tmp_path):
    config = tiny_config()
    model, opt, scheduler, ema = state(config)
    path = tmp_path / "recovery.pt"
    save_checkpoint(
        path,
        model=model,
        optimizer=opt,
        scheduler=scheduler,
        ema=ema,
        step=1,
        config=config.as_dict(),
        extra={"run_id": "trial-run", "generation_family": "trial-family"},
    )
    document = {
        "source_step": 1,
        "run_id": "trial-run",
        "generation_family": "trial-family",
        "recovery": {
            "path": str(path),
            "sha256": sha256_file(path),
            "bytes": path.stat().st_size,
        },
    }
    changed = replace(
        config,
        optimizer=replace(config.optimizer, adamw_lr=config.optimizer.adamw_lr * 2),
    )
    with pytest.raises(ValueError, match="optimizer"):
        trial.restore_training_state(
            changed, document, device=torch.device("cpu"), arm="global"
        )


def test_output_protection_and_tampered_copy(tmp_path):
    source = tmp_path / "production"
    source.mkdir()
    artifact = source / "model.pt"
    artifact.write_bytes(b"model")
    with pytest.raises(ValueError, match="overlaps"):
        trial._disjoint_output(source / "trial", [source])
    with pytest.raises(ValueError, match="checksum"):
        trial._copy_pin(artifact, tmp_path / "copy", expected="0" * 64)
    assert not (tmp_path / "copy").exists()


@pytest.mark.native
def test_preparation_copies_game_disjoint_all_cell_replay_and_survives_source_gc(
    tmp_path, monkeypatch
):
    from startrain.native import positions_from_native
    from startrain.replay import ReplaySample
    from startrain.replay_store import ReplayStore
    from startrain.runtime import RunIdentity

    native = pytest.importorskip("star_native")
    config = tiny_config()
    model, opt, scheduler, ema = state(config)
    production = tmp_path / "production"
    production.mkdir()
    profile = production / "profile.yaml"
    profile.write_text(yaml.safe_dump(config.as_dict()))
    run = RunIdentity(production / "run.json", "trial-run", "trial-family", 1)
    write_recovery_checkpoint(
        production / "learner",
        model=model,
        optimizer=opt,
        scheduler=scheduler,
        ema=ema,
        step=731,
        epoch=4,
        config=config.as_dict(),
        run_id=run.run_id,
        generation_family=run.generation_family,
        examples_consumed=731,
        global_batch_size=1,
    )
    with ReplayStore(production / "replay") as store:
        store.register_run(run)
        assert store.lease_generation(run, "actor") == 0
        for cell in trial.CELLS:
            ring, kind = cell.split("/")
            mode, rule = kind.split("-")
            states = native.StateBatch(
                int(ring[1:]),
                24,
                mode=mode,
                handicap=3 if rule == "handicap" else 1,
                pie=rule == "pie",
            )
            samples = []
            for index, position in enumerate(positions_from_native(states.data())):
                policy = np.ones(position.stones.numel(), dtype=np.float32)
                policy /= policy.sum()
                samples.append(
                    ReplaySample.from_position(
                        position,
                        policy=policy,
                        final_score=None,
                        search_provenance="trial-test",
                        policy_provenance="trial-test",
                        run_id=run.run_id,
                        generation_family=run.generation_family,
                        actor_id="actor",
                        generation=0,
                        game_id=f"game-{cell.replace('/', '-')}-{index}",
                        model_identity="sha256-" + "a" * 64,
                    )
                )
            store.append(
                samples,
                phase_min=0,
                phase_max=0,
                model_version="sha256-" + "a" * 64,
                model_step=700,
                model_identity="sha256-" + "a" * 64,
                run_id=run.run_id,
                generation_family=run.generation_family,
                actor_id="actor",
                generation=0,
            )
    args = SimpleNamespace(
        config=profile,
        recovery=production / "learner/recovery.json",
        replay_root=production / "replay",
        output_dir=tmp_path / "experiment",
        replay_cutoff=None,
        seed=17,
        train_samples_per_cell=2,
        validation_samples_per_cell=2,
    )
    monkeypatch.setattr(trial, "_validate_production_config", lambda _: None)
    result = trial.prepare(args)
    manifest = Path(result["manifest"])
    first, replay, cells = trial.load_frozen(manifest)
    assert len(cells) == 24
    for p in (production / "replay/shards").glob("*.npz"):
        p.unlink()
    second, _, again = trial.load_frozen(manifest)
    assert first == second
    train = {
        ref["game_id"] for value in first["cells"].values() for ref in value["train"]
    }
    validation = {
        ref["game_id"]
        for value in first["cells"].values()
        for ref in value["validation"]
    }
    assert not train & validation
    for cell in trial.CELLS:
        a = trial._helpers()._materialize(
            replay, cells[cell]["train"], start=0, batch_size=2, seed=17, augment=True
        )
        b = trial._helpers()._materialize(
            replay, again[cell]["train"], start=0, batch_size=2, seed=17, augment=True
        )
        assert torch.equal(a.inputs.node_features, b.inputs.node_features)
        assert torch.equal(a.targets.policy, b.targets.policy)
    tampered = json.loads(manifest.read_text())
    tampered["cells"][trial.CELLS[0]]["validation"][0] = tampered["cells"][
        trial.CELLS[0]
    ]["train"][0]
    bad = manifest.parent / "bad.json"
    bad.write_text(json.dumps(tampered))
    with pytest.raises(ValueError, match="leaks"):
        trial.load_frozen(bad)


def test_diagnostic_backward_preserves_model_and_optimizer_state():
    from startrain.features import DoubleStarPosition
    from startrain.replay import ReplaySample, collate_replay_samples
    from startrain.topology import get_topology

    config = tiny_config()
    model, opt, scheduler, ema = state(config)
    position = DoubleStarPosition(
        rings=4,
        stones=torch.full((get_topology(4).n,), -1, dtype=torch.int8),
        to_move=0,
        moves_left=1,
        opening=True,
        terminal=False,
    )
    policy = np.ones(position.stones.numel(), dtype=np.float32)
    policy /= policy.sum()
    sample = ReplaySample.from_position(
        position,
        policy=policy,
        final_score=None,
        search_provenance="trial-test",
        policy_provenance="trial-test",
    )
    batch = collate_replay_samples([sample])
    before = trial.state_fingerprint(
        [model.state_dict(), opt.state_dict(), scheduler.state_dict(), ema.state_dict()]
    )
    result = trial._diagnostic_backward(model, batch, opt, config)
    assert result["gradient_diagnostics"]["parameters"]
    assert before == trial.state_fingerprint(
        [model.state_dict(), opt.state_dict(), scheduler.state_dict(), ema.state_dict()]
    )


def trial_result(tmp_path, arm):
    folder = tmp_path / arm
    folder.mkdir()
    checkpoint = folder / "trial.pt"
    checkpoint.write_bytes(b"verified-checkpoint")
    losses = {
        cell: {"policy": 1.0, "value": 2.0, "composite": 3.0} for cell in trial.CELLS
    }
    result = {
        "format": "startrain.gradient-clipping-trial",
        "schema_version": 1,
        "status": "complete",
        "arm": arm,
        "diagnostic_only": False,
        "source_step": 731,
        "frozen_manifest_sha256": "a" * 64,
        "source_recovery": {"sha256": "b" * 64},
        "source_config": {"sha256": "c" * 64},
        "runtime_source_sha256": "d" * 64,
        "initial_state": {
            name: "e" * 64
            for name in ("model", "optimizer", "scheduler", "ema", "parameter_identity")
        },
        "rng_before_training": {"cpu": "f" * 64},
        "batch_pins": ["one"],
        "batch_schedule_sha256": trial.digest(["one"]),
        "completed_steps": 1,
        "steps": [{"trial_step": 1}],
        "cell_schedule": ["r10/classic-standard"],
        "precision": "bf16",
        "compile": True,
        "float32_matmul_precision": "high",
        "ownership_before": {"verified": True},
        "ownership_after": {"verified": True},
        "trial_checkpoint": {
            "path": str(checkpoint),
            "sha256": sha256_file(checkpoint),
            "bytes": checkpoint.stat().st_size,
        },
        "final_validation": {"raw": {"per_cell": losses}, "ema": {"per_cell": losses}},
        "timing": {"steady_seconds": 1},
    }
    path = folder / "result.json"
    path.write_text(json.dumps(result))
    return path, result


@pytest.mark.parametrize(
    "field",
    ["initial_state", "rng_before_training", "batch_pins", "runtime_source_sha256"],
)
def test_cross_arm_comparison_rejects_state_rng_batch_or_source_changes(
    tmp_path, field
):
    first, _ = trial_result(tmp_path, "global")
    second, result = trial_result(tmp_path, "adagc")
    assert trial.compare_trial_results([second, first])["comparable"] is True
    if field == "initial_state":
        result[field] = result[field] | {"optimizer": "f" * 64}
    elif field == "batch_pins":
        result[field] = ["different"]
        result["batch_schedule_sha256"] = trial.digest(result[field])
    else:
        result[field] = "changed"
    second.write_text(json.dumps(result))
    with pytest.raises(ValueError, match="not comparable"):
        trial.compare_trial_results([first, second])


def test_missing_incomplete_and_changed_checkpoint_results_fail_validation(tmp_path):
    path, result = trial_result(tmp_path, "global")
    trial.validate_result(
        path, arm="global", manifest_sha256="a" * 64, steps=1, diagnostic=False
    )
    with pytest.raises(ValueError, match="completed pinned"):
        trial.validate_result(
            path, arm="global", manifest_sha256="b" * 64, steps=1, diagnostic=False
        )
    Path(result["trial_checkpoint"]["path"]).write_bytes(b"changed")
    with pytest.raises(ValueError):
        trial.validate_result(
            path, arm="global", manifest_sha256="a" * 64, steps=1, diagnostic=False
        )


def test_rng_fingerprints_read_only_selected_cuda_device(monkeypatch):
    calls = []
    monkeypatch.setattr(
        torch.cuda,
        "get_rng_state",
        lambda device: calls.append(device) or torch.tensor([1, 2], dtype=torch.uint8),
    )
    monkeypatch.setattr(
        torch.cuda,
        "get_rng_state_all",
        lambda: pytest.fail("must not initialize other GPUs"),
    )
    device = torch.device("cuda:3")
    result = trial.rng_fingerprints(device)
    assert set(result) == {
        "python",
        "numpy",
        "torch_cpu",
        "cuda_selected",
    } and calls == [device]


def test_owned_timeout_is_reported_and_other_arm_does_not_start(tmp_path, monkeypatch):
    import subprocess
    from contextlib import contextmanager
    from startrain import training

    frozen = tmp_path / "frozen.json"
    frozen.write_text(
        json.dumps({"source_replay_root": str(tmp_path / "production/replay")})
    )
    calls = []

    def execute(command, **kwargs):
        calls.append(command)
        assert command[command.index("--manifest-sha256") + 1] == sha256_file(frozen)
        raise subprocess.TimeoutExpired(
            command, kwargs["timeout"], output=b"partial", stderr=b"timeout"
        )

    @contextmanager
    def cache(_path):
        yield SimpleNamespace(environment={})

    monkeypatch.setattr(trial._control(), "_run_owned", execute)
    monkeypatch.setattr(training, "isolated_compile_cache", cache)
    output = tmp_path / "trial"
    assert (
        trial.main(
            [
                "--replay-manifest",
                str(frozen),
                "--output-dir",
                str(output),
                "--cpu-affinity",
                "0-31",
                "--steps",
                "1",
            ]
        )
        == 1
    )
    assert (
        len(calls) == 1
        and json.loads((output / "execution.json").read_text())[0]["status"]
        == "timeout"
    )
    assert (output / "global/stdout.log").read_text() == "partial"
