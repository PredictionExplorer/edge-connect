from __future__ import annotations

from dataclasses import asdict
from pathlib import Path

import pytest
import torch

from startrain.checkpoint import (
    ExponentialMovingAverage,
    collect_recovery_garbage,
    discover_resume_checkpoints,
    extract_verified_checkpoint_config,
    load_checkpoint,
    load_ema_checkpoint,
    normalize_model_config,
    save_checkpoint,
    write_recovery_checkpoint,
    write_resume_cutover,
)
from startrain.model import GraphResTNet, ModelConfig
from startrain.optim import OptimizerConfig, build_optimizer


def _state():
    model = torch.nn.Linear(3, 2)
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.01)
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda _step: 1.0)
    ema = ExponentialMovingAverage(model)
    return model, optimizer, scheduler, ema


def test_ema_state_can_require_configured_decay() -> None:
    model = torch.nn.Linear(3, 2)
    source = ExponentialMovingAverage(model, decay=0.99)
    target = ExponentialMovingAverage(model, decay=0.9)

    with pytest.raises(ValueError, match="configured training decay"):
        target.load_state_dict(source.state_dict(), expected_decay=target.decay)

    target.load_state_dict(source.state_dict())
    assert target.decay == 0.99


def test_checkpoint_binds_optimizer_routing_and_hyperparameters(
    tmp_path: Path,
) -> None:
    model = torch.nn.Linear(3, 2)
    optimizer = build_optimizer(
        model,
        OptimizerConfig(kind="adamw", adamw_lr=3e-4),
    )
    checkpoint = save_checkpoint(
        tmp_path / "optimizer-contract.pt",
        model=model,
        optimizer=optimizer,
        step=1,
    )

    matching_model = torch.nn.Linear(3, 2)
    matching = build_optimizer(
        matching_model,
        OptimizerConfig(kind="adamw", adamw_lr=3e-4),
    )
    load_checkpoint(checkpoint, model=matching_model, optimizer=matching)

    changed_model = torch.nn.Linear(3, 2)
    changed = build_optimizer(
        changed_model,
        OptimizerConfig(kind="adamw", adamw_lr=1e-3),
    )
    with pytest.raises(ValueError, match="hyperparameters"):
        load_checkpoint(checkpoint, model=changed_model, optimizer=changed)


def _write(root: Path, *, step: int):
    model, optimizer, scheduler, ema = _state()
    return write_recovery_checkpoint(
        root,
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
        ema=ema,
        step=step,
        epoch=step // 10,
        config={"model": {}, "game": {}},
        run_id="run-test",
        generation_family="family-test",
        examples_consumed=step * 4,
        global_batch_size=4,
    )


def test_recovery_pointer_journal_and_corrupt_head_fallback(tmp_path) -> None:
    first = _write(tmp_path, step=10)
    second = _write(tmp_path, step=20)
    assert second.step == 20

    candidates, failures = discover_resume_checkpoints(
        tmp_path,
        run_id="run-test",
        generation_family="family-test",
    )
    assert failures == []
    assert candidates[0].step == 20
    assert {candidate.step for candidate in candidates} >= {10, 20}

    (tmp_path / "recovery.json").write_text("{broken", encoding="utf-8")
    candidates, failures = discover_resume_checkpoints(
        tmp_path,
        run_id="run-test",
        generation_family="family-test",
    )
    assert any(failure.startswith("recovery.json:") for failure in failures)
    assert candidates[0].step == 20
    assert candidates[0].source.startswith("recovery-journal:")
    assert first.checkpoint.is_file()


def test_corrupt_newest_recovery_can_fall_back_to_previous(tmp_path) -> None:
    first = _write(tmp_path, step=10)
    second = _write(tmp_path, step=20)
    second.checkpoint.write_bytes(second.checkpoint.read_bytes() + b"corrupt")
    candidates, _ = discover_resume_checkpoints(
        tmp_path,
        run_id="run-test",
        generation_family="family-test",
    )
    accepted = None
    for candidate in candidates:
        model, optimizer, scheduler, ema = _state()
        try:
            metadata = load_checkpoint(
                candidate.checkpoint,
                model=model,
                optimizer=optimizer,
                scheduler=scheduler,
                ema=ema,
                expected_run_id="run-test",
                expected_generation_family="family-test",
                expected_sha256=candidate.checkpoint_sha256,
                expected_bytes=candidate.checkpoint_bytes,
            )
        except ValueError:
            continue
        accepted = int(metadata["step"])
        break
    assert accepted == 10
    assert first.checkpoint.is_file()
    collected = collect_recovery_garbage(tmp_path, retain_checkpoints=2, dry_run=False)
    assert collected["valid_recovery_checkpoints"] == 1
    assert not second.checkpoint.exists()


def test_recovery_identity_and_retention_are_strict(tmp_path) -> None:
    for step in (10, 20, 30):
        _write(tmp_path, step=step)
    candidates, failures = discover_resume_checkpoints(
        tmp_path,
        run_id="other-run",
        generation_family="family-test",
    )
    assert candidates == []
    assert failures

    dry_run = collect_recovery_garbage(tmp_path, retain_checkpoints=2, dry_run=True)
    assert dry_run["recovery_checkpoints"] == 1
    collected = collect_recovery_garbage(tmp_path, retain_checkpoints=2, dry_run=False)
    assert collected["deleted_recovery_checkpoints"] == 1
    assert len(list((tmp_path / "recovery").glob("sha256-*.pt"))) == 2


def test_resume_cutover_prevents_rejected_high_step_resurrection(tmp_path) -> None:
    champion = _write(tmp_path, step=10)
    rejected = _write(tmp_path, step=20)
    write_resume_cutover(
        tmp_path,
        manifest=champion,
        run_id="run-test",
        generation_family="family-test",
    )
    candidates, _ = discover_resume_checkpoints(
        tmp_path,
        run_id="run-test",
        generation_family="family-test",
    )
    assert candidates[0].step == 10
    assert all(candidate.checkpoint != rejected.checkpoint for candidate in candidates)

    continued = _write(tmp_path, step=15)
    candidates, _ = discover_resume_checkpoints(
        tmp_path,
        run_id="run-test",
        generation_family="family-test",
    )
    assert candidates[0].checkpoint == continued.checkpoint
    collect_recovery_garbage(
        tmp_path,
        retain_checkpoints=1,
        dry_run=False,
    )
    assert champion.checkpoint.is_file()
    assert continued.checkpoint.is_file()
    assert not rejected.checkpoint.exists()


def test_recovery_interval_rejects_cross_directory_path(tmp_path) -> None:
    recovery = _write(tmp_path, step=10)
    payload = (tmp_path / "recovery.json").read_text(encoding="utf-8")
    (tmp_path / "recovery.json").write_text(
        payload.replace(
            f"recovery/{recovery.checkpoint.name}",
            f"../{recovery.checkpoint.name}",
        ),
        encoding="utf-8",
    )
    candidates, failures = discover_resume_checkpoints(
        tmp_path,
        run_id="run-test",
        generation_family="family-test",
    )
    assert any("escaped" in failure for failure in failures)
    assert candidates


def test_verified_checkpoint_config_is_normalized_and_contract_bound(
    tmp_path: Path,
) -> None:
    model_config = ModelConfig(
        width=16,
        rrt_groups=1,
        attention_heads=4,
        kv_heads=1,
    )
    model = GraphResTNet(model_config)
    ema = ExponentialMovingAverage(model)
    checkpoint = save_checkpoint(
        tmp_path / "architecture.pt",
        model=model,
        ema=ema,
        step=7,
        config={
            "model": {
                "width": 16,
                "rrt_groups": 1,
                "attention_heads": 4,
                "kv_heads": 1,
            },
            "game": {},
        },
        extra={
            "run_id": "run-architecture",
            "generation_family": "family-architecture",
        },
    )

    verified = extract_verified_checkpoint_config(
        checkpoint,
        expected_run_id="run-architecture",
        expected_generation_family="family-architecture",
    )

    assert verified.model == model_config
    assert verified.model_config == asdict(model_config)
    assert verified.game_config == {
        "mode": "double",
        "pie_rule": False,
        "handicap": 1,
        "rings": (4, 6, 8, 10),
        "variants": {
            "modes": ("classic", "double"),
            "handicap_min": 1,
            "handicap_max": 9,
            "pie_allowed": True,
        },
    }
    assert verified.evaluation_contract["action_layout_version"] == 1
    assert verified.game_contract["rules_schema"] == "edgeconnect.star.rules.v3"
    assert verified.input_contract["feature_schema_version"] == 4
    assert set(verified.evaluation_contract) == (
        set(verified.game_contract) | set(verified.input_contract)
    )


def test_verified_checkpoint_config_rejects_unsafe_or_incomplete_metadata(
    tmp_path: Path,
) -> None:
    assert normalize_model_config({}) == asdict(ModelConfig())
    with pytest.raises(ValueError, match="unsupported keys"):
        normalize_model_config({"unknown_architecture_key": 1})
    with pytest.raises(ValueError, match="must be integer"):
        normalize_model_config({"width": True})

    model_config = ModelConfig(
        width=16,
        rrt_groups=1,
        attention_heads=4,
        kv_heads=1,
    )
    model = GraphResTNet(model_config)
    checkpoint = save_checkpoint(
        tmp_path / "no-ema.pt",
        model=model,
        step=1,
        config={"model": asdict(model_config), "game": {}},
        extra={
            "run_id": "run-no-ema",
            "generation_family": "family-no-ema",
        },
    )
    with pytest.raises(ValueError, match="no EMA"):
        extract_verified_checkpoint_config(checkpoint)
    with pytest.raises(ValueError, match="run_id"):
        extract_verified_checkpoint_config(
            checkpoint,
            expected_run_id="wrong-run",
            require_ema=False,
        )

    payload = torch.load(checkpoint, weights_only=True)
    payload["action_layout_version"] = 999
    torch.save(payload, checkpoint)
    with pytest.raises(ValueError, match="action layout version"):
        extract_verified_checkpoint_config(checkpoint, require_ema=False)

    partial_checkpoint = save_checkpoint(
        tmp_path / "partial.pt",
        model=model,
        ema=ExponentialMovingAverage(model),
        step=2,
        config={"model": asdict(model_config), "game": {}},
    )
    partial = torch.load(partial_checkpoint, weights_only=True)
    partial["model"].pop(next(iter(partial["model"])))
    torch.save(partial, partial_checkpoint)
    with pytest.raises(ValueError, match="model keys do not match"):
        load_ema_checkpoint(
            partial_checkpoint,
            model=GraphResTNet(model_config),
            expected_model_config=asdict(model_config),
            expected_game_config={},
        )
