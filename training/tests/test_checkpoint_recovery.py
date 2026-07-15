from __future__ import annotations

from pathlib import Path

import torch

from startrain.checkpoint import (
    ExponentialMovingAverage,
    collect_recovery_garbage,
    discover_resume_checkpoints,
    load_checkpoint,
    write_recovery_checkpoint,
    write_resume_cutover,
)


def _state():
    model = torch.nn.Linear(3, 2)
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.01)
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda _step: 1.0)
    ema = ExponentialMovingAverage(model)
    return model, optimizer, scheduler, ema


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
