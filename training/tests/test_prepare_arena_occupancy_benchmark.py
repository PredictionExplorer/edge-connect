from __future__ import annotations

import json
import os
import stat
from pathlib import Path
from types import SimpleNamespace

import pytest
import yaml

import scripts.prepare_arena_occupancy_benchmark as benchmark_plan_module
from scripts.prepare_arena_occupancy_benchmark import (
    CONTROL_ARM,
    TREATMENT_ARM,
    prepare_arena_occupancy_benchmark,
    verify_arena_occupancy_plan,
)

CONFIGS = Path(__file__).parents[1] / "configs"


def _fake_selection() -> SimpleNamespace:
    profile = (CONFIGS / "h100-8gpu-ring10-only.yaml").resolve()
    return SimpleNamespace(
        candidates=(SimpleNamespace(model_identity="candidate"),),
        plan_digest="a" * 64,
        evaluation_profile=SimpleNamespace(verify=lambda: profile),
    )


def _patch_selection_plan(
    monkeypatch: pytest.MonkeyPatch,
    selection: SimpleNamespace,
) -> None:
    monkeypatch.setattr(
        benchmark_plan_module,
        "plan_archived_manifest_evaluation",
        lambda **_kwargs: selection,
    )
    monkeypatch.setattr(
        benchmark_plan_module,
        "verify_selection_plan",
        lambda _plan, **_kwargs: selection,
    )
    monkeypatch.setattr(
        benchmark_plan_module,
        "_git_revision",
        lambda: ("c" * 40, True),
    )
    monkeypatch.setattr(
        benchmark_plan_module,
        "_native_extension_artifact",
        lambda: {
            "path": "/test/star_native.so",
            "bytes": 1,
            "sha256": "d" * 64,
            "rules_hash": "fnv1a64:2da3783519381453",
        },
    )

    def freeze(path: Path, _selection: object) -> None:
        path.write_text('{"frozen":true}\n', encoding="utf-8")
        os.chmod(path, 0o444)

    monkeypatch.setattr(benchmark_plan_module, "freeze_selection_plan", freeze)


def test_git_revision_accepts_sha256_repository(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    completed = iter(
        [
            SimpleNamespace(stdout=f"{'e' * 64}\n"),
            SimpleNamespace(stdout=""),
        ]
    )
    monkeypatch.setattr(
        benchmark_plan_module.subprocess,
        "run",
        lambda *_args, **_kwargs: next(completed),
    )

    assert benchmark_plan_module._git_revision(tmp_path) == ("e" * 64, True)


def test_native_identity_resolves_compiled_extension_not_package_wrapper(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    suffix = benchmark_plan_module.importlib.machinery.EXTENSION_SUFFIXES[0]
    extension = tmp_path / f"star_native{suffix}"
    extension.write_bytes(b"compiled-extension")
    monkeypatch.setattr(
        benchmark_plan_module.importlib.util,
        "find_spec",
        lambda _name: SimpleNamespace(origin=str(extension)),
    )

    resolved = benchmark_plan_module._native_extension_path(
        SimpleNamespace(
            __name__="star_native",
            __file__=str(tmp_path / "__init__.py"),
        )
    )
    assert resolved == extension.resolve()
    assert resolved.name != "__init__.py"


def test_prepare_freezes_registered_arena_occupancy_arms(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source = tmp_path / "source"
    source.mkdir()
    candidate = tmp_path / "candidate.json"
    candidate.write_text("{}\n", encoding="utf-8")
    selection = _fake_selection()
    _patch_selection_plan(monkeypatch, selection)
    output = tmp_path / "benchmark-plan"

    payload = prepare_arena_occupancy_benchmark(
        source_run_root=source,
        profile=CONFIGS / "h100-8gpu-ring10-only.yaml",
        candidate_manifest=candidate,
        output_dir=output,
        repeats=4,
    )

    assert [arm["name"] for arm in payload["arms"]] == [
        CONTROL_ARM,
        TREATMENT_ARM,
    ]
    assert [arm["total_pair_count"] for arm in payload["arms"]] == [50, 50]
    assert [arm["chunk_pair_count"] for arm in payload["arms"]] == [25, 50]
    assert payload["schedule"] == [
        {
            "repeat": 0,
            "pair_start": 50,
            "arm_order": [CONTROL_ARM, TREATMENT_ARM],
        },
        {
            "repeat": 1,
            "pair_start": 100,
            "arm_order": [TREATMENT_ARM, CONTROL_ARM],
        },
        {
            "repeat": 2,
            "pair_start": 50,
            "arm_order": [TREATMENT_ARM, CONTROL_ARM],
        },
        {
            "repeat": 3,
            "pair_start": 100,
            "arm_order": [CONTROL_ARM, TREATMENT_ARM],
        },
    ]
    assert payload["deployment_policy"]["treatment_deployable"] is False
    assert payload["selection_plan"]["plan_digest"] == selection.plan_digest
    plan_path = output / "benchmark-plan.json"
    assert stat.S_IMODE(plan_path.stat().st_mode) == 0o444

    verified = verify_arena_occupancy_plan(plan_path)
    assert verified.selection is selection
    assert verified.experiment.orchestration.training_objective == "ring10_only"


def test_verify_rejects_modified_benchmark_plan(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source = tmp_path / "source"
    source.mkdir()
    candidate = tmp_path / "candidate.json"
    candidate.write_text("{}\n", encoding="utf-8")
    selection = _fake_selection()
    _patch_selection_plan(monkeypatch, selection)
    output = tmp_path / "benchmark-plan"
    prepare_arena_occupancy_benchmark(
        source_run_root=source,
        profile=CONFIGS / "h100-8gpu-ring10-only.yaml",
        candidate_manifest=candidate,
        output_dir=output,
        repeats=4,
    )
    plan_path = output / "benchmark-plan.json"
    payload = json.loads(plan_path.read_text(encoding="utf-8"))
    payload["arms"][1]["chunk_pair_count"] = 49
    os.chmod(plan_path, 0o644)
    plan_path.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(ValueError, match="digest mismatch"):
        verify_arena_occupancy_plan(plan_path)


def test_verify_rejects_semantic_drift_with_recomputed_digest(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source = tmp_path / "source"
    source.mkdir()
    candidate = tmp_path / "candidate.json"
    candidate.write_text("{}\n", encoding="utf-8")
    selection = _fake_selection()
    _patch_selection_plan(monkeypatch, selection)
    output = tmp_path / "benchmark-plan"
    prepare_arena_occupancy_benchmark(
        source_run_root=source,
        profile=CONFIGS / "h100-8gpu-ring10-only.yaml",
        candidate_manifest=candidate,
        output_dir=output,
        repeats=4,
    )
    plan_path = output / "benchmark-plan.json"
    payload = json.loads(plan_path.read_text(encoding="utf-8"))
    payload["statistical_contract"]["simulations"] = 512
    payload["plan_digest"] = benchmark_plan_module._digest_payload(payload)
    os.chmod(plan_path, 0o644)
    plan_path.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(ValueError, match="statistical contract"):
        verify_arena_occupancy_plan(plan_path)


def test_prepare_refuses_live_root_and_invalid_repeat_count(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source = tmp_path / "source"
    source.mkdir()
    candidate = tmp_path / "candidate.json"
    candidate.write_text("{}\n", encoding="utf-8")
    _patch_selection_plan(monkeypatch, _fake_selection())

    with pytest.raises(ValueError, match="outside the source run root"):
        prepare_arena_occupancy_benchmark(
            source_run_root=source,
            profile=CONFIGS / "h100-8gpu-ring10-only.yaml",
            candidate_manifest=candidate,
            output_dir=source / "benchmark",
        )
    with pytest.raises(ValueError, match="outside the Git repository"):
        prepare_arena_occupancy_benchmark(
            source_run_root=source,
            profile=CONFIGS / "h100-8gpu-ring10-only.yaml",
            candidate_manifest=candidate,
            output_dir=benchmark_plan_module.REPOSITORY_ROOT / "benchmark-output-test",
        )
    with pytest.raises(ValueError, match="exactly 4"):
        prepare_arena_occupancy_benchmark(
            source_run_root=source,
            profile=CONFIGS / "h100-8gpu-ring10-only.yaml",
            candidate_manifest=candidate,
            output_dir=tmp_path / "other",
            repeats=5,
        )


def test_prepare_refuses_active_source_lock(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source = tmp_path / "source"
    source.mkdir()
    (source / "coordinator.lock").write_text("active\n", encoding="utf-8")
    candidate = tmp_path / "candidate.json"
    candidate.write_text("{}\n", encoding="utf-8")
    _patch_selection_plan(monkeypatch, _fake_selection())

    with pytest.raises(ValueError, match="coordinator lock"):
        prepare_arena_occupancy_benchmark(
            source_run_root=source,
            profile=CONFIGS / "h100-8gpu-ring10-only.yaml",
            candidate_manifest=candidate,
            output_dir=tmp_path / "output",
        )


def test_prepare_rejects_chunked_profile_that_cannot_form_the_large_batch(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source = tmp_path / "source"
    source.mkdir()
    candidate = tmp_path / "candidate.json"
    candidate.write_text("{}\n", encoding="utf-8")
    raw = yaml.safe_load(
        (CONFIGS / "h100-8gpu-ring10-only.yaml").read_text(encoding="utf-8")
    )
    raw["arena"]["pair_chunk_size"] = 25
    profile = tmp_path / "chunked.yaml"
    profile.write_text(yaml.safe_dump(raw, sort_keys=False), encoding="utf-8")
    _patch_selection_plan(monkeypatch, _fake_selection())

    with pytest.raises(ValueError, match="unchunked production"):
        prepare_arena_occupancy_benchmark(
            source_run_root=source,
            profile=profile,
            candidate_manifest=candidate,
            output_dir=tmp_path / "output",
        )


def test_prepare_requires_exact_200_pair_range(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source = tmp_path / "source"
    source.mkdir()
    candidate = tmp_path / "candidate.json"
    candidate.write_text("{}\n", encoding="utf-8")
    raw = yaml.safe_load(
        (CONFIGS / "h100-8gpu-ring10-only.yaml").read_text(encoding="utf-8")
    )
    raw["arena"]["max_pairs_per_ring"] = 201
    profile = tmp_path / "oversized.yaml"
    profile.write_text(yaml.safe_dump(raw, sort_keys=False), encoding="utf-8")
    _patch_selection_plan(monkeypatch, _fake_selection())

    with pytest.raises(ValueError, match="200-pair range"):
        prepare_arena_occupancy_benchmark(
            source_run_root=source,
            profile=profile,
            candidate_manifest=candidate,
            output_dir=tmp_path / "output",
        )
