from __future__ import annotations

import json
import subprocess
import sys
from collections.abc import Callable
from pathlib import Path
from types import SimpleNamespace

import pytest

import startrain.cli as cli_module
import startrain.orchestration as orchestration_module
from starserve.cli import main as starserve_main
from startrain.cli import (
    actor_main,
    arena_main,
    main,
    selfplay_main,
    train_main,
)
from startrain.distill import distill_main
from startrain.orchestration import (
    FATAL_WORKER_EXIT_CODE,
    TRANSIENT_WORKER_EXIT_CODE,
    WORKER_FAILURE_PATH_ENV,
    WORKER_NAME_ENV,
    WORKER_ROLE_ENV,
    orchestrate_main,
)
from startrain.preflight import preflight_main
from startrain.promotion import promotion_main
from startrain.publish import publish_browser_main


@pytest.mark.parametrize(
    "entrypoint",
    [
        selfplay_main,
        train_main,
        actor_main,
        arena_main,
        distill_main,
        promotion_main,
        publish_browser_main,
        orchestrate_main,
        preflight_main,
        starserve_main,
    ],
)
def test_every_operator_entrypoint_has_parseable_help(
    entrypoint: Callable[[list[str] | None], None],
    capsys: pytest.CaptureFixture[str],
) -> None:
    with pytest.raises(SystemExit) as stopped:
        entrypoint(["--help"])
    assert stopped.value.code == 0
    output = capsys.readouterr().out
    assert "usage:" in output.lower()
    assert "--help" in output


def test_preflight_reports_detection_and_config_resolution(
    capsys: pytest.CaptureFixture[str],
) -> None:
    config = Path(__file__).parents[1] / "configs" / "small.yaml"
    preflight_main(["--config", str(config)])
    report = json.loads(capsys.readouterr().out)
    assert report["schema_version"] == 1
    assert report["detected"]["preferred_device"] in ("cuda", "mps", "cpu")
    assert report["learner"]["device"] == "cpu"
    assert report["learner"]["precision"] == "fp32"
    assert report["orchestration"] == {"enabled": False}


def test_preflight_exercise_proves_the_host_device(
    capsys: pytest.CaptureFixture[str],
) -> None:
    preflight_main(["--exercise"])
    report = json.loads(capsys.readouterr().out)
    results = report["exercise"]
    assert len(results) == 1
    assert results[0]["ok"] is True
    assert results[0]["device"] == report["detected"]["preferred_device"]


def test_dispatcher_rejects_missing_and_unknown_commands() -> None:
    with pytest.raises(SystemExit, match="expected one of"):
        main([])
    with pytest.raises(SystemExit, match="unknown startrain command"):
        main(["not-a-command"])


@pytest.mark.parametrize(
    ("error", "expected_class", "expected_exit_code"),
    [
        (
            ValueError("checkpoint schema is incompatible"),
            "fatal",
            FATAL_WORKER_EXIT_CODE,
        ),
        (
            RuntimeError("DataLoader worker exited unexpectedly"),
            "transient",
            TRANSIENT_WORKER_EXIT_CODE,
        ),
    ],
)
def test_dispatcher_persists_classified_worker_failure(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    error: Exception,
    expected_class: str,
    expected_exit_code: int,
) -> None:
    failure_path = tmp_path / "learner.failure.json"

    def fail(_arguments: list[str] | None = None) -> None:
        raise error

    monkeypatch.setattr(cli_module, "train_main", fail)
    monkeypatch.setenv(WORKER_FAILURE_PATH_ENV, str(failure_path))
    monkeypatch.setenv(WORKER_NAME_ENV, "learner")
    monkeypatch.setenv(WORKER_ROLE_ENV, "learner")

    with pytest.raises(SystemExit) as stopped:
        main(["train"])

    assert stopped.value.code == expected_exit_code
    payload = json.loads(failure_path.read_text(encoding="utf-8"))
    assert payload["failure_class"] == expected_class
    assert payload["exit_code"] == expected_exit_code
    assert payload["exception_type"] == type(error).__name__
    assert payload["reason"] == str(error)
    assert str(error) in capsys.readouterr().err


def test_orchestrator_config_value_error_uses_fatal_exit_code(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    def reject_config(_path: str) -> None:
        raise ValueError("configuration migration required")

    monkeypatch.setattr(orchestration_module, "load_config", reject_config)

    with pytest.raises(SystemExit) as stopped:
        orchestrate_main(["--config", "invalid.yaml"])

    assert stopped.value.code == FATAL_WORKER_EXIT_CODE
    assert "configuration migration required" in capsys.readouterr().err


def test_python_module_entrypoints_fail_cleanly_without_arguments() -> None:
    startrain = subprocess.run(
        [sys.executable, "-m", "startrain.cli"],
        check=False,
        capture_output=True,
        text=True,
    )
    assert startrain.returncode != 0
    assert "expected one of" in startrain.stderr

    starserve = subprocess.run(
        [sys.executable, "-m", "starserve", "--help"],
        check=False,
        capture_output=True,
        text=True,
    )
    assert starserve.returncode == 0
    assert "usage:" in starserve.stdout.lower()


def test_arena_architecture_mode_is_explicit_and_diagnostic_only(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    config = Path(__file__).parents[1] / "configs" / "small.yaml"
    candidate_manifest = SimpleNamespace(
        path=tmp_path / "candidate-manifest.json",
        artifact_manifest=None,
        manifest_sha256="a" * 64,
        manifest_bytes=101,
        model_identity="sha256-" + "a" * 64,
        model_version="sha256-" + "a" * 64,
    )
    baseline_manifest = SimpleNamespace(
        path=tmp_path / "baseline-manifest.json",
        artifact_manifest=None,
        manifest_sha256="b" * 64,
        manifest_bytes=102,
        model_identity="sha256-" + "b" * 64,
        model_version="sha256-" + "b" * 64,
    )
    manifests = {
        "candidate.json": candidate_manifest,
        "baseline.json": baseline_manifest,
    }
    evaluator_modes = []
    writes = []
    common_contract = {
        "rules_hash_wire": "fnv1a64:test",
        "feature_schema_hash": 1,
        "action_layout_version": 1,
        "game": {"mode": "double", "pie_rule": False, "rings": (4, 6, 8, 10)},
    }

    monkeypatch.setattr(
        cli_module,
        "load_model_manifest",
        lambda path: manifests[str(path)],
    )

    def extract(manifest, **_options):
        return SimpleNamespace(
            model_config={
                "width": 16 if manifest is candidate_manifest else 24,
            },
            evaluation_contract=common_contract,
        )

    monkeypatch.setattr(cli_module, "extract_verified_manifest_config", extract)

    def load_evaluator(
        _experiment,
        manifest,
        *,
        device,
        allow_heterogeneous_model=False,
    ):
        evaluator_modes.append((manifest, device, allow_heterogeneous_model))
        return SimpleNamespace(model_version=manifest.model_version)

    monkeypatch.setattr(cli_module, "load_manifest_evaluator", load_evaluator)
    monkeypatch.setattr(
        cli_module,
        "load_star_native",
        lambda *, required: SimpleNamespace(required=required),
    )

    class DiagnosticArena:
        def __init__(self, **_options):
            pass

        def run(self):
            return {
                "schema_version": 3,
                "candidate": candidate_manifest.model_identity,
                "baseline": baseline_manifest.model_identity,
                "baseline_metadata": {"kind": "checkpoint"},
                "aggregate": {"pairs": 1},
                "promotion": {"decision": "promote"},
                "pairs": [],
                "games": [],
            }

    monkeypatch.setattr(cli_module, "ArenaRunner", DiagnosticArena)
    monkeypatch.setattr(
        cli_module,
        "atomic_json",
        lambda path, payload: writes.append((Path(path), payload)),
    )

    arena_main(
        [
            "--config",
            str(config),
            "--candidate",
            "candidate.json",
            "--baseline",
            "baseline.json",
            "--output",
            str(tmp_path / "homogeneous.json"),
            "--device",
            "cpu",
        ]
    )
    capsys.readouterr()
    assert evaluator_modes[-2:] == [
        (candidate_manifest, "cpu", False),
        (baseline_manifest, "cpu", False),
    ]
    assert writes[-1][1]["promotion"]["decision"] == "promote"

    arena_main(
        [
            "--config",
            str(config),
            "--candidate",
            "candidate.json",
            "--baseline",
            "baseline.json",
            "--output",
            str(tmp_path / "architecture.json"),
            "--device",
            "cpu",
            "--evaluation-mode",
            "architecture",
        ]
    )
    summary = json.loads(capsys.readouterr().out)
    assert evaluator_modes[-2:] == [
        (candidate_manifest, "cpu", True),
        (baseline_manifest, "cpu", True),
    ]
    result = writes[-1][1]
    assert result["diagnostic_only"] is True
    assert result["promotion_authorized"] is False
    assert result["adoption_authorized"] is False
    assert result["diagnostic_assessment"]["decision"] == "promote"
    assert result["promotion"] == {
        "decision": "evaluation",
        "authorized": False,
        "reason": "architecture evaluations cannot promote models",
    }
    assert summary["decision"] == "evaluation"

    with pytest.raises(SystemExit):
        arena_main(
            [
                "--config",
                str(config),
                "--candidate",
                "candidate.json",
                "--baseline-kind",
                "uniform",
                "--output",
                str(tmp_path / "invalid.json"),
                "--evaluation-mode",
                "architecture",
            ]
        )
