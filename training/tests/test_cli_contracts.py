from __future__ import annotations

import json
import subprocess
import sys
from collections.abc import Callable
from pathlib import Path

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
