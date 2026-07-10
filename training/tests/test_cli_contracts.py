from __future__ import annotations

import subprocess
import sys
from collections.abc import Callable

import pytest

from starserve.cli import main as starserve_main
from startrain.cli import (
    actor_main,
    arena_main,
    main,
    selfplay_main,
    train_main,
)
from startrain.distill import distill_main
from startrain.orchestration import orchestrate_main
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


def test_dispatcher_rejects_missing_and_unknown_commands() -> None:
    with pytest.raises(SystemExit, match="expected one of"):
        main([])
    with pytest.raises(SystemExit, match="unknown startrain command"):
        main(["not-a-command"])


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
