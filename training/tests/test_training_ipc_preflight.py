from __future__ import annotations

from types import SimpleNamespace

import pytest

import scripts.check_training_ipc as ipc


def _completed(stdout: str, *, returncode: int = 0, stderr: str = ""):
    return SimpleNamespace(stdout=stdout, stderr=stderr, returncode=returncode)


def test_effective_remove_ipc_parses_authoritative_logind_property(
    monkeypatch,
) -> None:
    monkeypatch.setattr(
        ipc.subprocess,
        "run",
        lambda *_args, **_kwargs: _completed("b false\n"),
    )

    assert ipc.effective_remove_ipc() is False


def test_non_system_training_user_requires_remove_ipc_disabled(monkeypatch) -> None:
    monkeypatch.setattr(
        ipc.pwd,
        "getpwnam",
        lambda _user: SimpleNamespace(pw_uid=1_000),
    )
    monkeypatch.setattr(ipc, "effective_remove_ipc", lambda: True)

    with pytest.raises(RuntimeError, match="RemoveIPC=yes"):
        ipc.check_training_ipc("ubuntu")


def test_system_user_is_exempt_but_policy_is_reported(monkeypatch) -> None:
    monkeypatch.setattr(
        ipc.pwd,
        "getpwnam",
        lambda _user: SimpleNamespace(pw_uid=999),
    )
    monkeypatch.setattr(ipc, "effective_remove_ipc", lambda: True)

    report = ipc.check_training_ipc("trainer")

    assert report["status"] == "passed"
    assert report["effective_remove_ipc"] is True
    assert report["system_user_exempt"] is True


def test_unqueryable_logind_fails_closed(monkeypatch) -> None:
    monkeypatch.setattr(
        ipc.subprocess,
        "run",
        lambda *_args, **_kwargs: _completed(
            "",
            returncode=1,
            stderr="D-Bus unavailable",
        ),
    )

    with pytest.raises(RuntimeError, match="cannot query"):
        ipc.effective_remove_ipc()
