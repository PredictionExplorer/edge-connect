from __future__ import annotations

import subprocess
import sys
from concurrent.futures import Future
from pathlib import Path

import pytest

from startrain.arena import _wait_for_future


def test_arena_future_wait_preserves_task_timeout_error():
    future = Future()
    error = TimeoutError("the task itself timed out")
    future.set_exception(error)
    with pytest.raises(TimeoutError) as captured:
        _wait_for_future(future)
    assert captured.value is error


@pytest.mark.skipif(sys.platform == "win32", reason="requires POSIX thread signals")
def test_worker_directed_sigterm_interrupts_main_arena_future_wait():
    script = r"""
import ctypes
import os
import signal
import sys
import threading
import time

from startrain.arena import ArenaRunner
from startrain.runtime import SignalLatch
from startrain.selfplay import GameVariant

stop = SignalLatch()
stop.install()
worker_ready = threading.Event()
worker_ids = []

class WaitingArena(ArenaRunner):
    def _pair_specifications(self, *_args):
        return []

    def _play_specifications(self, *_args, stop_requested, **_options):
        worker_ids.append((threading.get_native_id(), threading.get_ident()))
        worker_ready.set()
        while not stop_requested():
            time.sleep(0.01)
        return False

subject = object.__new__(WaitingArena)
subject._shared_broker = object()
subject.parallel_variant_groups = 1

def deliver():
    assert worker_ready.wait(2)
    # Allow main to enter Future.result before directing the kernel signal to
    # the worker. Only main may execute the Python SignalLatch handler.
    time.sleep(0.15)
    native_id, python_id = worker_ids[0]
    if sys.platform == "linux":
        libc = ctypes.CDLL(None, use_errno=True)
        libc.tgkill.argtypes = [ctypes.c_int, ctypes.c_int, ctypes.c_int]
        libc.tgkill.restype = ctypes.c_int
        assert libc.tgkill(os.getpid(), native_id, signal.SIGTERM) == 0
    else:
        signal.pthread_kill(python_id, signal.SIGTERM)

sender = threading.Thread(target=deliver)
sender.start()
started = time.monotonic()
completed = subject._play_balanced_groups(
    4, {GameVariant(): [0]}, [], [], progress=None,
    inference_executor=None, stop_requested=stop.is_set,
)
sender.join(timeout=2)
assert not sender.is_alive()
assert not completed
assert stop.signal_number == signal.SIGTERM
assert time.monotonic() - started < 2
print("gracefully stopped after worker-directed SIGTERM")
"""
    result = subprocess.run(
        [sys.executable, "-c", script],
        cwd=Path(__file__).resolve().parents[1],
        capture_output=True,
        text=True,
        timeout=15,
        check=True,
    )
    assert "gracefully stopped" in result.stdout
