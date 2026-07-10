from __future__ import annotations

import importlib.util
import os

import pytest
import torch


def pytest_addoption(parser: pytest.Parser) -> None:
    parser.addoption(
        "--require-native",
        action="store_true",
        help="fail collection unless the compiled star_native extension is importable",
    )
    parser.addoption(
        "--run-soak",
        action="store_true",
        default=os.environ.get("STARTRAIN_RUN_SOAK") == "1",
        help="run long target-host reliability tests",
    )


def pytest_configure(config: pytest.Config) -> None:
    if (
        config.getoption("--require-native")
        and importlib.util.find_spec("star_native") is None
    ):
        raise pytest.UsageError(
            "--require-native was set but the star_native extension is not importable"
        )


def pytest_collection_modifyitems(
    config: pytest.Config, items: list[pytest.Item]
) -> None:
    native_available = importlib.util.find_spec("star_native") is not None
    cuda_devices = torch.cuda.device_count() if torch.cuda.is_available() else 0
    run_soak = bool(config.getoption("--run-soak"))

    skips = {
        "native": pytest.mark.skip(reason="requires compiled star_native extension"),
        "cuda": pytest.mark.skip(reason="requires a CUDA-capable GPU"),
        "multi_gpu": pytest.mark.skip(reason="requires at least two CUDA-capable GPUs"),
        "soak": pytest.mark.skip(reason="requires --run-soak or STARTRAIN_RUN_SOAK=1"),
    }
    for item in items:
        if item.get_closest_marker("native") and not native_available:
            item.add_marker(skips["native"])
        if item.get_closest_marker("cuda") and cuda_devices < 1:
            item.add_marker(skips["cuda"])
        if item.get_closest_marker("multi_gpu") and cuda_devices < 2:
            item.add_marker(skips["multi_gpu"])
        if item.get_closest_marker("soak") and not run_soak:
            item.add_marker(skips["soak"])
