"""Command-line entry point for the single-process model service."""

from __future__ import annotations

import argparse
import json
from dataclasses import replace

import uvicorn

from .app import create_app
from .config import load_server_config


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description="Serve an EMA Double *Star model with native Gumbel search"
    )
    parser.add_argument("--config", required=True)
    parser.add_argument("--host")
    parser.add_argument("--port", type=int)
    parser.add_argument(
        "--device",
        help="override the configured PyTorch device, for example mps or cpu",
    )
    parser.add_argument("--check-config", action="store_true")
    parser.add_argument("--log-level", default="info")
    arguments = parser.parse_args(argv)
    config = load_server_config(arguments.config)
    if arguments.device is not None:
        config = replace(config, device=arguments.device)
    if arguments.check_config:
        print(
            json.dumps(
                {
                    "status": "valid",
                    "experiment_config": str(config.experiment_config),
                    "model_manifest": str(config.model_manifest),
                    "device": config.device,
                },
                sort_keys=True,
            )
        )
        return
    app = create_app(config)
    uvicorn.run(
        app,
        host=arguments.host or config.host,
        port=arguments.port or config.port,
        workers=1,
        log_level=arguments.log_level,
        proxy_headers=False,
        server_header=False,
    )


if __name__ == "__main__":
    main()
