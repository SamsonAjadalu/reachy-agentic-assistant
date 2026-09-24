"""Entry point: ``python -m vision_sidecar``."""

from __future__ import annotations

import argparse
import os

import uvicorn

from app.logging_config import configure_logging
from security.redaction import register_secret
from vision_sidecar.app import create_app
from vision_sidecar.config import SidecarSettings


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Reachy perception sidecar")
    parser.add_argument("--host", default=None)
    parser.add_argument("--port", type=int, default=None)
    parser.add_argument("--reload", action="store_true")
    args = parser.parse_args(argv)

    settings = SidecarSettings()
    if args.host:
        settings.vision_sidecar_host = args.host
    if args.port:
        settings.vision_sidecar_port = args.port
    configure_logging(settings.app_log_level, settings.app_log_format)
    if settings.auth_token:
        register_secret(settings.auth_token)
    settings.ensure_directories()
    settings.apply_hf_env()
    if settings.vision_allow_downloads:
        os.environ.pop("HF_HUB_OFFLINE", None)
        os.environ.pop("TRANSFORMERS_OFFLINE", None)
    else:
        os.environ["HF_HUB_OFFLINE"] = "1"
        os.environ["TRANSFORMERS_OFFLINE"] = "1"
    app = create_app(settings)
    uvicorn.run(
        app,
        host=settings.vision_sidecar_host,
        port=settings.vision_sidecar_port,
        reload=args.reload,
        log_config=None,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
