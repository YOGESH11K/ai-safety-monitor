"""CLI entry point.

Examples:
    python run.py                          # webcam 0, stub mode if no model
    python run.py --test                   # synthetic feed + demo mode
    python run.py --source path/video.mp4
    python run.py --source rtsp://...
    python run.py --config config.yaml --host 0.0.0.0 --port 8000
"""
from __future__ import annotations

import argparse
import logging
import os
import sys
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
logger = logging.getLogger("run")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Real-time AI safety monitoring (documentation only).")
    parser.add_argument("--config", default="config.yaml", help="Path to config.yaml")
    parser.add_argument("--source", default=None, help="Camera source: index, video path, rtsp:// URL, or 'synthetic'")
    parser.add_argument("--test", action="store_true", help="Use the synthetic test feed (no camera needed)")
    parser.add_argument("--host", default=None, help="API host (default from config)")
    parser.add_argument("--port", type=int, default=None, help="API port (default from config)")
    parser.add_argument("--db", default="data/safety.db", help="SQLite database path")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)

    if args.source is not None:
        os.environ["CAMERA_SOURCE"] = str(args.source)
    if args.test:
        os.environ["CAMERA_SOURCE"] = "synthetic"

    from app.config import AppConfig

    logger.info("Loading config from %s", args.config)
    config = AppConfig.load(args.config)

    if args.host:
        config.api.host = args.host
    else:
        config.api.host = os.environ.get("API_HOST", config.api.host)
    if args.port:
        config.api.port = args.port

    if config.api.require_auth_token and not config.api.auth_token:
        logger.warning(
            "api.require_auth_token is true but no auth_token set; the API is open."
            " Set API_AUTH_TOKEN in the environment or .env."
        )

    from app.main import create_app
    import uvicorn

    app = create_app(config, db_path=args.db)
    logger.info(
        "Dashboard: http://%s:%s  |  API: http://%s:%s/api/status",
        config.api.host, config.api.port, config.api.host, config.api.port,
    )
    if config.detection.detector_mode == "yolo" or config.model_path.exists():
        logger.info("Weapon model: %s", config.model_path)
    else:
        logger.info("No weapon model at %s -> running in STUB demo mode.", config.model_path)
    uvicorn.run(app, host=config.api.host, port=config.api.port, log_level=config.log_level.lower())
    return 0


if __name__ == "__main__":
    sys.exit(main())