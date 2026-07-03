"""WSGI entry point for local Flask and Render Gunicorn deployment."""
from __future__ import annotations

import atexit

from src.api import create_app
from src.config import Config
from src.logger import get_logger

config = Config()
logger = get_logger("app", config)
app = create_app(config)


@atexit.register
def shutdown_log() -> None:
    logger.info("Application shutdown")


if __name__ == "__main__":
    app.run(host=config.api_host, port=config.api_port, debug=config.debug)
