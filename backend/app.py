"""Local development entry point and WSGI app object."""
from __future__ import annotations

import atexit

from src.api import create_app
from src.config import Config
from src.logging import get_logger

config = Config()
logger = get_logger("app", config)
app = create_app(config)


@atexit.register
def shutdown_log() -> None:
    logger.info("shutdown")


if __name__ == "__main__":
    app.run(host=config.api_host, port=config.api_port, debug=config.debug)
