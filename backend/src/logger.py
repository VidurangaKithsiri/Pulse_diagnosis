"""Logging utilities for the Pulse Diagnosis backend."""
from __future__ import annotations

import logging
from logging.handlers import RotatingFileHandler

from src.config import Config, ensure_directories


def get_logger(name: str = "pulse_diagnosis", config: Config = Config()) -> logging.Logger:
    """Return a configured rotating-file logger."""
    ensure_directories(config)
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    logger.propagate = False

    if logger.handlers:
        return logger

    formatter = logging.Formatter(
        "%(asctime)s | %(levelname)s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    file_handler = RotatingFileHandler(
        config.log_file, maxBytes=2_000_000, backupCount=5, encoding="utf-8"
    )
    file_handler.setFormatter(formatter)

    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(formatter)

    logger.addHandler(file_handler)
    logger.addHandler(stream_handler)
    return logger
