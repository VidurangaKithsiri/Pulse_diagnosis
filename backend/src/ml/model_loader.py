"""Thread-safe ML model loading."""
from __future__ import annotations

from threading import Lock
from typing import Any

import joblib

from src.config import Config
from src.logging import get_logger
from src.utils import load_json


class ModelNotLoadedError(RuntimeError):
    """Raised when trained model artifacts are unavailable."""


class ModelLoader:
    """Load and hold model artifacts for the life of the process."""

    def __init__(self, config: Config = Config()) -> None:
        self.config = config
        self.logger = get_logger(self.__class__.__name__, config)
        self.model: Any | None = None
        self.scaler: Any | None = None
        self.metadata: dict[str, Any] = {}
        self.load_error: str | None = None
        self._lock = Lock()
        self.load()

    @property
    def is_loaded(self) -> bool:
        return self.model is not None and self.scaler is not None

    def load(self) -> None:
        with self._lock:
            try:
                if not self.config.model_path.exists():
                    raise ModelNotLoadedError(f"Missing model file: {self.config.model_path}")
                if not self.config.scaler_path.exists():
                    raise ModelNotLoadedError(f"Missing scaler file: {self.config.scaler_path}")
                self.model = joblib.load(self.config.model_path)
                self.scaler = joblib.load(self.config.scaler_path)
                self.metadata = load_json(self.config.metadata_path)
                self.load_error = None
                self.logger.info("model_loaded")
            except Exception as exc:
                self.model = None
                self.scaler = None
                self.metadata = {}
                self.load_error = str(exc)
                self.logger.warning("model_load_failed: %s", exc)

    def ensure_loaded(self) -> None:
        if not self.is_loaded:
            raise ModelNotLoadedError(self.load_error or "Model or scaler is not loaded.")
