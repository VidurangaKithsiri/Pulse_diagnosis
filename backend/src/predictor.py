"""Prediction engine for the trained pulse diagnosis model."""
from __future__ import annotations

from typing import Any

import joblib

from src.config import CLASS_NAMES, Config, FEATURE_NAMES
from src.logger import get_logger
from src.utils import features_to_frame, load_json, validate_feature_payload


class ModelNotLoadedError(RuntimeError):
    """Raised when trained model artifacts are unavailable."""


class PredictionError(RuntimeError):
    """Raised when prediction execution fails."""


class PulsePredictor:
    """Load model artifacts once and serve prediction requests."""

    def __init__(self, config: Config = Config()) -> None:
        self.config = config
        self.logger = get_logger(self.__class__.__name__, config)
        self.model: Any | None = None
        self.scaler: Any | None = None
        self.metadata: dict[str, Any] = {}
        self.load_error: str | None = None
        self.load_artifacts()

    @property
    def is_loaded(self) -> bool:
        return self.model is not None and self.scaler is not None

    def load_artifacts(self) -> None:
        """Load model and scaler once; keep startup alive if artifacts are absent."""
        try:
            if not self.config.model_path.exists():
                raise ModelNotLoadedError(f"Missing model file: {self.config.model_path}")
            if not self.config.scaler_path.exists():
                raise ModelNotLoadedError(f"Missing scaler file: {self.config.scaler_path}")
            self.model = joblib.load(self.config.model_path)
            self.scaler = joblib.load(self.config.scaler_path)
            self.metadata = load_json(self.config.metadata_path) if self.config.metadata_path.exists() else {}
            self.load_error = None
            self.logger.info("Model and scaler loaded from %s", self.config.models_dir)
        except Exception as exc:
            self.model = None
            self.scaler = None
            self.load_error = str(exc)
            self.logger.warning("Model artifacts are not ready: %s", exc)

    def ensure_loaded(self) -> None:
        if not self.is_loaded:
            message = self.load_error or "Model or scaler is not loaded."
            raise ModelNotLoadedError(message)

    def predict(self, payload: dict[str, Any]) -> dict[str, Any]:
        self.ensure_loaded()
        try:
            features = validate_feature_payload(payload)
            frame = features_to_frame(features)
            scaled = self.scaler.transform(frame[FEATURE_NAMES])
            probabilities = self.model.predict_proba(scaled)[0]
            normal_probability = float(probabilities[0])
            abnormal_probability = float(probabilities[1])
            prediction_index = int(probabilities.argmax())
            prediction = CLASS_NAMES[prediction_index]
            confidence = float(max(normal_probability, abnormal_probability))
            return {
                "prediction": prediction,
                "confidence": round(confidence, 4),
                "risk_level": self._risk_level(abnormal_probability),
                "prediction_probability": {
                    "Normal": round(normal_probability, 4),
                    "Abnormal": round(abnormal_probability, 4),
                },
            }
        except ModelNotLoadedError:
            raise
        except Exception as exc:
            raise PredictionError(str(exc)) from exc

    def model_info(self) -> dict[str, Any]:
        return {
            "algorithm": self.metadata.get("algorithm", "RandomForestClassifier"),
            "training_accuracy": self.metadata.get("training_accuracy"),
            "feature_names": self.metadata.get("feature_names", FEATURE_NAMES),
            "model_version": self.metadata.get("model_version", self.config.model_version),
            "training_date": self.metadata.get("training_date"),
            "model_loaded": self.is_loaded,
            "load_error": self.load_error,
        }

    @staticmethod
    def _risk_level(abnormal_probability: float) -> str:
        if abnormal_probability < 0.35:
            return "Low"
        if abnormal_probability < 0.70:
            return "Medium"
        return "High"
