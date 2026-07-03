"""Prediction service layer."""
from __future__ import annotations

from time import perf_counter
from typing import Any

from src.config import CLASS_NAMES, Config, FEATURE_NAMES
from src.ml import ModelLoader, ModelNotLoadedError
from src.utils import features_to_frame, utc_now_iso, validate_feature_payload


class PredictionError(RuntimeError):
    """Raised when prediction execution fails."""


class PredictionService:
    """Validate features, invoke the ML model, and shape stable responses."""

    def __init__(self, model_loader: ModelLoader, config: Config = Config()) -> None:
        self.model_loader = model_loader
        self.config = config

    def predict(self, payload: dict[str, Any]) -> dict[str, Any]:
        started = perf_counter()
        self.model_loader.ensure_loaded()
        try:
            features = validate_feature_payload(payload)
            frame = features_to_frame(features)
            scaled = self.model_loader.scaler.transform(frame[FEATURE_NAMES])
            probabilities = self.model_loader.model.predict_proba(scaled)[0]
            normal_probability = float(probabilities[0])
            abnormal_probability = float(probabilities[1])
            prediction_index = int(probabilities.argmax())
            prediction = CLASS_NAMES[prediction_index]
            confidence = float(max(normal_probability, abnormal_probability))
            processing_time_ms = round((perf_counter() - started) * 1000, 3)
            explanation = self._explain(prediction, abnormal_probability)
            probability = {"Normal": round(normal_probability, 4), "Abnormal": round(abnormal_probability, 4)}
            return {
                "prediction": prediction,
                "confidence": round(confidence, 4),
                "probability": probability,
                "prediction_probability": probability,
                "risk_level": self._risk_level(abnormal_probability),
                "model_version": self.config.model_version,
                "processing_time_ms": processing_time_ms,
                "timestamp": utc_now_iso(),
                "http_status": 200,
                "features": features,
                "explanation": explanation,
            }
        except ModelNotLoadedError:
            raise
        except Exception as exc:
            raise PredictionError(str(exc)) from exc

    @staticmethod
    def _risk_level(abnormal_probability: float) -> str:
        if abnormal_probability < 0.35:
            return "Low"
        if abnormal_probability < 0.70:
            return "Medium"
        return "High"

    @staticmethod
    def _explain(prediction: str, abnormal_probability: float) -> str:
        if prediction == "Normal":
            return "The extracted pulse features are closest to the model's Normal class pattern."
        if abnormal_probability >= 0.70:
            return "The extracted pulse features show a high abnormal probability according to the trained model."
        return "The extracted pulse features show moderate abnormal probability and should be reviewed with context."
