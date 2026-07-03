"""REST API routes."""
from __future__ import annotations

from collections.abc import Callable
from typing import Any

from flask import Blueprint, Response, jsonify, redirect, request

from src.config import Config
from src.database import DatabaseError, PredictionRepository
from src.logging import get_logger
from src.ml import ModelLoader, ModelNotLoadedError
from src.services import PredictionError, PredictionService
from src.utils import error_response, load_json, validate_feature_payload
from src.validators import ValidationError


def create_api_blueprint(
    prediction_service: PredictionService,
    model_loader: ModelLoader,
    repository: PredictionRepository,
    config: Config,
    limiter: Any | None = None,
) -> Blueprint:
    blueprint = Blueprint("api", __name__)
    logger = get_logger("api", config)

    def apply_predict_limit(view: Callable[..., Any]) -> Callable[..., Any]:
        return limiter.limit(config.rate_limit_predict)(view) if limiter is not None else view

    @blueprint.before_request
    def log_request() -> None:
        request.start_time_marker = None
        logger.info("request method=%s path=%s ip=%s", request.method, request.path, request.remote_addr)

    @blueprint.route("/", methods=["GET"])
    def index() -> tuple[Response, int]:
        return jsonify({
            "service": "Pulse Diagnosis Prediction API",
            "version": config.model_version,
            "docs": "/docs",
            "swagger": "/apidocs/",
            "health": "/health",
        }), 200

    @blueprint.route("/docs", methods=["GET"])
    @blueprint.route("/swagger", methods=["GET"])
    def docs():  # type: ignore[no-untyped-def]
        return redirect("/apidocs/")

    @blueprint.route("/version", methods=["GET"])
    def version() -> tuple[Response, int]:
        return jsonify({"api_version": config.api_version, "model_version": config.model_version}), 200

    @blueprint.route("/health", methods=["GET"])
    @blueprint.route("/api/v1/health", methods=["GET"])
    def health() -> tuple[Response, int]:
        try:
            database_status = repository.health_check()
        except DatabaseError:
            logger.exception("database_health_failed")
            database_status = "error"
        status = "healthy" if model_loader.is_loaded and database_status == "connected" else "degraded"
        return jsonify({
            "status": status,
            "model_loaded": model_loader.is_loaded,
            "database": database_status,
            "version": config.model_version,
        }), 200

    @blueprint.route("/model-info", methods=["GET"])
    @blueprint.route("/api/v1/model-info", methods=["GET"])
    def model_info() -> tuple[Response, int]:
        metadata = model_loader.metadata
        return jsonify({
            "algorithm": metadata.get("algorithm", "RandomForestClassifier"),
            "training_accuracy": metadata.get("training_accuracy"),
            "feature_names": metadata.get("feature_names"),
            "model_version": config.model_version,
            "training_date": metadata.get("training_date"),
            "model_loaded": model_loader.is_loaded,
            "load_error": model_loader.load_error,
        }), 200

    @blueprint.route("/metrics", methods=["GET"])
    def metrics() -> tuple[Response, int]:
        return jsonify(load_json(config.metrics_path)), 200

    @blueprint.route("/history", methods=["GET"])
    @blueprint.route("/api/v1/history", methods=["GET"])
    def history() -> tuple[Response, int]:
        try:
            return jsonify({"history": repository.get_history()}), 200
        except DatabaseError as exc:
            logger.exception("history_read_failed")
            return error_response("database_failure", str(exc), 500)

    @blueprint.route("/history", methods=["DELETE"])
    @blueprint.route("/api/v1/history", methods=["DELETE"])
    def delete_history() -> tuple[Response, int]:
        try:
            return jsonify({"deleted": repository.clear_history()}), 200
        except DatabaseError as exc:
            logger.exception("history_delete_failed")
            return error_response("database_failure", str(exc), 500)

    @blueprint.route("/predict", methods=["POST"])
    @blueprint.route("/api/v1/predict", methods=["POST"])
    @apply_predict_limit
    def predict() -> tuple[Response, int]:
        """Predict pulse status.
        ---
        consumes:
          - application/json
        parameters:
          - in: body
            name: body
            required: true
            schema:
              type: object
              required: [mean, std, variance, min, max, energy]
              properties:
                mean: {type: number, example: 72.5}
                std: {type: number, example: 4.2}
                variance: {type: number, example: 17.6}
                min: {type: number, example: 63}
                max: {type: number, example: 82}
                energy: {type: number, example: 5200}
        responses:
          200: {description: Prediction result}
          400: {description: Invalid input}
          503: {description: Model unavailable}
        """
        if not request.is_json:
            return error_response("missing_json", "Request body must be JSON.", 400)
        payload = request.get_json(silent=True)
        if payload is None:
            return error_response("invalid_json", "Malformed JSON body.", 400)
        try:
            features = validate_feature_payload(payload)
            result = prediction_service.predict(payload)
            repository.save_prediction(features, result, request.remote_addr, config.api_version)
            public_result = {key: value for key, value in result.items() if key != "features"}
            logger.info("prediction prediction=%s confidence=%s time_ms=%s", result["prediction"], result["confidence"], result["processing_time_ms"])
            return jsonify(public_result), 200
        except ValidationError as exc:
            return error_response("invalid_request", str(exc), 400)
        except ModelNotLoadedError as exc:
            logger.warning("model_unavailable: %s", exc)
            return error_response("model_unavailable", str(exc), 503)
        except PredictionError as exc:
            logger.exception("prediction_failed")
            return error_response("prediction_failed", str(exc), 500)
        except DatabaseError as exc:
            logger.exception("database_write_failed")
            return error_response("database_failure", str(exc), 500)

    return blueprint
