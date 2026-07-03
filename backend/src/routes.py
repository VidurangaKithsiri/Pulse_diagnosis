"""Flask route registration."""
from __future__ import annotations

from flask import Blueprint, Response, jsonify, request

from src.config import Config
from src.database import DatabaseError, PredictionDatabase
from src.logger import get_logger
from src.predictor import ModelNotLoadedError, PredictionError, PulsePredictor
from src.utils import ValidationError, validate_feature_payload


def create_routes(predictor: PulsePredictor, database: PredictionDatabase, config: Config) -> Blueprint:
    blueprint = Blueprint("api", __name__)
    logger = get_logger("api", config)

    @blueprint.before_request
    def log_request() -> None:
        logger.info("%s %s from %s", request.method, request.path, request.remote_addr)

    @blueprint.route("/", methods=["GET"])
    def index() -> tuple[Response, int]:
        return jsonify({"service": "Pulse Diagnosis Prediction API", "docs": "/apidocs/", "health": "/health"}), 200

    @blueprint.route("/health", methods=["GET"])
    @blueprint.route("/api/v1/health", methods=["GET"])
    def health() -> tuple[Response, int]:
        try:
            database_status = database.health_check()
        except DatabaseError:
            logger.exception("Database health check failed")
            database_status = "error"
        status = "healthy" if predictor.is_loaded and database_status == "connected" else "degraded"
        return jsonify({
            "status": status,
            "model_loaded": predictor.is_loaded,
            "database": database_status,
            "version": config.model_version,
        }), 200

    @blueprint.route("/model-info", methods=["GET"])
    @blueprint.route("/api/v1/model-info", methods=["GET"])
    def model_info() -> tuple[Response, int]:
        return jsonify(predictor.model_info()), 200

    @blueprint.route("/history", methods=["GET"])
    @blueprint.route("/api/v1/history", methods=["GET"])
    def history() -> tuple[Response, int]:
        try:
            return jsonify({"history": database.get_history()}), 200
        except DatabaseError as exc:
            logger.exception("Database failure while reading history")
            return jsonify({"error": "database_failure", "message": str(exc)}), 500

    @blueprint.route("/history", methods=["DELETE"])
    @blueprint.route("/api/v1/history", methods=["DELETE"])
    def delete_history() -> tuple[Response, int]:
        try:
            deleted = database.clear_history()
            return jsonify({"deleted": deleted}), 200
        except DatabaseError as exc:
            logger.exception("Database failure while deleting history")
            return jsonify({"error": "database_failure", "message": str(exc)}), 500

    @blueprint.route("/predict", methods=["POST"])
    @blueprint.route("/api/v1/predict", methods=["POST"])
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
          200:
            description: Prediction result
          400:
            description: Invalid input
          503:
            description: Model unavailable
        """
        if not request.is_json:
            return jsonify({"error": "missing_json", "message": "Request body must be JSON."}), 400
        payload = request.get_json(silent=True)
        if payload is None:
            return jsonify({"error": "invalid_json", "message": "Malformed JSON body."}), 400
        try:
            features = validate_feature_payload(payload)
            result = predictor.predict(payload)
            record_id = database.insert_prediction(features, result)
            logger.info("Prediction %s saved with id=%s", result["prediction"], record_id)
            return jsonify(result), 200
        except ValidationError as exc:
            return jsonify({"error": "invalid_request", "message": str(exc)}), 400
        except ModelNotLoadedError as exc:
            logger.warning("Prediction blocked because model is unavailable: %s", exc)
            return jsonify({"error": "model_unavailable", "message": str(exc)}), 503
        except PredictionError as exc:
            logger.exception("Prediction execution failed")
            return jsonify({"error": "prediction_failed", "message": str(exc)}), 500
        except DatabaseError as exc:
            logger.exception("Database failure while saving prediction")
            return jsonify({"error": "database_failure", "message": str(exc)}), 500

    return blueprint
