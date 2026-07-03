"""Flask application factory."""
from __future__ import annotations

from flask import Flask, jsonify
from flask_cors import CORS
from flasgger import Swagger
from flask_limiter import Limiter
from flask_limiter.errors import RateLimitExceeded
from flask_limiter.util import get_remote_address
from werkzeug.exceptions import RequestEntityTooLarge

from src.config import Config, ensure_directories
from src.database import DatabaseError, PredictionDatabase
from src.logger import get_logger
from src.predictor import PulsePredictor
from src.routes import create_routes
from src.utils import ValidationError


def create_app(config: Config = Config()) -> Flask:
    ensure_directories(config)
    logger = get_logger("app", config)
    app = Flask(__name__)
    app.config["MAX_CONTENT_LENGTH"] = config.max_content_length
    app.config["SWAGGER"] = {"title": "Pulse Diagnosis Prediction API", "uiversion": 3}

    origins = "*" if config.cors_origins == "*" else [item.strip() for item in config.cors_origins.split(",")]
    CORS(app, resources={r"/*": {"origins": origins}})
    Swagger(app)

    limiter = Limiter(
        key_func=get_remote_address,
        default_limits=[limit.strip() for limit in config.rate_limit_default.split(";") if limit.strip()],
        storage_uri="memory://",
    )
    limiter.init_app(app)

    predictor = PulsePredictor(config)
    database = PredictionDatabase(config)
    blueprint = create_routes(predictor, database, config)
    blueprint.view_functions["predict"] = limiter.limit(config.rate_limit_predict)(blueprint.view_functions["predict"])
    app.register_blueprint(blueprint)

    @app.errorhandler(404)
    def not_found(error):  # type: ignore[no-untyped-def]
        return jsonify({"error": "not_found", "message": "Invalid endpoint."}), 404

    @app.errorhandler(405)
    def method_not_allowed(error):  # type: ignore[no-untyped-def]
        return jsonify({"error": "method_not_allowed", "message": "HTTP method is not allowed."}), 405

    @app.errorhandler(ValidationError)
    def validation_error(error):  # type: ignore[no-untyped-def]
        return jsonify({"error": "invalid_request", "message": str(error)}), 400

    @app.errorhandler(RequestEntityTooLarge)
    def request_too_large(error):  # type: ignore[no-untyped-def]
        return jsonify({"error": "request_too_large", "message": "Request body is too large."}), 413

    @app.errorhandler(RateLimitExceeded)
    def rate_limited(error):  # type: ignore[no-untyped-def]
        return jsonify({"error": "rate_limit_exceeded", "message": str(error.description)}), 429

    @app.errorhandler(DatabaseError)
    def database_error(error):  # type: ignore[no-untyped-def]
        logger.exception("Database error")
        return jsonify({"error": "database_failure", "message": str(error)}), 500

    @app.errorhandler(Exception)
    def unhandled(error):  # type: ignore[no-untyped-def]
        logger.exception("Unhandled server error")
        return jsonify({"error": "internal_server_error", "message": "Unexpected server error."}), 500

    logger.info("Application startup complete on port %s", config.api_port)
    return app
