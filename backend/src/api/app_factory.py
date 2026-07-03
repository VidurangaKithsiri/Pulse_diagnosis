"""Flask application factory."""
from __future__ import annotations

from flask import Flask, jsonify
from flask_cors import CORS
from flasgger import Swagger
from flask_limiter import Limiter
from flask_limiter.errors import RateLimitExceeded
from flask_limiter.util import get_remote_address
from werkzeug.exceptions import RequestEntityTooLarge

from src.api.routes import create_api_blueprint
from src.config import Config, ensure_directories
from src.database import DatabaseError, PredictionRepository
from src.logging import get_logger
from src.middleware import register_security_headers
from src.ml import ModelLoader
from src.services import PredictionService
from src.utils import error_response
from src.validators import ValidationError


def create_app(config: Config = Config()) -> Flask:
    ensure_directories(config)
    logger = get_logger("app", config)
    app = Flask(__name__)
    app.config["MAX_CONTENT_LENGTH"] = config.max_content_length
    app.config["SWAGGER"] = {"title": "Pulse Diagnosis Prediction API", "uiversion": 3}

    origins = "*" if config.cors_origins == "*" else [item.strip() for item in config.cors_origins.split(",")]
    CORS(app, resources={r"/*": {"origins": origins}})
    Swagger(app)
    register_security_headers(app)

    limiter = Limiter(
        key_func=get_remote_address,
        default_limits=[limit.strip() for limit in config.rate_limit_default.split(";") if limit.strip()],
        storage_uri="memory://",
    )
    limiter.init_app(app)

    model_loader = ModelLoader(config)
    prediction_service = PredictionService(model_loader, config)
    repository = PredictionRepository(config)
    app.register_blueprint(create_api_blueprint(prediction_service, model_loader, repository, config, limiter))

    @app.errorhandler(404)
    def not_found(error):  # type: ignore[no-untyped-def]
        return error_response("not_found", "Invalid endpoint.", 404)

    @app.errorhandler(405)
    def method_not_allowed(error):  # type: ignore[no-untyped-def]
        return error_response("method_not_allowed", "HTTP method is not allowed.", 405)

    @app.errorhandler(ValidationError)
    def validation_error(error):  # type: ignore[no-untyped-def]
        return error_response("invalid_request", str(error), 400)

    @app.errorhandler(RequestEntityTooLarge)
    def request_too_large(error):  # type: ignore[no-untyped-def]
        return error_response("request_too_large", "Request body is too large.", 413)

    @app.errorhandler(RateLimitExceeded)
    def rate_limited(error):  # type: ignore[no-untyped-def]
        return error_response("rate_limit_exceeded", str(error.description), 429)

    @app.errorhandler(DatabaseError)
    def database_error(error):  # type: ignore[no-untyped-def]
        logger.exception("database_error")
        return error_response("database_failure", str(error), 500)

    @app.errorhandler(Exception)
    def unhandled(error):  # type: ignore[no-untyped-def]
        logger.exception("unhandled_error")
        return error_response("internal_server_error", "Unexpected server error.", 500)

    logger.info("startup_complete port=%s environment=%s", config.api_port, config.environment)
    return app
