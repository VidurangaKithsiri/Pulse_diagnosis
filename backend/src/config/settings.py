"""Environment-driven backend configuration."""
from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parents[2]

FEATURE_NAMES = ["mean", "std", "variance", "min", "max", "energy"]
DISPLAY_FEATURE_NAMES = ["Mean", "Standard Deviation", "Variance", "Minimum", "Maximum", "Energy"]
CLASS_NAMES = ["Normal", "Abnormal"]
TARGET_COLUMN_CANDIDATES = ["label", "target", "diagnosis", "status", "class", "condition", "result", "prediction"]


def _path_env(name: str, default: Path) -> Path:
    value = os.getenv(name)
    return Path(value) if value else default


@dataclass(frozen=True)
class Config:
    """Central configuration for local, Render, and future Docker deployments."""

    base_dir: Path = BASE_DIR
    dataset_dir: Path = _path_env("DATASET_DIR", BASE_DIR / "datasets")
    models_dir: Path = _path_env("MODEL_DIR", BASE_DIR / "trained_models")
    logs_dir: Path = _path_env("LOG_DIR", BASE_DIR / "logs")
    database_path: Path = _path_env("DATABASE_PATH", BASE_DIR / "pulse_diagnosis.db")
    dataset_path: Path = _path_env("DATASET_PATH", BASE_DIR / "datasets" / "pulse_features.csv")
    model_path: Path = _path_env("MODEL_PATH", BASE_DIR / "trained_models" / "model.pkl")
    scaler_path: Path = _path_env("SCALER_PATH", BASE_DIR / "trained_models" / "scaler.pkl")
    metadata_path: Path = _path_env("MODEL_METADATA_PATH", BASE_DIR / "trained_models" / "metadata.json")
    metrics_path: Path = _path_env("METRICS_PATH", BASE_DIR / "trained_models" / "metrics.json")
    confusion_matrix_path: Path = BASE_DIR / "trained_models" / "confusion_matrix.png"
    roc_curve_path: Path = BASE_DIR / "trained_models" / "roc_curve.png"
    feature_importance_path: Path = BASE_DIR / "trained_models" / "feature_importance.png"
    log_file: Path = _path_env("LOG_FILE", BASE_DIR / "logs" / "pulse_diagnosis.log")
    model_version: str = os.getenv("MODEL_VERSION", "1.0.0")
    api_version: str = os.getenv("API_VERSION", "v1")
    api_host: str = os.getenv("HOST", os.getenv("PULSE_API_HOST", "0.0.0.0"))
    api_port: int = int(os.getenv("PORT", os.getenv("PULSE_API_PORT", "5000")))
    debug: bool = os.getenv("PULSE_API_DEBUG", "false").lower() == "true"
    test_size: float = float(os.getenv("PULSE_TEST_SIZE", "0.2"))
    random_state: int = int(os.getenv("PULSE_RANDOM_STATE", "42"))
    cv_folds: int = int(os.getenv("PULSE_CV_FOLDS", "5"))
    max_content_length: int = int(os.getenv("MAX_CONTENT_LENGTH", "16384"))
    rate_limit_default: str = os.getenv("RATE_LIMIT_DEFAULT", "200 per day;50 per hour")
    rate_limit_predict: str = os.getenv("RATE_LIMIT_PREDICT", "30 per minute")
    cors_origins: str = os.getenv("CORS_ORIGINS", "*")
    environment: str = os.getenv("APP_ENV", "production")


def ensure_directories(config: Config = Config()) -> None:
    """Create all runtime directories."""
    for directory in (config.dataset_dir, config.models_dir, config.logs_dir, config.database_path.parent):
        directory.mkdir(parents=True, exist_ok=True)
