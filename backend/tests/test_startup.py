from __future__ import annotations

from src.api import create_app
from src.config import Config


def test_app_factory_starts_with_missing_model(tmp_path):
    config = Config(
        base_dir=tmp_path,
        dataset_dir=tmp_path / "datasets",
        models_dir=tmp_path / "trained_models",
        logs_dir=tmp_path / "logs",
        database_path=tmp_path / "test.db",
        model_path=tmp_path / "trained_models" / "missing_model.pkl",
        scaler_path=tmp_path / "trained_models" / "missing_scaler.pkl",
        metadata_path=tmp_path / "trained_models" / "metadata.json",
        metrics_path=tmp_path / "trained_models" / "metrics.json",
    )
    app = create_app(config)
    response = app.test_client().get("/health")
    assert response.status_code == 200
    assert response.get_json()["model_loaded"] is False
