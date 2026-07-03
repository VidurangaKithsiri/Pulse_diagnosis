from __future__ import annotations

from pathlib import Path

import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler

from src.api import create_app
from src.config import Config
from src.utils import save_json


def build_test_config(tmp_path: Path) -> Config:
    x = [[70, 3, 9, 65, 78, 4900], [95, 20, 400, 40, 140, 18000], [72, 4, 16, 64, 82, 5300], [100, 25, 625, 30, 155, 22000]]
    y = [0, 1, 0, 1]
    scaler = StandardScaler().fit(x)
    model = RandomForestClassifier(n_estimators=20, random_state=1).fit(scaler.transform(x), y)
    models_dir = tmp_path / "models"
    logs_dir = tmp_path / "logs"
    models_dir.mkdir()
    logs_dir.mkdir()
    model_path = models_dir / "model.pkl"
    scaler_path = models_dir / "scaler.pkl"
    metadata_path = models_dir / "model_metadata.json"
    joblib.dump(model, model_path)
    joblib.dump(scaler, scaler_path)
    save_json(metadata_path, {"algorithm": "RandomForestClassifier", "training_accuracy": 1.0, "feature_names": ["mean", "std", "variance", "min", "max", "energy"], "model_version": "test", "training_date": "2026-07-02"})
    return Config(base_dir=tmp_path, dataset_dir=tmp_path, models_dir=models_dir, logs_dir=logs_dir, database_path=tmp_path / "test.db", model_path=model_path, scaler_path=scaler_path, metadata_path=metadata_path)


def test_predict_endpoint(tmp_path):
    app = create_app(build_test_config(tmp_path))
    client = app.test_client()
    response = client.post("/predict", json={"mean": 72.5, "std": 4.2, "variance": 17.6, "min": 63, "max": 82, "energy": 5200})
    assert response.status_code == 200
    assert "prediction" in response.get_json()


def test_predict_endpoint_rejects_missing_json(tmp_path):
    app = create_app(build_test_config(tmp_path))
    response = app.test_client().post("/predict", data="not-json")
    assert response.status_code == 400
