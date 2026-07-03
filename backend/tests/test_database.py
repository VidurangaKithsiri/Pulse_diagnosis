from __future__ import annotations

from src.config import Config
from src.database import PredictionDatabase


def test_database_insert_get_and_clear(tmp_path):
    config = Config(base_dir=tmp_path, dataset_dir=tmp_path, models_dir=tmp_path, logs_dir=tmp_path, database_path=tmp_path / "test.db")
    database = PredictionDatabase(config)
    features = {"mean": 1.0, "std": 1.0, "variance": 1.0, "min": 1.0, "max": 2.0, "energy": 3.0}
    result = {"prediction": "Normal", "confidence": 0.9, "risk_level": "Low"}
    assert database.insert_prediction(features, result) == 1
    assert len(database.get_history()) == 1
    assert database.clear_history() == 1
