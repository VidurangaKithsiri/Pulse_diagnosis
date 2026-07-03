"""Compatibility prediction module."""
from src.ml import ModelNotLoadedError
from src.services import PredictionError, PredictionService
from src.config import Config
from src.ml import ModelLoader


class PulsePredictor(PredictionService):
    """Backward-compatible predictor facade."""

    def __init__(self, config: Config = Config()) -> None:
        self.loader = ModelLoader(config)
        super().__init__(self.loader, config)

    @property
    def model(self):
        return self.loader.model

    @property
    def scaler(self):
        return self.loader.scaler

    @property
    def is_loaded(self) -> bool:
        return self.loader.is_loaded

    @property
    def metadata(self):
        return self.loader.metadata

    @property
    def load_error(self):
        return self.loader.load_error

    def model_info(self):
        return {
            "algorithm": self.metadata.get("algorithm", "RandomForestClassifier"),
            "training_accuracy": self.metadata.get("training_accuracy"),
            "feature_names": self.metadata.get("feature_names"),
            "model_version": self.config.model_version,
            "training_date": self.metadata.get("training_date"),
            "model_loaded": self.is_loaded,
            "load_error": self.load_error,
        }
