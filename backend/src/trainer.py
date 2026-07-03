"""Random Forest training pipeline for pulse diagnosis."""
from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV, StratifiedKFold, cross_val_score, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from src.config import Config, FEATURE_NAMES, ensure_directories
from src.evaluation import evaluate_classifier, save_confusion_matrix, save_feature_importance, save_roc_curve
from src.logger import get_logger
from src.preprocess import DatasetPreprocessor
from src.utils import save_json


class PulseModelTrainer:
    """Train, evaluate, and persist the pulse diagnosis classifier."""

    def __init__(self, config: Config = Config()) -> None:
        self.config = config
        self.logger = get_logger(self.__class__.__name__, config)
        ensure_directories(config)

    def train(self) -> dict[str, Any]:
        self.logger.info("Training started")
        prepared = DatasetPreprocessor().prepare(self.config.dataset_path)
        x_train, x_test, y_train, y_test = train_test_split(
            prepared.x,
            prepared.y,
            test_size=self.config.test_size,
            random_state=self.config.random_state,
            stratify=prepared.y,
        )

        pipeline = Pipeline(
            steps=[
                ("scaler", StandardScaler()),
                ("classifier", RandomForestClassifier(random_state=self.config.random_state, class_weight="balanced")),
            ]
        )
        grid = {
            "classifier__n_estimators": [80],
            "classifier__max_depth": [None, 12],
            "classifier__min_samples_split": [2],
            "classifier__min_samples_leaf": [1],
        }
        cv_splits = min(3, self.config.cv_folds, int(prepared.y.value_counts().min()))
        if cv_splits < 2:
            raise ValueError("Not enough samples per class for stratified cross validation.")
        cv = StratifiedKFold(n_splits=cv_splits, shuffle=True, random_state=self.config.random_state)
        search = GridSearchCV(
            pipeline, grid, cv=cv, scoring="f1", n_jobs=1, verbose=0, error_score="raise"
        )
        search.fit(x_train, y_train)
        best_pipeline: Pipeline = search.best_estimator_
        metrics = evaluate_classifier(best_pipeline, x_test, y_test)
        cv_scores = cross_val_score(best_pipeline, prepared.x, prepared.y, cv=cv, scoring="accuracy")

        predictions = best_pipeline.predict(x_test)
        probabilities = best_pipeline.predict_proba(x_test)[:, 1]
        save_confusion_matrix(y_test, predictions, self.config.confusion_matrix_path)
        save_roc_curve(y_test, probabilities, self.config.roc_curve_path)
        save_feature_importance(best_pipeline.named_steps["classifier"], self.config.feature_importance_path)

        scaler = best_pipeline.named_steps["scaler"]
        model = best_pipeline.named_steps["classifier"]
        joblib.dump(model, self.config.model_path)
        joblib.dump(scaler, self.config.scaler_path)

        metadata = {
            "algorithm": "RandomForestClassifier",
            "training_accuracy": metrics["accuracy"],
            "feature_names": FEATURE_NAMES,
            "model_version": self.config.model_version,
            "training_date": datetime.now(timezone.utc).isoformat(),
            "best_params": search.best_params_,
            "cross_validation_accuracy_mean": float(cv_scores.mean()),
            "cross_validation_accuracy_std": float(cv_scores.std()),
            "label_source": prepared.label_source,
            "training_rows": int(len(prepared.x)),
        }
        save_json(self.config.metadata_path, metadata)
        save_json(self.config.metrics_path, {**metrics, "metadata": metadata})
        self.logger.info("Training completed with accuracy %.4f", metrics["accuracy"])
        return {"metrics": metrics, "metadata": metadata}
