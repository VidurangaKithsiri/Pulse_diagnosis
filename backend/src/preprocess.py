"""Dataset loading, cleaning, validation, and label preparation."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.impute import SimpleImputer

from src.config import CLASS_NAMES, FEATURE_NAMES, TARGET_COLUMN_CANDIDATES
from src.logger import get_logger
from src.utils import normalize_column_name


@dataclass
class PreparedDataset:
    """Cleaned feature matrix and encoded target vector."""

    x: pd.DataFrame
    y: pd.Series
    label_source: str


class DatasetPreprocessor:
    """Prepare pulse feature datasets for traditional machine-learning training."""

    def __init__(self) -> None:
        self.logger = get_logger(self.__class__.__name__)

    def load_dataset(self, dataset_path: Path) -> pd.DataFrame:
        if not dataset_path.exists():
            raise FileNotFoundError(f"Dataset not found: {dataset_path}")
        if dataset_path.suffix.lower() == ".csv":
            dataframe = pd.read_csv(dataset_path)
        elif dataset_path.suffix.lower() in {".xlsx", ".xls"}:
            dataframe = pd.read_excel(dataset_path)
        else:
            raise ValueError("Dataset must be a CSV or Excel file.")
        dataframe = dataframe.rename(columns={column: normalize_column_name(column) for column in dataframe.columns})
        self.logger.info("Loaded dataset with %s rows and %s columns", *dataframe.shape)
        return dataframe

    def prepare(self, dataset_path: Path) -> PreparedDataset:
        dataframe = self.load_dataset(dataset_path)
        dataframe = dataframe.drop_duplicates().reset_index(drop=True)
        missing_features = [feature for feature in FEATURE_NAMES if feature not in dataframe.columns]
        if missing_features:
            raise ValueError(f"Dataset missing required feature columns: {missing_features}")

        features = dataframe[FEATURE_NAMES].apply(pd.to_numeric, errors="coerce")
        features = features.replace([np.inf, -np.inf], np.nan)
        imputer = SimpleImputer(strategy="median")
        features = pd.DataFrame(imputer.fit_transform(features), columns=FEATURE_NAMES)
        features = self._clip_extreme_outliers(features)

        target_column = self._find_target_column(dataframe)
        if target_column:
            labels = self._encode_labels(dataframe[target_column])
            label_source = f"column:{target_column}"
        else:
            labels = self._bootstrap_anomaly_labels(features)
            label_source = "bootstrap:IsolationForest"

        prepared = pd.concat([features, labels.rename("target")], axis=1).dropna()
        x = prepared[FEATURE_NAMES].astype(float)
        y = prepared["target"].astype(int)
        if y.nunique() < 2:
            raise ValueError("Training requires both Normal and Abnormal classes.")
        self.logger.info("Prepared dataset: %s rows, label source=%s", len(x), label_source)
        return PreparedDataset(x=x, y=y, label_source=label_source)

    def _find_target_column(self, dataframe: pd.DataFrame) -> str | None:
        for candidate in TARGET_COLUMN_CANDIDATES:
            if candidate in dataframe.columns:
                return candidate
        return None

    def _encode_labels(self, labels: pd.Series) -> pd.Series:
        normalized = labels.astype(str).str.strip().str.lower()
        mapping = {
            "normal": 0, "healthy": 0, "0": 0, "false": 0, "low": 0,
            "abnormal": 1, "unhealthy": 1, "1": 1, "true": 1, "high": 1, "risk": 1,
        }
        encoded = normalized.map(mapping)
        if encoded.isna().any():
            bad = sorted(normalized[encoded.isna()].unique().tolist())[:10]
            raise ValueError(f"Unsupported label values: {bad}. Expected {CLASS_NAMES}.")
        return encoded.astype(int)

    def _clip_extreme_outliers(self, features: pd.DataFrame) -> pd.DataFrame:
        clipped = features.copy()
        for column in FEATURE_NAMES:
            q1 = clipped[column].quantile(0.25)
            q3 = clipped[column].quantile(0.75)
            iqr = q3 - q1
            if iqr == 0:
                continue
            lower = q1 - 3.0 * iqr
            upper = q3 + 3.0 * iqr
            clipped[column] = clipped[column].clip(lower, upper)
        return clipped

    def _bootstrap_anomaly_labels(self, features: pd.DataFrame) -> pd.Series:
        contamination = min(0.35, max(0.08, 20 / max(len(features), 1)))
        detector = IsolationForest(contamination=contamination, random_state=42)
        predictions = detector.fit_predict(features)
        labels = pd.Series(np.where(predictions == -1, 1, 0), index=features.index)
        if labels.nunique() < 2 and len(labels) >= 4:
            energy_threshold = features["energy"].quantile(0.85)
            labels = (features["energy"] >= energy_threshold).astype(int)
        return labels
