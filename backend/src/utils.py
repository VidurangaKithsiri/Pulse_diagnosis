"""Validation and serialization helpers."""
from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from src.config import FEATURE_NAMES


class ValidationError(ValueError):
    """Raised when a client or dataset payload is invalid."""


def normalize_column_name(name: str) -> str:
    """Normalize dataset column names to the backend feature convention."""
    key = name.strip().lower().replace(" ", "_").replace("-", "_")
    aliases = {
        "standard_deviation": "std",
        "sd": "std",
        "minimum": "min",
        "maximum": "max",
        "mean_value": "mean",
        "var": "variance",
    }
    return aliases.get(key, key)


def validate_feature_payload(payload: dict[str, Any]) -> dict[str, float]:
    """Validate the six numerical pulse features submitted to the API."""
    if not isinstance(payload, dict):
        raise ValidationError("JSON body must be an object.")

    validated: dict[str, float] = {}
    missing = [feature for feature in FEATURE_NAMES if feature not in payload]
    if missing:
        raise ValidationError(f"Missing required field(s): {', '.join(missing)}.")

    for feature in FEATURE_NAMES:
        value = payload[feature]
        if isinstance(value, bool):
            raise ValidationError(f"{feature} must be numeric, not boolean.")
        try:
            number = float(value)
        except (TypeError, ValueError) as exc:
            raise ValidationError(f"{feature} must be a valid number.") from exc
        if math.isnan(number) or math.isinf(number):
            raise ValidationError(f"{feature} cannot be NaN or Infinity.")
        if number < 0:
            raise ValidationError(f"{feature} cannot be negative.")
        validated[feature] = number

    if validated["min"] > validated["max"]:
        raise ValidationError("min cannot be greater than max.")
    return validated


def features_to_frame(features: dict[str, float]) -> pd.DataFrame:
    """Convert validated feature values into a single-row DataFrame."""
    return pd.DataFrame([[features[name] for name in FEATURE_NAMES]], columns=FEATURE_NAMES)


def convert_numpy_types(value: Any) -> Any:
    """Convert NumPy/Pandas values into JSON-serializable Python values."""
    if isinstance(value, dict):
        return {key: convert_numpy_types(item) for key, item in value.items()}
    if isinstance(value, list):
        return [convert_numpy_types(item) for item in value]
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, (np.ndarray,)):
        return value.tolist()
    return value


def save_json(path: Path, data: dict[str, Any]) -> None:
    """Write a dictionary as formatted JSON."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(convert_numpy_types(data), indent=2), encoding="utf-8")


def load_json(path: Path) -> dict[str, Any]:
    """Read a JSON object from disk."""
    return json.loads(path.read_text(encoding="utf-8"))
