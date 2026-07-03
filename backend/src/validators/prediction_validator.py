"""Prediction request validation and feature ordering."""
from __future__ import annotations

import math
from typing import Any

import pandas as pd

from src.config import FEATURE_NAMES


class ValidationError(ValueError):
    """Raised when a request or dataset payload is invalid."""


def normalize_column_name(name: str) -> str:
    key = name.strip().lower().replace(" ", "_").replace("-", "_")
    aliases = {"standard_deviation": "std", "sd": "std", "minimum": "min", "maximum": "max", "mean_value": "mean", "var": "variance"}
    return aliases.get(key, key)


def validate_feature_payload(payload: dict[str, Any]) -> dict[str, float]:
    if not isinstance(payload, dict):
        raise ValidationError("JSON body must be an object.")
    if not payload:
        raise ValidationError("JSON body cannot be empty.")
    missing = [feature for feature in FEATURE_NAMES if feature not in payload]
    if missing:
        raise ValidationError(f"Missing required field(s): {', '.join(missing)}.")
    extra = sorted(set(payload.keys()) - set(FEATURE_NAMES))
    if extra:
        raise ValidationError(f"Unsupported field(s): {', '.join(extra)}.")
    validated: dict[str, float] = {}
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
    return pd.DataFrame([[features[name] for name in FEATURE_NAMES]], columns=FEATURE_NAMES)
