from __future__ import annotations

import pytest

from src.utils import ValidationError, validate_feature_payload


def test_validate_feature_payload_accepts_valid_input():
    payload = {"mean": 72.5, "std": 4.2, "variance": 17.6, "min": 63, "max": 82, "energy": 5200}
    assert validate_feature_payload(payload)["mean"] == 72.5


def test_validate_feature_payload_rejects_missing_field():
    with pytest.raises(ValidationError):
        validate_feature_payload({"mean": 1})


def test_validate_feature_payload_rejects_nan():
    payload = {"mean": float("nan"), "std": 1, "variance": 1, "min": 1, "max": 2, "energy": 3}
    with pytest.raises(ValidationError):
        validate_feature_payload(payload)
