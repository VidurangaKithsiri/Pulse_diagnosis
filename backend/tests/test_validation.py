from __future__ import annotations

import pytest

from src.validators import ValidationError, validate_feature_payload


def test_valid_payload_is_ordered_and_numeric():
    payload = {"mean": 72.5, "std": 4.2, "variance": 17.6, "min": 63, "max": 82, "energy": 5200}
    assert validate_feature_payload(payload)["energy"] == 5200.0


def test_rejects_missing_values():
    with pytest.raises(ValidationError):
        validate_feature_payload({"mean": 1})


def test_rejects_extra_fields():
    with pytest.raises(ValidationError):
        validate_feature_payload({"mean": 1, "std": 1, "variance": 1, "min": 1, "max": 2, "energy": 3, "extra": 4})
