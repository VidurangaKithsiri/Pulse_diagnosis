"""Pulse waveform feature extraction."""
from __future__ import annotations

from collections.abc import Sequence

import numpy as np

from src.config import FEATURE_NAMES
from src.utils import ValidationError


class PulseFeatureExtractor:
    """Extract the six required numerical features from pulse waveform samples."""

    def extract(self, waveform: Sequence[float]) -> dict[str, float]:
        values = np.asarray(waveform, dtype=float)
        if values.ndim != 1 or values.size == 0:
            raise ValidationError("waveform must be a non-empty one-dimensional sequence.")
        if np.isnan(values).any() or np.isinf(values).any():
            raise ValidationError("waveform contains NaN or Infinity values.")

        features = {
            "mean": float(np.mean(values)),
            "std": float(np.std(values)),
            "variance": float(np.var(values)),
            "min": float(np.min(values)),
            "max": float(np.max(values)),
            "energy": float(np.sum(np.square(values))),
        }
        return {name: features[name] for name in FEATURE_NAMES}
