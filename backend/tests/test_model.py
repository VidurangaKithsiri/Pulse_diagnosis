from __future__ import annotations

import pandas as pd

from src.preprocess import DatasetPreprocessor


def test_preprocessor_uses_label_column(tmp_path):
    path = tmp_path / "data.csv"
    pd.DataFrame({
        "mean": [1, 2, 100, 110],
        "std": [1, 1, 10, 11],
        "variance": [1, 1, 100, 121],
        "min": [1, 1, 80, 90],
        "max": [3, 4, 130, 140],
        "energy": [10, 12, 10000, 12000],
        "diagnosis": ["Normal", "Normal", "Abnormal", "Abnormal"],
    }).to_csv(path, index=False)
    prepared = DatasetPreprocessor().prepare(path)
    assert prepared.x.shape == (4, 6)
    assert sorted(prepared.y.unique().tolist()) == [0, 1]
