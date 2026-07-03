"""JSON and NumPy serialization helpers."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np


def convert_numpy_types(value: Any) -> Any:
    if isinstance(value, dict):
        return {key: convert_numpy_types(item) for key, item in value.items()}
    if isinstance(value, list):
        return [convert_numpy_types(item) for item in value]
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    return value


def save_json(path: Path, data: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(convert_numpy_types(data), indent=2), encoding="utf-8")


def load_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))
