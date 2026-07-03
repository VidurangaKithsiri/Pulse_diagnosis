"""Command-line entry point for model training."""
from __future__ import annotations

from pprint import pprint

from src.trainer import PulseModelTrainer


def main() -> None:
    print("Starting pulse diagnosis model training...", flush=True)
    result = PulseModelTrainer().train()
    print("Training completed successfully.", flush=True)
    pprint(result["metadata"])
    pprint(result["metrics"])


if __name__ == "__main__":
    main()
