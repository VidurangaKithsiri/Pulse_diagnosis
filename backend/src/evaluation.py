"""Model evaluation and visualization."""
from __future__ import annotations

from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)

from src.config import CLASS_NAMES, FEATURE_NAMES


def evaluate_classifier(model: Any, x_test: pd.DataFrame, y_test: pd.Series) -> dict[str, Any]:
    predictions = model.predict(x_test)
    probabilities = model.predict_proba(x_test)[:, 1]
    return {
        "accuracy": accuracy_score(y_test, predictions),
        "precision": precision_score(y_test, predictions, zero_division=0),
        "recall": recall_score(y_test, predictions, zero_division=0),
        "f1_score": f1_score(y_test, predictions, zero_division=0),
        "auc": roc_auc_score(y_test, probabilities) if y_test.nunique() == 2 else 0.0,
        "classification_report": classification_report(
            y_test, predictions, target_names=CLASS_NAMES, zero_division=0, output_dict=True
        ),
        "confusion_matrix": confusion_matrix(y_test, predictions).tolist(),
    }


def save_confusion_matrix(y_true: pd.Series, y_pred: pd.Series, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    matrix = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(6, 5))
    sns.heatmap(matrix, annot=True, fmt="d", cmap="Blues", xticklabels=CLASS_NAMES, yticklabels=CLASS_NAMES)
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()


def save_roc_curve(y_true: pd.Series, probabilities: pd.Series, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fpr, tpr, _ = roc_curve(y_true, probabilities)
    auc = roc_auc_score(y_true, probabilities) if y_true.nunique() == 2 else 0.0
    plt.figure(figsize=(6, 5))
    plt.plot(fpr, tpr, label=f"AUC = {auc:.3f}")
    plt.plot([0, 1], [0, 1], linestyle="--", color="gray")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.legend(loc="lower right")
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()


def save_feature_importance(model: Any, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    importances = getattr(model, "feature_importances_", None)
    if importances is None:
        return
    frame = pd.DataFrame({"feature": FEATURE_NAMES, "importance": importances}).sort_values(
        "importance", ascending=False
    )
    plt.figure(figsize=(7, 5))
    sns.barplot(data=frame, x="importance", y="feature", color="#2f6f8f")
    plt.xlabel("Importance")
    plt.ylabel("Feature")
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()
