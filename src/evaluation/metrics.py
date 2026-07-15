"""Shared evaluation metrics for all detection methods.

All methods report on the same binary scale: 1 = hallucinated, 0 = faithful
(see src/data/loader.py for the label-convention note).
"""

from __future__ import annotations

from dataclasses import dataclass, asdict

from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)


@dataclass
class Metrics:
    accuracy: float
    precision: float
    recall: float
    f1: float

    def as_dict(self) -> dict:
        return asdict(self)


def compute_metrics(labels: list[int], predictions: list[int], pos_label: int = 1) -> Metrics:
    return Metrics(
        accuracy=accuracy_score(labels, predictions),
        precision=precision_score(labels, predictions, pos_label=pos_label, zero_division=0),
        recall=recall_score(labels, predictions, pos_label=pos_label, zero_division=0),
        f1=f1_score(labels, predictions, pos_label=pos_label, zero_division=0),
    )


def print_report(labels: list[int], predictions: list[int], title: str = "") -> None:
    if title:
        print("\n" + "=" * 70)
        print(title)
        print("=" * 70)
    m = compute_metrics(labels, predictions)
    print(f"Accuracy:  {m.accuracy:.4f}")
    print(f"Precision: {m.precision:.4f}")
    print(f"Recall:    {m.recall:.4f}")
    print(f"F1:        {m.f1:.4f}")
    print("\nConfusion matrix (rows=true, cols=pred) [0=faithful, 1=hallucinated]:")
    print(confusion_matrix(labels, predictions, labels=[0, 1]))
    print("\n" + classification_report(
        labels, predictions, labels=[0, 1], target_names=["faithful", "hallucinated"], zero_division=0
    ))
