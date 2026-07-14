"""Comparison plots across detection methods. Saves PNGs into outputs/figures/."""

from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

FIGURES_DIR = Path(__file__).resolve().parents[2] / "outputs" / "figures"


def plot_method_comparison(results: dict, out_name: str = "method_comparison.png") -> Path:
    """results: {method_name: {"accuracy": .., "precision": .., "recall": .., "f1": ..}}"""
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    methods = list(results.keys())
    metric_names = ["accuracy", "precision", "recall", "f1"]
    x = range(len(methods))
    width = 0.2

    fig, ax = plt.subplots(figsize=(max(8, len(methods) * 1.8), 5))
    for i, metric in enumerate(metric_names):
        values = [results[m].get(metric, 0) for m in methods]
        offsets = [xi + (i - 1.5) * width for xi in x]
        ax.bar(offsets, values, width=width, label=metric)

    ax.set_xticks(list(x))
    ax.set_xticklabels(methods, rotation=20, ha="right")
    ax.set_ylim(0, 1)
    ax.set_ylabel("Score")
    ax.set_title("Faithfulness Detection: Method Comparison")
    ax.legend()
    fig.tight_layout()

    out_path = FIGURES_DIR / out_name
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    return out_path


def plot_confusion_matrix(cm, labels=("faithful", "hallucinated"), out_name: str = "confusion_matrix.png") -> Path:
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(4, 4))
    im = ax.imshow(cm, cmap="Blues")
    ax.set_xticks(range(len(labels)))
    ax.set_yticks(range(len(labels)))
    ax.set_xticklabels(labels)
    ax.set_yticklabels(labels)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    for i in range(len(labels)):
        for j in range(len(labels)):
            ax.text(j, i, str(cm[i][j]), ha="center", va="center")
    fig.colorbar(im, ax=ax, fraction=0.046)
    fig.tight_layout()

    out_path = FIGURES_DIR / out_name
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    return out_path
