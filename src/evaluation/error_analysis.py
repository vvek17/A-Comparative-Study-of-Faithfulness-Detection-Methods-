"""Lightweight error categorization for qualitative analysis of a method's mistakes."""

from __future__ import annotations

import re
from collections import Counter

from datasets import Dataset

_NUMBER_RE = re.compile(r"\b\d[\d,.]*\b")
_DATE_HINT_RE = re.compile(
    r"\b(19|20)\d{2}\b|\b(january|february|march|april|may|june|july|august|"
    r"september|october|november|december)\b",
    re.IGNORECASE,
)


def categorize_example(answer: str) -> str:
    """Heuristic single-label categorization of the kind of claim an answer makes.

    Not mutually-exclusive in reality, but a single dominant category is
    assigned (first match wins) so error counts are easy to read as a table.
    """
    if _DATE_HINT_RE.search(answer):
        return "date"
    if _NUMBER_RE.search(answer):
        return "numerical"
    if answer.strip() and answer.strip()[0].isupper() and len(answer.split()) <= 4:
        return "named_entity"
    return "reasoning_based"


def error_breakdown(ds: Dataset, predictions: list[int]) -> dict:
    """Count errors (pred != true label) per heuristic category.

    Returns {category: {"errors": n, "total": n, "error_rate": float}}.
    """
    totals = Counter()
    errors = Counter()
    for ex, pred in zip(ds, predictions):
        cat = categorize_example(ex["answer"])
        totals[cat] += 1
        if pred != ex["labels"]:
            errors[cat] += 1

    return {
        cat: {
            "errors": errors.get(cat, 0),
            "total": totals[cat],
            "error_rate": errors.get(cat, 0) / totals[cat] if totals[cat] else 0.0,
        }
        for cat in totals
    }
