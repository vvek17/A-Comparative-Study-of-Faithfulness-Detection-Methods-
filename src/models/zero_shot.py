"""Zero-shot NLI faithfulness detection: document-level and SummaC-Zero sentence-level.

A (knowledge, answer) pair is classified faithful/hallucinated by comparing the
NLI model's contradiction probability against its entailment probability:
contradiction > entailment  => hallucinated (label 1)
otherwise                   => faithful     (label 0)

This mirrors Table 1 / Section 3.1 of docs/report.pdf. Two label conventions
appear in earlier notebook drafts (some cells return "yes"/"no" strings, others
0/1 ints); this module standardizes on the same int convention as
src/data/loader.py: 1 = hallucinated, 0 = faithful.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

import torch
from datasets import Dataset
from transformers import pipeline
from tqdm import tqdm


def _pipeline_device():
    """HF pipeline device arg: CUDA index, 'mps', or -1 (CPU)."""
    if torch.cuda.is_available():
        return 0
    if torch.backends.mps.is_available():
        return "mps"
    return -1


@dataclass
class ZeroShotResult:
    predictions: list[int]
    labels: list[int]


def _label_scores(result_item) -> dict:
    """Normalize a text-classification pipeline output into {label_lower: score}."""
    scores_list = result_item[0] if isinstance(result_item, list) and result_item and isinstance(result_item[0], list) else result_item
    if isinstance(scores_list, dict):
        scores_list = [scores_list]
    return {d["label"].lower(): d["score"] for d in scores_list}


def run_zero_shot_nli(
    ds: Dataset,
    model_name: str = "facebook/bart-large-mnli",
    batch_size: int = 16,
) -> ZeroShotResult:
    """Document-level zero-shot NLI over the whole (knowledge, answer) pair."""
    device = _pipeline_device()
    classifier = pipeline(
        "text-classification", model=model_name, device=device, top_k=None
    )

    pairs = [{"text": ex["knowledge"], "text_pair": ex["answer"]} for ex in ds]
    predictions = []
    for i in tqdm(range(0, len(pairs), batch_size), desc=f"zero-shot ({model_name})"):
        batch = pairs[i : i + batch_size]
        results = classifier(batch, batch_size=batch_size)
        for r in results:
            scores = _label_scores(r)
            pred = 1 if scores.get("contradiction", 0) > scores.get("entailment", 0) else 0
            predictions.append(pred)

    return ZeroShotResult(predictions=predictions, labels=list(ds["labels"]))


_SENTENCE_SPLIT_RE = re.compile(r"(?<=[.!?])\s+")


def split_sentences(text: str) -> list[str]:
    sentences = [s.strip() for s in _SENTENCE_SPLIT_RE.split(text.strip()) if s.strip()]
    return sentences or [text.strip()]


def run_summac_zero(
    ds: Dataset,
    model_name: str = "facebook/bart-large-mnli",
    batch_size: int = 32,
) -> ZeroShotResult:
    """SummaC-Zero-style sentence-level aggregation (Laban et al., 2021).

    For each answer sentence, take the max entailment-vs-contradiction margin
    across all knowledge sentences, then flag the pair as hallucinated if any
    answer sentence is, on balance, contradicted rather than entailed by the
    single best-matching knowledge sentence.
    """
    device = _pipeline_device()
    classifier = pipeline(
        "text-classification", model=model_name, device=device, top_k=None
    )

    predictions = []
    for ex in tqdm(ds, desc=f"summac-zero ({model_name})"):
        knowledge_sents = split_sentences(ex["knowledge"])
        answer_sents = split_sentences(ex["answer"])

        pairs = [
            {"text": k_sent, "text_pair": a_sent}
            for a_sent in answer_sents
            for k_sent in knowledge_sents
        ]
        results = classifier(pairs, batch_size=batch_size)

        # margin[i][j] = entailment(k_j -> a_i) - contradiction(k_j -> a_i)
        n_k = len(knowledge_sents)
        is_hallucinated = False
        for i in range(len(answer_sents)):
            row = results[i * n_k : (i + 1) * n_k]
            margins = [
                _label_scores(r).get("entailment", 0) - _label_scores(r).get("contradiction", 0)
                for r in row
            ]
            best_margin = max(margins)  # most-supportive knowledge sentence
            if best_margin < 0:
                is_hallucinated = True
                break

        predictions.append(1 if is_hallucinated else 0)

    return ZeroShotResult(predictions=predictions, labels=list(ds["labels"]))
