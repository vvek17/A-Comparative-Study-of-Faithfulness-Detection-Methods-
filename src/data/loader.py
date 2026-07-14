"""Dataset loading for HaluEval and FEVER.

Label convention (verified against pminervini/HaluEval qa_samples):
    hallucination == "yes"  -> the `answer` field is NOT supported by `knowledge` (hallucinated)
    hallucination == "no"   -> the `answer` field IS supported by `knowledge` (faithful)

This is the opposite polarity of what earlier drafts of the written report described
("yes" = faithful) -- that prose was inverted relative to the actual dataset and the
labeling used throughout this codebase. All code here treats hallucination="yes" as
the positive ("hallucinated") class, label = 1.
"""

from __future__ import annotations

from datasets import load_dataset, Dataset, DatasetDict

HALUEVAL_REPO = "pminervini/HaluEval"
HALUEVAL_CONFIG = "qa_samples"


def load_halueval_qa(num_samples: int | None = 10_000, seed: int = 42) -> Dataset:
    """Load the HaluEval QA subset with an integer `labels` column added.

    labels: 1 = hallucinated (unsupported by knowledge), 0 = faithful.
    """
    ds = load_dataset(HALUEVAL_REPO, HALUEVAL_CONFIG, split="data")
    ds = ds.map(lambda ex: {"labels": 1 if ex["hallucination"] == "yes" else 0})
    if num_samples is not None and num_samples < len(ds):
        ds = ds.shuffle(seed=seed).select(range(num_samples))
    return ds


def split_halueval(ds: Dataset, scheme: str = "70/15/15", seed: int = 42) -> DatasetDict:
    """Split into train/validation/test.

    scheme: "80/20" -> train/test only (validation == test, kept separate is not
            meaningful for that scheme so only 'train' and 'test' keys are returned).
            "70/15/15" -> train/validation/test.
    """
    if scheme == "80/20":
        split = ds.train_test_split(test_size=0.2, seed=seed)
        return DatasetDict(train=split["train"], test=split["test"])
    if scheme == "70/15/15":
        split1 = ds.train_test_split(test_size=0.3, seed=seed)
        split2 = split1["test"].train_test_split(test_size=0.5, seed=seed)
        return DatasetDict(
            train=split1["train"], validation=split2["train"], test=split2["test"]
        )
    raise ValueError(f"Unknown split scheme: {scheme!r}")


def load_fever(split: str = "dev", num_samples: int | None = 2_000, seed: int = 42) -> Dataset:
    """Load FEVER for cross-domain generalization testing.

    Uses pietrolesci/nli_fever, a pre-converted NLI-formatted mirror of FEVER
    (the original `fever` dataset repo relies on a loading script that current
    `datasets` versions refuse to execute). Its `premise` column is actually the
    claim and `hypothesis` is the Wikipedia evidence sentence(s) -- this loader
    renames them into the same (knowledge, question, answer) shape used by
    HaluEval, with the claim treated as the "answer" to be checked against the
    evidence "knowledge":
        fever_gold_label == "REFUTES"         -> 1 (hallucinated / unfaithful)
        fever_gold_label == "SUPPORTS"        -> 0 (faithful)
        fever_gold_label == "NOT ENOUGH INFO" -> dropped (no HaluEval analogue --
                                                   neither faithful nor contradicted)
    """
    ds = load_dataset("pietrolesci/nli_fever", split=split)
    ds = ds.filter(lambda ex: ex["fever_gold_label"] in ("SUPPORTS", "REFUTES"))
    ds = ds.map(
        lambda ex: {
            "knowledge": ex["hypothesis"],
            "question": "",
            "answer": ex["premise"],
            "labels": 1 if ex["fever_gold_label"] == "REFUTES" else 0,
        }
    )
    if num_samples is not None and num_samples < len(ds):
        ds = ds.shuffle(seed=seed).select(range(num_samples))
    return ds
