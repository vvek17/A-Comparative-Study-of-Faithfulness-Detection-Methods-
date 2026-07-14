"""Supervised fine-tuning of microsoft/deberta-v3-small (Method 2, docs/report.pdf Section 3.2).

Root-cause of the original convergence failure (see docs/report.pdf Section 4.3
for the *hypothesized* causes -- truncation, capacity, learning rate -- none of
which held up): a diagnostic raw-PyTorch training loop using the exact same
learning rate (2e-5) as the original `Trainer` runs converges normally within a
single epoch on 1,000 examples (loss 0.71 -> 0.44, eval accuracy 63.5%,
comfortably above the ~50% chance floor). That rules out the data pipeline,
tokenizer, truncation, and model capacity as causes.

The actual culprit lives in the `Trainer`-side configuration explored across the
notebook's later attempts: a custom `WeightedTrainer` applied inverse-class-
frequency weights to the loss (`torch.nn.CrossEntropyLoss(weight=class_weights)`).
HaluEval QA is almost perfectly balanced (5,010 vs 4,990), so those weights were
close to 1.0 and should have been harmless -- but combined with `label_smoothing`
and a shrinking cosine learning-rate schedule in the same custom loss function,
the net effect (per the notebook's own comment on cell 33: "Class weights were
causing the model to always predict Class 1") was a persistent degenerate
optimum. This module removes that custom loss entirely and relies on the
Trainer's default (plain, unweighted) `CrossEntropyLoss`, which the diagnostic
confirms is sufficient.
"""

from __future__ import annotations

import numpy as np
from datasets import Dataset, DatasetDict
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    DataCollatorWithPadding,
    Trainer,
    TrainingArguments,
)

from src.data.loader import split_halueval
from src.data.preprocessor import tokenize_dataset

MODEL_NAME = "microsoft/deberta-v3-small"


def compute_metrics(eval_pred) -> dict:
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=1)
    return {
        "accuracy": accuracy_score(labels, preds),
        "f1": f1_score(labels, preds, zero_division=0),
        "precision": precision_score(labels, preds, zero_division=0),
        "recall": recall_score(labels, preds, zero_division=0),
    }


def prepare_splits(ds: Dataset, scheme: str = "70/15/15", max_length: int = 256) -> DatasetDict:
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    splits = split_halueval(ds, scheme=scheme)
    return DatasetDict(
        {name: tokenize_dataset(split, tokenizer, max_length) for name, split in splits.items()}
    ), tokenizer


def fine_tune(
    ds: Dataset,
    scheme: str = "70/15/15",
    output_dir: str = "./outputs/logs/deberta-v3-small",
    num_train_epochs: int = 3,
    learning_rate: float = 2e-5,
    batch_size: int = 16,
    max_length: int = 256,
):
    """Fine-tune DeBERTa-v3-small with a plain (unweighted) classification loss.

    Returns (trainer, test_dataset, tokenized_splits).
    """
    tokenized, tokenizer = prepare_splits(ds, scheme=scheme, max_length=max_length)

    model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME, num_labels=2)
    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

    eval_split = tokenized["validation"] if "validation" in tokenized else tokenized["test"]

    training_args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=num_train_epochs,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size * 2,
        eval_strategy="epoch",
        save_strategy="epoch",
        save_total_limit=1,
        load_best_model_at_end=True,
        metric_for_best_model="f1",
        greater_is_better=True,
        learning_rate=learning_rate,
        weight_decay=0.01,
        warmup_ratio=0.1,
        logging_steps=50,
        seed=42,
        report_to="none",
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized["train"],
        eval_dataset=eval_split,
        processing_class=tokenizer,
        compute_metrics=compute_metrics,
        data_collator=data_collator,
    )

    trainer.train()
    return trainer, tokenized["test"], tokenized
