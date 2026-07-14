"""Text formatting and tokenization helpers shared by all detection methods."""

from __future__ import annotations

from datasets import Dataset
from transformers import PreTrainedTokenizerBase


def build_nli_pair(knowledge: str, question: str, answer: str) -> tuple[str, str]:
    """Build the (premise, hypothesis) pair used for zero-shot NLI.

    The premise is the retrieved knowledge; the hypothesis is the generated
    answer. The question is not included in the NLI pair -- the model isn't
    asked to answer the question, only whether `answer` is entailed by
    `knowledge`. (Question context is used instead in the fine-tuning input,
    see `build_finetune_pair`, and in the LLM-as-Judge prompt.)
    """
    return knowledge, answer


def build_finetune_pair(question: str, answer: str, knowledge: str) -> tuple[str, str]:
    """Build the (text_a, text_b) segment pair used for supervised fine-tuning.

    text_a = "{question} {answer}" (segment A / token_type_ids=0)
    text_b = knowledge              (segment B / token_type_ids=1)

    Putting question+answer first keeps them intact even when knowledge is
    long: the tokenizer's default 'longest_first' truncation strategy trims
    whichever segment is currently longer, and knowledge is almost always the
    longer segment (see docs/report.pdf Section 4.3 for the length analysis
    that motivated this ordering).
    """
    return f"{question} {answer}", knowledge


def make_tokenize_fn(tokenizer: PreTrainedTokenizerBase, max_length: int = 256):
    def tokenize(batch: dict) -> dict:
        text_a = [f"{q} {a}" for q, a in zip(batch["question"], batch["answer"])]
        text_b = batch["knowledge"]
        return tokenizer(
            text_a, text_b, truncation=True, padding="max_length", max_length=max_length
        )

    return tokenize


def tokenize_dataset(
    ds: Dataset, tokenizer: PreTrainedTokenizerBase, max_length: int = 256
) -> Dataset:
    tokenize = make_tokenize_fn(tokenizer, max_length)
    tokenized = ds.map(tokenize, batched=True)
    cols = [
        c
        for c in ["input_ids", "attention_mask", "token_type_ids", "labels"]
        if c in tokenized.column_names
    ]
    tokenized.set_format("torch", columns=cols)
    return tokenized
