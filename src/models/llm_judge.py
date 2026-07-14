"""LLM-as-Judge faithfulness detection (Method 3, planned in docs/proposal.pdf).

Prompts Claude with the (knowledge, question, answer) triple and a rubric,
asking for a binary faithfulness judgment with a short chain-of-thought
rationale, then parses out the final label.

This module only builds the prompt and makes the API calls -- it does not run
automatically. Each call costs money and hits Anthropic's API, so callers must
supply their own ANTHROPIC_API_KEY and explicitly invoke `run_llm_judge`.
Start with a small `limit` (e.g. 20-50) before scaling up to the full
evaluation set; see notebooks/03_llm_judge.ipynb.
"""

from __future__ import annotations

import json
import os
import re
import time
from dataclasses import dataclass

from datasets import Dataset
from tqdm import tqdm

JUDGE_MODEL = "claude-sonnet-5"

PROMPT_TEMPLATE = """You are evaluating whether a generated answer is faithful to (fully \
supported by) a source passage, for a hallucination-detection benchmark.

Source passage:
\"\"\"{knowledge}\"\"\"

Question: {question}

Generated answer: "{answer}"

Judge whether the generated answer is fully supported by the source passage \
(FAITHFUL) or contradicts / introduces information not found in the passage \
(HALLUCINATED). Briefly explain your reasoning in 1-2 sentences, then give your \
final verdict.

Respond with exactly this JSON structure and nothing else:
{{"reasoning": "<1-2 sentence rationale>", "verdict": "FAITHFUL" or "HALLUCINATED"}}"""


@dataclass
class JudgeResult:
    predictions: list[int]
    labels: list[int]
    rationales: list[str]
    raw_responses: list[str]


def _parse_verdict(text: str) -> tuple[int, str]:
    """Return (label, rationale). label: 1 = hallucinated, 0 = faithful."""
    match = re.search(r"\{.*\}", text, re.DOTALL)
    if match:
        try:
            obj = json.loads(match.group(0))
            verdict = str(obj.get("verdict", "")).strip().upper()
            rationale = str(obj.get("reasoning", "")).strip()
            if "HALLUCIN" in verdict:
                return 1, rationale
            if "FAITHFUL" in verdict:
                return 0, rationale
        except json.JSONDecodeError:
            pass
    # Fallback: look for the words directly in unstructured output.
    upper = text.upper()
    if "HALLUCIN" in upper:
        return 1, text.strip()
    return 0, text.strip()


def run_llm_judge(
    ds: Dataset,
    limit: int | None = 20,
    model: str = JUDGE_MODEL,
    max_retries: int = 3,
    sleep_between_calls: float = 0.0,
) -> JudgeResult:
    """Run the LLM-as-Judge evaluation. Requires ANTHROPIC_API_KEY to be set.

    Costs real money per call -- always pass a small `limit` first.
    """
    try:
        import anthropic
    except ImportError as e:
        raise ImportError("pip install anthropic to use run_llm_judge") from e

    if not os.environ.get("ANTHROPIC_API_KEY"):
        raise RuntimeError(
            "ANTHROPIC_API_KEY is not set. This method calls a paid API -- "
            "set the key explicitly before running it."
        )

    client = anthropic.Anthropic()

    subset = ds if limit is None else ds.select(range(min(limit, len(ds))))

    predictions, labels, rationales, raw_responses = [], [], [], []
    for ex in tqdm(subset, desc=f"LLM-as-Judge ({model})"):
        prompt = PROMPT_TEMPLATE.format(
            knowledge=ex["knowledge"], question=ex["question"], answer=ex["answer"]
        )

        response_text = ""
        for attempt in range(max_retries):
            try:
                response = client.messages.create(
                    model=model,
                    max_tokens=300,
                    messages=[{"role": "user", "content": prompt}],
                )
                response_text = response.content[0].text
                break
            except Exception:
                if attempt == max_retries - 1:
                    raise
                time.sleep(2 ** attempt)

        pred, rationale = _parse_verdict(response_text)
        predictions.append(pred)
        labels.append(ex["labels"])
        rationales.append(rationale)
        raw_responses.append(response_text)

        if sleep_between_calls:
            time.sleep(sleep_between_calls)

    return JudgeResult(
        predictions=predictions, labels=labels, rationales=rationales, raw_responses=raw_responses
    )
