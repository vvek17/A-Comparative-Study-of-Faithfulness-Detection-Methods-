"""LLM-as-Judge faithfulness detection using Groq (free tier).

Same rubric/prompt as src/models/llm_judge.py (the Anthropic version), but
calls Groq's free, OpenAI-compatible API instead of a paid one -- no cost per
call, so this can run on the full 10,000-example evaluation set directly
rather than requiring a small-sample cost check first.

Requires GROQ_API_KEY (get one free at https://console.groq.com). Loaded from
a local .env file if present.
"""

from __future__ import annotations

import os
import time

from datasets import Dataset
from tqdm import tqdm

from src.models.llm_judge import PROMPT_TEMPLATE, JudgeResult, _parse_verdict

GROQ_MODEL = "llama-3.1-8b-instant"
GROQ_MODEL_LARGE = "llama-3.3-70b-versatile"  # higher quality; separate daily quota bucket


def run_llm_judge_groq(
    ds: Dataset,
    limit: int | None = 20,
    model: str = GROQ_MODEL,
    max_retries: int = 3,
    sleep_between_calls: float = 0.0,
    stop_on_rate_limit: bool = True,
) -> JudgeResult:
    """Run the LLM-as-Judge evaluation via Groq's free API.

    Requires GROQ_API_KEY to be set (env var or .env file).

    The free tier has a daily token quota in addition to the per-minute one.
    If it's hit mid-run, `stop_on_rate_limit=True` (default) returns whatever
    was completed so far as a partial JudgeResult instead of raising and
    losing that work -- check `len(result.predictions)` against `limit` to see
    whether the run was cut short.
    """
    try:
        from groq import Groq, RateLimitError
    except ImportError as e:
        raise ImportError("pip install groq to use run_llm_judge_groq") from e

    try:
        from dotenv import load_dotenv

        load_dotenv()
    except ImportError:
        pass

    if not os.environ.get("GROQ_API_KEY"):
        raise RuntimeError(
            "GROQ_API_KEY is not set. Get a free key at https://console.groq.com "
            "and set it as an environment variable or in a local .env file."
        )

    client = Groq(api_key=os.environ["GROQ_API_KEY"])

    subset = ds if limit is None else ds.select(range(min(limit, len(ds))))

    predictions, labels, rationales, raw_responses = [], [], [], []
    for ex in tqdm(subset, desc=f"LLM-as-Judge ({model}, Groq)"):
        prompt = PROMPT_TEMPLATE.format(
            knowledge=ex["knowledge"], question=ex["question"], answer=ex["answer"]
        )

        response_text = ""
        hit_rate_limit = False
        for attempt in range(max_retries):
            try:
                response = client.chat.completions.create(
                    model=model,
                    max_tokens=300,
                    messages=[{"role": "user", "content": prompt}],
                )
                response_text = response.choices[0].message.content or ""
                break
            except RateLimitError:
                if stop_on_rate_limit:
                    hit_rate_limit = True
                    break
                if attempt == max_retries - 1:
                    raise
                time.sleep(2 ** attempt)
            except Exception:
                if attempt == max_retries - 1:
                    raise
                time.sleep(2 ** attempt)

        if hit_rate_limit:
            print(
                f"\nHit Groq rate limit after {len(predictions)}/{len(subset)} examples -- "
                "returning partial results instead of losing them."
            )
            break

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
