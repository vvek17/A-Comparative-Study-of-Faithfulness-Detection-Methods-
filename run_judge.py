import json

from src.data.loader import load_halueval_qa
from src.models.llm_judge_groq import GROQ_MODEL, run_llm_judge_groq
from src.evaluation.metrics import compute_metrics, print_report

MODEL = GROQ_MODEL  # llama-3.1-8b-instant; pass GROQ_MODEL_LARGE for higher quality
                     # once that model's separate daily quota resets
LIMIT = 50

ds = load_halueval_qa(num_samples=LIMIT, seed=42)
result = run_llm_judge_groq(ds, limit=LIMIT, sleep_between_calls=3.2, model=MODEL)

n = len(result.predictions)
title = f"LLM-as-Judge (Groq, {MODEL}, n={n})"
print_report(result.labels, result.predictions, title=title)

if n > 0:
    with open("outputs/tables/llm_judge_groq_results.json") as f:
        existing = json.load(f)
    existing[title] = compute_metrics(result.labels, result.predictions).as_dict()
    with open("outputs/tables/llm_judge_groq_results.json", "w") as f:
        json.dump(existing, f, indent=2)
    print(f"\nSaved '{title}' to outputs/tables/llm_judge_groq_results.json")
