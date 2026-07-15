"""Fast, network/model-free sanity tests for the preprocessing and evaluation logic."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.data.preprocessor import build_finetune_pair, build_nli_pair
from src.evaluation.error_analysis import categorize_example
from src.evaluation.metrics import compute_metrics, print_report
from src.models.llm_judge import _parse_verdict
from src.models.llm_judge_groq import GROQ_MODEL, run_llm_judge_groq


def test_build_nli_pair():
    premise, hypothesis = build_nli_pair("The sky is blue.", "Paris", "The sky is red.")
    assert premise == "The sky is blue."
    assert hypothesis == "The sky is red."


def test_build_finetune_pair():
    text_a, text_b = build_finetune_pair("Where is X?", "In Y.", "X is in Y.")
    assert text_a == "Where is X? In Y."
    assert text_b == "X is in Y."


def test_compute_metrics_perfect():
    m = compute_metrics([1, 0, 1, 0], [1, 0, 1, 0])
    assert m.accuracy == 1.0
    assert m.f1 == 1.0


def test_compute_metrics_all_wrong():
    m = compute_metrics([1, 0, 1, 0], [0, 1, 0, 1])
    assert m.accuracy == 0.0


def test_categorize_example_date():
    assert categorize_example("It happened in 1994.") == "date"


def test_categorize_example_numerical():
    assert categorize_example("There were 42 people present.") == "numerical"


def test_parse_verdict_json():
    label, rationale = _parse_verdict('{"reasoning": "no support", "verdict": "HALLUCINATED"}')
    assert label == 1
    assert "no support" in rationale


def test_parse_verdict_faithful():
    label, _ = _parse_verdict('{"reasoning": "matches", "verdict": "FAITHFUL"}')
    assert label == 0


def test_parse_verdict_fallback_unstructured():
    label, _ = _parse_verdict("This answer is clearly a hallucination.")
    assert label == 1


def test_groq_judge_requires_api_key(monkeypatch):
    import dotenv

    monkeypatch.delenv("GROQ_API_KEY", raising=False)
    monkeypatch.setattr(dotenv, "load_dotenv", lambda *a, **k: None)
    try:
        run_llm_judge_groq(ds=[], limit=0)
        assert False, "expected RuntimeError"
    except RuntimeError as e:
        assert "GROQ_API_KEY" in str(e)


def test_groq_model_constant():
    assert GROQ_MODEL == "llama-3.1-8b-instant"


def test_print_report_single_class_sample(capsys):
    # Regression test: a sample with only one true/predicted class (e.g. a
    # partial run cut short by a rate limit) must not crash sklearn's
    # classification_report with a labels-mismatch ValueError.
    print_report([1], [1], title="single-example sample")
    captured = capsys.readouterr()
    assert "Accuracy" in captured.out


if __name__ == "__main__":
    import pytest

    sys.exit(pytest.main([__file__, "-v"]))
