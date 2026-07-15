# Hallucination Detection in RAG Systems

Detecting Factual Hallucinations in Retrieval-Augmented Generation (RAG)

A comprehensive comparative study evaluating multiple approaches to identify when AI language models generate false or unsupported information despite having access to source documents.

---

## Quick Stats

| Metric | Value |
|--------|-------|
| Best Accuracy (no training) | 68.8% (Zero-Shot NLI) |
| Best Accuracy (with training) | 97.1% (Fine-Tuned DeBERTa, fixed -- 3k-sample validation run) |
| Dataset Size | 10,000 HaluEval examples |
| Methods Evaluated | 3 (NLI, Fine-tuning, LLM-as-Judge) |
| Precision (Faithful Class, Zero-Shot) | 0.730 |
| F1-Score (Zero-Shot) | 0.657 |

---

## What is Hallucination Detection?

### The Problem

Large Language Models (LLMs) often generate plausible-sounding but factually incorrect answers, even when given source documents. This is called hallucination.

Example:
```
Question: "Who won the 2024 Olympics swimming championship?"
Source Document: "Michael Phelps retired in 2015. Katie Ledecky won 2 gold medals."
LLM Output: "Michael Phelps won with a record time." [HALLUCINATION]
```

### Why It Matters

- Healthcare: Wrong medical advice can harm patients
- Legal: Incorrect legal information has serious consequences
- Finance: Bad financial guidance causes losses
- Research: Fake citations damage scientific integrity
- Customer Service: Misinformation destroys trust

### The Solution

Faithfulness Detection: Automatically determine if a generated answer is supported by the source document.

---

## Three Detection Methods Evaluated

### Method 1: Zero-Shot NLI [BEST]

How it works:
- Uses pre-trained Natural Language Inference models
- No additional training required
- Treats faithfulness as an entailment problem
- Input: (passage, answer) -> Output: Faithful or Hallucinated

Models tested:
- facebook/bart-large-mnli
- MoritzLaurer/DeBERTa-v3-large-mnli-fever-anli-ling-wanli

Results:
- [SUCCESS] 68.8% accuracy
- [SUCCESS] No training needed
- [SUCCESS] Works immediately on any domain
- [SUCCESS] Production-ready baseline
- [WARNING] Lower accuracy than fine-tuned models

Code Example:
```python
from transformers import pipeline

# Initialize zero-shot classifier
classifier = pipeline(
    "zero-shot-classification",
    model="facebook/bart-large-mnli"
)

# Check faithfulness
passage = "The Earth orbits the Sun."
answer = "The Sun orbits the Earth."

result = classifier(
    f"{passage} [SEP] {answer}",
    ["faithful", "hallucinated"]
)

print(result)  # Outputs: hallucinated
```

When to use:
- Quick deployment needed
- No training budget available
- Multiple domains/languages
- Human-in-the-loop systems

---

### Method 2: Fine-Tuned DeBERTa [FIXED]

How it works:
- Takes a smaller pre-trained model
- Fine-tunes on HaluEval training data
- Learns task-specific patterns

Setup:
- Model: microsoft/deberta-v3-small (184M parameters)
- Data splits: 80/20 and 70/15/15
- Optimizer: AdamW
- Loss: cross-entropy (see root-cause note below)

Original result (see `docs/report.pdf`):
- [FAILED] ~50% accuracy (chance level)
- [FAILED] Training loss stuck at 0.693 (= ln 2, the loss of a coin flip)
- [FAILED] Predicted a single class for every example

Root cause (diagnosed, not just hypothesized):
The report's own hypotheses -- input truncation, insufficient model capacity, a
mis-tuned learning rate -- were tested empirically and **ruled out**: a raw
PyTorch training loop using the *same* learning rate (2e-5), the *same* data
pipeline, and no `Trainer` converged normally within under one epoch on just
1,000 examples (loss 0.71 -> 0.44, eval accuracy 63.5%). The actual cause lived
in the `Trainer`-side code: a custom `WeightedTrainer` applied inverse-class-
frequency loss weighting to an already-balanced dataset (5,010 vs 4,990
examples) stacked with label smoothing and a cosine LR schedule. The fix in
`src/models/fine_tune.py` removes that custom loss entirely and uses the
`Trainer`'s default unweighted cross-entropy. A validation run on a 3,000-example
subset (70/15/15 split, 3 epochs) confirms the fix: loss drops from 0.70 to
0.10 by epoch 2, with non-degenerate predictions on both classes. Run
`notebooks/02_fine_tuning.ipynb` on the full 10,000-example set for
paper-quality numbers.

Key Lesson:
The "fine-tuning is hard" framing in the original report was itself a red
herring -- the model was never given a fair chance to learn because a
loss-reweighting scheme, not the learning task, was the actual blocker. Always
isolate `Trainer`-level configuration from data/model-level configuration when
diagnosing a degenerate-prediction failure.

**Validated result** (3,000-example subset, 70/15/15 split, 3 epochs, on-disk at
`outputs/tables/finetune_validation_results.json`):

| Metric | Value |
|--------|-------|
| Accuracy | 97.1% |
| Precision | 0.974 |
| Recall | 0.969 |
| F1 | 0.971 |
| Prediction distribution | 227 / 223 (balanced, not degenerate) |

Once fixed, supervised fine-tuning doesn't just "work" -- it decisively beats
the zero-shot baseline (68.8%) on in-domain data, which is the expected
outcome for a supervised method and is consistent with the proposal's original
hypothesis. Run `notebooks/02_fine_tuning.ipynb` on the full 10,000-example set
to get the paper-quality number for both the 80/20 and 70/15/15 splits.

Also corrected: the report's Section 3 prose describes HaluEval's
`hallucination` field as `"yes"` = faithful / `"no"` = hallucinated. Direct
inspection of `pminervini/HaluEval` shows the opposite -- `"yes"` marks the
`answer` field as the *fabricated* one (e.g. row 0: knowledge says Arthur's
Magazine (1844) predates First for Women, but the answer claims the reverse,
and `hallucination="yes"`). The fine-tuning code's label mapping
(`1 if hallucination == "yes" else 0`) already had this right; only the
written description was inverted. `src/data/loader.py` documents this
explicitly.

---

### Method 3: LLM-as-Judge [IMPLEMENTED]

How it works:
- Prompts a large instruction-following model as an evaluator
- Structured prompt with rubric + 1-2 sentence chain-of-thought rationale
- Response parsed to a binary faithful/hallucinated label

Two backends, same prompt/rubric (`src/models/llm_judge.py::PROMPT_TEMPLATE`):
- **Groq (free)** -- `src/models/llm_judge_groq.py`, defaults to
  `llama-3.1-8b-instant` via Groq's free API (get a key at console.groq.com).
  No cost, so it can run on larger samples than the paid option -- but the
  free tier has a **daily token quota per model** (100k tokens/day) in
  addition to the ~12,000/min and 1,000 requests/day limits; a few hundred
  judge calls in one day can exhaust a single model's quota. The quota is
  tracked **separately per model** -- `llama-3.3-70b-versatile`'s quota ran
  out from earlier testing the same day, but `llama-3.1-8b-instant` still had
  headroom and completed cleanly, which is why it's the default. Pass
  `model="llama-3.3-70b-versatile"` explicitly once that model's quota resets
  if you want the larger model's judgments instead.
  `run_llm_judge_groq()` catches a mid-run quota exhaustion and returns
  whatever it completed instead of losing the run (check
  `len(result.predictions)` against your requested `limit`) --
  `notebooks/03_llm_judge.ipynb` / `run_judge.py` pace calls to stay under
  the per-minute limit, but the full 10,000-example set needs multiple days
  (or a paid Groq tier) to clear the daily cap.
- **Claude (paid)** -- `src/models/llm_judge.py`, requires `ANTHROPIC_API_KEY`
  and costs money per call; not auto-run, start with a small sample.

Approach:
```
Input: 
  - Source passage
  - Generated answer
  - Evaluation rubric

Output:
  - Faithful/Hallucinated label
  - Reasoning chain
```

Results (see `outputs/tables/llm_judge_groq_results.json`), both comfortably
above the zero-shot NLI baseline (68.8%):
- `llama-3.3-70b-versatile`, n=20: 85.0% accuracy, F1 = 0.842
- `llama-3.1-8b-instant`, n=50 (all 50 completed cleanly): 86.0% accuracy,
  F1 = 0.844, **precision = 1.00** (zero false hallucination flags), recall
  0.731 (misses ~27% of actual hallucinations)

Both are small-sample directional estimates, not final numbers -- run at
larger scale (spread across multiple days, given the daily quota) before
citing these in the report.

---

## Performance Comparison

### All Methods Side-by-Side

```
Method                                       Accuracy  Precision  Recall  F1-Score
BART-MNLI (Zero-Shot)                        68.8%     0.730      0.598   0.657
DeBERTa-v3-Large (Zero-Shot)                 68.8%     0.730      0.598   0.657
BART + SummaC-Zero                           67.2%     0.709      0.587   0.642
DeBERTa-v3-small (80/20, ORIGINAL/BROKEN)    49.4%     0.494      1.000   0.661
DeBERTa-v3-small (70/15/15, ORIGINAL/BROKEN) 50.1%     0.501      1.000   0.667
DeBERTa-v3-small (FIXED, 3k-sample validation) 97.1%   0.974      0.969   0.971
LLM-as-Judge                                 —         —          —       —
```

The "ORIGINAL/BROKEN" rows are the results reported in `docs/report.pdf`,
kept here for comparison. The "FIXED" row is the corrected implementation in
`src/models/fine_tune.py`, run on a 3,000-example subset as a validation check
(see the root-cause note above) -- rerun `notebooks/02_fine_tuning.ipynb` on
the full 10,000 examples for the final paper-quality number.

### Per-Class Breakdown

Hallucinated Examples (Negative Class):
- Precision: 0.66 (66% are truly hallucinated)
- Recall: 0.78 (catches 78% of hallucinations)
- F1: 0.71

Faithful Examples (Positive Class):
- Precision: 0.73 (73% are truly faithful)
- Recall: 0.60 (misses 40% of faithful responses)
- F1: 0.66

Implication: Models are conservative - better to incorrectly flag faithful responses than miss hallucinations!

---

## Implementation Guide

### Installation

```bash
# Clone repository
git clone https://github.com/vvek17/hallucination-detection-rag.git
cd hallucination-detection-rag

# Create virtual environment (recommended)
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### Requirements

```
torch>=2.0.0
transformers>=4.30.0
datasets>=2.13.0
scikit-learn>=1.3.0
pandas>=2.0.0
numpy>=1.24.0
tqdm>=4.65.0
```

### Quick Start - 3 Lines of Code

```python
from transformers import pipeline
from datasets import load_dataset

# 1. Load model
classifier = pipeline("zero-shot-classification", 
                      model="facebook/bart-large-mnli")

# 2. Load data
dataset = load_dataset("pminervini/HaluEval", "qa_samples")

# 3. Evaluate
for example in dataset['data'][:5]:
    passage = example['document']
    answer = example['response']
    result = classifier(f"{passage} [SEP] {answer}", 
                       ["faithful", "hallucinated"])
    print(f"Answer: {answer[:50]}...")
    print(f"Prediction: {result['labels'][0]}")
    print(f"Score: {result['scores'][0]:.2f}\n")
```

### Advanced Usage

```python
from transformers import pipeline, AutoTokenizer
from datasets import load_dataset
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score

# Initialize classifier
classifier = pipeline(
    "zero-shot-classification",
    model="facebook/bart-large-mnli"
)

# Load dataset
dataset = load_dataset("pminervini/HaluEval", "qa_samples")
data = dataset['data'][:1000]  # Use 1000 examples

# Make predictions
predictions = []
true_labels = []

for example in data:
    passage = example['knowledge']
    answer = example['answer']
    label = 1 if example['hallucination'] == 'yes' else 0  # 1=hallucinated, 0=faithful
    
    # Get prediction
    result = classifier(
        f"{passage} [SEP] {answer}",
        ["faithful", "hallucinated"],
        multi_class=False
    )
    
    pred = 1 if result['labels'][0] == 'hallucinated' else 0
    predictions.append(pred)
    true_labels.append(label)

# Calculate metrics
print(f"Accuracy:  {accuracy_score(true_labels, predictions):.3f}")
print(f"Precision: {precision_score(true_labels, predictions):.3f}")
print(f"Recall:    {recall_score(true_labels, predictions):.3f}")
print(f"F1-Score:  {f1_score(true_labels, predictions):.3f}")
```

---

## Dataset Overview

### HaluEval QA Benchmark

| Aspect | Details |
|--------|---------|
| Total Examples | 35,000 (10,000 used in study) |
| Task Type | Question Answering |
| Faithful Examples | 5,010 (50.1%) |
| Hallucinated Examples | 4,990 (49.9%) |
| Generation Method | ChatGPT (for hallucinations) |
| Annotation | Human-verified |

### Data Structure

```json
{
  "knowledge": "The Eiffel Tower is located in Paris, France...",
  "question": "Where is the Eiffel Tower?",
  "answer": "The Eiffel Tower is in London, England.",
  "hallucination": "yes"
}
```

Note: `hallucination` is a string (`"yes"`/`"no"`), not an int, and `"yes"` means
the `answer` field is the fabricated one (see the Method 2 root-cause note above).

---

## Key Takeaways and Recommendations

### DO:

1. Start with Zero-Shot NLI
   - No training required
   - Production-ready immediately
   - 68.8% accuracy baseline

2. Fine-tune once you have labeled in-domain data
   - 97.1% accuracy on a validation run once the loss-weighting bug was fixed
   - Use the `Trainer`'s default unweighted cross-entropy unless you've
     confirmed class imbalance is severe enough to need it
   - Sanity-check prediction distribution every run -- a single-class
     collapse is a configuration bug, not a data limit

3. Use for Human-in-the-Loop Systems
   - Flag suspicious responses
   - Route for human review
   - Improve over time

4. Combine Multiple Methods
   - Ensemble different detectors
   - Increase robustness
   - Better coverage

5. Monitor and Evaluate
   - Track precision/recall
   - Domain-specific validation
   - User feedback loops

### DON'T:

1. Don't add loss-reweighting/label-smoothing/schedule complexity before
   checking the basics
   - This project's fine-tuning "failure" was entirely self-inflicted by a
     custom class-weighted loss on an already-balanced dataset, not a
     fundamental limitation of small models
   - Isolate `Trainer`-level config from data/model config when debugging a
     degenerate-prediction failure

2. Don't ignore class imbalance
   - Precision/recall tradeoff
   - False negatives more costly in high-stakes
   - Balance metrics carefully

3. Don't use as-is in production
   - 31.2% error rate still high
   - Needs human validation layer
   - Domain-specific tuning recommended

4. Don't forget context
   - Evaluate on your actual data
   - Cross-domain performance varies
   - Test on target documents

---

## Technical Details

### Input Format

Zero-Shot NLI:
```
[Passage] [SEP] [Generated Answer]
DOWN
NLI Model
DOWN
Entailment vs Contradiction probabilities
```

Fine-Tuning (if retrying):
```
[CLS] [Passage] [SEP] [Answer] [SEP]
DOWN
DeBERTa Encoder
DOWN
Binary Classification Head
DOWN
Faithful/Hallucinated
```

### Evaluation Metrics Explained

| Metric | Formula | Use Case |
|--------|---------|----------|
| Accuracy | (TP+TN)/(Total) | Overall performance |
| Precision | TP/(TP+FP) | Quality of positive predictions |
| Recall | TP/(TP+FN) | Coverage of positives |
| F1 | 2x(P x R)/(P+R) | Balanced metric |

Which to optimize?
- High-stakes (medical): Maximize recall (catch all hallucinations)
- General QA: Balance precision-recall
- User experience: Maximize precision (trust in system)

---

## Educational Value

### What You'll Learn

1. RAG Systems: How grounded generation works
2. Hallucination: Types, causes, detection methods
3. NLI Models: Entailment, contradiction, neutrality
4. Transfer Learning: Using pre-trained models
5. Fine-tuning: When/how to train models
6. Evaluation: Metrics beyond accuracy
7. Benchmarking: Comparing approaches

### Research Applications

- Build fact-checking systems
- Improve LLM reliability
- Domain-specific hallucination detection
- Ensemble methods for robustness
- Active learning for annotation

---

## Background and Related Work

### Key Papers

1. RAG Framework - Lewis et al. (2020)
   - Original RAG architecture
   - Dense passage retrieval
   - Generator-based decoding

2. Consistency Detection - Laban et al. (2021)
   - SummaC method
   - Sentence-level NLI
   - Factual consistency

3. HaluEval Benchmark - Li et al. (2023)
   - 35K labeled examples
   - QA, dialogue, summarization
   - ChatGPT-generated hallucinations

4. FEVER Dataset - Thorne et al. (2018)
   - Fact extraction and verification
   - 145K examples
   - Cross-domain testing

---

## Future Directions

### Immediate (Next Sprint)
- Hyperparameter grid search for fine-tuning
- Sequence truncation analysis
- FEVER cross-domain evaluation
- LLM-as-Judge implementation

### Short-term (Next 3 months)
- Test larger model variants (DeBERTa-base/large)
- Prompt engineering experiments
- Error categorization analysis
- Latency vs accuracy benchmarks

### Long-term (Next 6-12 months)
- Multi-task learning approaches
- Domain-specific fine-tuning
- Active learning for annotation
- Production deployment pipeline

---

## Project Structure

```
A-Comparative-Study-of-Faithfulness-Detection-Methods/
|
├── README.md
├── requirements.txt
├── setup.py
|
├── notebooks/
|   ├── 01_zero_shot_nli.ipynb        # Method 1: BART-MNLI, DeBERTa-MNLI, SummaC-Zero
|   ├── 02_fine_tuning.ipynb          # Method 2: fixed DeBERTa-v3-small fine-tuning
|   ├── 03_llm_judge.ipynb            # Method 3: LLM-as-Judge (Groq, free; Claude, paid/opt-in)
|   ├── 04_fever_cross_dataset.ipynb  # cross-domain generalization on FEVER
|   └── 05_analysis.ipynb             # aggregate comparison + error analysis
|
├── src/
|   ├── models/
|   |   ├── zero_shot.py     # document-level + SummaC-Zero sentence-level NLI
|   |   ├── fine_tune.py     # supervised fine-tuning (root-cause fix applied)
|   |   ├── llm_judge.py     # Claude-based judge, no auto-run (paid API)
|   |   └── llm_judge_groq.py # Groq-based judge (free, same prompt/rubric)
|   |
|   ├── data/
|   |   ├── loader.py        # HaluEval + FEVER loaders, label-convention notes
|   |   ├── preprocessor.py  # NLI-pair / fine-tune-pair formatting, tokenization
|   |   └── utils.py         # seeding, device selection
|   |
|   └── evaluation/
|       ├── metrics.py         # accuracy/precision/recall/F1 + reports
|       ├── error_analysis.py  # heuristic error categorization
|       └── visualize.py       # comparison + confusion-matrix plots
|
├── tests/
|   └── test_basic.py   # fast, network-free sanity tests
|
├── outputs/
|   ├── figures/   # saved plots
|   ├── tables/     # saved per-method results (json)
|   └── logs/       # Trainer checkpoints/logs
|
├── legacy/
|   └── rag_hallucination_original.ipynb  # original exploratory notebook, kept for reference
|
└── docs/
    ├── report.pdf
    └── proposal.pdf
```

---

## Important Limitations

1. Dataset Size: Only 10,000 of 35,000 examples used
2. Domain Specificity: Optimized for QA, may not transfer to summarization
3. Cross-Domain: FEVER evaluation incomplete
4. Inference Speed: Not benchmarked for latency
5. LLM-Judge: Not evaluated (performance unknown)
6. Fine-tuning: Failure not fully diagnosed

---

## How to Contribute

### Bug Reports
Found an issue? Open a GitHub issue with:
- Minimal reproducible example
- Error message
- Environment (Python version, GPU, etc.)

### Improvements
Want to improve results? Try:
- Different hyperparameters
- Larger models
- Alternative architectures
- Domain-specific tuning

### Research
Want to extend this? Consider:
- Multi-lingual evaluation
- Real-time detection
- Explanation generation
- Ensemble methods

---

## Contact and Support

Author: Vivek Solanki
Email: vivekksolankii691@gmail.com (tuu31610@temple.edu for course-related matters)
GitHub: @vvek17
Institution: Temple University, Philadelphia, PA
Course: CIS 5523 - Knowledge Discovery and Data Mining

---

## Citation

If you use this work, please cite:

```
@report{solanki2026hallucination,
  title={Detecting Factual Hallucinations in Retrieval-Augmented Generation Systems: 
         A Comparative Study of Faithfulness Detection Methods},
  author={Solanki, Vivek},
  institution={Temple University},
  course={CIS 5523: Knowledge Discovery and Data Mining},
  year={2026},
  month={Spring}
}
```

---

## Documentation Links

- Full Report: [docs/report.pdf](docs/report.pdf)
- Project Proposal: [docs/proposal.pdf](docs/proposal.pdf)
- Notebooks: See [/notebooks](notebooks/) directory
- Code: See [/src](src/) directory

---

## License

This project is provided for academic and research purposes.

- Datasets: HaluEval (Li et al. 2023), FEVER (Thorne et al. 2018) - refer to original licenses
- Models: BART (Meta), DeBERTa (Microsoft) - refer to HuggingFace model cards
- Code: MIT License

---

## Summary

| Question | Answer |
|----------|--------|
| What | Detecting AI hallucinations in RAG systems |
| How | Zero-shot NLI (68.8% accuracy) |
| When | Production ready for human review |
| Why | Critical for high-stakes applications |
| Where | Healthcare, legal, finance, research |
| Who | Researchers, practitioners, ML engineers |

---

Last Updated: Spring 2026
Status: Active Research

---

If this helps your research, please star the repository!
