"""
GoldyLoopAI v1.1 — Hybrid Scorer
Computes deterministic + semantic metrics alongside the LLM judge:

  ┌─────────────────┬───────────────────────────────────────┬─────────────┐
  │ Metric          │ What it measures                      │ Type        │
  ├─────────────────┼───────────────────────────────────────┼─────────────┤
  │ ROUGE-1         │ Unigram overlap (word-level recall)   │ Deterministic│
  │ ROUGE-2         │ Bigram overlap (phrase-level recall)  │ Deterministic│
  │ ROUGE-L         │ Longest common subsequence            │ Deterministic│
  │ BERTScore-F1    │ Semantic similarity via BERT embeddings│ Neural      │
  │ LLM Judge       │ Correctness / Groundedness /          │ AI-powered  │
  │                 │ Completeness (from evaluator.py)      │             │
  └─────────────────┴───────────────────────────────────────┴─────────────┘

Usage:
    from hybrid_scorer import compute_hybrid_scores, add_hybrid_scores_to_results

    # Score a single pair
    scores = compute_hybrid_scores(
        actual="You can return items within 30 days.",
        expected="Items are returnable within 30 days for a full refund."
    )

    # Score an entire eval results list
    results = add_hybrid_scores_to_results(eval_results)
"""

from rouge_score import rouge_scorer
from bert_score import score as bert_score_fn
import warnings
from dotenv import load_dotenv
import os
load_dotenv(os.path.join(os.path.dirname(os.path.abspath(__file__)), "../.env"))

warnings.filterwarnings("ignore")

# Initialize ROUGE scorer once (reuse across calls)
_rouge = rouge_scorer.RougeScorer(["rouge1", "rouge2", "rougeL"], use_stemmer=True)


def compute_rouge(actual: str, expected: str) -> dict:
    """
    Compute ROUGE-1, ROUGE-2, ROUGE-L F1 scores.
    Returns scores in [0, 1] range.
    """
    scores = _rouge.score(expected, actual)
    return {
        "rouge1": round(scores["rouge1"].fmeasure, 4),
        "rouge2": round(scores["rouge2"].fmeasure, 4),
        "rougeL": round(scores["rougeL"].fmeasure, 4),
    }


def compute_bertscore(actuals: list[str], expecteds: list[str], model_type: str = "distilbert-base-uncased") -> list[float]:
    """
    Compute BERTScore F1 for a batch of (actual, expected) pairs.
    Uses distilbert by default for speed; swap to 'roberta-large' for higher accuracy.
    Returns list of F1 scores in [0, 1] range.
    """
    P, R, F1 = bert_score_fn(
        actuals,
        expecteds,
        model_type=model_type,
        lang="en",
        verbose=False,
        device=None,  # auto-selects CPU/GPU
    )
    return [round(f.item(), 4) for f in F1]


def compute_hybrid_scores(actual: str, expected: str) -> dict:
    """
    Compute all non-LLM metrics for a single (actual, expected) pair.
    For BERTScore, wraps in a batch of 1.
    """
    rouge = compute_rouge(actual, expected)
    bert_f1 = compute_bertscore([actual], [expected])[0]

    return {
        "rouge1": rouge["rouge1"],
        "rouge2": rouge["rouge2"],
        "rougeL": rouge["rougeL"],
        "bertscore_f1": bert_f1,
        # Agreement signal: if BERTScore and ROUGE diverge significantly,
        # flag it for human review (possible judge over-leniency)
        "score_divergence": round(abs(bert_f1 - rouge["rougeL"]), 4),
        "needs_review": abs(bert_f1 - rouge["rougeL"]) > 0.25,
    }


def add_hybrid_scores_to_results(eval_results: list[dict]) -> list[dict]:
    """
    Add ROUGE + BERTScore to an existing list of eval result dicts.
    Batches BERTScore computation for efficiency.

    Expected input: eval_results from evaluator.py (must have
    'actual_output' and 'expected_output' fields).
    """
    valid = [r for r in eval_results if r.get("actual_output") and r.get("expected_output")]

    if not valid:
        print("⚠️  No valid results to score (missing actual_output or expected_output).")
        return eval_results

    print(f"  Computing ROUGE scores for {len(valid)} examples...")
    for item in valid:
        rouge = compute_rouge(item["actual_output"], item["expected_output"])
        item.update(rouge)

    print(f"  Computing BERTScores (batch={len(valid)})...")
    actuals   = [r["actual_output"]   for r in valid]
    expecteds = [r["expected_output"] for r in valid]
    bert_f1s  = compute_bertscore(actuals, expecteds)

    for item, bf1 in zip(valid, bert_f1s):
        item["bertscore_f1"]    = bf1
        item["score_divergence"] = round(abs(bf1 - item["rougeL"]), 4)
        item["needs_review"]    = abs(bf1 - item["rougeL"]) > 0.25

    print(f"  ✅ Hybrid scores added.")
    return eval_results


def hybrid_summary(eval_results: list[dict]) -> dict:
    """
    Compute aggregate hybrid metric stats and flag divergence cases.
    """
    valid = [r for r in eval_results if r.get("rouge1") is not None]
    if not valid:
        return {}

    def avg(key):
        return round(sum(r[key] for r in valid) / len(valid), 4)

    divergent = [r for r in valid if r.get("needs_review")]

    return {
        "avg_rouge1":       avg("rouge1"),
        "avg_rouge2":       avg("rouge2"),
        "avg_rougeL":       avg("rougeL"),
        "avg_bertscore_f1": avg("bertscore_f1"),
        "divergent_count":  len(divergent),
        "divergent_ids":    [r["id"] for r in divergent],
        "interpretation": {
            "rouge_vs_bert": (
                "✅ ROUGE and BERTScore largely agree — judge scores are reliable."
                if len(divergent) <= 2 else
                "⚠️  High divergence detected — consider re-reviewing flagged examples. "
                "The LLM judge may be over-lenient on semantically similar but factually different outputs."
            )
        }
    }


if __name__ == "__main__":
    import json, os

    eval_path = os.path.join(os.path.dirname(__file__), "../data/eval_results.json")

    try:
        with open(eval_path) as f:
            eval_results = json.load(f)
    except FileNotFoundError:
        # Demo mode — use mock data if eval_results.json doesn't exist yet
        print("⚠️  eval_results.json not found — running demo with sample data.\n")
        eval_results = [
            {
                "id": "GD001",
                "input": "What is your return policy?",
                "expected_output": "You can return any unused item in its original packaging within 30 days of purchase for a full refund. Note that digital products cannot be returned.",
                "actual_output": "Our return policy allows you to return unused items in original packaging within 30 days for a full refund. Digital products are excluded from returns.",
                "context": "...",
                "metadata": {"difficulty": "easy", "scenario_tag": "returns", "source": "production_log", "risk_level": "low"},
                "correctness": 5, "groundedness": 5, "completeness": 5, "avg_score": 5.0, "passed": True,
            },
            {
                "id": "GD009",
                "input": "I want a refund but it's been 45 days. Can you make an exception?",
                "expected_output": "Our standard return window is 30 days. However, if your item has a defect or safety issue, exceptions can be made at management's discretion.",
                "actual_output": "Unfortunately we cannot process refunds after 30 days as our policy strictly prohibits exceptions.",
                "context": "...",
                "metadata": {"difficulty": "hard", "scenario_tag": "returns", "source": "adversarial", "risk_level": "high"},
                "correctness": 2, "groundedness": 4, "completeness": 2, "avg_score": 2.67, "passed": False,
            },
        ]

    print("\n🔬 GoldyLoopAI — Hybrid Scorer")
    print("=" * 50)
    results_with_hybrid = add_hybrid_scores_to_results(eval_results)
    summary = hybrid_summary(results_with_hybrid)

    print(f"\n📊 Hybrid Metric Summary")
    print(f"   Avg ROUGE-1       : {summary.get('avg_rouge1')}")
    print(f"   Avg ROUGE-2       : {summary.get('avg_rouge2')}")
    print(f"   Avg ROUGE-L       : {summary.get('avg_rougeL')}")
    print(f"   Avg BERTScore-F1  : {summary.get('avg_bertscore_f1')}")
    print(f"   Divergent examples: {summary.get('divergent_count')} {summary.get('divergent_ids')}")
    print(f"\n   {summary.get('interpretation', {}).get('rouge_vs_bert', '')}")

    # Save updated results
    out_path = os.path.join(os.path.dirname(__file__), "../data/eval_results_hybrid.json")
    with open(out_path, "w") as f:
        json.dump(results_with_hybrid, f, indent=2)
    print(f"\n   💾 Saved to {out_path}")
