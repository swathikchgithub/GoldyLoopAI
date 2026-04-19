"""
GoldyLoopAI - LLM-as-a-Judge Evaluator
Scores model outputs using a judge LLM across three dimensions:
  - Correctness: Does it match the expected answer?
  - Groundedness: Does it stick to the provided context?
  - Completeness: Does it cover all key points?
"""

import os
from dotenv import load_dotenv
load_dotenv(os.path.join(os.path.dirname(os.path.abspath(__file__)), "../.env"))
import json
import time
from openai import OpenAI
from rouge_score import rouge_scorer
from bert_score import score as bert_score

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

JUDGE_PROMPT = """You are an expert evaluator for a customer support AI assistant.

Evaluate the ACTUAL ANSWER against the EXPECTED ANSWER on three dimensions. Return a JSON object only.

Question: {question}
Context: {context}
Expected Answer: {expected_output}
Actual Answer: {actual_output}

Score each dimension from 1-5:
- correctness: Does the actual answer convey the same key facts as the expected answer? (1=completely wrong, 5=perfectly correct)
- groundedness: Does the actual answer stay faithful to the provided context without hallucinating? (1=makes things up, 5=fully grounded)
- completeness: Does the actual answer cover all the important points from the expected answer? (1=misses most points, 5=covers everything)

Return ONLY this JSON (no markdown, no explanation):
{{"correctness": <1-5>, "groundedness": <1-5>, "completeness": <1-5>, "reason": "<one sentence explaining any issues or confirming quality>"}}"""


def judge_single(item: dict, judge_model: str = "gpt-4o", pass_cutoff: float = 3.5) -> dict:
    """Run the judge on a single evaluated item."""
    prompt = JUDGE_PROMPT.format(
        question=item["input"],
        context=item["context"],
        expected_output=item["expected_output"],
        actual_output=item.get("actual_output", ""),
    )

    response = client.chat.completions.create(
        model=judge_model,
        messages=[{"role": "user", "content": prompt}],
        temperature=0.0,
        max_tokens=200,
    )

    raw = response.choices[0].message.content.strip()
    scores = json.loads(raw)
    scores["avg_score"] = round(
        (scores["correctness"] + scores["groundedness"] + scores["completeness"]) / 3, 2
    )
    scores["passed"] = scores["avg_score"] >= pass_cutoff
    return scores


def run_evaluation(pipeline_outputs: list[dict], judge_model: str = "gpt-4o", pass_cutoff: float = 3.5) -> list[dict]:
    """Run the judge over all pipeline outputs and calculate hybrid semantic metrics."""
    results = []
    
    # Initialize ROUGE
    r_scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
    
    # Compute BERTScore in batch for performance
    print("🧠 Computing BERTScore and ROUGE hybrid metrics...")
    actuals = [str(item.get("actual_output", "")) for item in pipeline_outputs]
    expecteds = [str(item.get("expected_output", "")) for item in pipeline_outputs]
    
    # We use a smaller fast model to avoid huge downloads during dev
    P, R, F1 = bert_score(actuals, expecteds, lang="en", model_type="distilbert-base-uncased", verbose=False)
    
    for i, item in enumerate(pipeline_outputs):
        print(f"Judging [{i+1}/{len(pipeline_outputs)}] {item['id']}...")
        
        # Hybrid Metrics
        rouge_scores = r_scorer.score(expecteds[i], actuals[i])
        rouge_1 = round(rouge_scores['rouge1'].fmeasure, 2)
        rouge_2 = round(rouge_scores['rouge2'].fmeasure, 2)
        rouge_l = round(rouge_scores['rougeL'].fmeasure, 2)
        bert_f1 = round(F1[i].item(), 2)
        
        try:
            scores = judge_single(item, judge_model, pass_cutoff)
            
            # Simple heuristic for 'needs_review' check logic
            normalized_judge = scores["avg_score"] / 5.0
            divergence = round(abs(normalized_judge - bert_f1), 2)
            needs_review = divergence > 0.3 or (not scores["passed"] and bert_f1 >= 0.70)
            
            results.append({
                **item, 
                **scores, 
                "judge_model": judge_model,
                "rouge1": rouge_1,
                "rouge2": rouge_2,
                "rougeL": rouge_l,
                "bertscore_f1": bert_f1,
                "score_divergence": divergence,
                "needs_review": needs_review
            })
        except Exception as e:
            print(f"  ⚠️ Error judging {item['id']}: {e}")
            results.append({
                **item,
                "correctness": None, "groundedness": None,
                "completeness": None, "avg_score": None,
                "passed": False, "reason": str(e),
                "judge_model": judge_model,
                "rouge1": rouge_1,
                "rouge2": rouge_2,
                "rougeL": rouge_l,
                "bertscore_f1": bert_f1,
                "score_divergence": 0.0,
                "needs_review": True
            })
        time.sleep(0.5)  # Rate limit buffer
    return results



def compute_summary(eval_results: list[dict]) -> dict:
    """Compute aggregate metrics and slice-based breakdowns."""
    valid = [r for r in eval_results if r.get("avg_score") is not None]

    def avg(key):
        return round(sum(r[key] for r in valid) / len(valid), 2) if valid else 0

    # Overall metrics
    summary = {
        "total": len(eval_results),
        "evaluated": len(valid),
        "pass_rate": round(sum(1 for r in valid if r["passed"]) / len(valid) * 100, 1),
        "avg_correctness": avg("correctness"),
        "avg_groundedness": avg("groundedness"),
        "avg_completeness": avg("completeness"),
        "avg_overall": avg("avg_score"),
    }

    # Slice by difficulty
    for difficulty in ["easy", "medium", "hard"]:
        slice_items = [r for r in valid if r["metadata"]["difficulty"] == difficulty]
        if slice_items:
            summary[f"avg_{difficulty}"] = round(
                sum(r["avg_score"] for r in slice_items) / len(slice_items), 2
            )
            summary[f"count_{difficulty}"] = len(slice_items)

    # Slice by scenario_tag
    tags = set(r["metadata"]["scenario_tag"] for r in valid)
    summary["by_scenario"] = {}
    for tag in tags:
        tag_items = [r for r in valid if r["metadata"]["scenario_tag"] == tag]
        summary["by_scenario"][tag] = {
            "count": len(tag_items),
            "avg_score": round(sum(r["avg_score"] for r in tag_items) / len(tag_items), 2),
            "pass_rate": round(sum(1 for r in tag_items if r["passed"]) / len(tag_items) * 100, 1),
        }

    return summary


if __name__ == "__main__":
    pipeline_path = os.path.join(os.path.dirname(__file__), "../data/pipeline_outputs.json")
    with open(pipeline_path, "r") as f:
        pipeline_outputs = json.load(f)

    eval_results = run_evaluation(pipeline_outputs)
    summary = compute_summary(eval_results)

    # Save detailed results
    results_path = os.path.join(os.path.dirname(__file__), "../data/eval_results.json")
    with open(results_path, "w") as f:
        json.dump(eval_results, f, indent=2)

    # Save summary
    summary_path = os.path.join(os.path.dirname(__file__), "../data/eval_summary.json")
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)

    print(f"\n✅ Evaluation complete!")
    print(f"   Pass rate: {summary['pass_rate']}%")
    print(f"   Avg score: {summary['avg_overall']}/5")
    print(f"   Results saved to {results_path}")
