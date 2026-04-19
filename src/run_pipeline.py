"""
GoldyLoopAI - Full Pipeline Runner (v1.1)
Runs the complete eval loop end-to-end:
  1. Validate golden dataset
  2. Run RAG app on every golden example
  3. Score outputs with LLM-as-a-judge
  4. Add ROUGE + BERTScore hybrid metrics  ← NEW in v1.1
  5. Generate report + CSV

Usage:
    python src/run_pipeline.py
    python src/run_pipeline.py --model gpt-4o --judge gpt-4o
    python src/run_pipeline.py --skip-hybrid   # skip ROUGE/BERTScore (faster)
"""

import os
import sys
import json
import argparse
from dotenv import load_dotenv
load_dotenv(os.path.join(os.path.dirname(os.path.abspath(__file__)), "../.env"))

from golden_builder import validate_dataset
from app import run_on_golden_dataset
from evaluator import run_evaluation, compute_summary
from hybrid_scorer import add_hybrid_scores_to_results, hybrid_summary
from report import generate_report


def main(app_model: str = "gpt-4o-mini", judge_model: str = "gpt-4o", skip_hybrid: bool = False, min_pass_rate: float = 0.0, pass_cutoff: float = 3.5):
    base_dir = os.path.dirname(os.path.abspath(__file__))
    data_dir = os.path.join(base_dir, "../data")
    dataset_path = os.path.join(data_dir, "golden_dataset.json")

    print("\n🌟 GoldyLoopAI — Starting Evaluation Loop (v1.1)")
    print("=" * 55)

    # Step 1: Validate golden dataset
    print("\n[1/5] Validating golden dataset...")
    with open(dataset_path) as f:
        dataset = json.load(f)
    report = validate_dataset(dataset)
    if not report["is_valid"]:
        print(f"❌ Dataset validation failed: {report['errors']}")
        return
    print(f"  ✅ {report['total_items']} examples valid | Tags: {report['scenario_tags']}")

    # Step 2: Run RAG pipeline
    print(f"\n[2/5] Running RAG pipeline (model={app_model})...")
    pipeline_outputs = run_on_golden_dataset(dataset_path, model=app_model)
    pipeline_path = os.path.join(data_dir, "pipeline_outputs.json")
    with open(pipeline_path, "w") as f:
        json.dump(pipeline_outputs, f, indent=2)
    print(f"  ✅ {len(pipeline_outputs)} outputs generated")

    # Step 3: LLM-as-a-Judge evaluation
    print(f"\n[3/5] Running LLM-as-a-Judge (judge={judge_model})...")
    eval_results = run_evaluation(pipeline_outputs, judge_model=judge_model, pass_cutoff=pass_cutoff)
    summary = compute_summary(eval_results)

    # Step 4: Hybrid scoring (ROUGE + BERTScore)
    if not skip_hybrid:
        print(f"\n[4/5] Computing hybrid metrics (ROUGE + BERTScore)...")
        eval_results = add_hybrid_scores_to_results(eval_results)
        h_summary = hybrid_summary(eval_results)
        summary["hybrid"] = h_summary
        print(f"  ✅ Avg ROUGE-L: {h_summary.get('avg_rougeL')} | Avg BERTScore-F1: {h_summary.get('avg_bertscore_f1')}")
        if h_summary.get("divergent_count", 0) > 0:
            print(f"  ⚠️  {h_summary['divergent_count']} divergent examples flagged: {h_summary['divergent_ids']}")
    else:
        print(f"\n[4/5] Skipping hybrid metrics (--skip-hybrid flag set)")

    # Save results
    eval_path = os.path.join(data_dir, "eval_results.json")
    with open(eval_path, "w") as f:
        json.dump(eval_results, f, indent=2)
    summary_path = os.path.join(data_dir, "eval_summary.json")
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)

    # Step 5: Generate report
    print(f"\n[5/5] Generating report...")
    generate_report(eval_path, data_dir)

    print(f"\n🎉 GoldyLoopAI eval complete!")
    print(f"   LLM Judge Pass rate : {summary['pass_rate']}%")
    print(f"   Avg judge score     : {summary['avg_overall']}/5")
    if not skip_hybrid and "hybrid" in summary:
        print(f"   Avg ROUGE-L         : {summary['hybrid'].get('avg_rougeL')}")
        print(f"   Avg BERTScore-F1    : {summary['hybrid'].get('avg_bertscore_f1')}")
        print(f"   {summary['hybrid'].get('interpretation', {}).get('rouge_vs_bert', '')}")
        
    if min_pass_rate > 0.0:
        if summary['pass_rate'] < min_pass_rate:
            print(f"\n❌ Pipeline failed: Pass rate {summary['pass_rate']}% is below required minimum of {min_pass_rate}%")
            sys.exit(1)
        else:
            print(f"\n✅ Pipeline passed: Pass rate {summary['pass_rate']}% meets required minimum of {min_pass_rate}%")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="GoldyLoopAI pipeline runner v1.1")
    parser.add_argument("--model", default="gpt-4o-mini", help="App model for RAG pipeline")
    parser.add_argument("--judge", default="gpt-4o", help="Judge model for evaluation")
    parser.add_argument("--skip-hybrid", action="store_true", help="Skip ROUGE/BERTScore (faster runs)")
    parser.add_argument("--min-pass-rate", type=float, default=0.0, help="Fail the pipeline if overall pass rate is below this percentage (0-100)")
    parser.add_argument("--pass-cutoff", type=float, default=3.5, help="Avg score out of 5 required for an item to pass (default 3.5)")
    args = parser.parse_args()
    main(app_model=args.model, judge_model=args.judge, skip_hybrid=args.skip_hybrid, min_pass_rate=args.min_pass_rate, pass_cutoff=args.pass_cutoff)
