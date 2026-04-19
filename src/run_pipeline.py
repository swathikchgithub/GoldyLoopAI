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
from rich import print


def estimate_cost(dataset: list, app_model: str, judge_model: str) -> float:
    import tiktoken
    try:
        enc_app = tiktoken.encoding_for_model(app_model)
    except KeyError:
        enc_app = tiktoken.get_encoding("cl100k_base")
    try:
        enc_judge = tiktoken.encoding_for_model(judge_model)
    except KeyError:
        enc_judge = tiktoken.get_encoding("cl100k_base")

    # Appoximate tokens for app context and judge evaluation
    app_prompt_tokens = sum(len(enc_app.encode(item["input"] + (" " * 800))) for item in dataset)
    app_comp_tokens = len(dataset) * 150

    judge_prompt_tokens = sum(len(enc_judge.encode(item["input"] + item.get("expected_output", "") + (" " * 800))) for item in dataset) + app_comp_tokens
    judge_comp_tokens = len(dataset) * 50

    # Pricing estimates
    app_cost = (app_prompt_tokens / 1_000_000 * 0.150) + (app_comp_tokens / 1_000_000 * 0.600)
    judge_cost = (judge_prompt_tokens / 1_000_000 * 5.0) + (judge_comp_tokens / 1_000_000 * 15.0)

    return round(app_cost + judge_cost, 4)


def main(app_model: str = "gpt-4o-mini", judge_model: str = "gpt-4o", skip_hybrid: bool = False, min_pass_rate: float = 0.0, pass_cutoff: float = 3.5, weights: dict = None, dry_run: bool = False):
    base_dir = os.path.dirname(os.path.abspath(__file__))
    data_dir = os.path.join(base_dir, "../data")
    dataset_path = os.path.join(data_dir, "golden_dataset.json")

    print("\n[bold magenta]🌟 GoldyLoopAI — Starting Evaluation Loop (v1.2)[/bold magenta]")
    print("[dim]" + "=" * 55 + "[/dim]")

    # Step 1: Validate golden dataset
    print("\n[bold cyan][1/5] Validating golden dataset...[/bold cyan]")
    with open(dataset_path) as f:
        dataset = json.load(f)
    report = validate_dataset(dataset)
    if not report["is_valid"]:
        print(f"[bold red]❌ Dataset validation failed:[/bold red] {report['errors']}")
        return
    print(f"  [green]✅[/green] {report['total_items']} examples valid | Tags: {report['scenario_tags']}")
    
    # Estimate Cost (Feature #7)
    est_cost = estimate_cost(dataset, app_model, judge_model)
    print(f"  [bold yellow]💸 Estimated API Cost:[/bold yellow] ~${est_cost} USD")
    
    if dry_run:
        print("\n[bold yellow]🛑 Dry run active. Exiting before API calls.[/bold yellow]")
        return

    # Step 2: Run RAG pipeline
    print(f"\n[bold cyan][2/5] Running RAG pipeline (model={app_model})...[/bold cyan]")
    pipeline_outputs = run_on_golden_dataset(dataset_path, model=app_model)
    pipeline_path = os.path.join(data_dir, "pipeline_outputs.json")
    with open(pipeline_path, "w") as f:
        json.dump(pipeline_outputs, f, indent=2)
    print(f"  [green]✅[/green] {len(pipeline_outputs)} outputs generated")

    # Step 3: LLM-as-a-Judge evaluation
    print(f"\n[bold cyan][3/5] Running LLM-as-a-Judge (judge={judge_model})...[/bold cyan]")
    eval_results = run_evaluation(pipeline_outputs, judge_model=judge_model, pass_cutoff=pass_cutoff, weights=weights)
    summary = compute_summary(eval_results)

    # Step 4: Hybrid scoring (ROUGE + BERTScore)
    if not skip_hybrid:
        print(f"\n[bold cyan][4/5] Computing hybrid metrics (ROUGE + BERTScore)...[/bold cyan]")
        eval_results = add_hybrid_scores_to_results(eval_results)
        h_summary = hybrid_summary(eval_results)
        summary["hybrid"] = h_summary
        print(f"  [green]✅[/green] Avg ROUGE-L: [bold]{h_summary.get('avg_rougeL')}[/bold] | Avg BERTScore-F1: [bold]{h_summary.get('avg_bertscore_f1')}[/bold]")
        if h_summary.get("divergent_count", 0) > 0:
            print(f"  [bold yellow]⚠️  {h_summary['divergent_count']} divergent examples flagged:[/bold yellow] {h_summary['divergent_ids']}")
    else:
        print(f"\n[bold cyan][4/5] Skipping hybrid metrics (--skip-hybrid flag set)[/bold cyan]")

    # Save results
    eval_path = os.path.join(data_dir, "eval_results.json")
    with open(eval_path, "w") as f:
        json.dump(eval_results, f, indent=2)
    summary_path = os.path.join(data_dir, "eval_summary.json")
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)

    # Step 5: Generate report
    print(f"\n[bold cyan][5/5] Generating report...[/bold cyan]")
    generate_report(eval_path, data_dir)

    print(f"\n[bold green]🎉 GoldyLoopAI eval complete![/bold green]")
    print(f"   [bold]LLM Judge Pass rate[/bold] : {summary['pass_rate']}%")
    print(f"   [bold]Avg judge score[/bold]     : {summary['avg_overall']}/5")
    if not skip_hybrid and "hybrid" in summary:
        print(f"   [bold]Avg ROUGE-L[/bold]         : {summary['hybrid'].get('avg_rougeL')}")
        print(f"   [bold]Avg BERTScore-F1[/bold]    : {summary['hybrid'].get('avg_bertscore_f1')}")
        print(f"   [dim]{summary['hybrid'].get('interpretation', {}).get('rouge_vs_bert', '')}[/dim]")
        
    if min_pass_rate > 0.0:
        if summary['pass_rate'] < min_pass_rate:
            print(f"\n[bold red]❌ Pipeline failed:[/bold red] Pass rate {summary['pass_rate']}% is below required minimum of {min_pass_rate}%")
            sys.exit(1)
        else:
            print(f"\n[bold green]✅ Pipeline passed:[/bold green] Pass rate {summary['pass_rate']}% meets required minimum of {min_pass_rate}%")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="GoldyLoopAI pipeline runner v1.1")
    parser.add_argument("--model", default="gpt-4o-mini", help="App model for RAG pipeline")
    parser.add_argument("--judge", default="gpt-4o", help="Judge model for evaluation")
    parser.add_argument("--skip-hybrid", action="store_true", help="Skip ROUGE/BERTScore (faster runs)")
    parser.add_argument("--dry-run", action="store_true", help="Validate dataset and estimate costs without calling APIs")
    parser.add_argument("--min-pass-rate", type=float, default=0.0, help="Fail the pipeline if overall pass rate is below this percentage (0-100)")
    parser.add_argument("--pass-cutoff", type=float, default=3.5, help="Avg score out of 5 required for an item to pass (default 3.5)")
    parser.add_argument("--weight-correctness", type=float, default=1.0, help="Weight for correctness score")
    parser.add_argument("--weight-groundedness", type=float, default=1.0, help="Weight for groundedness score")
    parser.add_argument("--weight-completeness", type=float, default=1.0, help="Weight for completeness score")
    args = parser.parse_args()
    
    weights = {
        "correctness": args.weight_correctness,
        "groundedness": args.weight_groundedness,
        "completeness": args.weight_completeness
    }
    
    main(
        app_model=args.model, 
        judge_model=args.judge, 
        skip_hybrid=args.skip_hybrid, 
        min_pass_rate=args.min_pass_rate, 
        pass_cutoff=args.pass_cutoff,
        weights=weights,
        dry_run=args.dry_run
    )
