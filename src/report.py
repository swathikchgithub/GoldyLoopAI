"""
GoldyLoopAI - Report Generator (v1.1)
Generates a slice-based eval report with hybrid metrics and exports to CSV.
Run after run_pipeline.py has produced eval_results.json.
"""

import os
from dotenv import load_dotenv
load_dotenv(os.path.join(os.path.dirname(os.path.abspath(__file__)), "../.env"))
import json
import csv
from datetime import datetime


def generate_report(eval_results_path: str, output_dir: str) -> None:
    with open(eval_results_path, "r") as f:
        results = json.load(f)

    valid = [r for r in results if r.get("avg_score") is not None]
    has_hybrid = any(r.get("rouge1") is not None for r in valid)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # ── CSV Export ──────────────────────────────────────────────────────
    fieldnames = [
        "id", "scenario_tag", "difficulty", "risk_level", "source", "model",
        "correctness", "groundedness", "completeness", "avg_score", "passed",
        # v1.1 hybrid metrics
        "rouge1", "rouge2", "rougeL", "bertscore_f1", "score_divergence", "needs_review",
        "reason", "input", "expected_output", "actual_output",
    ]
    csv_path = os.path.join(output_dir, f"eval_report_{timestamp}.csv")
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for r in valid:
            writer.writerow({
                "id": r["id"],
                "scenario_tag": r["metadata"]["scenario_tag"],
                "difficulty": r["metadata"]["difficulty"],
                "risk_level": r["metadata"]["risk_level"],
                "source": r["metadata"]["source"],
                "model": r.get("model", ""),
                "correctness": r.get("correctness"),
                "groundedness": r.get("groundedness"),
                "completeness": r.get("completeness"),
                "avg_score": r.get("avg_score"),
                "passed": r.get("passed"),
                "rouge1": r.get("rouge1", "N/A"),
                "rouge2": r.get("rouge2", "N/A"),
                "rougeL": r.get("rougeL", "N/A"),
                "bertscore_f1": r.get("bertscore_f1", "N/A"),
                "score_divergence": r.get("score_divergence", "N/A"),
                "needs_review": r.get("needs_review", "N/A"),
                "reason": r.get("reason", ""),
                "input": r["input"],
                "expected_output": r["expected_output"],
                "actual_output": r.get("actual_output", ""),
            })
    print(f"📊 CSV report saved: {csv_path}")

    # ── Text Summary ────────────────────────────────────────────────────
    def avg(key):
        vals = [r[key] for r in valid if r.get(key) is not None]
        return round(sum(vals) / len(vals), 4) if vals else 0

    lines = [
        "=" * 62,
        "  GoldyLoopAI — Evaluation Report (v1.1)",
        f"  Generated: {datetime.now().strftime('%Y-%m-%d %H:%M')}",
        "=" * 62,
        f"  Total examples   : {len(results)}",
        f"  Evaluated        : {len(valid)}",
        f"  Pass rate        : {round(sum(1 for r in valid if r['passed']) / len(valid) * 100, 1)}%",
        "",
        "── LLM Judge Scores ───────────────────────────────────────",
        f"  Avg Correctness  : {avg('correctness')} / 5",
        f"  Avg Groundedness : {avg('groundedness')} / 5",
        f"  Avg Completeness : {avg('completeness')} / 5",
        f"  Avg Overall      : {avg('avg_score')} / 5",
    ]

    if has_hybrid:
        divergent = [r for r in valid if r.get("needs_review")]
        lines += [
            "",
            "── Hybrid Metrics (v1.1) ──────────────────────────────────",
            f"  Avg ROUGE-1      : {avg('rouge1')}",
            f"  Avg ROUGE-2      : {avg('rouge2')}",
            f"  Avg ROUGE-L      : {avg('rougeL')}",
            f"  Avg BERTScore-F1 : {avg('bertscore_f1')}",
            f"  Divergent (⚠️ )  : {len(divergent)} examples need review",
        ]
        if divergent:
            for r in divergent:
                lines.append(f"    → {r['id']} | judge={r.get('avg_score')} | rougeL={r.get('rougeL')} | bert={r.get('bertscore_f1')} | Δ={r.get('score_divergence')}")

    lines += [
        "",
        "── By Difficulty ──────────────────────────────────────────",
    ]
    for diff in ["easy", "medium", "hard"]:
        items = [r for r in valid if r["metadata"]["difficulty"] == diff]
        if items:
            score = round(sum(r["avg_score"] for r in items) / len(items), 2)
            lines.append(f"  {diff.capitalize():8s} ({len(items)} items): {score}/5")

    lines += ["", "── By Scenario ────────────────────────────────────────────"]
    tags = sorted(set(r["metadata"]["scenario_tag"] for r in valid))
    for tag in tags:
        tag_items = [r for r in valid if r["metadata"]["scenario_tag"] == tag]
        score = round(sum(r["avg_score"] for r in tag_items) / len(tag_items), 2)
        pass_rate = round(sum(1 for r in tag_items if r["passed"]) / len(tag_items) * 100)
        lines.append(f"  {tag:12s} ({len(tag_items)} items): {score}/5  pass={pass_rate}%")

    lines += ["", "── Failures ───────────────────────────────────────────────"]
    failures = [r for r in valid if not r["passed"]]
    if failures:
        for r in failures:
            lines.append(f"  ❌ {r['id']} [{r['metadata']['difficulty']}] judge={r['avg_score']}", )
            if has_hybrid and r.get("rougeL"):
                lines.append(f"     ROUGE-L={r.get('rougeL')} | BERTScore={r.get('bertscore_f1')}")
            lines.append(f"     Reason: {r.get('reason', 'N/A')}")
    else:
        lines.append("  🎉 All examples passed!")

    lines.append("=" * 62)
    report_text = "\n".join(lines)
    print("\n" + report_text)

    txt_path = os.path.join(output_dir, f"eval_report_{timestamp}.txt")
    with open(txt_path, "w") as f:
        f.write(report_text)
    print(f"\n📄 Text report saved: {txt_path}")

    # ── Dashboard Data (app_data.js) ────────────────────────────────────
    dashboard_dir = os.path.join(output_dir, "../dashboard")
    os.makedirs(dashboard_dir, exist_ok=True)
    app_data_path = os.path.join(dashboard_dir, "app_data.js")

    cleaned_results = []
    for r in valid:
        cleaned_results.append({
            "id": r["id"],
            "input": r["input"],
            "tag": r["metadata"]["scenario_tag"],
            "difficulty": r["metadata"]["difficulty"],
            "source": r["metadata"]["source"],
            "risk": r["metadata"]["risk_level"],
            "correctness": r.get("correctness"),
            "groundedness": r.get("groundedness"),
            "completeness": r.get("completeness"),
            "avg": r.get("avg_score"),
            "rougeL": r.get("rougeL"),
            "bertscore": r.get("bertscore_f1"),
            "needs_review": r.get("needs_review", False),
            "passed": r.get("passed", False),
            "reason": r.get("reason", "")
        })

    dataset_path = os.path.join(output_dir, "golden_dataset.json")
    try:
        with open(dataset_path, "r") as f:
            golden_data = json.load(f)
            clean_golden = []
            for g in golden_data:
                clean_golden.append({
                    "id": g["id"],
                    "input": g.get("input", ""),
                    "tag": g.get("metadata", {}).get("scenario_tag", "unknown"),
                    "difficulty": g.get("metadata", {}).get("difficulty", "medium"),
                    "source": g.get("metadata", {}).get("source", ""),
                    "risk": g.get("metadata", {}).get("risk_level", "low")
                })
    except Exception:
        clean_golden = []

    # ── History Archiving ───────────────────────────────────────────────
    runs_dir = os.path.join(output_dir, "runs")
    os.makedirs(runs_dir, exist_ok=True)
    
    current_run_overview = {
        "timestamp": timestamp,
        "pass_rate": round(sum(1 for r in valid if r.get('passed', False)) / max(len(valid), 1) * 100, 1),
        "avg_score": round(sum(r.get('avg_score', 0) for r in valid) / max(len(valid), 1), 2)
    }

    current_run_file = os.path.join(runs_dir, f"run_{timestamp}.json")
    with open(current_run_file, "w") as f:
        json.dump({"summary": current_run_overview, "results": cleaned_results}, f)

    # Load all historical runs to generate trend and regression diff
    all_runs = []
    for fname in sorted(os.listdir(runs_dir)):
        if fname.endswith(".json"):
            try:
                with open(os.path.join(runs_dir, fname), "r") as f:
                    all_runs.append(json.load(f))
            except Exception:
                pass

    previous_eval_data = []
    if len(all_runs) > 1:
        previous_eval_data = all_runs[-2].get("results", [])

    history_summaries = [run["summary"] for run in all_runs if "summary" in run]

    js_content = f"window.goldenDataset = {json.dumps(clean_golden)};\n"
    js_content += f"window.evalData = {json.dumps(cleaned_results)};\n"
    js_content += f"window.previousEvalData = {json.dumps(previous_eval_data)};\n"
    js_content += f"window.runHistory = {json.dumps(history_summaries)};\n"

    with open(app_data_path, "w") as f:
        f.write(js_content)
    print(f"🖥️  Dashboard data updated: dashboard/app_data.js")


if __name__ == "__main__":
    eval_path = os.path.join(os.path.dirname(__file__), "../data/eval_results.json")
    output_dir = os.path.join(os.path.dirname(__file__), "../data")
    generate_report(eval_path, output_dir)
