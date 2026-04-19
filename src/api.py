import os
import json
from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse

app = FastAPI(title="GoldyLoopAI SaaS Backend")

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "../data")
DASHBOARD_DIR = os.path.join(BASE_DIR, "../dashboard")

from fastapi import Response

@app.get("/app_data.js")
def get_app_data():
    eval_path = os.path.join(DATA_DIR, "eval_results.json")
    dataset_path = os.path.join(DATA_DIR, "golden_dataset.json")
    runs_dir = os.path.join(DATA_DIR, "runs")

    # Load Golden
    clean_golden = []
    try:
        with open(dataset_path, "r") as f:
            for g in json.load(f):
                clean_golden.append({
                    "id": g["id"],
                    "input": g.get("input", ""),
                    "tag": g.get("metadata", {}).get("scenario_tag", "unknown"),
                    "difficulty": g.get("metadata", {}).get("difficulty", "medium"),
                    "source": g.get("metadata", {}).get("source", ""),
                    "risk": g.get("metadata", {}).get("risk_level", "low")
                })
    except Exception:
        pass

    # Load Eval
    cleaned_results = []
    try:
        with open(eval_path, "r") as f:
            valid = [r for r in json.load(f) if r.get("avg_score") is not None]
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
    except Exception:
        pass

    # Load History
    all_runs = []
    if os.path.exists(runs_dir):
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
    
    return Response(content=js_content, media_type="application/javascript")

# Mount static dashboard folder to /
app.mount("/", StaticFiles(directory=DASHBOARD_DIR, html=True), name="dashboard")

if __name__ == "__main__":
    import uvicorn
    print("\n🔥 Starting GoldyLoopAI Fast API Server 🔥")
    print("Dashboard available at: http://127.0.0.1:8000")
    uvicorn.run(app, host="127.0.0.1", port=8000)
