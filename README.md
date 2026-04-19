<div align="center">

<img src="https://img.shields.io/badge/GoldyLoopAI-v1.2.0-f5c842?style=for-the-badge&labelColor=0f0f0d" alt="version"/>
<img src="https://img.shields.io/badge/Python-3.10+-blue?style=for-the-badge&logo=python&logoColor=white" alt="python"/>
<img src="https://img.shields.io/badge/OpenAI-GPT--4o-412991?style=for-the-badge&logo=openai&logoColor=white" alt="openai"/>
<img src="https://img.shields.io/badge/License-MIT-green?style=for-the-badge" alt="license"/>

# 🌟 GoldyLoopAI (v1.2)

**Build golden datasets. Close the eval loop.**

A complete, production-grade LLM evaluation framework that takes you from raw Q&A pairs to a slice-based regression dashboard, fortified with CI/CD PR Gates, automated cost estimators, historical data archiving, and semantic metrics scoring.

[Live Dashboard](#dashboard) · [Quick Start](#quick-start) · [Core Concepts](#how-it-works) · [Project Structure](#project-structure)

</div>

---

## What is GoldyLoopAI?

GoldyLoopAI is a hands-on project that demonstrates the **complete LLM evaluation lifecycle**:

1. 🗂️ **Golden Dataset** — curated Q&A pairs with ground truth, difficulty tags, and metadata
2. 🤖 **RAG Pipeline** — an LLM-powered Q&A system that answers from context
3. ⚖️ **LLM-as-a-Judge** — a stronger model scores outputs across 3 dimensions
4. 📊 **Slice Analysis** — break down failures by scenario, difficulty, and risk level
5. 🔁 **The Loop** — every regression discovered becomes a new golden example

> Built for learning, interviews, and demos. No fine-tuning required — just an OpenAI API key.

---

## Quick Start

```bash
# 1. Clone the repo
git clone https://github.com/YOUR_USERNAME/GoldyLoopAI.git
cd GoldyLoopAI

# 2. Create virtual environment
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Set your API key
cp .env.example .env
# Edit .env and add your OPENAI_API_KEY

# 5. Run the full eval loop!
python src/run_pipeline.py --model gpt-4o-mini --judge gpt-4o
```

Then open `dashboard/index.html` in your browser to explore the results. ✨

### 🛡️ Advanced CLI Controls (v1.2)
GoldyLoopAI supports advanced CLI controls for integration into GitHub Actions, CI/CD pipelines, and cost-constrained environments:

```bash
# Dry run to estimate API costs via Tiktoken before executing
python src/run_pipeline.py --dry-run

# Adjust pass thresholds and dimension weights (prioritize groundedness heavily)
python src/run_pipeline.py --pass-cutoff 4.0 --weight-groundedness 2.0 --weight-completeness 0.5 

# Enforce a CI/CD Pull Request Quality Gate (Exits code 1 if pass rate drops)
python src/run_pipeline.py --min-pass-rate 95.0
```

---

## How It Works

```
Golden Dataset → RAG Pipeline → LLM-as-Judge → Slice Report → Fix & Loop
     ↑                                                              |
     └──────────────── New golden examples from failures ──────────┘
```

### The Golden Dataset
A golden dataset is a **stable, versioned collection of inputs + expected outputs** used to make release decisions. Each example includes:
- `input` — the user's question
- `context` — the retrieved knowledge base snippet
- `expected_output` — the ideal answer (ground truth)
- `metadata` — difficulty, scenario tag, source, risk level

### LLM-as-a-Judge
The judge model (GPT-4o) scores each response on **3 dimensions**:

| Dimension | What it measures | Score |
|---|---|---|
| **Correctness** | Are the key facts right? | 1–5 |
| **Groundedness** | Does it stay faithful to context (no hallucinations)? | 1–5 |
| **Completeness** | Are all important points covered? | 1–5 |

A response **passes** if its average score ≥ 3.5.

### Slice-Based Analysis
Instead of a single accuracy number, GoldyLoopAI breaks down scores by:
- Difficulty tier (easy / medium / hard)
- Scenario tag (returns, shipping, payments, etc.)
- Risk level (low / medium / high)

This tells you *where* the model fails, not just *that* it fails.

---


## Hybrid Scoring (v1.1)

GoldyLoopAI now supports **hybrid evaluation** by combining an LLM judge with deterministic and semantic metrics:

| Metric | Purpose |
|---|---|
| **ROUGE-1 / ROUGE-2 / ROUGE-L** | Measures lexical overlap between expected and actual answers |
| **BERTScore F1** | Measures semantic similarity using transformer embeddings |
| **LLM Judge** | Scores correctness, groundedness, and completeness |

Why this matters:
- **ROUGE** is fast and deterministic, good as a sanity check.
- **BERTScore** captures semantic similarity even when wording changes.
- **LLM-as-a-Judge** adds richer qualitative scoring.
- If these signals diverge a lot, GoldyLoopAI flags the example for **manual review**.

Run with hybrid metrics enabled by default:

```bash
python src/run_pipeline.py
```

Skip hybrid scoring for a faster run:

```bash
python src/run_pipeline.py --skip-hybrid
```

You can also run the hybrid scorer directly:

```bash
python src/hybrid_scorer.py
```

## Dashboard (v1.2 Upgrades)

Open `dashboard/index.html` in any browser — no server needed. The backend dynamically builds `dashboard/app_data.js` and securely loops your history into `data/runs/`.

Features:
- **Historical Trend Chart** — Watch your overall pass rate dynamically climb or regress over time on the main Overview.
- **Regression Diff Engine** — Mathematically maps your current evaluation against your previous `n-1` run and highlights score differences.
- **Interactive UI Filtering** — Search, filter by scenario tags instantly, or filter by pass/fail via dropdowns.
- **Expandable Judge Reasoning** — Click any table row to drop down the LLM Judge's textual reasoning parameter.
- **Export to CSV** — Download semantic scores and outputs with one click.

---

## Project Structure

```
GoldyLoopAI/
├── data/
│   ├── golden_dataset.json       ← The curated ground truth (versioned)
│   ├── pipeline_outputs.json     ← RAG app answers (generated, gitignored)
│   ├── eval_results.json         ← Judge scores (generated, gitignored)
│   └── validation_report.json   ← Dataset quality report
│
├── src/
│   ├── app.py                    ← RAG Q&A pipeline (OpenAI + context)
│   ├── evaluator.py              ← LLM-as-a-Judge scorer + summary
│   ├── hybrid_scorer.py          ← ROUGE + BERTScore hybrid metrics
│   ├── golden_builder.py         ← Dataset validation + synthetic generation
│   ├── report.py                 ← CSV + text report generator
│   └── run_pipeline.py           ← Full pipeline runner (one command)
│
├── dashboard/
│   └── index.html                ← Interactive eval results dashboard
│
├── notebooks/
│   └── eval_walkthrough.ipynb    ← Step-by-step Jupyter notebook
│
├── .env.example                  ← Environment variable template
├── requirements.txt
└── README.md
```

---

## Key Concepts (Interview Ready)

### Why golden datasets matter
> "A golden dataset is your contract with the model — a frozen, human-verified set of inputs and expected outputs. Every deployment decision, prompt change, and model swap gets tested against it. Without it, you're guessing."

### Why LLM-as-a-Judge?
> "Human evaluation doesn't scale to production traffic. An LLM judge runs at API speed, scores consistently, and surfaces regressions before they reach users. It's not perfect — judges can be biased toward longer outputs — but calibrating it against a subset of human labels keeps it honest."

### Why slice-based analysis?
> "A 90% pass rate looks great until you notice it's 100% on easy examples and 0% on adversarial ones. Slicing by difficulty, scenario, and risk level gives you actionable signal. You're not just measuring performance — you're finding exactly where to focus your prompt engineering."

### The loop
> "The most powerful part is the feedback loop. Every production failure becomes a new golden example. The dataset grows, coverage improves, and each release is tested against a more complete picture of real user behavior."

---

## Running Individual Steps

```bash
# Validate your golden dataset
python src/golden_builder.py

# Run just the RAG pipeline
python src/app.py

# Run just the judge evaluation
python src/evaluator.py

# Generate CSV + text report
python src/report.py

# Full pipeline with model overrides
python src/run_pipeline.py --model gpt-4o --judge gpt-4o
```

---

## Simulating a Regression

Change the system prompt in `app.py` (e.g., make it more restrictive) and re-run:
```bash
python src/run_pipeline.py
```
Compare the new scores against your previous run. The dashboard will show which scenarios regressed.

---

## Tech Stack

- **OpenAI API** — GPT-4o-mini (app) + GPT-4o (judge)
- **Python 3.10+** — Core pipeline
- **Pandas** — Slice analysis and CSV export
- **Chart.js** — Dashboard visualizations
- **FAISS** (optional) — Vector similarity for real RAG extension

---

## License

MIT — built by **Swathi Chadalavada** · [GoldyLoopAI](https://github.com/YOUR_USERNAME/GoldyLoopAI)

> *"Build golden datasets. Close the eval loop."* 🌟
