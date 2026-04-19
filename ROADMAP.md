# 🗺️ GoldyLoopAI Roadmap

Here is the complete future roadmap for GoldyLoopAI — organized from quick wins to big ambitious features:

## 🔧 Core Improvements (v1.1 — Quick Wins)
- [ ] Add ROUGE / BERTScore metrics alongside LLM-as-judge for hybrid scoring — deterministic + semantic
- [ ] Configurable pass threshold — make the `avg_score >= 3.5` cutoff a CLI param (`--pass-threshold 4.0`)
- [ ] Per-dimension weights — let users weight correctness more than completeness for safety-critical domains
- [ ] Retry logic + exponential backoff in `evaluator.py` for rate limit resilience
- [ ] Colored terminal output with `rich` library for a polished CLI experience
- [ ] `--dry-run` flag — validate dataset and show pipeline plan without calling the API
- [ ] Cost estimator — print estimated API cost before running the pipeline

## 📊 Dashboard Enhancements (v1.2)
- [ ] Regression diff view — compare two eval runs side-by-side, highlight which examples regressed
- [ ] Export to CSV/PDF button directly from dashboard UI
- [ ] Filter & search the results table by tag, difficulty, pass/fail
- [ ] Trend chart — track avg score over multiple runs (v1, v2, v3 history)
- [ ] Judge reasoning panel — click any row to see the full judge explanation inline
- [ ] Dark/light theme persistence using URL hash instead of localStorage

## 🗂️ Golden Dataset Builder (v1.3)
- [ ] Synthetic data generation CLI — `python src/golden_builder.py --generate 20 --kb docs/faq.txt`
- [ ] Duplicate & near-duplicate detection using embedding cosine similarity (not just string match)
- [ ] Inter-annotator agreement scoring — run two judges and compute Krippendorff's alpha to calibrate reliability
- [ ] Dataset versioning — auto-tag dataset with a version hash and store history in `data/versions/`
- [ ] Human review queue — export borderline silver examples to a simple web form for SME approval
- [ ] Domain-specific scenario templates — starter datasets for HR bots, legal Q&A, medical triage

## 🤖 Evaluation Engine Upgrades (v2.0)
- [ ] Multi-turn / conversational eval — score multi-step conversations, not just single Q&A pairs
- [ ] Agent eval mode — evaluate LLM agents with tool calls, not just single-output pipelines
- [ ] Component-level eval — separately score the retriever (recall@k, nDCG) vs. the generator (correctness)
- [ ] Multimodal support — accept image + text inputs for vision-capable models
- [ ] Custom judge prompts — YAML-defined rubrics per scenario tag, not one global judge prompt
- [ ] Multiple judge ensemble — run 3 judges and take majority vote to reduce judge bias
- [ ] Reference-free eval mode — score outputs without ground truth using coherence + self-consistency

## 🔁 CI/CD & Production Integration (v2.1)
- [ ] GitHub Actions workflow — auto-run eval on every PR that touches `src/` or `data/`
- [ ] Eval gates — fail the PR if pass rate drops below a configurable threshold
- [ ] Production log sampler — connect to an API endpoint or S3 bucket and sample live traffic
- [ ] Drift detector — alert when production score distribution shifts significantly from baseline
- [ ] Slack / webhook alerts — post regression summaries to a Slack channel on each run
- [ ] MLflow / W&B integration — log scores, metadata, and artifacts to experiment tracking tools

## 🛡️ Safety & Compliance Layer (v2.2)
- [ ] Toxicity judge — dedicated evaluator that flags harmful, biased, or unsafe outputs
- [ ] PII detection — auto-redact names, emails, phone numbers before storing production logs
- [ ] Hallucination scorer — specialized metric using NLI (Natural Language Inference) to detect fabricated facts
- [ ] Regulatory compliance checks — HIPAA / GDPR / SOC2 safe handling tags per example
- [ ] Risk-weighted scoring — penalize failures on `risk_level: high` examples more heavily in the overall score

## 🧪 Advanced Research Features (v3.0)
- [ ] Prompt sensitivity testing — automatically paraphrase each golden input 5 ways and measure score variance
- [ ] Model comparison leaderboard — run the same golden set against GPT-4o, Claude 3.5, Gemini, Llama and rank them
- [ ] Fine-tuning data exporter — convert high-scoring eval pairs into JSONL for OpenAI fine-tuning
- [ ] Active learning loop — identify the most uncertain examples and surface them for human review first
- [ ] Traceability graph — link each score back to exact prompt version + model version + dataset version
- [ ] Red-teaming module — auto-generate adversarial variants of every golden example to stress-test the model

## 🌐 Ecosystem & Community (v3.x)
- [ ] Plugin system — let users drop in custom evaluator classes via a simple `BaseEvaluator` interface
- [ ] HuggingFace dataset push — one-command publish your golden dataset to the HF Hub
- [ ] REST API mode — expose the eval pipeline as a FastAPI service for integration into existing MLOps stacks
- [ ] GoldyLoopAI Hub — a community repo of shareable golden datasets by domain
- [ ] VS Code extension — highlight failing golden examples in the editor as you edit your prompt files
