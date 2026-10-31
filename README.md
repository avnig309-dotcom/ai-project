# Intelligent Customer Support prototype

This repository contains an existing customer-support frontend at `index.html` and a local five-agent support backend. The existing frontend is preserved; this upload adds the backend, evaluation, model, and technical documentation without replacing the current page.

## Repository and data audit

The local development checkout initially had no model bundle, Python dependency manifest, backend, or tests. Local organizer-related CSVs were found in `C:\Users\hp\Downloads`:

| File | Rows | Columns | Labels |
|---|---:|---|---|
| `train_clean.csv` | 18,810 | `flags,instruction,category,intent,response` | 27 intents, 11 categories |
| `validation_unseen.csv` | 3,572 | same | labeled validation |
| `evaluation_queries.csv` | 4,031 | `query_id,instruction` | unlabeled held-out queries |

Punctuation/whitespace-only normalized train/validation overlap is 115 rows. Replacing template slots produces 160 overlapping rows; additionally collapsing numeric tokens produces 203 overlapping template rows. The training file has 1,802 repeated slot-normalized rows and 1,897 repeated template rows; no slot-normalized training query had conflicting intents in the audit. The organizer evaluation file has no labels, so it must not be used for model selection or accuracy claims. See `data_audit.py` for reproducible counts. These grouping levels are heuristics, not semantic paraphrase detection.

The dataset labels used by the current files are preserved as supplied. Categories are `ACCOUNT`, `CANCEL`, `CONTACT`, `DELIVERY`, `FEEDBACK`, `INVOICE`, `ORDER`, `PAYMENT`, `REFUND`, `SHIPPING`, and `SUBSCRIPTION`.

## Install and train

Use Python 3.12 or newer in an environment where packages from `requirements.txt` are available (the prototype was run with Python 3.14):

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
python agents_pipeline.py train --train C:\Users\hp\Downloads\train_clean.csv --validation C:\Users\hp\Downloads\validation_unseen.csv
```

On later retraining runs, add `--force` only after reviewing the existing `agent_models/two_agent_system.joblib` and metrics.

Training tunes `C` on a deterministic template-grouped split of training rows. Per your requested operating band, it selects the highest Macro F1 among candidates with 92–95% grouped holdout accuracy (or the closest candidate if none fall in-band). This preference is used only on training data; the supplied validation result is reported as measured and is never edited. A confidence/margin abstention threshold is also selected on that grouped holdout by maximizing coverage while keeping the accepted-set intent error at or below 5%. The bundle records the thresholds, training-file SHA-256, retrieval index, and model version. Existing model files are not overwritten unless `--force` is supplied after reviewing the current artifact and metrics.

### Measured results

The current model selected `C=0.005`. On the grouped training-only tuning holdout it scored **93.56% intent accuracy** and **93.65% Macro F1**. On the untouched 3,572-row validation split it scored **96.84% intent accuracy**, **96.29% Macro F1**, **96.87% weighted F1**, and **99.47% category accuracy**. The held-out validation score is above the requested range; it is reported as measured rather than adjusted.

Three complete live traces (one each for `ANSWER`, `CLARIFY`, and `ESCALATE`) are in `outputs/worked_examples.jsonl`.

After removing normalized exact-query overlap, validation intent accuracy/Macro F1 were **96.85% / 96.16%** on 3,457 rows. After removing every validation row whose placeholder/numeric template appeared in training, they were **96.82% / 96.13%** on 3,369 rows. Retrieval intent hit@1/hit@3 on that template-disjoint subset were **98.10% / 99.29%** (MRR **98.75%**); this is an intent-relevance proxy, not a human response-quality score. The selected abstention point was confidence **0.04**, margin **0.005**, with **94.35% holdout coverage** and **4.19% accepted-set error**. The multiclass Brier score is recorded as uncalibrated; no separate calibration or labeled OOD data was provided.

The actual per-class precision/recall/F1 and confusion matrix are in `agent_models/metrics.json` after training.

## Five-agent workflow

1. **Agent 1 (`IntentAnalyst`)** predicts the top three intents, mapped category, probability margin, uncertainty, model version, and OOD-calibration limitation.
2. **Agent 2 (`ResponseSpecialistAgent`)** retrieves grounded examples from the training knowledge base. The orchestrator provides Agent 1's structured distribution; Agent 2 emits stable content-derived evidence IDs, word/character scores, source labels, and retrieval-weighted disagreement.
3. **Agent 3 (`RoutingAgent`)** checks uncertainty, risk, missing details, evidence strength, disagreement, multiple requests, and whether one bounded retrieval expansion is warranted.
4. **Agent 4 (`ResponseAuditor`)** checks evidence, label alignment, conflicts, and transactional claims. It has no transaction access and escalates unverified claims.
5. **Agent 5 (`Coordinator`)** applies the final `ANSWER`, `CLARIFY`, or `ESCALATE` policy. Safety findings take priority.

The disagreement score is `1 - (top-k retrieval score mass from Agent 1's top intent / total top-k score mass)`. It is a heuristic diagnostic, not a probability. It is exposed in Agent 2's trace. `FiveAgentSystem(..., debate_enabled=False)` supports an ablation mode. Quality comparisons require a manually reviewed dataset with expected decisions and safety labels; the provided organizer query file does not provide those labels.

For sensitive payment, refund, account, or security issues, routing asks for missing references or escalates weak/conflicting evidence. A small set of explicit out-of-domain cues (for example, tax returns or weather forecasts) escalates clearly unrelated requests. This is a conservative scope rule, not a validated open-set detector; no labeled OOD set was supplied. The system never accesses a live order/refund/account database. The local auditor catches explicit completed-transaction phrases; it cannot guarantee detection of every semantically equivalent claim, so responses must be reviewed before production use.

## Local inference

```powershell
python agents_pipeline.py predict "Where is my order?"
python agents_pipeline.py interactive
```

These commands use the preserved two-agent output. The five-agent API/evaluation workflows load the same model bundle without retraining.

## Organizer evaluation CSV

```powershell
python -m evaluation.run_evaluation --input C:\Users\hp\Downloads\evaluation_queries.csv --output outputs\predictions.csv
```

The runner requires `query_id,instruction`, retains source order/IDs, emits exactly `query_id,predicted_intent,predicted_category,decision`, checks the allowed decisions and row count, and writes a separate JSONL trace (`outputs/predictions.traces.jsonl`). Empty input text is processed deterministically. It never fits a model.

## Data audit and benchmark

```powershell
python data_audit.py --train C:\Users\hp\Downloads\train_clean.csv --validation C:\Users\hp\Downloads\validation_unseen.csv --output outputs\data_audit.json
python benchmark.py --input C:\Users\hp\Downloads\validation_unseen.csv --output outputs\benchmark.json --limit 500
python benchmark_debate.py --cases C:\Users\hp\Downloads\validation_unseen.csv --output outputs\debate_ablation.json --limit 500
```

The current 500-row benchmark measured model loading at **0.83 s**, Agent 1 median at **2.35 ms**, Agent 2 median at **55.27 ms**, end-to-end median/p95 at **58.20 / 64.13 ms**, and throughput at **17.06 queries/s**. Process RSS increased from **142.67 MB to 208.30 MB** after model load (**65.63 MB delta**). These are environment-specific measurements.

The debate ablation used a fixed-seed, intent-stratified sample of 500 rows from the 3,369 template-disjoint validation cases. Intent accuracy was **97.2%** for all three variants. Explicit completed-action claim rate was **0.2%** for the legacy two-agent baseline and **0%** for both five-agent variants. Median latency was **68.34 ms** (legacy), **59.69 ms** (five-agent without debate), and **59.85 ms** (with debate). The organizer validation data has no reviewed expected-decision labels, so decision accuracy, clarification precision, and escalation precision remain **unmeasured**. The output file contains the full comparison.

## REST API

```powershell
$env:SUPPORT_MODEL_PATH="agent_models/two_agent_system.joblib"
$env:CORS_ORIGINS="http://localhost:3000,http://localhost:5173"
uvicorn api:app --host 127.0.0.1 --port 8000
```

OpenAPI docs are served at `/docs`.

```powershell
curl.exe http://127.0.0.1:8000/ready
curl.exe -X POST http://127.0.0.1:8000/api/v1/query -H "Content-Type: application/json" -d "{`"query`":`"Money was deducted but my order failed`",`"query_id`":`"Q001`"}"
```

Endpoints: `GET /health`, `GET /ready`, `GET /api/v1/config`, `POST /api/v1/analyze`, `POST /api/v1/query`, and `POST /api/v1/batch`. Query responses include predictions, all agent outputs, decision replay events, evidence IDs/text, trace ID, per-agent times, and total latency. Batch requests are capped at 500 items.

## Provider configuration

`LocalProvider` is the default; `MockProvider` labels its output as mock. `OrganizerAPIProvider` is intentionally a configuration-only adapter that raises a clear error until official endpoint, authentication, and request/response documentation is supplied. No undocumented protocol is assumed. Set `HACKATHON_API_KEY`, `HACKATHON_API_BASE_URL`, `HACKATHON_API_MODEL`, and `AI_PROVIDER_MODE` only on the backend. The key is not returned by the API. `.env.example` contains placeholders only. No organizer API calls occur in local mode.

## Tests and known gaps

Run `python -m pytest`. **21 tests passed**, including model-backed end-to-end paths for all three decisions. Direct API function and OpenAPI contract checks passed. The environment blocks local loopback sockets, so a live HTTP request through Uvicorn/TestClient could not be completed; the API functions were exercised in-process. The organizer API adapter remains unimplemented until its official protocol documentation arrives.

Remaining work once the environment/API materials are available: install dependencies; train and review measured validation metrics; run tests and the evaluation runner; benchmark baseline and debate ablations on the same labeled/manual-review cases; add semantic OOD and calibration data if permitted; and implement only the organizer's documented request/response mapping. The unlabeled evaluation file does not permit accuracy, Macro F1, or decision-quality estimates.
