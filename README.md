# @AppleSupport Customer Support AI Agent & Evaluation Harness

[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org/downloads/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.110%2B-009688.svg)](https://fastapi.tiangolo.com)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

> **Hiver SDE Intern Take-Home Assignment Solution**  
> A customer support AI pipeline for **@AppleSupport** featuring 7-class intent classification, TF-IDF retrieval grounding, deterministic safety escalation, Gemini LLM-as-a-Judge evaluation with human inter-rater reliability measurement, and an interactive web dashboard.

---

## 🚀 Quickstart

### 1. Clone & Install Dependencies
```bash
git clone https://github.com/<your-repo>/hiver-sde-intern-assignment.git
cd hiver-sde-intern-assignment
pip install -r requirements.txt
```

### 2. Configure API Keys

**Kaggle (for real dataset ingestion):**
```bash
# Option A: Place credentials file
# Download from https://www.kaggle.com/settings → API → Create New Token
# Save to ~/.kaggle/kaggle.json

# Option B: Environment variables
export KAGGLE_USERNAME="your_username"
export KAGGLE_KEY="your_api_key"
```

**Gemini LLM (for LLM judge & reply generation):**
```bash
# Get a free API key at https://aistudio.google.com/
export GEMINI_API_KEY="your_gemini_api_key"
```

> **Note:** The system works without API keys — it falls back to a synthetic corpus and heuristic scoring. API keys enable the full pipeline with real data and LLM features.

### 3. Initialise Data Pipeline (Optional)

> [!NOTE]
> **Step 3 is only needed if you want to regenerate the dataset from scratch using your own Kaggle API credentials.**  
> The repository already ships with the pre-generated, real, 1,504-record deduplicated Kaggle dataset (`data/apple_support_corpus.json` and `data/golden_eval_set.json`), so reviewers can **skip straight to Step 4**. Running this step without Kaggle credentials will overwrite the real dataset with a synthetic fallback.

```bash
# Optional: only run if regenerating from raw Kaggle data with credentials
python -m src.data_pipeline
```

### 4. Run Automated Test Suite
```bash
python -m pytest -v
```

### 5. Run Headline Benchmark Evaluation
```bash
python -m src.eval_harness
```

### 6. Launch Interactive Web Dashboard
```bash
python -m uvicorn src.server:app --host 127.0.0.1 --port 8000
```
Open **[http://127.0.0.1:8000](http://127.0.0.1:8000)** to test live queries, compare baselines side-by-side, and view benchmark charts.

---

## 📊 Data Pipeline & Dataset

The system ingests the real **[Kaggle Customer Support on Twitter](https://www.kaggle.com/datasets/thoughtvector/customer-support-on-twitter)** dataset:

1. **Download** via `kagglehub` (cached after first download)
2. **Thread reconstruction** from flat tweet CSV using `response_tweet_id` / `in_response_to_tweet_id`
3. **Filter** to @AppleSupport brand conversations
4. **Clean** real tweet noise: anonymised handles (`@115712`), t.co URLs, emoji, whitespace
5. **Label** intents and escalation decisions via keyword heuristics
6. **Deduplicate** training corpus against golden eval set (0% leakage verified)

**Fallback mode:** If Kaggle credentials are not available, the pipeline uses a synthetic corpus (clearly labeled) for offline reproducibility.

---

## 🏗️ System Architecture

```
                                  [ Inbound Customer Tweet ]
                                              │
                   ┌──────────────────────────┴──────────────────────────┐
                   ▼                                                     ▼
   [ Deterministic Safety Filter ]                           [ Feature Vectoriser ]
   (Battery Swelling / Fire / PII)                           (1-3 Gram Sublinear TF-IDF)
                   │                                                     │
                   │ (If Critical Hazard)                                ▼
                   │                                          [ Intent Classifier ]
                   │                                       (7-Class Calibrated LogReg)
                   │                                                     │
                   ▼                                                     ▼
     [ Immediate Safety Escalation ]                          [ TF-IDF Retrieval Engine ]
     (Override with Urgent Warning)                       (Historical @AppleSupport Pairs)
                   │                                                     │
                   └──────────────────────────┬──────────────────────────┘
                                              ▼
                                 [ Escalation Policy Engine ]
                                 (Auto-Handle vs Human + Stated Reason)
                                              │
                                              ▼
                                 [ Draft Reply Generation ]
                                 (Gemini LLM or Template Fallback)
                                              │
                                              ▼
                                 [ LLM-as-a-Judge Evaluation ]
                                 (Gemini API or Heuristic Fallback)
```

---

## 📁 Repository Structure

```
├── data/
│   ├── apple_support_corpus.json          # Training corpus (real Kaggle data or synthetic fallback)
│   ├── golden_eval_set.json               # 200 annotated test cases (JSON)
│   ├── golden_eval_set.csv                # 200 annotated test cases (CSV)
│   ├── human_judge_ratings.json           # 50 scored samples (20 manual + 30 programmatic)
│   ├── leakage_verification.json          # Train/eval leakage check results
│   └── evaluation_benchmark_results.json  # Benchmark output
├── src/
│   ├── __init__.py
│   ├── config.py                          # 7-class taxonomy, escalation reasons, brand metadata
│   ├── data_pipeline.py                   # Kaggle ingestion, thread parsing, corpus building
│   ├── intent_classifier.py               # TF-IDF + LogReg + regex heuristic intent engine
│   ├── retrieval_engine.py                # TF-IDF cosine similarity retrieval
│   ├── escalation_policy.py               # Deterministic & risk-aware triage router
│   ├── agent.py                           # Unified support pipeline (LLM or template reply)
│   ├── llm_client.py                      # Gemini API wrapper with graceful degradation
│   ├── judge.py                           # Gemini LLM judge + heuristic fallback scorer
│   ├── eval_harness.py                    # Multi-baseline benchmarking with leakage check
│   └── server.py                          # FastAPI backend
├── web/
│   ├── index.html                         # Interactive web application
│   ├── style.css                          # Dark theme stylesheet
│   └── app.js                             # Client-side state & visualisation engine
├── tests/
│   ├── test_agent.py                      # End-to-end agent tests
│   ├── test_escalation.py                 # Escalation policy safety tests
│   ├── test_intent.py                     # Intent classification unit tests
│   ├── test_judge.py                      # Judge rubric tests
│   ├── test_failure_modes.py              # Regression tests for 5 documented failure modes
│   ├── test_server.py                     # FastAPI endpoint integration tests
│   ├── test_data_pipeline.py              # Data pipeline & leakage verification tests
│   └── test_retrieval_engine.py           # Retrieval engine tests
├── REPORT.md                              # Technical & evaluation report
├── DECISION_LOG.md                        # 18 engineering decisions & trade-offs
├── README.md                              # This file
└── requirements.txt                       # Project dependencies
```

---

## 🎯 Intent Taxonomy (7 Classes)

1. `OS_UPDATE_BUG`: Issues after iOS/macOS updates, boot loops, freezes, app crashes.
2. `HARDWARE_BATTERY`: Degraded battery health, overheating, swollen enclosures, physical ports.
3. `ACCOUNT_ICLOUD_SECURITY`: Locked Apple ID, 2FA, forgotten passwords, phishing alerts.
4. `BILLING_SUBSCRIPTIONS`: Duplicate charges, App Store refund requests, subscription cancellation.
5. `CONNECTIVITY_SETUP`: Bluetooth dropouts, AirPods/Apple Watch pairing, Wi-Fi glitches, CarPlay.
6. `REPAIR_WARRANTY_STATUS`: AppleCare+ claims, Genius Bar appointments, repair tracking.
7. `GENERAL_FEEDBACK_CHURN`: Brand sentiment, feature requests, competitor churn threats.

---

## 🛡️ Escalation Policies

| Reason | Risk Level | Trigger |
|:---|:---|:---|
| `HARDWARE_PHYSICAL_DAMAGE` | Critical | Battery swelling, liquid submersion, thermal hazard |
| `HIGH_SENTIMENT_CHURN_RISK` | High | Legal threats, extreme frustration, competitor churn |
| `BILLING_REFUND_AUTH` | Medium | Financial disputes, duplicate charges |
| `PII_SECURITY_DM` | Medium | Apple ID recovery, serial number collection |
| `LOW_CONFIDENCE_AMBIGUITY` | Low | Model confidence < 0.35, ambiguous intent |
| `STANDARD_TROUBLESHOOTING` | Auto-Handle | Self-serve guides, force restarts |
| `PUBLIC_DOCS_GUIDE` | Auto-Handle | Official Apple knowledgebase links |

---

## 📄 Documentation
- **Full Report:** [REPORT.md](./REPORT.md)
- **Decision Log:** [DECISION_LOG.md](./DECISION_LOG.md)
