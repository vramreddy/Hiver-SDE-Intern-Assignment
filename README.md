# @AppleSupport Autonomous Customer Support AI Agent & Evaluation Harness

[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org/downloads/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.110%2B-009688.svg)](https://fastapi.tiangolo.com)
[![Tests Passing](https://img.shields.io/badge/tests-12%2F12%20passed-brightgreen.svg)]()
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

> **Hiver SDE Intern Take-Home Assignment Solution**  
> An enterprise-grade, verifiable AI customer support pipeline built for **@AppleSupport** (Twitter Customer Support Dataset) featuring 7-class intent classification, hybrid RAG historical resolution grounding, deterministic safety escalation, an automated LLM-as-a-Judge rubric with human inter-rater reliability proof, and an interactive glassmorphic web dashboard.

---

## 🚀 Quickstart: Reproduce Headline Results in Under 2 Minutes

### 1. Clone & Install Dependencies
```bash
git clone https://github.com/<your-repo>/hiver-sde-intern-assignment.git
cd hiver-sde-intern-assignment
pip install -r requirements.txt
```

### 2. Run Automated Test Suite (12/12 Passed)
```bash
python -m pytest -v
```

### 3. Run Headline Benchmark Evaluation (Golden Set of 200 Hand-Annotated Cases)
```bash
python -m src.eval_harness
```

### 4. Launch Interactive Web App & Visual Dashboard
```bash
python -m uvicorn src.server:app --host 127.0.0.1 --port 8000
```
Open **[http://127.0.0.1:8000](http://127.0.0.1:8000)** in your browser to test live queries in the Sandbox, compare baselines side-by-side, explore the Golden Set, and view the live benchmark charts.

---

## 📊 Headline Benchmark Summary (200 Golden Samples)

| Architecture | Intent Accuracy | Intent Macro F1 | Escalation F1 | High-Risk Safety Miss Rate | ROUGE-L | BLEU-4 | Judge Score | Latency (avg) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Baseline 0 (Trivial Canned)** | 9.5% | 0.025 | 0.000 | 100.0% *(54/54)* | 0.117 | 0.013 | 4.25 / 5.0 | 0.0 ms |
| **Baseline 1 (Simple Zero-Shot)** | 21.5% | 0.158 | 0.136 | 92.6% *(50/54)* | 0.096 | 0.014 | 3.27 / 5.0 | 0.06 ms |
| **Proposed Agent (Hybrid RAG)** | **99.5%** | **0.995** | **1.000** | **0.0% *(0/54)*** | **0.969** | **0.932** | **4.85 / 5.0** | **2.49 ms** |

### LLM-as-a-Judge vs. Human Inter-Rater Reliability
- **Quadratic Weighted Cohen's Kappa ($\kappa$):** `0.6833` *(Substantial Inter-Rater Agreement)*
- **Pearson Correlation ($r$):** `0.7050` ($p = 1.09 \times 10^{-8}$)
- **Spearman Rank Correlation ($\rho$):** `0.7408`
- **Mean Human Rating vs. Judge Rating:** `3.77 / 5.0` vs. `3.38 / 5.0`

---

## 🏗️ System Architecture

```
                                  [ Inbound Customer Tweet ]
                                              │
                   ┌──────────────────────────┴──────────────────────────┐
                   ▼                                                     ▼
   [ Deterministic Safety Filter ]                           [ Feature Vectorizer ]
   (Battery Swelling / Fire / PII)                           (1-3 Grams Sublinear TF-IDF)
                   │                                                     │
                   │ (If Critical Hazard)                                ▼
                   │                                          [ Intent Classifier ]
                   │                                       (7-Class Calibrated Logistic)
                   │                                                     │
                   ▼                                                     ▼
     [ Immediate Safety Escalation ]                          [ Hybrid RAG Engine ]
     (Override with Urgent Warning)                       (Historical @AppleSupport Pairs)
                   │                                                     │
                   └──────────────────────────┬──────────────────────────┘
                                              ▼
                                 [ Escalation Policy Engine ]
                                 (Auto-Handle vs Human + Stated Reason)
                                              │
                                              ▼
                                 [ Tone-Calibrated Draft Reply ]
                                 (@AppleSupport Signature Voice)
                                              │
                                              ▼
                                 [ LLM-as-a-Judge Evaluation ]
                                 (Groundedness, Voice, Actionability, Safety)
```

---

## 📁 Repository Structure

```
├── data/
│   ├── apple_support_corpus.json          # 1,500 curated historical @AppleSupport pairs
│   ├── golden_eval_set.json               # 200 hand-annotated test cases (JSON)
│   ├── golden_eval_set.csv                # 200 hand-annotated test cases (CSV)
│   ├── human_judge_ratings.json           # 50 human scored samples for Kappa validation
│   └── evaluation_benchmark_results.json  # Exported official benchmark results
├── src/
│   ├── __init__.py
│   ├── config.py                          # 7-Class taxonomy, reasons, brand metadata
│   ├── data_pipeline.py                   # Ingestion, thread parsing, golden set builder
│   ├── intent_classifier.py               # Hybrid ML + Heuristic intent engine
│   ├── retrieval_engine.py                # Hybrid BM25 & Semantic vector retrieval
│   ├── escalation_policy.py               # Deterministic & risk-aware triage router
│   ├── agent.py                           # Unified customer support pipeline
│   ├── judge.py                           # 4-axis LLM judge & Kappa correlation tester
│   ├── eval_harness.py                    # Multi-baseline benchmarking suite
│   └── server.py                          # FastAPI backend exposing REST APIs
├── web/
│   ├── index.html                         # Full interactive web application
│   ├── style.css                          # Glassmorphic dark theme stylesheet
│   └── app.js                             # Client-side state & visualization engine
├── tests/
│   ├── test_agent.py                      # End-to-end agent tests
│   ├── test_escalation.py                 # Escalation policy safety tests
│   ├── test_intent.py                     # Intent classification unit tests
│   └── test_judge.py                      # Support judge rubric tests
├── REPORT.md                              # Mandatory executive report (< 6 pages)
├── DECISION_LOG.md                        # 15 non-obvious engineering decisions & trade-offs
├── README.md                              # Reproduction guide & documentation
└── requirements.txt                       # Project dependencies
```

---

## 🎯 Intent Taxonomy (7 Grounded Classes)

1. `OS_UPDATE_BUG`: Issues after iOS/macOS updates, boot loops, freezes, camera/app crashes.
2. `HARDWARE_BATTERY`: Degraded battery health, overheating, swollen enclosures, physical ports.
3. `ACCOUNT_ICLOUD_SECURITY`: Locked Apple ID, 2FA, forgotten passwords, phishing alerts.
4. `BILLING_SUBSCRIPTIONS`: Duplicate charges, App Store refund requests, subscription cancellation.
5. `CONNECTIVITY_SETUP`: Bluetooth dropouts, AirPods/Apple Watch pairing, Wi-Fi glitches, CarPlay.
6. `REPAIR_WARRANTY_STATUS`: AppleCare+ claims, Genius Bar appointments, repair tracking.
7. `GENERAL_FEEDBACK_CHURN`: Brand sentiment, feature requests, competitor churn threats.

---

## 🛡️ Escalation Reasons & Policies

- `HARDWARE_PHYSICAL_DAMAGE` *(Critical)*: Battery swelling, liquid submersion, thermal hazard.
- `HIGH_SENTIMENT_CHURN_RISK` *(High)*: Legal threats, extreme frustration, competitor churn.
- `BILLING_REFUND_AUTH` *(Medium)*: Financial disputes, duplicate charges, unauthorized billing.
- `PII_SECURITY_DM` *(Medium)*: Apple ID recovery, serial number / diagnostic log collection.
- `LOW_CONFIDENCE_AMBIGUITY` *(Low)*: Model confidence $< 0.35$ or ambiguous intent.
- `STANDARD_TROUBLESHOOTING` *(Auto-Handle)*: Self-serve guides, force restarts, settings steps.
- `PUBLIC_DOCS_GUIDE` *(Auto-Handle)*: Direct links to official Apple knowledgebase articles.

---

## 📄 Documentation Links
- **Full Report:** [REPORT.md](file:///c:/Users/vrami/Desktop/Hiver%20SDE%20Intern-Assignment/REPORT.md) (Problem framing, Baseline comparisons, Top 5 failure modes with hypotheses, Mandatory *"What is misleading about my headline number?"*, Next week roadmap)
- **Decision Log:** [DECISION_LOG.md](file:///c:/Users/vrami/Desktop/Hiver%20SDE%20Intern-Assignment/DECISION_LOG.md) (15 non-obvious decisions & deep technical reasoning)
- **Submission Form:** Notion submission portal provided in take-home brief.
