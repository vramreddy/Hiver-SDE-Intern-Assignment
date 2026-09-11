# Technical & Evaluation Report: Customer Support AI Agent for @AppleSupport

**Author:** Hiver SDE Intern Candidate  
**Target Brand:** @AppleSupport (Twitter Customer Support Dataset)  
**Deliverable Type:** Engineering & Empirical Evaluation Report (Max 6 Pages)  

---

## Executive Summary

This report presents the architecture and empirical evaluation of a Customer Support AI Agent built for **@AppleSupport** using the [Kaggle Customer Support on Twitter](https://www.kaggle.com/datasets/thoughtvector/customer-support-on-twitter) dataset. The system reconstructs multi-turn tweet threads, extracts Apple Support conversations, trains a hybrid TF-IDF + Logistic Regression intent classifier, and evaluates responses using a Gemini LLM-as-a-Judge rubric.

Evaluated against a 200-sample **Golden Evaluation Set** (deduplicated against the training corpus with verified 0% leakage), the system is benchmarked against a trivial canned baseline and a simple keyword baseline.

> **Note on LLM integration:** Reply generation and judge scoring use the **Gemini 2.0 Flash API** when `GEMINI_API_KEY` is set. Without it, the system falls back to template-based retrieval (for replies) and a heuristic keyword scorer (for judge), both clearly labeled in output metadata.

---

## 1. Problem Framing & Operational Scope

### 1.1 What "Good" Means for @AppleSupport

On public Twitter customer support channels, "good" support differs sharply from generic chat:

1. **Absolute Public Privacy & Zero PII Leakage:** The agent must never request or expose sensitive credentials in public tweets.
2. **Actionable First-Turn Triage:** Replies must provide official, verified Apple troubleshooting subdomains (`apple.co`, `locate.apple.com`, `iforgot.apple.com`, `reportaproblem.apple.com`) or concise self-serve steps.
3. **Safety-Critical Hardware Interception:** Catastrophic hardware hazards (e.g., lithium battery swelling, thermal anomalies) must bypass automated resolution immediately with urgent safety advisories.
4. **Sub-10ms High-Throughput Triage:** Twitter support queues experience burst traffic during major OS releases.

### 1.2 Data Pipeline: From Raw Kaggle Dataset to Clean Corpus

The system ingests the real **thoughtvector/customer-support-on-twitter** dataset (3M+ tweets):

1. Downloaded via `kagglehub` with automatic caching
2. Flat tweet CSV reconstructed into multi-turn threads using `response_tweet_id` / `in_response_to_tweet_id`
3. Filtered to @AppleSupport brand conversations (identified by tweet content heuristics)
4. Real tweet noise handled: anonymised handles (`@115712` → `@user`), t.co URLs → `[URL]`, whitespace normalisation
5. Each pair labeled with intent (7-class heuristic) and escalation decision
6. Training corpus explicitly deduplicated against golden eval set (verified 0% leakage)

**Fallback:** When Kaggle credentials are unavailable, the pipeline uses a clearly-labeled synthetic corpus for offline reproducibility.

### 1.3 What We Intentionally Chose NOT to Build

- **Open-Ended Conversational Chitchat** — increases hallucination risk without customer value.
- **Automated Public Password Resets** — prevents social engineering vectors.
- **Unbounded Multi-Turn Autonomous Dialogue** — high-stakes disputes escalated to humans.

---

## 2. Empirical Benchmark Results vs. Baselines

Three architectures evaluated on the 200-sample Golden Evaluation Set:

1. **Baseline 0 (Trivial):** Majority class predictor (`OS_UPDATE_BUG`), static auto-handle, canned reply.
2. **Baseline 1 (Simple):** Keyword classifier, punctuation escalation rule, generic ungrounded reply.
3. **Proposed System:** 7-class TF-IDF + Calibrated Logistic Regression + Heuristic Boosting, TF-IDF cosine retrieval, deterministic multi-criteria escalation router, Gemini LLM reply generation (or template fallback).

> **Benchmark numbers update after re-running with leakage-free data.** Results will be populated by running `python -m src.eval_harness`.

### 2.1 Judge Validation & Inter-Rater Reliability

The judge evaluation uses either:
- **Gemini LLM** (when `GEMINI_API_KEY` is set): Structured rubric prompt scoring 4 dimensions
- **Heuristic fallback** (no key): Keyword-based deterministic scorer

Inter-rater agreement against 50 human-annotated samples (methodology: 20 genuine author self-annotations + 30 calibrated programmatic scores, transparently documented in `data/human_judge_ratings.json`):

- **Quadratic Weighted Cohen's Kappa (κ):** Measured at runtime
- **Pearson Correlation (r):** Measured at runtime
- **Methodology disclosure:** The "human" ratings combine 20 genuine manual author ratings (with specific per-case annotation notes) and 30 calibrated programmatic ratings. This is explicitly labeled in the JSON output (`annotation_method: "manual"` vs `"programmatic_rubric"`).

---

## 3. In-Depth Failure Analysis: Top 5 Failure Modes

Through stress-testing, we identified 5 distinct edge-case failure modes with regression tests in `tests/test_failure_modes.py`:

### 1. Adversarial Multi-Intent Queries (Software vs. Physical Hazard)
- **Example (constructed adversarial):** *"@AppleSupport WatchOS 10 completely ruined my battery life and my Apple Watch battery swells up when on the magnetic charger!!"*
- **Root Cause:** The message contains software update keywords alongside severe physical damage. Standard NLP models prioritise `OS_UPDATE_BUG` and miss the fire hazard.
- **Mitigation:** Priority-Ordered Safety Pre-Filter intercepts thermal/swelling keywords before intent classification, enforcing immediate hardware escalation.
- **Regression test:** `test_failure_modes.py::TestAdversarialMultiIntent`

### 2. Sarcasm and Inverted Polarity
- **Example (constructed adversarial):** *"@AppleSupport Brilliant new feature in iOS 17 where alarms just decide to stay completely silent and make me late for work! 😡"*
- **Root Cause:** Sarcastic phrasing causes lexical models to classify as positive feedback.
- **Mitigation:** N-gram TF-IDF (1-3 tokens) with domain context preserves technical bug keywords over superficial cheerfulness.
- **Regression test:** `test_failure_modes.py::TestSarcasmInvertedPolarity`

### 3. Public Password Reset & Phishing Lures
- **Example (constructed adversarial):** *"@AppleSupport I forgot my Apple ID passcode. Can you reset it for me over Twitter?"*
- **Root Cause:** Generic agents attempt to assist by soliciting credentials publicly.
- **Mitigation:** Strict policy guardrails prohibit interactive credential verification, routing to `iforgot.apple.com`.
- **Regression test:** `test_failure_modes.py::TestPIIPhishingTrap`

### 4. Hardware Baseband vs. Software Network Glitches
- **Example (constructed adversarial):** *"@AppleSupport Wi-Fi button on my iPhone is greyed out and Bluetooth toggle spins forever."*
- **Root Cause:** Greyed-out Wi-Fi indicates physical desoldering of the Wi-Fi IC chip, not a software glitch.
- **Mitigation:** Pattern matching for `greyed out` / `toggle spins` triggers hardware diagnostic routing.
- **Regression test:** `test_failure_modes.py::TestHardwareSoftwareAmbiguity`

### 5. Multi-Turn Context Truncation
- **Example (constructed adversarial):** *"@AppleSupport Done that already. Still not working."*
- **Root Cause:** Single-turn inference lacks context from prior tweets.
- **Mitigation:** Low-confidence thresholding (< 0.35) catches context-deficient queries and routes to human queues.
- **Regression test:** `test_failure_modes.py::TestContextTruncation`

---

## 4. Mandatory Section: "What is misleading about my headline number?"

### 1. Train/Eval Data Leakage (Fixed)
- **The Problem:** In the initial version, 31.5% of the golden eval examples were exact duplicates of training corpus entries, inflating accuracy to 99.5%. This was the most critical issue.
- **The Fix:** The data pipeline now explicitly deduplicates the golden set against the training corpus and runs `verify_no_leakage()` before every benchmark. The leakage verification is saved to `data/leakage_verification.json` and logged in benchmark output.
- **Current Status:** 0% leakage verified.

### 2. Heuristic Intent Labels on Real Data
- **The Blind Spot:** Intent labels on real Kaggle tweets are assigned by keyword heuristics, not by human annotators reviewing each tweet. This means some labels may be incorrect, especially for ambiguous tweets.
- **Reality:** A human annotation pass on a random sample would likely reveal 5-15% label noise, which means real-world accuracy is lower than reported.

### 3. LLM Judge Availability
- **The Blind Spot:** If `GEMINI_API_KEY` is not set, the "LLM-as-a-Judge" falls back to a heuristic keyword scorer. This is not an LLM at all — it's a deterministic rule-based evaluator. The system is transparent about this in output metadata (`scorer: "heuristic_fallback"` vs `"gemini_llm"`).
- **Reality:** Full LLM judge results require a valid Gemini API key.

### 4. Synthetic Golden Set Realism Gap
- **The Blind Spot:** ~10 of the 200 golden eval cases are hand-crafted adversarial examples (clearly labeled as `source: "constructed_adversarial"`). These test important edge cases but don't represent natural tweet distribution.
- **Reality:** On purely organic, unseen Twitter streams, accuracy may be 10-15% lower due to OOV slang, emoji-only messages, and image-based error reporting.

### 5. ROUGE/BLEU Metric Traps
- **The Blind Spot:** ROUGE-L and BLEU-4 measure n-gram overlap, not semantic correctness. A technically wrong answer sharing 85% of words with the reference still scores high.
- **Reality:** ROUGE is a proxy for stylistic alignment, not task resolution. Real grounding needs live agent trial measurement.

### 6. Human Agreement Methodology
- **The Blind Spot:** The inter-rater agreement dataset uses a hybrid approach: 20 genuine author self-annotations and 30 calibrated programmatic ratings. The programmatic ratings are not from independent human raters — they follow a predetermined quality tier pattern.
- **Reality:** True inter-rater reliability requires multiple independent annotators. The current approach provides a calibration baseline, not a gold-standard agreement measurement.

---

## 5. What I Would Do Next with One More Week

1. **Full Human Annotation Pass:** Hire 2-3 annotators to independently label intent, escalation, and judge scores on 200 real tweets for proper inter-rater reliability.
2. **Multi-Turn Thread Context:** Integrate Twitter conversation ID graph traversal to reconstruct 3-5 turn thread history, resolving single-turn context truncation.
3. **Multilingual Embeddings:** Incorporate sentence transformers for multilingual support (Spanish, French, German, Japanese customer queries).
4. **Active Learning Feedback Loop:** Build pipeline to ingest human advisor corrections from DM queue to continuously retrain the classifier.
5. **Deploy Gemini Judge at Scale:** Batch-evaluate all 200 golden cases through the LLM judge and report both LLM and heuristic scores side-by-side.

---

## Conclusion

This system demonstrates that building a trustworthy AI support agent from real-world data requires handling messy tweet noise, preventing train/eval leakage, being honest about evaluation methodology, and providing transparent fallback behavior when external services (LLM APIs, Kaggle) are unavailable.
