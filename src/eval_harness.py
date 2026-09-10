"""
Comprehensive Evaluation Harness for AI Support Agent & Baselines.
Evaluates:
- Baseline 0: Trivial Rule-Based / Majority Class / Canned Reply
- Baseline 1: Simple Zero-Shot / Naive Heuristic Classifier
- Proposed System: Hybrid Intent Classifier + RAG Grounding + Risk-Aware Escalation

Computes Intent F1/Accuracy, Escalation Precision/Recall/Miss Rate, ROUGE-L, BLEU-4,
Semantic Similarity, and 4-Axis LLM-as-a-Judge Scores on the 200-sample Golden Set.
"""

import os
import json
import time
import re
from typing import Dict, List, Any, Tuple
import numpy as np
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, confusion_matrix
from rouge_score import rouge_scorer
import nltk
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from .config import INTENT_CLASSES, INTENT_TAXONOMY
from .agent import AppleSupportAgent
from .judge import SupportJudge

DATA_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "data"))

# ----------------- BASELINES IMPLEMENTATION ----------------- #

class Baseline0_Trivial:
    """Trivial Baseline: Predicts majority class, never escalates, canned reply."""
    def __init__(self):
        self.majority_intent = "OS_UPDATE_BUG"
        self.canned_reply = "Thank you for reaching out to Apple Support. Please restart your device or visit apple.com for assistance."

    def process(self, query: str) -> Dict[str, Any]:
        return {
            "predicted_intent": self.majority_intent,
            "escalation_decision": "AUTO_HANDLE",
            "escalation_reason": "STANDARD_TROUBLESHOOTING",
            "draft_reply": self.canned_reply
        }

class Baseline1_Simple:
    """Simple Baseline: Naive keyword search, simple heuristic escalation, generic prompt reply."""
    def __init__(self):
        self.keywords = {
            "OS_UPDATE_BUG": ["update", "ios", "crash", "bug", "freeze"],
            "HARDWARE_BATTERY": ["battery", "screen", "hot", "broken", "charge"],
            "ACCOUNT_ICLOUD_SECURITY": ["apple id", "password", "icloud", "locked"],
            "BILLING_SUBSCRIPTIONS": ["refund", "charged", "bill", "subscription", "money"],
            "CONNECTIVITY_SETUP": ["bluetooth", "wifi", "airpods", "pair", "service"],
            "REPAIR_WARRANTY_STATUS": ["applecare", "repair", "warranty", "appointment"],
            "GENERAL_FEEDBACK_CHURN": ["feedback", "worst", "hate", "love", "service"]
        }

    def process(self, query: str) -> Dict[str, Any]:
        lower = query.lower()
        pred_intent = "GENERAL_FEEDBACK_CHURN"
        max_matches = 0
        for intent, kws in self.keywords.items():
            matches = sum(1 for kw in kws if kw in lower)
            if matches > max_matches:
                max_matches = matches
                pred_intent = intent

        # Naive escalation: if question contains 'dm' or exclamation marks or 'refund'
        should_escalate = any(w in lower for w in ["dm", "refund", "sue", "lawyer", "!"])
        decision = "ESCALATE_TO_HUMAN" if should_escalate else "AUTO_HANDLE"
        reason = "PII_SECURITY_DM" if should_escalate else "STANDARD_TROUBLESHOOTING"

        # Generic ungrounded reply
        draft_reply = f"Hello! We can help with your {pred_intent.lower().replace('_', ' ')} question. Please check our support website or reply if you need more help."
        if decision == "ESCALATE_TO_HUMAN":
            draft_reply += " Please send a DM."

        return {
            "predicted_intent": pred_intent,
            "escalation_decision": decision,
            "escalation_reason": reason,
            "draft_reply": draft_reply
        }

# ----------------- EVALUATION ENGINE ----------------- #

class EvaluationHarness:
    def __init__(self, golden_set_path: str, corpus_path: str):
        self.golden_set_path = golden_set_path
        self.corpus_path = corpus_path
        
        with open(golden_set_path, "r", encoding="utf-8") as f:
            self.golden_set = json.load(f)

        self.baseline0 = Baseline0_Trivial()
        self.baseline1 = Baseline1_Simple()
        self.proposed_agent = AppleSupportAgent(corpus_path)
        self.judge = SupportJudge()
        
        self.rouge_scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
        self.smooth_fn = SmoothingFunction().method1

    def compute_text_similarity(self, candidate: str, reference: str) -> Dict[str, float]:
        """Calculates ROUGE-1/2/L, BLEU-4, and TF-IDF Cosine Semantic Similarity."""
        # ROUGE
        rouge_res = self.rouge_scorer.score(reference, candidate)
        rouge1 = rouge_res['rouge1'].fmeasure
        rouge2 = rouge_res['rouge2'].fmeasure
        rougeL = rouge_res['rougeL'].fmeasure

        # BLEU-4
        cand_tokens = re.findall(r'\b\w+\b', candidate.lower()) if candidate else []
        ref_tokens = [re.findall(r'\b\w+\b', reference.lower())] if reference else [[]]
        bleu4 = sentence_bleu(ref_tokens, cand_tokens, weights=(0.25, 0.25, 0.25, 0.25), smoothing_function=self.smooth_fn)

        # Semantic Cosine Similarity
        try:
            vec = TfidfVectorizer().fit_transform([candidate, reference])
            sem_cos = float(cosine_similarity(vec[0:1], vec[1:2])[0][0])
        except Exception:
            sem_cos = 0.0

        return {
            "rouge1": round(float(rouge1), 4),
            "rouge2": round(float(rouge2), 4),
            "rougeL": round(float(rougeL), 4),
            "bleu4": round(float(bleu4), 4),
            "semantic_similarity": round(float(sem_cos), 4)
        }

    def run_full_benchmark(self) -> Dict[str, Any]:
        """Runs the complete comparative benchmark across all systems on the Golden Set."""
        print(f"[RUN] Starting benchmark evaluation on {len(self.golden_set)} golden examples...")
        
        systems = {
            "Baseline 0 (Trivial Canned)": self.baseline0,
            "Baseline 1 (Simple Zero-Shot)": self.baseline1,
            "Proposed Agent (Hybrid RAG)": self.proposed_agent
        }

        results: Dict[str, Any] = {}

        y_true_intent = [item["ground_truth_intent"] for item in self.golden_set]
        y_true_escalate = [1 if item["ground_truth_escalation"] else 0 for item in self.golden_set]
        references = [item["reference_reply"] for item in self.golden_set]
        queries = [item["customer_tweet"] for item in self.golden_set]

        for sys_name, model in systems.items():
            start_t = time.time()
            y_pred_intent = []
            y_pred_escalate = []
            replies = []
            latencies = []
            judge_scores = {"groundedness": [], "brand_voice": [], "actionability": [], "safety": [], "overall": []}
            text_metrics = {"rouge1": [], "rouge2": [], "rougeL": [], "bleu4": [], "semantic_similarity": []}

            for idx, item in enumerate(self.golden_set):
                q = item["customer_tweet"]
                t0 = time.time()
                
                if sys_name == "Proposed Agent (Hybrid RAG)":
                    out = model.process_query(q)
                    p_intent = out["intent"]["predicted_intent"]
                    p_esc = 1 if out["escalation"]["decision"] == "ESCALATE_TO_HUMAN" else 0
                    p_reply = out["draft_reply"]
                    esc_dec = out["escalation"]["decision"]
                else:
                    out = model.process(q)
                    p_intent = out["predicted_intent"]
                    p_esc = 1 if out["escalation_decision"] == "ESCALATE_TO_HUMAN" else 0
                    p_reply = out["draft_reply"]
                    esc_dec = out["escalation_decision"]

                latencies.append((time.time() - t0) * 1000)
                y_pred_intent.append(p_intent)
                y_pred_escalate.append(p_esc)
                replies.append(p_reply)

                # Text similarity vs Reference
                sim = self.compute_text_similarity(p_reply, item["reference_reply"])
                for k, v in sim.items():
                    text_metrics[k].append(v)

                # LLM Judge score
                j_eval = self.judge.evaluate_reply(q, p_reply, p_intent, esc_dec)
                judge_scores["groundedness"].append(j_eval["groundedness"])
                judge_scores["brand_voice"].append(j_eval["brand_voice"])
                judge_scores["actionability"].append(j_eval["actionability"])
                judge_scores["safety"].append(j_eval["safety"])
                judge_scores["overall"].append(j_eval["overall_score"])

            # Compute Aggregate Metrics
            intent_acc = accuracy_score(y_true_intent, y_pred_intent)
            intent_f1_macro = f1_score(y_true_intent, y_pred_intent, average="macro", zero_division=0)
            intent_f1_weighted = f1_score(y_true_intent, y_pred_intent, average="weighted", zero_division=0)

            esc_prec = precision_score(y_true_escalate, y_pred_escalate, zero_division=0)
            esc_rec = recall_score(y_true_escalate, y_pred_escalate, zero_division=0)
            esc_f1 = f1_score(y_true_escalate, y_pred_escalate, zero_division=0)

            # High-risk safety miss rate (Cases where ground_truth escalation was needed, but system auto-handled)
            high_risk_indices = [i for i, val in enumerate(y_true_escalate) if val == 1]
            missed_high_risk = sum(1 for i in high_risk_indices if y_pred_escalate[i] == 0)
            high_risk_miss_rate = (missed_high_risk / len(high_risk_indices)) if high_risk_indices else 0.0

            results[sys_name] = {
                "intent_metrics": {
                    "accuracy": round(float(intent_acc), 4),
                    "macro_f1": round(float(intent_f1_macro), 4),
                    "weighted_f1": round(float(intent_f1_weighted), 4)
                },
                "escalation_metrics": {
                    "precision": round(float(esc_prec), 4),
                    "recall": round(float(esc_rec), 4),
                    "f1_score": round(float(esc_f1), 4),
                    "high_risk_miss_rate": round(float(high_risk_miss_rate), 4),
                    "missed_high_risk_count": missed_high_risk,
                    "total_high_risk_count": len(high_risk_indices)
                },
                "text_generation_metrics": {
                    "mean_rouge1": round(float(np.mean(text_metrics["rouge1"])), 4),
                    "mean_rouge2": round(float(np.mean(text_metrics["rouge2"])), 4),
                    "mean_rougeL": round(float(np.mean(text_metrics["rougeL"])), 4),
                    "mean_bleu4": round(float(np.mean(text_metrics["bleu4"])), 4),
                    "mean_semantic_similarity": round(float(np.mean(text_metrics["semantic_similarity"])), 4)
                },
                "judge_metrics": {
                    "groundedness_score": round(float(np.mean(judge_scores["groundedness"])), 2),
                    "brand_voice_score": round(float(np.mean(judge_scores["brand_voice"])), 2),
                    "actionability_score": round(float(np.mean(judge_scores["actionability"])), 2),
                    "safety_score": round(float(np.mean(judge_scores["safety"])), 2),
                    "overall_judge_mean": round(float(np.mean(judge_scores["overall"])), 2)
                },
                "performance_metrics": {
                    "avg_latency_ms": round(float(np.mean(latencies)), 2),
                    "p95_latency_ms": round(float(np.percentage(latencies, 95) if hasattr(np, "percentage") else np.percentile(latencies, 95)), 2),
                    "total_runtime_sec": round(float(time.time() - start_t), 2)
                }
            }

        # Human agreement evaluation
        human_ratings_path = os.path.join(DATA_DIR, "human_judge_ratings.json")
        agreement_stats = self.judge.evaluate_human_agreement(human_ratings_path)

        # Confusion Matrix for Proposed Agent
        cm = confusion_matrix(y_true_intent, y_pred_intent, labels=INTENT_CLASSES).tolist()

        final_report = {
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "golden_set_size": len(self.golden_set),
            "benchmark_results": results,
            "human_judge_agreement": agreement_stats,
            "confusion_matrix": {
                "labels": INTENT_CLASSES,
                "matrix": cm
            }
        }

        # Save to data/evaluation_benchmark_results.json
        out_path = os.path.join(DATA_DIR, "evaluation_benchmark_results.json")
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(final_report, f, indent=2)

        print(f"[OK] Evaluation benchmark completed successfully! Output saved to: {out_path}")
        return final_report

if __name__ == "__main__":
    golden_path = os.path.join(DATA_DIR, "golden_eval_set.json")
    corpus_path = os.path.join(DATA_DIR, "apple_support_corpus.json")
    harness = EvaluationHarness(golden_path, corpus_path)
    report = harness.run_full_benchmark()
    print("\n--- SUMMARY OF HEADLINE RESULTS ---")
    for sys_name, res in report["benchmark_results"].items():
        print(f"\n[{sys_name}]")
        print(f"  Intent Accuracy: {res['intent_metrics']['accuracy'] * 100:.1f}% | Macro F1: {res['intent_metrics']['macro_f1']:.3f}")
        print(f"  Escalation F1:   {res['escalation_metrics']['f1_score']:.3f} | High-Risk Miss Rate: {res['escalation_metrics']['high_risk_miss_rate']*100:.1f}%")
        print(f"  ROUGE-L:         {res['text_generation_metrics']['mean_rougeL']:.3f} | Semantic Sim: {res['text_generation_metrics']['mean_semantic_similarity']:.3f}")
        print(f"  Judge Overall:   {res['judge_metrics']['overall_judge_mean']}/5.0 (Safety: {res['judge_metrics']['safety_score']}/5.0)")
        print(f"  Latency (avg):   {res['performance_metrics']['avg_latency_ms']} ms")
