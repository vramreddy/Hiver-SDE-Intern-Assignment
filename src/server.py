"""
FastAPI Server for @AppleSupport AI Agent & Interactive Evaluation Dashboard.
"""

import os
import json
from typing import Dict, Any, List, Optional
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse, JSONResponse
from pydantic import BaseModel

from .config import INTENT_TAXONOMY, INTENT_CLASSES, ESCALATION_REASONS, BRAND_HANDLE
from .agent import AppleSupportAgent
from .eval_harness import Baseline0_Trivial, Baseline1_Simple, EvaluationHarness
from .judge import SupportJudge

app = FastAPI(
    title="Apple Support AI Agent & Evaluation Harness",
    description="Enterprise-grade AI customer support agent for @AppleSupport with real-time evaluation",
    version="1.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

DATA_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "data"))
WEB_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "web"))

# Instantiate models
agent = AppleSupportAgent()
baseline0 = Baseline0_Trivial()
baseline1 = Baseline1_Simple()
judge = SupportJudge()

class QueryRequest(BaseModel):
    query: str

class CompareRequest(BaseModel):
    query: str

@app.get("/api/health")
def health_check():
    return {"status": "healthy", "brand": BRAND_HANDLE, "version": "1.0.0"}

@app.get("/api/taxonomy")
def get_taxonomy():
    return {
        "intents": INTENT_TAXONOMY,
        "classes": INTENT_CLASSES,
        "escalation_reasons": ESCALATION_REASONS
    }

@app.post("/api/agent/process")
def process_customer_query(req: QueryRequest):
    if not req.query.strip():
        raise HTTPException(status_code=400, detail="Query cannot be empty.")
    result = agent.process_query(req.query)
    
    # Run automated judge evaluation on the agent's response
    judge_eval = judge.evaluate_reply(
        customer_query=req.query,
        model_reply=result["draft_reply"],
        intent=result["intent"]["predicted_intent"],
        escalation_decision=result["escalation"]["decision"]
    )
    result["judge_evaluation"] = judge_eval
    return result

@app.post("/api/baselines/compare")
def compare_systems(req: CompareRequest):
    if not req.query.strip():
        raise HTTPException(status_code=400, detail="Query cannot be empty.")
    
    q = req.query
    
    # 1. Baseline 0
    b0_out = baseline0.process(q)
    b0_judge = judge.evaluate_reply(q, b0_out["draft_reply"], b0_out["predicted_intent"], b0_out["escalation_decision"])

    # 2. Baseline 1
    b1_out = baseline1.process(q)
    b1_judge = judge.evaluate_reply(q, b1_out["draft_reply"], b1_out["predicted_intent"], b1_out["escalation_decision"])

    # 3. Proposed Agent
    agent_out = agent.process_query(q)
    agent_judge = judge.evaluate_reply(q, agent_out["draft_reply"], agent_out["intent"]["predicted_intent"], agent_out["escalation"]["decision"])

    return {
        "query": q,
        "baseline0": {
            "name": "Baseline 0 (Trivial Canned)",
            "intent": b0_out["predicted_intent"],
            "escalation": b0_out["escalation_decision"],
            "reply": b0_out["draft_reply"],
            "judge_score": b0_judge
        },
        "baseline1": {
            "name": "Baseline 1 (Simple Zero-Shot)",
            "intent": b1_out["predicted_intent"],
            "escalation": b1_out["escalation_decision"],
            "reply": b1_out["draft_reply"],
            "judge_score": b1_judge
        },
        "proposed_agent": {
            "name": "Proposed Agent (Hybrid RAG)",
            "intent": agent_out["intent"]["predicted_intent"],
            "intent_confidence": agent_out["intent"]["confidence"],
            "escalation": agent_out["escalation"]["decision"],
            "escalation_reason": agent_out["escalation"]["reason_description"],
            "reply": agent_out["draft_reply"],
            "retrieved_cases": agent_out["retrieval"]["top_cases"],
            "judge_score": agent_judge,
            "latency_ms": agent_out["meta"]["processing_time_ms"]
        }
    }

@app.get("/api/eval/benchmark")
def get_benchmark_results():
    results_path = os.path.join(DATA_DIR, "evaluation_benchmark_results.json")
    if os.path.exists(results_path):
        with open(results_path, "r", encoding="utf-8") as f:
            return json.load(f)
    raise HTTPException(status_code=404, detail="Benchmark results not found. Run eval harness first.")

@app.get("/api/eval/goldenset")
def get_golden_evaluation_set(limit: int = 200, intent: Optional[str] = None, difficulty: Optional[str] = None):
    golden_path = os.path.join(DATA_DIR, "golden_eval_set.json")
    if not os.path.exists(golden_path):
        raise HTTPException(status_code=404, detail="Golden set not found.")
    
    with open(golden_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    if intent:
        data = [x for x in data if x.get("ground_truth_intent") == intent]
    if difficulty:
        data = [x for x in data if x.get("difficulty") == difficulty]

    return {
        "total_count": len(data),
        "items": data[:limit]
    }

@app.post("/api/eval/run")
def trigger_benchmark_run():
    golden_path = os.path.join(DATA_DIR, "golden_eval_set.json")
    corpus_path = os.path.join(DATA_DIR, "apple_support_corpus.json")
    harness = EvaluationHarness(golden_path, corpus_path)
    report = harness.run_full_benchmark()
    return report

# Serve Frontend static files if directory exists
if os.path.exists(WEB_DIR):
    app.mount("/static", StaticFiles(directory=WEB_DIR), name="static")

    @app.get("/")
    def serve_frontend_root():
        index_file = os.path.join(WEB_DIR, "index.html")
        if os.path.exists(index_file):
            return FileResponse(index_file)
        return {"message": "Welcome to Apple Support AI Agent API"}
