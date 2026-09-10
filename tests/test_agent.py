"""
End-to-end tests for Unified Apple Support Agent.
"""

import pytest
from src.agent import AppleSupportAgent

@pytest.fixture
def agent():
    return AppleSupportAgent()

def test_full_agent_query_processing(agent):
    query = "@AppleSupport My iPhone 15 Pro gets hot while charging and battery health dropped."
    res = agent.process_query(query)
    
    assert "intent" in res
    assert "retrieval" in res
    assert "escalation" in res
    assert "draft_reply" in res
    assert len(res["draft_reply"]) > 10
    assert res["meta"]["processing_time_ms"] >= 0
    assert res["intent"]["predicted_intent"] in ["HARDWARE_BATTERY", "OS_UPDATE_BUG"]

def test_airpods_troubleshooting_reply(agent):
    query = "@AppleSupport My AirPods Pro right earbud has zero sound."
    res = agent.process_query(query)
    assert res["intent"]["predicted_intent"] == "CONNECTIVITY_SETUP"
    assert "airpods" in res["draft_reply"].lower() or "reset" in res["draft_reply"].lower()
