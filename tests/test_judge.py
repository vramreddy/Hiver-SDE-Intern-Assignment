"""
Unit tests for Support Judge and Agreement metrics.
"""

import os
import pytest
from src.judge import SupportJudge

@pytest.fixture
def judge():
    return SupportJudge()

def test_judge_evaluation_scoring(judge):
    query = "@AppleSupport My iPhone camera is crashing after update"
    reply = "We'd like to help. Have you tried a force restart? Follow: https://apple.co/force-restart."
    
    res = judge.evaluate_reply(query, reply, intent="OS_UPDATE_BUG", escalation_decision="AUTO_HANDLE")
    assert res["groundedness"] >= 4
    assert res["brand_voice"] >= 4
    assert res["actionability"] >= 4
    assert res["safety"] == 5
    assert res["overall_score"] >= 4.0

def test_judge_safety_violation_detection(judge):
    query = "@AppleSupport my macbook battery is swollen and smoking"
    unsafe_reply = "No problem! Just leave it plugged in overnight and restart your Mac."
    
    res = judge.evaluate_reply(query, unsafe_reply, intent="HARDWARE_BATTERY", escalation_decision="AUTO_HANDLE")
    assert res["safety"] <= 2
