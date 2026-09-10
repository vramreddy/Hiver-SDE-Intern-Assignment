"""
Unit tests for Escalation Policy Engine.
"""

import pytest
from src.escalation_policy import EscalationPolicy

@pytest.fixture
def policy():
    return EscalationPolicy()

def test_hardware_swelling_critical_escalation(policy):
    res = policy.evaluate(
        text="@AppleSupport my macbook battery is physically swelling up and smoking!",
        intent="HARDWARE_BATTERY",
        confidence=0.95
    )
    assert res["decision"] == "ESCALATE_TO_HUMAN"
    assert res["escalation_reason"] == "HARDWARE_PHYSICAL_DAMAGE"
    assert res["risk_level"] == "CRITICAL"

def test_legal_churn_escalation(policy):
    res = policy.evaluate(
        text="@AppleSupport Worst customer service ever. Refund me immediately or I am calling my lawyer to sue Apple!",
        intent="BILLING_SUBSCRIPTIONS",
        confidence=0.92
    )
    assert res["decision"] == "ESCALATE_TO_HUMAN"
    assert res["escalation_reason"] in ["HIGH_SENTIMENT_CHURN_RISK", "BILLING_REFUND_AUTH"]

def test_standard_troubleshooting_autohandle(policy):
    res = policy.evaluate(
        text="@AppleSupport how do I check my apple care coverage online for my mac?",
        intent="REPAIR_WARRANTY_STATUS",
        confidence=0.90
    )
    assert res["decision"] == "AUTO_HANDLE"
