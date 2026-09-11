"""
Deterministic & Risk-Aware Escalation Policy Engine.
Evaluates customer messages against safety constraints, PII requirements,
financial authorization thresholds, hardware damage hazards, and sentiment churn risk.
"""

import re
from typing import Dict, Any, Tuple
from .config import ESCALATION_REASONS, AUTO_HANDLE_REASONS

class EscalationPolicy:
    def __init__(self, confidence_threshold: float = 0.30):
        self.confidence_threshold = confidence_threshold
        
        # High Risk / Escalation Triggers
        self.pii_patterns = [
            re.compile(r'\b(dm\s*me|send\s*dm|serial\s*number|imei|apple\s*id\s*email|locked\s*out|recovery\s*phone|hacked|stolen)\b', re.I),
            re.compile(r'\b(repair\s*#|repair\s*id|fedex|shipping\s*address|tracking\s*number)\b', re.I)
        ]
        
        self.hardware_damage_patterns = [
            re.compile(r'\b(swells?|swelling|swollen|smoke|smoking|spark|burning|burn\s*mark|exploded|fire)\b', re.I),
            re.compile(r'\b(submerged|bathtub|dropped\s*in\s*water|liquid\s*damage|shattered\s*back|hinge\s*lines)\b', re.I),
            re.compile(r'\b(wi-?fi|wifi|bluetooth).{0,60}(grey|gray)e?d?\s*out\b', re.I),
            re.compile(r'\b(grey|gray)e?d?\s*out\b', re.I),
        ]

        self.billing_auth_patterns = [
            re.compile(r'\b(unauthorized\s*charge|charged\s*4\s*times|double\s*charge|dispute|fraud|bank\s*blocked)\b', re.I),
            re.compile(r'\b(charged\s*twice|refund\s*me\s*now|unauthorized\s*apple\s*services)\b', re.I)
        ]

        self.sentiment_churn_patterns = [
            re.compile(r'\b(lawyer|attorney|sue|court|legal\s*action|bbb|ftc)\b', re.I),
            re.compile(r'\b(worst\s*(service|experience)|unacceptable|disgusting|supervisor|manager|tim\s*cook|switching\s*to\s*android|switching\s*to\s*samsung)\b', re.I)
        ]

    def evaluate(self, text: str, intent: str, confidence: float, top_retrieved: list = None) -> Dict[str, Any]:
        """
        Evaluates whether a message should be Auto-Handled or Escalated to a Human Advisor.
        Returns decision, stated reason, risk score, and suggested action.
        """
        # 1. Physical Hardware & Safety Hazard (Critical Priority)
        for pat in self.hardware_damage_patterns:
            if pat.search(text):
                return {
                    "decision": "ESCALATE_TO_HUMAN",
                    "escalation_reason": "HARDWARE_PHYSICAL_DAMAGE",
                    "reason_description": ESCALATION_REASONS["HARDWARE_PHYSICAL_DAMAGE"],
                    "suggested_action": "Direct customer to halt usage immediately and arrange emergency Genius Bar inspection / priority mail-in repair.",
                    "risk_level": "CRITICAL"
                }

        # 2. Severe Frustration / Legal / Churn Risk
        for pat in self.sentiment_churn_patterns:
            if pat.search(text):
                return {
                    "decision": "ESCALATE_TO_HUMAN",
                    "escalation_reason": "HIGH_SENTIMENT_CHURN_RISK",
                    "reason_description": ESCALATION_REASONS["HIGH_SENTIMENT_CHURN_RISK"],
                    "suggested_action": "Route to Senior Customer Relations Specialist with empathy-focused outreach via secure DM.",
                    "risk_level": "HIGH"
                }

        # 3. High Billing / Unauthorized Charges
        for pat in self.billing_auth_patterns:
            if pat.search(text):
                return {
                    "decision": "ESCALATE_TO_HUMAN",
                    "escalation_reason": "BILLING_REFUND_AUTH",
                    "reason_description": ESCALATION_REASONS["BILLING_REFUND_AUTH"],
                    "suggested_action": "Request Apple ID verification via DM to inspect duplicate transactions and initiate financial reversal.",
                    "risk_level": "MEDIUM"
                }

        # 4. PII / Security Verification Required
        for pat in self.pii_patterns:
            if pat.search(text):
                return {
                    "decision": "ESCALATE_TO_HUMAN",
                    "escalation_reason": "PII_SECURITY_DM",
                    "reason_description": ESCALATION_REASONS["PII_SECURITY_DM"],
                    "suggested_action": "Invite customer to secure DM to gather Apple ID / hardware serial number / diagnostic logs safely.",
                    "risk_level": "MEDIUM"
                }

        # 5. Low Model Confidence / Ambiguity
        if confidence < self.confidence_threshold:
            return {
                "decision": "ESCALATE_TO_HUMAN",
                "escalation_reason": "LOW_CONFIDENCE_AMBIGUITY",
                "reason_description": ESCALATION_REASONS["LOW_CONFIDENCE_AMBIGUITY"],
                "suggested_action": "Route query to human triage queue due to low model confidence and semantic ambiguity.",
                "risk_level": "LOW"
            }

        # 6. Intent-specific policy defaults
        # If retrieved historical context strongly suggests auto-handling
        if top_retrieved and len(top_retrieved) > 0:
            top_match = top_retrieved[0]
            if not top_match.get("auto_handle", True):
                return {
                    "decision": "ESCALATE_TO_HUMAN",
                    "escalation_reason": top_match.get("escalation_reason", "PII_SECURITY_DM"),
                    "reason_description": ESCALATION_REASONS.get(top_match.get("escalation_reason"), "Requires specialist human intervention."),
                    "suggested_action": "Escalate to tier-2 human support based on historical resolution pattern.",
                    "risk_level": "MEDIUM"
                }

        # Default Auto-Handle
        return {
            "decision": "AUTO_HANDLE",
            "escalation_reason": "STANDARD_TROUBLESHOOTING",
            "reason_description": AUTO_HANDLE_REASONS["STANDARD_TROUBLESHOOTING"],
            "suggested_action": "Deploy grounded step-by-step resolution draft with relevant official Apple support documentation links.",
            "risk_level": "MINIMAL"
        }
