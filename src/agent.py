"""
Unified Customer Support Agent Pipeline for @AppleSupport.
Integrates Intent Classification, RAG Historical Grounding, Escalation Policy,
and Tone-Calibrated Draft Generation.
"""

import os
import time
import re
from typing import Dict, Any, List, Optional
from .intent_classifier import IntentClassifier
from .retrieval_engine import RetrievalEngine
from .escalation_policy import EscalationPolicy
from .config import BRAND_HANDLE, BRAND_NAME, INTENT_TAXONOMY

class AppleSupportAgent:
    def __init__(self, corpus_path: Optional[str] = None):
        self.intent_classifier = IntentClassifier(corpus_path)
        self.retrieval_engine = RetrievalEngine(corpus_path)
        self.escalation_policy = EscalationPolicy()

    def generate_draft_reply(self, query: str, intent: str, escalation: Dict[str, Any], retrieved_cases: List[Dict[str, Any]]) -> str:
        """
        Drafts a response grounded in historical resolutions and calibrated to Apple Support brand voice.
        """
        # If we have a very high quality top retrieved case (similarity > 0.65), adapt from historical resolution
        if retrieved_cases and retrieved_cases[0]["similarity_score"] > 0.40:
            best_match = retrieved_cases[0]
            base_reply = best_match["agent_text"]
            
            # If escalated to human for PII or high sentiment, ensure standard DM greeting is present
            if escalation["decision"] == "ESCALATE_TO_HUMAN":
                if "dm" not in base_reply.lower() and "message" not in base_reply.lower():
                    base_reply += " Please send us a DM so an advisor can securely review your account details."
            return base_reply

        # Fallback calibrated template per intent and escalation decision
        if escalation["decision"] == "ESCALATE_TO_HUMAN":
            reason = escalation["escalation_reason"]
            if reason == "HARDWARE_PHYSICAL_DAMAGE":
                return "Your safety and device care are our priority. Please disconnect the charger and stop using the device. Send us a DM or visit https://locate.apple.com to book an immediate service appointment."
            elif reason == "HIGH_SENTIMENT_CHURN_RISK":
                return "We sincerely apologize for the frustrating experience. We want to make this right. Please send us a DM with your case details and phone number so a senior specialist can assist you directly."
            elif reason == "BILLING_REFUND_AUTH":
                return "We want to help resolve this billing concern quickly. Please submit a refund request at https://reportaproblem.apple.com or send us a DM with your Apple ID email for review."
            else:
                return "We'd like to look into this with you. Please send us a DM with your device model and iOS version: https://apple.co/dm."
        else:
            # Auto-handle standard troubleshooting templates
            if intent == "OS_UPDATE_BUG":
                return "We'd like to help get this resolved. Have you tried a force restart on your device? Follow these steps: https://apple.co/force-restart. Let us know if the issue persists!"
            elif intent == "HARDWARE_BATTERY":
                return "We can help you review your hardware and battery health. You can check warranty status and find authorized service providers near you at https://locate.apple.com."
            elif intent == "ACCOUNT_ICLOUD_SECURITY":
                return "Account security is our priority. You can manage your Apple ID and verify security settings securely at https://appleid.apple.com."
            elif intent == "BILLING_SUBSCRIPTIONS":
                return "You can view your active subscriptions, recent purchases, and submit refund requests directly at https://reportaproblem.apple.com."
            elif intent == "CONNECTIVITY_SETUP":
                return "Let's troubleshoot your connection. Try toggling Airplane Mode, restarting your device, and checking for carrier settings updates in Settings > General > About."
            elif intent == "REPAIR_WARRANTY_STATUS":
                return "You can easily check your warranty coverage, AppleCare+ status, and schedule Genius Bar appointments at https://checkcoverage.apple.com."
            else:
                return "Thank you for reaching out to Apple Support! You can submit direct product feedback to our engineering teams anytime at https://apple.com/feedback."

    def process_query(self, query: str) -> Dict[str, Any]:
        """
        Executes the full AI support agent pipeline for an incoming customer tweet.
        """
        start_time = time.time()
        
        # 1. Intent Classification
        intent_res = self.intent_classifier.classify(query)
        predicted_intent = intent_res["predicted_intent"]
        confidence = intent_res["confidence"]

        # 2. Historical Retrieval Grounding
        retrieved = self.retrieval_engine.retrieve(query, top_k=3, filter_intent=predicted_intent)

        # 3. Escalation Decision & Stated Reason
        escalation_res = self.escalation_policy.evaluate(
            text=query,
            intent=predicted_intent,
            confidence=confidence,
            top_retrieved=retrieved
        )

        # 4. Tone-Calibrated Draft Reply Generation
        draft_reply = self.generate_draft_reply(
            query=query,
            intent=predicted_intent,
            escalation=escalation_res,
            retrieved_cases=retrieved
        )

        elapsed_ms = round((time.time() - start_time) * 1000, 2)

        return {
            "customer_query": query,
            "intent": {
                "predicted_intent": predicted_intent,
                "intent_name": intent_res["intent_name"],
                "confidence": confidence,
                "probabilities": intent_res["probabilities"],
                "ranked_intents": intent_res["ranked_intents"]
            },
            "retrieval": {
                "num_retrieved": len(retrieved),
                "top_cases": retrieved
            },
            "escalation": {
                "decision": escalation_res["decision"],
                "reason_code": escalation_res["escalation_reason"],
                "reason_description": escalation_res["reason_description"],
                "suggested_action": escalation_res["suggested_action"],
                "risk_level": escalation_res["risk_level"]
            },
            "draft_reply": draft_reply,
            "meta": {
                "brand": BRAND_HANDLE,
                "processing_time_ms": elapsed_ms,
                "pipeline_version": "1.0.0-prod"
            }
        }
