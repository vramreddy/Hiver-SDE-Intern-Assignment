"""
Unified Customer Support Agent Pipeline for @AppleSupport.

Integrates Intent Classification, Historical Retrieval Grounding,
Escalation Policy, and Draft Reply Generation.

Reply generation:
  - When GEMINI_API_KEY is set: Uses Gemini LLM grounded in retrieved cases
  - When not set: Uses template-based retrieval fallback (clearly labeled)
"""

import os
import time
import re
import logging
from typing import Dict, Any, List, Optional

from .intent_classifier import IntentClassifier
from .retrieval_engine import RetrievalEngine
from .escalation_policy import EscalationPolicy
from .config import BRAND_HANDLE, BRAND_NAME, INTENT_TAXONOMY
from . import llm_client

logger = logging.getLogger(__name__)

# System instruction for Gemini reply generation
_REPLY_SYSTEM_PROMPT = """\
You are @AppleSupport on Twitter. Draft a concise, empathetic customer support reply.

Rules:
- Maximum 280 characters (Twitter limit)
- Never ask for PII (passwords, serial numbers, Apple ID emails) in public tweets
- If escalation is needed, direct the customer to send a DM
- Only use verified Apple URLs: apple.co/*, support.apple.com, locate.apple.com, \
iforgot.apple.com, appleid.apple.com, reportaproblem.apple.com, checkcoverage.apple.com
- Match @AppleSupport's warm, professional, empathetic tone
- Provide specific, actionable steps when possible
- For safety hazards (swollen battery, fire, smoke): immediately warn to stop using the device

Respond with ONLY the reply text. No quotes, no metadata.
"""


class AppleSupportAgent:
    def __init__(self, corpus_path: Optional[str] = None):
        self.intent_classifier = IntentClassifier(corpus_path)
        self.retrieval_engine = RetrievalEngine(corpus_path)
        self.escalation_policy = EscalationPolicy()
        self._llm_available = llm_client.is_available()

        if self._llm_available:
            logger.info("Agent: Gemini LLM available for reply generation.")
        else:
            logger.info("Agent: No GEMINI_API_KEY — using template-based retrieval fallback.")

    def generate_draft_reply(
        self,
        query: str,
        intent: str,
        escalation: Dict[str, Any],
        retrieved_cases: List[Dict[str, Any]],
    ) -> str:
        """
        Drafts a response to a customer query.

        When Gemini is available: generates a grounded LLM reply.
        When not: uses retrieval + template fallback.
        """
        if self._llm_available:
            llm_reply = self._llm_generate_reply(query, intent, escalation, retrieved_cases)
            if llm_reply:
                return llm_reply
            logger.warning("Gemini generation failed — falling back to templates.")

        return self._template_generate_reply(query, intent, escalation, retrieved_cases)

    # ------------------------------------------------------------------ #
    #  Gemini LLM Reply Generation
    # ------------------------------------------------------------------ #

    def _llm_generate_reply(
        self,
        query: str,
        intent: str,
        escalation: Dict[str, Any],
        retrieved_cases: List[Dict[str, Any]],
    ) -> Optional[str]:
        """Generates a reply using Gemini, grounded in retrieved historical cases."""
        # Build context from retrieved cases
        context_lines = []
        for i, case in enumerate(retrieved_cases[:3]):
            context_lines.append(
                f"Historical case {i+1}:\n"
                f"  Customer: {case['customer_text'][:200]}\n"
                f"  Agent reply: {case['agent_text'][:200]}"
            )
        context = "\n".join(context_lines) if context_lines else "No historical cases retrieved."

        esc_desc = escalation.get("reason_description", "Standard troubleshooting")
        esc_decision = escalation.get("decision", "AUTO_HANDLE")

        prompt = (
            f"Customer tweet: {query}\n"
            f"Classified intent: {intent}\n"
            f"Escalation decision: {esc_decision}\n"
            f"Escalation reason: {esc_desc}\n\n"
            f"Historical resolutions for grounding:\n{context}\n\n"
            f"Draft the @AppleSupport reply."
        )

        reply = llm_client.generate(
            prompt,
            system_instruction=_REPLY_SYSTEM_PROMPT,
            temperature=0.4,
            max_output_tokens=256,
        )

        if reply and len(reply) > 10:
            # Strip any accidental quotes
            reply = reply.strip('"').strip("'").strip()
            return reply
        return None

    # ------------------------------------------------------------------ #
    #  Template-Based Retrieval Fallback
    # ------------------------------------------------------------------ #

    def _template_generate_reply(
        self,
        query: str,
        intent: str,
        escalation: Dict[str, Any],
        retrieved_cases: List[Dict[str, Any]],
    ) -> str:
        """
        Template-based reply generation grounded in historical resolutions.
        Used when GEMINI_API_KEY is not set.
        """
        q_lower = query.lower()

        # Dedicated escalation routing replies
        if escalation["decision"] == "ESCALATE_TO_HUMAN":
            reason = escalation["escalation_reason"]
            if reason == "HARDWARE_PHYSICAL_DAMAGE":
                return ("Your safety and device care are our priority. Please disconnect the charger "
                        "and stop using the device. Send us a DM or visit https://locate.apple.com "
                        "to book an immediate service appointment.")
            elif reason == "HIGH_SENTIMENT_CHURN_RISK":
                return ("We sincerely apologize for the frustrating experience. We want to make this right. "
                        "Please send us a DM with your case details and phone number so a senior specialist "
                        "can assist you directly.")
            elif reason == "BILLING_REFUND_AUTH":
                return ("We want to help resolve this billing concern quickly. Please submit a refund request "
                        "at https://reportaproblem.apple.com or send us a DM with your Apple ID email for review.")
            elif reason == "LOW_CONFIDENCE_AMBIGUITY":
                return ("We'd like to look into this with you! Please send us a DM with more details about "
                        "your device model and what troubleshooting steps you've tried so far: https://apple.co/dm.")
            else:
                return ("We'd like to look into this securely with you. Please send us a DM with your details "
                        "or visit https://iforgot.apple.com for account security: https://apple.co/dm.")

        # If we have a high quality retrieved case for non-escalated queries, adapt from historical resolution
        if retrieved_cases and retrieved_cases[0]["similarity_score"] > 0.45:
            best_match = retrieved_cases[0]
            return best_match["agent_text"]

        # Fallback calibrated template per intent
        if "airpod" in q_lower or "earbud" in q_lower:
            return ("We'd like to help with your AirPods. Place both AirPods in the case, open the lid, "
                    "and hold the setup button for 15 seconds to reset them: https://apple.co/reset-airpods.")

        if "password" in q_lower or "passcode" in q_lower or "apple id" in q_lower or "locked" in q_lower:
            return ("Account security is our top priority. We cannot process credentials over Twitter. "
                    "You can securely reset your password or manage your account at https://iforgot.apple.com "
                    "or DM us for guidance.")

        templates = {
            "OS_UPDATE_BUG": ("We'd like to help get this resolved. Have you tried a force restart on your "
                              "device? Follow these steps: https://apple.co/force-restart. Let us know if "
                              "the issue persists!"),
            "HARDWARE_BATTERY": ("We can help you review your hardware and battery health. You can check "
                                 "warranty status and find authorized service providers near you at "
                                 "https://locate.apple.com."),
            "ACCOUNT_ICLOUD_SECURITY": ("Account security is our priority. You can securely reset passwords "
                                        "at https://iforgot.apple.com and manage security at https://appleid.apple.com."),
            "BILLING_SUBSCRIPTIONS": ("You can view your active subscriptions, recent purchases, and submit "
                                      "refund requests directly at https://reportaproblem.apple.com."),
            "CONNECTIVITY_SETUP": ("Let's troubleshoot your connection. For AirPods, reset them in the case "
                                   "(https://apple.co/reset-airpods). For cellular or Wi-Fi, toggle Airplane Mode "
                                   "and check Settings > General > About for carrier updates."),
            "REPAIR_WARRANTY_STATUS": ("You can easily check your warranty coverage, AppleCare+ status, "
                                       "and schedule Genius Bar appointments at "
                                       "https://checkcoverage.apple.com."),
        }
        return templates.get(
            intent,
            ("Thank you for reaching out to Apple Support! You can submit direct product feedback "
             "to our engineering teams anytime at https://apple.com/feedback.")
        )

    # ------------------------------------------------------------------ #
    #  Full Pipeline Execution
    # ------------------------------------------------------------------ #

    def process_query(self, query: str) -> Dict[str, Any]:
        """Executes the full AI support agent pipeline for an incoming customer tweet."""
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
            top_retrieved=retrieved,
        )

        # 4. Draft Reply Generation (LLM or template fallback)
        draft_reply = self.generate_draft_reply(
            query=query,
            intent=predicted_intent,
            escalation=escalation_res,
            retrieved_cases=retrieved,
        )

        elapsed_ms = round((time.time() - start_time) * 1000, 2)

        return {
            "customer_query": query,
            "intent": {
                "predicted_intent": predicted_intent,
                "intent_name": intent_res["intent_name"],
                "confidence": confidence,
                "probabilities": intent_res["probabilities"],
                "ranked_intents": intent_res["ranked_intents"],
            },
            "retrieval": {
                "num_retrieved": len(retrieved),
                "top_cases": retrieved,
            },
            "escalation": {
                "decision": escalation_res["decision"],
                "reason_code": escalation_res["escalation_reason"],
                "reason_description": escalation_res["reason_description"],
                "suggested_action": escalation_res["suggested_action"],
                "risk_level": escalation_res["risk_level"],
            },
            "draft_reply": draft_reply,
            "meta": {
                "brand": BRAND_HANDLE,
                "processing_time_ms": elapsed_ms,
                "pipeline_version": "2.0.0",
                "reply_mode": "gemini_llm" if self._llm_available else "template_fallback",
            },
        }
