"""
Regression tests for the 5 documented failure modes from REPORT.md.

1. Adversarial multi-intent (software + hardware hazard)
2. Sarcasm / inverted polarity
3. PII phishing trap (public password reset request)
4. Hardware vs. software ambiguity (greyed-out Wi-Fi)
5. Context truncation (isolated follow-up tweet)
"""

import pytest
from src.agent import AppleSupportAgent
from src.escalation_policy import EscalationPolicy
from src.intent_classifier import IntentClassifier


@pytest.fixture
def agent():
    return AppleSupportAgent()


@pytest.fixture
def policy():
    return EscalationPolicy()


@pytest.fixture
def classifier():
    return IntentClassifier()


# ---------- Failure Mode 1: Adversarial Multi-Intent ---------- #

class TestAdversarialMultiIntent:
    """
    Failure Mode: A tweet mentions both a software update AND a physical
    battery hazard. The system must prioritise the safety-critical hardware
    escalation over software troubleshooting.
    """

    def test_swelling_battery_with_software_mention_escalates(self, policy):
        """Battery swelling must trigger HARDWARE_PHYSICAL_DAMAGE even when
        software update is mentioned."""
        res = policy.evaluate(
            text="@AppleSupport WatchOS 10 ruined my battery life and "
                 "my Apple Watch battery swells up on the charger!!",
            intent="OS_UPDATE_BUG",
            confidence=0.85,
        )
        assert res["decision"] == "ESCALATE_TO_HUMAN"
        assert res["escalation_reason"] == "HARDWARE_PHYSICAL_DAMAGE"

    def test_smoke_with_update_mention_escalates(self, policy):
        """Smoke report must override any OS update context."""
        res = policy.evaluate(
            text="@AppleSupport after the iOS 17 update my phone started smoking "
                 "while charging, what should I do?",
            intent="OS_UPDATE_BUG",
            confidence=0.90,
        )
        assert res["decision"] == "ESCALATE_TO_HUMAN"
        assert res["escalation_reason"] == "HARDWARE_PHYSICAL_DAMAGE"
        assert res["risk_level"] == "CRITICAL"


# ---------- Failure Mode 2: Sarcasm / Inverted Polarity ---------- #

class TestSarcasmInvertedPolarity:
    """
    Failure Mode: Sarcastic phrasing like 'Brilliant new feature' should be
    classified as a software bug, not positive feedback.
    """

    def test_sarcastic_alarm_bug(self, classifier):
        """Sarcastic alarm complaint must map to OS_UPDATE_BUG, not GENERAL_FEEDBACK."""
        res = classifier.classify(
            "@AppleSupport Brilliant new feature in iOS 17 where alarms "
            "just decide to stay completely silent and make me late for work! 😡"
        )
        assert res["predicted_intent"] == "OS_UPDATE_BUG"

    def test_sarcastic_crash_complaint(self, classifier):
        """Another sarcastic pattern — 'love how Safari crashes' is still a bug."""
        res = classifier.classify(
            "@AppleSupport Love how Safari crashes every single time I open "
            "a new tab on my MacBook after the macOS Sonoma update 🙄"
        )
        assert res["predicted_intent"] == "OS_UPDATE_BUG"


# ---------- Failure Mode 3: PII Phishing Trap ---------- #

class TestPIIPhishingTrap:
    """
    Failure Mode: Customer asks agent to reset their password over Twitter.
    The agent must NEVER attempt a public password reset — must redirect
    to iforgot.apple.com or DM.
    """

    def test_password_reset_over_twitter_rejected(self, agent):
        """Agent must not include any password-handling language in a public reply."""
        res = agent.process_query(
            "@AppleSupport I forgot my Apple ID passcode and my device is disabled. "
            "Can you reset it for me over Twitter?"
        )
        reply_lower = res["draft_reply"].lower()
        # Must NOT attempt to handle the password publicly
        assert "your password is" not in reply_lower
        assert "enter your password" not in reply_lower
        # Should redirect to secure channel
        assert "iforgot" in reply_lower or "dm" in reply_lower

    def test_serial_number_not_requested_publicly(self, agent):
        """Agent must not ask for serial number in a public tweet."""
        res = agent.process_query(
            "@AppleSupport My iPhone keeps shutting down randomly. What's wrong?"
        )
        reply_lower = res["draft_reply"].lower()
        assert "serial number" not in reply_lower or "dm" in reply_lower


# ---------- Failure Mode 4: Hardware vs Software Ambiguity ---------- #

class TestHardwareSoftwareAmbiguity:
    """
    Failure Mode: Greyed-out Wi-Fi button indicates a hardware IC failure,
    not a software network glitch. Must route to hardware diagnostic, not
    generic 'Reset Network Settings' advice.
    """

    def test_greyed_wifi_triggers_escalation(self, policy):
        """Greyed Wi-Fi should escalate for hardware diagnostic."""
        res = policy.evaluate(
            text="@AppleSupport Wi-Fi button on my iPhone is greyed out "
                 "and Bluetooth toggle spins forever.",
            intent="CONNECTIVITY_SETUP",
            confidence=0.80,
        )
        # Should escalate due to PII/diagnostic DM pattern
        assert res["decision"] == "ESCALATE_TO_HUMAN"

    def test_greyed_wifi_agent_mentions_hardware_or_dm(self, agent):
        """Agent reply should mention DM or hardware diagnostic, not just
        'reset network settings'."""
        res = agent.process_query(
            "@AppleSupport Wi-Fi button on my iPhone is greyed out "
            "and Bluetooth toggle spins forever."
        )
        reply_lower = res["draft_reply"].lower()
        assert "dm" in reply_lower or "hardware" in reply_lower or "diagnostic" in reply_lower


# ---------- Failure Mode 5: Context Truncation ---------- #

class TestContextTruncation:
    """
    Failure Mode: An isolated follow-up tweet like 'Done that already.
    Still not working.' has no prior context. The system should recognise
    low confidence and escalate to a human.
    """

    def test_isolated_followup_low_confidence(self, classifier):
        """Ambiguous single-turn followup should yield low confidence."""
        res = classifier.classify("@AppleSupport Done that already. Still not working.")
        # The classifier should either have low confidence or classify as general
        # (either is acceptable — the key is that the escalation policy catches it)
        assert res["confidence"] < 0.70 or res["predicted_intent"] == "GENERAL_FEEDBACK_CHURN"

    def test_vague_query_escalates(self, agent):
        """Vague, context-free queries should trigger escalation or DM redirect."""
        res = agent.process_query("@AppleSupport Tried that. Didn't work. Help?")
        reply_lower = res["draft_reply"].lower()
        # Should ask for more details or redirect to DM
        assert "dm" in reply_lower or "details" in reply_lower or "more" in reply_lower
