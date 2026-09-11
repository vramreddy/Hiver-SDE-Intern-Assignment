"""
Tests for the data pipeline: cleaning, leakage verification, golden set properties.
"""

import pytest
from src.data_pipeline import (
    clean_tweet_text,
    verify_no_leakage,
    label_intent_heuristic,
    label_escalation_heuristic,
    ADVERSARIAL_GOLDEN_CASES,
)
from src.config import INTENT_CLASSES


class TestCleanTweetText:
    def test_url_replacement(self):
        text = "Check this out https://t.co/abc123 for more info"
        cleaned = clean_tweet_text(text)
        assert "https://" not in cleaned
        assert "[URL]" in cleaned

    def test_whitespace_normalization(self):
        text = "Hello   world  from   Apple"
        cleaned = clean_tweet_text(text)
        assert "  " not in cleaned

    def test_empty_string(self):
        assert clean_tweet_text("") == ""

    def test_none_input(self):
        assert clean_tweet_text(None) == ""

    def test_anonymised_handle_normalization(self):
        text = "@115712 my iPhone is broken"
        cleaned = clean_tweet_text(text)
        assert "@115712" not in cleaned
        assert "@user" in cleaned

    def test_preserves_emoji(self):
        text = "Love the new update 🔥🔥🔥"
        cleaned = clean_tweet_text(text)
        assert "🔥" in cleaned


class TestVerifyNoLeakage:
    def test_clean_data_passes(self):
        corpus = [
            {"customer_text": "My iPhone battery is draining fast"},
            {"customer_text": "Safari keeps crashing on macOS"},
        ]
        golden = [
            {"customer_tweet": "AirPods won't connect to my Mac"},
            {"customer_tweet": "How do I check my warranty?"},
        ]
        result = verify_no_leakage(corpus, golden)
        assert result["status"] == "CLEAN"
        assert result["exact_duplicates"] == 0
        assert result["leakage_percentage"] == 0.0

    def test_leakage_detected(self):
        corpus = [
            {"customer_text": "My iPhone battery is draining fast"},
            {"customer_text": "Safari keeps crashing on macOS"},
        ]
        golden = [
            {"customer_tweet": "My iPhone battery is draining fast"},  # exact duplicate
            {"customer_tweet": "AirPods won't connect"},
        ]
        result = verify_no_leakage(corpus, golden)
        assert result["status"] == "LEAKAGE_DETECTED"
        assert result["exact_duplicates"] >= 1
        assert result["leakage_percentage"] > 0

    def test_case_insensitive_leakage(self):
        corpus = [{"customer_text": "My iPhone Battery Is Draining Fast"}]
        golden = [{"customer_tweet": "my iphone battery is draining fast"}]
        result = verify_no_leakage(corpus, golden)
        assert result["status"] == "LEAKAGE_DETECTED"


class TestLabelIntentHeuristic:
    def test_os_update_bug(self):
        intent, score = label_intent_heuristic("My iPhone crashes after iOS 17.2 update")
        assert intent == "OS_UPDATE_BUG"
        assert score > 0

    def test_hardware_battery(self):
        intent, score = label_intent_heuristic("Battery health dropped and phone is overheating")
        assert intent == "HARDWARE_BATTERY"
        assert score > 0

    def test_account_security(self):
        intent, score = label_intent_heuristic("My Apple ID is locked and I need password reset")
        assert intent == "ACCOUNT_ICLOUD_SECURITY"
        assert score > 0

    def test_billing(self):
        intent, score = label_intent_heuristic("I was charged twice for my subscription refund needed")
        assert intent == "BILLING_SUBSCRIPTIONS"
        assert score > 0

    def test_unknown_fallback(self):
        intent, score = label_intent_heuristic("Hello there")
        assert intent == "GENERAL_FEEDBACK_CHURN"
        assert score == 0


class TestLabelEscalationHeuristic:
    def test_swelling_battery_escalates(self):
        escalate, reason = label_escalation_heuristic(
            "My battery is swelling up!", "HARDWARE_BATTERY"
        )
        assert escalate is True
        assert reason == "HARDWARE_PHYSICAL_DAMAGE"

    def test_legal_threat_escalates(self):
        escalate, reason = label_escalation_heuristic(
            "I am calling my lawyer to sue you!", "BILLING_SUBSCRIPTIONS"
        )
        assert escalate is True
        assert reason == "HIGH_SENTIMENT_CHURN_RISK"

    def test_normal_query_does_not_escalate(self):
        escalate, reason = label_escalation_heuristic(
            "How do I update my iPhone?", "OS_UPDATE_BUG"
        )
        assert escalate is False


class TestAdversarialGoldenCases:
    def test_adversarial_cases_cover_all_failure_modes(self):
        """Verify the adversarial cases cover key failure patterns."""
        notes = [c["notes"].lower() for c in ADVERSARIAL_GOLDEN_CASES]
        all_notes = " ".join(notes)
        assert "multi-intent" in all_notes or "adversarial" in all_notes
        assert "sarcas" in all_notes
        assert "phishing" in all_notes or "pii" in all_notes
        assert "truncation" in all_notes or "context" in all_notes

    def test_adversarial_cases_have_hard_difficulty(self):
        hard_count = sum(1 for c in ADVERSARIAL_GOLDEN_CASES if c["difficulty"] == "HARD")
        assert hard_count >= 5  # Most adversarial cases should be hard
