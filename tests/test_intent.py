"""
Unit tests for Intent Classifier.
"""

import pytest
from src.intent_classifier import IntentClassifier
from src.config import INTENT_CLASSES

@pytest.fixture
def classifier():
    return IntentClassifier()

def test_intent_classes_loaded(classifier):
    assert len(classifier.classes) == 7
    assert "OS_UPDATE_BUG" in classifier.classes
    assert "HARDWARE_BATTERY" in classifier.classes

def test_os_update_intent(classifier):
    res = classifier.classify("@AppleSupport iPhone 14 Pro freezing on lock screen after iOS 17.2 update")
    assert res["predicted_intent"] == "OS_UPDATE_BUG"
    assert res["confidence"] > 0.40

def test_hardware_battery_intent(classifier):
    res = classifier.classify("@AppleSupport My battery health dropped to 72% and maximum capacity says Service")
    assert res["predicted_intent"] == "HARDWARE_BATTERY"
    assert res["confidence"] > 0.40

def test_billing_intent(classifier):
    res = classifier.classify("@AppleSupport I was charged twice for my Apple Music subscription and need a refund")
    assert res["predicted_intent"] == "BILLING_SUBSCRIPTIONS"
    assert res["confidence"] > 0.40

def test_account_security_intent(classifier):
    res = classifier.classify("@AppleSupport Someone changed my Apple ID password and I am locked out of iCloud")
    assert res["predicted_intent"] == "ACCOUNT_ICLOUD_SECURITY"
    assert res["confidence"] > 0.40
