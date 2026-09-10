"""
Configuration and Taxonomy Definitions for @AppleSupport AI Agent.
"""

from typing import Dict, List
from pydantic import BaseModel

BRAND_HANDLE = "@AppleSupport"
BRAND_NAME = "Apple Support"

# 7-Class Intent Taxonomy grounded in Twitter Customer Support Data
INTENT_TAXONOMY: Dict[str, Dict[str, str]] = {
    "OS_UPDATE_BUG": {
        "name": "OS Update & Software Glitches",
        "description": "Issues arising after iOS/macOS/watchOS/iPadOS updates, app crashing, system freezing, lag, boot loops, or software bugs.",
        "typical_keywords": ["update", "ios", "macos", "version", "lag", "crash", "freeze", "bug", "glitch", "battery drain after update", "reboot"]
    },
    "HARDWARE_BATTERY": {
        "name": "Hardware & Battery Health",
        "description": "Physical device defects, degraded battery capacity, overheating, charging port issues, speaker/mic failure, broken glass.",
        "typical_keywords": ["battery", "draining", "health", "screen", "cracked", "overheating", "hot", "charger", "cable", "speaker", "microphone", "camera"]
    },
    "ACCOUNT_ICLOUD_SECURITY": {
        "name": "Apple ID, iCloud & Account Security",
        "description": "Locked Apple ID, forgotten passwords, 2-Factor Authentication issues, iCloud storage full/sync errors, suspicious login alerts.",
        "typical_keywords": ["apple id", "password", "locked", "icloud", "storage", "sync", "backup", "hacked", "2fa", "verification code", "disabled"]
    },
    "BILLING_SUBSCRIPTIONS": {
        "name": "Billing, App Store & Subscriptions",
        "description": "Unexpected App Store charges, subscription management/cancellation, refund requests, payment method declines, double billing.",
        "typical_keywords": ["charged", "refund", "subscription", "cancel", "bill", "invoice", "receipt", "purchase", "credit card", "payment declined"]
    },
    "CONNECTIVITY_SETUP": {
        "name": "Connectivity, Bluetooth & Accessories Setup",
        "description": "Wi-Fi dropouts, cellular data issues, Bluetooth pairing with AirPods/Apple Watch/CarPlay, AirDrop failures.",
        "typical_keywords": ["wifi", "wi-fi", "bluetooth", "airpods", "apple watch", "pairing", "connect", "airdrop", "cellular", "no service", "hotspot"]
    },
    "REPAIR_WARRANTY_STATUS": {
        "name": "Repair, AppleCare+ & Genius Bar",
        "description": "Checking warranty coverage, AppleCare+ claims, Genius Bar appointment scheduling, repair turnaround status, service quotes.",
        "typical_keywords": ["applecare", "warranty", "genius bar", "appointment", "repair", "service center", "quote", "replacement", "cost to fix"]
    },
    "GENERAL_FEEDBACK_CHURN": {
        "name": "General Inquiries & Customer Feedback",
        "description": "Brand feedback, product feature praise or complaints, feature requests, high-level non-technical queries, or general sentiment.",
        "typical_keywords": ["hate", "love", "feedback", "feature request", "why did apple", "switching to android", "worst", "recommend", "suggestion"]
    }
}

INTENT_CLASSES = list(INTENT_TAXONOMY.keys())

# Escalation Trigger Categories
ESCALATION_REASONS = {
    "PII_SECURITY_DM": "Requires customer Private Identifying Information (Apple ID email, serial number, IMEI) or secure diagnostic links.",
    "BILLING_REFUND_AUTH": "Requires financial authorization, chargeback review, or direct App Store refund processing.",
    "HARDWARE_PHYSICAL_DAMAGE": "Physical hardware damage or safety hazard (swollen battery, cracked enclosure) needing in-person Genius Bar inspection.",
    "HIGH_SENTIMENT_CHURN_RISK": "Extreme customer frustration, legal threats, or active churn risk requiring senior human empathy.",
    "LOW_CONFIDENCE_AMBIGUITY": "Customer query is ambiguous, contradictory, or model confidence is below reliability threshold.",
    "MULTI_INTENT_COMPLEX": "Complex multi-faceted issue with compounding root causes."
}

# Auto-Handle Eligible Scenarios (Step-by-step self-serve guides, standard public documentation links, general troubleshooting)
AUTO_HANDLE_REASONS = {
    "STANDARD_TROUBLESHOOTING": "Standard self-serve troubleshooting steps available (e.g., force restart, network reset, software update check).",
    "PUBLIC_DOCS_GUIDE": "Public Apple support guide and knowledge-base article directly addresses the inquiry.",
    "GENERAL_INFORMATIONAL": "General feature question or non-sensitive status inquiry with clear resolution."
}
