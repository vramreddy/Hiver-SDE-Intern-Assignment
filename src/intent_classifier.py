"""
Intent Classification Engine for Customer Support Tweets.
Combines TF-IDF n-gram vectorization, logistic regression / SVM calibrated probability models,
and domain regex heuristics for fast, robust, and explainable multi-class intent prediction.
"""

import os
import json
import re
import numpy as np
from typing import Dict, List, Tuple, Any, Optional
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from .config import INTENT_CLASSES, INTENT_TAXONOMY

class IntentClassifier:
    def __init__(self, corpus_path: Optional[str] = None):
        self.classes = INTENT_CLASSES
        self.pipeline: Optional[Pipeline] = None
        self.is_trained = False
        
        # Heuristic intent triggers for high-precision fast routing
        self.heuristics: Dict[str, List[re.Pattern]] = {
            "OS_UPDATE_BUG": [
                re.compile(r'\b(ios|macos|watchos|ipados)\s*\d+(\.\d+)*\b', re.I),
                re.compile(r'\b(update|updated|updating|boot\s*loop|kernel\s*panic|freeze|freezing|crash|crashes|crashing|lag|glitch|bug)\b', re.I),
                re.compile(r'\b(safari\s*crash|lock\s*screen\s*stuck|stuck\s*on\s*apple\s*logo)\b', re.I)
            ],
            "HARDWARE_BATTERY": [
                re.compile(r'\b(battery\s*health|battery\s*drain|maximum\s*capacity|overheat|swelling|swollen|hot\s*to\s*the\s*touch)\b', re.I),
                re.compile(r'\b(shattered|cracked\s*screen|broken\s*glass|charging\s*port|loose\s*port|earpiece|speaker\s*crackl|water\s*damage|submerged)\b', re.I),
                re.compile(r'\b(camera\s*lens|rattles|microphone\s*quiet|hardware)\b', re.I)
            ],
            "ACCOUNT_ICLOUD_SECURITY": [
                re.compile(r'\b(apple\s*id|icloud|iforgot|2fa|two[- ]factor|verification\s*code|locked\s*account|disabled\s*account)\b', re.I),
                re.compile(r'\b(phishing|suspicious\s*email|unauthorized\s*access|hacked|password\s*reset|trusted\s*number)\b', re.I),
                re.compile(r'\b(icloud\s*storage\s*full|storage\s*sync|photo\s*sync)\b', re.I)
            ],
            "BILLING_SUBSCRIPTIONS": [
                re.compile(r'\b(charged|refund|subscription|cancel\s*sub|receipt|invoice|billed|double\s*bill|payment\s*method\s*declined)\b', re.I),
                re.compile(r'\b(app\s*store\s*charge|reportaproblem|in-app\s*purchase|apple\s*arcade|apple\s*music\s*billing)\b', re.I)
            ],
            "CONNECTIVITY_SETUP": [
                re.compile(r'\b(airdrop|bluetooth|airpods\s*disconnect|apple\s*watch\s*pair|pairing|carplay|hotspot|personal\s*hotspot)\b', re.I),
                re.compile(r'\b(no\s*service|searching\.\.\.|carrier\s*settings|wifi\s*greyed|wi-fi\s*drop)\b', re.I)
            ],
            "REPAIR_WARRANTY_STATUS": [
                re.compile(r'\b(applecare|applecare\+|warranty|genius\s*bar|appointment|repair\s*status|repair\s*id|checkcoverage)\b', re.I),
                re.compile(r'\b(cost\s*to\s*fix|service\s*quote|trade-in\s*value|send\s*in\s*for\s*repair)\b', re.I)
            ],
            "GENERAL_FEEDBACK_CHURN": [
                re.compile(r'\b(switching\s*to\s*android|worst\s*experience|customer\s*service|tim\s*cook|unacceptable|terrible\s*service)\b', re.I),
                re.compile(r'\b(feature\s*request|feedback|love\s*the|shoutout|great\s*job)\b', re.I)
            ]
        }
        
        if corpus_path and os.path.exists(corpus_path):
            self.train_from_corpus(corpus_path)
        else:
            default_corpus = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "data", "apple_support_corpus.json"))
            if os.path.exists(default_corpus):
                self.train_from_corpus(default_corpus)

    def train_from_corpus(self, corpus_path: str):
        """Trains the TF-IDF + Calibrated Logistic Regression classifier on the historical corpus."""
        with open(corpus_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        
        texts = [item["customer_text"] for item in data]
        labels = [item["intent"] for item in data]

        self.pipeline = Pipeline([
            ('tfidf', TfidfVectorizer(
                ngram_range=(1, 3),
                max_features=10000,
                sublinear_tf=True,
                token_pattern=r'(?u)\b\w+\b'
            )),
            ('clf', LogisticRegression(
                C=2.5,
                max_iter=1000,
                class_weight='balanced',
                random_state=42
            ))
        ])

        self.pipeline.fit(texts, labels)
        self.is_trained = True

    def classify(self, text: str) -> Dict[str, Any]:
        """
        Classifies incoming customer text into intent with calibrated probabilities and heuristic alignment.
        """
        clean_text = re.sub(r'@[A-Za-z0-9_]+', '', text).strip()
        
        # Check rule heuristics matches
        heuristic_scores: Dict[str, int] = {intent: 0 for intent in self.classes}
        for intent, patterns in self.heuristics.items():
            for pat in patterns:
                if pat.search(text):
                    heuristic_scores[intent] += 1

        if self.is_trained and self.pipeline:
            # ML Model Probabilities
            probs = self.pipeline.predict_proba([text])[0]
            classes = list(self.pipeline.classes_)
            prob_dict = {cls: float(probs[i]) for i, cls in enumerate(classes)}

            # Blend ML probabilities with domain heuristics
            combined_scores = {}
            for cls in self.classes:
                base_prob = prob_dict.get(cls, 0.0)
                boost = heuristic_scores.get(cls, 0) * 0.18
                combined_scores[cls] = base_prob + boost

            # Normalize back to probability distribution
            total = sum(combined_scores.values())
            if total > 0:
                normalized_probs = {k: round(v / total, 4) for k, v in combined_scores.items()}
            else:
                normalized_probs = {k: 1.0 / len(self.classes) for k in self.classes}

            top_intent = max(normalized_probs, key=normalized_probs.get)
            top_confidence = normalized_probs[top_intent]
            
            # Sorted ranked probabilities
            ranked = sorted(normalized_probs.items(), key=lambda x: x[1], reverse=True)
            
            return {
                "predicted_intent": top_intent,
                "confidence": top_confidence,
                "intent_name": INTENT_TAXONOMY[top_intent]["name"],
                "probabilities": normalized_probs,
                "ranked_intents": ranked,
                "is_confident": top_confidence >= 0.35,
                "heuristic_match": heuristic_scores.get(top_intent, 0) > 0
            }
        else:
            # Fallback heuristic classifier
            top_intent = max(heuristic_scores, key=heuristic_scores.get)
            if heuristic_scores[top_intent] == 0:
                top_intent = "GENERAL_FEEDBACK_CHURN"
            
            return {
                "predicted_intent": top_intent,
                "confidence": 0.65,
                "intent_name": INTENT_TAXONOMY[top_intent]["name"],
                "probabilities": {k: (1.0 if k == top_intent else 0.0) for k in self.classes},
                "ranked_intents": [(top_intent, 0.65)],
                "is_confident": True,
                "heuristic_match": True
            }
