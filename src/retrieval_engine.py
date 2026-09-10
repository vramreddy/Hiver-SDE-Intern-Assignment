"""
Hybrid Retrieval Engine for Grounded Historical Resolution Matching.
Indexes historical @AppleSupport interactions to retrieve the most semantically
and lexically relevant past agent resolutions for context grounding and few-shot calibration.
"""

import os
import json
import math
import re
from typing import List, Dict, Any, Optional
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

class RetrievalEngine:
    def __init__(self, corpus_path: Optional[str] = None):
        self.corpus: List[Dict[str, Any]] = []
        self.vectorizer: Optional[TfidfVectorizer] = None
        self.tfidf_matrix = None
        
        if corpus_path and os.path.exists(corpus_path):
            self.load_corpus(corpus_path)
        else:
            default_corpus = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "data", "apple_support_corpus.json"))
            if os.path.exists(default_corpus):
                self.load_corpus(default_corpus)

    def load_corpus(self, corpus_path: str):
        """Loads and indexes the historical resolution corpus."""
        with open(corpus_path, "r", encoding="utf-8") as f:
            self.corpus = json.load(f)

        # Build TF-IDF index over customer problem descriptions
        corpus_texts = [item["customer_text"] for item in self.corpus]
        self.vectorizer = TfidfVectorizer(
            ngram_range=(1, 2),
            stop_words="english",
            sublinear_tf=True
        )
        self.tfidf_matrix = self.vectorizer.fit_transform(corpus_texts)

    def retrieve(self, query: str, top_k: int = 3, filter_intent: Optional[str] = None) -> List[Dict[str, Any]]:
        """
        Retrieves top_k most relevant historical cases for a given customer query.
        Applies hybrid semantic similarity and intent-aware scoring.
        """
        if not self.corpus or self.vectorizer is None or self.tfidf_matrix is None:
            return []

        query_vec = self.vectorizer.transform([query])
        sim_scores = cosine_similarity(query_vec, self.tfidf_matrix)[0]

        # Apply intent bonus if filter_intent is provided
        ranked_indices = np.argsort(sim_scores)[::-1]
        
        results: List[Dict[str, Any]] = []
        for idx in ranked_indices:
            score = float(sim_scores[idx])
            item = self.corpus[idx]
            
            # Intent alignment bonus
            if filter_intent and item.get("intent") == filter_intent:
                effective_score = score + 0.15
            else:
                effective_score = score

            results.append({
                "id": item.get("id", f"hist_{idx}"),
                "customer_text": item["customer_text"],
                "agent_text": item["agent_text"],
                "intent": item["intent"],
                "auto_handle": item.get("auto_handle", True),
                "escalation_reason": item.get("escalation_reason", "STANDARD_TROUBLESHOOTING"),
                "similarity_score": round(effective_score, 4),
                "raw_cosine_score": round(score, 4)
            })

            if len(results) >= top_k * 3:
                break

        # Re-sort by effective score and return top_k
        results.sort(key=lambda x: x["similarity_score"], reverse=True)
        return results[:top_k]
