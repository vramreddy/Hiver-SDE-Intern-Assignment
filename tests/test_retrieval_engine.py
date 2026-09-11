"""
Tests for the Retrieval Engine (TF-IDF cosine similarity).
"""

import pytest
from src.retrieval_engine import RetrievalEngine


@pytest.fixture
def engine():
    return RetrievalEngine()


class TestRetrievalBasics:
    def test_retrieval_returns_results(self, engine):
        """A valid Apple Support query should return non-empty results."""
        if not engine.corpus:
            pytest.skip("No corpus loaded — run data pipeline first.")
        results = engine.retrieve(
            "@AppleSupport My iPhone battery health dropped to 72%", top_k=3
        )
        assert len(results) > 0
        assert len(results) <= 3

    def test_retrieval_result_format(self, engine):
        """Each result should have the required keys."""
        if not engine.corpus:
            pytest.skip("No corpus loaded — run data pipeline first.")
        results = engine.retrieve(
            "@AppleSupport How do I reset my AirPods Pro?", top_k=1
        )
        if results:
            result = results[0]
            assert "customer_text" in result
            assert "agent_text" in result
            assert "intent" in result
            assert "similarity_score" in result
            assert "raw_cosine_score" in result
            assert isinstance(result["similarity_score"], float)

    def test_retrieval_empty_query(self, engine):
        """Empty query should still return results (TF-IDF handles it)."""
        if not engine.corpus:
            pytest.skip("No corpus loaded — run data pipeline first.")
        results = engine.retrieve("", top_k=3)
        # May return empty or low-score results — just shouldn't crash
        assert isinstance(results, list)


class TestIntentFilterBonus:
    def test_intent_filter_boosts_matching_results(self, engine):
        """Results with matching intent should have higher effective score
        than raw cosine score due to the +0.15 bonus."""
        if not engine.corpus:
            pytest.skip("No corpus loaded — run data pipeline first.")
        results = engine.retrieve(
            "@AppleSupport My iPhone battery is swelling",
            top_k=5,
            filter_intent="HARDWARE_BATTERY",
        )
        for result in results:
            if result["intent"] == "HARDWARE_BATTERY":
                assert result["similarity_score"] >= result["raw_cosine_score"]

    def test_no_filter_returns_diverse_intents(self, engine):
        """Without intent filter, results may span multiple intents."""
        if not engine.corpus:
            pytest.skip("No corpus loaded — run data pipeline first.")
        results = engine.retrieve(
            "@AppleSupport Help me with my device please", top_k=5
        )
        # Just verify it runs without error; diversity depends on corpus
        assert isinstance(results, list)


class TestRetrievalEdgeCases:
    def test_no_corpus_returns_empty(self):
        """Engine with no corpus should return empty results, not crash."""
        empty_engine = RetrievalEngine.__new__(RetrievalEngine)
        empty_engine.corpus = []
        empty_engine.vectorizer = None
        empty_engine.tfidf_matrix = None

        results = empty_engine.retrieve("test query", top_k=3)
        assert results == []
