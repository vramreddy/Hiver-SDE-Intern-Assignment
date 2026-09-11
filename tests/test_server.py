"""
Integration tests for the FastAPI server endpoints.
"""

import pytest
from fastapi.testclient import TestClient
from src.server import app


@pytest.fixture
def client():
    return TestClient(app)


class TestHealthEndpoint:
    def test_health_returns_200(self, client):
        resp = client.get("/api/health")
        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == "healthy"
        assert "version" in data

    def test_health_includes_brand(self, client):
        resp = client.get("/api/health")
        assert "@AppleSupport" in resp.json()["brand"]


class TestTaxonomyEndpoint:
    def test_taxonomy_returns_7_intents(self, client):
        resp = client.get("/api/taxonomy")
        assert resp.status_code == 200
        data = resp.json()
        assert len(data["classes"]) == 7
        assert "OS_UPDATE_BUG" in data["classes"]
        assert "HARDWARE_BATTERY" in data["classes"]


class TestAgentProcessEndpoint:
    def test_process_valid_query(self, client):
        resp = client.post(
            "/api/agent/process",
            json={"query": "@AppleSupport My iPhone battery health dropped to 70%."},
        )
        assert resp.status_code == 200
        data = resp.json()
        assert "intent" in data
        assert "draft_reply" in data
        assert "escalation" in data
        assert len(data["draft_reply"]) > 10

    def test_process_empty_query_returns_400(self, client):
        resp = client.post("/api/agent/process", json={"query": ""})
        assert resp.status_code == 400

    def test_process_whitespace_query_returns_400(self, client):
        resp = client.post("/api/agent/process", json={"query": "   "})
        assert resp.status_code == 400


class TestBaselineCompareEndpoint:
    def test_compare_returns_all_systems(self, client):
        resp = client.post(
            "/api/baselines/compare",
            json={"query": "@AppleSupport How do I check my AppleCare+ warranty?"},
        )
        assert resp.status_code == 200
        data = resp.json()
        assert "baseline0" in data
        assert "baseline1" in data
        assert "proposed_agent" in data
        assert data["baseline0"]["reply"] is not None
        assert data["proposed_agent"]["reply"] is not None

    def test_compare_empty_query_returns_400(self, client):
        resp = client.post("/api/baselines/compare", json={"query": ""})
        assert resp.status_code == 400
