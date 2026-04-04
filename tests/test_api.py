"""
Tests for the FastAPI endpoints
Run with: pytest tests/ -v
"""

import pytest
from unittest.mock import MagicMock, patch, AsyncMock
from fastapi.testclient import TestClient


@pytest.fixture
def client():
    """Test client with mocked RAG pipeline."""
    with patch("backend.main.RAGPipeline") as MockPipeline:
        mock_pipeline = MockPipeline.return_value

        # Mock the run method to return a string (non-stream) + empty sources
        mock_pipeline.run.return_value = (
            "The Finance department had a $730 budget overspend in Q1 2022.",
            [],
        )
        mock_pipeline.openai_client = MagicMock()
        mock_pipeline.search_client = MagicMock()
        mock_pipeline.openai_client.get_embedding.return_value = [0.1] * 1536
        mock_pipeline.search_client.hybrid_search.return_value = []

        from backend.main import app
        with TestClient(app) as c:
            yield c


def test_health_check(client):
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "ok"
    assert data["service"] == "FinSight AI"


def test_root_endpoint(client):
    response = client.get("/")
    assert response.status_code == 200
    assert "text/html" in response.headers["content-type"]


def test_chat_empty_question(client):
    response = client.post("/chat", json={"question": ""})
    assert response.status_code == 422  # Pydantic validation error


def test_chat_question_too_long(client):
    response = client.post("/chat", json={"question": "a" * 1001})
    assert response.status_code == 422


def test_search_endpoint(client):
    response = client.get("/search?q=Finance+budget")
    assert response.status_code == 200
    data = response.json()
    assert "query" in data
    assert "results" in data
