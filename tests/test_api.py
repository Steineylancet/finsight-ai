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


class _FakeGraph:
    """Stands in for the compiled LangGraph: replays a fixed astream sequence."""

    def __init__(self, final_state, extra_events=()):
        self.final_state = final_state
        self.extra_events = extra_events

    async def astream(self, state, stream_mode=None, subgraphs=False):
        from langchain_core.messages import AIMessageChunk
        yield (), "custom", {"type": "mode", "mode": "Variance Explainer", "intent": "variance"}
        yield ("variance:1",), "custom", {"type": "table", "content": "| a |\n|---|\n| 1 |"}
        yield ("variance:1",), "messages", (AIMessageChunk(content="routing"), {"tags": []})
        yield ("variance:1",), "messages", (AIMessageChunk(content="Over "), {"tags": ["final_answer"]})
        yield ("variance:1",), "messages", (AIMessageChunk(content="budget."), {"tags": ["final_answer"]})
        for ev in self.extra_events:
            yield ev
        yield (), "values", self.final_state

    async def ainvoke(self, state):
        return self.final_state


def _sse_events(response):
    import json
    return [json.loads(line[6:]) for line in response.text.splitlines()
            if line.startswith("data: ") and line != "data: [DONE]"]


def test_chat_streams_mode_table_tokens_and_done(client):
    import backend.main as main
    main.agent_graph = _FakeGraph({"narrative": "Over budget.", "sql_query": "get_variance_breakdown({})"})
    response = client.post("/chat", json={"question": "Why did IT overspend in Q2 FY2025?"})
    assert response.status_code == 200
    events = _sse_events(response)
    types = [e["type"] for e in events]
    assert types == ["mode", "table", "token", "token", "done"]
    assert "".join(e["content"] for e in events if e["type"] == "token") == "Over budget."
    assert events[-1]["sql_query"].startswith("get_variance_breakdown")
    assert response.text.rstrip().endswith("data: [DONE]")


def test_chat_surfaces_flow_errors(client):
    import backend.main as main
    main.agent_graph = _FakeGraph({"error": '"Sales" matches more than one department'})
    events = _sse_events(client.post("/chat", json={"question": "Why did Sales overspend?"}))
    assert {"type": "error", "message": '"Sales" matches more than one department'} in events


def test_agent_endpoint_returns_json(client):
    import backend.main as main
    main.agent_graph = _FakeGraph({"mode_label": "SQL Agent", "intent": "agent",
                                   "narrative": "Done.", "formatted_table": "| a |"})
    data = client.post("/agent", json={"question": "Top vendors Q1 FY2025"}).json()
    assert data["mode"] == "SQL Agent" and data["narrative"] == "Done." and data["error"] is None
