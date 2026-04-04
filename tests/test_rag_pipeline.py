"""
Tests for the RAG Pipeline
Run with: pytest tests/ -v
"""

import pytest
from unittest.mock import MagicMock, patch
from backend.rag_pipeline import RAGPipeline
from backend.models import ConversationTurn


@pytest.fixture
def mock_pipeline():
    """RAGPipeline with mocked Azure clients."""
    with patch("backend.rag_pipeline.AzureOpenAIClient") as MockOAI, \
         patch("backend.rag_pipeline.AzureSearchClient") as MockSearch:

        mock_oai = MockOAI.return_value
        mock_search = MockSearch.return_value

        # Mock embedding returns a 1536-dim vector
        mock_oai.get_embedding.return_value = [0.1] * 1536

        # Mock search returns 2 fake chunks
        mock_search.hybrid_search.return_value = [
            {
                "id": "chunk-001",
                "content": "Finance dept (RC-0001) had budget $36,136 and actuals $36,866 in P01 2022.",
                "title": "Planning — LE-US01, RC-0001, FY2022 P1, People Costs",
                "data_type": "planning",
                "entity": "Crestwood Capital Group USA",
                "department": "RC-0001",
                "fiscal_year": "2022",
                "fiscal_period": "P01",
                "expense_category": "People Costs",
                "score": 0.95,
            },
            {
                "id": "chunk-002",
                "content": "Workday Inc vendor invoice of $9,154 posted in Jan 2022 for Recruitment.",
                "title": "GL Transaction JE-202201-10000001 — Recruitment & Hiring",
                "data_type": "gl_transaction",
                "entity": "Crestwood Capital Group USA",
                "department": "Finance",
                "fiscal_year": "2022",
                "fiscal_period": "P01",
                "expense_category": "People Costs",
                "score": 0.88,
            },
        ]

        # Mock chat completion returns a simple string (non-stream)
        mock_oai.chat_completion.return_value = (
            "The Finance department had a budget of $36,136 and actuals of $36,866 in P1 2022, "
            "resulting in a $730 overspend (2.02% over budget)."
        )

        pipeline = RAGPipeline()
        pipeline.openai_client = mock_oai
        pipeline.search_client = mock_search
        yield pipeline


def test_retrieve_returns_chunks_and_sources(mock_pipeline):
    chunks, sources = mock_pipeline.retrieve("Finance budget Q1 2022")
    assert len(chunks) == 2
    assert len(sources) == 2
    assert sources[0].data_type == "planning"
    assert sources[1].data_type == "gl_transaction"


def test_embedding_dimension(mock_pipeline):
    embedding = mock_pipeline.openai_client.get_embedding("test")
    assert len(embedding) == 1536


def test_build_messages_includes_context(mock_pipeline):
    chunks, _ = mock_pipeline.retrieve("test question")
    messages = mock_pipeline.build_messages("test question", chunks, [])
    assert messages[0]["role"] == "system"
    assert "FinSight AI" in messages[0]["content"]
    user_msg = messages[-1]["content"]
    assert "test question" in user_msg
    assert "Finance dept" in user_msg  # Context chunk is included


def test_build_messages_with_history(mock_pipeline):
    history = [
        ConversationTurn(role="user", content="What is the budget?"),
        ConversationTurn(role="assistant", content="The budget is $36,136."),
    ]
    chunks, _ = mock_pipeline.retrieve("Follow-up question")
    messages = mock_pipeline.build_messages("Follow-up question", chunks, history)
    roles = [m["role"] for m in messages]
    assert "system" in roles
    assert roles.count("user") >= 2  # History + current question


def test_run_returns_response_and_sources(mock_pipeline):
    response, sources = mock_pipeline.run(
        question="What was Finance overspend in Q1 2022?",
        stream=False
    )
    assert isinstance(response, str)
    assert len(response) > 0
    assert len(sources) == 2


def test_run_empty_search_returns_fallback(mock_pipeline):
    mock_pipeline.search_client.hybrid_search.return_value = []
    response, sources = mock_pipeline.run("obscure query with no match", stream=False)
    assert "don't have enough data" in response.lower() or "couldn't find" in response.lower()
    assert sources == []
