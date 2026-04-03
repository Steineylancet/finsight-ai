"""
Pydantic Models — Request and Response schemas for FinSight AI API
"""

from pydantic import BaseModel, Field
from typing import Optional


class ConversationTurn(BaseModel):
    role: str  # "user" or "assistant"
    content: str


class ChatRequest(BaseModel):
    question: str = Field(..., min_length=1, max_length=1000)
    conversation_history: list[ConversationTurn] = Field(default_factory=list)


class Source(BaseModel):
    title: str
    data_type: str
    entity: str
    department: str
    fiscal_year: str
    fiscal_period: str
    expense_category: str
    preview: str  # First 200 chars of chunk content


class ChatResponse(BaseModel):
    answer: str
    sources: list[Source]
    question: str


class SearchRequest(BaseModel):
    query: str = Field(..., min_length=1)
    top_k: int = Field(default=5, ge=1, le=20)


class HealthResponse(BaseModel):
    status: str
    service: str
    version: str
