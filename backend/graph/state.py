"""
FinSight AI — LangGraph State Definition
Typed state shared across all nodes in the agent graph.
"""

from typing import TypedDict, Optional


class AgentState(TypedDict):
    # ── Input ─────────────────────────────────────────────────────────────────
    query: str
    conversation_history: list  # list of ConversationTurn dicts

    # ── Routing ───────────────────────────────────────────────────────────────
    intent: str          # "rag" | "sql" | "variance" | "anomaly" | "commentary"
    mode_label: str      # Human-readable badge: "RAG" | "SQL Query" | "Variance Flow" | etc.

    # ── Extracted parameters (populated by classify node) ─────────────────────
    department: Optional[str]
    fiscal_year: Optional[str]
    quarter: Optional[str]
    threshold_pct: Optional[float]   # for anomaly scanner (default 10.0)
    min_amount: Optional[float]      # for anomaly scanner (default 10000.0)

    # ── Results ───────────────────────────────────────────────────────────────
    formatted_table: Optional[str]   # Markdown table string from pandas tool
    narrative: Optional[str]         # GPT-generated explanation / commentary
    sources: list                    # RAG source citations (Source model dicts)
    sql_query: Optional[str]         # Pandas expression shown to user (collapsed)
    tool_results: Optional[dict]     # Raw tool output — passed between subgraph nodes
                                      # (variance/anomaly/commentary flows)

    # ── Error ─────────────────────────────────────────────────────────────────
    error: Optional[str]
