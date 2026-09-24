"""
FinSight AI — LangGraph State Definition
Typed state shared across all nodes in the agent graph. LangGraph only
propagates keys declared here, so every field a node writes must be listed.
"""

from typing import Optional, TypedDict


class AgentState(TypedDict, total=False):
    # ── Input ─────────────────────────────────────────────────────────────────
    query: str
    conversation_history: list       # [{"role": "user"|"assistant", "content": str}]

    # ── Understanding (populated by the understand node) ──────────────────────
    standalone_query: str            # follow-ups rewritten to be self-contained
    intent: str                      # "rag" | "variance" | "anomaly" | "commentary" | "agent"
    mode_label: str                  # badge shown to the user
    department: Optional[str]
    fiscal_year: Optional[str]
    quarter: Optional[str]
    threshold_pct: Optional[float]
    min_amount: Optional[float]

    # ── Results ───────────────────────────────────────────────────────────────
    formatted_table: Optional[str]   # markdown table(s) built from tool output
    narrative: Optional[str]         # the answer text shown to the user
    sources: list                    # document citations (Source model dicts)
    sql_query: Optional[str]         # tool call(s) made, shown for transparency
    tool_results: Optional[dict]     # raw tool output — passed between nodes, used by evals

    # ── Error ─────────────────────────────────────────────────────────────────
    error: Optional[str]
