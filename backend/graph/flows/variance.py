"""
FinSight AI — Variance Explainer Flow
LangGraph subgraph: given a department + period, decompose why actuals
deviated from budget and generate a plain-English narrative.

Steps:
  1. extract_params   — mini LLM extracts dept / FY / quarter from query
  2. query_data       — pandas tool: actuals vs budget by expense category
  3. rank_drivers     — pure Python: sort by |variance|, pick top drivers
  4. draft_narrative  — full LLM: write explanation in FP&A language
"""

import json
import logging
import os
from langgraph.graph import StateGraph, END
from langchain_openai import AzureChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage

from backend.graph.state import AgentState
from backend.graph.tools import get_variance_breakdown

logger = logging.getLogger(__name__)


def _mini_llm():
    return AzureChatOpenAI(
        azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT", "PLACEHOLDER"),
        azure_deployment=os.getenv("AZURE_OPENAI_MINI_DEPLOYMENT", "gpt-5-4-mini"),
        api_key=os.getenv("AZURE_OPENAI_API_KEY", "PLACEHOLDER"),
        api_version=os.getenv("AZURE_OPENAI_API_VERSION", "2024-12-01-preview"),
        temperature=0,
    )


def _full_llm():
    return AzureChatOpenAI(
        azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT", "PLACEHOLDER"),
        azure_deployment=os.getenv("AZURE_OPENAI_FULL_DEPLOYMENT", "gpt-4o"),
        api_key=os.getenv("AZURE_OPENAI_API_KEY", "PLACEHOLDER"),
        api_version=os.getenv("AZURE_OPENAI_API_VERSION", "2024-12-01-preview"),
        temperature=0.3,
    )


# ─────────────────────────────────────────────────────────────────────────────
# Node 1 — Extract parameters if not already present in state
# ─────────────────────────────────────────────────────────────────────────────

def extract_params(state: AgentState) -> AgentState:
    """
    If department / fiscal_year / quarter are already in state (set by router),
    pass through. Otherwise, ask mini LLM to extract them from the query.
    """
    if state.get("department") and state.get("fiscal_year") and state.get("quarter"):
        return state

    prompt = f"""Extract the following from this financial query. Return JSON only.
Fields: department (string or null), fiscal_year (4-digit string or null), quarter (Q1/Q2/Q3/Q4 or null).

Query: {state["query"]}

Example output: {{"department": "Software Engineering", "fiscal_year": "2025", "quarter": "Q2"}}"""

    try:
        response = _mini_llm().invoke([HumanMessage(content=prompt)])
        parsed = json.loads(response.content.strip().strip("```json").strip("```"))
        return {
            **state,
            "department": parsed.get("department") or state.get("department"),
            "fiscal_year": parsed.get("fiscal_year") or state.get("fiscal_year"),
            "quarter": parsed.get("quarter") or state.get("quarter"),
        }
    except Exception as e:
        logger.warning(f"extract_params failed: {e}")
        return {**state, "error": f"Could not extract parameters from query: {e}"}


# ─────────────────────────────────────────────────────────────────────────────
# Node 2 — Query data via pandas tool
# ─────────────────────────────────────────────────────────────────────────────

def query_data(state: AgentState) -> AgentState:
    if state.get("error"):
        return state

    dept = state.get("department")
    fy = state.get("fiscal_year")
    q = state.get("quarter")

    if not all([dept, fy, q]):
        return {
            **state,
            "error": "Missing department, fiscal year, or quarter. Please specify all three.",
        }

    result = get_variance_breakdown.invoke({
        "department": dept,
        "fiscal_year": fy,
        "quarter": q,
    })

    if "error" in result:
        return {**state, "error": result["error"]}

    return {**state, "tool_results": result}


# ─────────────────────────────────────────────────────────────────────────────
# Node 3 — Build markdown table from ranked drivers
# ─────────────────────────────────────────────────────────────────────────────

def build_table(state: AgentState) -> AgentState:
    if state.get("error"):
        return state

    drivers = state["tool_results"].get("drivers", [])
    if not drivers:
        return {**state, "error": "No variance data available."}

    header = "| Expense Category | Budget ($) | Actuals ($) | Variance ($) | Variance % |"
    sep    = "|---|---|---|---|---|"
    rows = []
    for d in drivers:
        variance_flag = "▲" if d["Variance_BvA"] > 0 else "▼"
        rows.append(
            f"| {d['Expense_Category']} "
            f"| {d['Budget_USD']:,.0f} "
            f"| {d['Actuals_USD']:,.0f} "
            f"| {variance_flag} {abs(d['Variance_BvA']):,.0f} "
            f"| {d['Variance_BvA_Pct']:+.1f}% |"
        )

    table = "\n".join([header, sep] + rows)
    return {**state, "formatted_table": table}


# ─────────────────────────────────────────────────────────────────────────────
# Node 4 — Draft narrative with full LLM
# ─────────────────────────────────────────────────────────────────────────────

def draft_narrative(state: AgentState) -> AgentState:
    if state.get("error"):
        return state

    drivers = state["tool_results"].get("drivers", [])
    dept = state["department"]
    fy = state["fiscal_year"]
    q = state["quarter"]

    top_3 = drivers[:3]
    top_3_text = "\n".join(
        f"- {d['Expense_Category']}: ${d['Variance_BvA']:+,.0f} ({d['Variance_BvA_Pct']:+.1f}%)"
        for d in top_3
    )

    prompt = f"""You are an FP&A analyst writing a variance commentary for {dept}, {q} FY{fy}.

The top variance drivers (actuals vs budget) are:
{top_3_text}

▲ = over budget (unfavourable), ▼ = under budget (favourable).

Write 2–3 concise sentences explaining these variances in plain business English.
- Be specific: name the categories and cite the figures.
- Do not invent reasons beyond what the numbers show.
- Formal, factual tone. No filler phrases."""

    try:
        response = _full_llm().invoke([
            SystemMessage(content="You are a precise FP&A financial analyst."),
            HumanMessage(content=prompt),
        ])
        return {**state, "narrative": response.content.strip()}
    except Exception as e:
        logger.error(f"draft_narrative failed: {e}")
        return {**state, "error": f"Narrative generation failed: {e}"}


# ─────────────────────────────────────────────────────────────────────────────
# Graph assembly
# ─────────────────────────────────────────────────────────────────────────────

def _should_continue(state: AgentState) -> str:
    return "error" if state.get("error") else "continue"


def build_variance_graph():
    graph = StateGraph(AgentState)

    graph.add_node("extract_params", extract_params)
    graph.add_node("query_data", query_data)
    graph.add_node("build_table", build_table)
    graph.add_node("draft_narrative", draft_narrative)

    graph.set_entry_point("extract_params")
    graph.add_edge("extract_params", "query_data")
    graph.add_edge("query_data", "build_table")
    graph.add_edge("build_table", "draft_narrative")
    graph.add_edge("draft_narrative", END)

    return graph.compile()
