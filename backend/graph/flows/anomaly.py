"""
FinSight AI — Anomaly Scanner Flow
LangGraph subgraph: scan all departments for a given period, flag those
exceeding the variance threshold, and return a ranked severity list.

Steps:
  1. extract_params   — mini LLM extracts FY / quarter / threshold from query
  2. scan             — pandas tool: compute variance% across all departments
  3. build_table      — format ranked flagged list as markdown
  4. summarise        — mini LLM writes a 1–2 sentence summary line
"""

import json
import logging
import os
from langgraph.graph import StateGraph, END
from langchain_openai import AzureChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage

from backend.graph.state import AgentState
from backend.graph.tools import get_anomalies

logger = logging.getLogger(__name__)


def _mini_llm():
    return AzureChatOpenAI(
        azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT", "PLACEHOLDER"),
        azure_deployment=os.getenv("AZURE_OPENAI_MINI_DEPLOYMENT", "gpt-5-4-mini"),
        api_key=os.getenv("AZURE_OPENAI_API_KEY", "PLACEHOLDER"),
        api_version=os.getenv("AZURE_OPENAI_API_VERSION", "2024-12-01-preview"),
        temperature=0,
    )


# ─────────────────────────────────────────────────────────────────────────────
# Node 1 — Extract parameters
# ─────────────────────────────────────────────────────────────────────────────

def extract_params(state: AgentState) -> AgentState:
    if state.get("fiscal_year") and state.get("quarter"):
        return state

    prompt = f"""Extract from this query: fiscal_year (4-digit string), quarter (Q1/Q2/Q3/Q4),
and optionally threshold_pct (number, default 10) and min_amount (number, default 10000).
Return JSON only.

Query: {state["query"]}
Example: {{"fiscal_year": "2025", "quarter": "Q3", "threshold_pct": 10, "min_amount": 10000}}"""

    try:
        response = _mini_llm().invoke([HumanMessage(content=prompt)])
        parsed = json.loads(response.content.strip().strip("```json").strip("```"))
        return {
            **state,
            "fiscal_year": parsed.get("fiscal_year") or state.get("fiscal_year"),
            "quarter": parsed.get("quarter") or state.get("quarter"),
            "threshold_pct": parsed.get("threshold_pct") or state.get("threshold_pct") or 10.0,
            "min_amount": parsed.get("min_amount") or state.get("min_amount") or 10000.0,
        }
    except Exception as e:
        logger.warning(f"extract_params (anomaly) failed: {e}")
        return {**state, "error": f"Could not extract parameters: {e}"}


# ─────────────────────────────────────────────────────────────────────────────
# Node 2 — Run the anomaly scan
# ─────────────────────────────────────────────────────────────────────────────

def scan(state: AgentState) -> AgentState:
    if state.get("error"):
        return state

    fy = state.get("fiscal_year")
    q = state.get("quarter")

    if not all([fy, q]):
        return {**state, "error": "Please specify a fiscal year and quarter to scan."}

    result = get_anomalies.invoke({
        "fiscal_year": fy,
        "quarter": q,
        "threshold_pct": state.get("threshold_pct") or 10.0,
        "min_amount": state.get("min_amount") or 10000.0,
    })

    if "error" in result:
        return {**state, "error": result["error"]}

    return {**state, "tool_results": result}


# ─────────────────────────────────────────────────────────────────────────────
# Node 3 — Build markdown table
# ─────────────────────────────────────────────────────────────────────────────

def build_table(state: AgentState) -> AgentState:
    if state.get("error"):
        return state

    anomalies = state["tool_results"].get("anomalies", [])
    result = state["tool_results"]

    if not anomalies:
        return {
            **state,
            "formatted_table": None,
            "narrative": (
                f"No anomalies found in {result['quarter']} FY{result['fiscal_year']}. "
                f"All departments are within the {result['threshold_pct']}% variance threshold."
            ),
        }

    header = "| Department | Budget ($) | Actuals ($) | Variance ($) | Variance % | Status |"
    sep    = "|---|---|---|---|---|---|"
    rows = []
    for a in anomalies:
        direction = "Over Budget ▲" if a["Variance_BvA"] > 0 else "Under Budget ▼"
        rows.append(
            f"| {a['Department']} "
            f"| {a['Budget_USD']:,.0f} "
            f"| {a['Actuals_USD']:,.0f} "
            f"| {abs(a['Variance_BvA']):,.0f} "
            f"| {a['Variance_BvA_Pct']:+.1f}% "
            f"| {direction} |"
        )

    table = "\n".join([header, sep] + rows)
    return {**state, "formatted_table": table}


# ─────────────────────────────────────────────────────────────────────────────
# Node 4 — Generate summary sentence
# ─────────────────────────────────────────────────────────────────────────────

def summarise(state: AgentState) -> AgentState:
    if state.get("error") or state.get("narrative"):
        return state

    result = state["tool_results"]
    anomalies = result.get("anomalies", [])

    if not anomalies:
        return state

    top_dept = anomalies[0]["Department"]
    top_variance = anomalies[0]["Variance_BvA"]
    top_pct = anomalies[0]["Variance_BvA_Pct"]
    count = result["flagged_count"]
    total = result["total_departments_scanned"]

    prompt = (
        f"{count} of {total} departments exceeded the {result['threshold_pct']}% variance "
        f"threshold in {result['quarter']} FY{result['fiscal_year']}. "
        f"The largest variance is {top_dept} at ${top_variance:+,.0f} ({top_pct:+.1f}%). "
        f"Write 1–2 concise sentences summarising this for an FP&A director. "
        f"Flag whether the pattern suggests systemic or isolated issues."
    )

    try:
        response = _mini_llm().invoke([
            SystemMessage(content="You are a concise FP&A analyst."),
            HumanMessage(content=prompt),
        ])
        return {**state, "narrative": response.content.strip()}
    except Exception as e:
        logger.error(f"summarise (anomaly) failed: {e}")
        return {**state, "narrative": f"{count} departments flagged — see table above."}


# ─────────────────────────────────────────────────────────────────────────────
# Graph assembly
# ─────────────────────────────────────────────────────────────────────────────

def build_anomaly_graph():
    graph = StateGraph(AgentState)

    graph.add_node("extract_params", extract_params)
    graph.add_node("scan", scan)
    graph.add_node("build_table", build_table)
    graph.add_node("summarise", summarise)

    graph.set_entry_point("extract_params")
    graph.add_edge("extract_params", "scan")
    graph.add_edge("scan", "build_table")
    graph.add_edge("build_table", "summarise")
    graph.add_edge("summarise", END)

    return graph.compile()
