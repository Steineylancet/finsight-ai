"""
FinSight AI — Anomaly Scanner Flow
LangGraph subgraph: scan all departments for a given period, flag those
exceeding the variance threshold, and return a ranked severity list.

Steps:
  1. validate_params  — require a period, apply default thresholds
  2. scan             — pandas tool: variance % across all departments
  3. build_table      — ranked flagged list, streamed to the UI
  4. summarise        — full model writes a short director-level summary (streamed)
"""

import json
import logging

from langchain_core.messages import HumanMessage, SystemMessage
from langgraph.config import get_stream_writer
from langgraph.graph import END, StateGraph

from backend.graph.llm import final_llm
from backend.graph.state import AgentState
from backend.graph.tools import get_anomalies, run_tool

logger = logging.getLogger(__name__)

DEFAULT_THRESHOLD_PCT = 10.0
DEFAULT_MIN_AMOUNT = 10000.0


def validate_params(state: AgentState) -> dict:
    if not (state.get("fiscal_year") and state.get("quarter")):
        return {"error": "Which period should I scan? Please give a quarter and fiscal year, "
                         "e.g. \"Which departments are over budget in Q2 FY2025?\""}
    return {"threshold_pct": state.get("threshold_pct") or DEFAULT_THRESHOLD_PCT,
            "min_amount": state.get("min_amount") or DEFAULT_MIN_AMOUNT}


def scan(state: AgentState) -> dict:
    if state.get("error"):
        return {}
    args = {"fiscal_year": state["fiscal_year"], "quarter": state["quarter"],
            "threshold_pct": state["threshold_pct"], "min_amount": state["min_amount"]}
    result = run_tool(get_anomalies, args)
    if "error" in result:
        return {"error": result["error"]}
    return {"tool_results": result, "sql_query": f"get_anomalies({json.dumps(args)})"}


def build_table(state: AgentState) -> dict:
    if state.get("error"):
        return {}
    r = state["tool_results"]
    if not r["anomalies"]:
        msg = (f"No departments breached the thresholds in {r['quarter']} FY{r['fiscal_year']} "
               f"(|variance| ≥ {r['threshold_pct']:g}% and ≥ ${r['min_amount']:,.0f}). "
               f"All {r['total_departments_scanned']} departments are within tolerance.")
        get_stream_writer()({"type": "token", "content": msg})
        return {"formatted_table": None, "narrative": msg}

    rows = []
    for a in r["anomalies"]:
        status = "Over ▲" if a["Variance_BvA"] > 0 else "Under ▼"
        rows.append(
            f"| {a['Department']} | {a['Budget_USD']:,.0f} | {a['Actuals_USD']:,.0f} "
            f"| {a['Variance_BvA']:+,.0f} | {a['Variance_BvA_Pct']:+.1f}% | {status} |"
        )
    table = "\n".join(
        ["| Department | Budget ($) | Actuals ($) | Variance ($) | Variance % | Status |",
         "|---|---|---|---|---|---|"] + rows
    )
    get_stream_writer()({"type": "table", "content": table})
    return {"formatted_table": table}


def summarise(state: AgentState) -> dict:
    if state.get("error") or state.get("narrative"):
        return {}
    r = state["tool_results"]
    over = [a for a in r["anomalies"] if a["Variance_BvA"] > 0]
    under = [a for a in r["anomalies"] if a["Variance_BvA"] < 0]
    top = r["anomalies"][0]
    prompt = (
        f"{r['flagged_count']} of {r['total_departments_scanned']} departments breached the "
        f"thresholds (|variance| ≥ {r['threshold_pct']:g}% and ≥ ${r['min_amount']:,.0f}) in "
        f"{r['quarter']} FY{r['fiscal_year']}: {len(over)} over budget, {len(under)} under budget. "
        f"Largest: {top['Department']} at ${top['Variance_BvA']:+,.0f} ({top['Variance_BvA_Pct']:+.1f}%).\n\n"
        f"Write 1–2 concise sentences for an FP&A director. Say whether the pattern looks "
        f"systemic (most departments) or isolated (a few), citing the figures exactly as given."
    )
    try:
        resp = final_llm(0.2).invoke([
            SystemMessage(content="You are a concise FP&A analyst."),
            HumanMessage(content=prompt),
        ])
        return {"narrative": resp.content.strip()}
    except Exception as e:
        logger.error(f"summarise (anomaly) failed: {e}")
        return {"narrative": f"{r['flagged_count']} departments flagged — see the table above."}


def build_anomaly_graph():
    graph = StateGraph(AgentState)
    graph.add_node("validate_params", validate_params)
    graph.add_node("scan", scan)
    graph.add_node("build_table", build_table)
    graph.add_node("summarise", summarise)
    graph.set_entry_point("validate_params")
    graph.add_edge("validate_params", "scan")
    graph.add_edge("scan", "build_table")
    graph.add_edge("build_table", "summarise")
    graph.add_edge("summarise", END)
    return graph.compile()
