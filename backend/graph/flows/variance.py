"""
FinSight AI — Variance Explainer Flow
LangGraph subgraph: given a department + period, decompose why actuals
deviated from budget and generate a plain-English narrative.

Steps:
  1. validate_params  — resolve department against the whitelist, require a period
  2. query_data       — pandas tool: variance by expense category, largest first
  3. build_table      — ranked markdown table, streamed to the UI
  4. draft_narrative  — full model explains the top drivers (streamed)
"""

import json
import logging

from langchain_core.messages import HumanMessage, SystemMessage
from langgraph.config import get_stream_writer
from langgraph.graph import END, StateGraph

from backend.graph.catalog import resolve_department
from backend.graph.llm import final_llm
from backend.graph.state import AgentState
from backend.graph.tools import get_variance_breakdown, run_tool

logger = logging.getLogger(__name__)


def validate_params(state: AgentState) -> dict:
    raw = state.get("department")
    if not raw:
        return {"error": "Which department should I analyse? For example: "
                         "\"Why did Software Engineering overspend in Q2 FY2025?\""}
    res = resolve_department(raw)
    if not res.ok:
        return {"error": res.message(raw)}
    if not (state.get("fiscal_year") and state.get("quarter")):
        return {"department": res.department,
                "error": f"Which period for {res.department}? Please give a quarter and "
                         f"fiscal year, e.g. Q2 FY2025."}
    return {"department": res.department}


def query_data(state: AgentState) -> dict:
    if state.get("error"):
        return {}
    args = {"department": state["department"], "fiscal_year": state["fiscal_year"],
            "quarter": state["quarter"]}
    result = run_tool(get_variance_breakdown, args)
    if "error" in result:
        return {"error": result["error"]}
    return {"tool_results": result,
            "sql_query": f"get_variance_breakdown({json.dumps(args)})"}


def build_table(state: AgentState) -> dict:
    if state.get("error"):
        return {}
    drivers = state["tool_results"].get("drivers", [])
    if not drivers:
        return {"error": "No variance data available for that period."}

    rows = []
    for d in drivers:
        flag = "▲ over" if d["Variance_BvA"] > 0 else "▼ under"
        rows.append(
            f"| {d['Expense_Category']} | {d['Budget_USD']:,.0f} | {d['Actuals_USD']:,.0f} "
            f"| {d['Variance_BvA']:+,.0f} | {d['Variance_BvA_Pct']:+.1f}% | {flag} |"
        )
    total = state["tool_results"]["total"]
    rows.append(
        f"| **TOTAL** | {total['Budget_USD']:,.0f} | {total['Actuals_USD']:,.0f} "
        f"| {total['Variance_BvA']:+,.0f} | {total['Variance_BvA_Pct']:+.1f}% | |"
    )
    table = "\n".join(
        ["| Expense Category | Budget ($) | Actuals ($) | Variance ($) | Variance % | |",
         "|---|---|---|---|---|---|"] + rows
    )
    get_stream_writer()({"type": "table", "content": table})
    return {"formatted_table": table}


def draft_narrative(state: AgentState) -> dict:
    if state.get("error"):
        return {}
    r = state["tool_results"]
    total = r["total"]
    drivers_text = "\n".join(
        f"- {d['Expense_Category']}: budget ${d['Budget_USD']:,.0f}, actuals "
        f"${d['Actuals_USD']:,.0f}, variance ${d['Variance_BvA']:+,.0f} ({d['Variance_BvA_Pct']:+.1f}%)"
        for d in r["drivers"][:3]
    )
    prompt = f"""Explain the budget variance for {r['department']}, {r['quarter']} FY{r['fiscal_year']}.

Total: budget ${total['Budget_USD']:,.0f}, actuals ${total['Actuals_USD']:,.0f}, variance ${total['Variance_BvA']:+,.0f} ({total['Variance_BvA_Pct']:+.1f}%).
Top drivers (positive variance = over budget, unfavourable):
{drivers_text}

Write 2–3 concise sentences in plain business English.
- Name the categories and cite the figures exactly as given.
- Do not invent causes the numbers don't show; you may say what further detail would confirm the cause.
- Formal, factual tone. No filler."""
    try:
        resp = final_llm(0.3).invoke([
            SystemMessage(content="You are a precise FP&A financial analyst."),
            HumanMessage(content=prompt),
        ])
        return {"narrative": resp.content.strip()}
    except Exception as e:
        logger.error(f"draft_narrative failed: {e}")
        return {"error": "The data was retrieved but the explanation could not be generated. "
                         "The table above shows the variance drivers."}


def build_variance_graph():
    graph = StateGraph(AgentState)
    graph.add_node("validate_params", validate_params)
    graph.add_node("query_data", query_data)
    graph.add_node("build_table", build_table)
    graph.add_node("draft_narrative", draft_narrative)
    graph.set_entry_point("validate_params")
    graph.add_edge("validate_params", "query_data")
    graph.add_edge("query_data", "build_table")
    graph.add_edge("build_table", "draft_narrative")
    graph.add_edge("draft_narrative", END)
    return graph.compile()
