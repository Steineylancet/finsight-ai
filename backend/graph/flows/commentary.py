"""
FinSight AI — Commentary Drafter Flow
LangGraph subgraph: pulls live numbers for a period, picks the most important
stories from the anomaly scan, then drafts CFO-level commentary.
Output is always clearly labelled as DRAFT.

Steps:
  1. validate_params   — require a period
  2. run_anomaly_scan  — reuses the anomaly tool at a lower (5%) threshold
  3. identify_stories  — mini model picks the top 3 departments (structured output)
  4. draft_commentary  — full model writes formal CFO commentary (streamed)
"""

import json
import logging

from langchain_core.messages import HumanMessage, SystemMessage
from langgraph.config import get_stream_writer
from langgraph.graph import END, StateGraph
from pydantic import BaseModel, Field

from backend.graph.catalog import resolve_department
from backend.graph.llm import final_llm, mini_llm
from backend.graph.state import AgentState
from backend.graph.tools import get_anomalies, run_tool

logger = logging.getLogger(__name__)

COMMENTARY_THRESHOLD_PCT = 5.0
COMMENTARY_MIN_AMOUNT = 5000.0
DRAFT_HEADER = "**⚠️ DRAFT — review and edit before distributing**\n\n"
DRAFT_FOOTER = "\n\n*AI-generated draft. Verify all figures against the table before use.*"


class Stories(BaseModel):
    departments: list[str] = Field(description="Exactly 3 department names, most important first.")


def validate_params(state: AgentState) -> dict:
    if not (state.get("fiscal_year") and state.get("quarter")):
        return {"error": "Which period should the commentary cover? Please give a quarter and "
                         "fiscal year, e.g. \"Draft the CFO commentary for Q2 FY2025\"."}
    return {}


def run_anomaly_scan(state: AgentState) -> dict:
    if state.get("error"):
        return {}
    args = {"fiscal_year": state["fiscal_year"], "quarter": state["quarter"],
            "threshold_pct": COMMENTARY_THRESHOLD_PCT, "min_amount": COMMENTARY_MIN_AMOUNT}
    result = run_tool(get_anomalies, args)
    if "error" in result:
        return {"error": result["error"]}
    return {"tool_results": result, "sql_query": f"get_anomalies({json.dumps(args)})"}


def identify_stories(state: AgentState) -> dict:
    if state.get("error"):
        return {}
    r = state["tool_results"]
    anomalies = r["anomalies"]
    by_size = sorted(anomalies, key=lambda a: abs(a["Variance_BvA"]), reverse=True)
    fallback = [a["Department"] for a in by_size[:3]]
    if len(anomalies) <= 3:
        return {"tool_results": {**r, "stories": fallback}}

    listing = "\n".join(
        f"- {a['Department']}: ${a['Variance_BvA']:+,.0f} ({a['Variance_BvA_Pct']:+.1f}%)"
        for a in by_size[:10]
    )
    try:
        picked = mini_llm().with_structured_output(Stories).invoke(
            f"Pick the 3 departments that matter most for CFO commentary on "
            f"{r['quarter']} FY{r['fiscal_year']}, weighing dollar magnitude, direction "
            f"(over vs under budget) and a mix of functions:\n{listing}"
        )
        flagged = {a["Department"] for a in anomalies}
        stories = []
        for name in picked.departments:
            res = resolve_department(name)
            if res.ok and res.department in flagged and res.department not in stories:
                stories.append(res.department)
        stories = (stories + [d for d in fallback if d not in stories])[:3]
    except Exception as e:
        logger.warning(f"identify_stories failed, using largest variances: {e}")
        stories = fallback
    return {"tool_results": {**r, "stories": stories}}


def draft_commentary(state: AgentState) -> dict:
    if state.get("error"):
        return {}
    r = state["tool_results"]
    total = r["company_total"]
    featured = [a for a in r["anomalies"] if a["Department"] in r.get("stories", [])]
    writer = get_stream_writer()

    if featured:
        table = "\n".join(
            ["| Department | Budget ($) | Actuals ($) | Variance ($) | Variance % |",
             "|---|---|---|---|---|"]
            + [f"| {a['Department']} | {a['Budget_USD']:,.0f} | {a['Actuals_USD']:,.0f} "
               f"| {a['Variance_BvA']:+,.0f} | {a['Variance_BvA_Pct']:+.1f}% |" for a in featured]
            + [f"| **All departments** | {total['Budget_USD']:,.0f} | {total['Actuals_USD']:,.0f} "
               f"| {total['Variance_BvA']:+,.0f} | {total['Variance_BvA_Pct']:+.1f}% |"]
        )
        writer({"type": "table", "content": table})
    else:
        table = None

    context = "\n".join(
        f"- {a['Department']}: budget ${a['Budget_USD']:,.0f}, actuals ${a['Actuals_USD']:,.0f}, "
        f"variance ${a['Variance_BvA']:+,.0f} ({a['Variance_BvA_Pct']:+.1f}%)"
        for a in featured
    ) or "- No department exceeded the 5% variance threshold."

    prompt = f"""Draft CFO-level management commentary on operating expenses for {r['quarter']} FY{r['fiscal_year']}.

Company total (all {r['total_departments_scanned']} departments): budget ${total['Budget_USD']:,.0f}, actuals ${total['Actuals_USD']:,.0f}, variance ${total['Variance_BvA']:+,.0f} ({total['Variance_BvA_Pct']:+.1f}%).
{r['flagged_count']} of {r['total_departments_scanned']} departments were more than {COMMENTARY_THRESHOLD_PCT:g}% off budget.
Featured departments:
{context}

Instructions:
- 3–4 short paragraphs in formal FP&A register.
- Opening: overall quarter result, favourable or unfavourable.
- One paragraph covering the featured departments, citing figures exactly as given.
- Closing: watch items and suggested follow-ups. Do not invent causes or forward-looking numbers.
- No headings, no filler."""

    writer({"type": "token", "content": DRAFT_HEADER})
    try:
        resp = final_llm(0.4).invoke([
            SystemMessage(content="You are a senior FP&A analyst drafting CFO management commentary."),
            HumanMessage(content=prompt),
        ])
    except Exception as e:
        logger.error(f"draft_commentary failed: {e}")
        return {"formatted_table": table, "error": "Commentary generation failed — please retry."}
    writer({"type": "token", "content": DRAFT_FOOTER})
    return {"formatted_table": table,
            "narrative": DRAFT_HEADER + resp.content.strip() + DRAFT_FOOTER}


def build_commentary_graph():
    graph = StateGraph(AgentState)
    graph.add_node("validate_params", validate_params)
    graph.add_node("run_anomaly_scan", run_anomaly_scan)
    graph.add_node("identify_stories", identify_stories)
    graph.add_node("draft_commentary", draft_commentary)
    graph.set_entry_point("validate_params")
    graph.add_edge("validate_params", "run_anomaly_scan")
    graph.add_edge("run_anomaly_scan", "identify_stories")
    graph.add_edge("identify_stories", "draft_commentary")
    graph.add_edge("draft_commentary", END)
    return graph.compile()
