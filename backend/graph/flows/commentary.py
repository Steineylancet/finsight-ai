"""
FinSight AI — Commentary Drafter Flow
LangGraph subgraph: pulls live numbers for a period, identifies the top
narrative stories from the anomaly scan, then drafts CFO-level commentary.
Output is always clearly labelled as DRAFT.

Steps:
  1. extract_params       — mini LLM extracts FY / quarter
  2. run_anomaly_scan     — reuses anomaly scanner tool
  3. identify_stories     — mini LLM selects top 3 narrative hooks
  4. draft_commentary     — full LLM writes formal CFO commentary
"""

import json
import logging
import os
from langgraph.graph import StateGraph, END
from langchain_openai import AzureChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage

from backend.graph.state import AgentState
from backend.graph.tools import get_anomalies, get_department_quarterly_summary

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
        temperature=0.4,
    )


# ─────────────────────────────────────────────────────────────────────────────
# Node 1 — Extract period
# ─────────────────────────────────────────────────────────────────────────────

def extract_params(state: AgentState) -> AgentState:
    if state.get("fiscal_year") and state.get("quarter"):
        return state

    prompt = f"""Extract fiscal_year (4-digit string) and quarter (Q1/Q2/Q3/Q4) from this query.
Return JSON only. Query: {state["query"]}
Example: {{"fiscal_year": "2025", "quarter": "Q3"}}"""

    try:
        response = _mini_llm().invoke([HumanMessage(content=prompt)])
        parsed = json.loads(response.content.strip().strip("```json").strip("```"))
        return {
            **state,
            "fiscal_year": parsed.get("fiscal_year") or state.get("fiscal_year"),
            "quarter": parsed.get("quarter") or state.get("quarter"),
        }
    except Exception as e:
        return {**state, "error": f"Could not extract period from query: {e}"}


# ─────────────────────────────────────────────────────────────────────────────
# Node 2 — Run anomaly scan to surface key stories
# ─────────────────────────────────────────────────────────────────────────────

def run_anomaly_scan(state: AgentState) -> AgentState:
    if state.get("error"):
        return state

    fy = state.get("fiscal_year")
    q = state.get("quarter")

    if not all([fy, q]):
        return {**state, "error": "Fiscal year and quarter are required for commentary drafting."}

    result = get_anomalies.invoke({
        "fiscal_year": fy,
        "quarter": q,
        "threshold_pct": 5.0,   # Lower threshold for commentary — catch more stories
        "min_amount": 5000.0,
    })

    if "error" in result:
        return {**state, "error": result["error"]}

    return {**state, "tool_results": result}


# ─────────────────────────────────────────────────────────────────────────────
# Node 3 — Identify top narrative stories
# ─────────────────────────────────────────────────────────────────────────────

def identify_stories(state: AgentState) -> AgentState:
    if state.get("error"):
        return state

    anomalies = state["tool_results"].get("anomalies", [])
    fy = state["fiscal_year"]
    q = state["quarter"]

    if not anomalies:
        state["tool_results"]["stories"] = []
        return state

    anomaly_text = "\n".join(
        f"- {a['Department']}: ${a['Variance_BvA']:+,.0f} ({a['Variance_BvA_Pct']:+.1f}%)"
        for a in anomalies[:8]
    )

    prompt = f"""You are an FP&A analyst preparing CFO commentary for {q} FY{fy}.
These departments have notable budget variances:

{anomaly_text}

Select the top 3 stories that matter most for CFO-level commentary.
Consider: magnitude of variance, direction (over/under), and mix of departments.
Return a JSON list of department names only.
Example: ["Software Engineering", "Sales - Americas", "IT Infrastructure"]"""

    try:
        response = _mini_llm().invoke([HumanMessage(content=prompt)])
        content = response.content.strip().strip("```json").strip("```")
        stories = json.loads(content)
        state["tool_results"]["stories"] = stories
    except Exception as e:
        logger.warning(f"identify_stories failed: {e} — using top 3 by variance")
        state["tool_results"]["stories"] = [
            a["Department"] for a in anomalies[:3]
        ]

    return state


# ─────────────────────────────────────────────────────────────────────────────
# Node 4 — Draft CFO commentary
# ─────────────────────────────────────────────────────────────────────────────

def draft_commentary(state: AgentState) -> AgentState:
    if state.get("error"):
        return state

    stories = state["tool_results"].get("stories", [])
    anomalies = state["tool_results"].get("anomalies", [])
    fy = state["fiscal_year"]
    q = state["quarter"]

    # Pull summary data for each featured department
    dept_summaries = []
    for dept in stories:
        result = get_department_quarterly_summary.invoke({
            "department": dept,
            "fiscal_year": fy,
            "quarter": q,
        })
        if "error" not in result:
            total = next(
                (r for r in result.get("rows", []) if r.get("Expense_Category") == "TOTAL"),
                None,
            )
            if total:
                dept_summaries.append(
                    f"- {dept}: Budget ${total['Budget_USD']:,.0f} | "
                    f"Actuals ${total['Actuals_USD']:,.0f} | "
                    f"Variance ${total['Variance_BvA']:+,.0f} ({total['Variance_BvA_Pct']:+.1f}%)"
                )

    context = "\n".join(dept_summaries) if dept_summaries else "Data not available for featured departments."

    total_depts = state["tool_results"].get("total_departments_scanned", "N/A")
    flagged = state["tool_results"].get("flagged_count", 0)

    prompt = f"""Draft CFO-level management commentary for {q} FY{fy}.

Key department performance (actuals vs budget):
{context}

Overall: {flagged} of {total_depts} departments had notable variances this quarter.

Instructions:
- Write 3–4 paragraphs in formal FP&A register
- Opening paragraph: overall quarter summary (favourable or unfavourable, why)
- One paragraph per featured department: cite specific figures
- Closing paragraph: outlook and watch items
- Factual, concise — no filler phrases, no invented information
- Use dollar amounts and percentages exactly as given above"""

    try:
        response = _full_llm().invoke([
            SystemMessage(content=(
                "You are a senior FP&A analyst drafting CFO management commentary. "
                "Be precise, cite figures, and write in formal finance register."
            )),
            HumanMessage(content=prompt),
        ])
        draft = (
            "⚠️ DRAFT — Review and edit before distributing\n"
            "─────────────────────────────────────────────\n\n"
            + response.content.strip()
            + "\n\n─────────────────────────────────────────────\n"
            "⚠️ This is an AI-generated draft. Verify all figures before use."
        )
        return {**state, "narrative": draft}
    except Exception as e:
        logger.error(f"draft_commentary failed: {e}")
        return {**state, "error": f"Commentary generation failed: {e}"}


# ─────────────────────────────────────────────────────────────────────────────
# Graph assembly
# ─────────────────────────────────────────────────────────────────────────────

def build_commentary_graph():
    graph = StateGraph(AgentState)

    graph.add_node("extract_params", extract_params)
    graph.add_node("run_anomaly_scan", run_anomaly_scan)
    graph.add_node("identify_stories", identify_stories)
    graph.add_node("draft_commentary", draft_commentary)

    graph.set_entry_point("extract_params")
    graph.add_edge("extract_params", "run_anomaly_scan")
    graph.add_edge("run_anomaly_scan", "identify_stories")
    graph.add_edge("identify_stories", "draft_commentary")
    graph.add_edge("draft_commentary", END)

    return graph.compile()
