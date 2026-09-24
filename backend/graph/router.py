"""
FinSight AI — LangGraph Router (Main Agent Graph)

understand ──┬─ rag         document Q&A over the policy / memo corpus
             ├─ variance    subgraph: why one department deviated from budget
             ├─ anomaly     subgraph: which departments breached thresholds
             ├─ commentary  subgraph: CFO commentary draft
             └─ agent       tool-calling agent over the 5 data tools + document
                            search, for lookups and multi-part questions that
                            need numbers and policy together

`understand` is one structured-output call on the mini model: it rewrites
follow-ups into standalone questions using the conversation history,
classifies intent and extracts parameters. Deterministic routing keeps the
common paths cheap and predictable; the agent handles everything open-ended.
"""

import json
import logging
from datetime import date
from typing import Literal, Optional

from langchain.agents import create_agent
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage
from langgraph.config import get_stream_writer
from langgraph.errors import GraphRecursionError
from langgraph.graph import END, StateGraph
from pydantic import BaseModel, Field

from backend.data_loader import DataLoader
from backend.graph.catalog import DEPARTMENTS
from backend.graph.flows.anomaly import build_anomaly_graph
from backend.graph.flows.commentary import build_commentary_graph
from backend.graph.flows.variance import build_variance_graph
from backend.graph.formatting import tool_result_table
from backend.graph.llm import final_llm, mini_llm
from backend.graph.retriever import FinSightRetriever, format_documents, make_search_tool
from backend.graph.state import AgentState
from backend.graph.tools import DATA_TOOLS, norm_fy

logger = logging.getLogger(__name__)

HISTORY_TURNS = 6
HISTORY_CHARS_PER_TURN = 1500
TOOL_OUTPUT_CHARS = 6000
AGENT_RECURSION_LIMIT = 12

MODE_LABELS = {
    "rag": "Document Q&A (RAG)",
    "variance": "Variance Explainer",
    "anomaly": "Anomaly Scan",
    "commentary": "Commentary Draft",
    "agent": "SQL Agent",
}


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _history(state: AgentState) -> list[dict]:
    turns = state.get("conversation_history") or []
    return [
        {"role": t["role"], "content": t["content"][:HISTORY_CHARS_PER_TURN]}
        for t in turns[-HISTORY_TURNS:]
        if t.get("role") in ("user", "assistant") and t.get("content")
    ]


def _emit(event: dict) -> None:
    get_stream_writer()(event)


# ─────────────────────────────────────────────────────────────────────────────
# Node 1 — understand (rewrite + classify + extract, one mini-model call)
# ─────────────────────────────────────────────────────────────────────────────

class QueryUnderstanding(BaseModel):
    standalone_question: str = Field(
        description="The user's latest message rewritten as a fully self-contained question, "
                    "carrying over department, period and topic from earlier turns when the "
                    "message refers to them. Repeat it unchanged if already self-contained.")
    intent: Literal["rag", "variance", "anomaly", "commentary", "agent"]
    department: Optional[str] = Field(description="Department named or implied, else null.")
    fiscal_year: Optional[str] = Field(description="4-digit fiscal year, e.g. '2025', else null.")
    quarter: Optional[Literal["Q1", "Q2", "Q3", "Q4"]] = Field(description="Quarter, else null.")
    threshold_pct: Optional[float] = Field(description="Variance % threshold if stated, else null.")


def _understand_prompt() -> str:
    lfy, lq = DataLoader.get().latest_actuals_period()
    return f"""You route questions for FinSight, an FP&A assistant for Crestwood Capital Group.
Today is {date.today():%d %B %Y}. Planning data (budget, actuals, forecast by department and
quarter) covers FY2022–FY2026; fiscal years follow calendar years. The latest quarter with
actuals booked is {lq} FY{lfy} — use it for "this quarter", "latest" or "most recent".

Departments: {", ".join(DEPARTMENTS)}.
Use the exact department name when the user clearly means one; otherwise copy their wording.

Intents:
- rag: answered from documents alone, with no specific department figures needed —
  expense/procurement policies, approval limits, per diems, variance-threshold rules,
  vendor profiles, qualitative memo or outlook commentary.
  e.g. "What is the APAC per diem?", "Who approves a $30K software purchase?"
- variance: WHY one specific department was over/under budget in a period.
  e.g. "Why did Legal overspend in Q1 FY2025?"
- anomaly: WHICH departments breached a variance threshold / are over or under budget in a period.
  e.g. "Which departments were >15% over budget in Q2 FY2024?"
- commentary: draft CFO, board or management commentary/narrative for a period.
- agent: everything else that needs numbers — actuals, budget, forecast, YTD, vendor spend,
  comparisons across periods or departments. ALWAYS agent when a question needs a
  department's actual figures AND a policy or rule to judge them, even if it mentions a policy.
  e.g. "Top 5 vendors in Q1 FY2025", "Is IT's Q2 FY2025 variance above the CFO-briefing
  threshold?", "Did Software Engineering's spend with its top vendor need VP approval?"
  Also greetings and anything off-topic.

Rewrite follow-ups ("what about Q3?", "and for EMEA?") into standalone questions using the
conversation so far."""


def understand(state: AgentState) -> dict:
    messages = [SystemMessage(content=_understand_prompt()), *_history(state),
                HumanMessage(content=state["query"])]
    try:
        u = mini_llm().with_structured_output(QueryUnderstanding).invoke(messages)
    except Exception as e:
        logger.warning(f"understand failed, falling back to agent: {e}")
        u = QueryUnderstanding(standalone_question=state["query"], intent="agent", department=None,
                               fiscal_year=None, quarter=None, threshold_pct=None)

    label = MODE_LABELS[u.intent]
    _emit({"type": "mode", "mode": label, "intent": u.intent})
    return {
        "standalone_query": u.standalone_question,
        "intent": u.intent,
        "mode_label": label,
        "department": u.department,
        "fiscal_year": norm_fy(u.fiscal_year) if u.fiscal_year else None,
        "quarter": u.quarter,
        "threshold_pct": u.threshold_pct,
    }


def route(state: AgentState) -> str:
    return state.get("intent") if state.get("intent") in MODE_LABELS else "agent"


# ─────────────────────────────────────────────────────────────────────────────
# Node 2a — RAG (LangChain retriever over Azure AI Search hybrid search)
# ─────────────────────────────────────────────────────────────────────────────

NO_DOCS_MESSAGE = ("I couldn't find anything relevant in the policy and memo documents. "
                   "Try naming a specific policy, department or quarter.")


def make_rag_node(retriever: FinSightRetriever, rag_pipeline):
    def rag(state: AgentState) -> dict:
        question = state.get("standalone_query") or state["query"]
        try:
            docs = retriever.invoke(question)
        except Exception as e:
            logger.error(f"retrieval failed: {e}")
            return {"error": "Document search is unavailable right now — please try again."}
        if not docs:
            _emit({"type": "token", "content": NO_DOCS_MESSAGE})
            return {"narrative": NO_DOCS_MESSAGE, "sources": []}

        sources = [d.metadata["source"] for d in docs]
        _emit({"type": "sources", "sources": sources})
        chunks = [{"title": d.metadata["title"], "content": d.page_content} for d in docs]
        messages = rag_pipeline.build_messages(question, chunks, _history(state))
        resp = final_llm(0.3).invoke(messages)
        return {"narrative": resp.content.strip(), "sources": sources,
                "tool_results": {"documents": chunks}}
    return rag


# ─────────────────────────────────────────────────────────────────────────────
# Node 2b — Agent (mini model picks tools, full model writes the answer)
# ─────────────────────────────────────────────────────────────────────────────

AGENT_SYSTEM = """You are FinSight's research agent for Crestwood Capital Group FP&A.
Gather the facts needed to answer the user's question using the tools:
- Data tools: planning data FY2022–FY2026 by department and quarter (budget, actuals,
  forecast, variances) and GL vendor spend. Use exact department names from the tool schema.
- search_documents: policies (approval limits, per diems, variance thresholds, procurement
  rules), quarterly memos, budget outlooks, vendor profiles.
Call as many tools as needed — for questions that combine numbers and policy, call both.
Never estimate numbers yourself. If a required department or period is genuinely missing
and cannot be inferred, say exactly what is needed. Finish with a brief factual summary."""

SYNTH_SYSTEM = """You are FinSight AI, an FP&A assistant for Crestwood Capital Group.
Answer the question using ONLY the evidence provided.
- Cite figures exactly as they appear in the evidence; do not recompute or round differently.
- When evidence comes from a document, name it (e.g. "Per the Travel & Entertainment Policy...").
- If the evidence notes a quarter has no actuals yet, say so plainly.
- If the evidence is insufficient, say what is missing instead of guessing.
- No characterisations, causes or conclusions the evidence doesn't support.
- Be concise: 2–5 sentences or a short list. The user already sees any data tables."""


def _parse_tool_output(msg: ToolMessage):
    if msg.name == "search_documents":
        return msg.content
    try:
        return json.loads(msg.content)
    except (TypeError, ValueError):
        return msg.content


def make_agent_node(agent):
    def agent_node(state: AgentState) -> dict:
        question = state.get("standalone_query") or state["query"]
        try:
            result = agent.invoke(
                {"messages": [*_history(state), {"role": "user", "content": question}]},
                {"recursion_limit": AGENT_RECURSION_LIMIT},
            )
        except GraphRecursionError:
            return {"error": "That question needed more steps than allowed. "
                             "Try splitting it into smaller questions."}

        messages = result["messages"]
        calls = {tc["id"]: tc for m in messages if isinstance(m, AIMessage) for tc in m.tool_calls}
        tool_msgs = [m for m in messages if isinstance(m, ToolMessage)]

        records, tables, sources, evidence = [], [], [], []
        for tm in tool_msgs:
            call = calls.get(tm.tool_call_id, {})
            output = _parse_tool_output(tm)
            records.append({"name": tm.name, "args": call.get("args", {}), "output": output})
            if tm.name == "search_documents":
                sources.extend(tm.artifact or [])
            elif isinstance(output, dict):
                if table := tool_result_table(output):
                    tables.append(table)
                    _emit({"type": "table", "content": table})
            evidence.append(
                f"### {tm.name}({json.dumps(call.get('args', {}))})\n{str(tm.content)[:TOOL_OUTPUT_CHARS]}"
            )

        used = {r["name"] for r in records}
        has_docs = "search_documents" in used
        has_data = bool(used - {"search_documents"})
        label = ("Hybrid Agent (SQL + RAG)" if has_docs and has_data
                 else "SQL Agent" if has_data
                 else "Agent (RAG)" if has_docs else "Agent")
        _emit({"type": "mode", "mode": label, "intent": "agent"})
        if sources:
            _emit({"type": "sources", "sources": sources})

        update = {
            "mode_label": label,
            "formatted_table": "\n\n".join(tables) or None,
            "sources": sources,
            "sql_query": "; ".join(f"{r['name']}({json.dumps(r['args'])})" for r in records) or None,
            "tool_results": {"calls": records},
        }

        if not records:
            reply = messages[-1].content if messages else ""
            _emit({"type": "token", "content": reply})
            return {**update, "narrative": reply}

        resp = final_llm(0.2).invoke([
            SystemMessage(content=SYNTH_SYSTEM),
            HumanMessage(content=f"Question: {question}\n\nEvidence:\n\n" + "\n\n".join(evidence)),
        ])
        return {**update, "narrative": resp.content.strip()}
    return agent_node


# ─────────────────────────────────────────────────────────────────────────────
# Graph assembly
# ─────────────────────────────────────────────────────────────────────────────

def build_agent_graph(rag_pipeline):
    retriever = FinSightRetriever(pipeline=rag_pipeline)
    agent = create_agent(
        mini_llm(),
        [*DATA_TOOLS, make_search_tool(retriever)],
        system_prompt=AGENT_SYSTEM,
        name="finsight_agent",
    )

    graph = StateGraph(AgentState)
    graph.add_node("understand", understand)
    graph.add_node("rag", make_rag_node(retriever, rag_pipeline))
    graph.add_node("agent", make_agent_node(agent))
    graph.add_node("variance", build_variance_graph())
    graph.add_node("anomaly", build_anomaly_graph())
    graph.add_node("commentary", build_commentary_graph())

    graph.set_entry_point("understand")
    graph.add_conditional_edges("understand", route, {k: k for k in MODE_LABELS})
    for node in MODE_LABELS:
        graph.add_edge(node, END)
    return graph.compile(name="finsight")
