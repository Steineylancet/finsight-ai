"""
FinSight AI — LangGraph Router (Main Agent Graph)
Classifies each user query and dispatches to the right execution path:

  rag        → Existing RAG pipeline (document Q&A)
  sql        → LangChain tool calling with pandas DataFrames
  variance   → variance_explainer subgraph
  anomaly    → anomaly_scanner subgraph
  commentary → commentary_drafter subgraph
"""

import json
import logging
import os
from langgraph.graph import StateGraph, END
from langchain_openai import AzureChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage

from backend.graph.state import AgentState
from backend.graph.tools import ALL_TOOLS
from backend.graph.flows.variance import build_variance_graph
from backend.graph.flows.anomaly import build_anomaly_graph
from backend.graph.flows.commentary import build_commentary_graph

logger = logging.getLogger(__name__)

# Pre-compile subgraphs once at import time
_variance_graph = build_variance_graph()
_anomaly_graph = build_anomaly_graph()
_commentary_graph = build_commentary_graph()


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
# Node 1 — Classify intent
# ─────────────────────────────────────────────────────────────────────────────

CLASSIFY_SYSTEM = """You classify financial queries into exactly one of these intents:

- rag         Questions about policies, procedures, per diem rates, vendor profiles,
               procurement rules, approval limits, or anything in policy documents.
               Examples: "What is the T&E per diem?", "Who approves a $50K purchase?"

- variance    Questions asking WHY a specific department deviated from budget.
               Requires a department + period.
               Examples: "Why did Software Engineering overspend in Q3?",
               "What drove the IT variance in FY2025 Q2?"

- anomaly     Questions asking to flag or identify which departments are over/under budget.
               Scans across multiple departments.
               Examples: "Which departments are over budget?",
               "Flag cost centres exceeding 10% variance in Q1 FY2026."

- commentary  Requests to draft CFO commentary, board narrative, or monthly review text.
               Examples: "Draft the Q3 FY2025 CFO commentary.",
               "Write the board deck variance summary for Q2."

- sql         All other data questions requiring live numbers:
               actuals lookups, budget figures, YTD summaries, vendor spend,
               headcount data, cross-period comparisons.
               Examples: "What were total actuals for Finance in FY2025?",
               "Show me the top 5 vendors by spend in Q2."

Return JSON: {"intent": "<one of the above>", "department": "<or null>",
              "fiscal_year": "<4-digit or null>", "quarter": "<Q1-Q4 or null>"}"""


def classify_intent(state: AgentState) -> AgentState:
    prompt = f"Query: {state['query']}"
    try:
        response = _mini_llm().invoke([
            SystemMessage(content=CLASSIFY_SYSTEM),
            HumanMessage(content=prompt),
        ])
        content = response.content.strip().strip("```json").strip("```")
        parsed = json.loads(content)
        intent = parsed.get("intent", "sql")
        mode_labels = {
            "rag": "RAG",
            "variance": "Variance Flow",
            "anomaly": "Anomaly Scan",
            "commentary": "Commentary Draft",
            "sql": "SQL Query",
        }
        return {
            **state,
            "intent": intent,
            "mode_label": mode_labels.get(intent, "SQL Query"),
            "department": parsed.get("department"),
            "fiscal_year": parsed.get("fiscal_year"),
            "quarter": parsed.get("quarter"),
        }
    except Exception as e:
        logger.warning(f"classify_intent failed: {e} — defaulting to sql")
        return {**state, "intent": "sql", "mode_label": "SQL Query"}


# ─────────────────────────────────────────────────────────────────────────────
# Node 2a — RAG node (delegates to existing RAG pipeline)
# ─────────────────────────────────────────────────────────────────────────────

def rag_node(state: AgentState) -> AgentState:
    """Lazy-imports RAG pipeline to avoid circular imports."""
    from backend.rag_pipeline import RAGPipeline
    try:
        pipeline = RAGPipeline()
        response, sources = pipeline.run(
            question=state["query"],
            conversation_history=state.get("conversation_history", []),
            stream=False,
        )
        narrative = response if isinstance(response, str) else str(response)
        return {
            **state,
            "narrative": narrative,
            "sources": [s.model_dump() for s in sources],
        }
    except Exception as e:
        logger.error(f"rag_node failed: {e}")
        return {**state, "error": f"RAG pipeline error: {e}"}


# ─────────────────────────────────────────────────────────────────────────────
# Node 2b — SQL agent node (LangChain tool calling)
# ─────────────────────────────────────────────────────────────────────────────

SQL_SYSTEM = """You are an FP&A data analyst with access to financial data tools.
Use the available tools to answer the user's question.
- Call the most appropriate tool with the correct parameters.
- After getting results, write a concise plain-English interpretation (2–3 sentences).
- Always cite specific figures from the tool output.
- If you are unsure of a parameter (e.g., department name), use the closest match
  from what the user mentioned. Do not guess fiscal year or quarter — ask if unclear.
"""


def sql_node(state: AgentState) -> AgentState:
    llm_with_tools = _mini_llm().bind_tools(ALL_TOOLS)

    messages = [
        SystemMessage(content=SQL_SYSTEM),
        HumanMessage(content=state["query"]),
    ]

    try:
        # Step 1: LLM selects a tool and generates call
        ai_response = llm_with_tools.invoke(messages)

        if not ai_response.tool_calls:
            # Model chose not to call a tool — treat response as direct answer
            return {**state, "narrative": ai_response.content, "formatted_table": None}

        # Step 2: Execute the selected tool
        tool_call = ai_response.tool_calls[0]
        tool_name = tool_call["name"]
        tool_args = tool_call["args"]

        tool_map = {t.name: t for t in ALL_TOOLS}
        if tool_name not in tool_map:
            return {**state, "error": f"Tool '{tool_name}' not found."}

        tool_result = tool_map[tool_name].invoke(tool_args)

        if "error" in tool_result:
            return {**state, "error": tool_result["error"]}

        # Step 3: Build markdown table from tool result rows
        formatted_table = _format_tool_result(tool_result)

        # Step 4: Full LLM interprets the result
        interp_prompt = (
            f"The user asked: {state['query']}\n\n"
            f"Tool used: {tool_name}\n"
            f"Result:\n{json.dumps(tool_result, indent=2, default=str)}\n\n"
            f"Write 2–3 sentences interpreting this data for an FP&A analyst. "
            f"Be specific — cite figures. No filler."
        )
        interp = _full_llm().invoke([
            SystemMessage(content="You are a concise FP&A analyst."),
            HumanMessage(content=interp_prompt),
        ])

        return {
            **state,
            "formatted_table": formatted_table,
            "narrative": interp.content.strip(),
            "sql_query": f"{tool_name}({json.dumps(tool_args, default=str)})",
        }

    except Exception as e:
        logger.error(f"sql_node failed: {e}")
        return {**state, "error": f"Data query failed: {e}"}


def _format_tool_result(result: dict) -> str | None:
    """Convert tool output dict to a markdown table string."""
    rows_key = next((k for k in ["rows", "drivers", "anomalies", "vendors"] if k in result), None)
    if not rows_key:
        return None

    rows = result[rows_key]
    if not rows:
        return None

    headers = list(rows[0].keys())
    header_row = "| " + " | ".join(str(h) for h in headers) + " |"
    sep_row = "|" + "|".join("---" for _ in headers) + "|"

    data_rows = []
    for row in rows:
        cells = []
        for v in row.values():
            if isinstance(v, float):
                cells.append(f"{v:,.1f}")
            elif isinstance(v, int):
                cells.append(f"{v:,}")
            else:
                cells.append(str(v))
        data_rows.append("| " + " | ".join(cells) + " |")

    return "\n".join([header_row, sep_row] + data_rows)


# ─────────────────────────────────────────────────────────────────────────────
# Nodes 2c / 2d / 2e — Delegate to compiled subgraphs
# ─────────────────────────────────────────────────────────────────────────────

def variance_node(state: AgentState) -> AgentState:
    return _variance_graph.invoke(state)


def anomaly_node(state: AgentState) -> AgentState:
    return _anomaly_graph.invoke(state)


def commentary_node(state: AgentState) -> AgentState:
    return _commentary_graph.invoke(state)


# ─────────────────────────────────────────────────────────────────────────────
# Routing function — reads intent from state and returns next node name
# ─────────────────────────────────────────────────────────────────────────────

def route(state: AgentState) -> str:
    return {
        "rag": "rag_node",
        "sql": "sql_node",
        "variance": "variance_node",
        "anomaly": "anomaly_node",
        "commentary": "commentary_node",
    }.get(state.get("intent", "sql"), "sql_node")


# ─────────────────────────────────────────────────────────────────────────────
# Graph assembly
# ─────────────────────────────────────────────────────────────────────────────

def build_agent_graph():
    graph = StateGraph(AgentState)

    graph.add_node("classify_intent", classify_intent)
    graph.add_node("rag_node", rag_node)
    graph.add_node("sql_node", sql_node)
    graph.add_node("variance_node", variance_node)
    graph.add_node("anomaly_node", anomaly_node)
    graph.add_node("commentary_node", commentary_node)

    graph.set_entry_point("classify_intent")

    graph.add_conditional_edges(
        "classify_intent",
        route,
        {
            "rag_node": "rag_node",
            "sql_node": "sql_node",
            "variance_node": "variance_node",
            "anomaly_node": "anomaly_node",
            "commentary_node": "commentary_node",
        },
    )

    for node in ["rag_node", "sql_node", "variance_node", "anomaly_node", "commentary_node"]:
        graph.add_edge(node, END)

    return graph.compile()
