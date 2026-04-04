"""
FinSight AI — Streamlit Frontend
Chat interface for the financial RAG chatbot
"""

import streamlit as st
import requests
import json

# ── Config ────────────────────────────────────────────────────────────────────
BACKEND_URL = "http://localhost:8000"

st.set_page_config(
    page_title="FinSight AI",
    page_icon="🏦",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Custom CSS ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
    /* ── Global ── */
    html, body, [data-testid="stAppViewContainer"] {
        background-color: #0d1117;
        color: #e6edf3;
    }

    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #161b22 0%, #0d1117 100%);
        border-right: 1px solid #21262d;
    }

    /* ── Header ── */
    .hero {
        background: linear-gradient(135deg, #1a237e 0%, #0d47a1 50%, #006064 100%);
        border-radius: 16px;
        padding: 28px 32px;
        margin-bottom: 24px;
        box-shadow: 0 4px 24px rgba(0,0,0,0.4);
    }
    .hero h1 {
        font-size: 2.2rem;
        font-weight: 800;
        color: #ffffff;
        margin: 0 0 6px 0;
        letter-spacing: -0.5px;
    }
    .hero p {
        font-size: 1rem;
        color: rgba(255,255,255,0.75);
        margin: 0;
    }

    /* ── Stats bar ── */
    .stats-bar {
        display: flex;
        gap: 12px;
        margin-bottom: 20px;
        flex-wrap: wrap;
    }
    .stat-chip {
        background: #161b22;
        border: 1px solid #30363d;
        border-radius: 20px;
        padding: 6px 14px;
        font-size: 0.78rem;
        color: #8b949e;
        display: flex;
        align-items: center;
        gap: 6px;
    }
    .stat-chip span { color: #58a6ff; font-weight: 600; }

    /* ── Source cards ── */
    .source-card {
        background: #161b22;
        border: 1px solid #21262d;
        border-left: 3px solid #58a6ff;
        border-radius: 8px;
        padding: 10px 14px;
        margin: 6px 0;
        font-size: 0.82rem;
        color: #c9d1d9;
    }
    .source-card strong { color: #e6edf3; }
    .source-card em { color: #8b949e; font-size: 0.78rem; }

    .badge {
        border-radius: 4px;
        padding: 2px 8px;
        font-size: 0.72rem;
        font-weight: 600;
        margin-right: 6px;
    }
    .badge-gl {
        background: rgba(88,166,255,0.15);
        color: #58a6ff;
        border: 1px solid rgba(88,166,255,0.3);
    }
    .badge-plan {
        background: rgba(63,185,80,0.15);
        color: #3fb950;
        border: 1px solid rgba(63,185,80,0.3);
    }

    /* ── Chat messages ── */
    [data-testid="stChatMessage"] {
        background: #161b22 !important;
        border: 1px solid #21262d !important;
        border-radius: 12px !important;
        margin-bottom: 8px !important;
    }

    /* ── Sidebar elements ── */
    .sidebar-title {
        font-size: 1.3rem;
        font-weight: 700;
        color: #e6edf3;
        margin-bottom: 4px;
    }
    .sidebar-sub {
        font-size: 0.8rem;
        color: #58a6ff;
        margin-bottom: 16px;
    }

    /* ── Input box ── */
    [data-testid="stChatInput"] {
        border-radius: 12px !important;
        border: 1px solid #30363d !important;
        background: #161b22 !important;
    }

    /* ── Expander ── */
    [data-testid="stExpander"] {
        background: #161b22 !important;
        border: 1px solid #21262d !important;
        border-radius: 8px !important;
    }

    /* ── Buttons ── */
    .stButton > button {
        background: #21262d !important;
        color: #c9d1d9 !important;
        border: 1px solid #30363d !important;
        border-radius: 8px !important;
        font-size: 0.82rem !important;
        text-align: left !important;
    }
    .stButton > button:hover {
        background: #30363d !important;
        border-color: #58a6ff !important;
        color: #e6edf3 !important;
    }
</style>
""", unsafe_allow_html=True)

# ── Session State ─────────────────────────────────────────────────────────────
if "messages" not in st.session_state:
    st.session_state.messages = []
if "sources_history" not in st.session_state:
    st.session_state.sources_history = {}

# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown('<div class="sidebar-title">🏦 FinSight AI</div>', unsafe_allow_html=True)
    st.markdown('<div class="sidebar-sub">Financial Intelligence Assistant</div>', unsafe_allow_html=True)
    st.divider()

    st.markdown("**About**")
    st.markdown(
        "<span style='color:#8b949e;font-size:0.85rem;'>"
        "Answers questions over <strong style='color:#c9d1d9;'>Crestwood Capital Group's</strong> "
        "financial data — GL transactions, budget vs actuals, forecasts — "
        "powered by <strong style='color:#58a6ff;'>Azure OpenAI GPT-4o</strong> "
        "and <strong style='color:#58a6ff;'>Azure AI Search</strong>."
        "</span>",
        unsafe_allow_html=True
    )

    st.divider()
    st.markdown("**💡 Try these questions**")

    example_questions = [
        "What was the budget vs actuals variance for Finance in Q1 2023?",
        "Which departments had the highest People Costs overspend?",
        "Show me large vendor invoices from January 2023.",
        "Which cost centers are at risk of going over budget?",
        "What is the total forecast for Salaries and Wages in Q2 2023?",
        "Compare budget vs actuals for GL account 6100.",
    ]

    for q in example_questions:
        if st.button(q, key=q, use_container_width=True):
            st.session_state["prefill_question"] = q

    st.divider()
    if st.button("🗑️ Clear Chat", use_container_width=True):
        st.session_state.messages = []
        st.session_state.sources_history = {}
        st.rerun()

    st.markdown(
        "<div style='font-size:0.72rem;color:#484f58;margin-top:8px;'>"
        "Azure OpenAI · Azure AI Search · FastAPI · Streamlit"
        "</div>",
        unsafe_allow_html=True
    )


# ── Hero Header ───────────────────────────────────────────────────────────────
st.markdown("""
<div class="hero">
    <h1>🏦 FinSight AI</h1>
    <p>Ask anything about Crestwood Capital Group's financial data — GL transactions, budgets, forecasts, and variance analysis</p>
</div>
""", unsafe_allow_html=True)

# ── Stats Bar ─────────────────────────────────────────────────────────────────
st.markdown("""
<div class="stats-bar">
    <div class="stat-chip">📄 GL Transactions <span>5,000</span> indexed</div>
    <div class="stat-chip">📊 Planning Records <span>4,500</span> indexed</div>
    <div class="stat-chip">📅 Coverage <span>FY2023 – FY2026</span></div>
    <div class="stat-chip">🔍 Search <span>Hybrid Vector + BM25</span></div>
    <div class="stat-chip">🤖 Model <span>GPT-4o</span></div>
</div>
""", unsafe_allow_html=True)

# ── Chat History ──────────────────────────────────────────────────────────────
for i, msg in enumerate(st.session_state.messages):
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

        if msg["role"] == "assistant" and i in st.session_state.sources_history:
            sources = st.session_state.sources_history[i]
            if sources:
                with st.expander(f"📎 {len(sources)} Source(s) Retrieved"):
                    for src in sources:
                        badge_class = "badge-gl" if src["data_type"] == "gl_transaction" else "badge-plan"
                        badge_label = "GL Transaction" if src["data_type"] == "gl_transaction" else "Planning Data"
                        st.markdown(
                            f'<div class="source-card">'
                            f'<span class="badge {badge_class}">{badge_label}</span>'
                            f'<strong>{src["title"]}</strong><br>'
                            f'<span style="color:#8b949e;">Entity: {src["entity"]} &nbsp;|&nbsp; '
                            f'Dept: {src["department"]} &nbsp;|&nbsp; '
                            f'FY{src["fiscal_year"]} P{src["fiscal_period"]}</span><br>'
                            f'<em>{src["preview"]}...</em>'
                            f'</div>',
                            unsafe_allow_html=True
                        )

# ── Chat Input ────────────────────────────────────────────────────────────────
prefill = st.session_state.pop("prefill_question", "")
question = st.chat_input("Ask a financial question about Crestwood Capital Group...") or prefill

if question:
    st.session_state.messages.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.markdown(question)

    with st.chat_message("assistant"):
        message_placeholder = st.empty()
        full_response = ""
        retrieved_sources = []

        try:
            history_payload = [
                {"role": m["role"], "content": m["content"]}
                for m in st.session_state.messages[:-1][-10:]
            ]

            with requests.post(
                f"{BACKEND_URL}/chat",
                json={"question": question, "conversation_history": history_payload},
                stream=True,
                timeout=60,
            ) as resp:
                resp.raise_for_status()
                for line in resp.iter_lines():
                    if line:
                        line = line.decode("utf-8")
                        if line.startswith("data: "):
                            data_str = line[6:]
                            if data_str == "[DONE]":
                                break
                            try:
                                data = json.loads(data_str)
                                if data["type"] == "sources":
                                    retrieved_sources = data["sources"]
                                elif data["type"] == "token":
                                    full_response += data["content"]
                                    message_placeholder.markdown(full_response + "▌")
                            except json.JSONDecodeError:
                                pass

            message_placeholder.markdown(full_response)

        except requests.exceptions.ConnectionError:
            full_response = "⚠️ Cannot connect to the backend. Make sure the FastAPI server is running: `uvicorn backend.main:app --reload`"
            message_placeholder.error(full_response)

        except Exception as e:
            full_response = f"⚠️ Error: {str(e)}"
            message_placeholder.error(full_response)

        if retrieved_sources:
            with st.expander(f"📎 {len(retrieved_sources)} Source(s) Retrieved"):
                for src in retrieved_sources:
                    badge_class = "badge-gl" if src["data_type"] == "gl_transaction" else "badge-plan"
                    badge_label = "GL Transaction" if src["data_type"] == "gl_transaction" else "Planning Data"
                    st.markdown(
                        f'<div class="source-card">'
                        f'<span class="badge {badge_class}">{badge_label}</span>'
                        f'<strong>{src["title"]}</strong><br>'
                        f'<span style="color:#8b949e;">Entity: {src["entity"]} &nbsp;|&nbsp; '
                        f'Dept: {src["department"]} &nbsp;|&nbsp; '
                        f'FY{src["fiscal_year"]} P{src["fiscal_period"]}</span><br>'
                        f'<em>{src["preview"]}...</em>'
                        f'</div>',
                        unsafe_allow_html=True
                    )

    msg_index = len(st.session_state.messages)
    st.session_state.messages.append({"role": "assistant", "content": full_response})
    st.session_state.sources_history[msg_index] = retrieved_sources
