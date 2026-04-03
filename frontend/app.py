"""
FinSight AI — Streamlit Frontend
Chat interface for the financial RAG chatbot
"""

import streamlit as st
import requests
import json

# ── Config ────────────────────────────────────────────────────────────────────
BACKEND_URL = "http://localhost:8000"  # Change to Azure App Service URL after deployment

st.set_page_config(
    page_title="FinSight AI",
    page_icon="🏦",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Custom CSS ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
    .main-header { font-size: 2rem; font-weight: 700; color: #1a1a2e; }
    .sub-header  { font-size: 1rem; color: #555; margin-bottom: 1rem; }
    .source-card {
        background: #f0f4ff;
        border-left: 4px solid #4361ee;
        border-radius: 6px;
        padding: 10px 14px;
        margin: 6px 0;
        font-size: 0.85rem;
    }
    .badge {
        background: #4361ee;
        color: white;
        border-radius: 4px;
        padding: 2px 8px;
        font-size: 0.75rem;
        margin-right: 4px;
    }
    .stChatMessage { border-radius: 12px; }
</style>
""", unsafe_allow_html=True)

# ── Session State ─────────────────────────────────────────────────────────────
if "messages" not in st.session_state:
    st.session_state.messages = []
if "sources_history" not in st.session_state:
    st.session_state.sources_history = {}

# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 🏦 FinSight AI")
    st.markdown("*Financial Knowledge Assistant*")
    st.markdown("---")

    st.markdown("### About")
    st.markdown(
        "FinSight AI answers questions about **Nexgen Corporation's** "
        "financial data including GL transactions, budget vs actuals, "
        "and forecasts — powered by Azure OpenAI + Azure AI Search."
    )

    st.markdown("---")
    st.markdown("### 💡 Example Questions")

    example_questions = [
        "What was the budget vs actuals variance for Finance in Q1 2022?",
        "Which departments had the highest People Costs overspend?",
        "Show me large vendor invoices from January 2022.",
        "Which cost centers are at risk of going over budget?",
        "What is the total forecast for Salaries and Wages in Q2 2022?",
        "Compare budget vs actuals for GL account 6100.",
    ]

    for q in example_questions:
        if st.button(q, key=q, use_container_width=True):
            st.session_state["prefill_question"] = q

    st.markdown("---")
    if st.button("🗑️ Clear Chat", use_container_width=True):
        st.session_state.messages = []
        st.session_state.sources_history = {}
        st.rerun()

    st.markdown("---")
    st.caption("Built with Azure OpenAI · Azure AI Search · FastAPI · Streamlit")


# ── Main Area ─────────────────────────────────────────────────────────────────
st.markdown('<p class="main-header">🏦 FinSight AI</p>', unsafe_allow_html=True)
st.markdown('<p class="sub-header">Ask anything about Nexgen Corporation\'s financial data</p>', unsafe_allow_html=True)

# Display chat history
for i, msg in enumerate(st.session_state.messages):
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

        # Show sources under assistant messages
        if msg["role"] == "assistant" and i in st.session_state.sources_history:
            sources = st.session_state.sources_history[i]
            if sources:
                with st.expander(f"📎 {len(sources)} Source(s) Used"):
                    for src in sources:
                        badge_type = "GL Transaction" if src["data_type"] == "gl_transaction" else "Planning Data"
                        st.markdown(
                            f'<div class="source-card">'
                            f'<span class="badge">{badge_type}</span>'
                            f'<strong>{src["title"]}</strong><br>'
                            f'Entity: {src["entity"]} | Dept: {src["department"]} | '
                            f'FY: {src["fiscal_year"]} P{src["fiscal_period"]}<br>'
                            f'<em>{src["preview"]}...</em>'
                            f'</div>',
                            unsafe_allow_html=True
                        )


# ── Chat Input ────────────────────────────────────────────────────────────────
prefill = st.session_state.pop("prefill_question", "")
question = st.chat_input("Ask a financial question about Nexgen Corporation...") or prefill

if question:
    # Add user message
    st.session_state.messages.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.markdown(question)

    # Call backend and stream response
    with st.chat_message("assistant"):
        message_placeholder = st.empty()
        full_response = ""
        retrieved_sources = []

        try:
            history_payload = [
                {"role": m["role"], "content": m["content"]}
                for m in st.session_state.messages[:-1][-10:]  # Last 10 turns
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
            full_response = (
                "⚠️ Cannot connect to the backend. "
                "Make sure the FastAPI server is running: `uvicorn backend.main:app --reload`"
            )
            message_placeholder.error(full_response)

        except Exception as e:
            full_response = f"⚠️ Error: {str(e)}"
            message_placeholder.error(full_response)

        # Show sources
        if retrieved_sources:
            with st.expander(f"📎 {len(retrieved_sources)} Source(s) Used"):
                for src in retrieved_sources:
                    badge_type = "GL Transaction" if src["data_type"] == "gl_transaction" else "Planning Data"
                    st.markdown(
                        f'<div class="source-card">'
                        f'<span class="badge">{badge_type}</span>'
                        f'<strong>{src["title"]}</strong><br>'
                        f'Entity: {src["entity"]} | Dept: {src["department"]} | '
                        f'FY: {src["fiscal_year"]} P{src["fiscal_period"]}<br>'
                        f'<em>{src["preview"]}...</em>'
                        f'</div>',
                        unsafe_allow_html=True
                    )

    # Save to session state
    msg_index = len(st.session_state.messages)
    st.session_state.messages.append({"role": "assistant", "content": full_response})
    st.session_state.sources_history[msg_index] = retrieved_sources
