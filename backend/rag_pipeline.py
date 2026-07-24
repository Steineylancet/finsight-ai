"""
FinSight AI — Core RAG Pipeline
Orchestrates: embed → search → prompt → stream response
"""

import logging
from backend.azure_openai_client import AzureOpenAIClient
from backend.azure_search import AzureSearchClient
from backend.models import Source, ConversationTurn

logger = logging.getLogger(__name__)

SYSTEM_PROMPT = """You are FinSight AI, an expert financial analyst assistant for Crestwood Capital Group.
You answer questions using internal management documents covering FY2025–FY2026, including:

QUARTERLY DEPARTMENT MEMOS:
- Management commentary memos for 13 departments: Finance, Human Resources, Executive,
  Sales (Americas / EMEA / APAC), Marketing - Americas, Customer Success (Americas / EMEA),
  Software Engineering, IT Infrastructure, Finance Operations, Data & Analytics.
- Each memo covers budget vs actuals vs forecast by expense category, key variances,
  vendor spend, and outlook — for Q1–Q4 of FY2025 and FY2026.

EXPENSE & PROCUREMENT POLICIES:
- Travel & Entertainment Policy (per diem rates, approval thresholds)
- Software & Cloud Procurement Policy (approval limits, preferred vendors)
- Professional Services Engagement Policy (SOW requirements, rate benchmarks)
- Vendor Management Policy (preferred vendor list, RFP thresholds)
- People Costs & Headcount Policy (compensation bands, benefits load assumptions)
- Budget Planning & Forecasting Guidelines (variance thresholds, reforecast cadence)
- Expense Approval Authority Matrix (who approves what at which dollar amount)
- Marketing & Events Expense Policy (event budgets, sponsorship approvals)

VENDOR PROFILES:
- Internal profiles for the top 10 vendors by spend (FY2025–2026), covering services
  provided, total spend, top departments, and contract notes.

Guidelines:
- Always ground your answers in the retrieved documents. Do not invent numbers.
- Cite the source document naturally in your answer (e.g., "According to the Q2 FY2026
  Software Engineering review..." or "Per the Travel & Entertainment Policy...").
- For budget vs actuals questions, reference the specific department memo.
- For policy questions (approval limits, per diems, headcount), reference the policy doc.
- For vendor questions, reference the vendor profile.
- The data covers OPERATING EXPENSES and department-level performance only.
  It does NOT include Balance Sheet items, COGS, or financing/tax lines —
  full EBITDA or Net Income cannot be computed. If asked, explain this clearly.
- If the retrieved documents don't contain enough detail, say so clearly and suggest
  a more specific question (e.g., a specific department, quarter, or expense category).
- Be concise and professional. Format numbers with commas and dollar signs.
"""


class RAGPipeline:
    def __init__(self):
        self.openai_client = AzureOpenAIClient()
        self.search_client = AzureSearchClient()

    def retrieve(self, question: str, top_k: int = 5) -> tuple[list[dict], list[Source]]:
        """Embed the question and retrieve top_k relevant chunks from Azure AI Search."""
        query_vector = self.openai_client.get_embedding(question)
        chunks = self.search_client.hybrid_search(
            query_text=question,
            query_vector=query_vector,
            top_k=top_k,
        )

        sources = [
            Source(
                title=c["title"],
                data_type=c["data_type"],
                entity=c["entity"],
                department=c["department"],
                fiscal_year=c["fiscal_year"],
                fiscal_period=c["fiscal_period"],
                expense_category=c["expense_category"],
                preview=c["content"][:200],
            )
            for c in chunks
        ]

        return chunks, sources

    def build_messages(
        self,
        question: str,
        chunks: list[dict],
        conversation_history: list[ConversationTurn],
    ) -> list[dict]:
        """Build the full message list for GPT-4o."""

        # Format retrieved context
        context_parts = []
        for i, chunk in enumerate(chunks, 1):
            context_parts.append(
                f"[Source {i}: {chunk['title']}]\n{chunk['content']}"
            )
        context_block = "\n\n".join(context_parts)

        # Keep only last 5 turns of conversation history to control token usage
        recent_history = conversation_history[-5:] if len(conversation_history) > 5 else conversation_history

        messages = [{"role": "system", "content": SYSTEM_PROMPT}]

        # Add conversation history
        for turn in recent_history:
            messages.append({"role": turn.role, "content": turn.content})

        # Add context + current question
        user_message = (
            f"Use the following financial data to answer the question:\n\n"
            f"{context_block}\n\n"
            f"Question: {question}"
        )
        messages.append({"role": "user", "content": user_message})

        return messages

    def run(self, question: str, conversation_history: list[ConversationTurn] = None, stream: bool = True):
        """
        Full RAG pipeline.
        Returns (stream_or_response, sources)
        """
        if conversation_history is None:
            conversation_history = []

        logger.info(f"RAG query: {question[:80]}...")

        # Retrieve
        chunks, sources = self.retrieve(question)

        if not chunks:
            logger.warning("No chunks retrieved from search index.")
            fallback = "I couldn't find relevant financial data to answer that question. Try rephrasing or asking about a specific department, period, or GL account."
            return fallback, []

        # Build messages
        messages = self.build_messages(question, chunks, conversation_history)

        # Generate response
        response = self.openai_client.chat_completion(
            messages=messages,
            stream=stream,
            max_tokens=800,
            temperature=0.3,
        )

        return response, sources
