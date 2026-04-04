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
You have access to Crestwood Capital Group's internal financial database covering FY2023–FY2026, including:
- General Ledger (GL) transactions across all departments, cost centers, and legal entities
- Budget vs Actuals planning data with variance analysis
- Rolling forecasts by GL account, responsibility center, and fiscal period

Guidelines:
- Always ground your answers in the retrieved context. Do not make up numbers.
- When referencing figures, always mention the department, fiscal period, and GL account where relevant.
- The internal database covers operating expenses (People Costs, Technology, Facilities, etc.) and planning data. It does NOT contain revenue accounts, so metrics like Net Operating Income, Net Profit, or EBITDA cannot be calculated. If asked, clearly explain this limitation — do NOT ask the user to provide data, as you already have access to the full internal database.
- If the retrieved context doesn't contain enough detail, say: "The current index doesn't contain sufficient data to answer this accurately. Try asking about a specific department, GL account, cost center, or fiscal period."
- Always be concise and professional. Format numbers with commas and dollar signs.
- At the end of your answer, mention which data sources (GL transactions / planning data) you used.
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
