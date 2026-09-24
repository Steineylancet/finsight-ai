"""
FinSight AI — Layer 1 (document RAG) as LangChain components.

FinSightRetriever wraps the existing Azure AI Search hybrid query (vector +
BM25 + semantic reranking) as a standard LangChain BaseRetriever, so the RAG
path and the agent share one retrieval implementation. LangChain's stock Azure
Search vector store isn't used because it doesn't expose this exact
hybrid + semantic-reranker configuration.
"""

from typing import Any

from langchain_core.callbacks import CallbackManagerForRetrieverRun
from langchain_core.documents import Document
from langchain_core.retrievers import BaseRetriever
from langchain_core.tools import BaseTool, tool
from pydantic import Field

MAX_CHARS_PER_CHUNK = 2500


class FinSightRetriever(BaseRetriever):
    pipeline: Any = Field(description="RAGPipeline exposing retrieve(question, top_k)")
    top_k: int = 5

    def _get_relevant_documents(
        self, query: str, *, run_manager: CallbackManagerForRetrieverRun
    ) -> list[Document]:
        chunks, sources = self.pipeline.retrieve(query, top_k=self.top_k)
        return [
            Document(
                page_content=chunk["content"],
                metadata={"title": chunk["title"], "source": source.model_dump()},
            )
            for chunk, source in zip(chunks, sources)
        ]


def format_documents(docs: list[Document]) -> str:
    return "\n\n".join(
        f"[Source {i}: {d.metadata['title']}]\n{d.page_content[:MAX_CHARS_PER_CHUNK]}"
        for i, d in enumerate(docs, 1)
    )


def make_search_tool(retriever: FinSightRetriever) -> BaseTool:
    @tool(response_format="content_and_artifact")
    def search_documents(query: str) -> tuple[str, list[dict]]:
        """
        Search Crestwood Capital Group's internal documents: expense and
        procurement policies (approval limits, per diems, variance thresholds),
        quarterly department memos, budget outlooks and vendor profiles.
        Use for any policy, procedure, or qualitative question.
        """
        docs = retriever.invoke(query)
        if not docs:
            return "No relevant documents found.", []
        return format_documents(docs), [d.metadata["source"] for d in docs]

    return search_documents
