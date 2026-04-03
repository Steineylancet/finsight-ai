"""
Azure AI Search Client Wrapper
Handles index creation, document upload, and hybrid search for FinSight AI
"""

import os
import logging
from azure.search.documents import SearchClient
from azure.search.documents.indexes import SearchIndexClient
from azure.search.documents.indexes.models import (
    SearchIndex,
    SearchField,
    SearchFieldDataType,
    SimpleField,
    SearchableField,
    VectorSearch,
    HnswAlgorithmConfiguration,
    VectorSearchProfile,
    SemanticConfiguration,
    SemanticSearch,
    SemanticPrioritizedFields,
    SemanticField,
)
from azure.search.documents.models import VectorizedQuery
from azure.core.credentials import AzureKeyCredential
from dotenv import load_dotenv

load_dotenv()
logger = logging.getLogger(__name__)


class AzureSearchClient:
    def __init__(self):
        self.endpoint = os.getenv("AZURE_SEARCH_ENDPOINT")
        self.api_key = os.getenv("AZURE_SEARCH_API_KEY")
        self.index_name = os.getenv("AZURE_SEARCH_INDEX_NAME", "finsight-index")
        self.credential = AzureKeyCredential(self.api_key)

        self.index_client = SearchIndexClient(
            endpoint=self.endpoint,
            credential=self.credential
        )
        self.search_client = SearchClient(
            endpoint=self.endpoint,
            index_name=self.index_name,
            credential=self.credential
        )

    def create_index(self):
        """Create the Azure AI Search index with vector + keyword search support."""
        fields = [
            SimpleField(name="id", type=SearchFieldDataType.String, key=True, filterable=True),
            SearchableField(name="content", type=SearchFieldDataType.String),
            SearchableField(name="title", type=SearchFieldDataType.String),
            SimpleField(name="data_type", type=SearchFieldDataType.String, filterable=True, facetable=True),
            SimpleField(name="entity", type=SearchFieldDataType.String, filterable=True, facetable=True),
            SimpleField(name="department", type=SearchFieldDataType.String, filterable=True, facetable=True),
            SimpleField(name="fiscal_year", type=SearchFieldDataType.String, filterable=True, facetable=True),
            SimpleField(name="fiscal_period", type=SearchFieldDataType.String, filterable=True),
            SimpleField(name="expense_category", type=SearchFieldDataType.String, filterable=True, facetable=True),
            SimpleField(name="chunk_index", type=SearchFieldDataType.Int32),
            SearchField(
                name="embedding",
                type=SearchFieldDataType.Collection(SearchFieldDataType.Single),
                searchable=True,
                vector_search_dimensions=1536,
                vector_search_profile_name="finsight-vector-profile",
            ),
        ]

        vector_search = VectorSearch(
            algorithms=[HnswAlgorithmConfiguration(name="finsight-hnsw")],
            profiles=[VectorSearchProfile(
                name="finsight-vector-profile",
                algorithm_configuration_name="finsight-hnsw"
            )],
        )

        semantic_config = SemanticConfiguration(
            name="finsight-semantic-config",
            prioritized_fields=SemanticPrioritizedFields(
                content_fields=[SemanticField(field_name="content")],
                title_field=SemanticField(field_name="title"),
                keywords_fields=[
                    SemanticField(field_name="department"),
                    SemanticField(field_name="expense_category"),
                ]
            )
        )

        index = SearchIndex(
            name=self.index_name,
            fields=fields,
            vector_search=vector_search,
            semantic_search=SemanticSearch(configurations=[semantic_config]),
        )

        try:
            self.index_client.create_or_update_index(index)
            logger.info(f"Index '{self.index_name}' created/updated successfully.")
        except Exception as e:
            logger.error(f"Failed to create index: {e}")
            raise

    def upload_documents(self, documents: list[dict], batch_size: int = 100):
        """Upload documents (chunks + embeddings) to the index in batches."""
        total = len(documents)
        for i in range(0, total, batch_size):
            batch = documents[i: i + batch_size]
            try:
                result = self.search_client.upload_documents(documents=batch)
                succeeded = sum(1 for r in result if r.succeeded)
                logger.info(f"Batch {i // batch_size + 1}: uploaded {succeeded}/{len(batch)} documents.")
            except Exception as e:
                logger.error(f"Upload batch failed: {e}")
                raise

    def hybrid_search(
        self,
        query_text: str,
        query_vector: list[float],
        top_k: int = 5,
        filter_expr: str = None,
    ) -> list[dict]:
        """
        Hybrid search: vector similarity + BM25 keyword + semantic reranking.
        Returns top_k results with content and metadata.
        """
        vector_query = VectorizedQuery(
            vector=query_vector,
            k_nearest_neighbors=top_k,
            fields="embedding"
        )

        try:
            results = self.search_client.search(
                search_text=query_text,
                vector_queries=[vector_query],
                query_type="semantic",
                semantic_configuration_name="finsight-semantic-config",
                top=top_k,
                filter=filter_expr,
                select=["id", "content", "title", "data_type", "entity",
                        "department", "fiscal_year", "fiscal_period", "expense_category"],
            )

            chunks = []
            for r in results:
                chunks.append({
                    "id": r["id"],
                    "content": r["content"],
                    "title": r["title"],
                    "data_type": r.get("data_type", ""),
                    "entity": r.get("entity", ""),
                    "department": r.get("department", ""),
                    "fiscal_year": r.get("fiscal_year", ""),
                    "fiscal_period": r.get("fiscal_period", ""),
                    "expense_category": r.get("expense_category", ""),
                    "score": r.get("@search.score", 0),
                })
            return chunks

        except Exception as e:
            logger.error(f"Search failed: {e}")
            raise
