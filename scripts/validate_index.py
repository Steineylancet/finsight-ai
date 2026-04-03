"""
FinSight AI — Index Validation Script
Run this after ingest.py to confirm the search index is working correctly.

    python scripts/validate_index.py
"""

import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from backend.azure_openai_client import AzureOpenAIClient
from backend.azure_search import AzureSearchClient
from dotenv import load_dotenv

load_dotenv()

TEST_QUERIES = [
    "What was the budget vs actuals variance for Finance in Q1 2022?",
    "Which departments had People Costs overspend?",
    "Show vendor invoices from Workday Inc",
    "What is the forecast for Salaries and Wages?",
    "Which cost centers are at risk of going over budget?",
]


def main():
    print("\n" + "=" * 60)
    print("FinSight AI — Index Validation")
    print("=" * 60)

    openai_client = AzureOpenAIClient()
    search_client = AzureSearchClient()

    for query in TEST_QUERIES:
        print(f"\n🔍 Query: {query}")
        print("-" * 50)

        query_vector = openai_client.get_embedding(query)
        results = search_client.hybrid_search(
            query_text=query,
            query_vector=query_vector,
            top_k=3
        )

        if not results:
            print("  ❌ No results returned!")
        else:
            for i, r in enumerate(results, 1):
                print(f"  [{i}] {r['title']}")
                print(f"      Type: {r['data_type']} | Dept: {r['department']} | FY: {r['fiscal_year']}")
                print(f"      Score: {r['score']:.4f}")
                print(f"      Preview: {r['content'][:150]}...")
                print()

    print("=" * 60)
    print("Validation complete! If results look relevant, you're ready to run the app.")
    print("=" * 60)


if __name__ == "__main__":
    main()
