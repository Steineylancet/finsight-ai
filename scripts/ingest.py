"""
FinSight AI — Data Ingestion Pipeline
Converts Crestwood Capital Group financial CSVs into searchable RAG chunks
and indexes them into Azure AI Search.

Run this script ONCE after setting up Azure resources:
    python scripts/ingest.py

Steps:
  1. Load all CSV files
  2. Convert structured rows into human-readable text chunks
  3. Generate embeddings via Azure OpenAI
  4. Create Azure AI Search index
  5. Upload chunks + embeddings to the index
"""

import os
import sys
import uuid
import logging
import pandas as pd

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from backend.azure_openai_client import AzureOpenAIClient
from backend.azure_search import AzureSearchClient
from dotenv import load_dotenv

load_dotenv()
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

# ── Paths ────────────────────────────────────────────────────────────────────
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
GL_DATA_PATH = os.path.join(BASE_DIR, "gl_data", "fact_gl_transactions.csv")
PLANNING_PATH = os.path.join(BASE_DIR, "planning_data", "fact_planning_combined.csv")
MASTER_RC_PATH = os.path.join(BASE_DIR, "master_data", "dim_responsibility_center.csv")
MASTER_GL_PATH = os.path.join(BASE_DIR, "master_data", "dim_gl_account.csv")
MASTER_VENDOR_PATH = os.path.join(BASE_DIR, "master_data", "dim_vendor.csv")


# ── Text Conversion Functions ─────────────────────────────────────────────────

def gl_row_to_text(row) -> str:
    """Convert a GL transaction row into a human-readable sentence."""
    return (
        f"In {row['Month']}/{row['Fiscal_Year']} (Q{row['Quarter'][-1]}), "
        f"{row['Entity_Name']} ({row['Department']} dept, cost center {row['RC_Code']}) "
        f"posted a {row['Document_Type_Desc']} of ${row['Amount_USD']:,.2f} USD "
        f"for {row['Account_Name']} ({row['Expense_Category']}) "
        f"to GL account {row['GL_Account']}, "
        f"from vendor {row['Vendor_Name']}. "
        f"Description: {row['Description']}. "
        f"Status: {row['Status']}. Clearing: {row['Clearing_Status']}. "
        f"Posted by: {row['Posted_By']}. Reference: {row['Reference']}."
    )


def planning_group_to_text(group_df) -> str:
    """Convert a group of planning rows (same RC + period + category) into a paragraph."""
    row = group_df.iloc[0]
    lines = [
        f"Financial Planning Summary — {row['Entity_ID']}, Cost Center {row['RC_Code']}, "
        f"Fiscal Year {row['Fiscal_Year']}, Period {row['Fiscal_Period']} ({row['Quarter']}):",
    ]
    for _, r in group_df.iterrows():
        variance_dir = "over budget" if r['Variance_BvA'] > 0 else "under budget"
        lines.append(
            f"  GL {r['GL_Account']} ({r['Expense_Category']}): "
            f"Budget ${r['Budget_USD']:,.2f}, "
            f"Actuals ${r['Actuals_USD']:,.2f}, "
            f"Forecast ${r['Forecast_USD']:,.2f}. "
            f"Budget vs Actuals variance: ${abs(r['Variance_BvA']):,.2f} ({abs(r['Variance_BvA_Pct']):.1f}% {variance_dir}). "
            f"Status: {r['Status']}."
        )
    return "\n".join(lines)


# ── Chunking Functions ────────────────────────────────────────────────────────

def chunk_gl_transactions(df: pd.DataFrame, sample_size: int = 30000) -> list[dict]:
    """
    Sample GL transactions and convert to text chunks.
    We sample to stay within reasonable indexing size and cost.
    """
    df = df[df["Fiscal_Year"] >= 2023]
    logger.info(f"GL transactions (FY2023+): {len(df)} rows. Sampling {sample_size}...")
    df_sample = df.sample(n=min(sample_size, len(df)), random_state=42)

    chunks = []
    for _, row in df_sample.iterrows():
        text = gl_row_to_text(row)
        chunks.append({
            "id": str(uuid.uuid4()),
            "content": text,
            "title": f"GL Transaction {row['Journal_Entry_ID']} — {row['Account_Name']}",
            "data_type": "gl_transaction",
            "entity": str(row.get("Entity_Name", "")),
            "department": str(row.get("Department", "")),
            "fiscal_year": str(row.get("Fiscal_Year", "")),
            "fiscal_period": str(row.get("Fiscal_Period", "")),
            "expense_category": str(row.get("Expense_Category", "")),
            "chunk_index": 0,
        })
    logger.info(f"Created {len(chunks)} GL chunks.")
    return chunks


def chunk_planning_data(df: pd.DataFrame) -> list[dict]:
    """
    Group planning rows by (RC_Code + Entity_ID + Fiscal_Year + Fiscal_Period + Expense_Category)
    and convert each group into a single text chunk.
    """
    df = df[df["Fiscal_Year"] >= 2023]
    logger.info(f"Planning data (FY2023+): {len(df)} rows. Grouping into chunks...")
    group_keys = ["RC_Code", "Entity_ID", "Fiscal_Year", "Fiscal_Period", "Expense_Category"]
    groups = df.groupby(group_keys)

    chunks = []
    for keys, group in groups:
        rc, entity, fy, fp, cat = keys
        text = planning_group_to_text(group)
        chunks.append({
            "id": str(uuid.uuid4()),
            "content": text,
            "title": f"Planning — {entity}, {rc}, FY{fy} P{fp}, {cat}",
            "data_type": "planning",
            "entity": str(entity),
            "department": str(group.iloc[0].get("RC_Code", "")),
            "fiscal_year": str(fy),
            "fiscal_period": str(fp),
            "expense_category": str(cat),
            "chunk_index": 0,
        })

    logger.info(f"Created {len(chunks)} planning chunks.")
    return chunks


# ── Main Pipeline ─────────────────────────────────────────────────────────────

def main():
    logger.info("=" * 60)
    logger.info("FinSight AI — Data Ingestion Pipeline Starting")
    logger.info("=" * 60)

    openai_client = AzureOpenAIClient()
    search_client = AzureSearchClient()

    # STEP 1 — Create the search index
    logger.info("\n[Step 1] Creating Azure AI Search index...")
    search_client.create_index()

    # STEP 2 — Load CSVs
    logger.info("\n[Step 2] Loading datasets...")
    df_gl = pd.read_csv(GL_DATA_PATH)
    df_planning = pd.read_csv(PLANNING_PATH)
    logger.info(f"  GL transactions: {len(df_gl):,} rows")
    logger.info(f"  Planning combined: {len(df_planning):,} rows")

    # STEP 3 — Convert to text chunks
    logger.info("\n[Step 3] Converting to text chunks...")
    gl_chunks = chunk_gl_transactions(df_gl, sample_size=5000)
    planning_chunks = chunk_planning_data(df_planning)
    # Cap planning chunks to stay within free tier 10K document limit
    if len(planning_chunks) > 4500:
        import random; random.seed(42)
        planning_chunks = random.sample(planning_chunks, 4500)
        logger.info(f"  Planning chunks capped to 4,500 for free tier.")
    all_chunks = gl_chunks + planning_chunks
    logger.info(f"  Total chunks to index: {len(all_chunks):,}")

    # STEP 4 — Generate embeddings
    logger.info("\n[Step 4] Generating embeddings (this will take a few minutes)...")
    texts = [c["content"] for c in all_chunks]
    embeddings = openai_client.get_embeddings_batch(texts, batch_size=16)

    for i, chunk in enumerate(all_chunks):
        chunk["embedding"] = embeddings[i]

    # STEP 5 — Upload to Azure AI Search
    logger.info("\n[Step 5] Uploading to Azure AI Search...")
    search_client.upload_documents(all_chunks, batch_size=100)

    logger.info("\n" + "=" * 60)
    logger.info("Ingestion complete!")
    logger.info(f"  GL chunks indexed:       {len(gl_chunks):,}")
    logger.info(f"  Planning chunks indexed: {len(planning_chunks):,}")
    logger.info(f"  Total indexed:           {len(all_chunks):,}")
    logger.info("Run scripts/validate_index.py to verify.")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
