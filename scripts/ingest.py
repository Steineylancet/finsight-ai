"""
FinSight AI — Data Ingestion Pipeline (v2 — Scenario-Based)
============================================================
Builds a focused, scenario-rich index for Crestwood Capital Group covering:

  1. Revenue planning   — synthetic Budget/Actuals/Forecast for revenue GL
                          accounts (Product, Service, Subscription, Consulting)
                          attached to Sales RC codes across 3 regions, FY25–26.

  2. Expense planning   — curated subset: 13 key departments × all expense
                          categories × FY2025–2026 (full 24 periods).

  3. GL transactions    — matching FY2025–2026 transactions for the same
                          departments, providing granular spend detail.

Estimated index size: ~6,000 chunks — well within the 10K free-tier limit.

Run:
    python scripts/ingest.py
"""

import os
import sys
import uuid
import logging
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from backend.azure_openai_client import AzureOpenAIClient
from backend.azure_search import AzureSearchClient
from dotenv import load_dotenv

load_dotenv()
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

# ── Paths ─────────────────────────────────────────────────────────────────────
BASE_DIR    = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
GL_PATH     = os.path.join(BASE_DIR, "gl_data",       "fact_gl_transactions.csv")
PLAN_PATH   = os.path.join(BASE_DIR, "planning_data", "fact_planning_combined.csv")
RC_PATH     = os.path.join(BASE_DIR, "master_data",   "dim_responsibility_center.csv")

# ── Curated RC scope ──────────────────────────────────────────────────────────
# These 20 cost centres cover the key business units that tell a complete
# financial story: revenue-generating (Sales), demand gen (Marketing),
# product build (Engineering, IT), customer retention (Customer Success),
# and corporate overhead (Finance, HR, Executive, Data & Analytics).
SELECTED_RCS = [
    "RC-0006", "RC-0007",   # Sales - Americas (US01)
    "RC-0022",              # Sales - EMEA (UK01)
    "RC-0034", "RC-0035",   # Sales - APAC (SG01)
    "RC-0010", "RC-0011",   # Marketing - Americas (US01)
    "RC-0014", "RC-0015",   # Customer Success - Americas (US01)
    "RC-0028",              # Customer Success - EMEA (UK01)
    "RC-0050", "RC-0051",   # Software Engineering (US01, UK01)
    "RC-0056", "RC-0057",   # IT Infrastructure (US01, UK01)
    "RC-0001",              # Finance (US01)
    "RC-0074", "RC-0075",   # Finance Operations (US01, UK01)
    "RC-0004",              # Executive (US01)
    "RC-0002",              # Human Resources (US01)
    "RC-0053",              # Data & Analytics (US01)
]

# ── Revenue scenario config ───────────────────────────────────────────────────
# Revenue is attached to the primary Sales RC per region.
# Scale factor is relative to Americas US (scale=1.0).
REVENUE_RCS = {
    "RC-0006": ("LE-US01", "Nexgen Corporation USA",    "Sales - Americas", 1.00),
    "RC-0007": ("LE-US01", "Nexgen Corporation USA",    "Sales - Americas", 0.75),
    "RC-0022": ("LE-UK01", "Nexgen Holdings UK Ltd",    "Sales - EMEA",     0.65),
    "RC-0034": ("LE-SG01", "Nexgen APAC Pte Ltd",       "Sales - APAC",     0.55),
    "RC-0035": ("LE-SG01", "Nexgen APAC Pte Ltd",       "Sales - APAC",     0.45),
}

REVENUE_ACCOUNTS = {
    4001: ("Product Revenue",      "Revenue", 0.38),
    4002: ("Service Revenue",      "Revenue", 0.28),
    4003: ("Subscription Revenue", "Revenue", 0.24),
    4004: ("Consulting Revenue",   "Revenue", 0.10),
}

# Base monthly revenue for LE-US01 RC-0006 at FY2025 baseline
BASE_MONTHLY_REVENUE = 7_800_000   # $7.8M / month for primary US Sales RC

# Seasonal monthly multipliers (Q4 end-of-year surge, Q1 dip)
MONTHLY_SEASONAL = {
    1: 0.82, 2: 0.78, 3: 1.04,
    4: 0.88, 5: 0.93, 6: 1.08,
    7: 0.86, 8: 0.91, 9: 1.06,
    10: 1.10, 11: 1.18, 12: 1.34,
}

MONTHS = ["Jan","Feb","Mar","Apr","May","Jun",
          "Jul","Aug","Sep","Oct","Nov","Dec"]
PERIOD_TO_MONTH  = {i+1: m for i, m in enumerate(MONTHS)}
PERIOD_TO_QUARTER = {1:"Q1",2:"Q1",3:"Q1",4:"Q2",5:"Q2",6:"Q2",
                     7:"Q3",8:"Q3",9:"Q3",10:"Q4",11:"Q4",12:"Q4"}


# ══════════════════════════════════════════════════════════════════════════════
# 1.  Revenue data generation
# ══════════════════════════════════════════════════════════════════════════════

def generate_revenue_planning() -> pd.DataFrame:
    """
    Synthesise FY2025–2026 revenue planning rows for each Sales RC code.
    Returns a DataFrame with the same schema as fact_planning_combined.csv,
    plus an Account_Name column used by the text formatter.
    """
    rng = np.random.default_rng(42)
    records = []
    plan_id = 900_000

    for fy in [2025, 2026]:
        # 12% top-line revenue growth in FY2026
        fy_growth = 1.00 if fy == 2025 else 1.12

        for period in range(1, 13):
            month   = PERIOD_TO_MONTH[period]
            quarter = PERIOD_TO_QUARTER[period]
            seasonal = MONTHLY_SEASONAL[period]

            for rc, (entity_id, entity_name, dept, scale) in REVENUE_RCS.items():
                rc_monthly_total = BASE_MONTHLY_REVENUE * scale * fy_growth * seasonal

                for gl_acct, (acct_name, category, rev_share) in REVENUE_ACCOUNTS.items():
                    budget = round(rc_monthly_total * rev_share, 2)

                    # Actuals: ±8% normal variation; slight positive bias (sales team hustle)
                    actual_factor = float(rng.normal(1.02, 0.07))
                    actuals = round(budget * max(actual_factor, 0.70), 2)

                    # Forecast: slightly optimistic projection
                    forecast = round(budget * float(rng.normal(1.04, 0.04)), 2)

                    variance_bva     = round(actuals - budget, 2)
                    variance_bva_pct = round(variance_bva / budget * 100, 2) if budget else 0
                    variance_bvf     = round(forecast - budget, 2)

                    if variance_bva_pct >= 5:
                        status = "Above Target"
                    elif variance_bva_pct <= -5:
                        status = "Below Target"
                    else:
                        status = "On Target"

                    records.append({
                        "PlanDim_ID":      f"REV-{plan_id:07d}",
                        "RC_Code":         rc,
                        "Entity_ID":       entity_id,
                        "Entity_Name":     entity_name,
                        "Department":      dept,
                        "Fiscal_Year":     fy,
                        "Fiscal_Period":   period,
                        "Quarter":         quarter,
                        "Month":           month,
                        "GL_Account":      gl_acct,
                        "Account_Name":    acct_name,
                        "Expense_Category": category,
                        "Budget_USD":      budget,
                        "Actuals_USD":     actuals,
                        "Forecast_USD":    forecast,
                        "Variance_BvA":    variance_bva,
                        "Variance_BvA_Pct": variance_bva_pct,
                        "Variance_BvF":    variance_bvf,
                        "Status":          status,
                    })
                    plan_id += 1

    df = pd.DataFrame(records)
    logger.info(f"Generated {len(df):,} synthetic revenue planning rows "
                f"({df['RC_Code'].nunique()} RC codes × 4 revenue lines × 24 periods).")
    return df


# ══════════════════════════════════════════════════════════════════════════════
# 2.  Text formatters
# ══════════════════════════════════════════════════════════════════════════════

def revenue_group_to_text(group_df: pd.DataFrame) -> str:
    """Format a revenue planning group (RC + FY + Period) as readable text."""
    row = group_df.iloc[0]
    total_budget  = group_df["Budget_USD"].sum()
    total_actuals = group_df["Actuals_USD"].sum()
    total_forecast= group_df["Forecast_USD"].sum()
    total_var_bva = group_df["Variance_BvA"].sum()
    overall_pct   = (total_var_bva / total_budget * 100) if total_budget else 0
    overall_status = "Above Target" if overall_pct >= 5 else ("Below Target" if overall_pct <= -5 else "On Target")

    lines = [
        f"Revenue Performance — {row['Entity_Name']} ({row['Entity_ID']}), "
        f"Cost Center {row['RC_Code']} ({row['Department']}), "
        f"Fiscal Year {row['Fiscal_Year']}, Period {row['Fiscal_Period']} "
        f"({row['Month']} {row['Quarter']}):",
        f"  Total Revenue: Budget ${total_budget:,.2f} | Actuals ${total_actuals:,.2f} | "
        f"Forecast ${total_forecast:,.2f}",
        f"  Overall vs Budget: ${abs(total_var_bva):,.2f} "
        f"({'above' if total_var_bva >= 0 else 'below'} target, "
        f"{abs(overall_pct):.1f}%) — {overall_status}",
        "",
    ]

    for _, r in group_df.iterrows():
        var_dir = "above target" if r["Variance_BvA"] >= 0 else "below target"
        lines.append(
            f"  {r['Account_Name']} (GL {r['GL_Account']}): "
            f"Budget ${r['Budget_USD']:,.2f}, "
            f"Actuals ${r['Actuals_USD']:,.2f}, "
            f"Forecast ${r['Forecast_USD']:,.2f}. "
            f"Variance: ${abs(r['Variance_BvA']):,.2f} "
            f"({abs(r['Variance_BvA_Pct']):.1f}% {var_dir}). "
            f"Status: {r['Status']}."
        )

    return "\n".join(lines)


def expense_group_to_text(group_df: pd.DataFrame) -> str:
    """Format an expense planning group as readable text."""
    row = group_df.iloc[0]
    lines = [
        f"Expense Planning Summary — {row['Entity_ID']}, "
        f"Cost Center {row['RC_Code']}, "
        f"Fiscal Year {row['Fiscal_Year']}, "
        f"Period {row['Fiscal_Period']} ({row['Quarter']}):",
    ]
    for _, r in group_df.iterrows():
        var_dir = "over budget" if r["Variance_BvA"] > 0 else "under budget"
        lines.append(
            f"  GL {r['GL_Account']} ({r['Expense_Category']}): "
            f"Budget ${r['Budget_USD']:,.2f}, "
            f"Actuals ${r['Actuals_USD']:,.2f}, "
            f"Forecast ${r['Forecast_USD']:,.2f}. "
            f"Budget vs Actuals: ${abs(r['Variance_BvA']):,.2f} "
            f"({abs(r['Variance_BvA_Pct']):.1f}% {var_dir}). "
            f"Status: {r['Status']}."
        )
    return "\n".join(lines)


def gl_row_to_text(row) -> str:
    """Convert a GL transaction row into a human-readable sentence."""
    return (
        f"In {row['Month']}/{row['Fiscal_Year']} ({row['Quarter']}), "
        f"{row['Entity_Name']} ({row['Department']} dept, cost center {row['RC_Code']}) "
        f"posted a {row['Document_Type_Desc']} of ${row['Amount_USD']:,.2f} USD "
        f"for {row['Account_Name']} ({row['Expense_Category']}) "
        f"to GL account {row['GL_Account']}, "
        f"vendor: {row['Vendor_Name']}. "
        f"Description: {row['Description']}. "
        f"Status: {row['Status']}. Clearing: {row['Clearing_Status']}."
    )


# ══════════════════════════════════════════════════════════════════════════════
# 3.  Chunking functions
# ══════════════════════════════════════════════════════════════════════════════

def chunk_revenue_planning(df: pd.DataFrame) -> list[dict]:
    """Group revenue planning rows by RC + FY + Period and build chunks."""
    group_keys = ["RC_Code", "Entity_ID", "Fiscal_Year", "Fiscal_Period"]
    chunks = []
    for keys, group in df.groupby(group_keys):
        rc, entity, fy, fp = keys
        row = group.iloc[0]
        text = revenue_group_to_text(group)
        chunks.append({
            "id":               str(uuid.uuid4()),
            "content":          text,
            "title":            f"Revenue — {entity} {rc}, FY{fy} P{fp} ({row['Month']} {row['Quarter']})",
            "data_type":        "planning_revenue",
            "entity":           str(entity),
            "department":       str(row.get("Department", "")),
            "fiscal_year":      str(fy),
            "fiscal_period":    str(fp),
            "expense_category": "Revenue",
            "chunk_index":      0,
        })
    logger.info(f"Created {len(chunks):,} revenue planning chunks.")
    return chunks


def chunk_expense_planning(df: pd.DataFrame) -> list[dict]:
    """Group expense planning rows by RC + Entity + FY + Period + Category."""
    df_filtered = df[
        (df["Fiscal_Year"] >= 2025) &
        (df["RC_Code"].isin(SELECTED_RCS))
    ].copy()
    logger.info(f"Expense planning rows (FY25-26, curated RCs): {len(df_filtered):,}")

    group_keys = ["RC_Code", "Entity_ID", "Fiscal_Year", "Fiscal_Period", "Expense_Category"]
    chunks = []
    for keys, group in df_filtered.groupby(group_keys):
        rc, entity, fy, fp, cat = keys
        text = expense_group_to_text(group)
        chunks.append({
            "id":               str(uuid.uuid4()),
            "content":          text,
            "title":            f"Expenses — {entity} {rc}, FY{fy} P{fp}, {cat}",
            "data_type":        "planning_expense",
            "entity":           str(entity),
            "department":       str(group.iloc[0].get("RC_Code", "")),
            "fiscal_year":      str(fy),
            "fiscal_period":    str(fp),
            "expense_category": str(cat),
            "chunk_index":      0,
        })
    logger.info(f"Created {len(chunks):,} expense planning chunks.")
    return chunks


def chunk_gl_transactions(df: pd.DataFrame, sample_size: int = 3000) -> list[dict]:
    """
    Sample GL transactions for the curated RC codes (FY2025–2026).
    Stratified sampling: proportional per department so no single dept dominates.
    """
    df_filtered = df[
        (df["Fiscal_Year"] >= 2025) &
        (df["RC_Code"].isin(SELECTED_RCS))
    ].copy()
    logger.info(f"GL rows available (FY25-26, curated RCs): {len(df_filtered):,}")

    # Stratified sample: up to sample_size total, proportional by department
    dept_counts = df_filtered["Department"].value_counts()
    total = len(df_filtered)
    sampled_parts = []
    for dept, count in dept_counts.items():
        n = max(1, round(sample_size * count / total))
        part = df_filtered[df_filtered["Department"] == dept].sample(
            n=min(n, count), random_state=42
        )
        sampled_parts.append(part)
    df_sample = pd.concat(sampled_parts).sample(
        n=min(sample_size, sum(len(p) for p in sampled_parts)), random_state=42
    )
    logger.info(f"Sampled {len(df_sample):,} GL transactions (stratified by department).")

    chunks = []
    for _, row in df_sample.iterrows():
        text = gl_row_to_text(row)
        chunks.append({
            "id":               str(uuid.uuid4()),
            "content":          text,
            "title":            f"GL {row['Journal_Entry_ID']} — {row['Account_Name']} ({row['Department']})",
            "data_type":        "gl_transaction",
            "entity":           str(row.get("Entity_Name", "")),
            "department":       str(row.get("Department", "")),
            "fiscal_year":      str(row.get("Fiscal_Year", "")),
            "fiscal_period":    str(row.get("Fiscal_Period", "")),
            "expense_category": str(row.get("Expense_Category", "")),
            "chunk_index":      0,
        })
    logger.info(f"Created {len(chunks):,} GL transaction chunks.")
    return chunks


# ══════════════════════════════════════════════════════════════════════════════
# 4.  Main
# ══════════════════════════════════════════════════════════════════════════════

def main():
    logger.info("=" * 65)
    logger.info("FinSight AI — Scenario-Based Ingestion Pipeline")
    logger.info("=" * 65)

    openai_client = AzureOpenAIClient()
    search_client = AzureSearchClient()

    # ── Step 1: Create index ──────────────────────────────────────────────────
    logger.info("\n[Step 1] Creating / updating Azure AI Search index...")
    search_client.create_index()

    # ── Step 2: Load source data ──────────────────────────────────────────────
    logger.info("\n[Step 2] Loading source data...")
    df_gl      = pd.read_csv(GL_PATH)
    df_plan    = pd.read_csv(PLAN_PATH)
    pd.read_csv(RC_PATH)  # loaded for reference; RC metadata embedded in source data

    logger.info(f"  GL transactions loaded:  {len(df_gl):,} rows")
    logger.info(f"  Planning data loaded:    {len(df_plan):,} rows")

    # ── Step 3: Build chunks ──────────────────────────────────────────────────
    logger.info("\n[Step 3] Building scenario-based chunks...")

    # 3a. Revenue planning (generated)
    df_revenue = generate_revenue_planning()
    revenue_chunks = chunk_revenue_planning(df_revenue)

    # 3b. Expense planning (curated, FY25-26)
    expense_chunks = chunk_expense_planning(df_plan)

    # 3c. GL transactions (curated, FY25-26, stratified sample)
    gl_chunks = chunk_gl_transactions(df_gl, sample_size=1500)

    all_chunks = revenue_chunks + expense_chunks + gl_chunks
    logger.info(f"\n  Revenue planning chunks:  {len(revenue_chunks):,}")
    logger.info(f"  Expense planning chunks:  {len(expense_chunks):,}")
    logger.info(f"  GL transaction chunks:    {len(gl_chunks):,}")
    logger.info(f"  ─────────────────────────────")
    logger.info(f"  TOTAL chunks to index:    {len(all_chunks):,}")

    if len(all_chunks) > 9_800:
        raise RuntimeError(
            f"Chunk count {len(all_chunks)} exceeds safe free-tier limit of 9,800. "
            "Reduce sample sizes before proceeding."
        )

    # ── Step 4: Embed ─────────────────────────────────────────────────────────
    logger.info("\n[Step 4] Generating embeddings (batches of 16)...")
    texts = [c["content"] for c in all_chunks]
    embeddings = openai_client.get_embeddings_batch(texts, batch_size=16)
    for i, chunk in enumerate(all_chunks):
        chunk["embedding"] = embeddings[i]

    # ── Step 5: Upload ────────────────────────────────────────────────────────
    logger.info("\n[Step 5] Uploading to Azure AI Search...")
    search_client.upload_documents(all_chunks, batch_size=100)

    logger.info("\n" + "=" * 65)
    logger.info("Ingestion complete!")
    logger.info(f"  Revenue planning chunks indexed:  {len(revenue_chunks):,}")
    logger.info(f"  Expense planning chunks indexed:  {len(expense_chunks):,}")
    logger.info(f"  GL transaction chunks indexed:    {len(gl_chunks):,}")
    logger.info(f"  Total indexed:                    {len(all_chunks):,}")
    logger.info("\nRun  python scripts/validate_index.py  to verify.")
    logger.info("=" * 65)


if __name__ == "__main__":
    main()
