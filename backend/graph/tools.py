"""
FinSight AI — LangChain Tool Definitions
Each tool wraps a validated pandas operation against the loaded DataFrames.
The LLM selects which tool to call with structured parameters — no arbitrary
code execution, no eval(). Safer and fully auditable.
"""

import logging
import re
from typing import Optional
from langchain_core.tools import tool

from backend.data_loader import DataLoader

logger = logging.getLogger(__name__)


def _df():
    return DataLoader.get()


def _norm_fy(fiscal_year: str) -> str:
    """Strip an optional 'FY' prefix so '2025' and 'FY2025' both match the data."""
    return re.sub(r"(?i)^fy", "", str(fiscal_year)).strip()


def _norm_quarter(quarter: str) -> str:
    """Normalise casing/whitespace so 'q2' and 'Q2' both match the data."""
    return str(quarter).strip().upper()


# ─────────────────────────────────────────────────────────────────────────────
# Tool 1 — Department quarterly summary
# ─────────────────────────────────────────────────────────────────────────────

@tool
def get_department_quarterly_summary(
    department: str,
    fiscal_year: str,
    quarter: str,
) -> dict:
    """
    Returns budget vs actuals vs forecast for a specific department and quarter,
    broken down by expense category. Use for questions like:
    'How did Software Engineering perform in Q2 FY2025?'
    """
    fiscal_year, quarter = _norm_fy(fiscal_year), _norm_quarter(quarter)
    df = _df().planning
    mask = (
        (df["Department"] == department) &
        (df["Fiscal_Year"] == fiscal_year) &
        (df["Quarter"] == quarter)
    )
    filtered = df[mask]

    if filtered.empty:
        return {"error": f"No data found for {department} | {quarter} FY{fiscal_year}"}

    summary = (
        filtered.groupby("Expense_Category")
        .agg(
            Budget_USD=("Budget_USD", "sum"),
            Actuals_USD=("Actuals_USD", "sum"),
            Forecast_USD=("Forecast_USD", "sum"),
        )
        .reset_index()
    )
    summary["Variance_BvA"] = summary["Actuals_USD"] - summary["Budget_USD"]
    summary["Variance_BvA_Pct"] = (
        (summary["Variance_BvA"] / summary["Budget_USD"].replace(0, float("nan"))) * 100
    ).round(1)

    total_row = {
        "Expense_Category": "TOTAL",
        "Budget_USD": summary["Budget_USD"].sum(),
        "Actuals_USD": summary["Actuals_USD"].sum(),
        "Forecast_USD": summary["Forecast_USD"].sum(),
        "Variance_BvA": summary["Variance_BvA"].sum(),
        "Variance_BvA_Pct": round(
            (summary["Variance_BvA"].sum() / summary["Budget_USD"].sum()) * 100, 1
        ) if summary["Budget_USD"].sum() != 0 else 0,
    }

    rows = summary.to_dict(orient="records")
    rows.append(total_row)

    return {
        "department": department,
        "fiscal_year": fiscal_year,
        "quarter": quarter,
        "rows": rows,
        "row_count": len(rows),
    }


# ─────────────────────────────────────────────────────────────────────────────
# Tool 2 — Variance breakdown ranked by driver
# ─────────────────────────────────────────────────────────────────────────────

@tool
def get_variance_breakdown(
    department: str,
    fiscal_year: str,
    quarter: str,
) -> dict:
    """
    Returns variance by expense category sorted by absolute variance (largest first).
    Use to decompose WHY a department was over/under budget.
    """
    result = get_department_quarterly_summary.invoke({
        "department": department,
        "fiscal_year": fiscal_year,
        "quarter": quarter,
    })

    if "error" in result:
        return result

    rows = [r for r in result["rows"] if r["Expense_Category"] != "TOTAL"]
    rows_sorted = sorted(rows, key=lambda r: abs(r["Variance_BvA"]), reverse=True)

    return {
        "department": department,
        "fiscal_year": fiscal_year,
        "quarter": quarter,
        "drivers": rows_sorted,
        "top_driver": rows_sorted[0] if rows_sorted else None,
    }


# ─────────────────────────────────────────────────────────────────────────────
# Tool 3 — Anomaly scanner
# ─────────────────────────────────────────────────────────────────────────────

@tool
def get_anomalies(
    fiscal_year: str,
    quarter: str,
    threshold_pct: float = 10.0,
    min_amount: float = 10000.0,
) -> dict:
    """
    Scans all departments for the given period and flags any where:
      - |Variance %| > threshold_pct  AND
      - |Variance_BvA| >= min_amount (USD)
    Returns a ranked list sorted by absolute variance (most severe first).
    Use for: 'Which departments are over budget?' or 'Flag anomalies in Q3 FY2025.'
    """
    fiscal_year, quarter = _norm_fy(fiscal_year), _norm_quarter(quarter)
    df = _df().planning
    mask = (
        (df["Fiscal_Year"] == fiscal_year) &
        (df["Quarter"] == quarter)
    )
    filtered = df[mask]

    if filtered.empty:
        return {"error": f"No data found for {quarter} FY{fiscal_year}"}

    summary = (
        filtered.groupby("Department")
        .agg(
            Budget_USD=("Budget_USD", "sum"),
            Actuals_USD=("Actuals_USD", "sum"),
        )
        .reset_index()
    )
    summary["Variance_BvA"] = summary["Actuals_USD"] - summary["Budget_USD"]
    summary["Variance_BvA_Pct"] = (
        (summary["Variance_BvA"] / summary["Budget_USD"].replace(0, float("nan"))) * 100
    ).round(1)

    flagged = summary[
        (summary["Variance_BvA_Pct"].abs() >= threshold_pct) &
        (summary["Variance_BvA"].abs() >= min_amount)
    ].copy()

    flagged_sorted = flagged.sort_values("Variance_BvA", ascending=False)

    return {
        "fiscal_year": fiscal_year,
        "quarter": quarter,
        "threshold_pct": threshold_pct,
        "min_amount": min_amount,
        "total_departments_scanned": len(summary),
        "flagged_count": len(flagged_sorted),
        "anomalies": flagged_sorted.to_dict(orient="records"),
    }


# ─────────────────────────────────────────────────────────────────────────────
# Tool 4 — Top vendors by spend
# ─────────────────────────────────────────────────────────────────────────────

@tool
def get_top_vendors(
    fiscal_year: str,
    quarter: str,
    department: Optional[str] = None,
    top_n: int = 5,
) -> dict:
    """
    Returns the top N vendors by actual spend for the given period.
    Optionally filtered to a specific department.
    """
    fiscal_year, quarter = _norm_fy(fiscal_year), _norm_quarter(quarter)
    gl = _df().gl
    mask = (
        (gl["Fiscal_Year"] == fiscal_year) &
        (gl["Quarter"] == quarter)
    )
    if department:
        mask &= gl["Department"] == department

    filtered = gl[mask]

    if filtered.empty:
        return {"error": f"No GL transactions found for {quarter} FY{fiscal_year}"}

    vendors = (
        filtered.groupby("Vendor_Name")
        .agg(Total_Spend=("Amount_USD", "sum"))
        .reset_index()
        .sort_values("Total_Spend", ascending=False)
        .head(top_n)
    )

    return {
        "fiscal_year": fiscal_year,
        "quarter": quarter,
        "department": department or "All Departments",
        "top_n": top_n,
        "vendors": vendors.to_dict(orient="records"),
    }


# ─────────────────────────────────────────────────────────────────────────────
# Tool 5 — YTD summary
# ─────────────────────────────────────────────────────────────────────────────

@tool
def get_ytd_summary(
    department: str,
    fiscal_year: str,
) -> dict:
    """
    Returns year-to-date actuals vs budget for a department across all quarters
    in the given fiscal year. Use for full-year performance questions.
    """
    fiscal_year = _norm_fy(fiscal_year)
    df = _df().planning
    mask = (
        (df["Department"] == department) &
        (df["Fiscal_Year"] == fiscal_year)
    )
    filtered = df[mask]

    if filtered.empty:
        return {"error": f"No data found for {department} FY{fiscal_year}"}

    by_quarter = (
        filtered.groupby("Quarter")
        .agg(
            Budget_USD=("Budget_USD", "sum"),
            Actuals_USD=("Actuals_USD", "sum"),
            Forecast_USD=("Forecast_USD", "sum"),
        )
        .reset_index()
        .sort_values("Quarter")
    )
    by_quarter["Variance_BvA"] = by_quarter["Actuals_USD"] - by_quarter["Budget_USD"]
    by_quarter["Variance_BvA_Pct"] = (
        (by_quarter["Variance_BvA"] / by_quarter["Budget_USD"].replace(0, float("nan"))) * 100
    ).round(1)

    ytd_total = {
        "Quarter": "YTD TOTAL",
        "Budget_USD": by_quarter["Budget_USD"].sum(),
        "Actuals_USD": by_quarter["Actuals_USD"].sum(),
        "Forecast_USD": by_quarter["Forecast_USD"].sum(),
        "Variance_BvA": by_quarter["Variance_BvA"].sum(),
        "Variance_BvA_Pct": round(
            (by_quarter["Variance_BvA"].sum() / by_quarter["Budget_USD"].sum()) * 100, 1
        ) if by_quarter["Budget_USD"].sum() != 0 else 0,
    }

    rows = by_quarter.to_dict(orient="records")
    rows.append(ytd_total)

    return {
        "department": department,
        "fiscal_year": fiscal_year,
        "rows": rows,
    }


# ── Tool registry (used by router to bind tools to the LLM) ──────────────────
ALL_TOOLS = [
    get_department_quarterly_summary,
    get_variance_breakdown,
    get_anomalies,
    get_top_vendors,
    get_ytd_summary,
]
