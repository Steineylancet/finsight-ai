"""
FinSight AI — LangChain Tool Definitions
Each tool wraps a validated pandas operation against the loaded DataFrames.
The LLM selects which tool to call with structured parameters — no arbitrary
code execution, no eval(). Department names are whitelisted twice: the JSON
schema the model sees lists the valid values, and every tool re-validates
server-side through the catalog resolver, so a hallucinated or ambiguous name
returns a clarification instead of silently matching nothing.
"""

import logging
import re
from typing import Literal, Optional

from langchain_core.tools import BaseTool, tool
from pydantic import BaseModel, Field, ValidationError, field_validator

from backend.data_loader import DataLoader
from backend.graph.catalog import DEPARTMENTS, resolve_department

logger = logging.getLogger(__name__)

DepartmentName = Literal[tuple(DEPARTMENTS)]
Quarter = Literal["Q1", "Q2", "Q3", "Q4"]


def _df():
    return DataLoader.get()


def norm_fy(fiscal_year: str) -> str:
    """Strip an optional 'FY' prefix so '2025' and 'FY2025' both match the data."""
    return re.sub(r"(?i)^fy", "", str(fiscal_year)).strip()


def _norm_quarter(quarter: str) -> str:
    return str(quarter).strip().upper()


class _PeriodArgs(BaseModel):
    fiscal_year: str = Field(description="4-digit fiscal year, e.g. '2025'.")
    quarter: Quarter = Field(description="Fiscal quarter.")

    @field_validator("fiscal_year", mode="before")
    @classmethod
    def _fy(cls, v):
        return norm_fy(v)

    @field_validator("quarter", mode="before")
    @classmethod
    def _q(cls, v):
        return _norm_quarter(v)


def _resolve_or_raise(v):
    """Map free text ('IT', 'marketing india') onto the whitelist, or explain why not."""
    if v is None:
        return v
    res = resolve_department(v)
    if res.ok:
        return res.department
    raise ValueError(res.message(v))


class DepartmentPeriodArgs(_PeriodArgs):
    department: DepartmentName = Field(description="Department name.")

    @field_validator("department", mode="before")
    @classmethod
    def _dept(cls, v):
        return _resolve_or_raise(v)


class AnomalyArgs(_PeriodArgs):
    threshold_pct: float = Field(10.0, description="Minimum |budget variance %| to flag.")
    min_amount: float = Field(10000.0, description="Minimum |budget variance $| to flag.")


class VendorArgs(_PeriodArgs):
    department: Optional[DepartmentName] = Field(None, description="Optional department filter.")
    top_n: int = Field(5, ge=1, le=25, description="Number of vendors to return.")

    @field_validator("department", mode="before")
    @classmethod
    def _dept(cls, v):
        return _resolve_or_raise(v)


class YtdArgs(BaseModel):
    department: DepartmentName = Field(description="Department name.")
    fiscal_year: str = Field(description="4-digit fiscal year, e.g. '2025'.")

    @field_validator("department", mode="before")
    @classmethod
    def _dept(cls, v):
        return _resolve_or_raise(v)

    @field_validator("fiscal_year", mode="before")
    @classmethod
    def _fy(cls, v):
        return norm_fy(v)


def run_tool(t: BaseTool, args: dict) -> dict:
    """Invoke a tool from graph code, turning argument validation failures into an error dict."""
    try:
        return t.invoke(args)
    except ValidationError as e:
        msg = e.errors()[0].get("msg", str(e))
        return {"error": msg.removeprefix("Value error, ")}


def _no_actuals_error(fiscal_year: str, quarter: str) -> dict | None:
    loader = _df()
    if loader.has_actuals(fiscal_year, quarter):
        return None
    lfy, lq = loader.latest_actuals_period()
    return {"error": f"{quarter} FY{fiscal_year} has no actuals booked yet (budget and forecast "
                     f"only), so there is no variance to analyse. The latest quarter with actuals "
                     f"is {lq} FY{lfy}."}


def _pct(numerator, denominator):
    return round(numerator / denominator * 100, 1) if denominator else 0.0


def _add_variances(df):
    df["Variance_BvA"] = df["Actuals_USD"] - df["Budget_USD"]
    df["Variance_BvA_Pct"] = (
        (df["Variance_BvA"] / df["Budget_USD"].replace(0, float("nan"))) * 100
    ).round(1)
    df["Variance_AvF"] = df["Actuals_USD"] - df["Forecast_USD"]
    return df


def _total_row(label_key: str, label: str, df) -> dict:
    budget = float(df["Budget_USD"].sum())
    actuals = float(df["Actuals_USD"].sum())
    forecast = float(df["Forecast_USD"].sum())
    return {
        label_key: label,
        "Budget_USD": budget,
        "Actuals_USD": actuals,
        "Forecast_USD": forecast,
        "Variance_BvA": actuals - budget,
        "Variance_BvA_Pct": _pct(actuals - budget, budget),
        "Variance_AvF": actuals - forecast,
    }


# ─────────────────────────────────────────────────────────────────────────────
# Tool 1 — Department quarterly summary
# ─────────────────────────────────────────────────────────────────────────────

@tool(args_schema=DepartmentPeriodArgs)
def get_department_quarterly_summary(department: str, fiscal_year: str, quarter: str) -> dict:
    """
    Budget vs actuals vs forecast for one department in one quarter, broken down
    by expense category, with a TOTAL row. Variance_BvA = actuals minus budget;
    Variance_AvF = actuals minus forecast. Use for 'How did X perform in Q2 FY2025?'
    or 'actuals vs forecast for X'.
    """
    df = _df().planning
    filtered = df[
        (df["Department"] == department)
        & (df["Fiscal_Year"] == fiscal_year)
        & (df["Quarter"] == quarter)
    ]
    if filtered.empty:
        return {"error": f"No planning data for {department} in {quarter} FY{fiscal_year}."}

    summary = _add_variances(
        filtered.groupby("Expense_Category")
        .agg(Budget_USD=("Budget_USD", "sum"), Actuals_USD=("Actuals_USD", "sum"),
             Forecast_USD=("Forecast_USD", "sum"))
        .reset_index()
    )
    rows = summary.to_dict(orient="records")
    rows.append(_total_row("Expense_Category", "TOTAL", summary))
    result = {"department": department, "fiscal_year": fiscal_year, "quarter": quarter,
              "rows": rows, "row_count": len(rows)}
    if not _df().has_actuals(fiscal_year, quarter):
        result["note"] = "No actuals booked yet for this quarter — budget and forecast only."
    return result


# ─────────────────────────────────────────────────────────────────────────────
# Tool 2 — Variance breakdown ranked by driver
# ─────────────────────────────────────────────────────────────────────────────

@tool(args_schema=DepartmentPeriodArgs)
def get_variance_breakdown(department: str, fiscal_year: str, quarter: str) -> dict:
    """
    Variance by expense category for one department and quarter, sorted by
    absolute budget variance (largest driver first). Use to explain WHY a
    department was over or under budget.
    """
    if err := _no_actuals_error(fiscal_year, quarter):
        return err
    result = get_department_quarterly_summary.invoke(
        {"department": department, "fiscal_year": fiscal_year, "quarter": quarter}
    )
    if "error" in result:
        return result

    rows = [r for r in result["rows"] if r["Expense_Category"] != "TOTAL"]
    total = next(r for r in result["rows"] if r["Expense_Category"] == "TOTAL")
    rows_sorted = sorted(rows, key=lambda r: abs(r["Variance_BvA"]), reverse=True)
    return {"department": result["department"], "fiscal_year": result["fiscal_year"],
            "quarter": result["quarter"], "drivers": rows_sorted,
            "top_driver": rows_sorted[0] if rows_sorted else None, "total": total}


# ─────────────────────────────────────────────────────────────────────────────
# Tool 3 — Anomaly scanner
# ─────────────────────────────────────────────────────────────────────────────

@tool(args_schema=AnomalyArgs)
def get_anomalies(fiscal_year: str, quarter: str, threshold_pct: float = 10.0,
                  min_amount: float = 10000.0) -> dict:
    """
    Scan every department for one quarter and flag those where BOTH
    |budget variance %| >= threshold_pct AND |budget variance $| >= min_amount.
    Returns flagged departments ranked by variance. Use for 'Which departments
    are over budget?' or 'Flag variances above 10% in Q3 FY2025'.
    """
    if err := _no_actuals_error(fiscal_year, quarter):
        return err
    df = _df().planning
    filtered = df[(df["Fiscal_Year"] == fiscal_year) & (df["Quarter"] == quarter)]
    if filtered.empty:
        return {"error": f"No planning data for {quarter} FY{fiscal_year}."}

    summary = _add_variances(
        filtered.groupby("Department")
        .agg(Budget_USD=("Budget_USD", "sum"), Actuals_USD=("Actuals_USD", "sum"),
             Forecast_USD=("Forecast_USD", "sum"))
        .reset_index()
    )
    flagged = summary[
        (summary["Variance_BvA_Pct"].abs() >= threshold_pct)
        & (summary["Variance_BvA"].abs() >= min_amount)
    ].sort_values("Variance_BvA", ascending=False)

    return {"fiscal_year": fiscal_year, "quarter": quarter, "threshold_pct": threshold_pct,
            "min_amount": min_amount, "total_departments_scanned": len(summary),
            "flagged_count": len(flagged), "anomalies": flagged.to_dict(orient="records"),
            "company_total": _total_row("Department", "ALL DEPARTMENTS", summary)}


# ─────────────────────────────────────────────────────────────────────────────
# Tool 4 — Top vendors by spend
# ─────────────────────────────────────────────────────────────────────────────

@tool(args_schema=VendorArgs)
def get_top_vendors(fiscal_year: str, quarter: str, department: Optional[str] = None,
                    top_n: int = 5) -> dict:
    """
    Top N vendors by actual GL spend for one quarter, optionally for a single
    department. Use for vendor spend questions.
    """
    gl = _df().gl
    mask = (gl["Fiscal_Year"] == fiscal_year) & (gl["Quarter"] == quarter)
    if department:
        mask &= gl["Department"] == department

    filtered = gl[mask]
    if filtered.empty:
        return {"error": f"No GL transactions for {quarter} FY{fiscal_year}."}

    vendors = (
        filtered.groupby("Vendor_Name")
        .agg(Total_Spend_USD=("Amount_USD", "sum"), Transactions=("Amount_USD", "size"))
        .reset_index()
        .sort_values("Total_Spend_USD", ascending=False)
        .head(top_n)
    )
    return {"fiscal_year": fiscal_year, "quarter": quarter,
            "department": department or "All Departments", "top_n": top_n,
            "vendors": vendors.to_dict(orient="records")}


# ─────────────────────────────────────────────────────────────────────────────
# Tool 5 — YTD summary
# ─────────────────────────────────────────────────────────────────────────────

@tool(args_schema=YtdArgs)
def get_ytd_summary(department: str, fiscal_year: str) -> dict:
    """
    Full-year view for one department: budget vs actuals vs forecast per quarter
    plus a YTD TOTAL row. Use for full-year or year-to-date performance and
    actual-vs-forecast questions when no single quarter is specified.
    """
    df = _df().planning
    filtered = df[(df["Department"] == department) & (df["Fiscal_Year"] == fiscal_year)]
    if filtered.empty:
        return {"error": f"No planning data for {department} in FY{fiscal_year}."}

    by_quarter = _add_variances(
        filtered.groupby("Quarter")
        .agg(Budget_USD=("Budget_USD", "sum"), Actuals_USD=("Actuals_USD", "sum"),
             Forecast_USD=("Forecast_USD", "sum"))
        .reset_index()
        .sort_values("Quarter")
    )
    rows = by_quarter.to_dict(orient="records")
    rows.append(_total_row("Quarter", "YTD TOTAL", by_quarter))
    result = {"department": department, "fiscal_year": fiscal_year, "rows": rows}
    pending = [r["Quarter"] for r in rows[:-1] if not _df().has_actuals(fiscal_year, r["Quarter"])]
    if pending:
        result["note"] = (f"No actuals booked yet for {', '.join(pending)} — those quarters show "
                          f"budget and forecast only, so the YTD actuals cover fewer quarters than budget.")
    return result


DATA_TOOLS = [
    get_department_quarterly_summary,
    get_variance_breakdown,
    get_anomalies,
    get_top_vendors,
    get_ytd_summary,
]
ALL_TOOLS = DATA_TOOLS
