"""
Data tools against the real planning/GL data (loaded locally, or from Blob
Storage in CI). No LLM calls.
"""

import pytest
from langchain_core.utils.function_calling import convert_to_openai_tool

from backend.data_loader import DataLoader
from backend.graph.catalog import DEPARTMENTS
from backend.graph.formatting import tool_result_table
from backend.graph.tools import (
    DATA_TOOLS, get_anomalies, get_department_quarterly_summary, get_top_vendors,
    get_variance_breakdown, get_ytd_summary, run_tool,
)


@pytest.fixture(scope="module", autouse=True)
def data():
    return DataLoader.get()


def test_catalog_matches_data(data):
    assert set(data.departments()) == set(DEPARTMENTS)


def test_department_enum_is_in_tool_schema():
    for t in DATA_TOOLS:
        props = convert_to_openai_tool(t)["function"]["parameters"]["properties"]
        if "department" in props:
            assert str(sorted(DEPARTMENTS)[0]) in str(props["department"]), t.name


def test_fy_prefix_and_quarter_case_are_normalised():
    a = run_tool(get_department_quarterly_summary,
                 {"department": "Finance", "fiscal_year": "FY2025", "quarter": "q1"})
    b = run_tool(get_department_quarterly_summary,
                 {"department": "Finance", "fiscal_year": "2025", "quarter": "Q1"})
    assert "error" not in a and a["rows"] == b["rows"]


def test_alias_resolves_inside_tool():
    r = run_tool(get_department_quarterly_summary,
                 {"department": "IT", "fiscal_year": "2025", "quarter": "Q2"})
    assert r["department"] == "IT Infrastructure"


def test_ambiguous_department_returns_clarification():
    r = run_tool(get_variance_breakdown,
                 {"department": "Marketing", "fiscal_year": "2025", "quarter": "Q2"})
    assert "error" in r and "Marketing - EMEA" in r["error"]


def test_summary_total_row_equals_sum_of_categories():
    r = run_tool(get_department_quarterly_summary,
                 {"department": "Software Engineering", "fiscal_year": "2025", "quarter": "Q2"})
    cats, total = r["rows"][:-1], r["rows"][-1]
    assert total["Expense_Category"] == "TOTAL"
    for col in ("Budget_USD", "Actuals_USD", "Forecast_USD"):
        assert total[col] == pytest.approx(sum(c[col] for c in cats))
    assert total["Variance_AvF"] == pytest.approx(total["Actuals_USD"] - total["Forecast_USD"])


def test_variance_drivers_sorted_by_absolute_variance():
    r = run_tool(get_variance_breakdown,
                 {"department": "IT Infrastructure", "fiscal_year": "2025", "quarter": "Q3"})
    sizes = [abs(d["Variance_BvA"]) for d in r["drivers"]]
    assert sizes == sorted(sizes, reverse=True)


def test_future_quarter_has_no_variance(data):
    lfy, lq = data.latest_actuals_period()
    assert (lfy, lq) == ("2026", "Q1")
    r = run_tool(get_variance_breakdown,
                 {"department": "Finance", "fiscal_year": "2026", "quarter": "Q3"})
    assert "no actuals" in r["error"]
    r = run_tool(get_anomalies, {"fiscal_year": "2026", "quarter": "Q4"})
    assert "no actuals" in r["error"]


def test_future_quarter_summary_still_returns_budget_and_forecast():
    r = run_tool(get_department_quarterly_summary,
                 {"department": "Finance", "fiscal_year": "2026", "quarter": "Q3"})
    assert "note" in r and r["rows"][-1]["Forecast_USD"] > 0


def test_anomaly_thresholds_applied():
    r = run_tool(get_anomalies, {"fiscal_year": "2025", "quarter": "Q2",
                                 "threshold_pct": 45, "min_amount": 1_000_000})
    for a in r["anomalies"]:
        assert abs(a["Variance_BvA_Pct"]) >= 45 and abs(a["Variance_BvA"]) >= 1_000_000
    assert r["company_total"]["Actuals_USD"] > 0


def test_top_vendors_respects_top_n_and_department():
    r = run_tool(get_top_vendors, {"fiscal_year": "2025", "quarter": "Q1",
                                   "department": "Finance", "top_n": 3})
    assert r["department"] == "Finance" and len(r["vendors"]) == 3
    spend = [v["Total_Spend_USD"] for v in r["vendors"]]
    assert spend == sorted(spend, reverse=True)


def test_ytd_includes_total_row():
    r = run_tool(get_ytd_summary, {"department": "Legal", "fiscal_year": "2024"})
    assert r["rows"][-1]["Quarter"] == "YTD TOTAL" and len(r["rows"]) == 5


def test_table_rendering_signs_variances():
    r = run_tool(get_department_quarterly_summary,
                 {"department": "Finance", "fiscal_year": "2025", "quarter": "Q1"})
    table = tool_result_table(r)
    assert "vs Budget ($)" in table and "| TOTAL |" in table
    assert "+" in table.splitlines()[-1]
