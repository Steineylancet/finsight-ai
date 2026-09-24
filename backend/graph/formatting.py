"""FinSight AI — markdown table rendering for tool results."""

_ROW_KEYS = ("rows", "drivers", "anomalies", "vendors")

_HEADERS = {
    "Expense_Category": "Expense Category",
    "Vendor_Name": "Vendor",
    "Budget_USD": "Budget ($)",
    "Actuals_USD": "Actuals ($)",
    "Forecast_USD": "Forecast ($)",
    "Variance_BvA": "vs Budget ($)",
    "Variance_BvA_Pct": "vs Budget %",
    "Variance_AvF": "vs Forecast ($)",
    "Total_Spend_USD": "Spend ($)",
}
_SIGNED = {"Variance_BvA", "Variance_AvF"}


def _cell(key: str, v) -> str:
    if v is None or (isinstance(v, float) and v != v):  # None or NaN
        return "—"
    if isinstance(v, bool) or not isinstance(v, (int, float)):
        return str(v)
    if key.endswith("_Pct"):
        return f"{v:+.1f}%"
    if key in _SIGNED:
        return f"{v:+,.0f}"
    if isinstance(v, float):
        return f"{v:,.0f}"
    return f"{v:,}"


def markdown_table(rows: list[dict]) -> str | None:
    if not rows:
        return None
    keys = list(rows[0].keys())
    lines = [
        "| " + " | ".join(_HEADERS.get(k, k.replace("_", " ")) for k in keys) + " |",
        "|" + "|".join("---" for _ in keys) + "|",
    ]
    for row in rows:
        lines.append("| " + " | ".join(_cell(k, row.get(k)) for k in keys) + " |")
    return "\n".join(lines)


def tool_result_table(result: dict) -> str | None:
    key = next((k for k in _ROW_KEYS if k in result), None)
    return markdown_table(result[key]) if key else None
