"""
FinSight 2.0 — Corpus Generation Script
Generates ~122 management documents from real CSV data:
  - 104 quarterly department expense memos (GPT-4o)
  - 10 vendor profiles (GPT-4o)
  - 8 static policy documents (no API cost)

Run:
    python scripts/generate_corpus.py

Resumable: skips files that already exist.
Estimated cost: ~$1.50 one-time (GPT-4o at current pricing).
"""

import os, sys, re, time, logging
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from backend.azure_openai_client import AzureOpenAIClient
from dotenv import load_dotenv

load_dotenv()
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

BASE_DIR    = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PLAN_PATH   = os.path.join(BASE_DIR, "planning_data", "fact_planning_combined.csv")
GL_PATH     = os.path.join(BASE_DIR, "gl_data", "fact_gl_transactions.csv")
RC_PATH     = os.path.join(BASE_DIR, "master_data", "dim_responsibility_center.csv")
CORPUS_DIR  = os.path.join(BASE_DIR, "corpus")

SELECTED_RCS = [
    "RC-0001", "RC-0002", "RC-0004",
    "RC-0006", "RC-0007", "RC-0010", "RC-0011",
    "RC-0014", "RC-0015", "RC-0022", "RC-0028",
    "RC-0034", "RC-0035", "RC-0050", "RC-0051",
    "RC-0053", "RC-0056", "RC-0057",
    "RC-0074", "RC-0075",
]

FISCAL_YEARS = [2025, 2026]
QUARTERS     = ["Q1", "Q2", "Q3", "Q4"]

ENTITY_NAMES = {
    "LE-US01": "Nexgen Corporation USA",
    "LE-UK01": "Nexgen Holdings UK Ltd",
    "LE-SG01": "Nexgen APAC Pte Ltd",
    "LE-AU01": "Nexgen Australia Pty Ltd",
    "LE-DE01": "Nexgen GmbH",
    "LE-CA01": "Nexgen Canada Inc",
}


def slug(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", text.lower()).strip("_")


def write_doc(path: str, meta: dict, content: str):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        f.write("---\n")
        for k, v in meta.items():
            f.write(f'{k}: "{v}"\n')
        f.write("---\n\n")
        f.write(content)
    logger.info(f"  Wrote: {os.path.relpath(path, BASE_DIR)}")


def load_data():
    df_plan = pd.read_csv(PLAN_PATH)
    df_gl   = pd.read_csv(GL_PATH)
    df_rc   = pd.read_csv(RC_PATH)[["RC_Code", "Department"]]
    df_plan = df_plan.merge(df_rc, on="RC_Code", how="left")
    return df_plan, df_gl


def expense_table(df: pd.DataFrame) -> str:
    grp = df.groupby("Expense_Category").agg(
        Budget=("Budget_USD", "sum"),
        Actuals=("Actuals_USD", "sum"),
        Forecast=("Forecast_USD", "sum"),
    ).reset_index()
    grp["Var_USD"] = grp["Actuals"] - grp["Budget"]
    grp["Var_Pct"] = (grp["Var_USD"] / grp["Budget"].replace(0, 1) * 100).round(1)
    grp["Status"]  = grp["Var_Pct"].apply(
        lambda x: "Unfavorable" if x > 5 else ("Favorable" if x < -5 else "On Track")
    )
    grp = grp.sort_values("Var_USD", ascending=False)

    lines = [
        "| Expense Category | Budget ($) | Actuals ($) | Variance ($) | Variance % | Status |",
        "|---|---|---|---|---|---|",
    ]
    for _, r in grp.iterrows():
        sign = "+" if r["Var_USD"] >= 0 else ""
        lines.append(
            f"| {r['Expense_Category']} | {r['Budget']:,.0f} | {r['Actuals']:,.0f} | "
            f"{sign}{r['Var_USD']:,.0f} | {sign}{r['Var_Pct']:.1f}% | {r['Status']} |"
        )
    tb = grp["Budget"].sum()
    ta = grp["Actuals"].sum()
    tv = ta - tb
    tp = tv / tb * 100 if tb else 0
    sign = "+" if tv >= 0 else ""
    lines.append(
        f"| **TOTAL** | **{tb:,.0f}** | **{ta:,.0f}** | "
        f"**{sign}{tv:,.0f}** | **{sign}{tp:.1f}%** | |"
    )
    return "\n".join(lines)


def top_vendors(df_gl: pd.DataFrame, dept: str, fy: int, quarter: str) -> str:
    df = df_gl[
        (df_gl["Department"] == dept) &
        (df_gl["Fiscal_Year"] == fy) &
        (df_gl["Quarter"] == quarter)
    ]
    if df.empty:
        return "No GL transaction detail available for this period."
    top = (df.groupby(["Vendor_Name", "Expense_Category"])["Amount_USD"]
             .sum().reset_index()
             .sort_values("Amount_USD", ascending=False)
             .head(5))
    lines = []
    for i, (_, r) in enumerate(top.iterrows(), 1):
        lines.append(f"{i}. {r['Vendor_Name']} — ${r['Amount_USD']:,.0f} ({r['Expense_Category']})")
    return "\n".join(lines)


def outlook_table(df: pd.DataFrame) -> str:
    """Budget/Forecast table for future quarters (no actuals booked yet)."""
    grp = df.groupby("Expense_Category").agg(
        Budget=("Budget_USD", "sum"),
        Forecast=("Forecast_USD", "sum"),
    ).reset_index()
    grp["Var_USD"] = grp["Forecast"] - grp["Budget"]
    grp["Var_Pct"] = (grp["Var_USD"] / grp["Budget"].replace(0, 1) * 100).round(1)
    grp = grp.sort_values("Budget", ascending=False)

    lines = [
        "| Expense Category | Budget ($) | Forecast ($) | Forecast vs Budget ($) | Forecast vs Budget % |",
        "|---|---|---|---|---|",
    ]
    for _, r in grp.iterrows():
        sign = "+" if r["Var_USD"] >= 0 else ""
        lines.append(
            f"| {r['Expense_Category']} | {r['Budget']:,.0f} | {r['Forecast']:,.0f} | "
            f"{sign}{r['Var_USD']:,.0f} | {sign}{r['Var_Pct']:.1f}% |"
        )
    tb, tf = grp["Budget"].sum(), grp["Forecast"].sum()
    tv = tf - tb
    sign = "+" if tv >= 0 else ""
    lines.append(
        f"| **TOTAL** | **{tb:,.0f}** | **{tf:,.0f}** | **{sign}{tv:,.0f}** | "
        f"**{sign}{(tv / tb * 100 if tb else 0):.1f}%** |"
    )
    return "\n".join(lines)


def gpt_outlook_memo(client, dept, fy, quarter, tbl):
    prompt = f"""Write a forward-looking Budget & Forecast Outlook memo using this data:

DEPARTMENT: {dept}
PERIOD: FY{fy} {quarter} (FUTURE QUARTER — actuals not yet booked)
COMPANY: Crestwood Capital Group

BUDGET & FORECAST:
{tbl}

Write a professional internal planning memo with these sections:
## Executive Summary
## Planned Spend by Category
## Forecast Assumptions & Drivers
## Risks & Watch Items

350-450 words. This is a FORWARD-LOOKING outlook — do NOT mention actuals or variances vs actuals.
Ground all figures in the data above. Invent realistic planning context (planned hires, upcoming
renewals, projects in flight, seasonality). Use a professional FP&A tone."""

    return client.chat_completion(
        messages=[
            {"role": "system", "content": (
                "You are a senior FP&A analyst writing forward-looking budget outlook memos "
                "for Crestwood Capital Group. Ground every number in the data provided. "
                "Never invent actuals for future periods."
            )},
            {"role": "user", "content": prompt},
        ],
        stream=False,
        max_tokens=650,
        temperature=0.4,
    )


def gpt_quarterly_memo(client, dept, fy, quarter, tbl, vendors_text):
    prompt = f"""Write a Quarterly Department Review memo using this data:

DEPARTMENT: {dept}
PERIOD: FY{fy} {quarter}
COMPANY: Crestwood Capital Group

EXPENSE PERFORMANCE:
{tbl}

TOP VENDORS:
{vendors_text}

Write a professional internal management memo with these sections:
## Executive Summary
## Expense Performance by Category
## Key Variances & Commentary
## Key Vendors
## Risks & Outlook

400-500 words. Ground all figures in the data above. Invent realistic business context for variances (headcount changes, new projects, vendor renegotiations, seasonal patterns). Use a professional CFO-report tone."""

    return client.chat_completion(
        messages=[
            {"role": "system", "content": (
                "You are a senior financial analyst writing internal management commentary memos "
                "for Crestwood Capital Group. Ground every number in the data provided. "
                "Invent plausible business context for variances."
            )},
            {"role": "user", "content": prompt},
        ],
        stream=False,
        max_tokens=700,
        temperature=0.4,
    )


def gpt_vendor_profile(client, vendor_name, total_spend, top_depts, top_category):
    prompt = f"""Write a vendor profile for Crestwood Capital Group's procurement records:

VENDOR: {vendor_name}
TOTAL SPEND FY2025-2026: ${total_spend:,.0f}
PRIMARY CATEGORY: {top_category}
TOP DEPARTMENTS: {top_depts}

Write an internal vendor profile with sections:
## Overview
## Services Provided
## Spend Summary
## Contract & Relationship Notes

150-200 words. Invent realistic contract terms, SLAs, and relationship context."""

    return client.chat_completion(
        messages=[
            {"role": "system", "content": (
                "You are writing internal vendor profiles for Crestwood Capital Group's procurement team."
            )},
            {"role": "user", "content": prompt},
        ],
        stream=False,
        max_tokens=350,
        temperature=0.4,
    )


# ══════════════════════════════════════════════════════════════════════════════
# Static Policy Documents
# ══════════════════════════════════════════════════════════════════════════════

POLICY_DOCS = {
    "travel_expense_policy.md": {
        "meta": {"doc_type": "expense_policy", "department": "All", "fiscal_year": "", "quarter": "", "title": "Travel & Entertainment (T&E) Policy"},
        "content": """# Travel & Entertainment (T&E) Policy
**Crestwood Capital Group | Finance Policy FP-001**
**Effective: January 1, 2025 | Reviewed Annually**

## Purpose
This policy governs all business travel, meals, and entertainment expenses incurred by employees of Crestwood Capital Group and its subsidiaries.

## Airline & Accommodation Standards
- **Domestic flights:** Economy class for flights under 5 hours. Business class permitted for flights over 5 hours with VP+ approval.
- **International flights:** Business class permitted for flights over 8 hours. Must book at least 14 days in advance.
- **Hotels:** Maximum nightly rate of $250 (domestic) / $350 (international). Exceptions require Director approval.
- **Rental cars:** Economy or compact class only. SUVs require prior approval.

## Per Diem Rates
- **Domestic (US):** $75/day meals and incidentals
- **UK / Western Europe:** $95/day
- **APAC (SG, AU):** $90/day
- **No receipts required** for expenses under $25 per transaction.

## Meals & Entertainment
- **Business meals (client-facing):** Up to $150/person with manager approval; $250/person with Director approval.
- **Team meals (internal):** Up to $50/person. Must have a clear business purpose.
- **Alcohol:** May be included in client entertainment; not reimbursable as a standalone expense.

## Approval Thresholds
| Expense Amount | Approver Required |
|---|---|
| Under $500 | Direct Manager |
| $500 – $2,500 | Department Director |
| $2,500 – $10,000 | VP Finance |
| Over $10,000 | CFO |

## Submission Requirements
- All expense reports must be submitted within 30 days of the expense date.
- Receipts required for all expenses over $25.
- Out-of-policy expenses require written justification and VP Finance pre-approval.

## Non-Reimbursable Items
Personal entertainment, fines and penalties, first-class upgrades (unless pre-approved), mini-bar charges, and personal care items.
"""
    },
    "software_procurement_policy.md": {
        "meta": {"doc_type": "expense_policy", "department": "All", "fiscal_year": "", "quarter": "", "title": "Software & Cloud Procurement Policy"},
        "content": """# Software & Cloud Procurement Policy
**Crestwood Capital Group | Finance Policy FP-002**
**Effective: January 1, 2025**

## Purpose
This policy governs the procurement of software licenses, SaaS subscriptions, and cloud infrastructure to ensure cost control, security compliance, and license optimization.

## Approval Thresholds
| Annual Contract Value | Approver |
|---|---|
| Under $5,000 | Department Manager |
| $5,000 – $25,000 | Director + IT Security review |
| $25,000 – $100,000 | VP Technology + Finance Business Partner |
| $100,000 – $500,000 | CTO + CFO |
| Over $500,000 | Board approval required |

## Preferred Vendors
The following vendors are on the Crestwood Capital Group preferred vendor list and do not require an RFP for renewals under $100,000:
- Microsoft (M365, Azure) — Primary cloud and productivity platform
- Salesforce — CRM platform
- Workday — HRIS and payroll
- Snowflake — Data warehousing
- GitHub — Source code management
- Datadog — Observability and monitoring

## Cloud Cost Controls
- All new cloud infrastructure provisioning over $2,000/month requires IT Infrastructure approval.
- Reserved instances and savings plans must be evaluated for any workload running over 6 months.
- Cloud spend forecasts are reviewed monthly in the IT Infrastructure budget review.
- Unused licenses must be reported quarterly to IT for reclamation.

## Security & Compliance Requirements
- All SaaS vendors must complete a vendor security assessment before contract signing.
- Vendors handling personal data require a Data Processing Agreement (DPA).
- Multi-factor authentication (MFA) must be supported by all procured tools.

## Contract Terms
- Minimum contract term: 1 year for software over $10,000/year.
- Auto-renewal clauses must be flagged during procurement review.
- Cancellation notice periods must not exceed 90 days.
"""
    },
    "professional_services_policy.md": {
        "meta": {"doc_type": "expense_policy", "department": "All", "fiscal_year": "", "quarter": "", "title": "Professional Services Engagement Policy"},
        "content": """# Professional Services Engagement Policy
**Crestwood Capital Group | Finance Policy FP-003**
**Effective: January 1, 2025**

## Purpose
This policy governs the engagement of external consultants, advisors, legal counsel, and other professional services providers.

## Definition
Professional Services include: management consulting, legal advisory, audit and assurance, tax advisory, IT consulting, recruitment (contingency and retained), and financial advisory services.

## Approval Requirements
| Engagement Value | Process Required |
|---|---|
| Under $10,000 | Manager approval + Purchase Order |
| $10,000 – $50,000 | Director approval + signed SOW |
| $50,000 – $200,000 | VP approval + Finance sign-off + signed SOW |
| Over $200,000 | C-Suite approval + Legal review + Finance sign-off |

## Statement of Work (SOW) Requirements
All engagements over $10,000 must have a signed SOW that includes:
- Clear scope of work and deliverables
- Fixed timeline with milestones
- Rate card or fixed fee structure
- Intellectual property ownership clause
- Termination clause (max 30-day notice for time-and-materials)

## Rate Card Benchmarks (FY2025)
| Consultant Level | Daily Rate Benchmark |
|---|---|
| Junior Analyst | $800 – $1,200 |
| Senior Consultant | $1,500 – $2,200 |
| Manager / Principal | $2,500 – $3,500 |
| Partner / Director | $4,000 – $6,000 |

Rates above benchmark require CFO approval and documented justification.

## Preferred Firms
Crestwood Capital Group maintains preferred engagements with: Deloitte (audit), PwC (tax advisory), Kirkland & Ellis (M&A legal), and Korn Ferry (executive search).

## Contractor vs Employee Classification
All engagements over 6 months must be reviewed by HR and Legal to assess worker classification risk under applicable employment law.
"""
    },
    "vendor_management_policy.md": {
        "meta": {"doc_type": "expense_policy", "department": "All", "fiscal_year": "", "quarter": "", "title": "Vendor Management & Preferred Vendor Policy"},
        "content": """# Vendor Management & Preferred Vendor Policy
**Crestwood Capital Group | Finance Policy FP-004**
**Effective: January 1, 2025**

## Purpose
To ensure consistent, cost-effective, and compliant vendor relationships across all Crestwood Capital Group entities.

## Vendor Onboarding
All new vendors must complete:
1. Vendor registration form (legal entity, bank details, tax ID)
2. Security assessment (for technology vendors)
3. Anti-bribery and sanctions screening
4. Certificate of Insurance (minimum $1M general liability)

New vendor setup requires Finance Operations approval.

## Preferred Vendor Program
Preferred vendors have pre-negotiated rates and do not require competitive RFP for standard renewals. Current preferred vendors by category:

**People & HR:** ADP Payroll Services, Workday, LinkedIn Talent Solutions, Korn Ferry
**Technology:** Microsoft, AWS, Salesforce, Snowflake, GitHub, Datadog
**Professional Services:** Deloitte, PwC, Kirkland & Ellis
**Insurance:** Aetna (health), MetLife (life/disability), Cigna (supplemental)
**Facilities:** CBRE (property management)

## Competitive Sourcing Requirements
| Annual Spend | Requirement |
|---|---|
| Under $25,000 | 2 quotes preferred |
| $25,000 – $100,000 | 3 formal quotes required |
| Over $100,000 | Full RFP process required |

Waivers available for sole-source situations with VP + Finance approval.

## Vendor Performance Reviews
Vendors with annual spend over $500,000 receive an annual performance review covering: SLA compliance, cost vs benchmark, relationship quality, and strategic alignment.

## Payment Terms
Standard payment terms are Net 30. Early payment discounts (2/10 Net 30) must be evaluated against working capital needs by Finance Operations.
"""
    },
    "people_costs_policy.md": {
        "meta": {"doc_type": "expense_policy", "department": "All", "fiscal_year": "", "quarter": "", "title": "People Costs & Headcount Policy"},
        "content": """# People Costs & Headcount Policy
**Crestwood Capital Group | Finance Policy FP-005**
**Effective: January 1, 2025**

## Scope
People Costs include: base salaries, bonuses, payroll taxes, benefits (health, dental, vision, life/disability), employer pension/401k contributions, recruitment fees, and training & development.

## Headcount Approval Process
All new headcount (permanent or contract over 6 months) requires:
1. **Headcount Request Form** submitted by the hiring manager
2. **Department Director** approval
3. **Finance Business Partner** confirmation of budget availability
4. **CHRO** approval for roles at VP level and above
5. **CEO** approval for new C-Suite roles

Backfill positions (replacing a departed employee at the same level) follow an expedited process requiring Director + Finance approval only.

## Compensation Bands (FY2025)
| Level | Base Salary Range (USD) |
|---|---|
| Analyst / Associate | $65,000 – $95,000 |
| Senior Analyst / Specialist | $90,000 – $130,000 |
| Manager | $120,000 – $165,000 |
| Senior Manager / Principal | $155,000 – $210,000 |
| Director | $200,000 – $280,000 |
| VP | $275,000 – $380,000 |
| SVP / EVP | $350,000 – $500,000 |

Offers above band midpoint require CHRO + CFO approval.

## Benefits Costs (Budgeting Assumptions)
- Employer benefits load (health + dental + vision): 18% of base salary
- Payroll taxes (FICA + FUTA, US entities): 8.5% of base salary
- 401k match: up to 4% of base salary
- Total fully-loaded cost multiplier: approximately 1.30x base salary

## Training & Development
- Individual training budget: $1,500/year per employee
- Conferences and external courses over $2,500 require Director approval
- Certifications with annual fees over $500 require manager approval

## Headcount Freeze Protocols
During periods of budget constraint, the CFO may declare a Headcount Freeze. During a freeze: all open roles are paused, backfills over $150K require CFO approval, and contractor conversions are suspended.
"""
    },
    "budget_planning_guidelines.md": {
        "meta": {"doc_type": "expense_policy", "department": "All", "fiscal_year": "", "quarter": "", "title": "Budget Planning & Forecasting Guidelines"},
        "content": """# Budget Planning & Forecasting Guidelines
**Crestwood Capital Group | Finance Policy FP-006**
**Effective: FY2025 Planning Cycle**

## Annual Budget Cycle
| Milestone | Timeline |
|---|---|
| Finance distributes templates | September 1 |
| Department submissions due | October 1 |
| Finance consolidation & review | October 1 – 15 |
| Management review & challenge | October 15 – 31 |
| Board budget approval | November 15 |
| Budget locked for FY | December 1 |

## Rolling Forecast
Crestwood Capital Group maintains a rolling quarterly forecast (Q+3 outlook):
- **Reforecast submitted:** 2 weeks after each quarter close
- **Departments must reforecast** if actual spend diverges from budget by more than 10% in any expense category
- Finance Business Partners issue reforecast packages to each department

## Variance Thresholds & Escalation
| Variance vs Budget | Action Required |
|---|---|
| < 5% | No action required; note in quarterly review |
| 5% – 10% | Department Director must provide written commentary |
| 10% – 20% | Finance Business Partner review + corrective action plan |
| > 20% | VP Finance escalation + CFO briefing required |

Favorable variances (under-spend) exceeding 15% must also be explained — unplanned savings may be subject to budget realignment.

## Budget Transfer Policy
- Transfers within the same expense category: Director approval
- Transfers across expense categories: VP Finance approval
- Transfers across departments: CFO approval
- No budget may be transferred to People Costs without CHRO sign-off

## Zero-Based Budgeting
The following departments operate on a zero-based budget cycle (every 3 years, justifying all spend from scratch): Marketing, Professional Services, and Facilities & Real Estate.
"""
    },
    "expense_approval_matrix.md": {
        "meta": {"doc_type": "expense_policy", "department": "All", "fiscal_year": "", "quarter": "", "title": "Expense Approval Authority Matrix"},
        "content": """# Expense Approval Authority Matrix
**Crestwood Capital Group | Finance Policy FP-007**
**Effective: January 1, 2025**

## Purpose
This matrix defines who has authority to approve operational expenses across all Crestwood Capital Group entities.

## Recurring Operating Expenses (within approved budget)
| Transaction Value | Approver |
|---|---|
| Under $1,000 | Employee's Direct Manager |
| $1,000 – $5,000 | Department Director |
| $5,000 – $25,000 | VP of the Business Unit |
| $25,000 – $100,000 | CFO |
| Over $100,000 | CEO + CFO dual approval |

## Capital Expenditure (CapEx)
| CapEx Amount | Approver |
|---|---|
| Under $10,000 | VP + Finance BP |
| $10,000 – $100,000 | CFO |
| Over $100,000 | Board approval |

## Unbudgeted Spend (not in approved annual budget)
All unbudgeted spend requires one level of approval higher than the standard matrix above, plus Finance Business Partner sign-off confirming available funds.

## Emergency Spend Authorization
In urgent situations, the CFO may grant emergency spend authorization of up to $250,000 with CEO notification within 24 hours. Such authorizations must be ratified at the next scheduled board/executive review.

## Invoice Payment Authorization
- Invoices must match an approved Purchase Order within 5% tolerance
- Invoices with no PO up to $2,500: Finance Operations can approve
- Invoices with no PO over $2,500: Department Director + Finance Operations required

## Signature Authority by Role
| Role | Maximum Signing Authority |
|---|---|
| Manager | $5,000 |
| Director | $25,000 |
| VP | $100,000 |
| SVP / EVP | $250,000 |
| CFO | $1,000,000 |
| CEO | $5,000,000 |
| Board | Unlimited |
"""
    },
    "marketing_event_policy.md": {
        "meta": {"doc_type": "expense_policy", "department": "Marketing - Americas", "fiscal_year": "", "quarter": "", "title": "Marketing & Events Expense Policy"},
        "content": """# Marketing & Events Expense Policy
**Crestwood Capital Group | Finance Policy FP-008**
**Effective: January 1, 2025**

## Scope
This policy covers all marketing program spend including: digital advertising, content production, events and conferences, sponsorships, trade shows, PR and analyst relations, and branded merchandise.

## Budget Ownership
Marketing budget is managed by the Chief Marketing Officer (CMO) with regional allocation to:
- Marketing - Americas (VP Marketing, Americas)
- Marketing - EMEA (VP Marketing, EMEA)
- Marketing - APAC (VP Marketing, APAC)

Each regional VP owns their budget and approves spend within their allocation.

## Events Spend Guidelines
| Event Type | Budget Per Event | Approver |
|---|---|---|
| Internal team events (< 20 people) | Up to $5,000 | Manager |
| Customer roundtables / dinners | Up to $15,000 | Director |
| Regional field events | Up to $75,000 | VP Marketing |
| National conferences (hosted) | Up to $250,000 | CMO + CFO |
| Global events (e.g., annual summit) | Over $250,000 | CMO + CEO + CFO |

## Sponsorship Policy
- Sponsorships under $25,000: CMO approval
- Sponsorships $25,000 – $100,000: CMO + CFO approval
- Sponsorships over $100,000: CEO + CFO approval
- All sponsorships must have a documented business case with expected lead generation or brand KPIs

## Digital Advertising
- Agency retainers over $50,000/month require CFO approval
- Programmatic and paid search budgets are managed monthly with weekly reporting to the CMO
- Attribution tracking is mandatory for all digital spend over $10,000/month

## Swag & Branded Merchandise
- Individual item value cap: $75/person
- Annual swag budget managed by Marketing Operations
- All merchandise must be approved by Brand team

## Lead Generation & Demand Gen Programs
Finance requires ROI reporting on all demand generation programs over $50,000: pipeline generated, cost-per-lead, and conversion to opportunity within 90 days.
"""
    },
}


# ══════════════════════════════════════════════════════════════════════════════
# Main
# ══════════════════════════════════════════════════════════════════════════════

def main():
    logger.info("=" * 65)
    logger.info("FinSight 2.0 — Corpus Generation")
    logger.info("=" * 65)

    client = AzureOpenAIClient()
    df_plan, df_gl = load_data()

    # Filter to selected RCs and FY2025-2026
    df_plan_filtered = df_plan[
        (df_plan["RC_Code"].isin(SELECTED_RCS)) &
        (df_plan["Fiscal_Year"].isin(FISCAL_YEARS))
    ].copy()
    df_gl_filtered = df_gl[
        (df_gl["Department"].isin(df_plan_filtered["Department"].dropna().unique())) &
        (df_gl["Fiscal_Year"].isin(FISCAL_YEARS))
    ].copy()

    departments = sorted(df_plan_filtered["Department"].dropna().unique())
    logger.info(f"Departments: {departments}")

    # ── 1. Quarterly department memos ────────────────────────────────────────
    logger.info(f"\n[1/3] Generating quarterly expense memos...")
    memo_dir = os.path.join(CORPUS_DIR, "generated", "memos")
    total_memos = len(departments) * len(FISCAL_YEARS) * len(QUARTERS)
    done = 0

    for dept in departments:
        for fy in FISCAL_YEARS:
            for quarter in QUARTERS:
                done += 1
                fname = f"{slug(dept)}_fy{fy}_{quarter.lower()}.md"
                fpath = os.path.join(memo_dir, fname)

                if os.path.exists(fpath):
                    logger.info(f"  [{done}/{total_memos}] Skip (exists): {fname}")
                    continue

                df_q = df_plan_filtered[
                    (df_plan_filtered["Department"] == dept) &
                    (df_plan_filtered["Fiscal_Year"] == fy) &
                    (df_plan_filtered["Quarter"] == quarter)
                ]
                if df_q.empty:
                    logger.info(f"  [{done}/{total_memos}] No data: {fname}")
                    continue

                is_future = df_q["Actuals_USD"].sum() == 0

                if is_future:
                    logger.info(f"  [{done}/{total_memos}] Generating (outlook): {fname}")
                    tbl = outlook_table(df_q)
                    try:
                        content = gpt_outlook_memo(client, dept, fy, quarter, tbl)
                    except Exception as e:
                        logger.error(f"    GPT error: {e}")
                        time.sleep(5)
                        continue
                    doc_kind    = "Budget & Forecast Outlook"
                    doc_type    = "outlook_memo"
                    tbl_heading = "Budget & Forecast Summary"
                    title       = f"Budget Outlook — {dept} FY{fy} {quarter}"
                else:
                    logger.info(f"  [{done}/{total_memos}] Generating: {fname}")
                    tbl     = expense_table(df_q)
                    vendors = top_vendors(df_gl_filtered, dept, fy, quarter)
                    try:
                        content = gpt_quarterly_memo(client, dept, fy, quarter, tbl, vendors)
                    except Exception as e:
                        logger.error(f"    GPT error: {e}")
                        time.sleep(5)
                        continue
                    doc_kind    = "Quarterly Department Review"
                    doc_type    = "quarterly_memo"
                    tbl_heading = "Expense Data Summary"
                    title       = f"Quarterly Review — {dept} FY{fy} {quarter}"

                # Prepend a structured header so the doc is self-contained
                header = (
                    f"# {doc_kind} — {dept} | FY{fy} {quarter}\n\n"
                    f"**Classification:** Internal | Management Use Only  \n"
                    f"**Department:** {dept}  \n"
                    f"**Period:** FY{fy} {quarter}  \n"
                    f"**Prepared by:** Office of Finance, Crestwood Capital Group  \n\n"
                    f"---\n\n"
                    f"## {tbl_heading}\n\n"
                    f"{tbl}\n\n"
                    f"---\n\n"
                )
                full_content = header + content

                meta = {
                    "doc_type":   doc_type,
                    "department": dept,
                    "fiscal_year": str(fy),
                    "quarter":    quarter,
                    "title":      title,
                }
                write_doc(fpath, meta, full_content)
                time.sleep(0.3)  # gentle rate limiting

    # ── 2. Vendor profiles ────────────────────────────────────────────────────
    logger.info(f"\n[2/3] Generating vendor profiles...")
    vendor_dir = os.path.join(CORPUS_DIR, "generated", "vendors")

    top_vendors_df = (
        df_gl_filtered
        .groupby(["Vendor_Name", "Expense_Category", "Department"])["Amount_USD"]
        .sum().reset_index()
    )
    top_by_spend = (
        top_vendors_df.groupby("Vendor_Name")["Amount_USD"]
        .sum().sort_values(ascending=False)
        .head(10)
    )

    for i, (vendor_name, total_spend) in enumerate(top_by_spend.items(), 1):
        fname = f"{slug(vendor_name)}.md"
        fpath = os.path.join(vendor_dir, fname)

        if os.path.exists(fpath):
            logger.info(f"  [{i}/10] Skip (exists): {fname}")
            continue

        logger.info(f"  [{i}/10] Generating: {fname}")
        vdf = top_vendors_df[top_vendors_df["Vendor_Name"] == vendor_name]
        top_cat   = vdf.groupby("Expense_Category")["Amount_USD"].sum().idxmax()
        top_depts = ", ".join(
            vdf.groupby("Department")["Amount_USD"].sum()
               .sort_values(ascending=False).head(3).index.tolist()
        )

        try:
            content = gpt_vendor_profile(client, vendor_name, total_spend, top_depts, top_cat)
        except Exception as e:
            logger.error(f"    GPT error: {e}")
            time.sleep(5)
            continue

        header = (
            f"# Vendor Profile — {vendor_name}\n\n"
            f"**Classification:** Internal | Procurement Use Only  \n"
            f"**Primary Category:** {top_cat}  \n"
            f"**Total Spend FY2025–2026:** ${total_spend:,.0f}  \n"
            f"**Top Departments:** {top_depts}  \n\n"
            f"---\n\n"
        )
        full_content = header + content

        meta = {
            "doc_type":    "vendor_profile",
            "department":  top_depts.split(",")[0].strip(),
            "fiscal_year": "2025-2026",
            "quarter":     "",
            "title":       f"Vendor Profile — {vendor_name}",
        }
        write_doc(fpath, meta, full_content)
        time.sleep(0.3)

    # ── 3. Policy documents (static, no API) ─────────────────────────────────
    logger.info(f"\n[3/3] Writing policy documents...")
    policy_dir = os.path.join(CORPUS_DIR, "policy")
    for fname, doc in POLICY_DOCS.items():
        fpath = os.path.join(policy_dir, fname)
        if os.path.exists(fpath):
            logger.info(f"  Skip (exists): {fname}")
            continue
        write_doc(fpath, doc["meta"], doc["content"])

    # ── Summary ───────────────────────────────────────────────────────────────
    n_memos   = len([f for f in os.listdir(memo_dir)   if f.endswith(".md")])
    n_vendors = len([f for f in os.listdir(vendor_dir) if f.endswith(".md")])
    n_policy  = len([f for f in os.listdir(policy_dir) if f.endswith(".md")])
    logger.info("\n" + "=" * 65)
    logger.info("Corpus generation complete!")
    logger.info(f"  Quarterly memos:   {n_memos}")
    logger.info(f"  Vendor profiles:   {n_vendors}")
    logger.info(f"  Policy documents:  {n_policy}")
    logger.info(f"  TOTAL documents:   {n_memos + n_vendors + n_policy}")
    logger.info("\nNext: python scripts/ingest_v2.py")
    logger.info("=" * 65)


if __name__ == "__main__":
    main()
