# FinSight AI — Full Project Context for VS Code
### Paste this into your VS Code AI chat at the start of every session

---

## What We Are Building

A production-ready **RAG (Retrieval-Augmented Generation) Financial Chatbot** called **FinSight AI**, deployed end-to-end on Microsoft Azure. This chatbot answers natural language questions over a real custom-generated internal financial dataset for a fictional company called **Nexgen Corporation**.

This is a resume project showcasing: Gen AI, RAG pattern, Azure cloud, FastAPI, Streamlit, CI/CD.

---

## The Dataset (Already Generated — DO NOT recreate)

The dataset is located at:
```
C:\Users\stein\OneDrive\Documents\DE\AI\Claude\Projects\fin_cb\
```

It is a complete corporate financial dataset for **Nexgen Corporation** with the following files:

### Fact Tables

| File | Rows | Description |
|------|------|-------------|
| `gl_data/fact_gl_transactions.csv` | 119,340 | Full General Ledger journal entries — every financial transaction posted |
| `planning_data/fact_planning_combined.csv` | 119,161 | Budget + Actuals + Forecast side by side with variance calculations |
| `planning_data/fact_actuals_planning.csv` | 119,161 | Actual spend by cost center, GL account, period |
| `planning_data/fact_budget.csv` | 119,161 | Approved budget by cost center, GL account, period |
| `planning_data/fact_forecast.csv` | 476,641 | Rolling forecasts — multiple forecast versions (Q1FC22, Q2FC22, etc.) |

### Master Data (Dimension Tables)

| File | Description |
|------|-------------|
| `master_data/dim_gl_account.csv` | Chart of accounts — GL codes, account names, categories (P&L / Balance Sheet) |
| `master_data/dim_legal_entity.csv` | Legal entities (Nexgen Corporation USA = LE-US01, etc.) |
| `master_data/dim_responsibility_center.csv` | Cost centers with manager names, departments, BU mapping |
| `master_data/dim_business_unit.csv` | Business unit hierarchy (Corporate, Regional, etc.) |
| `master_data/dim_geography.csv` | Geography hierarchy — region, country, city |
| `master_data/dim_planning_dimension.csv` | Planning dimensions linking cost centers to entities/BUs |
| `master_data/dim_vendor.csv` | Vendor master — names, categories, preferred vendor flags |

### Key Columns — fact_gl_transactions.csv
```
Journal_Entry_ID, Posting_Date, Document_Date, Fiscal_Year, Fiscal_Period,
Month, Quarter, Ledger, Company_Code, Entity_Name, RC_Code, Department,
GL_Account, Account_Name, Expense_Category, Document_Type, Document_Type_Desc,
Vendor_ID, Vendor_Name, Amount_USD, Amount_LCY, Currency,
Debit_USD, Credit_USD, Description, Reference, Posted_By, Status, Clearing_Status
```

### Key Columns — fact_planning_combined.csv
```
PlanDim_ID, RC_Code, Entity_ID, Fiscal_Year, Fiscal_Period, Quarter, Month,
GL_Account, Expense_Category,
Budget_USD, Actuals_USD, Forecast_USD,
Variance_BvA, Variance_BvA_Pct, Variance_BvF, Status
```

### Sample Questions This Chatbot Should Answer
- "What was the budget vs actuals variance for the Finance department in Q1 2022?"
- "Which responsibility center had the highest overspend last year?"
- "Show me all People Costs transactions above $10,000 in January 2022"
- "What is the total forecast for People Costs in Q2 2022?"
- "Which vendors had the most spend in the Finance cost center?"
- "What is Nexgen Corporation's total headcount cost for 2022?"
- "Which departments are at risk of going over budget?"
- "Compare budget vs forecast for GL account 6100 (Salaries & Wages)"

---

## Azure Infrastructure (Already Provisioned)

All Azure resources are live in resource group: `finsight-ai-rg` (region: eastus)

| Resource | Name | Purpose |
|----------|------|---------|
| Azure OpenAI | finsight-openai | GPT-4o (chat) + text-embedding-ada-002 (embeddings) |
| Azure AI Search | finsight-search | Vector index for RAG retrieval |
| Azure Blob Storage | finsightstorage1234 | Raw dataset storage |
| Azure App Service | finsight-ai-app | Host the web application |
| Resource Group | finsight-ai-rg | Container for all resources |

### Deployed Model Names
- Chat model deployment name: `gpt-4o`
- Embedding model deployment name: `text-embedding-ada-002`
- Azure OpenAI API version: `2024-02-01`

### Azure AI Search Index
- Index name: `finsight-index`
- Vector dimensions: 1536 (text-embedding-ada-002)
- Search type: Hybrid (vector + BM25 keyword + semantic reranking)

### Environment Variables (.env file exists at project root)
```env
AZURE_OPENAI_ENDPOINT=https://finsight-openai.openai.azure.com/
AZURE_OPENAI_API_KEY=<already set>
AZURE_OPENAI_API_VERSION=2024-02-01
AZURE_OPENAI_CHAT_DEPLOYMENT=gpt-4o
AZURE_OPENAI_EMBEDDING_DEPLOYMENT=text-embedding-ada-002
AZURE_SEARCH_ENDPOINT=https://finsight-search.search.windows.net
AZURE_SEARCH_API_KEY=<already set>
AZURE_SEARCH_INDEX_NAME=finsight-index
AZURE_STORAGE_CONNECTION_STRING=<already set>
AZURE_STORAGE_CONTAINER_NAME=financial-data
```

---

## Tech Stack

| Layer | Technology |
|-------|-----------|
| Language | Python 3.11 |
| Backend API | FastAPI + Uvicorn |
| Frontend | Streamlit |
| LLM | Azure OpenAI GPT-4o |
| Embeddings | Azure OpenAI text-embedding-ada-002 |
| Vector Store | Azure AI Search (hybrid search) |
| Blob Storage | Azure Blob Storage |
| Deployment | Azure App Service (Linux, B1) |
| CI/CD | GitHub Actions |
| RAG Framework | LangChain |
| Data Processing | Pandas, tiktoken |

---

## Project Folder Structure

```
finsight-ai/                          ← GitHub repo root
├── .env                              ← Azure credentials (NEVER commit)
├── .env.example                      ← Template (commit this)
├── .gitignore
├── requirements.txt
├── README.md
│
├── data/
│   ├── raw/                          ← Copy CSVs from fin_cb folder here
│   └── processed/                    ← Chunked text output from ingest.py
│
├── scripts/
│   ├── ingest.py                     ← STEP 1: Run this to build the index
│   └── validate_index.py             ← STEP 2: Verify the index is working
│
├── backend/
│   ├── __init__.py
│   ├── main.py                       ← FastAPI app entry point
│   ├── rag_pipeline.py               ← Core RAG logic
│   ├── azure_search.py               ← Azure AI Search client wrapper
│   ├── azure_openai_client.py        ← Azure OpenAI client wrapper
│   └── models.py                     ← Pydantic request/response models
│
├── frontend/
│   └── app.py                        ← Streamlit chat UI
│
├── tests/
│   ├── test_rag_pipeline.py
│   └── test_api.py
│
└── .github/
    └── workflows/
        └── deploy.yml                ← Auto-deploy to Azure on push to main
```

---

## Data Ingestion Strategy (How We Convert CSVs into RAG-ready Text)

Since the dataset is structured (CSVs, not documents), we convert rows into **human-readable text chunks** before embedding. This is critical — DO NOT embed raw CSV rows.

### Conversion approach for fact_gl_transactions:
Each transaction becomes a sentence like:
```
"In January 2022 (Q1), Nexgen Corporation USA (Finance department, cost center RC-0001)
posted a vendor invoice of $9,154.73 USD for Recruitment & Hiring expenses to GL account
6106, from vendor Workday Inc. Status: Posted. Reference: REF-564705."
```

### Conversion approach for fact_planning_combined:
Each planning row becomes:
```
"For GL account 6100 (Salaries & Wages, People Costs) in cost center RC-0001 (Finance),
Nexgen Corporation USA had a budget of $36,136.05 and actuals of $36,866.15 in Period 1
of 2022 (Q1). Budget vs Actuals variance: +$730.10 (2.02% overspend). Status: At Risk."
```

### Chunking rules:
- Target chunk size: ~400 tokens
- Group related rows together (same RC + period + GL account)
- Add metadata to every chunk: entity, department, fiscal_year, fiscal_period, data_type

---

## RAG Pipeline Flow

```
User question (natural language)
        ↓
[1] Embed question → text-embedding-ada-002 → 1536-dim vector
        ↓
[2] Hybrid search in Azure AI Search
    → Vector similarity + BM25 keyword match + semantic reranker
    → Return top 5 most relevant financial chunks
        ↓
[3] Build GPT-4o prompt:
    System: "You are FinSight, a financial analyst assistant for Nexgen Corporation.
             Answer questions using only the provided financial data. Always cite sources."
    Context: [paste the 5 retrieved chunks]
    History: [last 5 conversation turns]
    User: [the question]
        ↓
[4] Stream GPT-4o response
        ↓
[5] Return: answer text + source citations (entity, dept, period, data_type)
```

---

## Build Order (Files to Create, In This Exact Order)

1. `requirements.txt` — dependencies
2. `.gitignore` and `.env.example`
3. `backend/azure_openai_client.py` — OpenAI wrapper (embed + chat)
4. `backend/azure_search.py` — Search wrapper (create index + search + upload)
5. `scripts/ingest.py` — Load CSVs → convert to text → embed → index
6. `scripts/validate_index.py` — Test index with sample queries
7. `backend/models.py` — Pydantic models
8. `backend/rag_pipeline.py` — RAG orchestration
9. `backend/main.py` — FastAPI app
10. `frontend/app.py` — Streamlit chat UI
11. `tests/test_rag_pipeline.py` + `tests/test_api.py`
12. `.github/workflows/deploy.yml` — GitHub Actions CI/CD

---

## Budget Constraint
**Hard limit: $10 USD total Azure spend** (we are on the Azure free trial with $200 credits).
Always use: Free tier for Search, minimal token usage, cap max_tokens=800, batch API calls.

---

## Current Status
- ✅ Azure free trial account created and CLI logged in
- ✅ All Azure resources provisioned (OpenAI, AI Search, Blob Storage, App Service)
- ✅ Custom financial dataset already generated (Nexgen Corporation, ~900K total rows)
- ✅ Project plan finalized
- ⏳ Next step: Create GitHub repo + build ingestion pipeline (scripts/ingest.py)

---

## Resume Bullet Points (Final Goal)
```
FinSight AI — Financial RAG Chatbot (Azure)               [Live Demo] [GitHub]
• Built an end-to-end RAG chatbot over 900K+ rows of internal financial data
  (GL transactions, budgets, forecasts) for a fictional enterprise "Nexgen Corporation"
• Implemented hybrid vector + keyword search using Azure AI Search with semantic reranking
• Deployed FastAPI backend + Streamlit frontend to Azure App Service via GitHub Actions CI/CD
• Tech: Python, FastAPI, Streamlit, LangChain, Azure OpenAI (GPT-4o),
  Azure AI Search, Azure Blob Storage, Azure App Service, GitHub Actions
```

---

*Always reference this document at the start of every VS Code AI chat session.*
*Dataset path: C:\Users\stein\OneDrive\Documents\DE\AI\Claude\Projects\fin_cb*
