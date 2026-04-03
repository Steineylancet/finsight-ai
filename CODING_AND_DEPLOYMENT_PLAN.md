# FinSight AI — Full Coding & Deployment Plan
### End-to-End: Code → Test → Deploy → Production Ready

---

## Overview

```
Phase 1: Project Setup          → GitHub repo, venv, folder structure
Phase 2: Data Pipeline          → Download → Chunk → Embed → Index
Phase 3: Backend (FastAPI)      → RAG logic, API endpoints, Azure clients
Phase 4: Frontend (Streamlit)   → Chat UI, streaming, citations
Phase 5: Testing                → Unit + integration tests
Phase 6: Deployment             → Azure App Service + GitHub Actions CI/CD
Phase 7: Production Hardening   → Logging, error handling, rate limiting, monitoring
```

Total estimated time: **6–8 focused days** (a few hours each day)

---

## PHASE 1 — Project Setup (Day 1, ~1 hour)

### 1.1 Create GitHub Repository
- Create a new public repo: `finsight-ai`
- Add a `README.md`, `.gitignore` (Python template), `LICENSE` (MIT)
- Clone it locally: `git clone https://github.com/<your-username>/finsight-ai`

### 1.2 Set Up Python Virtual Environment
```bash
cd finsight-ai
python -m venv venv
venv\Scripts\activate          # Windows
pip install -r requirements.txt
```

### 1.3 Create Folder Structure
```
finsight-ai/
├── .env                        ← Your Azure keys (never commit!)
├── .env.example                ← Template (commit this)
├── .gitignore
├── requirements.txt
├── README.md
├── data/
│   ├── raw/                    ← Raw downloaded dataset
│   └── processed/              ← Chunked text ready for indexing
├── scripts/
│   ├── ingest.py               ← Full ingestion pipeline
│   └── validate_index.py       ← Test your Azure Search index
├── backend/
│   ├── __init__.py
│   ├── main.py                 ← FastAPI app
│   ├── rag_pipeline.py         ← Core RAG logic
│   ├── azure_search.py         ← Azure AI Search client wrapper
│   ├── azure_openai_client.py  ← Azure OpenAI client wrapper
│   └── models.py               ← Pydantic request/response models
├── frontend/
│   └── app.py                  ← Streamlit chat UI
├── tests/
│   ├── test_rag_pipeline.py
│   └── test_api.py
└── .github/
    └── workflows/
        └── deploy.yml          ← GitHub Actions CI/CD
```

### 1.4 Configure .gitignore
Key entries to include:
```
.env
venv/
__pycache__/
data/raw/
*.pyc
.DS_Store
```

---

## PHASE 2 — Data Pipeline (Day 2, ~2-3 hours)

### 2.1 Download the Dataset
We'll use the **Financial News and Stock Reaction** dataset from Kaggle.
- Install Kaggle CLI: `pip install kaggle`
- Download: `kaggle datasets download -d miguelaenlle/massive-stock-news-analysis-db-for-nlpbacktests`
- Unzip into `data/raw/`

### 2.2 scripts/ingest.py — The Full Pipeline

**What this script does, step by step:**

```
Raw CSV
   ↓
[1] Load & Filter
    → Keep only rows with meaningful text (>100 chars)
    → Filter for finance/market/economy topics
    → Drop duplicates
   ↓
[2] Chunk Text
    → Split articles into ~500 token chunks
    → 50 token overlap between chunks (so context isn't lost at boundaries)
    → Tag each chunk with: source, date, title, chunk_id
   ↓
[3] Generate Embeddings
    → Call Azure OpenAI text-embedding-ada-002
    → Batch in groups of 16 (API rate limit safety)
    → Each chunk gets a 1536-dimension vector
   ↓
[4] Upload to Azure Blob Storage
    → Store raw CSV in Azure Blob (for audit/reproducibility)
   ↓
[5] Index into Azure AI Search
    → Create index schema (if not exists)
    → Upload chunks + embeddings in batches of 100
    → Index supports: vector search + keyword (BM25) + semantic reranking
```

**Azure AI Search Index Schema:**
```json
{
  "fields": [
    { "name": "id",          "type": "string",     "key": true },
    { "name": "content",     "type": "string",     "searchable": true },
    { "name": "title",       "type": "string",     "searchable": true },
    { "name": "source",      "type": "string",     "filterable": true },
    { "name": "date",        "type": "string",     "filterable": true },
    { "name": "chunk_index", "type": "int32" },
    { "name": "embedding",   "type": "vector",     "dimensions": 1536 }
  ]
}
```

### 2.3 scripts/validate_index.py
- Query the index with a test question
- Print top 3 returned chunks
- Confirm embeddings and metadata look correct

---

## PHASE 3 — Backend: FastAPI (Day 3, ~3 hours)

### 3.1 backend/azure_openai_client.py
Wrapper around Azure OpenAI SDK:
- `get_embedding(text)` → returns 1536-dim vector
- `chat_completion(messages, stream=True)` → returns streamed GPT-4o response

### 3.2 backend/azure_search.py
Wrapper around Azure AI Search SDK:
- `create_index()` → creates the index schema
- `upload_documents(docs)` → batch upload chunks
- `hybrid_search(query_vector, query_text, top_k=5)` → runs hybrid search
  - Combines vector similarity + BM25 keyword + semantic reranker
  - Returns top K chunks with scores and metadata

### 3.3 backend/rag_pipeline.py
The core logic:
```
Input: user question + conversation history
   ↓
[1] Embed the question → vector
[2] Hybrid search Azure AI Search → top 5 chunks
[3] Build prompt:
    - System: "You are a financial analyst assistant..."
    - Context: paste the 5 retrieved chunks
    - History: last 5 conversation turns
    - User: the question
[4] Call GPT-4o with streaming
[5] Return: streamed answer + list of source citations
```

### 3.4 backend/models.py
Pydantic models for request/response validation:
```python
class ChatRequest:
    question: str
    conversation_history: list[dict]

class ChatResponse:
    answer: str
    sources: list[Source]

class Source:
    title: str
    date: str
    chunk: str
```

### 3.5 backend/main.py — FastAPI App
Endpoints:

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/` | Health check |
| GET | `/health` | Returns service status |
| POST | `/chat` | Main RAG chat (streaming) |
| GET | `/search?q=query` | Raw search (debug) |

Key features:
- **CORS middleware** so Streamlit can call it
- **Streaming responses** using `StreamingResponse`
- **Error handling** with proper HTTP status codes
- **Request validation** via Pydantic

---

## PHASE 4 — Frontend: Streamlit (Day 4, ~2 hours)

### 4.1 frontend/app.py — Chat UI

**Features:**
- Clean chat interface with message bubbles
- Streaming response display (text appears word by word)
- **Source citations** shown as expandable cards below each answer
- Conversation history maintained in session state
- Sidebar with: app info, example questions, "Clear chat" button
- Loading spinner while waiting for response

**UI Layout:**
```
┌─────────────────────────────────────────────┐
│  🏦 FinSight AI                      [Clear]│
│  Your Financial Knowledge Assistant         │
├──────────┬──────────────────────────────────┤
│          │                                  │
│ Sidebar  │  Chat Messages Area              │
│          │                                  │
│ About    │  User: What happened to...       │
│          │  Bot: Based on the data...       │
│ Example  │                                  │
│ Questions│  📎 Sources (expandable)         │
│          │  • Reuters, Jan 2023             │
│          │  • Bloomberg, Mar 2023           │
│          │                                  │
│          ├──────────────────────────────────│
│          │  [Type your question here...] ▶  │
└──────────┴──────────────────────────────────┘
```

---

## PHASE 5 — Testing (Day 5, ~2 hours)

### 5.1 tests/test_rag_pipeline.py
- Test embedding generation returns correct dimensions
- Test search returns results for known financial queries
- Test prompt construction includes all context chunks
- Test response contains source citations

### 5.2 tests/test_api.py
- Test `/health` returns 200
- Test `/chat` with a valid question returns streamed response
- Test `/chat` with empty question returns 422 validation error
- Test `/chat` handles Azure API errors gracefully

### 5.3 Run Tests
```bash
pip install pytest pytest-asyncio
pytest tests/ -v
```

---

## PHASE 6 — Deployment (Day 6, ~2-3 hours)

### 6.1 Prepare for Deployment

**Create startup.sh for Azure App Service:**
```bash
#!/bin/bash
cd /home/site/wwwroot
pip install -r requirements.txt
uvicorn backend.main:app --host 0.0.0.0 --port 8000
```

**Set environment variables on Azure App Service:**
```powershell
az webapp config appsettings set \
  --name finsight-ai-app \
  --resource-group finsight-ai-rg \
  --settings \
    AZURE_OPENAI_ENDPOINT="..." \
    AZURE_OPENAI_API_KEY="..." \
    AZURE_SEARCH_ENDPOINT="..." \
    AZURE_SEARCH_API_KEY="..." \
    AZURE_STORAGE_CONNECTION_STRING="..."
```

### 6.2 .github/workflows/deploy.yml — GitHub Actions CI/CD

**Trigger:** Push to `main` branch
**Pipeline:**
```
on: push to main
   ↓
[1] Checkout code
[2] Set up Python 3.11
[3] Install dependencies
[4] Run tests (pytest)
[5] If tests pass → Deploy to Azure App Service
[6] Post deployment URL as comment
```

**Secrets needed in GitHub repo settings:**
- `AZURE_WEBAPP_PUBLISH_PROFILE` (download from Azure Portal)
- `AZURE_OPENAI_API_KEY`
- `AZURE_SEARCH_API_KEY`
- `AZURE_STORAGE_CONNECTION_STRING`

### 6.3 Deploy Streamlit Frontend
Two options:
- **Option A**: Run Streamlit on the same App Service (different port, proxied)
- **Option B**: Deploy Streamlit to **Streamlit Cloud** (free, easiest for resume) ← Recommended

For Streamlit Cloud:
1. Push code to GitHub
2. Go to share.streamlit.io
3. Connect GitHub repo
4. Set environment variables in Streamlit Cloud dashboard
5. Get a public URL like `https://finsight-ai.streamlit.app`

---

## PHASE 7 — Production Hardening (Day 7, ~2 hours)

### 7.1 Logging
```python
import logging
# Structured logging for every request
# Log: question, search latency, GPT latency, total latency, error details
```

### 7.2 Error Handling
- Azure OpenAI rate limit → retry with exponential backoff
- Azure Search timeout → fallback response
- Empty search results → "I don't have information on that" graceful response
- Invalid input → 422 with clear error message

### 7.3 Rate Limiting
```python
from slowapi import Limiter
# Limit: 10 requests/minute per IP
# Prevents runaway Azure API costs
```

### 7.4 Cost Controls
- Cap max tokens per response: `max_tokens=800`
- Limit conversation history to last 5 turns
- Limit search to top 5 chunks (not 10+)
- Add request logging to monitor Azure spend

### 7.5 README Polish (for recruiters)
Your README must have:
- Project title + one-line description
- Architecture diagram (image)
- Live demo link
- Tech stack badges
- Setup instructions
- Example questions
- Screenshots of the UI
- Your name + LinkedIn

---

## Summary: File-by-File Build Order

| Order | File | What it does |
|-------|------|-------------|
| 1 | `requirements.txt` | All dependencies |
| 2 | `.env` / `.env.example` | Azure credentials |
| 3 | `backend/azure_openai_client.py` | OpenAI wrapper |
| 4 | `backend/azure_search.py` | Search wrapper |
| 5 | `scripts/ingest.py` | Data pipeline |
| 6 | `scripts/validate_index.py` | Test the index |
| 7 | `backend/models.py` | Pydantic models |
| 8 | `backend/rag_pipeline.py` | Core RAG logic |
| 9 | `backend/main.py` | FastAPI app |
| 10 | `frontend/app.py` | Streamlit UI |
| 11 | `tests/test_*.py` | Tests |
| 12 | `.github/workflows/deploy.yml` | CI/CD pipeline |

---

## What to Say on Your Resume

```
FinSight AI — Financial RAG Chatbot                          [Live Demo] [GitHub]
• Built an end-to-end Retrieval-Augmented Generation (RAG) chatbot on Azure that answers
  natural language questions from 50K+ financial news articles
• Implemented hybrid vector + keyword search using Azure AI Search with semantic reranking,
  achieving <2s response latency
• Deployed FastAPI backend + Streamlit frontend to Azure App Service with CI/CD via GitHub Actions
• Tech: Python, FastAPI, Streamlit, LangChain, Azure OpenAI (GPT-4o), Azure AI Search,
  Azure Blob Storage, Azure App Service, GitHub Actions
```

---

*FinSight AI | Coding & Deployment Plan v1.0*
