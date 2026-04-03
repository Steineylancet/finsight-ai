"""
FinSight AI — FastAPI Backend
Entry point for the RAG chatbot API
"""

import logging
import json
from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.util import get_remote_address
from slowapi.errors import RateLimitExceeded

from backend.models import ChatRequest, ChatResponse, SearchRequest, HealthResponse, Source
from backend.rag_pipeline import RAGPipeline

# ── Logging ──────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s"
)
logger = logging.getLogger(__name__)

# ── Rate Limiter ──────────────────────────────────────────────────────────────
limiter = Limiter(key_func=get_remote_address)

# ── App ───────────────────────────────────────────────────────────────────────
app = FastAPI(
    title="FinSight AI",
    description="RAG-powered financial chatbot for Nexgen Corporation",
    version="1.0.0",
)

app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Tighten in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Startup ───────────────────────────────────────────────────────────────────
rag_pipeline: RAGPipeline = None

@app.on_event("startup")
async def startup_event():
    global rag_pipeline
    logger.info("Initializing RAG pipeline...")
    rag_pipeline = RAGPipeline()
    logger.info("FinSight AI is ready.")


# ── Endpoints ─────────────────────────────────────────────────────────────────

@app.get("/", response_model=HealthResponse)
async def root():
    return HealthResponse(
        status="ok",
        service="FinSight AI",
        version="1.0.0"
    )


@app.get("/health", response_model=HealthResponse)
async def health():
    return HealthResponse(
        status="ok",
        service="FinSight AI",
        version="1.0.0"
    )


@app.post("/chat")
@limiter.limit("10/minute")
async def chat(request: Request, body: ChatRequest):
    """
    Main RAG chat endpoint. Streams GPT-4o response with sources.
    """
    try:
        stream, sources = rag_pipeline.run(
            question=body.question,
            conversation_history=body.conversation_history,
            stream=True,
        )

        # If fallback string returned (no chunks found)
        if isinstance(stream, str):
            return ChatResponse(
                answer=stream,
                sources=[],
                question=body.question,
            )

        # Serialize sources once
        sources_payload = json.dumps([s.model_dump() for s in sources])

        def generate():
            # First chunk: send sources metadata
            yield f"data: {json.dumps({'type': 'sources', 'sources': json.loads(sources_payload)})}\n\n"

            # Stream GPT-4o response
            for chunk in stream:
                delta = chunk.choices[0].delta
                if delta and delta.content:
                    yield f"data: {json.dumps({'type': 'token', 'content': delta.content})}\n\n"

            yield "data: [DONE]\n\n"

        return StreamingResponse(generate(), media_type="text/event-stream")

    except Exception as e:
        logger.error(f"Chat endpoint error: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Internal error: {str(e)}")


@app.get("/search")
@limiter.limit("20/minute")
async def search(request: Request, q: str, top_k: int = 5):
    """
    Debug endpoint: run a raw hybrid search and return raw chunks.
    """
    try:
        query_vector = rag_pipeline.openai_client.get_embedding(q)
        results = rag_pipeline.search_client.hybrid_search(
            query_text=q,
            query_vector=query_vector,
            top_k=top_k,
        )
        return {"query": q, "results": results}
    except Exception as e:
        logger.error(f"Search endpoint error: {e}")
        raise HTTPException(status_code=500, detail=str(e))
