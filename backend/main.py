"""
FinSight AI — FastAPI Backend

/chat   streams answers (SSE) from the LangGraph agent: document RAG, the SQL
        agent and the three analytical flows all go through one graph.
/agent  same graph, single JSON response — used by evals and scripts.
"""

import json
import logging
import os
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.errors import RateLimitExceeded
from slowapi.util import get_remote_address

from backend.data_loader import DataLoader
from backend.graph.catalog import DEPARTMENTS
from backend.graph.llm import FINAL_ANSWER_TAG
from backend.graph.router import build_agent_graph
from backend.models import ChatRequest, HealthResponse
from backend.rag_pipeline import RAGPipeline

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s"
)
logger = logging.getLogger(__name__)

APP_VERSION = "3.0.0"
GENERIC_ERROR = "Something went wrong answering that — please try again."

rag_pipeline: RAGPipeline = None
agent_graph = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global rag_pipeline, agent_graph
    logger.info("Initializing RAG pipeline...")
    rag_pipeline = RAGPipeline()
    logger.info("Loading financial data for the agentic layer...")
    loader = DataLoader.get()
    missing = set(loader.departments()) ^ set(DEPARTMENTS)
    if missing:
        logger.warning(f"Department catalog out of sync with data: {sorted(missing)}")
    logger.info("Compiling LangGraph agent...")
    agent_graph = build_agent_graph(rag_pipeline)
    logger.info(f"FinSight AI v{APP_VERSION} is ready.")
    yield


limiter = Limiter(key_func=get_remote_address)

app = FastAPI(
    title="FinSight AI",
    description="Agentic FP&A assistant for Crestwood Capital Group",
    version=APP_VERSION,
    lifespan=lifespan,
)
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["GET", "POST"],
    allow_headers=["Content-Type"],
)

_static_dir = os.path.join(os.path.dirname(__file__), "static")
app.mount("/static", StaticFiles(directory=_static_dir), name="static")


def _initial_state(body: ChatRequest) -> dict:
    return {
        "query": body.question,
        "conversation_history": [t.model_dump() for t in body.conversation_history],
    }


def _sse(payload) -> str:
    return f"data: {json.dumps(payload, default=str)}\n\n"


# ── Endpoints ─────────────────────────────────────────────────────────────────

@app.get("/", include_in_schema=False)
async def root():
    return FileResponse(os.path.join(_static_dir, "index.html"))


@app.get("/health", response_model=HealthResponse)
async def health():
    return HealthResponse(status="ok", service="FinSight AI", version=APP_VERSION)


@app.post("/chat")
@limiter.limit("10/minute")
async def chat(request: Request, body: ChatRequest):
    """
    Streams Server-Sent Events:
      mode    {mode, intent}     which path answered (may be refined once)
      sources {sources: [...]}   document citations
      table   {content}          markdown table built from tool output
      token   {content}          answer text, streamed
      error   {message}          user-facing error
      done    {sql_query}        tool calls made, for transparency
    """
    state = _initial_state(body)

    async def events():
        final_state, streamed_text = {}, False
        try:
            async for namespace, mode, chunk in agent_graph.astream(
                state, stream_mode=["messages", "custom", "values"], subgraphs=True
            ):
                if mode == "custom":
                    streamed_text |= chunk.get("type") == "token"
                    yield _sse(chunk)
                elif mode == "messages":
                    msg, meta = chunk
                    if FINAL_ANSWER_TAG in (meta.get("tags") or []) and msg.content:
                        streamed_text = True
                        yield _sse({"type": "token", "content": msg.content})
                elif mode == "values" and not namespace:
                    final_state = chunk
        except Exception as e:
            logger.error(f"/chat stream failed: {e}", exc_info=True)
            yield _sse({"type": "error", "message": GENERIC_ERROR})
            yield "data: [DONE]\n\n"
            return

        if final_state.get("error"):
            yield _sse({"type": "error", "message": final_state["error"]})
        elif not streamed_text and final_state.get("narrative"):
            yield _sse({"type": "token", "content": final_state["narrative"]})
        yield _sse({"type": "done", "sql_query": final_state.get("sql_query")})
        yield "data: [DONE]\n\n"

    return StreamingResponse(events(), media_type="text/event-stream",
                             headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"})


@app.post("/agent")
@limiter.limit("10/minute")
async def agent(request: Request, body: ChatRequest):
    """Same graph as /chat, returned as one JSON object."""
    try:
        result = await agent_graph.ainvoke(_initial_state(body))
    except Exception as e:
        logger.error(f"/agent failed: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=GENERIC_ERROR)
    return {
        "mode": result.get("mode_label"),
        "intent": result.get("intent"),
        "standalone_query": result.get("standalone_query"),
        "table": result.get("formatted_table"),
        "narrative": result.get("narrative"),
        "sources": result.get("sources") or [],
        "sql_query": result.get("sql_query"),
        "error": result.get("error"),
    }


@app.get("/search")
@limiter.limit("20/minute")
async def search(request: Request, q: str, top_k: int = 5):
    """Debug endpoint: raw hybrid search results."""
    try:
        query_vector = rag_pipeline.openai_client.get_embedding(q)
        results = rag_pipeline.search_client.hybrid_search(
            query_text=q, query_vector=query_vector, top_k=min(max(top_k, 1), 20),
        )
        return {"query": q, "results": results}
    except Exception as e:
        logger.error(f"Search endpoint error: {e}")
        raise HTTPException(status_code=500, detail=GENERIC_ERROR)
