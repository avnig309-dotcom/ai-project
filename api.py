"""FastAPI frontend contract. Set SUPPORT_MODEL_PATH to a trained joblib bundle."""
from __future__ import annotations
import os
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from orchestrator import FiveAgentSystem


class QueryRequest(BaseModel):
    query: str = ""
    query_id: str | None = None
    context: dict[str, Any] | None = None

class BatchRequest(BaseModel):
    queries: list[QueryRequest] = Field(default_factory=list, max_length=500)

system: FiveAgentSystem | None = None
load_error: str | None = None

def load_model():
    global system, load_error
    path = Path(os.getenv("SUPPORT_MODEL_PATH", "agent_models/two_agent_system.joblib"))
    if path.is_file():
        try: system = FiveAgentSystem.from_joblib(str(path))
        except Exception as exc: load_error = f"{type(exc).__name__}: {exc}"
    else: load_error = f"Model bundle not found: {path}"

@asynccontextmanager
async def lifespan(app):
    load_model()
    yield

app = FastAPI(title="Intelligent Customer Support API", version="1.0.0", lifespan=lifespan)
origins = [x.strip() for x in os.getenv("CORS_ORIGINS", "http://localhost:3000,http://localhost:5173").split(",") if x.strip()]
app.add_middleware(CORSMiddleware, allow_origins=origins, allow_credentials=False, allow_methods=["GET", "POST"], allow_headers=["Content-Type"])

def require_system() -> FiveAgentSystem:
    if system is None: raise HTTPException(status_code=503, detail="Model is not ready. Train a model and configure SUPPORT_MODEL_PATH.")
    return system

@app.get("/health")
def health(): return {"status": "ok"}

@app.get("/ready")
def ready(): return {"ready": system is not None, "model_loaded": system is not None, "error": load_error}

@app.get("/api/v1/config")
def config(): return {"provider_mode": os.getenv("AI_PROVIDER_MODE", "local"), "capabilities": ["intent-classification", "tfidf-retrieval", "five-agent-orchestration", "local-audit"], "organizer_api_configured": False}

@app.post("/api/v1/analyze")
def analyze(request: QueryRequest):
    active = require_system()
    result = active._analyze(request.query)
    return result.__dict__

@app.post("/api/v1/query")
def query(request: QueryRequest):
    return require_system().process_query(request.query, context=request.context, query_id=request.query_id)

@app.post("/api/v1/batch")
def batch(request: BatchRequest):
    active = require_system()
    items = [{"query": item.query, "context": item.context, "query_id": item.query_id} for item in request.queries]
    return {"results": active.process_batch(items)}
