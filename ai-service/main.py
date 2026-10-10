"""
Argus AI Service — Phase 1 stub.

This service will eventually own the whole perception -> retrieval pipeline
described in the PRD (frame sampling, detection, tracking, embeddings,
Knowledge Graph writes, hybrid retrieval, RAG). For Phase 1 it only exposes
the same HTTP contract the Spring Boot backend will keep calling later, but
returns canned/deterministic data so the full request path
(React -> Spring Boot -> Python -> back to chat) works end-to-end.

Run locally:
    uvicorn main:app --reload --port 8000

Nothing here talks to Postgres, pgvector or Neo4j yet — that arrives in
Phase 2 alongside real video processing.
"""

from datetime import datetime, timezone
from typing import Any, Optional

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

app = FastAPI(
    title="Argus AI Service",
    description="Video perception, retrieval and reasoning service for Argus.",
    version="0.1.0",
)

# Allow the Spring Boot backend to call this directly during local dev.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:8080", "http://localhost:5173"],
    allow_methods=["*"],
    allow_headers=["*"],
)


class QueryRequest(BaseModel):
    investigation_id: str
    message: str


class EvidenceResult(BaseModel):
    camera: str
    start_timestamp: str
    end_timestamp: str
    description: str
    relevance: float
    thumbnail_url: Optional[str] = None


class QueryResponse(BaseModel):
    reply: str
    results: Optional[list[EvidenceResult]] = None


@app.get("/health")
def health() -> dict[str, Any]:
    return {"status": "ok", "service": "argus-ai-service", "phase": 1}


@app.post("/query", response_model=QueryResponse)
def query(request: QueryRequest) -> QueryResponse:
    """
    Phase 1 stub: acknowledges the investigator's question and returns a
    single placeholder evidence card so the frontend result-card UI can be
    built and tested against a real response shape before retrieval exists.

    Phase 3 replaces this body with: query understanding -> entity/attribute/
    event extraction -> vector + Knowledge Graph hybrid retrieval -> ranking
    -> grounded LLM response.
    """
    canned_result = EvidenceResult(
        camera="Camera 01",
        start_timestamp="00:00:00",
        end_timestamp="00:00:12",
        description="Evidence retrieval is not yet implemented (Phase 3). "
                     "This is a placeholder result so the UI can be built end-to-end.",
        relevance=0.0,
    )

    return QueryResponse(
        reply=(
            f"Received your query for investigation {request.investigation_id}: "
            f"\u201c{request.message}\u201d. Evidence search isn't wired up yet "
            f"(that's Phase 3) — for now this confirms the chat pipeline works."
        ),
        results=[canned_result],
    )


@app.get("/")
def root() -> dict[str, str]:
    return {
        "service": "argus-ai-service",
        "docs": "/docs",
        "checked_at": datetime.now(timezone.utc).isoformat(),
    }
