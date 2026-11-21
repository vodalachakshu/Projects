"""FastAPI entry point for the Document Q&A RAG Service."""

from __future__ import annotations

import logging
import uuid
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, Request, status
from fastapi.responses import JSONResponse

from app.config import get_settings
from app.logging_config import setup_logging
from app.rag_engine import (
    DocumentQARAGEngine,
    RAGConfigurationError,
    RAGGenerationError,
    RAGRetrievalError,
)
from app.schemas import (
    AskRequest,
    AskResponse,
    CorpusStatusResponse,
    HealthResponse,
)

settings = get_settings()
logger = setup_logging(settings.log_level)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Initialize the RAG engine and index documents on startup."""
    logger.info("Starting %s v%s", settings.app_name, settings.app_version)
    try:
        app.state.rag_engine = DocumentQARAGEngine(settings=settings)
    except RAGConfigurationError as exc:
        logger.error("Startup configuration error: %s", exc)
        raise RuntimeError(str(exc)) from exc
    yield
    logger.info("Shutting down %s", settings.app_name)


app = FastAPI(
    title=settings.app_name,
    version=settings.app_version,
    description="Retrieval-Augmented Generation (RAG) Document Q&A Service over 5-10 documents.",
    lifespan=lifespan,
)


@app.middleware("http")
async def add_request_id_and_log(request: Request, call_next):
    """Attach a correlation request_id and log request duration/status."""
    request_id = request.headers.get("X-Request-ID") or str(uuid.uuid4())
    request.state.request_id = request_id
    logger.info("request_id=%s method=%s path=%s started", request_id, request.method, request.url.path)
    try:
        response = await call_next(request)
    except Exception:
        logger.exception("request_id=%s unhandled error during request processing", request_id)
        return JSONResponse(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            content={
                "error": "internal_server_error",
                "message": "An unexpected internal error occurred.",
                "request_id": request_id,
            },
        )
    response.headers["X-Request-ID"] = request_id
    logger.info(
        "request_id=%s method=%s path=%s status=%d completed",
        request_id,
        request.method,
        request.url.path,
        response.status_code,
    )
    return response


@app.get("/health", response_model=HealthResponse, tags=["System"])
def health_check(request: Request) -> HealthResponse:
    """Return service health and index statistics."""
    engine: DocumentQARAGEngine = request.app.state.rag_engine
    return HealthResponse(
        status="ok",
        service=settings.app_name,
        version=settings.app_version,
        indexed_documents=len(engine.doc_stats),
        indexed_chunks=len(engine.chunks),
    )


@app.get("/documents", response_model=CorpusStatusResponse, tags=["Documents"])
def list_documents(request: Request) -> CorpusStatusResponse:
    """List indexed documents and chunk counts."""
    engine: DocumentQARAGEngine = request.app.state.rag_engine
    status_data = engine.get_corpus_status()
    return CorpusStatusResponse(**status_data)


@app.post("/documents/reindex", response_model=CorpusStatusResponse, tags=["Documents"])
def reindex_documents(request: Request) -> CorpusStatusResponse:
    """Reload and re-index documents from disk."""
    request_id = getattr(request.state, "request_id", "unknown")
    engine: DocumentQARAGEngine = request.app.state.rag_engine
    try:
        engine.build_index()
        return CorpusStatusResponse(**engine.get_corpus_status
