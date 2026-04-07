import logging
import os
import time

from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException, Request
from pydantic import BaseModel
from qdrant_client import QdrantClient

from retriever import QueryEmbedder, RetrievedChunk, dense_search

logging.basicConfig(level=os.getenv("LOG_LEVEL", "INFO"), format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)

QDRANT_HOST = os.getenv("QDRANT_HOST", "localhost")
QDRANT_PORT = int(os.getenv("QDRANT_PORT", "6333"))


# Startup / shutdown

@asynccontextmanager
async def lifespan(app: FastAPI):
    log.info("Loading query embedder (shared model cache)...")
    app.state.embedder = QueryEmbedder()
    app.state.qdrant = QdrantClient(host=QDRANT_HOST, port=QDRANT_PORT)
    log.info("FastAPI ready")
    yield
    log.info("FastAPI shutting down")


app = FastAPI(title="NHL RAG API", version="0.2.0", lifespan=lifespan)


# Request / response models

class QueryRequest(BaseModel):
    question: str
    top_k: int = 50   # retrieval pool size; downstream reranking will narrow this


class ChunkResult(BaseModel):
    text: str
    score: float
    source: str
    date: str
    url: str
    doc_id: str
    chunk_index: int
    entity_tags: dict


class QueryResponse(BaseModel):
    question: str
    chunks: list[ChunkResult]
    total_hits: int
    latency_ms: float


class IngestRequest(BaseModel):
    text: str
    source: str = "manual"


# Routes

@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/query", response_model=QueryResponse)
def query(req: QueryRequest, request: Request):
    t0 = time.perf_counter()

    embedder: QueryEmbedder = request.app.state.embedder
    qdrant: QdrantClient = request.app.state.qdrant

    query_vector = embedder.embed(req.question)
    chunks: list[RetrievedChunk] = dense_search(qdrant, query_vector, top_k=req.top_k)

    latency_ms = (time.perf_counter() - t0) * 1000
    log.info(f"query '{req.question[:60]}': {len(chunks)} chunks in {latency_ms:.0f}ms")

    return QueryResponse(
        question=req.question,
        chunks=[
            ChunkResult(
                text=c.text,
                score=c.score,
                source=c.source,
                date=c.date,
                url=c.url,
                doc_id=c.doc_id,
                chunk_index=c.chunk_index,
                entity_tags=c.entity_tags,
            )
            for c in chunks
        ],
        total_hits=len(chunks),
        latency_ms=round(latency_ms, 2),
    )


@app.post("/ingest")
def ingest(req: IngestRequest):
    # TODO: chunk req.text, embed, upsert to Qdrant, publish to Redis stream
    raise HTTPException(status_code=501, detail="Not implemented yet")
