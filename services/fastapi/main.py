import os
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

app = FastAPI(title="NHL RAG API", version="0.1.0")

QDRANT_HOST = os.getenv("QDRANT_HOST", "localhost")
QDRANT_PORT = int(os.getenv("QDRANT_PORT", "6333"))
REDIS_URL = os.getenv("REDIS_URL", "redis://localhost:6379")


class QueryRequest(BaseModel):
    question: str
    top_k: int = 5

class QueryResponse(BaseModel):
    answer: str
    chunks: list[str]
    latency_ms: float

class IngestRequest(BaseModel):
    text: str
    source: str = "manual"


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/query", response_model=QueryResponse)
def query(req: QueryRequest):
    # TODO: embed req.question, hybrid search Qdrant, rerank, generate answer
    raise HTTPException(status_code=501, detail="Not implemented yet")


@app.post("/ingest")
def ingest(req: IngestRequest):
    # TODO: chunk req.text, embed, upsert to Qdrant, publish to Redis stream
    raise HTTPException(status_code=501, detail="Not implemented yet")
