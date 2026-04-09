from fastapi import FastAPI, HTTPException, Request
from pydantic import BaseModel

app = FastAPI()

# Request/response models

class QueryRequest(BaseModel):
    question: str
    top_k: int = 50  # dense + BM25 retrieval pool fed into RRF
    rerank_candidates: int = 20  # top RRF results passed to the cross-encoder
    top_n: int = 5  # final results returned after reranking
    # Metadata filters — all optional, all combinable
    date_from: str | None = None  # ISO "YYYY-MM-DD", inclusive
    date_to: str | None = None  # ISO "YYYY-MM-DD", inclusive
    sources: list[str] | None = None  # e.g. ["nhl_scores", "reddit_hockey"]
    teams: list[str] | None = None  # e.g. ["Boston Bruins"]
    players: list[str] | None = None  # e.g. ["Connor McDavid"]

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
    answer: str

class IngestRequest(BaseModel):
    text: str
    source: str = "manual"
    url: str
    date: str
    entity_tags: dict | None = None


@app.get("/health")
async def health():
    return {"status": "ok"}

@app.post("/query", response_model=QueryResponse)
async def query(req: QueryRequest):
    raise HTTPException(status_code=501, detail="Not implemented yet")

@app.post("/ingest")
async def ingest(req: IngestRequest):
    # chunk req.text, embed, upsert to Qdrant, publish to Redis stream
    raise HTTPException(status_code=501, detail="Not implemented yet")
