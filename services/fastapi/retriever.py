"""
Dense semantic retrieval for the NHL RAG pipeline.

Query embedding:
  Model: BAAI/bge-large-en-v1.5  (1024-dim, CPU)
  Prefix: "Represent this question for searching relevant passages: "
  BGE uses a different prefix at query time vs. indexing time.

Qdrant search:
  Collection: nhl_rag
  Metric: cosine similarity (vectors are L2-normalised)
  Default k: 50 — broad recall pool for downstream reranking
"""
import logging
import time
from dataclasses import dataclass

from qdrant_client import QdrantClient
from qdrant_client.models import ScoredPoint
from sentence_transformers import SentenceTransformer

log = logging.getLogger(__name__)

MODEL_NAME = "BAAI/bge-large-en-v1.5"
COLLECTION_NAME = "nhl_rag"
# BGE query prefix — different from the passage prefix used at index time
QUERY_PREFIX = "Represent this question for searching relevant passages: "


@dataclass
class RetrievedChunk:
    text: str
    score: float
    source: str
    date: str
    url: str
    doc_id: str
    chunk_index: int
    entity_tags: dict   # {"teams": [...], "players": [...]}


# Embedder

class QueryEmbedder:
    def __init__(self, model_name: str = MODEL_NAME):
        log.info(f"Loading query embedding model '{model_name}'...")
        t0 = time.perf_counter()
        self._model = SentenceTransformer(model_name, device="cpu")
        log.info(f"Query embedder ready in {time.perf_counter() - t0:.1f}s")

    def embed(self, query: str) -> list[float]:
        """Embed a natural-language question with the BGE query prefix.
        Returns a normalised 1024-dim vector."""
        vector = self._model.encode(
            QUERY_PREFIX + query,
            normalize_embeddings=True,
            show_progress_bar=False,
        )
        return vector.tolist()


# Search

def dense_search(
    client: QdrantClient,
    query_vector: list[float],
    top_k: int = 50,
) -> list[RetrievedChunk]:
    """Search nhl_rag by cosine similarity. Returns top_k chunks sorted by descending score."""
    t0 = time.perf_counter()
    hits: list[ScoredPoint] = client.search(
        collection_name=COLLECTION_NAME,
        query_vector=query_vector,
        limit=top_k,
        with_payload=True,
        with_vectors=False,
    )
    log.info(f"dense_search: {len(hits)} hits in {(time.perf_counter() - t0) * 1000:.0f}ms (top_k={top_k})")

    chunks = []
    for hit in hits:
        p = hit.payload or {}
        chunks.append(RetrievedChunk(
            text=p.get("chunk_text", ""),
            score=hit.score,
            source=p.get("source", ""),
            date=p.get("date", ""),
            url=p.get("url", ""),
            doc_id=p.get("doc_id", ""),
            chunk_index=p.get("chunk_index", 0),
            entity_tags=p.get("entity_tags", {"teams": [], "players": []}),
        ))
    return chunks
