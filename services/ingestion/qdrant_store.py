"""
Qdrant collection setup and upsert for the NHL RAG pipeline.

Collection config:
  - Vectors: 1024-dim dense, cosine similarity, HNSW index
  - HNSW: m=16, ef_construct=100 (good recall/speed tradeoff on CPU)
  - Payload indexes on source, date, url for fast metadata filtering

Payload schema per point:
  source (str)  — nhl_scores | nhl_standings | nhl_player_stats |
                         nhl_play_by_play | reddit_hockey | reddit_nhl | news
  date (str)  — ISO date of ingestion (e.g. "2026-03-25")
  chunk_text (str)  — the actual text that was embedded
  entity_tags (dict) — {"teams": [...], "players": [...]}
  url(str)  — original URL, empty string if not available
  doc_id (str)  — SHA-1 of source + text (links chunks from the same doc)
  chunk_index (int)  — position of this chunk within its source document
"""
import logging
import uuid

from qdrant_client import QdrantClient
from qdrant_client.models import (
    Distance,
    HnswConfigDiff,
    PayloadSchemaType,
    PointStruct,
    VectorParams,
)

from embedder import EmbeddedChunk

log = logging.getLogger(__name__)

COLLECTION_NAME = "nhl_rag"
VECTOR_DIM = 1024


def ensure_collection(client: QdrantClient):
    """
    Create the nhl_rag collection if it does not already exist.
    Safe to call on every startup — no-ops if collection is present.
    """
    existing = {c.name for c in client.get_collections().collections}
    if COLLECTION_NAME in existing:
        log.info(f"Qdrant collection '{COLLECTION_NAME}' already exists — skipping creation")
        return

    client.create_collection(
        collection_name=COLLECTION_NAME,
        vectors_config=VectorParams(
            size=VECTOR_DIM,
            distance=Distance.COSINE,
            hnsw_config=HnswConfigDiff(
                m=16,  # edges per node — higher = better recall, more RAM
                ef_construct=100,  # build-time search width — higher = better quality index
            ),
        ),
    )
    log.info(f"Created Qdrant collection '{COLLECTION_NAME}' (dim={VECTOR_DIM}, cosine, HNSW m=16)")

    # Payload indexes enable fast WHERE-style filtering during hybrid search
    _create_payload_indexes(client)


def _create_payload_indexes(client: QdrantClient):
    indexes = [
        ("source", PayloadSchemaType.KEYWORD),
        ("date", PayloadSchemaType.KEYWORD),
        ("url", PayloadSchemaType.KEYWORD),
        ("doc_id", PayloadSchemaType.KEYWORD),
    ]
    for field, schema_type in indexes:
        client.create_payload_index(
            collection_name=COLLECTION_NAME,
            field_name=field,
            field_schema=schema_type,
        )
        log.info(f"Payload index created: {field} ({schema_type})")


def upsert_chunks(client: QdrantClient, embedded_chunks: list[EmbeddedChunk]) -> int:
    """
    Upsert embedded chunks into Qdrant.

    Point IDs are deterministic UUIDs derived from doc_id + chunk_index,
    so re-upserting the same chunk overwrites rather than duplicates.

    Returns the number of points upserted.
    """
    points = []
    for ec in embedded_chunks:
        meta = ec.chunk.metadata
        point_id = str(uuid.uuid5(uuid.NAMESPACE_DNS, f"{meta.doc_id}:{meta.chunk_index}"))
        points.append(PointStruct(
            id=point_id,
            vector=ec.embedding,
            payload={
                "source": meta.source,
                "date": meta.date,
                "chunk_text": ec.chunk.text,
                "entity_tags": {"teams": meta.teams, "players": meta.players},
                "url": meta.url or "",
                "doc_id": meta.doc_id,
                "chunk_index": meta.chunk_index,
            },
        ))

    client.upsert(collection_name=COLLECTION_NAME, points=points)
    return len(points)
