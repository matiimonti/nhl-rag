"""
Integration tests for the full ingestion pipeline.

Path under test: publish() → Redis Stream → _process_message() → Qdrant

The Embedder is replaced with a fake that returns deterministic random
1024-dim vectors — this avoids the sentence-transformers / torch dependency
and the 1.3 GB model download at test time.

Requirements (install once in the project venv):
    pip install "testcontainers[redis]" qdrant-client redis numpy

Docker must be running. All tests are skipped automatically if testcontainers
is not installed.
"""
import asyncio
import json
import os
import sys
import time
from unittest.mock import MagicMock

# ---------------------------------------------------------------------------
# Mock heavy deps BEFORE any ingestion module is imported.
# sentence_transformers pulls in torch which is not installed in the test venv.
# ---------------------------------------------------------------------------
import numpy as np

sys.modules.setdefault("sentence_transformers", MagicMock())
sys.modules.setdefault("torch", MagicMock())
sys.modules.setdefault("apscheduler", MagicMock())
sys.modules.setdefault("apscheduler.schedulers", MagicMock())
sys.modules.setdefault("apscheduler.schedulers.asyncio", MagicMock())

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "services", "ingestion"))

import pytest
import redis.asyncio as aioredis
from qdrant_client import QdrantClient

try:
    from testcontainers.redis import RedisContainer
    from testcontainers.core.container import DockerContainer
    _TESTCONTAINERS_AVAILABLE = True
except ImportError:
    _TESTCONTAINERS_AVAILABLE = False

pytestmark = pytest.mark.skipif(
    not _TESTCONTAINERS_AVAILABLE,
    reason="testcontainers not installed — run: pip install 'testcontainers[redis]'",
)

from worker import publish, _process_message, STREAM_KEY
from dedup import Deduplicator
from qdrant_store import ensure_collection, COLLECTION_NAME, VECTOR_DIM
from chunker import Chunker
from embedder import EmbeddedChunk  # safe to import after mock


# ---------------------------------------------------------------------------
# Fake embedder
# ---------------------------------------------------------------------------

def _fake_embedder():
    """Embedder mock that returns deterministic random 1024-dim unit vectors."""
    mock = MagicMock()
    rng = np.random.default_rng(seed=42)

    def embed_chunks(chunks):
        vecs = rng.random((len(chunks), VECTOR_DIM)).astype(np.float32)
        # normalise so cosine similarity is well-defined
        vecs /= np.linalg.norm(vecs, axis=1, keepdims=True)
        return [EmbeddedChunk(chunk=c, embedding=v.tolist()) for c, v in zip(chunks, vecs)]

    mock.embed_chunks.side_effect = embed_chunks
    return mock


# ---------------------------------------------------------------------------
# Containers — one pair per test session
# ---------------------------------------------------------------------------

@pytest.fixture(scope="session")
def redis_container():
    with RedisContainer("redis:7-alpine") as container:
        yield container


@pytest.fixture(scope="session")
def qdrant_container():
    container = DockerContainer("qdrant/qdrant:latest")
    container.with_exposed_ports(6333)
    with container:
        host = container.get_container_host_ip()
        port = int(container.get_exposed_port(6333))
        deadline = time.time() + 30
        while time.time() < deadline:
            try:
                QdrantClient(host=host, port=port, timeout=2).get_collections()
                break
            except Exception:
                time.sleep(0.5)
        else:
            pytest.fail("Qdrant container did not become ready within 30s")
        yield container


# ---------------------------------------------------------------------------
# Per-test fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(scope="session")
def redis_url(redis_container):
    host = redis_container.get_container_host_ip()
    port = redis_container.get_exposed_port(6379)
    return f"redis://{host}:{port}"


@pytest.fixture(autouse=True)
def clean_redis(redis_url):
    """Flush all Redis keys before every test for clean isolation."""
    async def _flush():
        r = aioredis.from_url(redis_url)
        await r.flushdb()
        await r.aclose()
    asyncio.run(_flush())


@pytest.fixture
def qdrant_client(qdrant_container):
    """Fresh Qdrant collection for every test."""
    host = qdrant_container.get_container_host_ip()
    port = int(qdrant_container.get_exposed_port(6333))
    client = QdrantClient(host=host, port=port)
    try:
        client.delete_collection(COLLECTION_NAME)
    except Exception:
        pass
    ensure_collection(client)
    return client


def _run(coro):
    return asyncio.run(coro)


def _news_payloads(n: int) -> list[dict]:
    return [
        {
            "title": f"NHL story {i}",
            "description": f"Story details {i}",
            "url": f"https://example.com/story/{i}",
        }
        for i in range(n)
    ]


# ---------------------------------------------------------------------------
# publish() → Redis Stream
# ---------------------------------------------------------------------------

class TestPublish:
    def test_10_unique_docs_all_reach_stream(self, redis_url):
        async def _inner():
            r = aioredis.from_url(redis_url, decode_responses=True)
            dedup = Deduplicator(r)
            results = [await publish(r, dedup, "news", p) for p in _news_payloads(10)]
            stream_len = await r.xlen(STREAM_KEY)
            await r.aclose()
            return results, stream_len

        results, stream_len = _run(_inner())
        assert all(results)           # every doc returned True (published)
        assert stream_len == 10       # exactly 10 messages in stream

    def test_duplicate_dropped_before_stream(self, redis_url):
        async def _inner():
            r = aioredis.from_url(redis_url, decode_responses=True)
            dedup = Deduplicator(r)
            payload = {"title": "Repeated story", "url": "https://example.com"}
            first  = await publish(r, dedup, "news", payload)
            second = await publish(r, dedup, "news", payload)
            stream_len = await r.xlen(STREAM_KEY)
            await r.aclose()
            return first, second, stream_len

        first, second, stream_len = _run(_inner())
        assert first is True
        assert second is False          # dedup dropped it
        assert stream_len == 1          # only one message in stream

    def test_same_payload_different_source_both_published(self, redis_url):
        async def _inner():
            r = aioredis.from_url(redis_url, decode_responses=True)
            dedup = Deduplicator(r)
            payload = {"title": "Game recap", "url": "https://example.com"}
            r1 = await publish(r, dedup, "reddit_hockey", payload)
            r2 = await publish(r, dedup, "reddit_nhl", payload)
            stream_len = await r.xlen(STREAM_KEY)
            await r.aclose()
            return r1, r2, stream_len

        r1, r2, stream_len = _run(_inner())
        assert r1 is True
        assert r2 is True
        assert stream_len == 2


# ---------------------------------------------------------------------------
# _process_message() → Qdrant
# ---------------------------------------------------------------------------

class TestProcessMessage:
    def test_news_message_produces_vectors(self, qdrant_client):
        fields = {
            "source": "news",
            "payload": json.dumps({"title": "Bruins beat Leafs 3-2", "url": "https://tsn.ca"}),
            "ts": str(time.time()),
        }
        n = _run(_process_message(fields, Chunker(), _fake_embedder(), qdrant_client))
        assert n >= 1
        assert qdrant_client.count(COLLECTION_NAME).count >= 1

    def test_unknown_source_produces_no_vectors(self, qdrant_client):
        fields = {
            "source": "unknown_source",
            "payload": json.dumps({"title": "something"}),
            "ts": str(time.time()),
        }
        n = _run(_process_message(fields, Chunker(), _fake_embedder(), qdrant_client))
        assert n == 0
        assert qdrant_client.count(COLLECTION_NAME).count == 0

    def test_stored_vector_has_correct_payload_fields(self, qdrant_client):
        fields = {
            "source": "news",
            "payload": json.dumps({"title": "Maple Leafs win 4-1", "url": "https://tsn.ca/story"}),
            "ts": str(time.time()),
        }
        _run(_process_message(fields, Chunker(), _fake_embedder(), qdrant_client))
        points, _ = qdrant_client.scroll(
            COLLECTION_NAME, limit=5, with_payload=True, with_vectors=True
        )
        assert points
        p = points[0].payload
        assert p["source"] == "news"
        assert p["chunk_text"]
        assert p["date"]
        assert p["doc_id"]
        assert len(points[0].vector) == VECTOR_DIM

    def test_upsert_is_idempotent(self, qdrant_client):
        """Processing the same message twice must not create duplicate vectors."""
        fields = {
            "source": "news",
            "payload": json.dumps({"title": "Idempotent story", "url": "https://example.com"}),
            "ts": str(time.time()),
        }
        _run(_process_message(fields, Chunker(), _fake_embedder(), qdrant_client))
        count_after_first = qdrant_client.count(COLLECTION_NAME).count

        _run(_process_message(fields, Chunker(), _fake_embedder(), qdrant_client))
        count_after_second = qdrant_client.count(COLLECTION_NAME).count

        assert count_after_second == count_after_first


# ---------------------------------------------------------------------------
# Full pipeline: publish → stream → process → Qdrant
# ---------------------------------------------------------------------------

class TestFullPipeline:
    def test_10_docs_in_qdrant_within_30s(self, redis_url, qdrant_client):
        """
        Core integration test: publish 10 distinct documents, process each
        through the full chunker → fake embedder → Qdrant path, and assert
        >= 10 vectors are stored within 30 seconds.
        """
        t_start = time.time()

        async def _inner():
            r = aioredis.from_url(redis_url, decode_responses=True)
            dedup = Deduplicator(r)
            chunker = Chunker()
            embedder = _fake_embedder()

            for p in _news_payloads(10):
                await publish(r, dedup, "news", p)

            msgs = await r.xrange(STREAM_KEY, count=20)
            total = 0
            for _, fields in msgs:
                total += await _process_message(fields, chunker, embedder, qdrant_client)
            await r.aclose()
            return total

        total_vectors = _run(_inner())
        elapsed = time.time() - t_start

        assert elapsed < 30, f"Pipeline took {elapsed:.1f}s — exceeded 30s budget"
        assert total_vectors >= 10
        assert qdrant_client.count(COLLECTION_NAME).count >= 10

    def test_all_7_source_types_reach_qdrant(self, redis_url, qdrant_client):
        """One document from each of the 7 known sources must all land in Qdrant."""
        source_payloads = [
            ("nhl_scores",       {"game_state": "FINAL", "game_date": "2025-01-01",
                                  "home_team": {"abbrev": "TOR", "score": 3},
                                  "away_team": {"abbrev": "BOS", "score": 2}}),
            ("nhl_standings",    {"team_name": {"default": "Boston Bruins"},
                                  "wins": 30, "losses": 15, "ot_losses": 5,
                                  "points": 65, "games_played": 50}),
            ("nhl_player_stats", {"name": "Sidney Crosby", "team": "PIT",
                                  "goals": 20, "assists": 35, "points": 55,
                                  "games_played": 50}),
            ("nhl_play_by_play", {"game_id": "2025020001", "period": 2,
                                  "time_in_period": "10:00", "type_desc_key": "goal"}),
            ("reddit_hockey",    {"title": "Great game tonight", "selftext": "",
                                  "permalink": "/r/hockey/abc"}),
            ("reddit_nhl",       {"title": "Trade deadline talk", "selftext": "",
                                  "permalink": "/r/nhl/xyz"}),
            ("news",             {"title": "NHL Playoff picture", "url": "https://tsn.ca"}),
        ]

        async def _inner():
            r = aioredis.from_url(redis_url, decode_responses=True)
            dedup = Deduplicator(r)
            chunker = Chunker()
            embedder = _fake_embedder()
            for source, payload in source_payloads:
                await publish(r, dedup, source, payload)
            msgs = await r.xrange(STREAM_KEY, count=20)
            total = 0
            for _, fields in msgs:
                total += await _process_message(fields, chunker, embedder, qdrant_client)
            await r.aclose()
            return total

        total = _run(_inner())
        assert total >= len(source_payloads)
        assert qdrant_client.count(COLLECTION_NAME).count >= len(source_payloads)

    def test_dedup_prevents_double_ingestion(self, redis_url, qdrant_client):
        """
        Publishing the same document twice must not produce duplicate vectors.
        Dedup drops it before the stream; deterministic point IDs handle any
        re-upserts that slip through.
        """
        async def _inner():
            r = aioredis.from_url(redis_url, decode_responses=True)
            dedup = Deduplicator(r)
            chunker = Chunker()
            embedder = _fake_embedder()

            payload = {"title": "Sidney Crosby scores twice", "url": "https://nhl.com"}
            await publish(r, dedup, "news", payload)
            await publish(r, dedup, "news", payload)  # duplicate — should be dropped

            msgs = await r.xrange(STREAM_KEY, count=20)
            assert len(msgs) == 1  # dedup stopped it before the stream

            for _, fields in msgs:
                await _process_message(fields, chunker, embedder, qdrant_client)

            count = qdrant_client.count(COLLECTION_NAME).count
            await r.aclose()
            return count

        count_after_one_publish = _run(_inner())
        assert count_after_one_publish >= 1

        # Re-processing the same message is idempotent — deterministic UUIDs
        fields = {
            "source": "news",
            "payload": json.dumps({
                "title": "Sidney Crosby scores twice", "url": "https://nhl.com"
            }),
            "ts": str(time.time()),
        }
        _run(_process_message(fields, Chunker(), _fake_embedder(), qdrant_client))
        assert qdrant_client.count(COLLECTION_NAME).count == count_after_one_publish
