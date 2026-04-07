import asyncio
import json
import logging
import os
import time

import redis
import redis.asyncio as aioredis
from apscheduler.schedulers.asyncio import AsyncIOScheduler
from qdrant_client import QdrantClient

from chunker import Chunker
from dedup import Deduplicator
from embedder import Embedder
from qdrant_store import ensure_collection, upsert_chunks
from clients.nhl_client import NHLClient
from clients.news_client import NewsClient
from clients.reddit_client import RedditClient

logging.basicConfig(level=os.getenv("LOG_LEVEL", "INFO"), format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)

REDIS_URL = os.getenv("REDIS_URL", "redis://localhost:6379")
QDRANT_HOST = os.getenv("QDRANT_HOST", "localhost")
QDRANT_PORT = int(os.getenv("QDRANT_PORT", "6333"))
STREAM_KEY = "nhl:ingest"
CONSUMER_GROUP = "embed-workers"
CONSUMER_NAME = "worker-1"
STREAM_BATCH = 10  # messages per read
THROUGHPUT_LOG_N = 50  # log throughput every N docs


def check_connections():
    r = redis.Redis.from_url(REDIS_URL, socket_connect_timeout=3)
    r.ping()
    log.info("Redis: reachable")

    q = QdrantClient(host=QDRANT_HOST, port=QDRANT_PORT, timeout=3)
    q.get_collections()
    log.info("Qdrant: reachable")


async def publish(r: aioredis.Redis, dedup: Deduplicator, source: str, payload: dict) -> bool:
    """Publish a document to the Redis ingest stream, skipping duplicates.
    Returns True if published, False if dropped as duplicate."""
    if await dedup.check_and_mark(source, payload):
        return False
    msg_id = await r.xadd(STREAM_KEY, {
        "source": source,
        "payload": json.dumps(payload, default=str),
        "ts": str(time.time()),
    })
    log.debug(f"stream {STREAM_KEY} [{source}] → {msg_id}")
    return True


# Poll jobs

async def poll_scores(nhl: NHLClient, r: aioredis.Redis, dedup: Deduplicator):
    try:
        scores = await nhl.fetch_scores()
        n = sum([await publish(r, dedup, "nhl_scores", g.model_dump()) for g in scores.games])
        log.info(f"scores: {n} new, {len(scores.games) - n} duplicate(s) dropped")
    except Exception as e:
        log.warning(f"poll_scores failed: {e}")


async def poll_standings(nhl: NHLClient, r: aioredis.Redis, dedup: Deduplicator):
    try:
        standings = await nhl.fetch_standings()
        n = sum([await publish(r, dedup, "nhl_standings", t.model_dump()) for t in standings.standings])
        log.info(f"standings: {n} new, {len(standings.standings) - n} duplicate(s) dropped")
    except Exception as e:
        log.warning(f"poll_standings failed: {e}")


async def poll_player_stats(nhl: NHLClient, r: aioredis.Redis, dedup: Deduplicator):
    try:
        stats = await nhl.fetch_player_stats()
        n = sum([await publish(r, dedup, "nhl_player_stats", p.model_dump()) for p in stats.data])
        log.info(f"player_stats: {n} new, {len(stats.data) - n} duplicate(s) dropped")
    except Exception as e:
        log.warning(f"poll_player_stats failed: {e}")


async def poll_play_by_play(nhl: NHLClient, r: aioredis.Redis, dedup: Deduplicator):
    try:
        game_ids = await nhl.fetch_live_game_ids()
        if not game_ids:
            log.info("play_by_play: no live games")
            return
        for game_id in game_ids:
            pbp = await nhl.fetch_play_by_play(game_id)
            n = sum([await publish(r, dedup, "nhl_play_by_play", {"game_id": game_id, **e.model_dump()}) for e in pbp.plays])
            log.info(f"play_by_play: {n} new, {len(pbp.plays) - n} duplicate(s) dropped for game {game_id}")
    except Exception as e:
        log.warning(f"poll_play_by_play failed: {e}")


async def poll_reddit(reddit: RedditClient, r: aioredis.Redis, dedup: Deduplicator):
    try:
        pages = await reddit.fetch_all_subreddits()
        total_new, total_all = 0, 0
        for page in pages:
            n = sum([await publish(r, dedup, f"reddit_{page.subreddit}", p.model_dump()) for p in page.posts])
            total_new += n
            total_all += len(page.posts)
        log.info(f"reddit: {total_new} new, {total_all - total_new} duplicate(s) dropped")
    except Exception as e:
        log.warning(f"poll_reddit failed: {e}")


async def poll_news(news: NewsClient, r: aioredis.Redis, dedup: Deduplicator):
    try:
        result = await news.fetch_hockey_news()
        n = sum([await publish(r, dedup, "news", a.model_dump()) for a in result.articles])
        log.info(f"news: {n} new, {len(result.articles) - n} duplicate(s) dropped")
    except RuntimeError as e:
        log.warning(f"poll_news skipped: {e}")
    except Exception as e:
        log.warning(f"poll_news failed: {e}")


# Stream consumer

async def _with_backoff(fn, *args, max_retries: int = 3, base_delay: float = 1.0):
    """Run a sync function in a thread with exponential backoff on failure."""
    for attempt in range(max_retries):
        try:
            return await asyncio.to_thread(fn, *args)
        except Exception as e:
            if attempt == max_retries - 1:
                raise
            delay = base_delay * (2 ** attempt)
            log.warning(f"Retry {attempt + 1}/{max_retries} in {delay:.0f}s: {e}")
            await asyncio.sleep(delay)


async def _process_message(
    fields: dict,
    chunker: Chunker,
    embedder: Embedder,
    qdrant: QdrantClient,
):
    """Chunk -> embed -> upsert one stream message."""
    from datetime import date as _date
    source = fields.get("source", "")
    payload = json.loads(fields.get("payload", "{}"))
    today = _date.today().isoformat()

    chunks = await asyncio.to_thread(chunker.chunk_document, source, payload, today)
    if not chunks:
        return 0

    embedded = await _with_backoff(embedder.embed_chunks, chunks)
    n = await _with_backoff(upsert_chunks, qdrant, embedded)
    return n


async def consume_stream(
    stream: aioredis.Redis,
    chunker: Chunker,
    embedder: Embedder,
    qdrant: QdrantClient,
):
    """
    Read from nhl:ingest, chunk, embed, and upsert to Qdrant.
    Uses a consumer group for at-least-once delivery.
    XACKs only after a successful upsert.
    """
    try:
        await stream.xgroup_create(STREAM_KEY, CONSUMER_GROUP, id="0", mkstream=True)
        log.info(f"Consumer group '{CONSUMER_GROUP}' created")
    except Exception:
        log.info(f"Consumer group '{CONSUMER_GROUP}' already exists")

    total_docs = 0
    total_chunks = 0
    t_start = time.perf_counter()

    while True:
        try:
            response = await stream.xreadgroup(
                CONSUMER_GROUP, CONSUMER_NAME,
                {STREAM_KEY: ">"},
                count=STREAM_BATCH,
                block=2000,
            )
            if not response:
                continue

            for _stream_name, entries in response:
                for msg_id, fields in entries:
                    try:
                        n = await _process_message(fields, chunker, embedder, qdrant)
                        await stream.xack(STREAM_KEY, CONSUMER_GROUP, msg_id)
                        total_docs += 1
                        total_chunks += n
                    except Exception as e:
                        log.warning(f"Failed to process message {msg_id}: {e}")

            if total_docs > 0 and total_docs % THROUGHPUT_LOG_N == 0:
                elapsed = time.perf_counter() - t_start
                log.info(
                    f"Throughput: {total_docs / elapsed:.1f} docs/s | "
                    f"{total_chunks / elapsed:.1f} chunks/s | "
                    f"{total_docs} docs, {total_chunks} chunks total"
                )

        except Exception as e:
            log.warning(f"Consumer loop error: {e}")
            await asyncio.sleep(2)


# Main

async def main():
    log.info("Ingestion worker starting...")

    for attempt in range(5):
        try:
            check_connections()
            break
        except Exception as e:
            log.warning(f"Connection check failed (attempt {attempt + 1}/5): {e}")
            time.sleep(3)
    else:
        log.error("Could not reach Redis or Qdrant after 5 attempts — exiting")
        raise SystemExit(1)

    qdrant = QdrantClient(host=QDRANT_HOST, port=QDRANT_PORT)
    ensure_collection(qdrant)

    log.info("Loading embedding model (may take a moment on first run)...")
    embedder = await asyncio.to_thread(Embedder)
    chunker = Chunker()

    nhl = NHLClient()
    reddit = RedditClient()
    news = NewsClient(redis_url=REDIS_URL)
    stream = aioredis.from_url(REDIS_URL, decode_responses=True)
    dedup = Deduplicator(stream)

    scheduler = AsyncIOScheduler()
    scheduler.add_job(poll_scores, "interval", seconds=300, args=[nhl, stream, dedup])
    scheduler.add_job(poll_standings, "interval", seconds=300, args=[nhl, stream, dedup])
    scheduler.add_job(poll_player_stats, "interval", seconds=300, args=[nhl, stream, dedup])
    scheduler.add_job(poll_play_by_play, "interval", seconds=300, args=[nhl, stream, dedup])
    scheduler.add_job(poll_reddit, "interval", seconds=900, args=[reddit, stream, dedup])
    scheduler.add_job(poll_news, "interval", seconds=3600, args=[news, stream, dedup])

    scheduler.start()
    log.info("Scheduler started — NHL:5min  Reddit:15min  News:60min")

    asyncio.create_task(consume_stream(stream, chunker, embedder, qdrant))
    log.info("Stream consumer started")

    try:
        await asyncio.Event().wait()
    finally:
        await nhl.close()
        await reddit.close()
        await news.close()
        await stream.aclose()


if __name__ == "__main__":
    asyncio.run(main())
