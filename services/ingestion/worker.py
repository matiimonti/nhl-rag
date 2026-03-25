import asyncio
import json
import logging
import os
import time

import redis
import redis.asyncio as aioredis
from apscheduler.schedulers.asyncio import AsyncIOScheduler
from qdrant_client import QdrantClient

from nhl_client import NHLClient
from news_client import NewsClient
from reddit_client import RedditClient

logging.basicConfig(level=os.getenv("LOG_LEVEL", "INFO"), format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)

REDIS_URL = os.getenv("REDIS_URL", "redis://localhost:6379")
QDRANT_HOST = os.getenv("QDRANT_HOST", "localhost")
QDRANT_PORT = int(os.getenv("QDRANT_PORT", "6333"))
STREAM_KEY = "nhl:ingest"


def check_connections():
    r = redis.Redis.from_url(REDIS_URL, socket_connect_timeout=3)
    r.ping()
    log.info("Redis: reachable")

    q = QdrantClient(host=QDRANT_HOST, port=QDRANT_PORT, timeout=3)
    q.get_collections()
    log.info("Qdrant: reachable")


async def publish(r: aioredis.Redis, source: str, payload: dict):
    """Publish a document to the Redis ingest stream."""
    msg_id = await r.xadd(STREAM_KEY, {
        "source": source,
        "payload": json.dumps(payload, default=str),
        "ts": str(time.time()),
    })
    log.debug(f"stream {STREAM_KEY} [{source}] → {msg_id}")


# ── Poll jobs ─────────────────────────────────────────────────────────────────

async def poll_scores(nhl: NHLClient, r: aioredis.Redis):
    try:
        scores = await nhl.fetch_scores()
        for game in scores.games:
            await publish(r, "nhl_scores", game.model_dump())
        log.info(f"scores: published {len(scores.games)} game(s)")
    except Exception as e:
        log.warning(f"poll_scores failed: {e}")


async def poll_standings(nhl: NHLClient, r: aioredis.Redis):
    try:
        standings = await nhl.fetch_standings()
        for team in standings.standings:
            await publish(r, "nhl_standings", team.model_dump())
        log.info(f"standings: published {len(standings.standings)} team(s)")
    except Exception as e:
        log.warning(f"poll_standings failed: {e}")


async def poll_player_stats(nhl: NHLClient, r: aioredis.Redis):
    try:
        stats = await nhl.fetch_player_stats()
        for player in stats.data:
            await publish(r, "nhl_player_stats", player.model_dump())
        log.info(f"player_stats: published {len(stats.data)} skater(s)")
    except Exception as e:
        log.warning(f"poll_player_stats failed: {e}")


async def poll_play_by_play(nhl: NHLClient, r: aioredis.Redis):
    try:
        game_ids = await nhl.fetch_live_game_ids()
        if not game_ids:
            log.info("play_by_play: no live games")
            return
        for game_id in game_ids:
            pbp = await nhl.fetch_play_by_play(game_id)
            for event in pbp.plays:
                await publish(r, "nhl_play_by_play", {"game_id": game_id, **event.model_dump()})
            log.info(f"play_by_play: published {len(pbp.plays)} event(s) for game {game_id}")
    except Exception as e:
        log.warning(f"poll_play_by_play failed: {e}")


async def poll_reddit(reddit: RedditClient, r: aioredis.Redis):
    try:
        pages = await reddit.fetch_all_subreddits()
        total = 0
        for page in pages:
            for post in page.posts:
                await publish(r, f"reddit_{page.subreddit}", post.model_dump())
            total += len(page.posts)
        log.info(f"reddit: published {total} post(s)")
    except Exception as e:
        log.warning(f"poll_reddit failed: {e}")


async def poll_news(news: NewsClient, r: aioredis.Redis):
    try:
        result = await news.fetch_hockey_news()
        for article in result.articles:
            await publish(r, "news", article.model_dump())
        log.info(f"news: published {len(result.articles)} article(s)")
    except RuntimeError as e:
        log.warning(f"poll_news skipped: {e}")
    except Exception as e:
        log.warning(f"poll_news failed: {e}")


# ── Main ──────────────────────────────────────────────────────────────────────

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

    nhl = NHLClient()
    reddit = RedditClient()
    news = NewsClient(redis_url=REDIS_URL)
    stream = aioredis.from_url(REDIS_URL, decode_responses=True)

    scheduler = AsyncIOScheduler()
    scheduler.add_job(poll_scores, "interval", seconds=300, args=[nhl, stream])
    scheduler.add_job(poll_standings, "interval", seconds=300, args=[nhl, stream])
    scheduler.add_job(poll_player_stats, "interval", seconds=300, args=[nhl, stream])
    scheduler.add_job(poll_play_by_play, "interval", seconds=300, args=[nhl, stream])
    scheduler.add_job(poll_reddit, "interval", seconds=900, args=[reddit, stream])
    scheduler.add_job(poll_news, "interval", seconds=3600, args=[news, stream])

    scheduler.start()
    log.info("Scheduler started — NHL:5min  Reddit:15min  News:60min")

    try:
        await asyncio.Event().wait()
    finally:
        await nhl.close()
        await reddit.close()
        await news.close()
        await stream.aclose()


if __name__ == "__main__":
    asyncio.run(main())
