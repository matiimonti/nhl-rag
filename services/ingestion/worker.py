import asyncio
import logging
import os
import time

import redis
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


def check_connections():
    r = redis.Redis.from_url(REDIS_URL, socket_connect_timeout=3)
    r.ping()
    log.info("Redis: reachable")

    q = QdrantClient(host=QDRANT_HOST, port=QDRANT_PORT, timeout=3)
    q.get_collections()
    log.info("Qdrant: reachable")


# Poll jobs

async def poll_scores(client: NHLClient):
    try:
        scores = await client.fetch_scores()
        log.info(f"scores: {len(scores.games)} game(s) today")
    except Exception as e:
        log.warning(f"poll_scores failed: {e}")


async def poll_standings(client: NHLClient):
    try:
        standings = await client.fetch_standings()
        log.info(f"standings: {len(standings.standings)} team(s)")
    except Exception as e:
        log.warning(f"poll_standings failed: {e}")


async def poll_player_stats(client: NHLClient):
    try:
        stats = await client.fetch_player_stats()
        log.info(f"player_stats: {len(stats.data)} skater(s)")
    except Exception as e:
        log.warning(f"poll_player_stats failed: {e}")


async def poll_play_by_play(client: NHLClient):
    try:
        game_ids = await client.fetch_live_game_ids()
        if not game_ids:
            log.info("play_by_play: no live games")
            return
        for game_id in game_ids:
            pbp = await client.fetch_play_by_play(game_id)
            log.info(f"play_by_play: game {game_id} — {len(pbp.plays)} event(s)")
    except Exception as e:
        log.warning(f"poll_play_by_play failed: {e}")


async def poll_reddit(client: RedditClient):
    try:
        await client.fetch_all_subreddits()
    except Exception as e:
        log.warning(f"poll_reddit failed: {e}")


async def poll_news(client: NewsClient):
    try:
        await client.fetch_hockey_news()
    except RuntimeError as e:
        log.warning(f"poll_news skipped: {e}")
    except Exception as e:
        log.warning(f"poll_news failed: {e}")


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

    nhl = NHLClient()
    reddit = RedditClient()
    news = NewsClient(redis_url=REDIS_URL)

    scheduler = AsyncIOScheduler()
    scheduler.add_job(poll_scores,       "interval", seconds=60,   args=[nhl])
    scheduler.add_job(poll_standings,    "interval", seconds=300,  args=[nhl])
    scheduler.add_job(poll_player_stats, "interval", seconds=600,  args=[nhl])
    scheduler.add_job(poll_play_by_play, "interval", seconds=30,   args=[nhl])
    scheduler.add_job(poll_reddit,       "interval", seconds=120,  args=[reddit])
    scheduler.add_job(poll_news,         "interval", seconds=900,  args=[news])

    scheduler.start()
    log.info("Scheduler started — polling NHL API, Reddit, and NewsAPI")

    try:
        await asyncio.Event().wait()
    finally:
        await nhl.close()
        await reddit.close()
        await news.close()


if __name__ == "__main__":
    asyncio.run(main())
