"""
Data pipeline engine = what keeps the RAG system fed with fresh NHL data
"""

import os
import logging
import time

import redis
from apscheduler.schedulers.background import BackgroundScheduler
from apscheduler.triggers.interval import IntervalTrigger
from qdrant_client import QdrantClient

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


def tick():
    log.info("tick")


# Main

def main():
    for attempt in range(3):
        try:
            check_connections()
            break
        except Exception as e:
            log.warning(f"Connection check failed (attempt number {attempt + 1}/3)")
            time.sleep(3)
    else:
        log.error("Could not reach Redis or Qdrant after 3 attempts.")
        raise SystemExit(1)

    scheduler = BackgroundScheduler()
    scheduler.add_job(tick, IntervalTrigger(seconds=60))
    scheduler.start()
    log.info("Scheduler started. Worker is running.")

    try:
        while True:
            time.sleep(1)
    finally:
        scheduler.shutdown()


if __name__ == "__main__":
    main()
