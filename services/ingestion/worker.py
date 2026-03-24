import logging
import os
import time

import redis
from apscheduler.schedulers.blocking import BlockingScheduler
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


if __name__ == "__main__":
    log.info("Ingestion worker starting...")

    # Retry connection check on startup (services may still be warming up)
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

    scheduler = BlockingScheduler()
    scheduler.add_job(tick, "interval", seconds=60)
    log.info("Scheduler started — ticking every 60s")
    scheduler.start()
