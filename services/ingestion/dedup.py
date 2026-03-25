"""
Deduplication layer for the ingest pipeline.

Each raw document is hashed with SHA-256 over:
  source + sorted JSON payload (includes content fields and document timestamp)

The hash is stored in Redis with a configurable TTL. If the same document
arrives again within the TTL window, it is silently dropped before hitting
the stream — no duplicate embeddings in Qdrant.

TTL defaults:
  - play-by-play events: 6h  (game finishes, no need to keep longer)
  - scores / standings:  1h  (update frequently, allow re-publish after state change)
  - reddit / news:       24h (articles don't change)
  - player stats:        1h  (stats update throughout the day)
"""
import hashlib
import json
import logging

import redis.asyncio as aioredis

log = logging.getLogger(__name__)

DEDUP_PREFIX = "dedup:"

# TTL in seconds per source type
_TTL: dict[str, int] = {
    "nhl_play_by_play": 6 * 3600,
    "nhl_scores":       3600,
    "nhl_standings":    3600,
    "nhl_player_stats": 3600,
    "reddit_hockey":    24 * 3600,
    "reddit_nhl":       24 * 3600,
    "news":             24 * 3600,
}
_DEFAULT_TTL = 3600


class Deduplicator:
    def __init__(self, redis_client: aioredis.Redis):
        self._r = redis_client

    def _hash(self, source: str, payload: dict) -> str:
        """SHA-256 of source + deterministic JSON serialisation of payload."""
        content = source + json.dumps(payload, sort_keys=True, default=str)
        return hashlib.sha256(content.encode()).hexdigest()

    async def is_duplicate(self, source: str, payload: dict) -> bool:
        """Return True if this document has been seen recently."""
        key = DEDUP_PREFIX + self._hash(source, payload)
        return bool(await self._r.exists(key))

    async def mark_seen(self, source: str, payload: dict):
        """Record the document hash in Redis with the appropriate TTL."""
        key = DEDUP_PREFIX + self._hash(source, payload)
        ttl = _TTL.get(source, _DEFAULT_TTL)
        await self._r.setex(key, ttl, "1")

    async def check_and_mark(self, source: str, payload: dict) -> bool:
        """
        Atomic check-and-mark.
        Returns True if the document is a duplicate (should be skipped).
        Returns False if new (caller should publish, then the hash is recorded).
        """
        key = DEDUP_PREFIX + self._hash(source, payload)
        ttl = _TTL.get(source, _DEFAULT_TTL)
        # SET key 1 EX ttl NX — only sets if key does not exist
        inserted = await self._r.set(key, "1", ex=ttl, nx=True)
        is_dup = inserted is None  # None means key already existed
        if is_dup:
            log.debug(f"dedup: dropped duplicate [{source}]")
        return is_dup
