"""
NewsAPI client for hockey news articles.

Dev tier limits: 100 requests/day.
Budget is tracked in Redis with a key that expires at midnight.
Articles are cached in Redis (24h TTL) to avoid re-fetching duplicates.

Get a free API key at https://newsapi.org/register
Set NEWS_API_KEY in your .env file.
"""
import hashlib
import logging
import os
from datetime import date
from typing import Optional

import httpx
import redis.asyncio as aioredis
from pydantic import BaseModel

log = logging.getLogger(__name__)

NEWSAPI_BASE = "https://newsapi.org/v2"
DAILY_BUDGET = 90          # stay under the 100/day hard limit
CACHE_TTL = 86_400         # 24h in seconds
BUDGET_KEY_PREFIX = "newsapi:budget:"
ARTICLE_CACHE_PREFIX = "newsapi:article:"


# Pydantic models

class NewsArticle(BaseModel):
    source_name: str
    author: Optional[str] = None
    title: str
    description: Optional[str] = None
    url: str
    published_at: str
    content: Optional[str] = None


class NewsResponse(BaseModel):
    total_results: int
    articles: list[NewsArticle]


# Client

class NewsClient:
    def __init__(self, redis_url: str, api_key: Optional[str] = None):
        self._api_key = api_key or os.getenv("NEWS_API_KEY", "")
        self._redis = aioredis.from_url(redis_url, decode_responses=True)
        self._http = httpx.AsyncClient(
            timeout=10.0,
            follow_redirects=True,
            base_url=NEWSAPI_BASE,
        )

    async def close(self):
        await self._http.aclose()
        await self._redis.aclose()

    # Budget tracking

    def _budget_key(self) -> str:
        return f"{BUDGET_KEY_PREFIX}{date.today().isoformat()}"

    async def _requests_used_today(self) -> int:
        val = await self._redis.get(self._budget_key())
        return int(val) if val else 0

    async def _increment_budget(self):
        key = self._budget_key()
        pipe = self._redis.pipeline()
        pipe.incr(key)
        pipe.expireat(key, _next_midnight_ts())
        await pipe.execute()

    async def budget_remaining(self) -> int:
        return max(0, DAILY_BUDGET - await self._requests_used_today())

    # Article dedup cache

    def _article_key(self, url: str) -> str:
        digest = hashlib.sha1(url.encode()).hexdigest()[:12]
        return f"{ARTICLE_CACHE_PREFIX}{digest}"

    async def _is_cached(self, url: str) -> bool:
        return bool(await self._redis.exists(self._article_key(url)))

    async def _mark_cached(self, url: str):
        await self._redis.setex(self._article_key(url), CACHE_TTL, "1")

    # Fetch

    async def fetch_hockey_news(
        self,
        page_size: int = 20,
    ) -> NewsResponse:
        """
        Fetch latest hockey articles from NewsAPI.
        Returns only articles not seen in the last 24h.
        Raises RuntimeError if daily budget is exhausted.
        """
        if not self._api_key:
            raise RuntimeError("NEWS_API_KEY is not set — skipping NewsAPI poll")

        remaining = await self.budget_remaining()
        if remaining == 0:
            raise RuntimeError(f"NewsAPI daily budget exhausted ({DAILY_BUDGET} req/day)")

        log.debug(f"NewsAPI budget: {remaining} requests remaining today")

        resp = await self._http.get(
            "/everything",
            params={
                "q": "NHL OR hockey",
                "language": "en",
                "sortBy": "publishedAt",
                "pageSize": page_size,
                "apiKey": self._api_key,
            },
        )
        resp.raise_for_status()
        await self._increment_budget()

        raw = resp.json()
        all_articles = [
            NewsArticle(
                source_name=a["source"]["name"],
                author=a.get("author"),
                title=a["title"],
                description=a.get("description"),
                url=a["url"],
                published_at=a["publishedAt"],
                content=a.get("content"),
            )
            for a in raw.get("articles", [])
        ]

        # filter out already-seen articles
        new_articles = []
        for article in all_articles:
            if not await self._is_cached(article.url):
                new_articles.append(article)
                await self._mark_cached(article.url)

        log.info(
            f"news: {len(new_articles)} new article(s) "
            f"(skipped {len(all_articles) - len(new_articles)} cached)"
        )
        return NewsResponse(total_results=raw.get("totalResults", 0), articles=new_articles)


# Helper

def _next_midnight_ts() -> int:
    """Unix timestamp of next midnight UTC (for Redis EXPIREAT)."""
    import calendar
    from datetime import datetime, timezone
    tomorrow = date.today().toordinal() + 1
    midnight = datetime.fromordinal(tomorrow).replace(tzinfo=timezone.utc)
    return calendar.timegm(midnight.timetuple())
