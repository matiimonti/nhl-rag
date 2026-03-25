"""
Reddit public JSON client — no auth required.

Uses Reddit's undocumented but stable public JSON endpoints:
  https://www.reddit.com/r/<subreddit>/new.json

Rate limit: Reddit allows ~1 req/2s for unauthenticated clients.
A custom User-Agent is required or Reddit returns 429/403.

Swap to PRAW once an OAuth app is registered.
"""
import asyncio
import logging
from typing import Optional

import httpx
from pydantic import BaseModel, Field

log = logging.getLogger(__name__)

REDDIT_BASE = "https://www.reddit.com"
USER_AGENT = "nhl-rag-bot/0.1 (research project)"

SUBREDDITS = ["hockey", "nhl"]


# Pydantic models

class RedditPost(BaseModel):
    id: str
    title: str
    selftext: str = ""
    url: str
    score: int
    num_comments: int = Field(alias="num_comments")
    created_utc: float
    author: str
    subreddit: str
    permalink: str

    model_config = {"populate_by_name": True}


class SubredditPage(BaseModel):
    subreddit: str
    posts: list[RedditPost]
    after: Optional[str]   # pagination cursor for next page


# Client

class RedditClient:
    def __init__(self, timeout: float = 10.0, req_delay: float = 2.0):
        self._client = httpx.AsyncClient(
            timeout=timeout,
            follow_redirects=True,
            headers={"User-Agent": USER_AGENT},
        )
        self._req_delay = req_delay  # seconds between requests (rate limit)

    async def close(self):
        await self._client.aclose()

    async def fetch_new_posts(
        self,
        subreddit: str,
        limit: int = 25,
        after: Optional[str] = None,
    ) -> SubredditPage:
        """
        Fetch the newest posts from a subreddit.

        Args:
            subreddit: e.g. "hockey" or "nhl"
            limit:     max posts to fetch (Reddit cap: 100)
            after:     pagination cursor from a previous response

        Returns:
            SubredditPage with posts and next cursor.
        """
        url = f"{REDDIT_BASE}/r/{subreddit}/new.json"
        params: dict = {"limit": limit, "raw_json": 1}
        if after:
            params["after"] = after

        for attempt in range(3):
            resp = await self._client.get(url, params=params)

            if resp.status_code == 429:
                retry_after = int(resp.headers.get("Retry-After", 10))
                log.warning(f"Reddit 429 on r/{subreddit} — waiting {retry_after}s")
                await asyncio.sleep(retry_after)
                continue

            resp.raise_for_status()
            break
        else:
            raise RuntimeError(f"Reddit rate limit exceeded for r/{subreddit} after 3 attempts")

        body = resp.json()
        children = body["data"]["children"]
        posts = [RedditPost.model_validate(c["data"]) for c in children]

        return SubredditPage(
            subreddit=subreddit,
            posts=posts,
            after=body["data"].get("after"),
        )

    async def fetch_all_subreddits(
        self,
        limit: int = 25,
    ) -> list[SubredditPage]:
        """
        Fetch new posts from all tracked subreddits with a delay between
        requests to stay within Reddit's rate limit.
        """
        pages = []
        for i, subreddit in enumerate(SUBREDDITS):
            if i > 0:
                await asyncio.sleep(self._req_delay)
            try:
                page = await self.fetch_new_posts(subreddit, limit=limit)
                pages.append(page)
                log.info(f"reddit r/{subreddit}: {len(page.posts)} post(s)")
            except Exception as e:
                log.warning(f"reddit r/{subreddit} failed: {e}")
        return pages
