"""
JSONPlaceholder API client with async I/O and retry logic.

Design decisions:
- Async httpx for non-blocking network I/O (allows UI thread to remain
  responsive in future multi-threaded architectures).
- Pydantic v2 models for schema validation and IDE completion.
- tenacity for declarative retry with exponential backoff — avoids
  hand-rolled retry loops that make business logic harder to read.
- Explicit timeout config prevents indefinite hangs on flaky networks.
"""

from __future__ import annotations

import asyncio
from typing import Any

import httpx
from pydantic import BaseModel, Field, field_validator
from tenacity import (
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

from src.utils.logger import get_logger

logger = get_logger(__name__)

_BASE_URL = "https://jsonplaceholder.typicode.com"
_DEFAULT_TIMEOUT = 10.0  # seconds
_DEFAULT_RETRIES = 3


# ─── Data models ──────────────────────────────────────────────────────────────

class Post(BaseModel):
    """A single post from JSONPlaceholder /posts endpoint."""

    id: int
    user_id: int = Field(alias="userId")
    title: str
    body: str

    @field_validator("title", "body", mode="before")
    @classmethod
    def strip_whitespace(cls, v: Any) -> str:
        return str(v).strip() if v else ""

    def to_notepad_text(self) -> str:
        """Format post for Notepad output."""
        return (
            f"Title: {self.title}\n"
            f"{'─' * 60}\n"
            f"{self.body}\n"
        )

    model_config = {"populate_by_name": True}


# ─── Client ───────────────────────────────────────────────────────────────────

class PostsClient:
    """
    Async client for the JSONPlaceholder posts API.

    Usage::

        client = PostsClient()
        posts = asyncio.run(client.fetch_posts(limit=10))
    """

    def __init__(
        self,
        base_url: str = _BASE_URL,
        timeout: float = _DEFAULT_TIMEOUT,
        retries: int = _DEFAULT_RETRIES,
    ) -> None:
        self._base_url = base_url.rstrip("/")
        self._timeout = httpx.Timeout(timeout)
        self._retries = retries

    async def fetch_posts(self, limit: int = 10) -> list[Post]:
        """
        Fetch the first *limit* posts from /posts.

        Args:
            limit: Maximum number of posts to return. The API returns up to 100.

        Returns:
            List of Post objects, sorted by id ascending.

        Raises:
            httpx.HTTPStatusError: On 4xx/5xx after retries exhausted.
            httpx.RequestError:    On network errors after retries exhausted.
        """
        logger.info(f"Fetching {limit} posts from {self._base_url}/posts")
        raw = await self._get_with_retry(
            url=f"{self._base_url}/posts",
            params={"_limit": limit},
        )
        posts = [Post.model_validate(item) for item in raw]
        posts.sort(key=lambda p: p.id)
        logger.info(f"Retrieved {len(posts)} posts")
        return posts[:limit]

    async def fetch_post(self, post_id: int) -> Post:
        """Fetch a single post by ID."""
        raw = await self._get_with_retry(url=f"{self._base_url}/posts/{post_id}")
        return Post.model_validate(raw)

    # ─── Internal: async retry wrapper ────────────────────────────────────────

    async def _get_with_retry(
        self, url: str, params: dict | None = None
    ) -> Any:
        """
        Perform a GET request with exponential-backoff retry.

        tenacity decorators don't work cleanly on async methods defined
        inside a class, so we wrap the inner coroutine instead.
        """
        attempt = 0
        delay = 1.0

        while True:
            attempt += 1
            try:
                return await self._get(url, params)
            except (httpx.HTTPStatusError, httpx.RequestError) as e:
                if attempt >= self._retries:
                    logger.error(
                        f"GET {url} failed after {attempt} attempts: {e}"
                    )
                    raise
                wait = delay * (2 ** (attempt - 1))
                logger.warning(
                    f"GET {url} failed (attempt {attempt}/{self._retries}): "
                    f"{type(e).__name__}. Retrying in {wait:.1f}s…"
                )
                await asyncio.sleep(wait)

    async def _get(self, url: str, params: dict | None = None) -> Any:
        """Single GET request. Raises on non-2xx."""
        async with httpx.AsyncClient(timeout=self._timeout) as client:
            resp = await client.get(url, params=params)
            resp.raise_for_status()
            return resp.json()


# ─── Sync convenience wrapper ─────────────────────────────────────────────────

def fetch_posts_sync(limit: int = 10, **client_kwargs) -> list[Post]:
    """
    Synchronous convenience function — runs the async client in a new loop.
    Use this from non-async code (e.g., the workflow when run in a thread).
    """
    client = PostsClient(**client_kwargs)
    return asyncio.run(client.fetch_posts(limit=limit))
