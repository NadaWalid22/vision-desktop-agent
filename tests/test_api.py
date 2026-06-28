"""
Integration tests for the PostsClient.

Uses pytest-httpx (or httpx's own mock transport) to intercept HTTP calls
so tests run offline without hitting the real JSONPlaceholder API.
"""

from __future__ import annotations

import json

import pytest
import httpx

from src.api.posts_client import Post, PostsClient


# ─── Fixtures ─────────────────────────────────────────────────────────────────

SAMPLE_POSTS = [
    {"userId": 1, "id": i, "title": f"Post {i} title", "body": f"Body of post {i}"}
    for i in range(1, 21)
]


class _MockTransport(httpx.AsyncBaseTransport):
    """Intercept HTTP requests and return canned responses."""

    def __init__(self, payload: list, status_code: int = 200) -> None:
        self._payload = payload
        self._status_code = status_code

    async def handle_async_request(self, request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            self._status_code,
            content=json.dumps(self._payload).encode(),
            headers={"Content-Type": "application/json"},
        )


# We patch httpx.AsyncClient to use our mock transport
class _PatchedPostsClient(PostsClient):
    def __init__(self, payload, status_code=200, **kwargs):
        super().__init__(**kwargs)
        self._transport = _MockTransport(payload, status_code)

    async def _get(self, url, params=None):
        async with httpx.AsyncClient(
            transport=self._transport, timeout=self._timeout
        ) as client:
            resp = await client.get(url, params=params)
            resp.raise_for_status()
            return resp.json()


# ─── Post model tests ──────────────────────────────────────────────────────────

class TestPost:
    def test_parse_valid(self) -> None:
        data = {"userId": 1, "id": 1, "title": "Hello", "body": "World"}
        post = Post.model_validate(data)
        assert post.id == 1
        assert post.user_id == 1
        assert post.title == "Hello"
        assert post.body == "World"

    def test_strips_whitespace(self) -> None:
        data = {"userId": 1, "id": 1, "title": "  Hello  ", "body": "\nWorld\n"}
        post = Post.model_validate(data)
        assert post.title == "Hello"
        assert post.body == "World"

    def test_to_notepad_text(self) -> None:
        post = Post(id=1, user_id=1, title="Test", body="Some body")
        text = post.to_notepad_text()
        assert "Title: Test" in text
        assert "Some body" in text

    def test_model_alias(self) -> None:
        """userId field alias must work."""
        data = {"userId": 42, "id": 1, "title": "T", "body": "B"}
        post = Post.model_validate(data)
        assert post.user_id == 42


# ─── PostsClient tests ────────────────────────────────────────────────────────

@pytest.mark.asyncio
class TestPostsClient:
    async def test_fetch_posts_returns_limit(self) -> None:
        client = _PatchedPostsClient(SAMPLE_POSTS)
        posts = await client.fetch_posts(limit=10)
        assert len(posts) == 10

    async def test_fetch_posts_sorted_by_id(self) -> None:
        # Shuffle the payload
        shuffled = SAMPLE_POSTS[:]
        shuffled.reverse()
        client = _PatchedPostsClient(shuffled)
        posts = await client.fetch_posts(limit=10)
        ids = [p.id for p in posts]
        assert ids == sorted(ids)

    async def test_fetch_posts_validates_models(self) -> None:
        client = _PatchedPostsClient(SAMPLE_POSTS[:5])
        posts = await client.fetch_posts(limit=5)
        assert all(isinstance(p, Post) for p in posts)

    async def test_http_error_raises(self) -> None:
        client = _PatchedPostsClient([], status_code=500)
        client._retries = 1  # Don't wait for retries in unit test
        with pytest.raises(httpx.HTTPStatusError):
            await client.fetch_posts(limit=5)

    async def test_empty_response(self) -> None:
        client = _PatchedPostsClient([])
        posts = await client.fetch_posts(limit=10)
        assert posts == []

    async def test_limit_larger_than_response(self) -> None:
        client = _PatchedPostsClient(SAMPLE_POSTS[:3])
        posts = await client.fetch_posts(limit=10)
        assert len(posts) == 3  # Can't return more than available
