"""Embedding model choice: preference with fallback, per-collection pinning, health probe.

Transport-free: the gateway call is passed in as `request(model, texts, timeout)`,
returning one vector per text or raising.
"""
from __future__ import annotations

import asyncio
import logging
import time
from typing import Awaitable, Callable, Optional

log = logging.getLogger("index-shim")

Request = Callable[[str, list[str], Optional[float]], Awaitable[list[list[float]]]]


def gateway_name(model: str) -> str:
    """The gateway's strict-local name for a model: a collection embedded as
    local/<id> keeps working, since the gateway registers only strict-local/<id>."""
    return "strict-local/" + model.removeprefix("local/") if model.startswith("local/") else model


class EmbeddingUnavailable(Exception):
    """No candidate model answered."""


class Embedder:
    """Embeds through the gateway, remembering which model answered."""

    def __init__(self, models: list[str], request: Request) -> None:
        self.models = list(models)
        self.model: Optional[str] = None
        self._request = request

    def candidates(self, model: Optional[str] = None) -> list[str]:
        """Models to try in order: the pinned one alone, else the preference first."""
        if model:
            return [model]
        return ([self.model] if self.model else []) + [
            m for m in self.models if m != self.model
        ]

    async def embed(self, texts: list[str], *, model: Optional[str] = None,
                    remember: bool = True,
                    timeout: Optional[float] = None) -> tuple[list[list[float]], str]:
        """Vectors and the model that produced them.

        Args:
            model: one model, no fallback (a collection's own).
            remember: cache the answering model as the preference for the next
                unpinned call; off for health checks.
            timeout: per-request ceiling in seconds; None for the transport default."""
        if not texts:
            return [], model or self.model or ""
        last: str = "no embedding model configured"
        for candidate in self.candidates(model):
            try:
                vectors = await self._request(candidate, texts, timeout)
                if len(vectors) != len(texts):
                    raise ValueError(f"{len(vectors)} vectors for {len(texts)} inputs")
                if remember:
                    self.model = candidate
                return [list(map(float, v)) for v in vectors], candidate
            except Exception as exc:  # noqa: BLE001
                last = f"{candidate}: {exc}"
                log.info("embedding via %s failed: %s", candidate, exc)
                # A failed model loses its cached preference.
                if remember and self.model == candidate:
                    self.model = None
        raise EmbeddingUnavailable(f"embeddings unavailable ({last})")


#: A collection's embedding model: the one holding most rows
COLLECTION_MODEL_SQL = """
SELECT embed_model
  FROM chunks
 WHERE collection = $1 AND embed_model <> ''
 GROUP BY embed_model
 ORDER BY count(*) DESC
 LIMIT 1
"""


async def collection_model(conn, collection: str) -> str:
    """Model the collection is indexed with, '' for an empty collection."""
    return await conn.fetchval(COLLECTION_MODEL_SQL, collection) or ""


class EmbedProbe:
    """Embedding availability for /health: one probe per `ttl`, waited on for at most `wait`.

    A probe outliving `wait` keeps running in the background and the last result
    is reported; the backend is probed at most once per `ttl`."""

    def __init__(self, check: Callable[[], Awaitable[str]], *, ttl: float, wait: float,
                 clock: Callable[[], float] = time.monotonic) -> None:
        self._check = check
        self._ttl = ttl
        self._wait = wait
        self._clock = clock
        self._next = 0.0
        self._task: Optional[asyncio.Task] = None
        self.model = ""

    async def _run(self) -> None:
        try:
            self.model = await self._check()
        except Exception as exc:  # noqa: BLE001
            log.info("health: embedding probe failed: %s", exc)
            self.model = ""

    async def current(self) -> str:
        """Answering model, '' when none answered."""
        now = self._clock()
        if (self._task is None or self._task.done()) and now >= self._next:
            self._next = now + self._ttl
            self._task = asyncio.ensure_future(self._run())
        if self._task is not None and not self._task.done():
            try:
                await asyncio.wait_for(asyncio.shield(self._task), self._wait)
            except asyncio.TimeoutError:
                pass
        return self.model
