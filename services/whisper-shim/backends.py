"""Which whisper backend transcribes next, and the walk past unreachable ones.

Transport-free: health probes and requests are passed in. Healthy backends go
first, fewest in-flight requests first; the rest follow in configured order. A
backend that refused the connection counts as down until the next probe round.
"""
from __future__ import annotations

import asyncio
import logging
import time
from typing import Any, Awaitable, Callable

LOG = logging.getLogger("whisper-shim")


class Unreachable(Exception):
    """No backend accepted the connection."""

    def __init__(self, backends: list[str]) -> None:
        super().__init__(", ".join(backends))
        self.backends = backends


class Backends:
    def __init__(self, urls: list[str], probe: Callable[[str], Awaitable[bool]], *,
                 ttl: float, clock: Callable[[], float] = time.monotonic) -> None:
        self.urls = list(urls)
        self.health: dict[str, bool] = {}
        self.inflight: dict[str, int] = {u: 0 for u in self.urls}
        self._probe = probe
        self._ttl = ttl
        self._clock = clock
        self._checked_at = 0.0
        self._lock = asyncio.Lock()

    async def refresh(self) -> None:
        """Probe every backend if the cache is stale; concurrent callers share one round."""
        async with self._lock:
            if self.health and self._clock() < self._checked_at + self._ttl:
                return
            results = await asyncio.gather(*(self._probe(u) for u in self.urls))
            self.health = dict(zip(self.urls, results, strict=True))
            self._checked_at = self._clock()
            LOG.info("Whisper health: %s",
                     ", ".join(f"{u}={'up' if ok else 'down'}" for u, ok in self.health.items()))

    async def order(self) -> list[str]:
        """Backends to try, best first."""
        if len(self.urls) == 1:
            return list(self.urls)
        await self.refresh()
        healthy = sorted((u for u in self.urls if self.health.get(u)),
                         key=lambda u: self.inflight[u])
        if not healthy:
            LOG.warning("All whisper backends unhealthy; trying them in configured order")
        return healthy + [u for u in self.urls if u not in healthy]

    def mark_down(self, url: str) -> None:
        self.health[url] = False

    async def send(self, request: Callable[[str], Awaitable[Any]],
                   unreachable: tuple[type[BaseException], ...]) -> tuple[str, Any]:
        """(backend, result) from the first backend that accepts the connection.

        Errors in `unreachable` move on to the next backend; any other error is raised."""
        refused: list[str] = []
        for url in await self.order():
            self.inflight[url] += 1
            try:
                return url, await request(url)
            except unreachable as e:
                # Never reached the backend: nothing transcribed, the next one may be up
                LOG.warning("whisper backend %s unreachable: %s", url, e)
                self.mark_down(url)
                refused.append(url)
            finally:
                self.inflight[url] = max(0, self.inflight[url] - 1)
        raise Unreachable(refused)


def parse_urls(whisper_urls: str, placed: str, model_id: str = "whisper-large-v3") -> list[str]:
    """Backend URLs: the hand-run WHISPER_URLS, then the scheduler's VLLM_PLACED
    entries for ``model_id`` (``id=url``, ``local/`` prefix optional)."""
    urls = [u.strip().rstrip("/") for u in whisper_urls.split(",") if u.strip()]
    for item in placed.split(","):
        name, sep, url = item.strip().partition("=")
        url = url.strip().rstrip("/")
        if sep and url and name.strip().removeprefix("local/") == model_id and url not in urls:
            urls.append(url)
    return urls
