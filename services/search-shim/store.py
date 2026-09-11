"""Three guards between the callers and SearXNG.

`ResultCache` remembers a search for a while: the queries a model writes for
one topic converge, so the same words arrive many times an hour. An entry past
its fresh window is still kept for a longer stale window and handed out when
SearXNG cannot answer. `Coalescer` folds identical searches that are in flight
at the same moment into one upstream call. `Gate` caps how many searches reach
SearXNG at once — an engine bans on burst, not on volume — and makes the rest
wait a bounded time.
"""
from __future__ import annotations

import asyncio
import time
from collections import OrderedDict
from typing import Any, Awaitable, Callable


class Gate:
    """A counting semaphore with a bounded wait and live counters for /health."""

    def __init__(self, limit: int) -> None:
        self.limit = max(1, limit)
        self._slots = asyncio.Semaphore(self.limit)
        self.active = 0
        self.waiting = 0

    async def acquire(self, timeout: float) -> bool:
        self.waiting += 1
        try:
            await asyncio.wait_for(self._slots.acquire(), timeout=timeout)
        except asyncio.TimeoutError:
            return False
        finally:
            self.waiting -= 1
        self.active += 1
        return True

    def release(self) -> None:
        self.active -= 1
        self._slots.release()


class ResultCache:
    """Search answers by normalised request, newest kept. `fresh` for `ttl`
    seconds, still `stale` for `stale_ttl` seconds after that."""

    def __init__(self, ttl: float, stale_ttl: float, max_entries: int,
                 clock: Callable[[], float] = time.monotonic) -> None:
        self.ttl = ttl
        self.stale_ttl = max(ttl, stale_ttl)
        self.max_entries = max(1, max_entries)
        self._clock = clock
        self._rows: OrderedDict[tuple, tuple[float, Any]] = OrderedDict()

    @staticmethod
    def key(params: dict[str, str]) -> tuple:
        """Case and whitespace in the query never change what an engine returns
        enough to matter; every other parameter is taken as given."""
        q = " ".join(params.get("q", "").split()).casefold()
        rest = tuple(sorted((k, v) for k, v in params.items() if k != "q"))
        return (q, rest)

    def _age(self, key: tuple) -> float | None:
        row = self._rows.get(key)
        if row is None:
            return None
        return self._clock() - row[0]

    def fresh(self, key: tuple) -> Any | None:
        age = self._age(key)
        if age is None or age > self.ttl:
            return None
        self._rows.move_to_end(key)
        return self._rows[key][1]

    def stale(self, key: tuple) -> Any | None:
        age = self._age(key)
        if age is None:
            return None
        if age > self.stale_ttl:
            del self._rows[key]
            return None
        return self._rows[key][1]

    def put(self, key: tuple, value: Any) -> None:
        self._rows[key] = (self._clock(), value)
        self._rows.move_to_end(key)
        while len(self._rows) > self.max_entries:
            self._rows.popitem(last=False)

    def __len__(self) -> int:
        return len(self._rows)


class Coalescer:
    """Identical searches in flight at the same time share one upstream call.

    The call runs as its own task, so a caller that goes away does not cancel
    it for the others, and its outcome is always read, so a failure nobody
    waited for is not reported by the loop as never retrieved."""

    def __init__(self) -> None:
        self._inflight: dict[tuple, asyncio.Task] = {}

    @property
    def inflight(self) -> int:
        return len(self._inflight)

    def _done(self, key: tuple, task: asyncio.Task) -> None:
        if self._inflight.get(key) is task:
            del self._inflight[key]
        if not task.cancelled():
            task.exception()

    async def run(self, key: tuple, fn: Callable[[], Awaitable[Any]]) -> Any:
        task = self._inflight.get(key)
        if task is None:
            task = asyncio.create_task(fn())
            self._inflight[key] = task
            task.add_done_callback(lambda t: self._done(key, t))
        return await asyncio.shield(task)
