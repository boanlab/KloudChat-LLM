"""Two small guards in front of the one shared browser.

`Gate` caps how many pages render at once and makes the rest wait a bounded
time — past the cap every tab used to slow every other until all hit the page
timeout together. `PageCache` remembers a successful scrape for a while: the
top search results are the same pages for everyone asking about the same
thing that hour.
"""
from __future__ import annotations

import asyncio
import time
from collections import OrderedDict
from typing import Any, Callable


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


class PageCache:
    """Successful scrapes by (url, formats, main-content flag), newest kept,
    each good for `ttl` seconds."""

    def __init__(self, ttl: float, max_entries: int,
                 clock: Callable[[], float] = time.monotonic) -> None:
        self.ttl = ttl
        self.max_entries = max(1, max_entries)
        self._clock = clock
        self._rows: OrderedDict[tuple, tuple[float, dict[str, Any]]] = OrderedDict()

    @staticmethod
    def key(url: str, formats: list[str], only_main: bool) -> tuple:
        return (url, tuple(sorted(formats)), bool(only_main))

    def get(self, key: tuple) -> dict[str, Any] | None:
        row = self._rows.get(key)
        if row is None:
            return None
        stored_at, value = row
        if self._clock() - stored_at > self.ttl:
            del self._rows[key]
            return None
        self._rows.move_to_end(key)
        return value

    def put(self, key: tuple, value: dict[str, Any]) -> None:
        self._rows[key] = (self._clock(), value)
        self._rows.move_to_end(key)
        while len(self._rows) > self.max_entries:
            self._rows.popitem(last=False)

    def __len__(self) -> int:
        return len(self._rows)
