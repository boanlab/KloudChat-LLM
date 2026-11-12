"""Render concurrency cap, per-page time bounds and page cache for the shared browser.

`Gate`: pages rendering at once, with a bounded wait for the rest; past the cap
every tab slows toward the page timeout. `page_times`: a request's page timeout
and wait, kept within bounds so no caller holds a slot for long. `PageCache`:
successful scrapes, reused for a TTL; one too large to keep is not stored.
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


def _ms(value: Any, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, (int, float, str)):
        raise ValueError(f"{name} must be a number of milliseconds")
    try:
        return int(float(value))
    except (ValueError, OverflowError):
        raise ValueError(f"{name} must be a number of milliseconds") from None


def page_times(payload: dict[str, Any], default_ms: int,
               min_ms: int, max_ms: int) -> tuple[int, int]:
    """`(timeout, waitFor)` in ms from a request: timeout within [min_ms, max_ms]
    (default_ms when absent or 0), waitFor within [0, timeout]. ValueError on a
    value that is not a number."""
    timeout = _ms(payload.get("timeout") or default_ms, "timeout")
    timeout = min(max(timeout, min_ms), max_ms)
    wait = _ms(payload.get("waitFor") or 0, "waitFor")
    return timeout, min(max(wait, 0), timeout)


def _size(value: Any) -> int:
    """Bytes of text in a reply: every string, UTF-8 encoded."""
    if isinstance(value, str):
        return len(value.encode("utf-8", "surrogatepass"))
    if isinstance(value, dict):
        return sum(_size(v) for v in value.values())
    if isinstance(value, (list, tuple)):
        return sum(_size(v) for v in value)
    return 0


class PageCache:
    """Successful scrapes by (url, formats, main-content flag), newest kept,
    each good for `ttl` seconds; a scrape over `max_entry_bytes` of text is not kept."""

    def __init__(self, ttl: float, max_entries: int,
                 clock: Callable[[], float] = time.monotonic,
                 max_entry_bytes: int | None = None) -> None:
        self.ttl = ttl
        self.max_entries = max(1, max_entries)
        self.max_entry_bytes = max_entry_bytes
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

    def put(self, key: tuple, value: dict[str, Any]) -> bool:
        """Store `value`; False when it is too large to keep."""
        if self.max_entry_bytes is not None and _size(value) > self.max_entry_bytes:
            self._rows.pop(key, None)
            return False
        self._rows[key] = (self._clock(), value)
        self._rows.move_to_end(key)
        while len(self._rows) > self.max_entries:
            self._rows.popitem(last=False)
        return True

    def __len__(self) -> int:
        return len(self._rows)
