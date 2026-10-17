"""Which backend draws next when one served name runs on several nodes.

Each vllm-omni server draws one picture at a time and queues the rest, so the
next request goes to the backend with the fewest pictures in flight from this
shim (the only caller). Ties rotate. A backend that refused the connection sits
out for a cooldown unless every backend is out.
"""
from __future__ import annotations

import time
from typing import Callable


class Pool:
    def __init__(self, urls: list[str], cooldown: float = 30.0,
                 clock: Callable[[], float] = time.monotonic) -> None:
        self.urls = list(urls)
        self.busy = {u: 0 for u in self.urls}
        self._down_until = {u: 0.0 for u in self.urls}
        self._cooldown = cooldown
        self._clock = clock
        self._turn = 0

    def order(self) -> list[str]:
        """Backends to try, best first."""
        now = self._clock()
        n = len(self.urls)
        turn = self._turn
        self._turn = (turn + 1) % max(1, n)
        return sorted(
            self.urls,
            key=lambda u: (
                self._down_until[u] > now,
                self.busy[u],
                (self.urls.index(u) - turn) % n,
            ),
        )

    def mark_down(self, url: str) -> None:
        self._down_until[url] = self._clock() + self._cooldown

    def mark_up(self, url: str) -> None:
        self._down_until[url] = 0.0
