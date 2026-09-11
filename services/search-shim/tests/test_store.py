"""The cache tells fresh from stale, the coalescer shares one call, the gate refuses on time."""
import asyncio
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from store import Coalescer, Gate, ResultCache  # noqa: E402


class Clock:
    def __init__(self) -> None:
        self.now = 0.0

    def __call__(self) -> float:
        return self.now


def test_the_key_ignores_case_and_spacing_in_the_query_only():
    a = ResultCache.key({"q": " Kubernetes  security ", "format": "json", "pageno": "1"})
    b = ResultCache.key({"pageno": "1", "q": "kubernetes security", "format": "json"})
    c = ResultCache.key({"q": "kubernetes security", "format": "json", "pageno": "2"})
    assert a == b and a != c


def test_an_entry_is_fresh_then_stale_then_gone():
    clock = Clock()
    cache = ResultCache(ttl=10, stale_ttl=100, max_entries=8, clock=clock)
    key = ResultCache.key({"q": "x"})
    cache.put(key, {"results": [1]})
    assert cache.fresh(key) == {"results": [1]}
    clock.now = 11
    assert cache.fresh(key) is None
    assert cache.stale(key) == {"results": [1]}
    clock.now = 101
    assert cache.stale(key) is None
    assert len(cache) == 0


def test_the_oldest_entry_goes_first_past_the_limit():
    cache = ResultCache(ttl=10, stale_ttl=10, max_entries=2)
    for q in ("a", "b", "c"):
        cache.put(ResultCache.key({"q": q}), q)
    assert cache.fresh(ResultCache.key({"q": "a"})) is None
    assert cache.fresh(ResultCache.key({"q": "c"})) == "c"


def test_identical_searches_in_flight_share_one_call():
    async def scenario():
        co = Coalescer()
        calls = 0
        started = asyncio.Event()

        async def slow():
            nonlocal calls
            calls += 1
            started.set()
            await asyncio.sleep(0.05)
            return {"results": [calls]}

        key = ResultCache.key({"q": "same"})
        first = asyncio.create_task(co.run(key, slow))
        await started.wait()
        assert co.inflight == 1
        second = asyncio.create_task(co.run(key, slow))
        assert await first == await second == {"results": [1]}
        assert calls == 1 and co.inflight == 0
        # A later search runs again.
        assert await co.run(key, slow) == {"results": [2]}

    asyncio.run(scenario())


def test_a_shared_failure_reaches_every_waiter():
    async def scenario():
        co = Coalescer()
        started = asyncio.Event()

        async def failing():
            started.set()
            await asyncio.sleep(0.02)
            raise RuntimeError("upstream")

        key = ResultCache.key({"q": "boom"})
        first = asyncio.create_task(co.run(key, failing))
        await started.wait()
        second = asyncio.create_task(co.run(key, failing))
        for t in (first, second):
            try:
                await t
            except RuntimeError as e:
                assert str(e) == "upstream"
            else:
                raise AssertionError("expected the failure")
        assert co.inflight == 0

    asyncio.run(scenario())


def test_a_failure_nobody_waited_for_is_still_retrieved():
    """The loop reports a task whose exception was never read; the coalescer reads it."""
    async def scenario():
        co = Coalescer()

        async def failing():
            raise RuntimeError("alone")

        key = ResultCache.key({"q": "alone"})
        try:
            await co.run(key, failing)
        except RuntimeError:
            pass
        assert co.inflight == 0

    unretrieved = []
    loop = asyncio.new_event_loop()
    loop.set_exception_handler(lambda _l, ctx: unretrieved.append(ctx))
    try:
        loop.run_until_complete(scenario())
        loop.run_until_complete(asyncio.sleep(0))
    finally:
        loop.close()
    assert unretrieved == []


def test_a_caller_that_leaves_does_not_cancel_the_shared_call():
    async def scenario():
        co = Coalescer()
        started = asyncio.Event()

        async def slow():
            started.set()
            await asyncio.sleep(0.05)
            return "done"

        key = ResultCache.key({"q": "shared"})
        leader = asyncio.create_task(co.run(key, slow))
        await started.wait()
        follower = asyncio.create_task(co.run(key, slow))
        await asyncio.sleep(0)
        leader.cancel()
        assert await follower == "done"
        assert co.inflight == 0

    asyncio.run(scenario())


def test_the_gate_admits_up_to_its_limit_and_refuses_after_the_wait():
    async def scenario():
        gate = Gate(2)
        assert await gate.acquire(0.05) and await gate.acquire(0.05)
        assert gate.active == 2
        assert await gate.acquire(0.05) is False
        assert gate.waiting == 0
        gate.release()
        assert await gate.acquire(0.05) is True

    asyncio.run(scenario())
