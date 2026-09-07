"""The browser gate refuses past a bounded wait; the page cache forgets on time."""
import asyncio
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from gate import Gate, PageCache  # noqa: E402


def test_the_gate_admits_up_to_its_limit_and_refuses_after_the_wait():
    async def scenario():
        gate = Gate(2)
        assert await gate.acquire(0.05) and await gate.acquire(0.05)
        assert gate.active == 2
        # A third waits its bounded time and is turned away, not hung.
        assert await gate.acquire(0.05) is False
        assert gate.waiting == 0
        gate.release()
        assert await gate.acquire(0.05) is True
        assert gate.active == 2

    asyncio.run(scenario())


def test_a_waiter_gets_the_slot_when_one_frees_in_time():
    async def scenario():
        gate = Gate(1)
        assert await gate.acquire(0.05)

        async def free_soon():
            await asyncio.sleep(0.02)
            gate.release()

        asyncio.create_task(free_soon())
        assert await gate.acquire(0.5) is True

    asyncio.run(scenario())


def test_the_cache_answers_within_ttl_and_forgets_after():
    now = [100.0]
    cache = PageCache(ttl=10, max_entries=8, clock=lambda: now[0])
    key = PageCache.key("https://a.test/p", ["markdown"], True)
    assert cache.get(key) is None
    cache.put(key, {"success": True, "data": {"markdown": "본문"}})
    now[0] = 109.0
    assert cache.get(key)["data"]["markdown"] == "본문"
    now[0] = 111.0
    assert cache.get(key) is None
    assert len(cache) == 0


def test_the_cache_key_ignores_format_order_but_not_the_main_flag():
    a = PageCache.key("https://a.test/p", ["html", "markdown"], True)
    b = PageCache.key("https://a.test/p", ["markdown", "html"], True)
    c = PageCache.key("https://a.test/p", ["markdown", "html"], False)
    assert a == b and a != c


def test_the_oldest_entry_goes_first():
    cache = PageCache(ttl=100, max_entries=2, clock=lambda: 0.0)
    for i in range(3):
        cache.put(PageCache.key(f"https://a.test/{i}", ["markdown"], True), {"i": i})
    assert cache.get(PageCache.key("https://a.test/0", ["markdown"], True)) is None
    assert cache.get(PageCache.key("https://a.test/2", ["markdown"], True)) == {"i": 2}
