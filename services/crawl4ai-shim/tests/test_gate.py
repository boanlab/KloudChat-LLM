"""The browser gate refuses past a bounded wait; page times stay within bounds; the page
cache forgets on time and does not keep an oversized scrape."""
import asyncio
import pathlib
import sys

import pytest

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from gate import Gate, PageCache, page_times  # noqa: E402


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


def test_page_times_default_when_absent_or_zero():
    assert page_times({}, 30000, 1000, 60000) == (30000, 0)
    assert page_times({"timeout": 0, "waitFor": None}, 30000, 1000, 60000) == (30000, 0)


def test_page_times_are_held_within_bounds():
    assert page_times({"timeout": 3_600_000}, 30000, 1000, 60000) == (60000, 0)
    assert page_times({"timeout": 5}, 30000, 1000, 60000) == (1000, 0)
    assert page_times({"timeout": -5}, 30000, 1000, 60000) == (1000, 0)
    # waitFor never outlasts the page timeout
    assert page_times({"timeout": 20000, "waitFor": 3_600_000}, 30000, 1000, 60000) == (20000, 20000)
    assert page_times({"waitFor": -10}, 30000, 1000, 60000) == (30000, 0)
    assert page_times({"timeout": "15000", "waitFor": 2500.7}, 30000, 1000, 60000) == (15000, 2500)


@pytest.mark.parametrize("payload", [
    {"timeout": "soon"}, {"timeout": [1]}, {"timeout": {"ms": 1}}, {"timeout": True},
    {"timeout": "nan"}, {"timeout": "inf"}, {"waitFor": "later"},
])
def test_page_times_refuse_what_is_not_a_number(payload):
    with pytest.raises(ValueError):
        page_times(payload, 30000, 1000, 60000)


def test_an_oversized_scrape_is_not_kept():
    cache = PageCache(ttl=100, max_entries=8, clock=lambda: 0.0, max_entry_bytes=1000)
    small = PageCache.key("https://a.test/s", ["markdown"], True)
    big = PageCache.key("https://a.test/b", ["rawHtml"], True)
    assert cache.put(small, {"success": True, "data": {"markdown": "가" * 300}}) is True
    # 400 Hangul characters are 1200 bytes of UTF-8
    assert cache.put(big, {"success": True, "data": {"markdown": "x", "rawHtml": "가" * 400}}) is False
    assert cache.get(small) is not None and cache.get(big) is None
    assert len(cache) == 1


def test_an_oversized_scrape_leaves_no_stale_copy():
    cache = PageCache(ttl=100, max_entries=8, clock=lambda: 0.0, max_entry_bytes=10)
    key = PageCache.key("https://a.test/p", ["markdown"], True)
    cache.put(key, {"data": {"markdown": "short"}})
    cache.put(key, {"data": {"markdown": "much longer than ten"}})
    assert cache.get(key) is None


def test_a_cache_without_an_entry_limit_keeps_everything():
    cache = PageCache(ttl=100, max_entries=8, clock=lambda: 0.0)
    key = PageCache.key("https://a.test/p", ["rawHtml"], True)
    assert cache.put(key, {"data": {"rawHtml": "x" * 5_000_000}}) is True
    assert cache.get(key) is not None
