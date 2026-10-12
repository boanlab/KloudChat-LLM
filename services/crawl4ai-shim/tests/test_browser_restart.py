"""A dead browser or driver is replaced and the page retried once; one that stays dead
fails /health and, past the exit limit, ends the process. Needs the image's
dependencies; skipped where they are absent (the lint job), run inside the image:

    docker run --rm -v $PWD/services/crawl4ai-shim:/src -w /src kloudchat-crawl4ai-shim \\
        python -m pytest -q tests
"""
import asyncio
import pathlib
import sys
from types import SimpleNamespace

import pytest

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
pytest.importorskip("crawl4ai")

import app  # noqa: E402

GONE = "Browser.new_context: Target page, context or browser has been closed"
# The 2026-10-12 outage: the Playwright driver died and every call failed like this
DRIVER_GONE = ("Browser.new_context: unable to perform operation on "
               "<WriteUnixTransport closed=True reading=False 0x70e95aa7e260>; the handler is closed")


class _Page:
    async def set_content(self, html):
        pass

    async def evaluate(self, expr):
        return "ok"


class _Context:
    async def new_page(self):
        return _Page()

    async def close(self):
        pass


class _Browser:
    def __init__(self, works=True):
        self.works = works

    def is_connected(self):
        return True

    async def new_context(self):
        if not self.works:
            raise RuntimeError(DRIVER_GONE)
        return _Context()


def _with_browser(crawler, works):
    crawler.crawler_strategy = SimpleNamespace(browser_manager=SimpleNamespace(
        browser=_Browser(works), playwright=None))
    return crawler


class _Dead:
    def __init__(self, error=GONE, raises=False):
        self.error, self.raises, self.calls = error, raises, 0
        _with_browser(self, works=False)

    async def arun(self, url, config):
        self.calls += 1
        await asyncio.sleep(0)
        if self.raises:
            raise RuntimeError(self.error)
        return SimpleNamespace(success=False, error_message=self.error)

    async def close(self):
        raise RuntimeError(DRIVER_GONE)


class _Alive:
    def __init__(self):
        _with_browser(self, works=True)

    async def arun(self, url, config):
        return SimpleNamespace(success=True, error_message="", markdown="# 본문\n\n내용",
                               metadata={"title": "t"}, status_code=200, redirected_url=url,
                               html="<p>내용</p>", cleaned_html="<p>내용</p>")

    async def close(self):
        pass


def _setup(monkeypatch, first, *restarted):
    monkeypatch.setattr(app, "crawler", first)
    monkeypatch.setattr(app, "dead_in_a_row", 0)
    monkeypatch.setattr(app, "generation", 0)
    monkeypatch.setattr(app, "restart_lock", asyncio.Lock())
    monkeypatch.setattr(app, "_refusal", lambda url: None)
    monkeypatch.setattr(app.cache, "get", lambda key: None)
    monkeypatch.setattr(app.cache, "put", lambda key, value: None)
    queue = list(restarted)
    starts = []

    async def started(cfg):
        starts.append(cfg)
        await asyncio.sleep(0)
        return queue.pop(0) if len(queue) > 1 else queue[0]

    monkeypatch.setattr(app, "_started", started)
    return starts


def _health():
    response = SimpleNamespace(status_code=200)
    body = asyncio.run(app.health(response))
    return response.status_code, body


def test_a_dead_browser_is_restarted_and_the_page_read(monkeypatch):
    _setup(monkeypatch, _Dead(), _Alive())
    out = asyncio.run(app._scrape({"url": "https://example.com/a"}))
    assert out.get("success") is True
    assert app.generation == 1 and isinstance(app.crawler, _Alive)
    assert app.dead_in_a_row == 0


@pytest.mark.parametrize("raises", [False, True])
def test_a_dead_driver_connection_is_restarted(monkeypatch, raises):
    _setup(monkeypatch, _Dead(DRIVER_GONE, raises=raises), _Alive())
    out = asyncio.run(app._scrape({"url": "https://example.com/a"}))
    assert out.get("success") is True
    assert app.generation == 1 and isinstance(app.crawler, _Alive)
    assert _health()[0] == 200


def test_concurrent_failures_restart_once(monkeypatch):
    starts = _setup(monkeypatch, _Dead(DRIVER_GONE), _Alive())

    async def many():
        return await asyncio.gather(*(app._scrape({"url": f"https://example.com/{n}"})
                                      for n in range(6)))

    outs = asyncio.run(many())
    assert all(o.get("success") for o in outs)
    assert len(starts) == 1 and app.generation == 1


def test_a_browser_that_still_renders_is_kept(monkeypatch):
    # One page's target closed, the browser itself is fine
    first = _with_browser(_Dead(), works=True)
    starts = _setup(monkeypatch, first, _Alive())
    out = asyncio.run(app._scrape({"url": "https://example.com/a"}))
    assert out.get("success") is False
    assert starts == [] and app.crawler is first


def test_a_browser_that_stays_dead_fails_health(monkeypatch):
    _setup(monkeypatch, _Dead(), _Dead())
    assert _health()[0] == 200
    for n in range(2):
        out = asyncio.run(app._scrape({"url": f"https://example.com/{n}"}))
        assert out.get("success") is False
    status, body = _health()
    assert status == 503 and body["status"] == "browser_down"


def test_health_is_503_without_a_browser(monkeypatch):
    _setup(monkeypatch, None, _Alive())
    status, body = _health()
    assert status == 503 and body["status"] == "browser_down"


def test_the_check_replaces_a_browser_that_cannot_render(monkeypatch):
    _setup(monkeypatch, _Dead(DRIVER_GONE), _Alive())
    monkeypatch.setattr(app, "dead_in_a_row", app.DEAD_LIMIT)
    assert _health()[0] == 503
    asyncio.run(app._check())
    assert isinstance(app.crawler, _Alive) and app.dead_in_a_row == 0
    assert _health()[0] == 200


def test_the_check_exits_when_restarts_do_not_help(monkeypatch):
    _setup(monkeypatch, _Dead(DRIVER_GONE), _Dead(DRIVER_GONE))
    exits = []
    monkeypatch.setattr(app, "_exit", lambda: exits.append(1))
    for _ in range(app.EXIT_LIMIT - 1):
        asyncio.run(app._check())
    assert exits == [] and _health()[0] == 503
    asyncio.run(app._check())
    assert exits == [1]


def test_a_real_browser_recovers_from_a_killed_driver(monkeypatch):
    """Chromium in the image: kill the Playwright driver as in the outage, then recover."""
    cfg = app.BrowserConfig(headless=True, verbose=False, light_mode=True, text_mode=True)

    async def run():
        monkeypatch.setattr(app, "browser_config", cfg)
        monkeypatch.setattr(app, "restart_lock", asyncio.Lock())
        monkeypatch.setattr(app, "generation", 0)
        monkeypatch.setattr(app, "dead_in_a_row", 0)
        monkeypatch.setattr(app, "crawler", await app._started(cfg))
        try:
            assert await app._renders(app.crawler)
            manager = app.crawler.crawler_strategy.browser_manager
            manager.playwright._impl_obj._connection._transport._proc.kill()
            await asyncio.sleep(1)
            assert not await app._renders(app.crawler)
            await app._check()
            assert app.generation == 1 and app.dead_in_a_row == 0
            assert await app._renders(app.crawler)
        finally:
            if app.crawler is not None:
                await app._close_quietly(app.crawler)

    asyncio.run(run())
