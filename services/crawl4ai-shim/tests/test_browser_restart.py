"""A browser that dies is replaced and the page tried once more; one that stays dead
fails /health so the container restarts. On 2026-10-05 the browser died and every scrape
failed for fifteen hours while /health said ok."""
import asyncio
import pathlib
import sys
from types import SimpleNamespace

import pytest

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
pytest.importorskip("crawl4ai")

import app  # noqa: E402

GONE = "Browser.new_context: Target page, context or browser has been closed"


class _Dead:
    async def arun(self, url, config):
        return SimpleNamespace(success=False, error_message=GONE)

    async def close(self):
        pass


class _Alive:
    async def arun(self, url, config):
        return SimpleNamespace(success=True, error_message="", markdown="# 본문\n\n내용",
                               metadata={"title": "t"}, status_code=200, redirected_url=url,
                               html="<p>내용</p>", cleaned_html="<p>내용</p>")


def _setup(monkeypatch, first, restarted):
    monkeypatch.setattr(app, "crawler", first)
    monkeypatch.setattr(app, "dead_in_a_row", 0)
    monkeypatch.setattr(app, "generation", 0)
    monkeypatch.setattr(app, "_refusal", lambda url: None)
    monkeypatch.setattr(app.cache, "get", lambda key: None)

    async def started(cfg):
        return restarted

    monkeypatch.setattr(app, "_started", started)


def test_a_dead_browser_is_restarted_and_the_page_read(monkeypatch):
    _setup(monkeypatch, _Dead(), _Alive())
    out = asyncio.run(app._scrape({"url": "https://example.com/a"}))
    assert out.get("success") is True
    assert app.generation == 1 and isinstance(app.crawler, _Alive)


def test_a_browser_that_stays_dead_fails_health(monkeypatch):
    _setup(monkeypatch, _Dead(), _Dead())
    for n in range(2):
        out = asyncio.run(app._scrape({"url": f"https://example.com/{n}"}))
        assert out.get("success") is False
    response = SimpleNamespace(status_code=200)
    body = asyncio.run(app.health(response))
    assert response.status_code == 503 and body["status"] == "browser_down"
