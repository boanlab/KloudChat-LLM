"""A malformed scrape request is answered with 400; a caller's timeout is bounded before it
reaches the browser. Needs the image's dependencies; skipped where they are absent (the lint
job), run inside the image:

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
from fastapi.testclient import TestClient  # noqa: E402

client = TestClient(app.app)


class _Recording:
    def __init__(self):
        self.configs = []
        self.html = "<p>본문</p>"

    async def arun(self, url, config):
        self.configs.append(config)
        await asyncio.sleep(0)
        return SimpleNamespace(success=True, error_message="", markdown="본문",
                               metadata={}, status_code=200, redirected_url=url,
                               html=self.html, cleaned_html="<p>본문</p>")


@pytest.fixture
def browser(monkeypatch):
    crawler = _Recording()
    monkeypatch.setattr(app, "crawler", crawler)
    monkeypatch.setattr(app, "dead_in_a_row", 0)
    monkeypatch.setattr(app, "_refusal", lambda url: None)
    monkeypatch.setattr(app, "cache", app.PageCache(60, 8, max_entry_bytes=1000))
    return crawler


@pytest.mark.parametrize("body", [
    [1, 2], "page", 42, None,
    {"url": "https://example.com/", "timeout": "soon"},
    {"url": "https://example.com/", "timeout": [1]},
    {"url": "https://example.com/", "waitFor": {"ms": 1}},
    {"url": ["https://example.com/"]},
    {"url": "https://example.com/", "formats": "markdown"},
    {"url": "https://example.com/", "formats": [1]},
])
def test_a_malformed_request_is_400(browser, body):
    r = client.post("/v2/scrape", json=body)
    assert r.status_code == 400
    assert r.json()["success"] is False and r.json()["error"]
    assert browser.configs == []


def test_a_long_timeout_is_cut_to_the_ceiling(browser):
    r = client.post("/v1/scrape", json={"url": "https://example.com/",
                                       "timeout": 3_600_000, "waitFor": 3_600_000})
    assert r.status_code == 200 and r.json()["success"] is True
    config = browser.configs[0]
    assert config.page_timeout == app.MAX_TIMEOUT_MS
    assert config.delay_before_return_html == app.MAX_TIMEOUT_MS / 1000.0


def test_a_tiny_timeout_is_raised_to_the_floor(browser):
    client.post("/v2/scrape", json={"url": "https://example.com/", "timeout": 1})
    assert browser.configs[0].page_timeout == app.MIN_TIMEOUT_MS


def test_an_oversized_page_is_served_but_not_cached(browser):
    browser.html = "<p>" + "가" * 1000 + "</p>"
    for _ in range(2):
        r = client.post("/v2/scrape", json={"url": "https://example.com/big",
                                           "formats": ["markdown", "rawHtml"]})
        assert r.json()["success"] is True
    assert len(browser.configs) == 2


def test_a_small_page_is_served_from_cache(browser, monkeypatch):
    monkeypatch.setattr(app, "cache", app.PageCache(60, 8, max_entry_bytes=1_000_000))
    for _ in range(2):
        assert client.post("/v2/scrape", json={"url": "https://example.com/s"}).json()["success"]
    assert len(browser.configs) == 1
