"""Firecrawl-compatible scrape API over Crawl4AI, for KloudChat's document fetch.

Endpoints:
  POST /v2/scrape, /v1/scrape, /v0/scrape
  GET  /health

Request (firecrawl v2, the fields honoured): url, formats ["markdown", "html",
"rawHtml"], timeout (ms), waitFor (ms), onlyMainContent.
Response: {"success": true, "data": {"markdown", "html", "rawHtml", "metadata"}}
or {"success": false, "error"}.

One persistent headless Chromium crawler, shared by every request. At most
MAX_CONCURRENT_PAGES render at once (the rest wait up to QUEUE_TIMEOUT_MS and
are then told "busy"), and a successful scrape is reused for CACHE_TTL_S.

Page addresses are logged at DEBUG only; a warning names the host.
"""
from __future__ import annotations

import asyncio
import logging
import os
from contextlib import asynccontextmanager
from typing import Any
from urllib.parse import urlsplit

import netguard
from adultlist import AdultList
from crawl4ai import (
    AsyncWebCrawler,
    BrowserConfig,
    CacheMode,
    CrawlerRunConfig,
)
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from gate import Gate, PageCache

LOG_LEVEL = os.environ.get("LOG_LEVEL", "INFO").upper()
logging.basicConfig(level=LOG_LEVEL,
                    format="%(asctime)s %(levelname)s %(name)s %(message)s")
LOG = logging.getLogger("crawl4ai-shim")

DEFAULT_TIMEOUT_MS = int(os.environ.get("DEFAULT_TIMEOUT_MS", "30000"))
USER_AGENT = os.environ.get(
    "USER_AGENT",
    "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 KloudChat/1.0",
)

# Bearer token the gateway injects. Empty disables the check.
API_KEY = os.environ.get("SCRAPER_API_KEY", "")

# Pages rendering at once, and how long a request waits for a slot before it is
# refused. One Chromium on an 8-core host renders about eight pages in the time
# it takes to render one; beyond that every page slows towards its own timeout.
MAX_CONCURRENT_PAGES = int(os.environ.get("MAX_CONCURRENT_PAGES", "8"))
QUEUE_TIMEOUT_MS = int(os.environ.get("QUEUE_TIMEOUT_MS", "15000"))
# A successful scrape is answered from memory for this long.
CACHE_TTL_S = int(os.environ.get("CACHE_TTL_S", "900"))
CACHE_MAX_ENTRIES = int(os.environ.get("CACHE_MAX_ENTRIES", "512"))
# Hosts file of adult sites a scrape refuses; baked into the image.
ADULT_HOSTS_FILE = os.environ.get("ADULT_HOSTS_FILE", "/app/adult-hosts.txt")

crawler: AsyncWebCrawler | None = None
adult = AdultList()
gate = Gate(MAX_CONCURRENT_PAGES)
cache = PageCache(CACHE_TTL_S, CACHE_MAX_ENTRIES)


@asynccontextmanager
async def lifespan(_app: FastAPI):
    global crawler, adult
    try:
        adult = AdultList.from_file(ADULT_HOSTS_FILE)
        LOG.info("adult site list: %d domains from %s", len(adult), ADULT_HOSTS_FILE)
    except OSError as e:
        LOG.error("adult site list unreadable, running without one: %s", e)
    LOG.info("starting Crawl4AI (headless Chromium)")
    cfg = BrowserConfig(
        headless=True,
        verbose=False,
        user_agent=USER_AGENT,
        java_script_enabled=True,
        light_mode=True,
        # Images, fonts and media are never handed to the model — only the page's
        # markdown is — so the browser does not download them. Scripts still run.
        text_mode=True,
    )
    crawler = AsyncWebCrawler(config=cfg)
    await crawler.start()
    LOG.info("Crawl4AI ready")
    try:
        yield
    finally:
        LOG.info("shutting down Crawl4AI")
        await crawler.close()


app = FastAPI(lifespan=lifespan)


@app.get("/health")
async def health() -> dict[str, Any]:
    return {
        "status": "ok" if crawler is not None else "starting",
        "backend": "crawl4ai",
        "pages": {"active": gate.active, "waiting": gate.waiting, "limit": gate.limit},
        "cache": {"entries": len(cache), "ttl_s": CACHE_TTL_S},
        "adult_list": {"domains": len(adult)},
    }


def _refusal(url: str) -> str | None:
    """Why `url` may not be scraped: off the network, or an adult site."""
    return netguard.refusal(url) or adult.refusal(url)


def _site(url: str) -> str:
    """The host alone, for a log line."""
    try:
        return urlsplit(url).hostname or "?"
    except ValueError:
        return "?"


def _build_metadata(result: Any, url: str) -> dict[str, Any]:
    md = result.metadata if getattr(result, "metadata", None) else {}
    return {
        "title": md.get("title") if isinstance(md, dict) else None,
        "description": md.get("description") if isinstance(md, dict) else None,
        "language": md.get("language") if isinstance(md, dict) else None,
        "sourceURL": url,
        "statusCode": getattr(result, "status_code", None),
    }


async def _scrape(payload: dict[str, Any]) -> dict[str, Any]:
    url = payload.get("url")
    if not url:
        return {"success": False, "error": "url is required"}
    # Judged by resolved address and host before the browser opens it: this service is
    # reached without authentication and sits beside every other internal service.
    refused = await asyncio.to_thread(_refusal, url)
    if refused:
        LOG.warning("scrape refused for %s: %s", _site(url), refused)
        return {"success": False, "error": refused}

    formats = payload.get("formats") or ["markdown"]
    timeout_ms = payload.get("timeout") or DEFAULT_TIMEOUT_MS
    wait_for_ms = payload.get("waitFor") or 0
    only_main = bool(payload.get("onlyMainContent", True))

    excluded_tags = ["script", "style", "nav", "footer", "iframe", "noscript"] \
        if only_main else None

    run_cfg = CrawlerRunConfig(
        cache_mode=CacheMode.BYPASS,
        page_timeout=int(timeout_ms),
        delay_before_return_html=wait_for_ms / 1000.0 if wait_for_ms else 0,
        excluded_tags=excluded_tags,
        word_count_threshold=10,
        only_text=False,
        # Crawl4AI's own progress lines print the address
        verbose=False,
    )

    cache_key = PageCache.key(url, formats, only_main)
    if (hit := cache.get(cache_key)) is not None:
        LOG.debug("scrape %s served from cache", url)
        return hit

    LOG.debug("scrape %s (formats=%s, timeout=%dms, main=%s)",
              url, formats, timeout_ms, only_main)

    if not await gate.acquire(QUEUE_TIMEOUT_MS / 1000.0):
        LOG.warning("scrape refused, browser busy (%d rendering, %d waiting): %s",
                    gate.active, gate.waiting, _site(url))
        return {"success": False, "error": "busy: too many pages rendering"}
    try:
        result = await asyncio.wait_for(
            crawler.arun(url=url, config=run_cfg),
            timeout=(timeout_ms / 1000.0) + 10,
        )
    except asyncio.TimeoutError:
        LOG.warning("scrape timeout on %s", _site(url))
        return {"success": False, "error": "scrape timeout"}
    except Exception as e:
        LOG.warning("scrape error on %s: %r", _site(url), e)
        return {"success": False, "error": f"crawl4ai error: {e}"}
    finally:
        gate.release()

    if not getattr(result, "success", False):
        err = getattr(result, "error_message", None) or "crawl failed"
        LOG.warning("scrape failed on %s: %s", _site(url), err)
        return {"success": False, "error": err}
    # The browser follows redirects on its own; the address it ended up at is judged too.
    final_url = getattr(result, "redirected_url", None) or url
    if final_url != url:
        refused = await asyncio.to_thread(_refusal, final_url)
        if refused:
            LOG.warning("scrape refused after redirect %s -> %s: %s",
                        _site(url), _site(final_url), refused)
            return {"success": False, "error": refused}

    data: dict[str, Any] = {}

    markdown_obj = getattr(result, "markdown", None)
    if markdown_obj is not None:
        # fit_markdown: after the main-content filter; raw_markdown: plain html→md.
        if hasattr(markdown_obj, "fit_markdown") and markdown_obj.fit_markdown:
            data["markdown"] = markdown_obj.fit_markdown
        elif hasattr(markdown_obj, "raw_markdown"):
            data["markdown"] = markdown_obj.raw_markdown
        else:
            data["markdown"] = str(markdown_obj)

    if any(f in formats for f in ("html", "cleanedHtml")):
        data["html"] = getattr(result, "cleaned_html", None) or ""
    if "rawHtml" in formats:
        data["rawHtml"] = getattr(result, "html", None) or ""

    data["metadata"] = _build_metadata(result, url)

    response = {"success": True, "data": data}
    cache.put(cache_key, response)
    return response


async def _handle(req: Request) -> JSONResponse:
    if API_KEY and req.headers.get("authorization") != f"Bearer {API_KEY}":
        return JSONResponse({"success": False, "error": "unauthorized"},
                            status_code=401)
    try:
        payload = await req.json()
    except Exception as e:
        return JSONResponse({"success": False, "error": f"invalid json: {e}"},
                            status_code=400)
    return JSONResponse(await _scrape(payload))


@app.post("/v2/scrape")
async def scrape_v2(req: Request) -> JSONResponse:
    return await _handle(req)


@app.post("/v1/scrape")
async def scrape_v1(req: Request) -> JSONResponse:
    return await _handle(req)


@app.post("/v0/scrape")
async def scrape_v0(req: Request) -> JSONResponse:
    return await _handle(req)
