"""Firecrawl-compatible scrape API powered by Crawl4AI.

KloudChat doesn't have a Firecrawl Cloud subscription and uploading internal
queries to a third-party service is off-limits. The client's web_search only
natively supports a small set of scrapers (firecrawl / serper / jina / cohere),
so this shim implements the firecrawl HTTP surface using Crawl4AI underneath.

Endpoints implemented:
  POST /v2/scrape, /v1/scrape, /v0/scrape   ← firecrawl-compatible clients post here
  GET  /health

Request body (firecrawl v2 schema, partial — only the fields clients actually send):
  {
    "url":            "https://...",
    "formats":        ["markdown", "rawHtml"],
    "timeout":        7500,                  // ms
    "onlyMainContent": true,
    "waitFor":        0,                     // ms after load before extracting
    "headers":        {...},                 // optional extra request headers
    "skipTlsVerification": false,
    "mobile":         false,
    "blockAds":       false,
    "parsePDF":       true,
    ...
  }

Response body:
  {"success": true, "data": {"markdown": "...", "html": "...", "rawHtml": "...",
                              "metadata": {"title", "description", "language",
                                            "sourceURL", "statusCode"}}}
  {"success": false, "error": "..."}

The shim warms up one persistent AsyncWebCrawler at startup (single Chromium
context) and serializes requests through it. For low-volume use that
is sufficient; if we ever need concurrency we can run multiple replicas or
fork off an async pool.
"""
from __future__ import annotations

import asyncio
import logging
import os
from contextlib import asynccontextmanager
from typing import Any

from crawl4ai import (
    AsyncWebCrawler,
    BrowserConfig,
    CacheMode,
    CrawlerRunConfig,
)
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse

LOG_LEVEL = os.environ.get("LOG_LEVEL", "INFO").upper()
logging.basicConfig(level=LOG_LEVEL,
                    format="%(asctime)s %(levelname)s %(name)s %(message)s")
LOG = logging.getLogger("crawl4ai-shim")

# Tuneables (env-overridable).
DEFAULT_TIMEOUT_MS = int(os.environ.get("DEFAULT_TIMEOUT_MS", "30000"))
USER_AGENT = os.environ.get(
    "USER_AGENT",
    "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 KloudChat/1.0",
)

# The gateway injects this on every proxied request and it never leaves that hop.
# Empty disables the check — the shim stays runnable on its own, which is how the
# healthcheck and a local `docker run` reach it.
API_KEY = os.environ.get("SCRAPER_API_KEY", "")

crawler: AsyncWebCrawler | None = None


@asynccontextmanager
async def lifespan(_app: FastAPI):
    global crawler
    LOG.info("starting Crawl4AI (headless Chromium)")
    cfg = BrowserConfig(
        headless=True,
        verbose=False,
        user_agent=USER_AGENT,
        java_script_enabled=True,
        light_mode=True,
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
    }


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
    )

    LOG.info("scrape %s (formats=%s, timeout=%dms, main=%s)",
             url, formats, timeout_ms, only_main)

    try:
        result = await asyncio.wait_for(
            crawler.arun(url=url, config=run_cfg),
            timeout=(timeout_ms / 1000.0) + 10,
        )
    except asyncio.TimeoutError:
        LOG.warning("scrape timeout on %s", url)
        return {"success": False, "error": "scrape timeout"}
    except Exception as e:
        LOG.warning("scrape error on %s: %r", url, e)
        return {"success": False, "error": f"crawl4ai error: {e}"}

    if not getattr(result, "success", False):
        err = getattr(result, "error_message", None) or "crawl failed"
        LOG.warning("scrape failed on %s: %s", url, err)
        return {"success": False, "error": err}

    data: dict[str, Any] = {}

    markdown_obj = getattr(result, "markdown", None)
    if markdown_obj is not None:
        # crawl4ai returns a MarkdownGenerationResult: .raw_markdown is the plain
        # html→md conversion, .fit_markdown is the same after main-content filter.
        # Prefer fit when available (cleaner for LLM context).
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

    return {"success": True, "data": data}


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
