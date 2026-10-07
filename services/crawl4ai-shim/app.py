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
Every request a page makes passes the network guard before it leaves the browser.

Page addresses are logged at DEBUG only; a warning names the host.
"""
from __future__ import annotations

import asyncio
import logging
import os
from contextlib import asynccontextmanager
from typing import Any
from urllib.parse import urljoin, urlsplit

import netguard
from adultlist import AdultList
from crawl4ai import (
    AsyncWebCrawler,
    BrowserConfig,
    CacheMode,
    CrawlerRunConfig,
)
from fastapi import FastAPI, Request, Response
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

# Pages rendering at once, and the wait for a slot before a request is refused.
# One Chromium on an 8-core host renders about eight pages concurrently; past
# that every page slows towards its own timeout.
MAX_CONCURRENT_PAGES = int(os.environ.get("MAX_CONCURRENT_PAGES", "8"))
QUEUE_TIMEOUT_MS = int(os.environ.get("QUEUE_TIMEOUT_MS", "15000"))
# A successful scrape is answered from memory for this long.
CACHE_TTL_S = int(os.environ.get("CACHE_TTL_S", "900"))
CACHE_MAX_ENTRIES = int(os.environ.get("CACHE_MAX_ENTRIES", "512"))
# Redirect hops the route guard follows for one request
MAX_REDIRECTS = 10
# Hosts file of adult sites a scrape refuses; baked into the image.
ADULT_HOSTS_FILE = os.environ.get("ADULT_HOSTS_FILE", "/app/adult-hosts.txt")

crawler: AsyncWebCrawler | None = None
browser_config: BrowserConfig | None = None
#: Bumped on every browser restart, so concurrent failures restart it once.
generation = 0
restart_lock = asyncio.Lock()
#: Scrapes in a row that found the browser dead; past DEAD_LIMIT /health fails and the
#: container is restarted.
dead_in_a_row = 0
DEAD_LIMIT = 3
#: Playwright's words for a browser that is gone. The process can die (OOM, a crash)
#: while the service stays up; every later page then fails with these until restart.
_DEAD_BROWSER = ("has been closed", "Target closed", "Browser closed", "Connection closed")
adult = AdultList()
gate = Gate(MAX_CONCURRENT_PAGES)
cache = PageCache(CACHE_TTL_S, CACHE_MAX_ENTRIES)


@asynccontextmanager
async def lifespan(_app: FastAPI):
    global crawler, adult, browser_config
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
        # No image, font or media downloads: only the page's markdown is
        # returned. Scripts still run.
        text_mode=True,
    )
    browser_config = cfg
    crawler = await _started(cfg)
    LOG.info("Crawl4AI ready")
    try:
        yield
    finally:
        LOG.info("shutting down Crawl4AI")
        await crawler.close()


app = FastAPI(lifespan=lifespan)


async def _started(cfg: BrowserConfig) -> AsyncWebCrawler:
    started = AsyncWebCrawler(config=cfg)
    await started.start()
    # Every request a page makes is judged before it leaves the browser.
    started.crawler_strategy.set_hook("on_page_context_created", _guard_context)
    return started


def _browser_gone(error: str) -> bool:
    return any(words in error for words in _DEAD_BROWSER)


async def _restart_browser(seen: int) -> None:
    """A fresh browser in place of a dead one; a restart another request already made
    (the generation moved on) is not repeated."""
    global crawler, generation
    async with restart_lock:
        if generation != seen:
            return
        LOG.warning("browser is gone, starting a new one")
        old, crawler = crawler, None
        try:
            if old is not None:
                await old.close()
        except Exception as e:  # noqa: BLE001 — the old one is dead already
            LOG.debug("closing the dead browser: %r", e)
        generation += 1
        try:
            crawler = await _started(browser_config or BrowserConfig(headless=True))
        except Exception as e:  # noqa: BLE001 — /health reports it and the container restarts
            LOG.error("a new browser did not start: %r", e)
            return
        LOG.info("new browser ready")


@app.get("/health")
async def health(response: Response) -> dict[str, Any]:
    # A browser that stays dead after restarts fails the check, so the container restarts.
    down = dead_in_a_row >= DEAD_LIMIT
    if down:
        response.status_code = 503
    return {
        "status": "browser_down" if down else "ok" if crawler is not None else "starting",
        "deadInARow": dead_in_a_row,
        "backend": "crawl4ai",
        "pages": {"active": gate.active, "waiting": gate.waiting, "limit": gate.limit},
        "cache": {"entries": len(cache), "ttl_s": CACHE_TTL_S},
        "adult_list": {"domains": len(adult)},
    }


def _refusal(url: str) -> str | None:
    """Why `url` may not be scraped: off the network, or an adult site."""
    return netguard.refusal(url) or adult.refusal(url)


async def _guard_context(page, context=None, **_: Any):
    """Crawl4AI hook: every request of the new context passes the network guard.

    Playwright does not route redirected requests, so the guard follows redirects
    itself, judges each hop, and fulfils the route with the final response."""
    verdicts = netguard.HostVerdicts()
    target = context or page.context

    async def guard(route, request):
        url = request.url
        refused = await asyncio.to_thread(netguard.subrequest_refusal, url, verdicts)
        if refused:
            LOG.warning("request refused for %s: %s", _site(url), refused)
            await route.abort("blockedbyclient")
            return
        if not url.lower().startswith(("http://", "https://")):
            await route.continue_()
            return
        try:
            resp = await route.fetch(max_redirects=0)
            hops = 0
            while 300 <= resp.status < 400 and resp.headers.get("location") and hops < MAX_REDIRECTS:
                hop = urljoin(resp.url, resp.headers["location"])
                refused = await asyncio.to_thread(netguard.subrequest_refusal, hop, verdicts)
                if refused:
                    LOG.warning("redirect refused %s -> %s: %s", _site(url), _site(hop), refused)
                    await route.abort("blockedbyclient")
                    return
                method = "GET" if resp.status in (301, 302, 303) else None
                resp = await route.fetch(url=hop, method=method, max_redirects=0)
                hops += 1
            if 300 <= resp.status < 400 and resp.headers.get("location"):
                # Past MAX_REDIRECTS: never hand the browser a redirect to follow unrouted
                LOG.warning("redirect chain too long for %s", _site(url))
                await route.abort("failed")
                return
            await route.fulfill(response=resp)
        except Exception as e:  # noqa: BLE001
            LOG.debug("request failed for %s: %r", _site(url), e)
            await route.abort("failed")

    await target.route("**/*", guard)


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
        # Crawl4AI's progress lines carry the address
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
    global dead_in_a_row
    try:
        # A dead browser is restarted and the page tried once more.
        for attempt in (1, 2):
            seen = generation
            if crawler is None:
                dead_in_a_row += 1
                return {"success": False, "error": "browser is restarting"}
            try:
                result = await asyncio.wait_for(
                    crawler.arun(url=url, config=run_cfg),
                    timeout=(timeout_ms / 1000.0) + 10,
                )
            except asyncio.TimeoutError:
                LOG.warning("scrape timeout on %s", _site(url))
                return {"success": False, "error": "scrape timeout"}
            except Exception as e:
                if attempt == 1 and _browser_gone(str(e)):
                    dead_in_a_row += 1
                    await _restart_browser(seen)
                    continue
                LOG.warning("scrape error on %s: %r", _site(url), e)
                return {"success": False, "error": f"crawl4ai error: {e}"}
            error = str(getattr(result, "error_message", "") or "")
            if not getattr(result, "success", False) and _browser_gone(error):
                dead_in_a_row += 1
                if attempt == 1:
                    await _restart_browser(seen)
                    continue
            else:
                dead_in_a_row = 0
            break
    finally:
        gate.release()

    if not getattr(result, "success", False):
        err = getattr(result, "error_message", None) or "crawl failed"
        # ERR_BLOCKED_BY_CLIENT: the route guard aborted the navigation
        if "ERR_BLOCKED_BY_CLIENT" in err:
            err = netguard.INTERNAL
        LOG.warning("scrape failed on %s: %s", _site(url), err)
        return {"success": False, "error": err}
    # The address the browser ended up at is judged once more.
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
        # fit_markdown: after the main-content filter; raw_markdown: plain html to md.
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
