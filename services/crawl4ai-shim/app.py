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

A browser or Playwright driver that dies is replaced and the page retried once.
While no scrape proves the browser works, a local page is rendered every
BROWSER_PROBE_INTERVAL_S. /health answers 503 after BROWSER_DEAD_LIMIT failures
in a row, and the process exits after BROWSER_EXIT_LIMIT so Docker restarts it.

Page addresses are logged at DEBUG only; a warning names the host.
"""
from __future__ import annotations

import asyncio
import contextlib
import logging
import os
import time
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

# Pages rendering at once, and the wait for a slot before refusal
MAX_CONCURRENT_PAGES = int(os.environ.get("MAX_CONCURRENT_PAGES", "8"))
QUEUE_TIMEOUT_MS = int(os.environ.get("QUEUE_TIMEOUT_MS", "15000"))
# Lifetime of a cached successful scrape
CACHE_TTL_S = int(os.environ.get("CACHE_TTL_S", "900"))
CACHE_MAX_ENTRIES = int(os.environ.get("CACHE_MAX_ENTRIES", "512"))
# Redirect hops the route guard follows for one request
MAX_REDIRECTS = 10
# Adult-site hosts file, baked into the image
ADULT_HOSTS_FILE = os.environ.get("ADULT_HOSTS_FILE", "/app/adult-hosts.txt")

crawler: AsyncWebCrawler | None = None
browser_config: BrowserConfig | None = None
# Browser restart count; concurrent failures restart once
generation = 0
restart_lock = asyncio.Lock()
# Failures in a row with the browser unusable (scrapes and probes); a working render resets it
dead_in_a_row = 0
# When a render last showed the browser working (monotonic seconds)
last_ok = 0.0
# /health answers 503 from DEAD_LIMIT; at EXIT_LIMIT the process exits for Docker to restart
DEAD_LIMIT = int(os.environ.get("BROWSER_DEAD_LIMIT", "3"))
EXIT_LIMIT = int(os.environ.get("BROWSER_EXIT_LIMIT", "6"))
# Local render while no scrape has shown the browser working for this long
PROBE_INTERVAL_S = float(os.environ.get("BROWSER_PROBE_INTERVAL_S", "30"))
PROBE_TIMEOUT_S = 10.0
START_TIMEOUT_S = 120.0
# Playwright errors (lower case) for a dead browser or a dead driver connection
_DEAD_BROWSER = (
    "has been closed",
    "target closed",
    "browser closed",
    "connection closed",
    "handler is closed",
    "browser.new_context",
)
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
        # No image, font or media downloads; scripts still run
        text_mode=True,
    )
    browser_config = cfg
    crawler = await _started(cfg)
    _mark_ok()
    LOG.info("Crawl4AI ready")
    watcher = asyncio.create_task(_watch())
    try:
        yield
    finally:
        LOG.info("shutting down Crawl4AI")
        watcher.cancel()
        if crawler is not None:
            await _close_quietly(crawler)


app = FastAPI(lifespan=lifespan)


async def _started(cfg: BrowserConfig) -> AsyncWebCrawler:
    started = AsyncWebCrawler(config=cfg)
    await started.start()
    started.crawler_strategy.set_hook("on_page_context_created", _guard_context)
    return started


def _browser_gone(error: str) -> bool:
    error = error.lower()
    return any(words in error for words in _DEAD_BROWSER)


def _mark_ok() -> None:
    global dead_in_a_row, last_ok
    dead_in_a_row = 0
    last_ok = time.monotonic()


def _browser_of(c: Any) -> Any:
    manager = getattr(getattr(c, "crawler_strategy", None), "browser_manager", None)
    return getattr(manager, "browser", None)


async def _renders(c: Any) -> bool:
    """A local page renders in a fresh context: driver, browser and renderer answer. No network."""
    browser = _browser_of(c)
    if browser is None or not browser.is_connected():
        return False

    async def render() -> bool:
        context = await browser.new_context()
        try:
            page = await context.new_page()
            await page.set_content("<p>ok</p>")
            return await page.evaluate("document.body.textContent") == "ok"
        finally:
            await context.close()

    try:
        return await asyncio.wait_for(render(), PROBE_TIMEOUT_S)
    except Exception as e:  # noqa: BLE001 — any failure means it cannot render
        LOG.warning("browser probe failed: %r", e)
        return False


async def _close_quietly(c: Any) -> None:
    """Close a crawler whose browser or driver may be dead; stop its driver if close fails."""
    try:
        await asyncio.wait_for(c.close(), PROBE_TIMEOUT_S)
        return
    except Exception as e:  # noqa: BLE001 — the old one is dead already
        LOG.debug("closing the dead browser: %r", e)
    manager = getattr(getattr(c, "crawler_strategy", None), "browser_manager", None)
    driver = getattr(manager, "playwright", None)
    if driver is not None:
        with contextlib.suppress(Exception):
            await asyncio.wait_for(driver.stop(), PROBE_TIMEOUT_S)


async def _restart_browser(seen: int) -> bool:
    """Replace an unusable browser; True when one is in place afterwards.

    One restart per generation: requests that failed on the same browser wait on
    the lock and reuse the replacement. A browser that still renders is kept."""
    global crawler, generation
    async with restart_lock:
        if generation != seen:
            return crawler is not None
        if crawler is not None and await _renders(crawler):
            LOG.info("browser still renders, keeping it")
            return True
        LOG.warning("browser is unusable, starting a new one")
        old, crawler = crawler, None
        generation += 1
        if old is not None:
            await _close_quietly(old)
        try:
            crawler = await asyncio.wait_for(
                _started(browser_config or BrowserConfig(headless=True)), START_TIMEOUT_S)
        except Exception as e:  # noqa: BLE001 — /health reports it, the watcher retries
            LOG.error("a new browser did not start: %r", e)
            return False
        LOG.info("new browser ready")
        return True


def _exit() -> None:
    """End the process; Docker's restart policy starts a fresh container."""
    os._exit(1)


async def _check() -> None:
    """Render a local page; replace the browser if it cannot, exit if replacing has not helped."""
    global dead_in_a_row
    seen = generation
    if crawler is not None and await _renders(crawler):
        _mark_ok()
        return
    dead_in_a_row += 1
    LOG.warning("browser cannot render (%d in a row)", dead_in_a_row)
    if dead_in_a_row >= EXIT_LIMIT:
        LOG.error("browser still unusable after %d failures, exiting", dead_in_a_row)
        _exit()
        return
    if await _restart_browser(seen) and crawler is not None and await _renders(crawler):
        _mark_ok()


async def _watch() -> None:
    while True:
        await asyncio.sleep(PROBE_INTERVAL_S)
        if time.monotonic() - last_ok < PROBE_INTERVAL_S:
            continue
        try:
            await _check()
        except Exception as e:  # noqa: BLE001 — the watcher must outlive one bad check
            LOG.error("browser check failed: %r", e)


@app.get("/health")
async def health(response: Response) -> dict[str, Any]:
    down = crawler is None or dead_in_a_row >= DEAD_LIMIT
    if down:
        response.status_code = 503
    return {
        "status": "browser_down" if down else "ok",
        "deadInARow": dead_in_a_row,
        "lastOkAgoS": round(time.monotonic() - last_ok, 1) if last_ok else None,
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
            # A dead driver fails the abort too
            with contextlib.suppress(Exception):
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
        # Unusable browser: replace it and retry once
        for attempt in (1, 2):
            seen = generation
            current = crawler
            if current is None:
                error = "browser is restarting"
            else:
                try:
                    result = await asyncio.wait_for(
                        current.arun(url=url, config=run_cfg),
                        timeout=(timeout_ms / 1000.0) + 10,
                    )
                except asyncio.TimeoutError:
                    LOG.warning("scrape timeout on %s", _site(url))
                    return {"success": False, "error": "scrape timeout"}
                except Exception as e:
                    error = str(e)
                    if not _browser_gone(error):
                        LOG.warning("scrape error on %s: %r", _site(url), e)
                        return {"success": False, "error": f"crawl4ai error: {e}"}
                else:
                    error = str(getattr(result, "error_message", "") or "")
                    if getattr(result, "success", False) or not _browser_gone(error):
                        # The browser answered, even if the page failed
                        _mark_ok()
                        break
            dead_in_a_row += 1
            LOG.warning("browser unusable on %s (%d in a row): %s",
                        _site(url), dead_in_a_row, error.splitlines()[0][:200] if error else "")
            if attempt == 1 and await _restart_browser(seen):
                continue
            return {"success": False, "error": "browser unavailable, restarting"}
    finally:
        gate.release()

    if not getattr(result, "success", False):
        err = getattr(result, "error_message", None) or "crawl failed"
        # ERR_BLOCKED_BY_CLIENT: the route guard aborted the navigation
        if "ERR_BLOCKED_BY_CLIENT" in err:
            err = netguard.INTERNAL
        LOG.warning("scrape failed on %s: %s", _site(url), err)
        return {"success": False, "error": err}
    # Final address after redirects
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
        # fit_markdown: main-content filtered; raw_markdown: unfiltered
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
