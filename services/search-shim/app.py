"""SearXNG front for KloudChat's web search: result cache, coalescing, and a cap
on searches in flight.

Endpoints:
  GET/POST /search   SearXNG's /search, JSON or HTML. safesearch is forced to
                     SAFESEARCH whatever the caller sent, and a bare `ko` is
                     given its region. An answer with results is served from
                     cache for CACHE_TTL_S; identical searches in flight share
                     one upstream call; at most MAX_CONCURRENT_SEARCHES reach
                     SearXNG at once, the rest wait up to QUEUE_TIMEOUT_MS.
                     When SearXNG is busy or failing, an answer up to
                     STALE_TTL_S old is served instead; without one, 503/502.
                     SearXNG's own 4xx (a bad parameter) passes through as is.
  GET  /health
  anything else      forwarded to SearXNG unchanged, outside the cap.

The engines behind SearXNG ban a datacentre address on burst, so the point is
to keep the number of requests that actually leave this host small and even.
Search terms are logged at DEBUG only.
"""
from __future__ import annotations

import json
import logging
import os
from contextlib import asynccontextmanager
from dataclasses import dataclass
from typing import Any

import httpx
from fastapi import FastAPI, Request, Response
from fastapi.responses import JSONResponse
from store import Coalescer, Gate, ResultCache

LOG_LEVEL = os.environ.get("LOG_LEVEL", "INFO").upper()
logging.basicConfig(level=LOG_LEVEL,
                    format="%(asctime)s %(levelname)s %(name)s %(message)s")
LOG = logging.getLogger("search-shim")
# One line per upstream call otherwise
logging.getLogger("httpx").setLevel(logging.WARNING)

SEARXNG_URL = os.environ.get("SEARXNG_URL", "http://searxng:8080").rstrip("/")
# SearXNG's own outgoing.max_request_timeout is 15 s; this bounds the whole call.
UPSTREAM_TIMEOUT_S = float(os.environ.get("UPSTREAM_TIMEOUT_S", "20"))
# A SearXNG that is down or restarting is found out this fast, not after the
# full timeout with a gate slot held.
CONNECT_TIMEOUT_S = float(os.environ.get("CONNECT_TIMEOUT_S", "3"))

# Searches SearXNG runs at once, and how long a request waits for a slot.
MAX_CONCURRENT_SEARCHES = int(os.environ.get("MAX_CONCURRENT_SEARCHES", "12"))
QUEUE_TIMEOUT_MS = int(os.environ.get("QUEUE_TIMEOUT_MS", "10000"))
# A search with results is answered from memory for CACHE_TTL_S, and still
# handed out up to STALE_TTL_S after it was fetched when SearXNG cannot answer.
CACHE_TTL_S = int(os.environ.get("CACHE_TTL_S", "900"))
STALE_TTL_S = int(os.environ.get("STALE_TTL_S", "21600"))
# Answers are kept as the bytes SearXNG sent, tens of KB each.
CACHE_MAX_ENTRIES = int(os.environ.get("CACHE_MAX_ENTRIES", "2048"))
# SearXNG safe search on every search: 0 off, 1 moderate, 2 strict. The caller's
# value is replaced, not floored: the UI sends 1 by default.
SAFESEARCH = os.environ.get("SAFESEARCH", "2")
# A bare language is given its region: engines that take a market or country
# prefer Korean results only with the region present.
_LANGUAGE_REGION = {"ko": "ko-KR"}

# Hop-by-hop and framing headers; httpx has already decoded the body.
_DROP_HEADERS = {"connection", "keep-alive", "transfer-encoding", "content-encoding",
                 "content-length", "server", "date"}
# Every result article in SearXNG's HTML carries this class.
_HTML_RESULT = b'class="result result-'

search_client: httpx.AsyncClient | None = None
proxy_client: httpx.AsyncClient | None = None
gate = Gate(MAX_CONCURRENT_SEARCHES)
cache = ResultCache(CACHE_TTL_S, STALE_TTL_S, CACHE_MAX_ENTRIES)
coalescer = Coalescer()


@asynccontextmanager
async def lifespan(_app: FastAPI):
    global search_client, proxy_client
    # The search pool is the gate's size, so a slot never waits on a connection;
    # the passthrough has its own pool and cannot starve searches.
    search_client = httpx.AsyncClient(
        base_url=SEARXNG_URL,
        timeout=httpx.Timeout(UPSTREAM_TIMEOUT_S, connect=CONNECT_TIMEOUT_S,
                              pool=QUEUE_TIMEOUT_MS / 1000.0),
        limits=httpx.Limits(max_connections=MAX_CONCURRENT_SEARCHES),
    )
    proxy_client = httpx.AsyncClient(
        base_url=SEARXNG_URL,
        timeout=httpx.Timeout(UPSTREAM_TIMEOUT_S, connect=CONNECT_TIMEOUT_S),
        limits=httpx.Limits(max_connections=16),
    )
    LOG.info("search-shim in front of %s (limit %d, cache %ds/%ds)",
             SEARXNG_URL, MAX_CONCURRENT_SEARCHES, CACHE_TTL_S, STALE_TTL_S)
    try:
        yield
    finally:
        await search_client.aclose()
        await proxy_client.aclose()


app = FastAPI(lifespan=lifespan)


@app.get("/health")
async def health() -> dict[str, Any]:
    return {
        "status": "ok" if search_client is not None else "starting",
        "backend": SEARXNG_URL,
        "searches": {"active": gate.active, "waiting": gate.waiting, "limit": gate.limit,
                     "coalesced": coalescer.inflight},
        "cache": {"entries": len(cache), "ttl_s": CACHE_TTL_S, "stale_ttl_s": STALE_TTL_S},
    }


@dataclass
class Answer:
    """What SearXNG sent for a search, as sent."""
    status: int
    media_type: str
    body: bytes

    def has_results(self) -> bool:
        """An empty answer is what a suspended engine returns; remembering it
        would hide the engine's recovery."""
        if "json" in self.media_type:
            try:
                return bool(json.loads(self.body).get("results"))
            except (ValueError, AttributeError):
                return False
        return _HTML_RESULT in self.body

    def response(self) -> Response:
        return Response(self.body, status_code=self.status, media_type=self.media_type)


class UpstreamError(Exception):
    def __init__(self, status: int, error: str) -> None:
        super().__init__(error)
        self.status = status
        self.error = error


async def _search_params(req: Request) -> dict[str, str]:
    """Search parameters from the query string and a form body (SearXNG reads
    both), a repeated key keeping its first value as SearXNG does."""
    items = list(req.query_params.multi_items())
    if req.method == "POST":
        try:
            items += [(k, str(v)) for k, v in (await req.form()).multi_items()]
        except Exception:
            pass
    params: dict[str, str] = {}
    for k, v in items:
        params.setdefault(k, v)
    params["safesearch"] = SAFESEARCH
    language = _LANGUAGE_REGION.get(params.get("language", ""), params.get("language", ""))
    if language:
        params["language"] = language
    else:
        params.pop("language", None)
    return params


async def _fetch(params: dict[str, str]) -> Answer:
    """One search against SearXNG, behind the gate. SearXNG's 4xx is an answer;
    5xx and transport failures are not."""
    if not await gate.acquire(QUEUE_TIMEOUT_MS / 1000.0):
        LOG.warning("search refused, SearXNG busy (%d running, %d waiting)",
                    gate.active, gate.waiting)
        raise UpstreamError(503, "busy: too many searches in flight")
    try:
        r = await search_client.get("/search", params=params)
    except httpx.HTTPError as e:
        LOG.warning("search failed: %r", e)
        raise UpstreamError(502, f"searxng error: {e}") from e
    finally:
        gate.release()
    if r.status_code >= 500:
        LOG.warning("searxng answered %d", r.status_code)
        raise UpstreamError(502, f"searxng answered {r.status_code}")
    return Answer(r.status_code, r.headers.get("content-type", "text/html"), r.content)


async def _search(params: dict[str, str]) -> Response:
    key = ResultCache.key(params)
    if (hit := cache.fresh(key)) is not None:
        LOG.debug("search %r served from cache", params.get("q"))
        return hit.response()

    async def fetch() -> Answer:
        answer = await _fetch(params)
        if answer.status == 200 and answer.has_results():
            cache.put(key, answer)
        return answer

    try:
        answer = await coalescer.run(key, fetch)
    except UpstreamError as e:
        if (old := cache.stale(key)) is not None:
            LOG.info("a search served stale (%s)", e.error)
            return old.response()
        return JSONResponse({"error": e.error, "results": []}, status_code=e.status)
    LOG.debug("search %r: %d", params.get("q"), answer.status)
    return answer.response()


@app.api_route("/search", methods=["GET", "POST"])
async def search(req: Request) -> Response:
    return await _search(await _search_params(req))


@app.api_route("/{path:path}", methods=["GET", "POST", "HEAD"])
async def passthrough(req: Request, path: str) -> Response:
    """Everything else: SearXNG's own answer, as is."""
    headers = {k: v for k, v in req.headers.items()
               if k.lower() not in ("host", "content-length")}
    try:
        r = await proxy_client.request(req.method, "/" + path,
                                       params=req.query_params.multi_items(),
                                       content=await req.body(), headers=headers)
    except httpx.HTTPError as e:
        return JSONResponse({"error": f"searxng error: {e}"}, status_code=502)
    out = {k: v for k, v in r.headers.items() if k.lower() not in _DROP_HEADERS}
    return Response(r.content, status_code=r.status_code, headers=out)
