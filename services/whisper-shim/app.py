"""OpenAI-compatible Whisper aggregator in front of one or more whisper backends.

Each backend is an amd64 GPU node's transcription container, installed alongside
vLLM by `scripts/install-vllm.sh` (faster-whisper on the card) exposing
OpenAI-style `/v1/audio/transcriptions`. arm64 nodes run no backend — aarch64
ctranslate2 wheels are CPU-only, so STT is delegated to OpenRouter there and this
shim is not deployed at all. The shim multiplexes the same protocol across the
backends — no translation, just routing — so any client that speaks `WHISPER_URL`
gets a single endpoint while audio jobs spread across the GPU fleet.

Routing
───────
For each request, among backends that pass /health, pick the one with the
lowest in-flight count. Each backend serves a request on its GPU sequentially
(whisper's model state is per-process; concurrent requests queue inside the
worker), so steering to the least-busy node keeps a single hot node from
accumulating jobs while an idle node sits empty.

If every backend's health probe fails we still forward to the first one so the
caller gets a meaningful 5xx instead of a silent drop. STT is GPU-only — there
is no OpenRouter fallback (callers surface the failure directly).

No variant catalogue (every backend serves the same WHISPER_MODEL), no
prompt_id stickiness (each request is one HTTP round-trip and self-contained).
"""
from __future__ import annotations

import asyncio
import logging
import os
import time
from collections import defaultdict
from typing import Any

import httpx
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import Response

LOG = logging.getLogger("whisper-shim")
logging.basicConfig(level=os.getenv("LOG_LEVEL", "INFO").upper())


def _parse_backends() -> list[str]:
    """WHISPER_URLS — comma-separated whisper backend URLs (GPU nodes' faster-whisper).
    WHISPER_URL = a separate variable by which a consumer (youtube MCP) points at the shim — unused here."""
    raw = os.getenv("WHISPER_URLS", "")
    urls = [u.strip().rstrip("/") for u in raw.split(",") if u.strip()]
    if not urls:
        raise RuntimeError("WHISPER_URLS is empty")
    return urls


BACKENDS: list[str] = _parse_backends()
HEALTH_PROBE_TIMEOUT_SEC = float(os.getenv("HEALTH_PROBE_TIMEOUT_SEC", "2.0"))
HEALTH_CACHE_TTL_SEC    = float(os.getenv("HEALTH_CACHE_TTL_SEC", "10"))
TRANSCRIBE_TIMEOUT_SEC  = float(os.getenv("TRANSCRIBE_TIMEOUT_SEC", "900"))

# In-process inflight counter — accurate per shim replica. Multiple replicas
# wouldn't share this view, but we only run one shim container per stack so
# this is the authoritative load signal at the routing layer.
_INFLIGHT: dict[str, int] = defaultdict(int)
_INFLIGHT_LOCK = asyncio.Lock()

# Cached /health probe result so we don't hit every backend on every request
# (each call is a quick GET but they add up under bursty MCP usage). TTL is
# short enough that a node that goes down is dropped within ~10s.
_HEALTH: dict[str, bool] = {}
_HEALTH_AT: float = 0.0
_HEALTH_LOCK = asyncio.Lock()

LOG.info("Whisper backends: %s", ", ".join(BACKENDS))

app = FastAPI(title="KloudChat Whisper Shim", version="0.1.0")


# ──────────────────────────────────────────────────────────────────────
# Backend selection
# ──────────────────────────────────────────────────────────────────────

async def _probe_health(client: httpx.AsyncClient, backend: str) -> bool:
    try:
        r = await client.get(f"{backend}/health", timeout=HEALTH_PROBE_TIMEOUT_SEC)
        return r.status_code < 400
    except httpx.HTTPError:
        return False


async def _refresh_health(client: httpx.AsyncClient) -> None:
    """Refresh the cached health map if stale. Holds a lock so concurrent
    requests share one probe round."""
    global _HEALTH_AT
    async with _HEALTH_LOCK:
        fresh = _HEALTH and (time.monotonic() < _HEALTH_AT + HEALTH_CACHE_TTL_SEC)
        if fresh:
            return
        results = await asyncio.gather(*(_probe_health(client, b) for b in BACKENDS))
        _HEALTH.clear()
        for b, ok in zip(BACKENDS, results, strict=True):
            _HEALTH[b] = ok
        _HEALTH_AT = time.monotonic()
        LOG.info("Whisper health: %s",
                 ", ".join(f"{b}={'up' if ok else 'down'}" for b, ok in _HEALTH.items()))


async def _pick_backend(client: httpx.AsyncClient) -> str:
    """Pick the healthy backend with the fewest in-flight whisper requests."""
    if len(BACKENDS) == 1:
        return BACKENDS[0]

    await _refresh_health(client)
    healthy = [b for b in BACKENDS if _HEALTH.get(b)]
    if not healthy:
        LOG.warning("All whisper backends unhealthy; forwarding to %s anyway", BACKENDS[0])
        return BACKENDS[0]
    if len(healthy) == 1:
        return healthy[0]

    healthy.sort(key=lambda b: _INFLIGHT[b])
    chosen = healthy[0]
    LOG.info(
        "Routing to %s — candidates: %s",
        chosen,
        ", ".join(f"{b}[inflight={_INFLIGHT[b]}]" for b in healthy),
    )
    return chosen


async def _inc_inflight(backend: str) -> None:
    async with _INFLIGHT_LOCK:
        _INFLIGHT[backend] += 1


async def _dec_inflight(backend: str) -> None:
    async with _INFLIGHT_LOCK:
        _INFLIGHT[backend] = max(0, _INFLIGHT[backend] - 1)


# ──────────────────────────────────────────────────────────────────────
# Routes
# ──────────────────────────────────────────────────────────────────────

@app.get("/health")
async def health() -> dict[str, Any]:
    async with httpx.AsyncClient() as client:
        await _refresh_health(client)
    return {
        "status": "ok" if any(_HEALTH.values()) else "degraded",
        "backends": [{"url": b, "up": _HEALTH.get(b, False), "inflight": _INFLIGHT[b]} for b in BACKENDS],
    }


@app.post("/v1/audio/transcriptions")
async def transcribe(request: Request) -> Response:
    """Multipart passthrough — forward the raw request body and content-type
    to the chosen backend, then mirror its response.

    Reading the body into memory once is fine: whisper inputs are short audio
    files (the youtube MCP downloads bestaudio[ext=m4a]), and faster-whisper
    needs the whole file on disk anyway. Streaming through wouldn't save
    memory and would complicate the inflight accounting.
    """
    body = await request.body()
    content_type = request.headers.get("content-type", "application/octet-stream")

    async with httpx.AsyncClient(timeout=TRANSCRIBE_TIMEOUT_SEC) as client:
        backend = await _pick_backend(client)
        await _inc_inflight(backend)
        try:
            r = await client.post(
                f"{backend}/v1/audio/transcriptions",
                content=body,
                headers={"content-type": content_type},
            )
        except httpx.HTTPError as e:
            LOG.warning("whisper backend %s request failed: %s", backend, e)
            raise HTTPException(502, f"whisper backend {backend} unreachable: {e}") from e
        finally:
            await _dec_inflight(backend)

    # Mirror status + headers (Content-Type especially — backend may return
    # text/plain when caller asked for response_format=text). Strip hop-by-hop
    # headers httpx already collapsed; Content-Length is regenerated by
    # FastAPI from the body.
    excluded = {"content-length", "transfer-encoding", "connection"}
    headers = {k: v for k, v in r.headers.items() if k.lower() not in excluded}
    return Response(content=r.content, status_code=r.status_code, headers=headers,
                    media_type=r.headers.get("content-type"))
