"""OpenAI-compatible transcription front end over the GPU nodes' vllm-whisper backends.

One endpoint, `/v1/audio/transcriptions`, forwarded to the healthy backend with
the fewest in-flight requests. The `model` field is rewritten to the name every
backend serves, since vLLM 404s any other name and clients default to
`whisper-1`. With every backend unhealthy the first one is still tried, so the
caller gets a 5xx rather than a silent drop. No OpenRouter fallback here: an
empty WHISPER_URLS means this shim is not deployed and LiteLLM registers
OpenRouter STT instead.
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
    """Backend URLs from WHISPER_URLS (comma-separated)."""
    raw = os.getenv("WHISPER_URLS", "")
    urls = [u.strip().rstrip("/") for u in raw.split(",") if u.strip()]
    if not urls:
        raise RuntimeError("WHISPER_URLS is empty")
    return urls


BACKENDS: list[str] = _parse_backends()
#: Name put on every forwarded request; one of vllm-whisper's --served-model-name values.
MODEL_NAME: str = os.getenv("WHISPER_MODEL_NAME", "local/whisper-large-v3")
HEALTH_PROBE_TIMEOUT_SEC = float(os.getenv("HEALTH_PROBE_TIMEOUT_SEC", "2.0"))
HEALTH_CACHE_TTL_SEC    = float(os.getenv("HEALTH_CACHE_TTL_SEC", "10"))
TRANSCRIBE_TIMEOUT_SEC  = float(os.getenv("TRANSCRIBE_TIMEOUT_SEC", "900"))

# In-flight count per backend. In-process: one shim replica per stack.
_INFLIGHT: dict[str, int] = defaultdict(int)
_INFLIGHT_LOCK = asyncio.Lock()

# Cached backend health, refreshed every HEALTH_CACHE_TTL_SEC.
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
    """Refresh the health cache if stale; concurrent callers share one probe round."""
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
    """Healthy backend with the fewest in-flight requests."""
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


async def _rebuild_form(request: Request) -> tuple[dict, dict]:
    """(files, fields) for httpx, with `model` replaced by MODEL_NAME."""
    form = await request.form()
    files: dict[str, tuple] = {}
    fields: dict[str, str] = {}
    for key, value in form.multi_items():
        if hasattr(value, "filename"):
            files[key] = (value.filename, await value.read(),
                          value.content_type or "application/octet-stream")
        else:
            fields[key] = str(value)
    fields["model"] = MODEL_NAME
    return files, fields


@app.post("/v1/audio/transcriptions")
async def transcribe(request: Request) -> Response:
    """Forward the upload to the least-busy backend and mirror its response."""
    try:
        files, fields = await _rebuild_form(request)
    except Exception as e:  # noqa: BLE001
        raise HTTPException(400, f"could not read the upload: {e}") from e
    if not files:
        raise HTTPException(400, "no audio file in the request")

    async with httpx.AsyncClient(timeout=TRANSCRIBE_TIMEOUT_SEC) as client:
        backend = await _pick_backend(client)
        await _inc_inflight(backend)
        try:
            r = await client.post(
                f"{backend}/v1/audio/transcriptions",
                files=files, data=fields,
            )
        except httpx.HTTPError as e:
            LOG.warning("whisper backend %s request failed: %s", backend, e)
            raise HTTPException(502, f"whisper backend {backend} unreachable: {e}") from e
        finally:
            await _dec_inflight(backend)

    # Mirror status and headers (Content-Type varies with response_format);
    # hop-by-hop headers dropped, Content-Length regenerated.
    excluded = {"content-length", "transfer-encoding", "connection"}
    headers = {k: v for k, v in r.headers.items() if k.lower() not in excluded}
    return Response(content=r.content, status_code=r.status_code, headers=headers,
                    media_type=r.headers.get("content-type"))
