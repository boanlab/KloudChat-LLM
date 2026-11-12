"""OpenAI-compatible transcription front end over the GPU nodes' vllm-whisper backends.

`/v1/audio/transcriptions` goes to the healthy backend with the fewest in-flight
requests; with none healthy, to each in configured order. A backend that refuses
the connection is passed over for the next. Uploads are capped at
WHISPER_MAX_UPLOAD_MB (413) and at most WHISPER_MAX_CONCURRENT run at once (503).
`model` is rewritten to MODEL_NAME (vLLM 404s other names; clients default to
`whisper-1`). No OpenRouter fallback: audio never leaves the network, so with no
backend (WHISPER_URLS, or whisper-large-v3 in VLLM_PLACED) the shim is not
deployed and dictation is off.
"""
from __future__ import annotations

import logging
import os
from typing import Any

import httpx
from backends import Backends, Unreachable, parse_urls
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import Response
from upload import Slots, TooLarge, rebuild_form

LOG = logging.getLogger("whisper-shim")
logging.basicConfig(level=os.getenv("LOG_LEVEL", "INFO").upper())


def _parse_backends() -> list[str]:
    """Backend URLs from WHISPER_URLS (hand-run) and VLLM_PLACED (scheduler)."""
    urls = parse_urls(os.getenv("WHISPER_URLS", ""), os.getenv("VLLM_PLACED", ""))
    if not urls:
        raise RuntimeError("no transcription backend in WHISPER_URLS or VLLM_PLACED")
    return urls


BACKENDS: list[str] = _parse_backends()
#: Name put on every forwarded request; one of vllm-whisper's --served-model-name values.
MODEL_NAME: str = os.getenv("WHISPER_MODEL_NAME", "local/whisper-large-v3")
HEALTH_PROBE_TIMEOUT_SEC = float(os.getenv("HEALTH_PROBE_TIMEOUT_SEC", "2.0"))
HEALTH_CACHE_TTL_SEC    = float(os.getenv("HEALTH_CACHE_TTL_SEC", "10"))
TRANSCRIBE_TIMEOUT_SEC  = float(os.getenv("TRANSCRIBE_TIMEOUT_SEC", "900"))
#: Upload ceiling, the same figure the GPU nodes' vllm-whisper accepts
MAX_UPLOAD_BYTES = int(float(os.getenv("WHISPER_MAX_UPLOAD_MB", "100")) * 1024 * 1024)
#: Multipart framing allowed on top of the audio in Content-Length
FORM_OVERHEAD_BYTES = 64 * 1024
#: Transcriptions held at once (each holds its upload in memory); past it, 503
MAX_CONCURRENT = int(os.getenv("WHISPER_MAX_CONCURRENT", "8"))

LOG.info("Whisper backends: %s", ", ".join(BACKENDS))

app = FastAPI(title="KloudChat Whisper Shim")


async def _probe_health(backend: str) -> bool:
    try:
        async with httpx.AsyncClient(timeout=HEALTH_PROBE_TIMEOUT_SEC) as client:
            r = await client.get(f"{backend}/health")
        return r.status_code < 400
    except httpx.HTTPError:
        return False


backends = Backends(BACKENDS, _probe_health, ttl=HEALTH_CACHE_TTL_SEC)
slots = Slots(MAX_CONCURRENT)


@app.get("/health")
async def health() -> dict[str, Any]:
    await backends.refresh()
    return {
        "status": "ok" if any(backends.health.values()) else "degraded",
        "backends": [{"url": b, "up": backends.health.get(b, False),
                      "inflight": backends.inflight[b]} for b in BACKENDS],
    }


@app.post("/v1/audio/transcriptions")
async def transcribe(request: Request) -> Response:
    """Forward the upload to the least-busy reachable backend and mirror its response."""
    length = request.headers.get("content-length", "")
    if length.isdigit() and int(length) > MAX_UPLOAD_BYTES + FORM_OVERHEAD_BYTES:
        raise HTTPException(413, f"upload exceeds {MAX_UPLOAD_BYTES // (1024 * 1024)} MB")
    if not slots.take():
        raise HTTPException(503, "too many transcriptions in progress",
                            headers={"Retry-After": "5"})
    try:
        return await _transcribe(request)
    finally:
        slots.give()


async def _transcribe(request: Request) -> Response:
    try:
        files, fields = await rebuild_form(await request.form(), MODEL_NAME, MAX_UPLOAD_BYTES)
    except TooLarge as e:
        raise HTTPException(413, str(e)) from e
    except Exception as e:  # noqa: BLE001
        raise HTTPException(400, f"could not read the upload: {e}") from e
    if not files:
        raise HTTPException(400, "no audio file in the request")

    async with httpx.AsyncClient(timeout=TRANSCRIBE_TIMEOUT_SEC) as client:
        async def post(backend: str) -> httpx.Response:
            return await client.post(f"{backend}/v1/audio/transcriptions",
                                     files=files, data=fields)

        try:
            backend, r = await backends.send(
                post, unreachable=(httpx.ConnectError, httpx.ConnectTimeout))
        except Unreachable as e:
            raise HTTPException(502, f"whisper backends unreachable: {e}") from e
        except httpx.HTTPError as e:
            LOG.warning("whisper backend request failed: %s", e)
            raise HTTPException(502, f"whisper backend failed: {e}") from e

    # Backend status and headers (Content-Type varies with response_format), minus hop-by-hop
    excluded = {"content-length", "transfer-encoding", "connection"}
    headers = {k: v for k, v in r.headers.items() if k.lower() not in excluded}
    return Response(content=r.content, status_code=r.status_code, headers=headers,
                    media_type=r.headers.get("content-type"))
