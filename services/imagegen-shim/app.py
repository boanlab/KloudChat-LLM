"""Chat-shaped front end over the GPU nodes' vllm-omni image servers.

KloudChat asks for a picture the OpenRouter way: `/v1/chat/completions` with
`modalities: ["image", "text"]` and `image_config.aspect_ratio`, answered with
`message.images[].image_url.url` as a data URL. vllm-omni answers its chat route
with the image in `content` and takes the size as `WIDTHxHEIGHT`, so this shim
calls its `/v1/images/generations` instead and reshapes the reply.

IMAGEGEN_BACKENDS maps each served name to one backend:
`local/z-image-turbo=http://gpu-2:8092,local/flux2-klein-4b=http://gpu-2:8093`.
"""
from __future__ import annotations

import logging
import os
import time
from typing import Any

import httpx
from fastapi import FastAPI, HTTPException, Request
from shape import chat_reply, generation_body, parse_backends

LOG = logging.getLogger("imagegen-shim")
logging.basicConfig(level=os.getenv("LOG_LEVEL", "INFO").upper())

GENERATE_TIMEOUT_SEC = float(os.getenv("GENERATE_TIMEOUT_SEC", "600"))
HEALTH_PROBE_TIMEOUT_SEC = float(os.getenv("HEALTH_PROBE_TIMEOUT_SEC", "2.0"))

BACKENDS: dict[str, str] = parse_backends(os.getenv("IMAGEGEN_BACKENDS", ""))
LOG.info("Image backends: %s", ", ".join(f"{k}={v}" for k, v in BACKENDS.items()) or "none")

app = FastAPI(title="KloudChat Image Shim")


@app.get("/health")
async def health() -> dict[str, Any]:
    async with httpx.AsyncClient(timeout=HEALTH_PROBE_TIMEOUT_SEC) as client:
        rows = []
        for name, url in BACKENDS.items():
            try:
                up = (await client.get(f"{url}/health")).status_code < 400
            except httpx.HTTPError:
                up = False
            rows.append({"model": name, "url": url, "up": up})
    return {"status": "ok" if any(r["up"] for r in rows) else "degraded", "backends": rows}


@app.get("/v1/models")
async def models() -> dict[str, Any]:
    return {"object": "list", "data": [{"id": n, "object": "model", "owned_by": "kloudchat"} for n in BACKENDS]}


@app.post("/v1/chat/completions")
async def chat_completions(request: Request) -> dict[str, Any]:
    body = await request.json()
    model = str(body.get("model") or "")
    backend = BACKENDS.get(model)
    if backend is None:
        raise HTTPException(404, f"unknown image model {model!r}")
    payload = generation_body(model, body)
    if not payload["prompt"]:
        raise HTTPException(400, "no prompt in the request")

    started = time.monotonic()
    try:
        async with httpx.AsyncClient(timeout=GENERATE_TIMEOUT_SEC) as client:
            r = await client.post(f"{backend}/v1/images/generations", json=payload)
    except httpx.HTTPError as e:
        LOG.warning("image backend %s failed: %s", backend, e)
        raise HTTPException(502, f"image backend unreachable: {e}") from e
    if r.status_code >= 400:
        LOG.warning("image backend %s answered %s: %s", backend, r.status_code, r.text[:300])
        raise HTTPException(r.status_code, r.text[:300])
    data = (r.json().get("data") or [{}])[0]
    b64 = data.get("b64_json")
    if not b64:
        raise HTTPException(502, "image backend returned no image")
    LOG.info("%s %s in %.1fs", model, payload["size"], time.monotonic() - started)
    return chat_reply(model, b64)
