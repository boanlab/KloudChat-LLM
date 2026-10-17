"""Chat-shaped front end over the GPU nodes' vllm-omni image servers.

KloudChat asks for a picture the OpenRouter way: `/v1/chat/completions` with
`modalities: ["image", "text"]` and `image_config.aspect_ratio`, answered with
`message.images[].image_url.url` as a data URL. vllm-omni answers its chat route
with the image in `content` and takes the size as `WIDTHxHEIGHT`, so this shim
calls its `/v1/images/generations` instead and reshapes the reply.

IMAGEGEN_BACKENDS is comma-separated `name=url` pairs, a name repeated once per
node: `local/z-image-turbo=http://gpu-1:8091,local/z-image-turbo=http://gpu-2:8091,
local/qwen-image-2512=http://gpu-1:8092`.
A request goes to the backend with the fewest pictures in flight (pool.py).
"""
from __future__ import annotations

import logging
import os
import time
from typing import Any

import httpx
from fastapi import FastAPI, HTTPException, Request
from pool import Pool
from shape import chat_reply, generation_body, parse_backends

LOG = logging.getLogger("imagegen-shim")
logging.basicConfig(level=os.getenv("LOG_LEVEL", "INFO").upper())

GENERATE_TIMEOUT_SEC = float(os.getenv("GENERATE_TIMEOUT_SEC", "600"))
HEALTH_PROBE_TIMEOUT_SEC = float(os.getenv("HEALTH_PROBE_TIMEOUT_SEC", "2.0"))

BACKENDS: dict[str, list[str]] = parse_backends(os.getenv("IMAGEGEN_BACKENDS", ""))
POOLS: dict[str, Pool] = {name: Pool(urls) for name, urls in BACKENDS.items()}
LOG.info("Image backends: %s",
         ", ".join(f"{k}={'|'.join(v)}" for k, v in BACKENDS.items()) or "none")

app = FastAPI(title="KloudChat Image Shim")


@app.get("/health")
async def health() -> dict[str, Any]:
    async with httpx.AsyncClient(timeout=HEALTH_PROBE_TIMEOUT_SEC) as client:
        rows = []
        for name, urls in BACKENDS.items():
            for url in urls:
                try:
                    up = (await client.get(f"{url}/health")).status_code < 400
                except httpx.HTTPError:
                    up = False
                rows.append({"model": name, "url": url, "up": up,
                             "inFlight": POOLS[name].busy[url]})
    return {"status": "ok" if any(r["up"] for r in rows) else "degraded", "backends": rows}


@app.get("/v1/models")
async def models() -> dict[str, Any]:
    return {"object": "list", "data": [{"id": n, "object": "model", "owned_by": "kloudchat"} for n in BACKENDS]}


@app.post("/v1/chat/completions")
async def chat_completions(request: Request) -> dict[str, Any]:
    body = await request.json()
    model = str(body.get("model") or "")
    pool = POOLS.get(model)
    if pool is None:
        raise HTTPException(404, f"unknown image model {model!r}")
    payload = generation_body(model, body)
    if not payload["prompt"]:
        raise HTTPException(400, "no prompt in the request")

    started = time.monotonic()
    r = None
    unreachable: list[str] = []
    for backend in pool.order():
        pool.busy[backend] += 1
        try:
            async with httpx.AsyncClient(timeout=GENERATE_TIMEOUT_SEC) as client:
                r = await client.post(f"{backend}/v1/images/generations", json=payload)
        except (httpx.ConnectError, httpx.ConnectTimeout) as e:
            # Never reached the backend: nothing drawn, the next one may be up
            LOG.warning("image backend %s unreachable: %s", backend, e)
            pool.mark_down(backend)
            unreachable.append(backend)
            continue
        except httpx.HTTPError as e:
            LOG.warning("image backend %s failed: %s", backend, e)
            raise HTTPException(502, f"image backend failed: {e}") from e
        finally:
            pool.busy[backend] -= 1
        pool.mark_up(backend)
        break
    if r is None:
        raise HTTPException(502, f"image backend unreachable: {', '.join(unreachable)}")
    if r.status_code >= 400:
        LOG.warning("image backend %s answered %s: %s", backend, r.status_code, r.text[:300])
        raise HTTPException(r.status_code, r.text[:300])
    data = (r.json().get("data") or [{}])[0]
    b64 = data.get("b64_json")
    if not b64:
        raise HTTPException(502, "image backend returned no image")
    LOG.info("%s %s on %s in %.1fs", model, payload["size"], backend, time.monotonic() - started)
    return chat_reply(model, b64)
