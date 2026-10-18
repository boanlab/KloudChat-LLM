"""Prometheus text of the GPU nodes, found from the model endpoint settings.

Behind the gateway's `/metrics/*` (the tools key is checked there):
  /metrics/targets  → JSON list of targets: name, kind (vllm | gpu), node
  /metrics/<name>   → that target's text; image servers in vLLM's series names
A name is only ever one the settings produced: the request never picks an upstream.
"""

from __future__ import annotations

import logging
import os
import re
import time

import httpx
from fastapi import FastAPI, HTTPException
from fastapi.responses import PlainTextResponse
from targets import Target, discover, image_text

LOG = logging.getLogger("metrics-shim")
logging.basicConfig(level=os.getenv("LOG_LEVEL", "INFO").upper())

SCRAPE_TIMEOUT_SEC = float(os.getenv("SCRAPE_TIMEOUT_SEC", "5.0"))
IMAGEGEN_SHIM_URL = os.getenv("IMAGEGEN_SHIM_URL", "http://imagegen-shim:9100").rstrip("/")
#: Node hostnames from gpu-exporter, re-read after this long.
HOSTNAME_TTL_SEC = 300.0

TARGETS: dict[str, Target] = {t.name: t for t in discover(dict(os.environ))}
LOG.info("Metrics targets: %s", ", ".join(TARGETS) or "none")

_hostnames: dict[str, tuple[float, str]] = {}
_NODE_INFO = re.compile(r'^kloudchat_node_info\{[^}]*hostname="([^"]+)"', re.M)

app = FastAPI(title="KloudChat Metrics Shim")


async def _get(client: httpx.AsyncClient, url: str) -> httpx.Response | None:
    try:
        return await client.get(url, follow_redirects=False)
    except httpx.HTTPError as exc:
        LOG.info("scrape %s failed: %s", url, type(exc).__name__)
        return None


async def _hostname(client: httpx.AsyncClient, address: str) -> str:
    """gpu-exporter's hostname for one node, else the address."""
    cached = _hostnames.get(address)
    if cached and time.monotonic() - cached[0] < HOSTNAME_TTL_SEC:
        return cached[1]
    gpu = next((t for t in TARGETS.values() if t.kind == "gpu" and t.address == address), None)
    name = address
    if gpu and (response := await _get(client, f"{gpu.url}/metrics")) and response.status_code == 200:
        if match := _NODE_INFO.search(response.text):
            name = match.group(1)
    _hostnames[address] = (time.monotonic(), name)
    return name


@app.get("/health")
async def health() -> dict:
    return {"status": "ok", "targets": len(TARGETS)}


@app.get("/metrics/targets")
async def targets() -> dict:
    async with httpx.AsyncClient(timeout=SCRAPE_TIMEOUT_SEC) as client:
        rows = []
        for target in TARGETS.values():
            rows.append({
                "name": target.name,
                "kind": "gpu" if target.kind == "gpu" else "vllm",
                "node": await _hostname(client, target.address),
            })
    return {"targets": rows}


async def _in_flight(client: httpx.AsyncClient, url: str) -> int | None:
    response = await _get(client, f"{IMAGEGEN_SHIM_URL}/health")
    if not response or response.status_code != 200:
        return None
    for row in response.json().get("backends", []):
        if str(row.get("url", "")).rstrip("/") == url:
            return int(row.get("inFlight") or 0)
    return None


@app.get("/metrics/{name}", response_class=PlainTextResponse)
async def metrics(name: str) -> str:
    target = TARGETS.get(name)
    if target is None:
        raise HTTPException(404, "unknown target")
    async with httpx.AsyncClient(timeout=SCRAPE_TIMEOUT_SEC) as client:
        if target.kind != "image":
            response = await _get(client, f"{target.url}/metrics")
            if not response or response.status_code != 200:
                raise HTTPException(502, "target did not answer")
            return response.text
        health = await _get(client, f"{target.url}/health")
        if not health or health.status_code >= 400:
            raise HTTPException(502, "target did not answer")
        response = await _get(client, f"{target.url}/metrics")
        text = response.text if response and response.status_code == 200 else None
        return image_text(target.model, text, await _in_flight(client, target.url))
