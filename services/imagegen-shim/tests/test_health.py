"""/health probes every backend at once: a few unreachable nodes cost one probe timeout,
not one each. Needs FastAPI and httpx; skipped where they are absent (the lint job)."""

from __future__ import annotations

import asyncio
import sys
import time
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

pytest.importorskip("fastapi")
httpx = pytest.importorskip("httpx")
import app as shim  # noqa: E402
from pool import Pool  # noqa: E402

UP_1, DOWN_1 = "http://gpu-1:8091", "http://gpu-2:8091"
DOWN_2, DOWN_3 = "http://gpu-1:8092", "http://gpu-2:8092"
SLOW = 0.3


def _backends(monkeypatch, backends: dict[str, list[str]], answer) -> None:
    monkeypatch.setattr(shim, "BACKENDS", backends)
    monkeypatch.setattr(shim, "POOLS", {name: Pool(urls) for name, urls in backends.items()})
    real = httpx.AsyncClient

    def client(**kwargs):
        return real(transport=httpx.MockTransport(answer), **kwargs)

    monkeypatch.setattr(shim.httpx, "AsyncClient", client)


async def _answer(request: httpx.Request) -> httpx.Response:
    url = f"{request.url.scheme}://{request.url.host}:{request.url.port}"
    await asyncio.sleep(SLOW)
    if url == UP_1:
        return httpx.Response(200)
    if url == DOWN_3:
        return httpx.Response(503)
    raise httpx.ConnectTimeout("no route", request=request)


def test_backends_are_probed_at_once_and_reported_in_order(monkeypatch):
    _backends(monkeypatch, {"local/z": [UP_1, DOWN_1], "local/q": [DOWN_2, DOWN_3]}, _answer)
    shim.POOLS["local/z"].busy[UP_1] = 2
    started = time.monotonic()
    body = asyncio.run(shim.health())
    assert time.monotonic() - started < 2 * SLOW
    assert body["status"] == "ok"
    assert [(r["model"], r["url"], r["up"], r["inFlight"]) for r in body["backends"]] == [
        ("local/z", UP_1, True, 2),
        ("local/z", DOWN_1, False, 0),
        ("local/q", DOWN_2, False, 0),
        ("local/q", DOWN_3, False, 0),
    ]


def test_health_is_degraded_when_no_backend_answers(monkeypatch):
    _backends(monkeypatch, {"local/q": [DOWN_1, DOWN_2, DOWN_3]}, _answer)
    started = time.monotonic()
    body = asyncio.run(shim.health())
    assert time.monotonic() - started < 2 * SLOW
    assert body["status"] == "degraded" and not any(r["up"] for r in body["backends"])


def test_health_without_backends_is_degraded(monkeypatch):
    _backends(monkeypatch, {}, _answer)
    assert asyncio.run(shim.health()) == {"status": "degraded", "backends": []}
