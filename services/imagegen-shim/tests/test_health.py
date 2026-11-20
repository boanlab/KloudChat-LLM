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


def _edit_request(messages):
    class Req:
        async def json(self):
            return {"model": "local/qwen-image-edit-2511", "messages": messages}

    return Req()


def test_an_edit_goes_to_the_edits_route_as_multipart(monkeypatch):
    seen = {}

    async def answer(request: httpx.Request) -> httpx.Response:
        seen["path"] = request.url.path
        seen["type"] = request.headers["content-type"]
        seen["body"] = request.content
        return httpx.Response(200, json={"data": [{"b64_json": "UE5H"}]})

    _backends(monkeypatch, {"qwen-image-edit-2511": ["http://gpu-1:8093"]}, answer)
    messages = [{"role": "user", "content": [
        {"type": "text", "text": "make the sky pink"},
        {"type": "image_url", "image_url": {"url": "data:image/png;base64,QUJD"}},
    ]}]
    reply = asyncio.run(shim.chat_completions(_edit_request(messages)))
    assert seen["path"] == "/v1/images/edits"
    assert seen["type"].startswith("multipart/form-data")
    assert b'name="image"' in seen["body"] and b"ABC" in seen["body"]
    assert b"make the sky pink" in seen["body"] and b"auto" in seen["body"]
    assert reply["choices"][0]["message"]["images"][0]["image_url"]["url"].endswith("UE5H")


@pytest.mark.parametrize("content", ["no picture", [{"type": "image_url", "image_url": {"url": "https://x/y.png"}}]])
def test_an_edit_without_a_usable_picture_is_refused_before_any_call(monkeypatch, content):
    async def answer(request: httpx.Request) -> httpx.Response:
        raise AssertionError("no backend call expected")

    _backends(monkeypatch, {"qwen-image-edit-2511": ["http://gpu-1:8093"]}, answer)
    with pytest.raises(shim.HTTPException) as e:
        asyncio.run(shim.chat_completions(_edit_request([{"role": "user", "content": content}])))
    assert e.value.status_code == 400
