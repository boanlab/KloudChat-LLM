"""A request waits a bounded time for the page slot and is then refused with 503; a page
hung in its worker thread turns /health unhealthy. Needs FastAPI; skipped where it is
absent (the lint job), run inside the image:

    docker run --rm -v $PWD/services/ocr-shim/tests:/app/tests kloudchat-ocr-shim \
        python -m pytest -q tests
"""

from __future__ import annotations

import asyncio
import sys
import threading
import time
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

pytest.importorskip("fastapi")
import app as shim  # noqa: E402
from fastapi import HTTPException  # noqa: E402


@pytest.fixture
def slots(monkeypatch):
    monkeypatch.setattr(shim, "_GATE", asyncio.Semaphore(1))
    monkeypatch.setattr(shim, "_running", {})
    monkeypatch.setattr(shim, "QUEUE_WAIT_SEC", 0.1)
    monkeypatch.setattr(shim, "TIMEOUT_SEC", 0.05)
    monkeypatch.setattr(shim, "STUCK_SEC", 0.15)
    monkeypatch.setattr(shim, "EXIT_SEC", 60.0)


def _health() -> tuple[int, dict]:
    import json

    response = asyncio.run(shim.health())
    return response.status_code, json.loads(response.body)


def test_a_request_behind_a_hung_page_is_refused_busy(slots):
    hang = threading.Event()

    async def scenario():
        with pytest.raises(HTTPException) as first:
            await shim._run(hang.wait)
        assert first.value.status_code == 504
        started = time.monotonic()
        with pytest.raises(HTTPException) as second:
            await shim._run(lambda: "never")
        assert second.value.status_code == 503 and second.value.detail == "busy"
        assert time.monotonic() - started < 1
        hang.set()
        # The slot comes back once the thread really ends
        assert await shim._run(lambda: "done") == "done"

    try:
        asyncio.run(scenario())
    finally:
        hang.set()


def test_a_waiting_request_gets_the_slot_when_it_frees_in_time(slots, monkeypatch):
    monkeypatch.setattr(shim, "TIMEOUT_SEC", 1.0)
    monkeypatch.setattr(shim, "QUEUE_WAIT_SEC", 1.0)

    async def scenario():
        slow = asyncio.ensure_future(shim._run(time.sleep, 0.1))
        await asyncio.sleep(0.02)
        assert await shim._run(lambda: "next") == "next"
        await slow

    asyncio.run(scenario())


def test_health_is_ok_while_pages_run_in_time(slots):
    status, body = _health()
    assert status == 200 and body["status"] == "ok" and body["running"] == 0
    shim._running[1] = time.monotonic()
    status, body = _health()
    assert status == 200 and body["status"] == "ok" and body["running"] == 1


def test_health_fails_while_a_page_is_stuck(slots):
    hang = threading.Event()

    async def scenario():
        with pytest.raises(HTTPException):
            await shim._run(hang.wait)
        await asyncio.sleep(0.2)
        response = await shim.health()
        assert response.status_code == 503
        hang.set()
        while shim._running:
            await asyncio.sleep(0.01)
        assert (await shim.health()).status_code == 200

    try:
        asyncio.run(scenario())
    finally:
        hang.set()
    assert shim._running == {}
    status, body = _health()
    assert status == 200 and body["status"] == "ok"


def test_a_stuck_page_reports_its_age(slots):
    shim._running[1] = time.monotonic() - 1.0
    status, body = _health()
    assert status == 503 and body["status"] == "stuck" and body["oldestPageSec"] >= 1.0


def test_a_page_hung_past_exit_sec_ends_the_process(slots, monkeypatch):
    exits = []
    monkeypatch.setattr(shim, "EXIT_SEC", 0.5)
    monkeypatch.setattr(shim, "_exit", lambda: exits.append(1))
    shim._running[1] = time.monotonic() - 1.0

    async def scenario():
        response = await shim.health()
        await asyncio.sleep(0.7)
        return response.status_code

    assert asyncio.run(scenario()) == 503
    assert exits == [1]
