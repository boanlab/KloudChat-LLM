"""The next transcription goes to the least busy healthy backend; a refusing one is passed over."""

from __future__ import annotations

import asyncio
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from backends import Backends, Unreachable  # noqa: E402

A, B, C = "http://gpu-1:8095", "http://gpu-2:8095", "http://gpu-3:8095"


class Refused(Exception):
    pass


class Clock:
    def __init__(self):
        self.now = 100.0

    def __call__(self):
        return self.now


def prober(up):
    calls = []

    async def probe(url):
        calls.append(url)
        return url in up

    probe.calls = calls
    return probe


def run(coro):
    return asyncio.run(coro)


def test_the_least_busy_healthy_backend_goes_first():
    pool = Backends([A, B, C], prober({A, B, C}), ttl=10, clock=Clock())
    pool.inflight[A] = 2
    pool.inflight[C] = 1
    assert run(pool.order()) == [B, C, A]


def test_unhealthy_backends_follow_in_configured_order():
    pool = Backends([A, B, C], prober({C}), ttl=10, clock=Clock())
    assert run(pool.order()) == [C, A, B]


def test_with_none_healthy_every_backend_is_tried_in_order():
    pool = Backends([A, B], prober(set()), ttl=10, clock=Clock())
    assert run(pool.order()) == [A, B]


def test_a_single_backend_is_not_probed():
    probe = prober(set())
    pool = Backends([A], probe, ttl=10, clock=Clock())
    assert run(pool.order()) == [A]
    assert probe.calls == []


def test_health_is_cached_for_its_ttl():
    clock = Clock()
    probe = prober({A, B})
    pool = Backends([A, B], probe, ttl=10, clock=clock)
    run(pool.order())
    clock.now += 9
    run(pool.order())
    assert len(probe.calls) == 2
    clock.now += 2
    run(pool.order())
    assert len(probe.calls) == 4


def test_a_refused_connection_moves_on_and_marks_the_backend_down():
    pool = Backends([A, B], prober({A, B}), ttl=10, clock=Clock())
    tried = []

    async def request(url):
        tried.append(url)
        if url == A:
            raise Refused("connection refused")
        return "transcript"

    assert run(pool.send(request, unreachable=(Refused,))) == (B, "transcript")
    assert tried == [A, B]
    assert pool.health[A] is False
    assert pool.inflight == {A: 0, B: 0}
    # Within the cache window the refusing backend goes last.
    assert run(pool.order()) == [B, A]


def test_an_error_after_the_request_was_accepted_is_not_retried():
    pool = Backends([A, B], prober({A, B}), ttl=10, clock=Clock())
    tried = []

    async def request(url):
        tried.append(url)
        raise TimeoutError("read timeout")

    with pytest.raises(TimeoutError):
        run(pool.send(request, unreachable=(Refused,)))
    assert tried == [A]
    assert pool.inflight[A] == 0


def test_every_backend_refusing_raises_unreachable():
    pool = Backends([A, B], prober({A, B}), ttl=10, clock=Clock())

    async def request(url):
        raise Refused(url)

    with pytest.raises(Unreachable) as exc:
        run(pool.send(request, unreachable=(Refused,)))
    assert exc.value.backends == [A, B]


def test_inflight_counts_while_the_request_runs():
    pool = Backends([A, B], prober({A, B}), ttl=10, clock=Clock())
    seen = []

    async def request(url):
        seen.append(dict(pool.inflight))
        return "ok"

    run(pool.send(request, unreachable=(Refused,)))
    assert seen == [{A: 1, B: 0}]
    assert pool.inflight == {A: 0, B: 0}


def test_urls_come_from_whisper_urls_and_the_placed_whisper_entries():
    from backends import parse_urls

    got = parse_urls(
        "http://a:9000/",
        "qwen=http://q:8001,whisper-large-v3=http://b:9000,local/whisper-large-v3=http://a:9000",
    )
    assert got == ["http://a:9000", "http://b:9000"]
    assert parse_urls("", "qwen=http://q:8001") == []
