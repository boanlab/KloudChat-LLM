"""Embedding model preference, per-collection pinning, and the cached health probe."""

from __future__ import annotations

import asyncio
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from embedding import (  # noqa: E402
    COLLECTION_MODEL_SQL,
    Embedder,
    EmbeddingUnavailable,
    EmbedProbe,
    collection_model,
    gateway_name,
)

LOCAL, CLOUD = "local/bge-m3", "text-embedding-3-small"


class Gateway:
    """Fake /v1/embeddings: models in `down` fail, the rest answer."""

    def __init__(self, down=()):
        self.down = set(down)
        self.calls: list[tuple[str, float | None]] = []

    async def __call__(self, model, texts, timeout):
        self.calls.append((model, timeout))
        if model in self.down:
            raise ConnectionError(f"{model} down")
        return [[1.0, 2.0] for _ in texts]


def run(coro):
    return asyncio.run(coro)


def test_unpinned_call_falls_back_and_remembers_the_answering_model():
    gw = Gateway(down={LOCAL})
    emb = Embedder([LOCAL, CLOUD], gw)
    vectors, model = run(emb.embed(["a", "b"]))
    assert model == CLOUD and len(vectors) == 2
    assert emb.model == CLOUD
    gw.calls.clear()
    run(emb.embed(["c"]))
    assert [m for m, _ in gw.calls] == [CLOUD]


def test_pinned_collection_model_never_falls_back():
    gw = Gateway(down={LOCAL})
    emb = Embedder([LOCAL, CLOUD], gw)
    with pytest.raises(EmbeddingUnavailable, match=LOCAL):
        run(emb.embed(["a"], model=LOCAL))
    assert [m for m, _ in gw.calls] == [LOCAL]


def test_pinned_model_outside_the_preference_list_is_used_as_is():
    gw = Gateway()
    emb = Embedder([LOCAL], gw)
    _, model = run(emb.embed(["a"], model="old/model"))
    assert model == "old/model"


def test_a_failed_preferred_model_loses_its_preference():
    gw = Gateway()
    emb = Embedder([LOCAL, CLOUD], gw)
    emb.model = CLOUD
    gw.down = {CLOUD}
    _, model = run(emb.embed(["a"]))
    assert model == LOCAL and emb.model == LOCAL


def test_health_probe_leaves_the_preference_alone():
    gw = Gateway(down={LOCAL})
    emb = Embedder([LOCAL, CLOUD], gw)
    _, model = run(emb.embed(["health"], remember=False, timeout=3.0))
    assert model == CLOUD
    assert emb.model is None
    assert all(t == 3.0 for _, t in gw.calls)


def test_vector_count_mismatch_counts_as_a_failure():
    async def short(model, texts, timeout):
        return [[0.0]]

    emb = Embedder([LOCAL], short)
    with pytest.raises(EmbeddingUnavailable, match="1 vectors for 2 inputs"):
        run(emb.embed(["a", "b"]))


def test_no_texts_need_no_call():
    gw = Gateway()
    assert run(Embedder([LOCAL], gw).embed([], model=CLOUD)) == ([], CLOUD)
    assert gw.calls == []


class Conn:
    def __init__(self, value):
        self.value = value
        self.args = None

    async def fetchval(self, sql, *args):
        self.args = (sql, *args)
        return self.value


def test_collection_model_is_the_majority_model_or_empty():
    conn = Conn(LOCAL)
    assert run(collection_model(conn, "c1")) == LOCAL
    assert conn.args == (COLLECTION_MODEL_SQL, "c1")
    assert run(collection_model(Conn(None), "c2")) == ""


class Clock:
    def __init__(self):
        self.now = 1000.0

    def __call__(self):
        return self.now


def test_probe_result_is_cached_for_its_ttl():
    clock = Clock()
    calls = []

    async def check():
        calls.append(clock.now)
        return LOCAL

    async def scenario():
        probe = EmbedProbe(check, ttl=300, wait=3, clock=clock)
        assert await probe.current() == LOCAL
        clock.now += 299
        assert await probe.current() == LOCAL
        assert len(calls) == 1
        clock.now += 2
        assert await probe.current() == LOCAL
        assert len(calls) == 2

    run(scenario())


def test_a_slow_probe_does_not_hold_health_past_its_wait():
    gate = None

    async def check():
        await gate.wait()
        return CLOUD

    async def scenario():
        nonlocal gate
        gate = asyncio.Event()
        probe = EmbedProbe(check, ttl=300, wait=0.05, clock=Clock())
        loop = asyncio.get_running_loop()
        started = loop.time()
        assert await probe.current() == ""
        assert loop.time() - started < 1.0
        # The probe keeps running and its result is reported once it lands.
        gate.set()
        await asyncio.sleep(0)
        await asyncio.sleep(0)
        assert await probe.current() == CLOUD

    run(scenario())


def test_a_running_probe_is_not_started_twice():
    calls = []

    async def check():
        calls.append(1)
        await asyncio.sleep(0.1)
        return LOCAL

    async def scenario():
        clock = Clock()
        probe = EmbedProbe(check, ttl=0, wait=0.01, clock=clock)
        await asyncio.gather(*(probe.current() for _ in range(5)))
        assert len(calls) == 1

    run(scenario())


def test_a_failed_probe_reports_no_model():
    async def check():
        raise EmbeddingUnavailable("all down")

    probe = EmbedProbe(check, ttl=300, wait=3, clock=Clock())
    assert run(probe.current()) == ""


def test_a_collection_embedded_as_local_is_asked_for_by_its_strict_local_name():
    assert gateway_name("local/bge-m3") == "strict-local/bge-m3"
    assert gateway_name("strict-local/bge-m3") == "strict-local/bge-m3"
    assert gateway_name("other/model") == "other/model"
