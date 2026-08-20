"""Retrieval index for KloudChat's RAG: chunk, embed, store, search.

Here rather than in KloudChat because the embedding model is here — indexing is
one hop to the GPU instead of two across a network boundary, and KloudChat stays
deployable with no GPU stack, falling back to lexical search.

Endpoints:
  PUT    /documents            index or replace one document
  POST   /search               nearest passages within one collection
  DELETE /documents/{doc_id}   forget one document
  DELETE /collections/{name}   forget a whole shelf
  GET    /health               readiness, including whether embeddings answer

**Collections are the permission.** A collection name is an opaque id minted by
KloudChat per (owner, agent). Every operation is scoped to the one named in the
request; nothing lists collections or searches across them.

**KloudChat owns the text.** What is stored here is derived — chunks and their
vectors, rebuildable from the source rows. Losing this volume costs a re-index,
not a document.
"""
from __future__ import annotations

import logging
import os
import re
from contextlib import asynccontextmanager
from typing import Any, Optional

import asyncpg
import httpx
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

log = logging.getLogger("index-shim")
logging.basicConfig(level=os.getenv("LOG_LEVEL", "INFO"))

#: No fallback: a default with credentials in it is a credential in the source
#: tree, and compose always supplies this one.
DATABASE_URL = os.environ["INDEX_DATABASE_URL"]
#: The model gateway. Embeddings are requested by name, so which backend answers
#: — local vLLM or a commercial fallback — is LiteLLM's decision, not ours.
LITELLM_URL = os.getenv("LITELLM_URL", "http://litellm:8000")
LITELLM_KEY = os.getenv("LITELLM_MASTER_KEY", "")
#: In order of preference. The first that answers is used, and its name is stored
#: on every row so a later change can be re-indexed incrementally instead of
#: silently mixing two vector spaces.
#: Reranker, empty to skip the second stage. A vector search compares question
#: and passage in one shared space; a reranker reads the pair together, which is
#: why 2.2 GiB of it separates a relevant passage from an irrelevant one far more
#: sharply than cosine distance does — the gap the `max_distance` note below had
#: to be tuned by hand.
RERANK_MODEL = os.getenv("RERANK_MODEL", "local/bge-reranker-v2-m3").strip()
#: Candidates pulled from pgvector per requested passage before reranking. The
#: reranker can only reorder what the vector stage returned, so this is the recall
#: it gets to work with; past a point it costs latency for passages that were
#: never close.
RERANK_CANDIDATES = int(os.getenv("RERANK_CANDIDATES", "5"))
#: Reranked passages below this are dropped. The two stages cut on different
#: scales: cosine distance is the coarse recall filter deciding what the reranker
#: sees, the reranker's score decides the answer. Marginal candidates are what
#: the reranker is for, so the recall filter must stay loose.
#:
#: Measured reranker scores against a four-passage shelf:
#:
#:   question the shelf answers      0.73 – 0.94
#:   loosely related, no answer      0.0005 – 0.025
#:   topic not on the shelf at all   <= 0.0002
#:
#: 0.1 sits in the empty band between the first two. "Loosely related" is dropped
#: deliberately: a retrieval layer that always answers teaches the model that the
#: shelf is relevant when it is not.
RERANK_MIN_SCORE = float(os.getenv("RERANK_MIN_SCORE", "0.1"))
#: Cosine cut used while reranking is on. Deliberately loose: it exists to bound
#: how much the reranker reads, not to decide relevance.
RERANK_RECALL_DISTANCE = float(os.getenv("RERANK_RECALL_DISTANCE", "0.85"))

EMBED_MODELS = [
    m.strip() for m in os.getenv("EMBED_MODELS", "local/bge-m3,text-embedding-3-small").split(",")
    if m.strip()
]
#: bge-m3 and text-embedding-3-small are both 1024 and 1536 respectively, so the
#: column is sized for the largest and shorter vectors are padded. Declared once:
#: changing it is a migration, not a setting.
EMBED_DIM = int(os.getenv("EMBED_DIM", "1536"))

#: Characters per chunk, and how much each repeats of the one before. Matches
#: KloudChat's lexical chunker so a passage cited by one path looks the same
#: coming from the other.
CHUNK = int(os.getenv("INDEX_CHUNK_CHARS", "900"))
OVERLAP = int(os.getenv("INDEX_CHUNK_OVERLAP", "150"))
#: A ceiling on one document. Past this the tail is dropped and said so in the
#: response, rather than silently indexing a prefix.
MAX_CHARS = int(os.getenv("INDEX_MAX_DOC_CHARS", "2000000"))

_SCHEMA = """
CREATE EXTENSION IF NOT EXISTS vector;

CREATE TABLE IF NOT EXISTS chunks (
    id          bigserial PRIMARY KEY,
    collection  text        NOT NULL,
    doc_id      text        NOT NULL,
    doc_name    text        NOT NULL DEFAULT '',
    source_url  text,
    ordinal     int         NOT NULL,
    body        text        NOT NULL,
    embed_model text        NOT NULL DEFAULT '',
    embedding   vector(%(dim)s),
    created_at  timestamptz NOT NULL DEFAULT now()
);

-- Every read is "within one collection", so the collection leads every index.
CREATE INDEX IF NOT EXISTS ix_chunks_collection ON chunks (collection);
CREATE UNIQUE INDEX IF NOT EXISTS ux_chunks_doc_ordinal
    ON chunks (collection, doc_id, ordinal);
"""

#: Built separately: an HNSW build on an empty table is instant, but it must come
#: after the column exists and it names the operator class explicitly.
_ANN_INDEX = """
CREATE INDEX IF NOT EXISTS ix_chunks_embedding
    ON chunks USING hnsw (embedding vector_cosine_ops)
"""


def chunk_text(text: str) -> list[str]:
    """Overlapping windows ending at a paragraph or sentence break where near.

    Hard cut when a document has neither — extracted tables.
    """
    body = re.sub(r"\n{3,}", "\n\n", (text or "").strip())[:MAX_CHARS]
    if not body:
        return []
    out: list[str] = []
    start = 0
    while start < len(body):
        end = min(start + CHUNK, len(body))
        if end < len(body):
            window = body[start:end]
            cut = max(window.rfind("\n\n"), window.rfind(". "), window.rfind("다.\n"))
            # Only honour a break in the back half; one at character 20 would
            # produce a chunk that is a heading and nothing else.
            if cut > CHUNK // 2:
                end = start + cut
        piece = body[start:end].strip()
        if piece:
            out.append(piece)
        if end >= len(body):
            break
        start = max(end - OVERLAP, start + 1)
    return out


class _Embedder:
    """Calls the gateway, caching which model answered.

    The preference list is tried in order, so a deployment with no local model
    uses the commercial fallback and one with neither fails at the first index
    rather than storing rows with no vectors.
    """

    def __init__(self) -> None:
        self.model: Optional[str] = None

    async def embed(self, texts: list[str]) -> tuple[list[list[float]], str]:
        if not texts:
            return [], self.model or ""
        candidates = ([self.model] if self.model else []) + [
            m for m in EMBED_MODELS if m != self.model
        ]
        last: str = "no embedding model configured"
        headers = {"Authorization": f"Bearer {LITELLM_KEY}"} if LITELLM_KEY else {}
        async with httpx.AsyncClient(timeout=httpx.Timeout(120.0, connect=10.0)) as client:
            for model in candidates:
                try:
                    r = await client.post(
                        f"{LITELLM_URL.rstrip('/')}/v1/embeddings",
                        json={"model": model, "input": texts},
                        headers=headers,
                    )
                    r.raise_for_status()
                    rows = (r.json() or {}).get("data") or []
                    if len(rows) != len(texts):
                        raise ValueError(f"{len(rows)} vectors for {len(texts)} inputs")
                    self.model = model
                    return [list(map(float, row["embedding"])) for row in rows], model
                except (httpx.HTTPError, KeyError, ValueError, TypeError) as exc:
                    last = f"{model}: {exc}"
                    log.info("embedding via %s failed: %s", model, exc)
                    # A cached model that just failed must not be preferred again.
                    if self.model == model:
                        self.model = None
        raise HTTPException(status_code=503, detail=f"embeddings unavailable ({last})")


def _to_pgvector(values: list[float]) -> str:
    """pgvector literal, zero-padded to the column width.

    Cosine distance over a zero-padded tail is unchanged within one model, and
    the search filters on `embed_model` so models never mix.
    """
    padded = list(values[:EMBED_DIM]) + [0.0] * max(0, EMBED_DIM - len(values))
    return "[" + ",".join(f"{v:.7g}" for v in padded) + "]"


embedder = _Embedder()


#: Width of the existing `embedding` column. pgvector stores the declared
#: dimension in `atttypmod` directly — unlike varchar, there is no length header
#: to subtract, and subtracting one anyway reports every table as four short.
_COLUMN_DIM = """
SELECT a.atttypmod
  FROM pg_attribute a
  JOIN pg_class c ON c.oid = a.attrelid
 WHERE c.relname = 'chunks' AND a.attname = 'embedding' AND a.attnum > 0
"""


@asynccontextmanager
async def lifespan(app: FastAPI):
    app.state.pool = await asyncpg.create_pool(DATABASE_URL, min_size=1, max_size=8)
    #: Set when the table's vector width does not match EMBED_DIM. Reported by
    #: /health and refused by the write path, rather than left to surface as an
    #: asyncpg DataError on every single insert.
    app.state.dim_error = ""
    async with app.state.pool.acquire() as conn:
        await conn.execute(_SCHEMA % {"dim": EMBED_DIM})
        # CREATE TABLE IF NOT EXISTS does not widen an existing column, so a
        # changed EMBED_DIM leaves the old width and every insert fails with an
        # error naming neither the setting nor the fix. Checked once, in words.
        found = await conn.fetchval(_COLUMN_DIM)
        if found and int(found) != EMBED_DIM:
            app.state.dim_error = (
                f"table holds vector({found}) but EMBED_DIM is {EMBED_DIM}. "
                "Re-index into a fresh table, or set EMBED_DIM back."
            )
            log.error("dimension mismatch: %s", app.state.dim_error)
        try:
            await conn.execute(_ANN_INDEX)
        except asyncpg.PostgresError as exc:
            # A missing ANN index costs speed, not correctness — the search still
            # runs as an exact scan. Refusing to start over it would take
            # retrieval down for a tuning problem.
            log.warning("HNSW index unavailable, falling back to exact scan: %s", exc)
    log.info("index-shim ready (dim=%s, models=%s)", EMBED_DIM, ",".join(EMBED_MODELS))
    yield
    await app.state.pool.close()


app = FastAPI(title="KloudChat index shim", lifespan=lifespan)


class Document(BaseModel):
    collection: str = Field(min_length=1, max_length=200)
    doc_id: str = Field(min_length=1, max_length=200)
    name: str = Field(default="", max_length=300)
    text: str = ""
    source_url: Optional[str] = None


class Query(BaseModel):
    collection: str = Field(min_length=1, max_length=200)
    query: str = Field(min_length=1, max_length=4000)
    limit: int = Field(default=4, ge=1, le=20)
    #: Passages further than this in cosine distance are dropped.
    #:
    #: Measured against bge-m3: a question the shelf answers scores 0.50–0.55
    #: similarity, one it does not scores 0.31–0.32. The cut sits between them at
    #: 0.42 similarity — 0.58 distance.
    max_distance: float = Field(default=0.58, ge=0.0, le=2.0)


@app.get("/health")
async def health() -> dict[str, Any]:
    """Readiness and embedding availability, reported separately.

    A shim whose database is up but whose embedding backend is gone still serves
    deletes, and the caller decides whether to fall back.
    """
    ok_db = False
    try:
        async with app.state.pool.acquire() as conn:
            await conn.fetchval("SELECT 1")
        ok_db = True
    except Exception as exc:  # noqa: BLE001
        log.warning("health: database unreachable: %s", exc)

    embed_model = ""
    try:
        _, embed_model = await embedder.embed(["health"])
    except HTTPException:
        pass
    mismatch = getattr(app.state, "dim_error", "")
    return {
        "status": "ok" if ok_db and not mismatch else "degraded",
        "database": ok_db,
        "embeddings": bool(embed_model),
        "model": embed_model,
        **({"error": mismatch} if mismatch else {}),
    }


@app.put("/documents")
async def put_document(doc: Document) -> dict[str, Any]:
    """Index one document, replacing any earlier version.

    Replace, not append: a re-uploaded file would otherwise keep answering from
    text it no longer contains.
    """
    if app.state.dim_error:
        raise HTTPException(status_code=503, detail=app.state.dim_error)
    pieces = chunk_text(doc.text)
    async with app.state.pool.acquire() as conn:
        async with conn.transaction():
            await conn.execute(
                "DELETE FROM chunks WHERE collection = $1 AND doc_id = $2",
                doc.collection,
                doc.doc_id,
            )
            if not pieces:
                # Empty document: a successful delete, not an error. Extraction
                # produced nothing and the caller already knows.
                return {"chunks": 0, "model": ""}

            vectors, model = await embedder.embed(pieces)
            await conn.executemany(
                """
                INSERT INTO chunks
                    (collection, doc_id, doc_name, source_url, ordinal, body,
                     embed_model, embedding)
                VALUES ($1, $2, $3, $4, $5, $6, $7, $8::vector)
                """,
                [
                    (
                        doc.collection,
                        doc.doc_id,
                        doc.name,
                        doc.source_url,
                        i + 1,
                        piece,
                        model,
                        _to_pgvector(vector),
                    )
                    for i, (piece, vector) in enumerate(zip(pieces, vectors, strict=True))
                ],
            )
    return {"chunks": len(pieces), "model": model}


async def _rerank(query: str, passages: list[dict]) -> Optional[list[dict]]:
    """Passages reordered by the reranker, or None if it could not be used.

    None rather than an exception on purpose: retrieval that silently degrades to
    vector order is worse than no reranking, but a shelf that stops answering
    because its second stage is down is worse still.
    """
    if not RERANK_MODEL or len(passages) < 2:
        return None
    headers = {"Authorization": f"Bearer {LITELLM_KEY}"} if LITELLM_KEY else {}
    try:
        async with httpx.AsyncClient(timeout=httpx.Timeout(30.0, connect=5.0)) as client:
            r = await client.post(
                f"{LITELLM_URL.rstrip('/')}/v1/rerank",
                json={
                    "model": RERANK_MODEL,
                    "query": query,
                    "documents": [p["text"] for p in passages],
                },
                headers=headers,
            )
            r.raise_for_status()
            results = r.json().get("results") or []
    except Exception as exc:  # noqa: BLE001 — any failure falls back to vector order
        log.warning("rerank unavailable, falling back to vector order: %s", exc)
        return None

    ordered: list[dict] = []
    for item in sorted(results, key=lambda x: -float(x.get("relevance_score", 0.0))):
        idx = int(item.get("index", -1))
        score = float(item.get("relevance_score", 0.0))
        if 0 <= idx < len(passages) and score >= RERANK_MIN_SCORE:
            # The reranker's score replaces the cosine one. They are not the same
            # quantity and blending them would mean nothing.
            ordered.append({**passages[idx], "score": round(score, 4)})
    # An empty list is an answer — nothing on the shelf was relevant — so it is
    # returned rather than falling back to vector order, which would answer anyway.
    return ordered


@app.post("/search")
async def search(q: Query) -> dict[str, Any]:
    """Nearest passages inside one collection.

    Filtered by `embed_model` as well: after a model change the index holds two
    vector spaces, and comparing across them returns confident nonsense. Old
    rows stay invisible until re-indexed.
    """
    if app.state.dim_error:
        raise HTTPException(status_code=503, detail=app.state.dim_error)
    vectors, model = await embedder.embed([q.query])
    literal = _to_pgvector(vectors[0])
    async with app.state.pool.acquire() as conn:
        rows = await conn.fetch(
            """
            SELECT doc_name, source_url, ordinal, body,
                   embedding <=> $2::vector AS distance
              FROM chunks
             WHERE collection = $1
               AND embed_model = $3
               AND embedding IS NOT NULL
             ORDER BY embedding <=> $2::vector
             LIMIT $4
            """,
            q.collection,
            literal,
            model,
            # Over-fetch so the reranker has something to choose between. It can
            # only reorder what this stage returned, so asking for exactly `limit`
            # would make the second stage decorative.
            q.limit * RERANK_CANDIDATES if RERANK_MODEL else q.limit,
        )
    # While reranking, the cosine cut is loosened to a recall bound. Precision is
    # the reranker's job, and the tuned 0.58 was chosen for a stage that has to
    # decide alone.
    cut = RERANK_RECALL_DISTANCE if RERANK_MODEL else q.max_distance
    passages = [
        {
            "document": r["doc_name"],
            "index": r["ordinal"],
            "text": r["body"],
            "source_url": r["source_url"],
            # Reported as a similarity, so the caller can blend it with a lexical
            # score without knowing that pgvector counts the other way.
            "score": round(max(0.0, 1.0 - float(r["distance"])), 4),
        }
        for r in rows
        if float(r["distance"]) <= cut
    ]
    reranked = await _rerank(q.query, passages)
    if reranked is None:
        # No reranker, or it could not be reached. Fall back to vector order and
        # to the cut that stage was tuned for.
        passages = [p for p in passages if p["score"] >= 1.0 - q.max_distance]
    else:
        passages = reranked
    return {"passages": passages[: q.limit], "model": model,
            "reranked": reranked is not None}


@app.delete("/documents/{doc_id}")
async def delete_document(doc_id: str, collection: str) -> dict[str, int]:
    async with app.state.pool.acquire() as conn:
        tag = await conn.execute(
            "DELETE FROM chunks WHERE collection = $1 AND doc_id = $2", collection, doc_id
        )
    return {"deleted": int(tag.rsplit(" ", 1)[-1] or 0)}


@app.delete("/collections/{collection}")
async def delete_collection(collection: str) -> dict[str, int]:
    """Forget a whole shelf. Triggered by agent deletion — without it the
    vectors stay searchable by anyone holding the collection id."""
    async with app.state.pool.acquire() as conn:
        tag = await conn.execute("DELETE FROM chunks WHERE collection = $1", collection)
    return {"deleted": int(tag.rsplit(" ", 1)[-1] or 0)}
