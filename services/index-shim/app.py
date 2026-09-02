"""Retrieval index for KloudChat's RAG: chunk, embed, store, search.

Endpoints:
  PUT    /documents            index or replace one document
  POST   /search               nearest passages within one collection
  DELETE /documents/{doc_id}   forget one document
  DELETE /collections/{name}   forget a whole collection
  GET    /health               readiness, embedding availability

A collection is an opaque id minted by KloudChat per (owner, agent) and scopes
every operation; nothing lists or searches across collections. Stored data is
derived (chunks and vectors), rebuildable from KloudChat's source rows.
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

#: Required; compose supplies it. No default with credentials in the source tree.
DATABASE_URL = os.environ["INDEX_DATABASE_URL"]
#: Model gateway. Embeddings and reranking are requested by model name; LiteLLM
#: decides which backend answers.
LITELLM_URL = os.getenv("LITELLM_URL", "http://litellm:8000")
LITELLM_KEY = os.getenv("LITELLM_MASTER_KEY", "")
#: Reranker model. Empty skips the second stage.
RERANK_MODEL = os.getenv("RERANK_MODEL", "local/bge-reranker-v2-m3").strip()
#: Vector candidates fetched per requested passage before reranking.
RERANK_CANDIDATES = int(os.getenv("RERANK_CANDIDATES", "5"))
#: Reranker score floor. Measured with bge-reranker-v2-m3: passages that answer
#: the question score 0.73–0.94, loosely related ones <= 0.025.
RERANK_MIN_SCORE = float(os.getenv("RERANK_MIN_SCORE", "0.1"))
#: Cosine distance bound on reranker candidates. Loose on purpose: a recall
#: filter, not a relevance decision.
RERANK_RECALL_DISTANCE = float(os.getenv("RERANK_RECALL_DISTANCE", "0.85"))

#: Embedding models in preference order. The first that answers is used and its
#: name is stored on every row, so vector spaces never mix.
EMBED_MODELS = [
    m.strip() for m in os.getenv("EMBED_MODELS", "local/bge-m3,text-embedding-3-small").split(",")
    if m.strip()
]
#: Vector column width: the widest model in EMBED_MODELS (text-embedding-3-small
#: is 1536, bge-m3 is 1024 and zero-padded). Changing it is a migration.
EMBED_DIM = int(os.getenv("EMBED_DIM", "1536"))

#: Chunk size and overlap in characters. Matches KloudChat's lexical chunker.
CHUNK = int(os.getenv("INDEX_CHUNK_CHARS", "900"))
OVERLAP = int(os.getenv("INDEX_CHUNK_OVERLAP", "150"))
#: Per-document ceiling; the tail past it is dropped.
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

-- Every read is scoped to one collection.
CREATE INDEX IF NOT EXISTS ix_chunks_collection ON chunks (collection);
CREATE UNIQUE INDEX IF NOT EXISTS ux_chunks_doc_ordinal
    ON chunks (collection, doc_id, ordinal);
"""

#: HNSW index, created after the table exists.
_ANN_INDEX = """
CREATE INDEX IF NOT EXISTS ix_chunks_embedding
    ON chunks USING hnsw (embedding vector_cosine_ops)
"""


def chunk_text(text: str) -> list[str]:
    """Overlapping windows, cut at a nearby paragraph or sentence break."""
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
            # Breaks in the front half would yield heading-only chunks.
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
    """Embeds through the gateway, remembering which model answered."""

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
                    # Failed model loses its cached preference.
                    if self.model == model:
                        self.model = None
        raise HTTPException(status_code=503, detail=f"embeddings unavailable ({last})")


def _to_pgvector(values: list[float]) -> str:
    """pgvector literal, zero-padded to the column width."""
    padded = list(values[:EMBED_DIM]) + [0.0] * max(0, EMBED_DIM - len(values))
    return "[" + ",".join(f"{v:.7g}" for v in padded) + "]"


embedder = _Embedder()


#: Declared width of the existing `embedding` column (pgvector stores it in
#: atttypmod directly, no header offset).
_COLUMN_DIM = """
SELECT a.atttypmod
  FROM pg_attribute a
  JOIN pg_class c ON c.oid = a.attrelid
 WHERE c.relname = 'chunks' AND a.attname = 'embedding' AND a.attnum > 0
"""


@asynccontextmanager
async def lifespan(app: FastAPI):
    app.state.pool = await asyncpg.create_pool(DATABASE_URL, min_size=1, max_size=8)
    #: Set when the table's vector width differs from EMBED_DIM; reported by
    #: /health and refused by the write path.
    app.state.dim_error = ""
    async with app.state.pool.acquire() as conn:
        await conn.execute(_SCHEMA % {"dim": EMBED_DIM})
        # CREATE TABLE IF NOT EXISTS does not widen an existing column.
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
            # Without the ANN index the search runs as an exact scan.
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
    #: Cosine distance cut when no reranker runs. Measured with bge-m3: answered
    #: questions score 0.50–0.55 similarity, unanswered 0.31–0.32.
    max_distance: float = Field(default=0.58, ge=0.0, le=2.0)


@app.get("/health")
async def health() -> dict[str, Any]:
    """Database readiness and embedding availability, reported separately."""
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
    """Index one document, replacing any earlier version."""
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
                # Empty document: a successful delete.
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
    """Passages reordered by the reranker, or None when it is off or unreachable."""
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
    except Exception as exc:  # noqa: BLE001
        log.warning("rerank unavailable, falling back to vector order: %s", exc)
        return None

    ordered: list[dict] = []
    for item in sorted(results, key=lambda x: -float(x.get("relevance_score", 0.0))):
        idx = int(item.get("index", -1))
        score = float(item.get("relevance_score", 0.0))
        if 0 <= idx < len(passages) and score >= RERANK_MIN_SCORE:
            # Reranker score replaces the cosine score; the two are not comparable.
            ordered.append({**passages[idx], "score": round(score, 4)})
    # Empty is an answer (nothing relevant), not a fallback trigger.
    return ordered


@app.post("/search")
async def search(q: Query) -> dict[str, Any]:
    """Nearest passages inside one collection, from rows in the current model's vector space."""
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
            # Over-fetch so the reranker has candidates to reorder.
            q.limit * RERANK_CANDIDATES if RERANK_MODEL else q.limit,
        )
    # With a reranker the cosine cut is only a recall bound.
    cut = RERANK_RECALL_DISTANCE if RERANK_MODEL else q.max_distance
    passages = [
        {
            "document": r["doc_name"],
            "index": r["ordinal"],
            "text": r["body"],
            "source_url": r["source_url"],
            # Similarity, not distance, so callers can blend it with lexical scores.
            "score": round(max(0.0, 1.0 - float(r["distance"])), 4),
        }
        for r in rows
        if float(r["distance"]) <= cut
    ]
    reranked = await _rerank(q.query, passages)
    if reranked is None:
        # Vector order with the cut tuned for it.
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
    """Forget a whole collection (agent deletion)."""
    async with app.state.pool.acquire() as conn:
        tag = await conn.execute("DELETE FROM chunks WHERE collection = $1", collection)
    return {"deleted": int(tag.rsplit(" ", 1)[-1] or 0)}
