"""Document text into overlapping passages."""
from __future__ import annotations

import os
import re

#: Chunk size and overlap in characters, matching KloudChat's lexical chunker
CHUNK = int(os.getenv("INDEX_CHUNK_CHARS", "900"))
OVERLAP = int(os.getenv("INDEX_CHUNK_OVERLAP", "150"))
#: Per-document ceiling; the tail is dropped
MAX_CHARS = int(os.getenv("INDEX_MAX_DOC_CHARS", "2000000"))


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
            # A break in the front half would leave a heading-only chunk.
            if cut > CHUNK // 2:
                end = start + cut
        piece = body[start:end].strip()
        if piece:
            out.append(piece)
        if end >= len(body):
            break
        start = max(end - OVERLAP, start + 1)
    return out
