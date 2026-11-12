"""Instances already served outside the nodes, from VLLM_ENDPOINTS.

An entry is ``id=http://host:port`` (counted as given) or a bare
``http://host:port`` (asked which models it serves, through /v1/models). Each
distinct URL counts once per model. The placement treats these as instances
of that model.
"""

from __future__ import annotations

import json
import sys
import urllib.request
from typing import Callable, Iterable

#: Seconds to wait for one /v1/models answer
PROBE_TIMEOUT_SEC = 3.0


def model_id(served_name: str) -> str:
    """Catalogue id of a served name: without ``local/``, the last path segment."""
    return served_name.removeprefix("local/").rsplit("/", 1)[-1]


def served_models(url: str) -> list[str]:
    """Names a vLLM server lists at /v1/models; empty when it does not answer."""
    try:
        with urllib.request.urlopen(f"{url.rstrip('/')}/v1/models",
                                    timeout=PROBE_TIMEOUT_SEC) as r:
            data = json.load(r).get("data") or []
        return [str(m["id"]) for m in data if isinstance(m, dict) and m.get("id")]
    except Exception:  # noqa: BLE001 — unreachable reads as serving nothing
        return []


def count_instances(
    raw: str, *, serves: Callable[[str], Iterable[str]] = served_models,
) -> dict[str, int]:
    """Instances per model id from a VLLM_ENDPOINTS value."""
    urls: dict[str, set[str]] = {}
    for item in raw.split(","):
        item = item.strip().replace(" ", "")
        if not item:
            continue
        name, sep, url = item.partition("=")
        if sep:
            urls.setdefault(model_id(name), set()).add(url.rstrip("/"))
            continue
        found = [model_id(n) for n in serves(item)]
        if not found:
            print(f"VLLM_ENDPOINTS: {item} serves nothing it can name (write id={item})",
                  file=sys.stderr)
        for mid in found:
            urls.setdefault(mid, set()).add(item.rstrip("/"))
    return {mid: len(u) for mid, u in urls.items()}
