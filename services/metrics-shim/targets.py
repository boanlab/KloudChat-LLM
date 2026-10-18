"""Metrics targets from the model endpoint settings, and image-server text in vLLM's shape."""

from __future__ import annotations

import re
from dataclasses import dataclass
from urllib.parse import urlsplit

#: Model endpoint settings scanned for vLLM servers; each a comma-separated URL list.
VLLM_KEYS = (
    "VLLM_QWEN27B_URL",
    "VLLM_QWENFLASH_URL",
    "VLLM_CODERNEXT_URL",
    "VLLM_BGEM3_URL",
    "VLLM_RERANK_URL",
    "WHISPER_URLS",
)
GPU_EXPORTER_PORT = 9835
#: vllm-omni's request histogram for one picture.
IMAGE_HANDLER = "/v1/images/generations"


@dataclass(frozen=True)
class Target:
    name: str
    kind: str  # "vllm" | "image" | "gpu"
    url: str  # base URL, no path
    address: str  # node host or IP
    model: str = ""  # served name, image targets only


def _slug(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", text.lower()).strip("-")


def _base(raw: str) -> tuple[str, str, int] | None:
    """(base URL, host, port) of one endpoint URL, or None when unusable."""
    raw = raw.strip()
    if not raw:
        return None
    parts = urlsplit(raw if "://" in raw else f"http://{raw}")
    if not parts.hostname:
        return None
    port = parts.port or (443 if parts.scheme == "https" else 80)
    return f"{parts.scheme}://{parts.netloc}", parts.hostname, port


def _images(raw: str) -> list[tuple[str, str]]:
    """(served name, URL) pairs of IMAGEGEN_BACKENDS."""
    out = []
    for item in raw.split(","):
        name, sep, url = item.strip().partition("=")
        if sep and name.strip() and url.strip():
            out.append((name.strip(), url.strip()))
    return out


def discover(env: dict[str, str]) -> list[Target]:
    """Every model server and GPU node the settings name, once each.

    GPU nodes: hosts of the model endpoints plus NODES_VLLM (`user@host`), each
    with its gpu-exporter.
    """
    found: dict[str, Target] = {}
    hosts: list[str] = []

    def add(target: Target) -> None:
        found.setdefault(target.name, target)
        if target.kind != "gpu" and target.address not in hosts:
            hosts.append(target.address)

    for key in VLLM_KEYS:
        for raw in env.get(key, "").split(","):
            if parsed := _base(raw):
                url, host, port = parsed
                add(Target(f"vllm-{_slug(host)}-{port}", "vllm", url, host))
    for model, raw in _images(env.get("IMAGEGEN_BACKENDS", "")):
        if parsed := _base(raw):
            url, host, port = parsed
            add(Target(f"image-{_slug(host)}-{port}", "image", url, host, model))
    for node in env.get("NODES_VLLM", "").split(","):
        host = node.strip().rpartition("@")[2]
        if host and host not in hosts:
            hosts.append(host)
    port = int(env.get("GPU_EXPORTER_PORT") or GPU_EXPORTER_PORT)
    for host in hosts:
        if host in ("localhost", "127.0.0.1"):
            continue
        add(Target(f"gpu-{_slug(host)}", "gpu", f"http://{host}:{port}", host))
    return list(found.values())


# ── image servers as vLLM text ──────────────────────────────────────────

_SAMPLE = re.compile(r"^([a-zA-Z_:][\w:]*)(?:\{([^}]*)\})?\s+(\S+)")
_LABEL = re.compile(r'(\w+)="((?:[^"\\]|\\.)*)"')


def _quote(value: str) -> str:
    return value.replace("\\", "\\\\").replace('"', '\\"')


def image_text(model: str, metrics: str | None, in_flight: int | None) -> str:
    """vLLM-named series for one vllm-omni server: one picture at a time, the rest wait.

    The picture route's request count and total time carry over; its buckets stop
    at 1 s, too short for a picture, so they are left out.
    """
    label = f'model_name="{_quote(model)}"'
    lines = []
    if in_flight is not None:
        lines.append(f"vllm:num_requests_running{{{label}}} {min(in_flight, 1)}")
        lines.append(f"vllm:num_requests_waiting{{{label}}} {max(in_flight - 1, 0)}")
    for line in (metrics or "").splitlines():
        match = _SAMPLE.match(line)
        if not match or not match.group(1).startswith("http_request_duration_seconds_"):
            continue
        labels = dict(_LABEL.findall(match.group(2) or ""))
        if labels.get("handler") != IMAGE_HANDLER:
            continue
        suffix = match.group(1).removeprefix("http_request_duration_seconds")
        if suffix in ("_count", "_sum"):
            lines.append(f"vllm:e2e_request_latency_seconds{suffix}{{{label}}} {match.group(3)}")
    if not lines:
        lines.append(f"vllm:num_requests_running{{{label}}} 0")
    return "\n".join(lines) + "\n"
