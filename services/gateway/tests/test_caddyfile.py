"""Static checks on the gateway's /tools access control.

The behaviour itself (401 without the key, header stripped upstream) is
checked against a running Caddy; these keep the file from drifting.
"""

from __future__ import annotations

import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
CADDYFILE = (ROOT / "services/gateway/Caddyfile").read_text()
COMPOSE = (ROOT / "docker-compose.yml").read_text()


def _block(text: str, opener: str) -> str:
    """The brace-balanced block that starts at `opener`."""
    start = text.index(opener)
    depth = 0
    for i in range(text.index("{", start), len(text)):
        depth += {"{": 1, "}": -1}.get(text[i], 0)
        if depth == 0:
            return text[start : i + 1]
    raise AssertionError(f"unbalanced block: {opener}")


def test_every_tools_route_sits_behind_the_key():
    authorized = _block(_block(CADDYFILE, "handle /tools* {"), "handle @authorized {")
    metrics = _block(_block(CADDYFILE, "handle /metrics/* {"), "handle @metrics_authorized {")
    # Every upstream other than LiteLLM's two routes is inside an authorized block.
    outside = CADDYFILE.replace(authorized, "").replace(metrics, "")
    assert re.findall(r"reverse_proxy (\S+)", outside) == ["litellm:8000", "litellm:8000"]
    routes = re.findall(r"handle_path (/tools/\w+)\*", CADDYFILE)
    assert set(routes) == {
        "/tools/search",
        "/tools/fetch",
        "/tools/exec",
        "/tools/research",
        "/tools/stt",
        "/tools/index",
    }
    for route in routes:
        assert f"handle_path {route}*" in authorized


def test_the_matcher_compares_the_header_and_requires_a_long_key():
    matcher = re.search(r"@authorized expression `(.+)`", CADDYFILE)
    assert matcher
    expr = matcher.group(1)
    assert '{http.request.header.X-Tools-Key} == "{$TOOLS_GATEWAY_KEY}"' in expr
    assert 'size("{$TOOLS_GATEWAY_KEY}") >= 32' in expr


def test_the_key_is_stripped_before_every_tools_upstream():
    authorized = _block(_block(CADDYFILE, "handle /tools* {"), "handle @authorized {")
    proxies = re.findall(r"reverse_proxy [^\n{]+\{[^}]*\}", authorized)
    assert len(proxies) == 6
    for proxy in proxies:
        assert "header_up -X-Tools-Key" in proxy, proxy


def test_compose_refuses_to_start_the_gateway_without_a_key():
    gateway = COMPOSE.split("\n  gateway:\n", 1)[1].split("\n  # ──", 1)[0]
    assert re.search(r"TOOLS_GATEWAY_KEY: \$\{TOOLS_GATEWAY_KEY:\?", gateway)


METRICS_TARGETS = {
    "vllm-27b": "METRICS_VLLM_27B_URL",
    "vllm-122b": "METRICS_VLLM_122B_URL",
    "vllm-bgem3": "METRICS_VLLM_BGEM3_URL",
    "vllm-rerank": "METRICS_VLLM_RERANK_URL",
    "vllm-whisper": "METRICS_VLLM_WHISPER_URL",
    "gpu-1": "METRICS_GPU_1_URL",
    "gpu-2": "METRICS_GPU_2_URL",
}


def test_metrics_take_the_tools_key_and_answer_401_without_it():
    block = _block(CADDYFILE, "handle /metrics/* {")
    matcher = re.search(r"@metrics_authorized expression `(.+)`", block)
    tools = re.search(r"@authorized expression `(.+)`", CADDYFILE)
    assert matcher and tools and matcher.group(1) == tools.group(1)
    fallback = block.replace(_block(block, "handle @metrics_authorized {"), "")
    assert '`{"error":"unauthorized"}` 401' in fallback
    assert "reverse_proxy" not in fallback


def test_each_metrics_name_is_a_fixed_upstream_with_the_key_stripped():
    authorized = _block(_block(CADDYFILE, "handle /metrics/* {"), "handle @metrics_authorized {")
    names = re.findall(r"handle /metrics/([\w-]+) \{", authorized)
    assert set(names) == set(METRICS_TARGETS)
    for name, env in METRICS_TARGETS.items():
        route = _block(authorized, f"handle /metrics/{name} {{")
        assert "rewrite * /metrics" in route
        proxy = route[route.index("reverse_proxy ") :]
        assert proxy.startswith(f"reverse_proxy {{${env}}} {{\n"), proxy
        assert "header_up -X-Tools-Key" in proxy
    # No upstream is taken from the request
    assert "{http.request" not in authorized


def test_compose_gives_every_metrics_upstream_a_closed_default():
    gateway = COMPOSE.split("\n  gateway:\n", 1)[1].split("\n  # ──", 1)[0]
    for env in METRICS_TARGETS.values():
        assert f"{env}: ${{{env}:-127.0.0.1:9}}" in gateway
