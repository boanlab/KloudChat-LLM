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
    # Every upstream other than LiteLLM's two routes is inside the authorized block.
    outside = CADDYFILE.replace(authorized, "")
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
