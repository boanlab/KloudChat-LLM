"""Where a scrape may go: the public internet, nothing behind it.

The shim sits on the deployment's private network, next to LiteLLM, the databases and
the other tool services, and `/tools/fetch` reaches it without authentication. A URL
is therefore judged by the addresses its host resolves to, before the browser opens it;
every request the page then makes (navigation, redirect, script, XHR) passes through
`subrequest_refusal` on a Playwright route before it leaves the browser; and the address
the browser ended up at is checked once more on return.
"""
from __future__ import annotations

import ipaddress
import socket
from urllib.parse import urlsplit

_INTERNAL_HOSTS = frozenset({
    "localhost",
    "host.docker.internal",
    "gateway.docker.internal",
    "metadata.google.internal",
    "metadata",
})
_INTERNAL_SUFFIXES = (".localhost", ".local", ".internal", ".arpa")

SCHEME = "only http(s) URLs can be scraped"
INTERNAL = "internal network addresses cannot be scraped"
UNRESOLVED = "host could not be resolved"


def is_public(address: ipaddress.IPv4Address | ipaddress.IPv6Address) -> bool:
    """`is_global` covers the IANA special-purpose registries; multicast and IPv4-mapped are explicit."""
    mapped = getattr(address, "ipv4_mapped", None)
    if mapped is not None:
        return is_public(mapped)
    return address.is_global and not address.is_multicast


def _resolve(host: str) -> list[str]:
    try:
        found = socket.getaddrinfo(host, None, type=socket.SOCK_STREAM)
    except socket.gaierror:
        return []
    return [entry[4][0] for entry in found]


_MISSING = object()


class HostVerdicts:
    """Per-host refusal cache for one browser context; cleared whole at `limit`, no expiry."""

    def __init__(self, limit: int = 2048) -> None:
        self.limit = limit
        self._seen: dict[str, str | None] = {}

    def get(self, host: str):
        """The cached verdict, or `_MISSING`."""
        return self._seen.get(host, _MISSING)

    def put(self, host: str, verdict: str | None) -> None:
        if len(self._seen) >= self.limit:
            self._seen.clear()
        self._seen[host] = verdict


def subrequest_refusal(url: str, verdicts: HostVerdicts | None = None,
                       resolve=_resolve) -> str | None:
    """Why a request the page makes may not leave the browser; None when it may.

    Non-http(s) schemes (data:, blob:, about:) stay inside the browser and pass.
    Blocking: call it off the loop."""
    scheme = (url or "").split(":", 1)[0].lower()
    if scheme not in ("http", "https"):
        return None
    host = (urlsplit(url).hostname or "").rstrip(".").lower()
    if verdicts is not None:
        cached = verdicts.get(host)
        if cached is not _MISSING:
            return cached
    verdict = refusal(url, resolve=resolve)
    if verdicts is not None and host:
        verdicts.put(host, verdict)
    return verdict


def refusal(url: str, resolve=_resolve) -> str | None:
    """Why `url` may not be scraped; None when it may. Blocking: call it off the loop."""
    parts = urlsplit((url or "").strip())
    if parts.scheme.lower() not in ("http", "https"):
        return SCHEME
    try:
        host = (parts.hostname or "").rstrip(".").lower()
    except ValueError:
        return SCHEME
    if not host or parts.username is not None or parts.password is not None:
        return SCHEME
    if host in _INTERNAL_HOSTS or host.endswith(_INTERNAL_SUFFIXES):
        return INTERNAL
    try:
        literal = ipaddress.ip_address(host)
    except ValueError:
        literal = None
    if literal is not None:
        return None if is_public(literal) else INTERNAL
    # A dotless name is a service on this network (`litellm`, `code-interpreter`).
    if "." not in host:
        return INTERNAL
    addresses = resolve(host)
    if not addresses:
        return UNRESOLVED
    for text in addresses:
        try:
            address = ipaddress.ip_address(text.split("%", 1)[0])
        except ValueError:
            return UNRESOLVED
        if not is_public(address):
            return INTERNAL
    return None
