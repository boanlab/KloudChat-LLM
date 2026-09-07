"""Where a scrape may go: the public internet, nothing behind it.

The shim sits on the deployment's private network, next to LiteLLM, the databases and
the other tool services, and `/tools/fetch` reaches it without authentication. A URL
is therefore judged by the addresses its host resolves to, before the browser opens it,
and again on the address the browser ended up at after redirects.
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
    """`is_global` follows the IANA special-purpose registries (loopback, private,
    link-local, CGNAT, documentation, reserved); multicast and IPv4-in-IPv6 are explicit."""
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
