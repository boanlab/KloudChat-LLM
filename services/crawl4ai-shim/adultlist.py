"""Adult sites a scrape may not go to, judged at the same points as netguard: before
the browser opens a URL, and again on the address it ended up at after redirects.

The list is a hosts file (`0.0.0.0 host` lines; plain host names work too) baked into
the image at build time from a pinned commit of StevenBlack/hosts' porn-only list. It
keeps hosts-file semantics: a host is refused when it is listed, with `www.` folded on
both sides, and nothing else is inferred from an entry. Listing an apex does not cover
its subdomains: the list names shared platforms (fc2.com) and bare labels (www.sex)
that would otherwise take a whole blog host or top-level domain with them.
"""
from __future__ import annotations

import pathlib
from typing import Iterable
from urllib.parse import urlsplit

ADULT = "adult sites cannot be scraped"

_NOISE = frozenset({
    "localhost", "localhost.localdomain", "broadcasthost", "0.0.0.0", "ip6-localhost",
    "ip6-loopback", "ip6-localnet", "ip6-mcastprefix", "ip6-allnodes", "ip6-allrouters",
    "ip6-allhosts",
})


def _host(text: str) -> str:
    """One host, `www.` folded, lower-case; '' for what is not a host."""
    host = text.strip().strip(".").lower()
    if host.startswith("www."):
        host = host[4:]
    # A bare label is a top-level domain, not a site.
    if not host or "." not in host or host in _NOISE:
        return ""
    return host


def _entry(line: str) -> str:
    """One hosts-file line to a host; '' for comments, blanks and noise."""
    fields = line.split("#", 1)[0].split()
    if not fields:
        return ""
    return _host(fields[1] if len(fields) > 1 else fields[0])


class AdultList:
    def __init__(self, lines: Iterable[str] = ()) -> None:
        self._hosts = frozenset(h for h in (_entry(x) for x in lines) if h)

    @classmethod
    def from_file(cls, path: str | pathlib.Path) -> AdultList:
        with open(path, encoding="utf-8", errors="replace") as f:
            return cls(f)

    def __len__(self) -> int:
        return len(self._hosts)

    def listed(self, host: str) -> bool:
        return _host(host) in self._hosts

    def refusal(self, url: str) -> str | None:
        """Why `url` may not be scraped; None when it may."""
        try:
            host = urlsplit((url or "").strip()).hostname or ""
        except ValueError:
            return None
        return ADULT if self.listed(host) else None
