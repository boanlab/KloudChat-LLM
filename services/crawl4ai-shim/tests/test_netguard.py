"""A scrape reaches the public internet and nothing on the deployment's own network."""
import pathlib
import sys

import pytest

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

import netguard  # noqa: E402

_DNS = {
    "example.com": ["93.184.216.34"],
    "intranet.example": ["10.40.0.14"],
    "mixed.example": ["93.184.216.34", "127.0.0.1"],
    "mapped.example": ["::ffff:10.0.0.7"],
}


def _resolve(host):
    return _DNS.get(host, [])


@pytest.mark.parametrize("url", [
    "https://example.com/page",
    "http://example.com:8080/?q=1",
    "https://8.8.8.8/",
    "https://[2606:4700::1111]/",
])
def test_public_addresses_pass(url):
    assert netguard.refusal(url, _resolve) is None


@pytest.mark.parametrize(("url", "reason"), [
    ("file:///etc/passwd", netguard.SCHEME),
    ("raw:<html>", netguard.SCHEME),
    ("http://user:pw@example.com/", netguard.SCHEME),
    ("http://localhost:8000/health", netguard.INTERNAL),
    ("http://127.0.0.1:8000/", netguard.INTERNAL),
    ("http://[::1]/", netguard.INTERNAL),
    ("http://10.40.0.14:5433/", netguard.INTERNAL),
    ("http://169.254.169.254/latest/meta-data/", netguard.INTERNAL),
    ("http://[::ffff:127.0.0.1]/", netguard.INTERNAL),
    ("http://host.docker.internal:8100/", netguard.INTERNAL),
    ("http://litellm:8000/health", netguard.INTERNAL),
    ("http://code-interpreter:8000/exec", netguard.INTERNAL),
    ("http://0x7f000001/", netguard.INTERNAL),
    ("http://api.localhost/", netguard.INTERNAL),
    ("http://intranet.example/", netguard.INTERNAL),
    ("http://mixed.example/", netguard.INTERNAL),
    ("http://mapped.example/", netguard.INTERNAL),
    ("http://nowhere.example/", netguard.UNRESOLVED),
])
def test_everything_else_is_refused(url, reason):
    assert netguard.refusal(url, _resolve) == reason
