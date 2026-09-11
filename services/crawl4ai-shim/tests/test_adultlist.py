"""A scrape refuses a listed adult host, and only that host."""
import pathlib
import sys

import pytest

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from adultlist import ADULT, AdultList  # noqa: E402

HOSTS = """
# Title: StevenBlack/hosts (porn-only)
127.0.0.1 localhost
0.0.0.0 0.0.0.0
0.0.0.0 adult.example   # trailing comment
0.0.0.0 www.pics.example
0.0.0.0 fc2.example
0.0.0.0 www.sex
plain.example
"""


def test_hosts_file_lines_become_hosts_without_the_noise():
    lst = AdultList(HOSTS.splitlines())
    assert len(lst) == 4
    assert lst.listed("adult.example") and lst.listed("pics.example") and lst.listed("plain.example")
    assert not lst.listed("localhost")
    # A bare label would be a whole top-level domain.
    assert not lst.listed("sex") and not lst.listed("anything.sex")


@pytest.mark.parametrize("url", [
    "https://adult.example/",
    "http://ADULT.example:8080/page?x=1",
    "https://www.adult.example/",
    "https://www.pics.example/",
    "https://pics.example.",
])
def test_listed_hosts_are_refused_with_www_folded(url):
    assert AdultList(HOSTS.splitlines()).refusal(url) == ADULT


@pytest.mark.parametrize("url", [
    "https://example.com/",
    "https://notadult.example/",
    "https://adult.example.org/",
    # Hosts-file semantics: a listed apex says nothing about its subdomains.
    "https://blog.fc2.example/",
    "https://cdn.adult.example/a.jpg",
    "raw:<html>",
    "",
])
def test_other_hosts_pass(url):
    assert AdultList(HOSTS.splitlines()).refusal(url) is None


def test_the_list_loads_from_a_file(tmp_path):
    p = tmp_path / "hosts"
    p.write_text(HOSTS, encoding="utf-8")
    assert len(AdultList.from_file(p)) == 4
