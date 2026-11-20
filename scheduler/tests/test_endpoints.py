"""VLLM_ENDPOINTS read as instances per model."""

from __future__ import annotations

from scheduler import endpoints


def _none(url):
    return []


def test_a_named_entry_counts_for_its_model():
    got = endpoints.count_instances(
        "qwen3.8-27b=http://a:8001, local/qwen3.8-27b=http://b:8001,qwen3.8-27b=http://a:8001/",
        serves=_none,
    )
    assert got == {"qwen3.8-27b": 2}


def test_a_bare_entry_counts_for_every_model_it_serves():
    listed = {"http://c:8000": ["local/bge-m3", "Org/Other-7B"], "http://d:8000": ["local/bge-m3"]}
    got = endpoints.count_instances("http://c:8000,http://d:8000", serves=lambda u: listed[u])
    assert got == {"bge-m3": 2, "Other-7B": 1}


def test_a_bare_entry_that_names_nothing_counts_for_nothing(capsys):
    assert endpoints.count_instances("http://down:8000", serves=_none) == {}
    assert "write id=http://down:8000" in capsys.readouterr().err


def test_empty_is_no_instances():
    assert endpoints.count_instances("", serves=_none) == {}
    assert endpoints.count_instances(" , ", serves=_none) == {}
