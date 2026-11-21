"""Backends come from IMAGEGEN_BACKENDS (hand-run) and IMAGEGEN_PLACED (the scheduler's), merged."""

from __future__ import annotations

import importlib
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

pytest.importorskip("fastapi")
pytest.importorskip("httpx")
import app as shim  # noqa: E402


@pytest.fixture
def reload_app(monkeypatch):
    """Reload the module under the test's environment, and again under the original one."""
    def load(**env: str):
        for key in ("IMAGEGEN_BACKENDS", "IMAGEGEN_PLACED"):
            monkeypatch.delenv(key, raising=False)
        for key, value in env.items():
            monkeypatch.setenv(key, value)
        return importlib.reload(shim)

    yield load
    monkeypatch.undo()
    importlib.reload(shim)


def test_placed_backends_join_the_hand_run_ones(reload_app):
    app = reload_app(
        IMAGEGEN_BACKENDS="z-image-turbo=http://gpu-1:8091",
        IMAGEGEN_PLACED="z-image-turbo=http://gpu-2:8091,qwen-image-2512=http://gpu-2:8092",
    )
    assert app.BACKENDS == {
        "z-image-turbo": ["http://gpu-1:8091", "http://gpu-2:8091"],
        "qwen-image-2512": ["http://gpu-2:8092"],
    }


def test_either_list_alone_is_enough(reload_app):
    assert reload_app(IMAGEGEN_PLACED="z-image-turbo=http://gpu-2:8091").BACKENDS == {
        "z-image-turbo": ["http://gpu-2:8091"]}
    assert reload_app(IMAGEGEN_BACKENDS="z-image-turbo=http://gpu-1:8091").BACKENDS == {
        "z-image-turbo": ["http://gpu-1:8091"]}
    assert reload_app().BACKENDS == {}
