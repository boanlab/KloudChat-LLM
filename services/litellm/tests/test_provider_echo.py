from __future__ import annotations

import importlib
import sys
import types

import pytest


@pytest.fixture
def echo(monkeypatch: pytest.MonkeyPatch):
    class Out:
        provider = None

    class Handler:
        def chunk_parser(self, chunk):  # LiteLLM's own parser drops `provider`
            return Out()

    stubs = {
        "litellm": types.ModuleType("litellm"),
        "litellm.integrations": types.ModuleType("litellm.integrations"),
        "litellm.integrations.custom_logger": types.ModuleType("litellm.integrations.custom_logger"),
        "litellm.llms": types.ModuleType("litellm.llms"),
        "litellm.llms.openrouter": types.ModuleType("litellm.llms.openrouter"),
        "litellm.llms.openrouter.chat": types.ModuleType("litellm.llms.openrouter.chat"),
        "litellm.llms.openrouter.chat.transformation": types.ModuleType("t"),
    }
    stubs["litellm.integrations.custom_logger"].CustomLogger = type("CustomLogger", (), {})
    stubs["litellm.llms.openrouter.chat.transformation"].OpenRouterChatCompletionStreamingHandler = Handler
    for name, mod in stubs.items():
        monkeypatch.setitem(sys.modules, name, mod)
    monkeypatch.delitem(sys.modules, "services.litellm.callbacks.provider_echo", raising=False)
    module = importlib.import_module("services.litellm.callbacks.provider_echo")
    return module, Handler


def test_the_streamed_chunk_keeps_the_provider(echo) -> None:
    _, handler = echo
    assert handler().chunk_parser({"provider": "Wafer"}).provider == "Wafer"
    assert handler().chunk_parser({"id": "x"}).provider is None
    assert handler().chunk_parser({"provider": 5}).provider is None


def test_applying_twice_wraps_once(echo) -> None:
    module, handler = echo
    first = handler.chunk_parser
    assert module.apply() is True
    assert handler.chunk_parser is first


def test_a_missing_parser_is_skipped(monkeypatch: pytest.MonkeyPatch, echo) -> None:
    module, _ = echo
    monkeypatch.setitem(sys.modules, "litellm.llms.openrouter.chat.transformation", types.ModuleType("t"))
    assert module.apply() is False
