"""Provider echo: pass the provider that answered through LiteLLM's streaming responses.

OpenRouter names the provider that served a request in a top-level `provider` field of
every response and stream chunk. LiteLLM's OpenRouter stream parser rebuilds each chunk and
drops it, so a client streaming through the proxy cannot tell which provider answered (the
non-streaming response keeps it). This module patches the parser to carry the field over;
KloudChat reads `provider` from the chunks to show the answer's provider in 「이번 답」.

Loaded as a callback (`callbacks.provider_echo.echo_instance`) so every worker applies it at
start-up. If LiteLLM's internals move, the patch is skipped with a warning and nothing else
changes.
"""
from __future__ import annotations

import logging
from typing import Any

from litellm.integrations.custom_logger import CustomLogger

log = logging.getLogger("litellm-provider-echo")

_PATCHED = "_kchat_provider_echo"


def apply() -> bool:
    """Patch the OpenRouter stream parser once; False when LiteLLM has no such parser."""
    try:
        from litellm.llms.openrouter.chat.transformation import (
            OpenRouterChatCompletionStreamingHandler as Handler,
        )
    except Exception as exc:  # noqa: BLE001 - a moved import must not stop the proxy
        log.warning("provider echo not applied: %s", exc)
        return False
    if getattr(Handler.chunk_parser, _PATCHED, False):
        return True
    original = Handler.chunk_parser

    def chunk_parser(self: Any, chunk: dict) -> Any:
        out = original(self, chunk)
        provider = chunk.get("provider") if isinstance(chunk, dict) else None
        if isinstance(provider, str) and provider:
            try:
                out.provider = provider
            except Exception:  # noqa: BLE001
                pass
        return out

    setattr(chunk_parser, _PATCHED, True)
    Handler.chunk_parser = chunk_parser
    return True


class ProviderEcho(CustomLogger):
    """Applies the patch when the callback is constructed."""

    def __init__(self) -> None:
        super().__init__()
        apply()


echo_instance = ProviderEcho()
