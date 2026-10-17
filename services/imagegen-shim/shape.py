"""Request and reply shapes between KloudChat's chat-style image call and vllm-omni.

No third-party imports, so CI tests it without the image's dependencies.
"""
from __future__ import annotations

import os
import time
import uuid
from typing import Any

#: WIDTHxHEIGHT per aspect. Qwen-Image's are the ones its card lists; the rest
#: are about one megapixel in multiples of 16.
_QWEN_SIZES = {
    "1:1": "1328x1328", "16:9": "1664x928", "9:16": "928x1664", "4:3": "1472x1104",
    "3:4": "1104x1472", "3:2": "1584x1056", "2:3": "1056x1584",
}
_MEGAPIXEL_SIZES = {
    "1:1": "1024x1024", "16:9": "1360x768", "9:16": "768x1360", "4:3": "1152x864",
    "3:4": "864x1152", "3:2": "1248x832", "2:3": "832x1248", "4:5": "896x1120",
    "5:4": "1120x896", "21:9": "1568x672",
}

#: Per family: sizes and the sampling its card recommends.
FAMILIES: dict[str, dict[str, Any]] = {
    "qwen-image": {
        "sizes": _QWEN_SIZES,
        "params": {"num_inference_steps": int(os.getenv("QWEN_IMAGE_STEPS", "30")),
                   "true_cfg_scale": 4.0, "negative_prompt": " "},
    },
    "z-image": {
        "sizes": _MEGAPIXEL_SIZES,
        "params": {"num_inference_steps": 9, "guidance_scale": 0.0},
    },
    "flux2-klein": {
        "sizes": _MEGAPIXEL_SIZES,
        "params": {"num_inference_steps": 4, "guidance_scale": 1.0},
    },
}


def parse_backends(raw: str) -> dict[str, list[str]]:
    """`{served name: [base URL, ...]}` from `name=url|url,name=url`.

    A name given twice adds its URLs; duplicates are dropped.
    """
    out: dict[str, list[str]] = {}
    for item in raw.split(","):
        name, sep, urls = item.strip().partition("=")
        name = name.strip()
        if not (sep and name):
            continue
        for url in urls.split("|"):
            url = url.strip().rstrip("/")
            if url and url not in out.setdefault(name, []):
                out[name].append(url)
        if not out.get(name):
            out.pop(name, None)
    return out


def family_of(model: str) -> dict[str, Any]:
    """The family whose key appears in the served name; Z-Image's when none does."""
    name = model.lower()
    for key, family in FAMILIES.items():
        if key in name:
            return family
    return FAMILIES["z-image"]


def size_for(model: str, aspect: str) -> str:
    sizes = family_of(model)["sizes"]
    return sizes.get(aspect) or sizes["1:1"]


def prompt_of(messages: list[dict[str, Any]]) -> str:
    """Text of the last user message, its text parts joined."""
    for message in reversed(messages or []):
        if message.get("role") != "user":
            continue
        content = message.get("content")
        if isinstance(content, str):
            return content.strip()
        if isinstance(content, list):
            parts = [p.get("text", "") for p in content if isinstance(p, dict) and p.get("type") == "text"]
            return "\n".join(t for t in parts if t).strip()
    return ""


def generation_body(model: str, body: dict[str, Any]) -> dict[str, Any]:
    """vllm-omni `/v1/images/generations` request for a chat-shaped one."""
    aspect = str((body.get("image_config") or {}).get("aspect_ratio") or "")
    out: dict[str, Any] = {
        "prompt": prompt_of(body.get("messages") or []),
        "n": 1,
        "size": size_for(model, aspect),
        "response_format": "b64_json",
        **family_of(model)["params"],
    }
    if isinstance(body.get("seed"), int):
        out["seed"] = body["seed"]
    return out


def chat_reply(model: str, b64: str) -> dict[str, Any]:
    """OpenRouter-shaped completion carrying one PNG."""
    return {
        "id": f"chatcmpl-{uuid.uuid4().hex}",
        "object": "chat.completion",
        "created": int(time.time()),
        "model": model,
        "choices": [{
            "index": 0,
            "finish_reason": "stop",
            "message": {
                "role": "assistant",
                "content": "",
                "images": [{"type": "image_url", "image_url": {"url": f"data:image/png;base64,{b64}"}}],
            },
        }],
        "usage": {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0},
    }
