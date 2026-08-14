"""Core data types.

Memory accounting: ``need = weights + activation + KV(ctx)``. Byte counts are
ints throughout, so capacity comparisons are reproducible.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Optional

GB: int = 1024 ** 3


class Dtype(str, Enum):
    """Weight and KV dtype. BF16/FP16 2 B/elem, FP8 1 B/elem, NVFP4 0.5 B/elem."""

    BF16 = "bf16"
    FP16 = "fp16"
    FP8 = "fp8"
    NVFP4 = "nvfp4"

    @property
    def bytes_per_elem(self) -> float:
        return {"bf16": 2, "fp16": 2, "fp8": 1, "nvfp4": 0.5}[self.value]


@dataclass(frozen=True)
class ModelMetadata:
    """Architectural facts read from the checkpoint's config.json."""

    model_id: str
    #: KV-bearing (full-attention) layers, not the total — hybrid models carry
    #: KV on only some.
    n_layers: int
    #: KV head count under GQA, distinct from the attention head count
    n_kv_heads: int
    head_dim: int
    weight_dtype: Dtype
    weight_bytes: int
    #: Multi-head Latent Attention: one compressed latent per layer, replacing
    #: the 2*H*d term.
    kv_latent_dim: Optional[int] = None
    #: max_position_embeddings from config.json — the model's native context
    native_ctx: int = 0


@dataclass(frozen=True)
class ORTwin:
    """OpenRouter route for a model that cannot be hosted locally.

    Registered under the same ``local/<id>`` name, so callers cannot tell which
    side answered.
    """

    slug: str
    in_price_pm: float   # USD / 1M input tokens
    out_price_pm: float  # USD / 1M output tokens


@dataclass(frozen=True)
class NodeSpec:
    """Capacity of a probed GPU node."""

    node_id: str                  # short identifier, usually the last IPv4 octet
    hostname: str                 # host reachable over SSH
    gpu_class: str                # "gb10", "pro5000", ...
    total_vram_bytes: int         # physical capacity of a single GPU
    #: OS and page-cache headroom, used when usable_vram_bytes is unset
    reserved_bytes: int = 8 * GB
    #: Explicit planner ceiling, below physical capacity on unified-memory nodes
    usable_vram_bytes: Optional[int] = None
    gpu_count: int = 1
    #: "amd64" | "arm64" | "" on probe failure. Gates what the node can serve.
    arch: str = ""

    @property
    def planner_vram_bytes(self) -> int:
        """Packing capacity. An explicit ceiling wins."""
        if self.usable_vram_bytes is not None:
            return self.usable_vram_bytes
        return self.gpu_count * self.total_vram_bytes - self.reserved_bytes
