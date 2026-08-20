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
    #: Sliding-window attention layers and their window. These hold KV too, but
    #: only ``min(ctx, window)`` tokens of it, so the cost is per sequence rather
    #: than per token. Counting them as full-attention layers overcharges by
    #: orders of magnitude; counting them as zero undercharges a Gemma-shaped
    #: model, where they are five layers in six.
    sliding_layers: int = 0
    sliding_window: int = 0


@dataclass(frozen=True)
class ORTwin:
    """OpenRouter route for a model that cannot be hosted locally.

    Registered under the same ``local/<id>`` name, so callers cannot tell which
    side answered.
    """

    slug: str
    in_price_pm: float   # USD / 1M input tokens
    out_price_pm: float  # USD / 1M output tokens


#: Headroom held back on a discrete card: driver context, fragmentation, and the
#: gap between "total" and "free" on a card that also drives a display. A
#: fraction, because a fixed 8 GiB is 8% of a 96 GiB card and 33% of a 24 GiB
#: one — the same number meaning two different things is what kept 32 GiB cards
#: out of the cluster.
RESERVE_FRACTION: float = 0.08
RESERVE_MIN_BYTES: int = 1 * GB
RESERVE_MAX_BYTES: int = 8 * GB


def default_reserve_bytes(total_vram_bytes: int) -> int:
    """Headroom for a card of this size."""
    if total_vram_bytes <= 0:
        return 0
    scaled = int(total_vram_bytes * RESERVE_FRACTION)
    return max(RESERVE_MIN_BYTES, min(RESERVE_MAX_BYTES, scaled))


@dataclass(frozen=True)
class NodeSpec:
    """Capacity of a probed GPU node."""

    node_id: str                  # short identifier, usually the last IPv4 octet
    hostname: str                 # host reachable over SSH
    gpu_class: str                # "gb10", "pro5000", ...
    total_vram_bytes: int         # physical capacity of a single GPU
    #: OS and page-cache headroom, used when usable_vram_bytes is unset. None
    #: derives it from the card size — see ``default_reserve_bytes``.
    reserved_bytes: Optional[int] = None
    #: Explicit planner ceiling, below physical capacity on unified-memory nodes
    usable_vram_bytes: Optional[int] = None
    gpu_count: int = 1
    #: GPU memory held by processes this stack does not manage — a desktop
    #: session, somebody's notebook, another deployment. Subtracted from what the
    #: planner may hand out, because vLLM's utilisation fraction is of the card's
    #: total but the memory has to actually be free.
    foreign_vram_bytes: int = 0
    #: "amd64" | "arm64" | "" on probe failure. Gates what the node can serve.
    arch: str = ""

    @property
    def effective_reserve_bytes(self) -> int:
        """Headroom actually held back: the declared figure, or one for this card."""
        if self.reserved_bytes is not None:
            return self.reserved_bytes
        return default_reserve_bytes(self.total_vram_bytes)

    @property
    def planner_vram_bytes(self) -> int:
        """Packing capacity across every card on the node. An explicit ceiling wins."""
        if self.usable_vram_bytes is not None:
            base = self.usable_vram_bytes
        else:
            base = self.gpu_count * self.total_vram_bytes - self.effective_reserve_bytes
        return max(0, base - self.foreign_vram_bytes)

    @property
    def per_gpu_planner_bytes(self) -> int:
        """Packing capacity of one card.

        Tensor parallelism splits a model across cards, so what has to fit is the
        per-card share — a node pool large enough in total says nothing about
        whether one rank's weights, activation and KV shard fit on one card.
        """
        return self.planner_vram_bytes // max(1, self.gpu_count)
