"""Core data types. Memory model: ``need = weights + activation + KV(ctx)``, in bytes."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Optional

GB: int = 1024 ** 3


class Dtype(str, Enum):
    """Weight and KV dtype."""

    BF16 = "bf16"
    FP16 = "fp16"
    FP8 = "fp8"
    NVFP4 = "nvfp4"

    @property
    def bytes_per_elem(self) -> float:
        return {"bf16": 2, "fp16": 2, "fp8": 1, "nvfp4": 0.5}[self.value]


@dataclass(frozen=True)
class ModelMetadata:
    """Architecture facts from the checkpoint's config.json."""

    model_id: str
    #: KV-bearing (full-attention) layers only; hybrid models carry KV on a subset
    n_layers: int
    #: KV heads (GQA), not attention heads
    n_kv_heads: int
    head_dim: int
    weight_dtype: Dtype
    weight_bytes: int
    #: MLA: one compressed latent per layer in place of 2*H*d
    kv_latent_dim: Optional[int] = None
    #: max_position_embeddings — native context
    native_ctx: int = 0
    #: Sliding-window layers: KV bounded at ``window`` tokens per sequence, so
    #: charged per sequence rather than per token
    sliding_layers: int = 0
    sliding_window: int = 0


#: Discrete-card headroom (driver context, fragmentation, display): a fraction
#: of the card, clamped, so small and large cards are charged proportionally.
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
    hostname: str                 # SSH target
    gpu_class: str                # "gb10", "pro5000", ...
    total_vram_bytes: int         # one GPU
    #: Explicit headroom; None derives it from the card size
    reserved_bytes: Optional[int] = None
    #: Planner ceiling below physical capacity (unified-memory nodes)
    usable_vram_bytes: Optional[int] = None
    gpu_count: int = 1
    #: GPU memory held by processes outside this stack, subtracted from capacity
    foreign_vram_bytes: int = 0
    #: "amd64" | "arm64" | "" on probe failure
    arch: str = ""
    #: Checkpoint directories under VLLM_MODELS_ROOT. None: not probed, no
    #: filtering. A model placed without its weights restarts forever (Docker
    #: creates a missing bind-mount path empty).
    checkpoints: Optional[frozenset[str]] = None

    @property
    def effective_reserve_bytes(self) -> int:
        if self.reserved_bytes is not None:
            return self.reserved_bytes
        return default_reserve_bytes(self.total_vram_bytes)

    @property
    def planner_vram_bytes(self) -> int:
        """Packing capacity across all cards; an explicit ceiling wins."""
        if self.usable_vram_bytes is not None:
            base = self.usable_vram_bytes
        else:
            base = self.gpu_count * self.total_vram_bytes - self.effective_reserve_bytes
        return max(0, base - self.foreign_vram_bytes)

    @property
    def per_gpu_planner_bytes(self) -> int:
        """Packing capacity of one card — what a tensor-parallel rank must fit."""
        return self.planner_vram_bytes // max(1, self.gpu_count)
