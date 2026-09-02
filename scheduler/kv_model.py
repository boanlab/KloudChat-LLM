"""KV cache sizing.

    bytes/token = 2 * L_kv * H * d * beta_kv        (MHA/GQA)
                = L_kv * latent * beta_kv           (MLA)

L_kv: full-attention layers, H: KV heads, d: head dim, beta_kv: KV dtype size.
Sliding-window layers are charged per sequence instead (``sliding_bytes_per_sequence``).
"""

from __future__ import annotations

from scheduler.types import Dtype, ModelMetadata

#: Headroom for output tokens and scheduler overhead
ADMISSION_MARGIN: float = 1.10


def sliding_bytes_per_sequence(model: ModelMetadata, kv_dtype: Dtype) -> int:
    """KV held by the sliding-window layers for one sequence, at any length."""
    if not model.sliding_layers or not model.sliding_window:
        return 0
    beta = kv_dtype.bytes_per_elem
    per_token_per_layer = 2 * model.n_kv_heads * model.head_dim * beta
    return int(round(model.sliding_layers * per_token_per_layer * model.sliding_window))


def kv_bytes_per_token(model: ModelMetadata, kv_dtype: Dtype) -> int:
    """KV bytes per token across the full-attention layers."""
    beta = kv_dtype.bytes_per_elem
    if model.kv_latent_dim:
        return int(round(model.n_layers * model.kv_latent_dim * beta))
    return int(round(2 * model.n_layers * model.n_kv_heads * model.head_dim * beta))
