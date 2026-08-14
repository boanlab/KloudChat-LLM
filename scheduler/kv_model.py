"""KV cache sizing.

Per-token cost:

    M_KV/token = 2 * L_kv * H * d * beta_kv     (bytes)

L_kv is the full-attention layer count, H the KV head count, d the head
dimension, beta_kv the KV dtype size.
"""

from __future__ import annotations

from scheduler.types import Dtype, ModelMetadata

#: Headroom for output tokens and scheduler overhead. Below 1.0 the planner
#: over-commits.
ADMISSION_MARGIN: float = 1.10


def kv_bytes_per_token(model: ModelMetadata, kv_dtype: Dtype) -> int:
    """Bytes one token costs across every KV-bearing layer.

    MHA and GQA store K and V separately, hence the factor of two. MLA caches one
    compressed latent per layer, dropping both the head fan-out and that factor;
    using the MHA form for an MLA model overestimates by an order of magnitude
    and rejects placements that fit.
    """
    beta = kv_dtype.bytes_per_elem
    if model.kv_latent_dim:
        return int(round(model.n_layers * model.kv_latent_dim * beta))
    return int(round(2 * model.n_layers * model.n_kv_heads * model.head_dim * beta))
