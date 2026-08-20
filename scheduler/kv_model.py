"""KV cache sizing.

Per-token cost:

    M_KV/token = 2 * L_kv * H * d * beta_kv     (bytes)

L_kv is the full-attention layer count, H the KV head count, d the head
dimension, beta_kv the KV dtype size.

Sliding-window layers are not in that term. They keep a bounded window, so their
cost is flat per sequence — see ``sliding_bytes_per_sequence``.
"""

from __future__ import annotations

from scheduler.types import Dtype, ModelMetadata

#: Headroom for output tokens and scheduler overhead. Below 1.0 the planner
#: over-commits.
ADMISSION_MARGIN: float = 1.10


def sliding_bytes_per_sequence(model: ModelMetadata, kv_dtype: Dtype) -> int:
    """Bytes the sliding-window layers hold for one sequence, at any length.

    A sliding-window layer keeps its last ``window`` tokens and no more, so its
    cost stops growing with the sequence. Flat per sequence, not per token — the
    opposite shape from the full-attention term, which is why it is a separate
    figure rather than a correction to that one.

    Counting these layers as full attention overcharges by orders of magnitude;
    counting them as nothing undercharges a Gemma-shaped model, where they are
    five layers in six.
    """
    if not model.sliding_layers or not model.sliding_window:
        return 0
    beta = kv_dtype.bytes_per_elem
    per_token_per_layer = 2 * model.n_kv_heads * model.head_dim * beta
    return int(round(model.sliding_layers * per_token_per_layer * model.sliding_window))


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
