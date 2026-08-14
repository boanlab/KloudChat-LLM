"""Resolve transformer architecture metadata for KV-cache sizing.

Source of truth is the model's ``config.json``. We avoid taking a hard
dependency on the ``transformers`` library (too heavy for a scheduling tool),
parsing the file directly instead.

Lookup order:
    1. Local disk cache:  ~/.cache/kloudchat-scheduler/models/<id>.json
    2. Probe a node:      ssh <host> cat <models_root>/<dir>/config.json
    3. (no online lookup — air-gapped clusters need pre-populated cache)

The local cache is content-addressed by ``model_id`` (e.g. ``Qwen/Qwen3.6-35B-A3B``).
"""

from __future__ import annotations

import json
import os
import subprocess
from pathlib import Path
from typing import Optional

from scheduler.types import Dtype, ModelMetadata

_CACHE_DIR = Path(os.path.expanduser("~/.cache/kloudchat-scheduler/models"))


def _cache_path(model_id: str) -> Path:
    safe = model_id.replace("/", "__")
    return _CACHE_DIR / f"{safe}.json"


def _load_local(model_id: str) -> Optional[dict]:
    p = _cache_path(model_id)
    if not p.is_file():
        return None
    try:
        return json.loads(p.read_text())
    except (OSError, json.JSONDecodeError):
        return None


def _store_local(model_id: str, blob: dict) -> None:
    _CACHE_DIR.mkdir(parents=True, exist_ok=True)
    _cache_path(model_id).write_text(json.dumps(blob, indent=2))


def _probe_node(host: str, remote_path: str, timeout: int = 5) -> Optional[dict]:
    """``ssh <host> cat <remote_path>`` — returns parsed JSON or None on any error."""
    try:
        result = subprocess.run(
            ["ssh", "-o", "StrictHostKeyChecking=no",
             "-o", f"ConnectTimeout={timeout}",
             host, "cat", remote_path],
            capture_output=True, text=True, timeout=timeout + 5,
        )
        if result.returncode != 0:
            return None
        return json.loads(result.stdout)
    except (subprocess.TimeoutExpired, json.JSONDecodeError, OSError):
        return None


def _unwrap_text_config(cfg: dict) -> dict:
    """Some multimodal Qwen variants nest the LLM config under ``text_config``.

    We promote it to the top level so the rest of the parser stays uniform.
    Quantization metadata typically remains at the outer level, so we preserve
    it by merging (outer takes precedence except for keys we explicitly want
    from text_config).
    """
    inner = cfg.get("text_config")
    if not isinstance(inner, dict):
        return cfg
    merged = dict(cfg)
    for k, v in inner.items():
        merged.setdefault(k, v)
    # text_config's dtype is the LLM dtype, prefer it if outer didn't set one
    if "torch_dtype" not in merged and "dtype" in inner:
        merged["torch_dtype"] = inner["dtype"]
    return merged


def _parse_dtype(cfg: dict) -> Dtype:
    """Map config.json's torch_dtype + quantization_config to our Dtype enum.

    Quantized checkpoints (FP8, NVFP4) override torch_dtype, which typically
    still reads ``bfloat16`` for the unquantized linear layers.
    """
    qcfg = cfg.get("quantization_config") or {}
    method = (qcfg.get("quant_method") or "").lower()
    fmt = (qcfg.get("fmt") or "").lower()
    if "fp8" in method or "fp8" in fmt:
        return Dtype.FP8
    if "nvfp4" in method or "nvfp4" in fmt:
        return Dtype.NVFP4

    # compressed-tensors / modelopt configs name the width in
    # config_groups.*.weights.num_bits rather than quant_method. Without it the
    # fall-through reads torch_dtype, still "bfloat16" on a quantised checkpoint,
    # and over-estimates weights fourfold. Smallest width wins: mixed checkpoints
    # keep a few layers wide, but the bulk drives the footprint.
    widths = {
        int(g["weights"]["num_bits"])
        for g in (qcfg.get("config_groups") or {}).values()
        if isinstance(g, dict) and isinstance(g.get("weights"), dict)
        and str(g["weights"].get("num_bits", "")).isdigit()
    }
    if widths:
        return {4: Dtype.NVFP4, 8: Dtype.FP8}.get(min(widths), Dtype.BF16)

    tdtype = (cfg.get("torch_dtype") or cfg.get("dtype") or "").lower()
    if tdtype in ("bfloat16", "bf16"):
        return Dtype.BF16
    if tdtype in ("float16", "fp16"):
        return Dtype.FP16
    return Dtype.BF16


def _count_kv_bearing_layers(cfg: dict) -> int:
    """Number of layers that scale KV cache linearly with sequence length.

    Pure-MHA / GQA transformers: every layer is KV-bearing.
    Hybrid linear/full attention (Qwen3.6-35B-A3B, Jamba, Mamba-Hybrid): only the
    ``full_attention`` layers grow KV with sequence length; linear-attention
    layers keep a fixed-size state that we account for separately (and which
    is negligible at the scales we plan for).
    """
    layer_types = cfg.get("layer_types")
    if isinstance(layer_types, list) and layer_types:
        full = sum(1 for t in layer_types if t == "full_attention")
        if full > 0:
            return full
    return int(cfg.get("num_hidden_layers") or 0)


def _resolve_kv_heads(cfg: dict) -> int:
    """GQA-aware KV head count.

    For grouped-query attention, ``num_key_value_heads`` is the number we
    actually pay for in KV cache. For multi-head attention, it equals
    ``num_attention_heads``.
    """
    return int(cfg.get("num_key_value_heads")
               or cfg.get("num_attention_heads")
               or 1)


def _resolve_kv_latent_dim(cfg: dict):
    """MLA latent width per layer, or None for ordinary MHA/GQA models.

    DeepSeek-style Multi-head Latent Attention compresses K and V into a single
    ``kv_lora_rank`` vector and keeps the rotary slice (``qk_rope_head_dim``)
    uncompressed alongside it. Both are cached once per layer, so the per-token
    cost is ``L · (kv_lora_rank + qk_rope_head_dim)`` — no head fan-out, no factor
    of two. Absence of ``kv_lora_rank`` means the model is not MLA.
    """
    rank = cfg.get("kv_lora_rank")
    if not rank:
        return None
    return int(rank) + int(cfg.get("qk_rope_head_dim") or 0)


def _resolve_head_dim(cfg: dict) -> int:
    if "head_dim" in cfg:
        return int(cfg["head_dim"])
    hidden = int(cfg.get("hidden_size") or 0)
    n_heads = int(cfg.get("num_attention_heads") or 1)
    return hidden // n_heads if n_heads else 0


def _estimate_weight_bytes(cfg: dict, dtype: Dtype) -> int:
    """Rough analytic weight size (used when on-disk size is unknown).

    Counts only the dominant tensors (embed + L × per-layer) — overhead from
    LM head, layernorms, biases is ≤ 1% for the models in our catalog.

    For MoE checkpoints (Qwen3.6-35B-A3B, GLM-4.7-Flash), ``num_experts`` × ``moe_intermediate_size``
    replaces the dense FFN term; the planner does *not* assume any sparse-routing
    activation discount (vLLM keeps all experts resident).
    """
    L = int(cfg.get("num_hidden_layers") or 0)
    H = int(cfg.get("hidden_size") or 0)
    Iff = int(cfg.get("intermediate_size") or 4 * H)
    n_experts = int(cfg.get("num_experts") or cfg.get("num_local_experts") or 1)
    moe_iff = int(cfg.get("moe_intermediate_size") or Iff)
    vocab = int(cfg.get("vocab_size") or 0)
    attn_proj = 4 * H * H                    # Q/K/V/O
    ffn_dense = 3 * H * Iff                  # gate/up/down (SwiGLU)
    ffn_moe = 3 * H * moe_iff * n_experts if n_experts > 1 else 0
    per_layer = attn_proj + (ffn_moe or ffn_dense)
    total_params = L * per_layer + 2 * H * vocab
    return int(total_params * dtype.bytes_per_elem)


def fetch(
    model_id: str,
    *,
    probe_host: Optional[str] = None,
    probe_path: Optional[str] = None,
    on_disk_weight_bytes: Optional[int] = None,
) -> ModelMetadata:
    """Resolve metadata for ``model_id``.

    Args:
        model_id: HuggingFace-style id (e.g. ``Qwen/Qwen3.6-35B-A3B``).
        probe_host: ssh target if local cache misses (optional).
        probe_path: remote path to config.json (required if probing).
        on_disk_weight_bytes: actual checkpoint size if known (overrides the
            analytic estimate, which is approximate).

    Raises:
        FileNotFoundError: cache miss and no probe target / probe failed.
    """
    cfg = _load_local(model_id)
    if cfg is None and probe_host and probe_path:
        cfg = _probe_node(probe_host, probe_path)
        if cfg is not None:
            _store_local(model_id, cfg)
    if cfg is None:
        raise FileNotFoundError(
            f"no metadata for {model_id!r}: cache miss and no probe target "
            f"(populate {_cache_path(model_id)} or pass probe_host/probe_path)"
        )

    cfg = _unwrap_text_config(cfg)
    dtype = _parse_dtype(cfg)
    weight_bytes = on_disk_weight_bytes or _estimate_weight_bytes(cfg, dtype)
    return ModelMetadata(
        model_id=model_id,
        n_layers=_count_kv_bearing_layers(cfg),
        n_kv_heads=_resolve_kv_heads(cfg),
        head_dim=_resolve_head_dim(cfg),
        weight_dtype=dtype,
        weight_bytes=weight_bytes,
        kv_latent_dim=_resolve_kv_latent_dim(cfg),
        native_ctx=_resolve_native_ctx(cfg),
    )


def _resolve_native_ctx(cfg: dict) -> int:
    """The maximum position declared by config.json — the native context."""
    for key in ("max_position_embeddings", "max_sequence_length", "n_positions"):
        value = cfg.get(key)
        if isinstance(value, int) and value > 0:
            return value
    return 0


def measure_weight_bytes(host: str, path: str) -> Optional[int]:
    """Measured size of the checkpoint directory; None on failure, which falls
    back to an analytical estimate."""
    import subprocess
    try:
        r = subprocess.run(
            ["ssh", "-o", "StrictHostKeyChecking=no", "-o", "ConnectTimeout=6",
             "-o", "LogLevel=ERROR", host, f"du -sb {path} 2>/dev/null | cut -f1"],
            capture_output=True, text=True, timeout=30,
        )
    except (subprocess.TimeoutExpired, OSError):
        return None
    out = (r.stdout or "").strip()
    return int(out) if out.isdigit() else None
