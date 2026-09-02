"""Architecture metadata from a checkpoint's config.json, parsed without transformers.

Lookup order:
    1. Local cache:  ~/.cache/kloudchat-scheduler/models/<id>.json
    2. Node probe:   ssh <host> cat <models_root>/<dir>/config.json
No online lookup.
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
    """``ssh <host> cat <remote_path>`` as JSON; None on any error."""
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
    """Promote a nested ``text_config`` (multimodal checkpoints) to the top level; outer keys win."""
    inner = cfg.get("text_config")
    if not isinstance(inner, dict):
        return cfg
    merged = dict(cfg)
    for k, v in inner.items():
        merged.setdefault(k, v)
    # text_config's dtype is the LLM dtype
    if "torch_dtype" not in merged and "dtype" in inner:
        merged["torch_dtype"] = inner["dtype"]
    return merged


def _parse_dtype(cfg: dict) -> Dtype:
    """Dtype from quantization_config, falling back to torch_dtype."""
    qcfg = cfg.get("quantization_config") or {}
    method = (qcfg.get("quant_method") or "").lower()
    fmt = (qcfg.get("fmt") or "").lower()
    if "fp8" in method or "fp8" in fmt:
        return Dtype.FP8
    if "nvfp4" in method or "nvfp4" in fmt:
        return Dtype.NVFP4

    # compressed-tensors / modelopt: width in config_groups.*.weights.num_bits.
    # Smallest width wins — the bulk drives the footprint.
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
    """Layers whose KV grows with sequence length.

    Hybrid models declare them as a ``layer_types`` list or a
    ``full_attention_interval`` stride (only honoured alongside linear-attention
    keys); every other model is all layers. Encoder-decoders count the decoder.
    """
    layer_types = cfg.get("layer_types")
    if isinstance(layer_types, list) and layer_types:
        full = sum(1 for t in layer_types if t == "full_attention")
        if full > 0:
            return full

    total = int(cfg.get("num_hidden_layers") or 0)
    interval = int(cfg.get("full_attention_interval") or 0)
    if total and interval > 1 and cfg.get("linear_key_head_dim"):
        return max(1, total // interval)
    return total or int(cfg.get("decoder_layers") or 0)


def _count_sliding_layers(cfg: dict) -> tuple[int, int]:
    """(layer count, window) for sliding-window attention, or (0, 0)."""
    layer_types = cfg.get("layer_types")
    window = int(cfg.get("sliding_window") or 0)
    if not window or not isinstance(layer_types, list):
        return 0, 0
    sliding = sum(1 for t in layer_types if isinstance(t, str) and "sliding" in t)
    return (sliding, window) if sliding else (0, 0)


def _resolve_kv_heads(cfg: dict) -> int:
    """KV head count: GQA heads, else attention heads, else the decoder's (whisper)."""
    return int(cfg.get("num_key_value_heads")
               or cfg.get("num_attention_heads")
               or cfg.get("decoder_attention_heads")
               or 1)


def _resolve_kv_latent_dim(cfg: dict):
    """MLA latent width per layer (``kv_lora_rank + qk_rope_head_dim``), or None for MHA/GQA."""
    rank = cfg.get("kv_lora_rank")
    if not rank:
        return None
    return int(rank) + int(cfg.get("qk_rope_head_dim") or 0)


def _resolve_head_dim(cfg: dict) -> int:
    """Head width; ``d_model`` / ``decoder_attention_heads`` are the encoder-decoder spelling."""
    if "head_dim" in cfg:
        return int(cfg["head_dim"])
    hidden = int(cfg.get("hidden_size") or cfg.get("d_model") or 0)
    n_heads = int(cfg.get("num_attention_heads")
                  or cfg.get("decoder_attention_heads") or 1)
    return hidden // n_heads if n_heads else 0


def _estimate_weight_bytes(cfg: dict, dtype: Dtype) -> int:
    """Analytic weight size when the on-disk size is unknown.

    Dominant tensors only (embed + L × per-layer). MoE: all experts counted,
    since vLLM keeps them resident.
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
    """Metadata for ``model_id`` from the cache or a node probe.

    Args:
        probe_host: ssh target on a cache miss.
        probe_path: remote path to config.json.
        on_disk_weight_bytes: measured size, overriding the analytic estimate.

    Raises:
        FileNotFoundError: cache miss and no probe target, or probe failed.
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
    sliding_layers, sliding_window = _count_sliding_layers(cfg)
    return ModelMetadata(
        model_id=model_id,
        n_layers=_count_kv_bearing_layers(cfg),
        n_kv_heads=_resolve_kv_heads(cfg),
        head_dim=_resolve_head_dim(cfg),
        weight_dtype=dtype,
        weight_bytes=weight_bytes,
        kv_latent_dim=_resolve_kv_latent_dim(cfg),
        native_ctx=_resolve_native_ctx(cfg),
        sliding_layers=sliding_layers,
        sliding_window=sliding_window,
    )


def _resolve_native_ctx(cfg: dict) -> int:
    """Native context from config.json; ``max_target_positions`` is the decoder's (whisper: 448)."""
    for key in ("max_position_embeddings", "max_sequence_length", "n_positions",
                "max_target_positions"):
        value = cfg.get(key)
        if isinstance(value, int) and value > 0:
            return value
    return 0


#: Size of the safetensors vLLM loads (excluding fp32 copies), not the whole
#: directory; ``du`` fallback for checkpoints without safetensors.
_WEIGHT_BYTES = r"""
sum=$(find %(path)s -maxdepth 1 -name '*.safetensors' ! -name '*fp32*'         -printf '%%s
' 2>/dev/null | awk '{t+=$1} END {print t+0}')
if [ "${sum:-0}" -gt 0 ]; then echo "$sum"; else du -sb %(path)s 2>/dev/null | cut -f1; fi
"""


def measure_weight_bytes(host: str, path: str) -> Optional[int]:
    """Measured weight size on the node; None on failure."""
    import subprocess
    try:
        r = subprocess.run(
            ["ssh", "-o", "StrictHostKeyChecking=no", "-o", "ConnectTimeout=6",
             "-o", "LogLevel=ERROR", host, _WEIGHT_BYTES % {"path": path}],
            capture_output=True, text=True, timeout=30,
        )
    except (subprocess.TimeoutExpired, OSError):
        return None
    out = (r.stdout or "").strip().splitlines()
    return int(out[-1]) if out and out[-1].isdigit() else None
