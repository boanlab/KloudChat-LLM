#!/usr/bin/env bash
# Usage: download-vllm-models.sh [alias|all|recommended] [...]
#
# Downloads weights after checking the card (compute capability, usable memory).
# Weights this node cannot serve are skipped with the reason.
#
# No arguments: `recommended whisper-large-v3`.
#
# Aliases (lib.sh::VLLM_MODELS):
#   qwen3.6-35b-nvfp4   unsloth/Qwen3.6-35B-A3B-NVFP4          21 GB  chat (default)
#   qwen3.6-35b         Qwen/Qwen3.6-35B-A3B                   35 GB  chat, FP8
#   qwen3.6-35b-awq     QuantTrio/Qwen3.6-35B-A3B-AWQ          26 GB  chat, int4 (cards without FP4)
#   qwen3.5-122b-a10b   Qwen/Qwen3.5-122B-A10B-NVFP4           78 GB  top chat
#   qwen3-coder-next    Qwen/Qwen3-Coder-Next-FP8              75 GB  coding
#   qwen3-coder-30b     Qwen/Qwen3-Coder-30B-A3B-Instruct-FP8  33 GB  coding
#   qwen3.6-27b         Qwen/Qwen3.6-27B                       21 GB  dense chat
#   bge-m3 / bge-reranker-v2-m3                                 3 GB  retrieval
#   whisper-large-v3    openai/whisper-large-v3                 4 GB  transcription
#
# Special:
#   recommended       what this card can serve and fit together
#   all               every alias, minus anything unservable here
#
# Env:
#   HF_TOKEN          auto-loaded from .env (for gated models)
#   VLLM_MODELS_ROOT  storage path (default /var/lib/vllm/models)
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/lib.sh"

# VLLM_PREFERRED_MODELS filtered to what this card can execute and fit together.
recommended_vllm_set() {
  local alias weight used=0 budget out=()
  budget="$(gpu_usable_vram_gb)"
  for alias in "${VLLM_PREFERRED_MODELS[@]}"; do
    vllm_model_unservable_reason "$alias" >/dev/null || continue
    weight="${VLLM_MODEL_WEIGHT_GB[$alias]:-0}"
    (( budget > 0 && used + weight + VLLM_RUNTIME_HEADROOM_GB > budget )) && continue
    out+=("$alias"); used=$(( used + weight ))
  done
  echo "${out[*]:-}"
}

describe_gpu() {
  has_nvidia_gpu || { echo "no GPU"; return; }
  local cap; cap="$(gpu_compute_cap)"
  echo "$(get_gpu_name) · ${cap:+compute ${cap} · }$(gpu_usable_vram_gb)GiB usable"
}

if [[ "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then
  sed -n '2,/^[^#]/p' "$0" | sed -n 's/^# \{0,1\}//p'
  exit 0
fi

require_supported_platform
command -v uv &>/dev/null || { err "uv not found. curl -LsSf https://astral.sh/uv/install.sh | sh"; exit 1; }

HF_TOKEN="$(env_get HF_TOKEN)"
[[ -n "$HF_TOKEN" ]] || warn "HF_TOKEN not set — gated model downloads will fail."

mkdir -p "$VLLM_MODELS_ROOT"

hdr "GPU: $(describe_gpu)"

# Servability filter; skips print the reason.
add_target() {
  local alias="$1" reason
  if reason="$(vllm_model_unservable_reason "$alias")"; then
    TARGETS+=("$alias")
  else
    warn "skip ${alias} — ${reason}"
  fi
}

if [[ $# -eq 0 ]]; then
  set -- recommended whisper-large-v3
fi

TARGETS=()
while [[ $# -gt 0 ]]; do
  case "$1" in
    recommended)
      r="$(recommended_vllm_set)"
      [[ -n "$r" ]] || { err "this card cannot serve any model in the lineup ($(describe_gpu)) — name an alias explicitly, or serve through OpenRouter"; exit 1; }
      info "recommended: $r"
      for a in $r; do TARGETS+=("$a"); done
      ;;
    all)
      for a in "${!VLLM_MODELS[@]}"; do add_target "$a"; done ;;
    *)
      [[ -n "${VLLM_MODELS[$1]:-}" ]] || { err "Unknown alias: $1"; exit 1; }
      add_target "$1"
      ;;
  esac
  shift
done

pull_one() {
  local alias="$1" repo="${VLLM_MODELS[$1]}" dest="$VLLM_MODELS_ROOT/$1"
  hdr "$alias → $repo"
  echo "  dest: $dest"
  if [[ -f "$dest/config.json" ]] && compgen -G "$dest/*.safetensors" >/dev/null; then
    ok "already downloaded ($(du -sh "$dest" | cut -f1)) — to re-download, delete the directory and re-run"
    return 0
  fi
  # hf_xet: Xet-backed repos refuse plain HTTP. One --exclude per pattern (a
  # list is read as positional filters). Alternative serialisations are
  # excluded: vLLM loads only the safetensors, and extra files inflate what the
  # scheduler measures the model to weigh.
  HF_TOKEN="$HF_TOKEN" \
  HF_HUB_DOWNLOAD_TIMEOUT=180 \
  uv tool run --from "huggingface_hub[hf_xet]" hf download \
    "$repo" --local-dir "$dest" --max-workers 4 \
    --exclude "*.msgpack" --exclude "*.h5" --exclude "*.ot" --exclude "*.tflite" \
    --exclude "*fp32*" --exclude "*.pth" --exclude "*.gguf" --exclude "ggml*" \
    --exclude "coreml/*" --exclude "openvino/*" --exclude "onnx/*"
  # .bin only where a repo ships nothing else
  if compgen -G "$dest/*.safetensors" >/dev/null; then
    rm -f "$dest"/*.bin "$dest"/pytorch_model*.bin 2>/dev/null || true
  fi
  local loadable
  loadable="$(du -cb "$dest"/*.safetensors 2>/dev/null | tail -1 | cut -f1 || true)"
  if [[ "${loadable:-0}" =~ ^[0-9]+$ ]] && (( loadable > 0 )); then
    ok "received $(du -sh "$dest" | cut -f1), $(awk -v b="$loadable" 'BEGIN{printf "%.1fGB", b/1024/1024/1024}') of it safetensors"
  else
    ok "received $(du -sh "$dest" | cut -f1)"
  fi
}

if (( ${#TARGETS[@]} == 0 )); then
  err "nothing to download — this card cannot serve the requested models ($(describe_gpu))"
  exit 1
fi

for a in "${TARGETS[@]}"; do pull_one "$a"; done

hdr "done"
du -sh "$VLLM_MODELS_ROOT"/* 2>/dev/null | sort -k2 || echo "  (no models yet)"
