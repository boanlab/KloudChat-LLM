#!/usr/bin/env bash
# Usage: download-vllm-models.sh [alias|all|recommended|whisper] [...]
#
# Downloads the weights a GPU node will serve, after checking the card: FP4
# capability and usable memory. Weights this node could not serve are skipped
# with the reason rather than discovered at engine startup.
#
# With no arguments: the recommended set for this card (same as `recommended`).
#
# Aliases (lib.sh::VLLM_MODELS):
#   qwen3.6-35b-nvfp4  unsloth/Qwen3.6-35B-A3B-NVFP4          21 GB (chat: vision + 262K + coding)
#   qwen3.6-35b        Qwen/Qwen3.6-35B-A3B                   35 GB (fp8 — older engines)
#   glm-4.7-flash      unsloth/GLM-4.7-Flash-NVFP4            20 GB (cheap-decode floor, A3B)
#
# Special:
#   recommended       what this card can serve and fit together (same as no args)
#   all               every alias above, minus anything unservable here
#   whisper           prewarm transcription weights — the node's resident backend (amd64 only)
#
# Env:
#   HF_TOKEN          auto-loaded from .env (for gated models)
#   VLLM_MODELS_ROOT  storage path (default /var/lib/vllm/models)
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/lib.sh"

# What this card can execute, limited to what fits on it together. Derived from
# measured compute capability and usable memory rather than a GPU-class table, so
# an unlisted card still gets a verdict.
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

# One line of card specifications, so the chosen set is self-explanatory.
describe_gpu() {
  has_nvidia_gpu || { echo "no GPU"; return; }
  local cap; cap="$(gpu_compute_cap)"
  echo "$(get_gpu_name) · ${cap:+compute ${cap} · }$(gpu_usable_vram_gb)GiB usable"
}

# Header comment only, so --help stays usage rather than design notes.
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

# Filter by servability, printing the reason. Nothing is dropped silently.
WANT_WHISPER=0
add_target() {
  local alias="$1" reason
  if reason="$(vllm_model_unservable_reason "$alias")"; then
    TARGETS+=("$alias")
  else
    warn "skip ${alias} — ${reason}"
  fi
}

# No arguments: the recommended set for this card, plus the resident transcription backend.
if [[ $# -eq 0 ]]; then
  set -- recommended whisper
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
      # The whole catalogue, including both quantisations of the chat model.
      # add_target drops whatever this node cannot serve.
      for a in "${!VLLM_MODELS[@]}"; do add_target "$a"; done ;;
    whisper)
      WANT_WHISPER=1 ;;
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
  # hf_xet: Xet-backed repos refuse plain HTTP downloads ("file too large").
  # Legacy repos still use HTTP with it installed.
  HF_TOKEN="$HF_TOKEN" \
  HF_HUB_DOWNLOAD_TIMEOUT=180 \
  uv tool run --from "huggingface_hub[hf_xet]" hf download \
    "$repo" --local-dir "$dest" --max-workers 4
  ok "received $(du -sh "$dest" | cut -f1)"
}

if (( ${#TARGETS[@]} == 0 )) && (( WANT_WHISPER == 0 )); then
  err "nothing to download — this card cannot serve the requested models ($(describe_gpu))"
  exit 1
fi

for a in "${TARGETS[@]}"; do pull_one "$a"; done

# Transcription weights are a Hugging Face cache, fetched from inside the
# container so the download_root mapping matches app.py. Non-fatal: the first
# transcription call fetches them anyway.
pull_whisper() {
  local compose_file="${SCRIPT_DIR%/scripts}/docker-compose.vllm.yml"
  if [[ "$(detect_arch)" != amd64 ]]; then
    info "arm64 node — no transcription backend here (STT goes to OpenRouter); skipping"
    return 0
  fi
  docker compose -f "$compose_file" ps -q whisper 2>/dev/null | grep -q . || {
    warn "the transcription container is not running — run ./scripts/install-vllm.sh first"
    return 1
  }
  local model; model="$(env_get WHISPER_MODEL)"; model="${model:-large-v3}"
  info "prewarming ${model} (in the container)"
  # device=cpu with int8: download and verify only, no GPU.
  docker compose -f "$compose_file" exec -T whisper python3 - "$model" <<'PY'
import sys
from faster_whisper import WhisperModel
WhisperModel(sys.argv[1], device="cpu", compute_type="int8", download_root="/var/lib/whisper")
PY
}

if (( WANT_WHISPER )); then
  hdr "Transcription weights"
  pull_whisper || warn "transcription prewarm failed — the first call will fetch them"
fi

hdr "done"
du -sh "$VLLM_MODELS_ROOT"/* 2>/dev/null | sort -k2 || echo "  (no models yet)"
