#!/usr/bin/env bash
# Usage:
#   manage-vllm.sh up [--recreate] [svc...]     verify weights + compose up
#                                               (omit svc = every service with local weights)
#   manage-vllm.sh down [-v]                    stop + remove containers
#   manage-vllm.sh restart [svc]                restart
#   manage-vllm.sh logs [svc]                   follow logs
#   manage-vllm.sh status                       container + healthcheck status
#   manage-vllm.sh pull                         update image
#
# Services:
#   vllm-qwen35b     Qwen3.6-35B-A3B — chat, vision, coding
#   vllm-qwen122b    Qwen3.5-122B-A10B — top chat, 78 GiB, a card to itself
#   vllm-codernext   Qwen3-Coder-Next-80B — coding, 75 GiB, a card to itself
#   vllm-coder30b    Qwen3-Coder-30B-A3B — coding
#   vllm-qwen27b     Qwen3.6-27B — dense chat
#   vllm-bgem3       BAAI/bge-m3 — retrieval embeddings
#   vllm-rerank      BAAI/bge-reranker-v2-m3 — retrieval reranking
#   vllm-whisper     openai/whisper-large-v3 — transcription
# Placement across nodes is the scheduler's (python3 -m scheduler apply); this
# script is the per-node manual control. Compose project: kloudchat-vllm.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
COMPOSE_FILE="${PROJECT_DIR}/docker-compose.vllm.yml"
source "${SCRIPT_DIR}/lib.sh"

[[ -f "$COMPOSE_FILE" ]] || { err "$COMPOSE_FILE not found"; exit 1; }

# Every service in docker-compose.vllm.yml, in `up` and `status` order.
VLLM_SERVICES=(vllm-qwen35b vllm-qwen122b vllm-codernext vllm-coder30b vllm-qwen27b
               vllm-bgem3 vllm-rerank vllm-whisper)

usage() {
  sed -n '2,/^[^#]/p' "$0" | sed -n 's/^# \{0,1\}//p'
  exit "${1:-1}"
}

cmd_up() {
  local recreate=0 want=()
  while [[ $# -gt 0 ]]; do
    case "$1" in
      --recreate) recreate=1; shift ;;
      vllm-*)     want+=("$1"); shift ;;
      *) err "Unknown: $1"; usage ;;
    esac
  done

  # No service named: every service with weights, which can sum gpu_util past
  # 1.0 on a shared card.
  if (( ${#want[@]} == 0 )); then
    warn "no service named — starting every service with local weights."
    warn "  the scheduler decides placement: python -m scheduler apply"
    warn "  to drive this node by hand: manage-vllm.sh up vllm-qwen35b [vllm-qwen122b ...]"
  fi

  local root="${VLLM_MODELS_ROOT:-/var/lib/vllm/models}"

  # Weight directory: VLLM_<MODEL>_DIR from .env, else the compose default.
  local d
  declare -A svc_dir
  d="$(env_get VLLM_QWEN35B_DIR)";   svc_dir[vllm-qwen35b]="${d:-qwen3.6-35b-nvfp4}"
  d="$(env_get VLLM_QWEN122B_DIR)";  svc_dir[vllm-qwen122b]="${d:-qwen3.5-122b-a10b}"
  d="$(env_get VLLM_CODERNEXT_DIR)"; svc_dir[vllm-codernext]="${d:-qwen3-coder-next}"
  d="$(env_get VLLM_CODER30B_DIR)";  svc_dir[vllm-coder30b]="${d:-qwen3-coder-30b}"
  d="$(env_get VLLM_QWEN27B_DIR)";   svc_dir[vllm-qwen27b]="${d:-qwen3.6-27b}"
  d="$(env_get VLLM_BGEM3_DIR)";     svc_dir[vllm-bgem3]="${d:-bge-m3}"
  d="$(env_get VLLM_RERANK_DIR)";    svc_dir[vllm-rerank]="${d:-bge-reranker-v2-m3}"
  d="$(env_get VLLM_WHISPER_DIR)";   svc_dir[vllm-whisper]="${d:-whisper-large-v3}"

  local svc up_svcs=()
  # Usable-VRAM floor, by size rather than card name. An explicit service list
  # bypasses it (transcription runs on a small card).
  local usable; usable="$(gpu_usable_vram_gb)"
  if (( ${#want[@]} == 0 && usable > 0 && usable < VLLM_MIN_USABLE_VRAM_GB )); then
    err "GPU=$(detect_gpu_class) has ${usable}GiB usable — this catalogue needs ${VLLM_MIN_USABLE_VRAM_GB}GiB before a model places with room to run"
    exit 2
  fi
  for svc in "${VLLM_SERVICES[@]}"; do
    if (( ${#want[@]} )); then
      local hit=0 w
      for w in "${want[@]}"; do [[ "$w" == "$svc" ]] && hit=1; done
      (( hit )) || continue
    fi
    local d="${svc_dir[$svc]}"
    if [[ -f "$root/$d/config.json" ]]; then
      ok "weight: $d ($(du -sh "$root/$d" 2>/dev/null | cut -f1))"
      up_svcs+=("$svc")
    else
      warn "skip $svc — no weights at $root/$d/ (./scripts/download-vllm-models.sh $d)"
    fi
  done
  (( ${#up_svcs[@]} )) || { err "no vLLM weights on this node — run ./scripts/download-vllm-models.sh"; exit 2; }

  local recreate_args=()
  if (( recreate )); then
    info "force-recreate — reloading model (~3-5 min, qwen3.6-35b baseline)"
    recreate_args=(--force-recreate)
  fi
  docker compose -f "$COMPOSE_FILE" up -d --no-build "${recreate_args[@]}" "${up_svcs[@]}"

  echo
  echo "  → ./scripts/manage-vllm.sh status"
  echo "  → ./scripts/gen-litellm-config.sh && docker compose up -d --force-recreate litellm"
}

cmd_down()    { docker compose -f "$COMPOSE_FILE" down "$@"; }
cmd_restart() { docker compose -f "$COMPOSE_FILE" restart "$@"; }
cmd_logs()    { docker compose -f "$COMPOSE_FILE" logs -f "$@"; }
cmd_pull()    { docker compose -f "$COMPOSE_FILE" pull "$@"; }

cmd_status() {
  docker compose -f "$COMPOSE_FILE" ps
  echo
  for c in "${VLLM_SERVICES[@]}"; do
    s="$(docker inspect "$c" --format '{{.State.Health.Status}}' 2>/dev/null || echo missing)"
    printf "  %-15s %s\n" "$c:" "$s"
  done
}

[[ $# -eq 0 ]] && usage
sub="$1"; shift
case "$sub" in
  up)         cmd_up "$@" ;;
  down)       cmd_down "$@" ;;
  restart)    cmd_restart "$@" ;;
  logs)       cmd_logs "$@" ;;
  status|ps)  cmd_status ;;
  pull)       cmd_pull "$@" ;;
  -h|--help)  usage 0 ;;
  *)          err "unknown subcommand: $sub"; usage ;;
esac
