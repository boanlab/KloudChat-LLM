#!/usr/bin/env bash
# Usage:
#   manage-vllm.sh up [--recreate] [svc...]     verify weights + compose up
#                                               (svc = vllm-qwen35b | vllm-qwen122b | vllm-glmflash | whisper;
#                                                omit = every service with local weights — see the
#                                                warning in cmd_up, the scheduler owns placement)
#   manage-vllm.sh down [-v]                    stop + remove containers (transcription included)
#   manage-vllm.sh restart [svc]                restart
#   manage-vllm.sh logs [svc]                   follow logs
#   manage-vllm.sh status                       container + healthcheck status
#   manage-vllm.sh pull                         update image
#
# Services (no compose profiles — the lineup is small enough that both are default):
#   vllm-qwen35b     Qwen3.6-35B-A3B  chat + vision + deep-research + coding
#   vllm-qwen122b    Qwen3.5-122B-A10B top chat — 78 GiB, wants the card to itself
#   vllm-glmflash    GLM-4.7-Flash cheap-decode floor
#   vllm-gemma26b    Gemma-4-26B-A4B — a second model family
#   vllm-coder30b    Qwen3-Coder-30B-A3B — coding
#   vllm-qwen27b     Qwen3.6-27B — the one dense model
#   whisper          resident transcription backend (amd64 only — arm64 delegates STT to OpenRouter)
#
# Which vLLM subset lands on which node is the scheduler's call: `python -m scheduler
# plan`. Transcription is not placed: it is resident on every amd64 node.
#
# compose project name = kloudchat-vllm — lifecycle separated from the main stack.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
COMPOSE_FILE="${PROJECT_DIR}/docker-compose.vllm.yml"
source "${SCRIPT_DIR}/lib.sh"

[[ -f "$COMPOSE_FILE" ]] || { err "$COMPOSE_FILE not found"; exit 1; }

usage() {
  sed -n '2,/^[^#]/p' "$0" | sed -n 's/^# \{0,1\}//p'
  exit "${1:-1}"
}

cmd_up() {
  local recreate=0 want=() extra=()
  while [[ $# -gt 0 ]]; do
    case "$1" in
      --recreate) recreate=1; shift ;;
      vllm-*)     want+=("$1"); shift ;;
      # No weight check: the backend lazy-loads on the first call.
      whisper)    extra+=("$1"); shift ;;
      *) err "Unknown: $1"; usage ;;
    esac
  done

  # Placement belongs to the scheduler. With no service arguments this starts
  # every service whose weights are on the node, which on a shared card sums the
  # gpu_util fractions past 1.0 and OOMs. `up whisper` alone must not read as
  # "start everything".
  local only_extra=0
  if (( ${#want[@]} == 0 && ${#extra[@]} > 0 )); then only_extra=1; fi

  if (( ${#want[@]} == 0 && only_extra == 0 )); then
    warn "no service named — starting every service with local weights."
    warn "  the scheduler decides placement: python -m scheduler apply"
    warn "  to drive this node by hand: manage-vllm.sh up vllm-qwen35b [vllm-qwen122b ...]"
  fi

  local root="${VLLM_MODELS_ROOT:-/var/lib/vllm/models}"
  local gpu_class; gpu_class="$(detect_gpu_class)"

  # Explicit .env value over the default weight directory.
  local cd bd gd
  cd="$(env_get VLLM_QWEN35B_DIR)"; bd="$(env_get VLLM_QWEN122B_DIR)"
  gd="$(env_get VLLM_GEMMA26B_DIR)"
  local kd dd; kd="$(env_get VLLM_CODER30B_DIR)"; dd="$(env_get VLLM_QWEN27B_DIR)"
  declare -A svc_dir=(
    [vllm-qwen35b]="${cd:-qwen3.6-35b-nvfp4}"
    [vllm-glmflash]="$(env_get VLLM_GLMFLASH_DIR)"
    [vllm-qwen122b]="${bd:-qwen3.5-122b-a10b}"
    [vllm-gemma26b]="${gd:-gemma-4-26b}"
    [vllm-coder30b]="${kd:-qwen3-coder-30b}"
    [vllm-qwen27b]="${dd:-qwen3.6-27b}"
  )

  local svc up_svcs=()
  if (( only_extra == 0 )); then
    # By size, not by card name. FP4 was never the whole story: the int4 aliases
    # run on an Ada or Ampere card, and a 24 GiB one still places exactly one
    # model at its floor and 0.92 of the card. What is missing there is room.
    # Transcription still runs on such a card; `up whisper` never reaches here.
    local usable; usable="$(gpu_usable_vram_gb)"
    if (( usable > 0 && usable < VLLM_MIN_USABLE_VRAM_GB )); then
      die "GPU=$(detect_gpu_class) has ${usable}GiB usable — this catalogue needs ${VLLM_MIN_USABLE_VRAM_GB}GiB before a model places with room to run"
    fi
    for svc in vllm-qwen35b vllm-qwen122b vllm-glmflash vllm-gemma26b \
               vllm-coder30b vllm-qwen27b; do
      # An explicit list narrows the set; otherwise every service with weights
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
  fi
  if (( ${#extra[@]} )); then up_svcs+=("${extra[@]}"); fi

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
  for c in vllm-qwen35b vllm-qwen122b vllm-glmflash vllm-gemma26b \
           vllm-coder30b vllm-qwen27b whisper; do
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
