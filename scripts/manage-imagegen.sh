#!/usr/bin/env bash
# Usage:
#   manage-imagegen.sh download [alias...]      fetch weights (default: z-image-turbo)
#   manage-imagegen.sh up [--recreate] [alias...]
#                                               compose up (omit alias = every model with weights)
#   manage-imagegen.sh down                     stop + remove containers
#   manage-imagegen.sh restart [svc]            restart
#   manage-imagegen.sh logs [svc]               follow logs
#   manage-imagegen.sh status                   container + healthcheck status
#
# Aliases (lib.sh::IMAGEGEN_MODELS):
#   z-image-turbo     Tongyi-MAI/Z-Image-Turbo   31 GB  imagegen-z-image    :8091
#   qwen-image-2512   Qwen/Qwen-Image-2512       54 GB  imagegen-qwen-image :8092
#
# Per-node, outside the scheduler. Compose project: kloudchat-imagegen.
# On the orchestrator, list what runs where in IMAGEGEN_BACKENDS, then
#   ./scripts/gen-litellm-config.sh && docker compose up -d imagegen-shim
#   docker compose up -d --force-recreate litellm
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
COMPOSE_FILE="${PROJECT_DIR}/docker-compose.imagegen.yml"
source "${SCRIPT_DIR}/lib.sh"

[[ -f "$COMPOSE_FILE" ]] || { err "$COMPOSE_FILE not found"; exit 1; }

# `up` and `status` order
IMAGEGEN_ALIASES=(z-image-turbo qwen-image-2512)

usage() {
  sed -n '2,/^[^#]/p' "$0" | sed -n 's/^# \{0,1\}//p'
  exit "${1:-1}"
}

compose() { docker compose -f "$COMPOSE_FILE" "$@"; }

known_alias() {
  [[ -n "${IMAGEGEN_MODELS[$1]:-}" ]] || { err "Unknown alias: $1 (${IMAGEGEN_ALIASES[*]})"; exit 1; }
}

# Weight directory: IMAGEGEN_<MODEL>_DIR from .env, else the alias
weight_dir() {
  local d=""
  case "$1" in
    z-image-turbo)   d="$(env_get IMAGEGEN_ZIMAGE_DIR)" ;;
    qwen-image-2512) d="$(env_get IMAGEGEN_QWENIMAGE_DIR)" ;;
  esac
  echo "${VLLM_MODELS_ROOT}/${d:-$1}"
}

has_weights() {
  local dest; dest="$(weight_dir "$1")"
  [[ -f "$dest/model_index.json" ]] && compgen -G "$dest/transformer/*.safetensors" >/dev/null
}

cmd_download() {
  (( $# )) || set -- z-image-turbo
  command -v uv &>/dev/null || { err "uv not found. curl -LsSf https://astral.sh/uv/install.sh | sh"; exit 1; }
  local HF_TOKEN; HF_TOKEN="$(env_get HF_TOKEN)"
  mkdir -p "$VLLM_MODELS_ROOT"
  local alias dest need free
  for alias in "$@"; do
    known_alias "$alias"
    dest="$(weight_dir "$alias")"
    hdr "$alias → ${IMAGEGEN_MODELS[$alias]}"
    echo "  dest: $dest"
    if has_weights "$alias"; then
      ok "already downloaded ($(du -sh "$dest" | cut -f1)) — to re-download, delete the directory and re-run"
      continue
    fi
    need="${IMAGEGEN_MODEL_DISK_GB[$alias]}"
    free=$(( $(df -Pk "$VLLM_MODELS_ROOT" | awk 'NR==2 {print $4}') / 1024 / 1024 ))
    if (( free < need )); then
      err "needs ~${need}GiB of disk; ${VLLM_MODELS_ROOT} has ${free}GiB free"; exit 1
    fi
    # Sample images and the repo's own docs are not needed to serve
    HF_TOKEN="$HF_TOKEN" HF_HUB_DOWNLOAD_TIMEOUT=180 \
    uv tool run --from "huggingface_hub[hf_xet]" hf download \
      "${IMAGEGEN_MODELS[$alias]}" --local-dir "$dest" --max-workers 4 \
      --exclude "*.jpg" --exclude "*.png" --exclude "*.gguf" --exclude "*.pth"
    has_weights "$alias" || { err "$alias: no transformer weights after download"; exit 1; }
    ok "received $(du -sh "$dest" | cut -f1)"
  done
}

cmd_up() {
  local recreate=() want=() alias
  while [[ $# -gt 0 ]]; do
    case "$1" in
      --recreate) recreate=(--force-recreate); shift ;;
      *)          known_alias "$1"; want+=("$1"); shift ;;
    esac
  done
  if ! gpu_is_supported; then
    err "$(get_gpu_name) is not a supported card (GB10, RTX 5090, RTX PRO 5000/6000)"
    exit 2
  fi
  (( ${#want[@]} )) || want=("${IMAGEGEN_ALIASES[@]}")

  local svcs=()
  for alias in "${want[@]}"; do
    if has_weights "$alias"; then
      ok "weight: $alias ($(du -sh "$(weight_dir "$alias")" 2>/dev/null | cut -f1))"
      svcs+=("${IMAGEGEN_SERVICE[$alias]}")
    else
      warn "skip $alias — no weights at $(weight_dir "$alias")/ (./scripts/manage-imagegen.sh download $alias)"
    fi
  done
  (( ${#svcs[@]} )) || { err "no image weights on this node — run ./scripts/manage-imagegen.sh download"; exit 2; }
  (( ${#svcs[@]} < 2 )) || warn "more than one image model on this node: check the card holds both"

  compose up -d "${recreate[@]}" "${svcs[@]}"

  echo
  echo "  → ./scripts/manage-imagegen.sh status"
  echo "  → on the orchestrator: IMAGEGEN_BACKENDS in .env, then"
  echo "    ./scripts/gen-litellm-config.sh && docker compose up -d imagegen-shim && docker compose up -d --force-recreate litellm"
}

cmd_status() {
  compose ps
  echo
  local alias c s
  for alias in "${IMAGEGEN_ALIASES[@]}"; do
    c="${IMAGEGEN_SERVICE[$alias]}"
    s="$(docker inspect "$c" --format '{{if .State.Health}}{{.State.Health.Status}}{{else}}{{.State.Status}}{{end}}' 2>/dev/null)" || s=missing
    printf "  %-20s %-10s :%s local/%s\n" "$c:" "$s" "${IMAGEGEN_PORT[$alias]}" "$alias"
  done
}

[[ $# -eq 0 ]] && usage
sub="$1"; shift
case "$sub" in
  download)   cmd_download "$@" ;;
  up)         cmd_up "$@" ;;
  down)       compose down "$@" ;;
  restart)    compose restart "$@" ;;
  logs)       compose logs -f "$@" ;;
  status|ps)  cmd_status ;;
  -h|--help)  usage 0 ;;
  *)          err "unknown subcommand: $sub"; usage ;;
esac
