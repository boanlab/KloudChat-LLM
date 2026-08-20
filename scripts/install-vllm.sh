#!/usr/bin/env bash
# Usage: install-vllm.sh [--reinstall] [--image <tag>] [--no-whisper]
#
# Prepares a GPU node for everything it serves. Transcription is included: a GPU
# node has one role, and there is nothing else to install separately.
#
#   1. GPU runtime check and vLLM image pull
#   2. Model directory
#   3. Transcription backend — an amd64 container. arm64 (GB10) does not install
#      it and delegates STT to OpenRouter, because aarch64 ctranslate2 wheels are
#      CPU-only and cannot use the card.
#
# Weights are downloaded by download-vllm-models.sh; services are started by
# manage-vllm.sh.
#
# Environment:
#   VLLM_IMAGE        image override (default: chosen by architecture)
#   VLLM_MODELS_ROOT  model storage location (default /var/lib/vllm/models)
#
# Flags:
#   --reinstall       re-pull the image (also reinstalls the transcription backend)
#   --image <tag>     one-off override of the base image, same as VLLM_BASE_IMAGE
#   --no-whisper      skip the transcription backend
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
ENV_FILE="${PROJECT_DIR}/.env"
COMPOSE_FILE="${PROJECT_DIR}/docker-compose.vllm.yml"
source "${SCRIPT_DIR}/lib.sh"

REINSTALL=0
IMAGE_OVERRIDE=""
WITH_WHISPER=1
while [[ $# -gt 0 ]]; do
  case "$1" in
    --reinstall)  REINSTALL=1; shift ;;
    --image)      IMAGE_OVERRIDE="$2"; shift 2 ;;
    --no-whisper) WITH_WHISPER=0; shift ;;
    -h|--help)    sed -n '2,/^[^#]/p' "$0" | sed -n 's/^# \{0,1\}//p'; exit 0 ;;
    *)            err "unknown option: $1"; exit 1 ;;
  esac
done

hdr "0. Environment"
require_supported_platform
ok "OS / ARCH: $(detect_os) / $(detect_arch)"

has_nvidia_gpu || { err "no NVIDIA GPU detected — vLLM requires one"; exit 1; }
ok "GPU: $(get_gpu_name) (class=$(detect_gpu_class))"

command -v docker &>/dev/null || { err "Docker not found."; exit 1; }
ok "Docker $(docker --version | awk '{print $3}' | tr -d ',')"

# A real `docker run --gpus all` passthrough is the gate. The Runtimes line in
# `docker info` reports false negatives while the daemon is busy.
hdr "1. GPU runtime check"
if docker run --rm --gpus all --entrypoint nvidia-smi nvcr.io/nvidia/cuda:12.6.3-base-ubuntu24.04 -L &>/dev/null; then
  ok "GPU passthrough confirmed (--gpus all)"
elif docker info 2>/dev/null | grep -q "Runtimes:.*nvidia"; then
  # Registered runtime with a failed probe: most likely a CUDA base image pull.
  # Not fatal — the vLLM container settles it.
  warn "GPU passthrough probe failed but the nvidia runtime is registered — likely a CUDA base image pull issue, continuing"
else
  err "the nvidia container runtime is not working — install nvidia-container-toolkit and restart docker"
  echo "  → curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey | sudo gpg --dearmor -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg"
  echo "  → curl -fsSL https://nvidia.github.io/libnvidia-container/stable/deb/nvidia-container-toolkit.list | sed 's#deb #deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] #g' | sudo tee /etc/apt/sources.list.d/nvidia-container-toolkit.list"
  echo "  → sudo apt update && sudo apt install -y nvidia-container-toolkit"
  echo "  → sudo nvidia-ctk runtime configure --runtime=docker && sudo systemctl restart docker"
  exit 1
fi

hdr "2. vLLM image"
# Base and derived are separate tags: sharing one overwrites the pulled tag with
# a local build, which destroys the provenance the digest pin depends on.
VLLM_BASE_IMAGE="${IMAGE_OVERRIDE:-${VLLM_BASE_IMAGE:-$(vllm_default_image)}}"
[[ -n "$VLLM_BASE_IMAGE" ]] || { err "could not determine the vLLM base image — pass --image or set VLLM_BASE_IMAGE"; exit 1; }
VLLM_IMAGE="kloudchat-vllm:local"
echo "  base:    $VLLM_BASE_IMAGE"
echo "  derived: $VLLM_IMAGE"

if (( REINSTALL )) || ! docker image inspect "$VLLM_BASE_IMAGE" &>/dev/null; then
  echo "  → pull (~10 GB)"
  docker pull "$VLLM_BASE_IMAGE"
fi

# The digest the tag actually resolved to. Recorded so a rebuild on another node,
# or after the tag moves, is reproducible rather than "whatever nightly is today"
# — a moving base once relocated the tool-parser registry, which leaves every
# container healthy and every tool call silently unparsed.
BASE_DIGEST="$(image_base_digest "$VLLM_BASE_IMAGE")"
[[ -n "$BASE_DIGEST" ]] && echo "  digest:  $BASE_DIGEST"

# pytest layer over the base. Context is services/vllm for its patches/.
echo "  → building services/vllm/Dockerfile (pytest layer) onto the base"
docker build --quiet \
  --build-arg "BASE_IMAGE=$VLLM_BASE_IMAGE" \
  -t "$VLLM_IMAGE" \
  "${PROJECT_DIR}/services/vllm" >/dev/null

ok "image ready: $(docker image inspect "$VLLM_IMAGE" --format '{{.Size}}' | awk '{printf "%.1fGB",$1/1024/1024/1024}')"

# Recorded in .env for docker-compose.vllm.yml. VLLM_IMAGE is what compose runs;
# the base and its digest are recorded so the build can be reproduced.
env_set VLLM_IMAGE "$VLLM_IMAGE"
env_set VLLM_BASE_IMAGE "$VLLM_BASE_IMAGE"
[[ -n "$BASE_DIGEST" ]] && env_set VLLM_BASE_DIGEST "$BASE_DIGEST"

# MLA attention backend for this card. A hardware fact, so it is decided where
# the hardware is, not by a default in compose that happens to suit one card.
# Empty leaves vLLM to choose, which is what an unrecognised card should get.
MLA_BACKEND="$(mla_attention_backend)"
if [[ -n "$MLA_BACKEND" ]]; then
  echo "  MLA attention backend for $(detect_gpu_class): $MLA_BACKEND"
  env_set VLLM_GLMFLASH_ATTN_BACKEND "$MLA_BACKEND"
else
  warn "unrecognised card — leaving the MLA attention backend to vLLM"
fi

hdr "3. Model directory"
echo "  VLLM_MODELS_ROOT: $VLLM_MODELS_ROOT"
if [[ ! -d "$VLLM_MODELS_ROOT" ]]; then
  if [[ -w "$(dirname "$VLLM_MODELS_ROOT")" ]]; then
    mkdir -p "$VLLM_MODELS_ROOT"
  else
    sudo mkdir -p "$VLLM_MODELS_ROOT"
    sudo chown "$USER:$USER" "$VLLM_MODELS_ROOT"
  fi
fi
ok "directory ready: $VLLM_MODELS_ROOT ($(df -h "$VLLM_MODELS_ROOT" 2>/dev/null | awk 'NR==2 {print $4}' || echo '?') free)"

# ── 4. Transcription — same node, same card ─────────────────────────────────
#
# amd64 only: aarch64 ctranslate2 wheels are CPU-only. arm64 nodes keep no
# backend, and the resulting empty WHISPER_URLS routes STT to OpenRouter.
# A failure here leaves vLLM installed and is reported in the summary.
install_whisper() {
  local port model device compute img_ns img_tag data_root
  port="${WHISPER_PORT:-$(env_get WHISPER_PORT)}"; port="${port:-9000}"
  model="${WHISPER_MODEL:-$(env_get WHISPER_MODEL)}"; model="${model:-large-v3}"
  device="${WHISPER_DEVICE:-$(env_get WHISPER_DEVICE)}"; device="${device:-auto}"
  # float16: supported since Maxwell. int8_float16 returns a runtime 500 on some
  # GPU/CTranslate2 combinations.
  compute="${WHISPER_COMPUTE_TYPE:-$(env_get WHISPER_COMPUTE_TYPE)}"; compute="${compute:-float16}"
  data_root="${WHISPER_DATA_ROOT:-$(env_get WHISPER_DATA_ROOT)}"; data_root="${data_root:-/var/lib/whisper}"

  echo "  model=${model} device=${device} compute=${compute} :${port}"

  # Weight cache volume (HF_HOME). /var/lib usually needs root to mkdir.
  local sudo_cmd=""; [[ -w "$(dirname "$data_root")" ]] || sudo_cmd="sudo"
  $sudo_cmd mkdir -p "$data_root"

  img_ns="$(env_get KLOUDCHAT_IMAGE_NS)"; img_ns="${img_ns:-boanlab}"
  img_tag="$(env_get KLOUDCHAT_IMAGE_TAG)"; img_tag="${img_tag:-latest}"
  local wenv=(WHISPER_PORT="$port" WHISPER_DATA_ROOT="$data_root" WHISPER_MODEL="$model"
              WHISPER_DEVICE="$device" WHISPER_COMPUTE_TYPE="$compute" HF_TOKEN="$(env_get HF_TOKEN)"
              KLOUDCHAT_IMAGE_NS="$img_ns" KLOUDCHAT_IMAGE_TAG="$img_tag")

  # Docker Hub pull by default; --reinstall builds on the node.
  if (( REINSTALL )); then
    info "docker compose build --no-cache (cuda + faster-whisper, several minutes)"
    env "${wenv[@]}" docker compose -f "$COMPOSE_FILE" build --no-cache whisper || return 1
  else
    info "docker compose pull (${img_ns}/kloudchat-whisper)"
    env "${wenv[@]}" docker compose -f "$COMPOSE_FILE" pull whisper \
      || { err "pull failed — the image may not be published ('./scripts/build-push-images.sh whisper'), or build locally with '--reinstall'"; return 1; }
  fi
  env "${wenv[@]}" docker compose -f "$COMPOSE_FILE" up -d --no-build whisper || return 1

  local i
  for i in {1..60}; do
    curl -sf "http://localhost:${port}/health" &>/dev/null && break
    sleep 2
    (( i == 60 )) && { err "transcription backend is not responding — ./scripts/manage-vllm.sh logs whisper"; return 1; }
  done

  # Single-host wiring. Multi-node WHISPER_URLS comes from `scheduler apply`.
  if [[ -f "$ENV_FILE" ]]; then
    local shim_url="http://whisper-shim:9000"   # the shim always listens on 9000, independent of the backend port
    local cur; cur="$(env_get WHISPER_URL)"
    if [[ -z "$cur" || "$cur" == "http://host.docker.internal:"* || "$cur" == "$shim_url" ]]; then
      env_set WHISPER_URL "$shim_url"
    else
      warn "WHISPER_URL=$cur (custom). To route through the shim, set it to ${shim_url}."
    fi
    local local_be="http://host.docker.internal:${port}"
    local urls; urls="$(env_get WHISPER_URLS)"
    if [[ -z "$urls" ]]; then env_set WHISPER_URLS "$local_be"
    elif ! grep -qF "$local_be" <<<"$urls"; then env_set WHISPER_URLS "${urls},${local_be}"; fi
  fi

  # Same-host shim restart, clearing the 10s health cache and in-flight counters.
  # Skipped automatically when the shim runs elsewhere.
  if [[ -f "${PROJECT_DIR}/docker-compose.yml" ]] \
     && docker compose --project-directory "$PROJECT_DIR" ps -q whisper-shim 2>/dev/null | grep -q .; then
    info "restarting whisper-shim (refresh health cache and in-flight counters)"
    docker compose --project-directory "$PROJECT_DIR" restart whisper-shim || warn "whisper-shim restart failed"
  fi

  echo "  weights lazy-load into ${data_root} on the first call — prewarm: ./scripts/download-vllm-models.sh whisper"
  return 0
}

WHISPER_OK=skipped
if (( WITH_WHISPER )); then
  hdr "4. Transcription backend"
  if [[ "$(detect_arch)" != amd64 ]]; then
    WHISPER_OK="delegated (arm64 — OpenRouter voxtral)"
    info "arm64 node — no transcription backend here; STT goes to OpenRouter."
  elif install_whisper; then
    WHISPER_OK=ok
  else
    WHISPER_OK=failed
    warn "transcription backend failed to install — vLLM is fine. Retry: ./scripts/install-vllm.sh --reinstall"
  fi
fi

hdr "5. Next steps"
cat <<EOF

  ./scripts/download-vllm-models.sh                # only weights this card can serve
  ./scripts/manage-vllm.sh up                      # placement is the scheduler's job: python3 -m scheduler apply

  # Fill in VLLM_*_URL in .env, then re-run setup.sh or gen-litellm-config.sh.
  # LiteLLM load-balances across every deployment of the same model_name.

  transcription: ${WHISPER_OK}

EOF
