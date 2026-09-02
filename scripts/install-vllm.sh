#!/usr/bin/env bash
# Usage: install-vllm.sh [--reinstall] [--image <tag>]
#
# Prepares a GPU node: GPU runtime check, vLLM image (pull + this repo's layer),
# model directory. Weights: download-vllm-models.sh; services: manage-vllm.sh.
#
# Environment:
#   VLLM_BASE_IMAGE   upstream image (default: by architecture, lib.sh)
#   VLLM_MODELS_ROOT  model storage (default /var/lib/vllm/models)
#
# Flags:
#   --reinstall       re-pull the base image
#   --image <tag>     one-off base image override
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
ENV_FILE="${PROJECT_DIR}/.env"
COMPOSE_FILE="${PROJECT_DIR}/docker-compose.vllm.yml"
source "${SCRIPT_DIR}/lib.sh"

REINSTALL=0
IMAGE_OVERRIDE=""
while [[ $# -gt 0 ]]; do
  case "$1" in
    --reinstall)  REINSTALL=1; shift ;;
    --image)      IMAGE_OVERRIDE="$2"; shift 2 ;;
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

# A real --gpus passthrough is the gate; `docker info` Runtimes has false negatives.
hdr "1. GPU runtime check"
if docker run --rm --gpus all --entrypoint nvidia-smi nvcr.io/nvidia/cuda:12.6.3-base-ubuntu24.04 -L &>/dev/null; then
  ok "GPU passthrough confirmed (--gpus all)"
elif docker info 2>/dev/null | grep -q "Runtimes:.*nvidia"; then
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
# Base (pulled) and derived (built here) are separate tags.
VLLM_BASE_IMAGE="${IMAGE_OVERRIDE:-${VLLM_BASE_IMAGE:-$(vllm_default_image)}}"
[[ -n "$VLLM_BASE_IMAGE" ]] || { err "could not determine the vLLM base image — pass --image or set VLLM_BASE_IMAGE"; exit 1; }
VLLM_IMAGE="kloudchat-vllm:local"
echo "  base:    $VLLM_BASE_IMAGE"
echo "  derived: $VLLM_IMAGE"

if (( REINSTALL )) || ! docker image inspect "$VLLM_BASE_IMAGE" &>/dev/null; then
  echo "  → pull (~10 GB)"
  docker pull "$VLLM_BASE_IMAGE"
fi

# Digest the tag resolved to, recorded for reproducible rebuilds.
BASE_DIGEST="$(image_base_digest "$VLLM_BASE_IMAGE")"
[[ -n "$BASE_DIGEST" ]] && echo "  digest:  $BASE_DIGEST"

# This repo's layer over the base: pytest and the audio decoders the
# transcription endpoint needs.
echo "  → building services/vllm/Dockerfile onto the base"
docker build --quiet \
  --build-arg "BASE_IMAGE=$VLLM_BASE_IMAGE" \
  -t "$VLLM_IMAGE" \
  "${PROJECT_DIR}/services/vllm" >/dev/null

ok "image ready: $(docker image inspect "$VLLM_IMAGE" --format '{{.Size}}' | awk '{printf "%.1fGB",$1/1024/1024/1024}')"

# .env for docker-compose.vllm.yml
env_set VLLM_IMAGE "$VLLM_IMAGE"
env_set VLLM_BASE_IMAGE "$VLLM_BASE_IMAGE"
[[ -n "$BASE_DIGEST" ]] && env_set VLLM_BASE_DIGEST "$BASE_DIGEST"

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

hdr "4. Next steps"
cat <<EOF

  ./scripts/download-vllm-models.sh                # only weights this card can serve
  ./scripts/manage-vllm.sh up                      # placement is the scheduler's job: python3 -m scheduler apply

  # The scheduler writes VLLM_*_URL and WHISPER_URLS into the orchestrator's .env;
  # gen-litellm-config.sh registers from them.

EOF
