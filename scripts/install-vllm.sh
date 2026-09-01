#!/usr/bin/env bash
# Usage: install-vllm.sh [--reinstall] [--image <tag>]
#
# Prepares a GPU node for everything it serves. One image covers all of it,
# transcription included: whisper is a vLLM service like the rest.
#
#   1. GPU runtime check and vLLM image pull
#   2. Model directory
#
# Weights are downloaded by download-vllm-models.sh; services are started by
# manage-vllm.sh.
#
# Environment:
#   VLLM_IMAGE        image override (default: chosen by architecture)
#   VLLM_MODELS_ROOT  model storage location (default /var/lib/vllm/models)
#
# Flags:
#   --reinstall       re-pull the image
#   --image <tag>     one-off override of the base image, same as VLLM_BASE_IMAGE
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

# This repo's layer over the base: pytest, the audio decoders the transcription
# endpoint needs, and the GB10 MLA patch. Context is services/vllm for patches/.
echo "  → building services/vllm/Dockerfile onto the base"
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

hdr "4. Next steps"
cat <<EOF

  ./scripts/download-vllm-models.sh                # only weights this card can serve
  ./scripts/manage-vllm.sh up                      # placement is the scheduler's job: python3 -m scheduler apply

  # Fill in VLLM_*_URL in .env, then re-run setup.sh or gen-litellm-config.sh.
  # LiteLLM load-balances across every deployment of the same model_name.
  # WHISPER_URLS follows the transcription model's placement, written by the
  # scheduler; an empty value is what routes STT to OpenRouter.

EOF
