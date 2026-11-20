#!/usr/bin/env bash

[[ -n "${__KC_LIB_SH:-}" ]] && return 0
__KC_LIB_SH=1

__R='\033[0;31m'; __G='\033[0;32m'; __Y='\033[1;33m'
__B='\033[1;34m'; __N='\033[0m'

hdr()  { echo; echo -e "${__B}━━━ $* ━━━${__N}"; }
# stderr: the config generators capture stdout as YAML
ok()   { echo -e "${__G}✓${__N} $*" >&2; }
info() { echo -e "${__G}[INFO]${__N} $*" >&2; }
warn() { echo -e "${__Y}[WARN]${__N} $*" >&2; }
err()  { echo -e "${__R}✗${__N} $*" >&2; }

detect_os() {
  case "$(uname -s)" in Linux) echo linux ;; *) echo unsupported ;; esac
}

detect_arch() {
  case "$(uname -m)" in
    x86_64|amd64)  echo amd64 ;;
    aarch64|arm64) echo arm64 ;;
    *)             echo unsupported ;;
  esac
}

require_supported_platform() {
  if [[ "$(detect_os)" == unsupported || "$(detect_arch)" == unsupported ]]; then
    err "Unsupported: $(uname -s) $(uname -m) (only Linux amd64/arm64 supported)"
    exit 1
  fi
}

__PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
__DEFAULT_ENV_FILE="${__PROJECT_DIR}/.env"

env_get() {
  local file="${ENV_FILE:-$__DEFAULT_ENV_FILE}"
  [[ -f "$file" ]] || return 0
  grep -E "^$1=" "$file" | head -1 | cut -d= -f2- || true
}

env_set() {
  local key="$1" val="$2" file="${ENV_FILE:-$__DEFAULT_ENV_FILE}"
  [[ -f "$file" ]] || { echo "[env_set] error: ${file} not found" >&2; return 1; }
  if grep -qE "^${key}=" "$file"; then sed -i "s|^${key}=.*|${key}=${val}|" "$file"
  else
    # File without a trailing newline
    [[ -s "$file" && -n "$(tail -c 1 "$file")" ]] && printf '\n' >> "$file"
    printf '%s=%s\n' "$key" "$val" >> "$file"
  fi
}

# Read and write check on $1 before a tmp+mv regeneration
assert_regen_writable() {
  local f="$1" d owner me; d="$(dirname "$f")"; me="$(id -un)"
  if [[ -e "$f" && ! -r "$f" ]]; then
    owner="$(stat -c '%U:%G' "$f" 2>/dev/null || echo '?')"
    err "$f not readable (owner: $owner, current user: $me)."
    err "  Running the script with sudo makes it root-owned → fix: sudo chown $(id -u):$(id -g) \"$f\"   (then run without sudo)"
    return 1
  fi
  if [[ ! -w "$d" ]]; then
    owner="$(stat -c '%U:%G' "$d" 2>/dev/null || echo '?')"
    err "Directory $d not writable (owner: $owner, current user: $me) → cannot update $f."
    err "  Fix: sudo chown $(id -u):$(id -g) \"$d\""
    return 1
  fi
  return 0
}

has_nvidia_gpu() { command -v nvidia-smi &>/dev/null && nvidia-smi -L &>/dev/null; }

# ~/.local/bin (uv, HF CLI) on PATH; non-interactive ssh omits it
case ":${PATH}:" in
  *":${HOME}/.local/bin:"*) ;;
  *) [[ -d "${HOME}/.local/bin" ]] && PATH="${HOME}/.local/bin:${PATH}" && export PATH ;;
esac

has_gb10() {
  has_nvidia_gpu || return 1
  nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | head -1 | grep -q 'GB10'
}

get_gpu_name() {
  has_nvidia_gpu || { echo ""; return; }
  nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | head -1
}

detect_gpu_class() {
  has_nvidia_gpu || { echo none; return; }
  has_gb10 && { echo gb10; return; }
  case "$(get_gpu_name)" in
    *"RTX PRO 6000 Blackwell"*|*"RTX 6000 Pro Blackwell"*|*"RTX 6000 PRO Blackwell"*) echo pro6000 ;;
    *"RTX PRO 5000 Blackwell"*|*"RTX 5000 Pro Blackwell"*|*"RTX 5000 PRO Blackwell"*) echo pro5000 ;;
    *"RTX 5090"*) echo rtx5090 ;;
    *)            echo unsupported ;;
  esac
}

# Supported cards: GB10, RTX 5090, RTX PRO 5000/6000 Blackwell
gpu_is_supported() {
  case "$(detect_gpu_class)" in
    gb10|pro6000|pro5000|rtx5090) return 0 ;;
    *) return 1 ;;
  esac
}

get_free_disk_gb() {
  df -BG "${1:-.}" 2>/dev/null | awk 'NR==2 {gsub("G",""); print $4; exit}'
}

# Compute capability of GPU 0 ("12.0"); empty when the driver cannot answer
gpu_compute_cap() {
  has_nvidia_gpu || { echo ""; return; }
  nvidia-smi --query-gpu=compute_cap --format=csv,noheader 2>/dev/null \
    | head -1 | tr -dc '0-9.'
}

# OS reserve (GiB) on a unified-memory node; must equal
# scheduler/inventory.py::_UNIFIED_RESERVE_BYTES
UNIFIED_RESERVE_GB=12

# Usable GPU memory (GiB), 0 without a GPU; unified-memory cards report [N/A]
# to nvidia-smi, so MemTotal minus the OS reserve stands in
gpu_usable_vram_gb() {
  has_nvidia_gpu || { echo 0; return; }
  local mib kb total
  mib="$(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits 2>/dev/null | head -1 | tr -dc '0-9')"
  if [[ -n "$mib" ]]; then echo $(( mib / 1024 )); return; fi
  kb="$(awk '/^MemTotal:/ {print $2}' /proc/meminfo 2>/dev/null)"
  [[ -n "$kb" ]] || { echo 0; return; }
  total=$(( kb / 1024 / 1024 ))
  (( total > UNIFIED_RESERVE_GB )) && echo $(( total - UNIFIED_RESERVE_GB )) || echo 0
}

# OpenRouter commercial catalogue, one array per provider; array order is
# picker order. Ids: https://openrouter.ai/api/v1/models
# Only the providers in OR_PROVIDER_ALLOW are registered (see below); the
# TENCENT, XIAOMI and MINIMAX arrays stay so an operator can allow them again.
OPENAI_MODELS=(gpt-6-astra gpt-6.1-sol gpt-6.1-terra gpt-6-luna gpt-5.4-nano gpt-5.3-codex gpt-oss-120b gpt-oss-20b)
ANTHROPIC_MODELS=(claude-fable-5.2 claude-opus-5.5 claude-sonnet-5.5 claude-haiku-5.5)
GOOGLE_MODELS=(gemini-4-argon gemini-4-carbon gemini-4-hydrogen gemini-3.8-flash gemini-3.5-flash-lite gemma-4-31b-it gemma-4-26b-a4b-it gemma-3-27b-it gemma-3-12b-it
              gemma-3-4b-it)
XAI_MODELS=(grok-4.7)
PERPLEXITY_MODELS=(sonar sonar-pro)
MISTRALAI_MODELS=(mistral-large-4-0 mistral-medium-3-5 mistral-small-2603 ministral-14b-2512
                  ministral-8b-2512 ministral-3b-2512 codestral-2508 devstral-2512 mistral-nemo)
META_LLAMA_MODELS=(llama-4-maverick llama-4-scout llama-3.3-70b-instruct)
NVIDIA_MODELS=(nemotron-3-ultra-550b-a55b nemotron-3.5-super-vl nemotron-3.5-lightning)
DEEPSEEK_MODELS=(deepseek-v4-pro-0813 deepseek-v4.1-flash)
ZAI_MODELS=(glm-5.3 glm-5.3-flash)
MOONSHOTAI_MODELS=(kimi-k3.1-pro kimi-k3)
QWEN_MODELS=(qwen4-max qwen3.8-max-0902 qwen3-coder qwen3.6-35b-a3b qwen3.5-9b qwen3.5-35b-a3b
             qwen3-vl-235b-a22b-instruct)
BYTEDANCE_SEED_MODELS=(seed-2-1-turbo seed-2.0-lite)
TENCENT_MODELS=(hy4-preview)
XIAOMI_MODELS=(mimo-v2.6-flash)
MINIMAX_MODELS=(minimax-m3)
# "<OpenRouter provider>:<array prefix>" for each *_MODELS array above
OR_PROVIDERS=(openai:OPENAI anthropic:ANTHROPIC google:GOOGLE x-ai:XAI
              perplexity:PERPLEXITY nvidia:NVIDIA deepseek:DEEPSEEK z-ai:ZAI
              moonshotai:MOONSHOTAI qwen:QWEN bytedance-seed:BYTEDANCE_SEED
              meta-llama:META_LLAMA mistralai:MISTRALAI
              tencent:TENCENT xiaomi:XIAOMI minimax:MINIMAX)
# Providers the generator registers, commercial and free alike (comma-separated;
# `*` = every provider). Shell environment first, then .env; an empty value
# means this default, not "no limit".
OR_PROVIDER_ALLOW_DEFAULT="qwen,deepseek,anthropic,google,moonshotai,nvidia,openai,perplexity,x-ai,z-ai,bytedance-seed,meta-llama,mistralai"

# Image generation, cheapest per picture first (picker default). Criteria:
# Korean text rendered right, 16:9 ratio kept, served on chat/completions
# (GPT Image 2.5, FLUX 3 and Qwen Image 3 serve only OpenRouter's /images).
OR_IMAGE_MODELS=(
  bytedance-seed/seedream-5-0-flash
  bytedance-seed/seedream-5-5
  x-ai/grok-imagine-image-2.0
  google/gemini-nano-banana-2.1
  openai/gpt-5.4-image-2
  google/gemini-3-pro-image
)
# USD per image_output token. Gemini spends ~1290 tokens on a picture; the
# image-only models are priced as 4175 tokens per 1K picture (Seedream bills a
# flat $0.018 whatever token count it reports; KloudChat charges usage.cost).
# GPT-5.4 Image 2: $0.127 over 3866 output tokens for one 16:9 picture
declare -A MODEL_IMAGE_OUT_COST=(
  [bytedance-seed/seedream-5-0-flash]=0.0000043114
  [bytedance-seed/seedream-5-5]=0.0000047904
  [x-ai/grok-imagine-image-2.0]=0.0000095808
  [google/gemini-nano-banana-2.1]=0.00003
  [openai/gpt-5.4-image-2]=0.0000329196
  [google/gemini-3-pro-image]=0.00012
)
# Audio generation through chat/completions (streaming required): gpt-audio
# speech per token, lyria music per clip
OR_AUDIO_MODELS=(
  openai/gpt-audio-mini
  openai/gpt-audio
  google/lyria-3-clip-preview
)
# USD per 1M audio tokens, applied to the text portion as well
declare -A MODEL_AUDIO_OUT_PM=(
  [openai/gpt-audio-mini]=2.40
  [openai/gpt-audio]=64.00
  [google/lyria-3-clip-preview]=0.00
)
declare -A MODEL_AUDIO_IN_PM=(
  [openai/gpt-audio-mini]=0.60
  [openai/gpt-audio]=32.00
  [google/lyria-3-clip-preview]=0.00
)
# USD per clip; the catalogue states it in the model description, not `pricing`
declare -A MODEL_AUDIO_PER_CALL=(
  [google/lyria-3-clip-preview]=0.04
)

declare -A MODEL_IMAGE_IN_PM=(
  [bytedance-seed/seedream-5-0-flash]=0.00
  [bytedance-seed/seedream-5-5]=0.00
  [x-ai/grok-imagine-image-2.0]=0.00
  [google/gemini-nano-banana-2.1]=1.50
  [openai/gpt-5.4-image-2]=0.00
  [google/gemini-3-pro-image]=2.00
)

# vLLM catalogue: download alias → HF repo
declare -A VLLM_MODELS=(
  # Chat, NVFP4
  [qwen3.8-27b-nvfp4]="unsloth/Qwen3.8-27B-NVFP4"
  # Quality and judging, MoE, NVFP4 body and FP8 PLE table
  [qwen3.8-flash-next-nvfp4]="nvidia/Qwen3.8-Flash-Next-NVFP4"
  # Retrieval embeddings and reranking, BF16
  [bge-m3]="BAAI/bge-m3"
  [bge-reranker-v2-m3]="BAAI/bge-reranker-v2-m3"
  # Transcription, FP16
  [whisper-large-v3]="openai/whisper-large-v3"
)
: "${VLLM_MODELS_ROOT:=$(env_get VLLM_MODELS_ROOT)}"
: "${VLLM_MODELS_ROOT:=/var/lib/vllm/models}"

# Resident weight size (GiB) for the fit check; the checkpoint size except
# flash-next (PLE table off the GPU)
declare -A VLLM_MODEL_WEIGHT_GB=(
  [qwen3.8-27b-nvfp4]=22
  [qwen3.8-flash-next-nvfp4]=79
  [bge-m3]=3
  [bge-reranker-v2-m3]=3
  [whisper-large-v3]=4
)
# GiB over the weights: activations plus KV for one request
VLLM_RUNTIME_HEADROOM_GB=6

# Image models (vllm-omni, docker-compose.vllm-omni.yml): download alias → HF repo.
# Diffusers layout (model_index.json), weights under VLLM_MODELS_ROOT/<alias>.
declare -A OMNI_MODELS=(
  [z-image-turbo]="Tongyi-MAI/Z-Image-Turbo"
  [qwen-image-2512]="Qwen/Qwen-Image-2512"
  [qwen-image-edit-2511]="Qwen/Qwen-Image-Edit-2511"
)
# Download size (GiB), for the disk check
declare -A OMNI_MODEL_DISK_GB=(
  [z-image-turbo]=31
  [qwen-image-2512]=54
  [qwen-image-edit-2511]=58
)
# Compose service and served port per alias
declare -A OMNI_SERVICE=(
  [z-image-turbo]=omni-z-image
  [qwen-image-2512]=omni-qwen-image
  [qwen-image-edit-2511]=omni-qwen-image-edit
)
declare -A OMNI_PORT=(
  [z-image-turbo]=8091
  [qwen-image-2512]=8092
  [qwen-image-edit-2511]=8093
)

# Download set for a node without an explicit model list
VLLM_PREFERRED_MODELS=(qwen3.8-27b-nvfp4)

# Usable-VRAM floor (GiB): the RTX 5090
VLLM_MIN_USABLE_VRAM_GB=32
# System memory for Flash-Next on a discrete card (GiB): 48 GiB PLE table plus headroom
VLLM_FLASHNEXT_MIN_RAM_GB=64

# Prints why this node cannot serve alias $1 and returns 1; 0 silently otherwise
vllm_model_unservable_reason() {
  local alias="$1" weight="${VLLM_MODEL_WEIGHT_GB[$1]:-0}"
  local vram; vram="$(gpu_usable_vram_gb)"
  if ! has_nvidia_gpu; then
    echo "no NVIDIA GPU on this node"; return 1
  fi
  if ! gpu_is_supported; then
    echo "$(get_gpu_name) is not a supported card (GB10, RTX 5090, RTX PRO 5000/6000)"; return 1
  fi
  if (( vram > 0 && vram < VLLM_MIN_USABLE_VRAM_GB )); then
    echo "the card has ${vram}GiB usable; this catalogue needs ${VLLM_MIN_USABLE_VRAM_GB}GiB before anything places with room to run"
    return 1
  fi
  # 79 GiB resident: GB10 or RTX PRO 6000 only; on the latter the 48 GiB PLE
  # table is pinned in system RAM
  if [[ "$alias" == qwen3.8-flash-next-nvfp4 ]]; then
    local class; class="$(detect_gpu_class)"
    if ! [[ "$class" =~ ^(gb10|pro6000)$ ]]; then
      echo "needs a GB10 or an RTX PRO 6000"; return 1
    fi
    local ram_gb; ram_gb=$(( $(awk '/^MemTotal:/ {print $2}' /proc/meminfo 2>/dev/null || echo 0) / 1024 / 1024 ))
    if [[ "$class" == pro6000 ]] && (( ram_gb < VLLM_FLASHNEXT_MIN_RAM_GB )); then
      echo "needs ${VLLM_FLASHNEXT_MIN_RAM_GB}GiB of system memory for its PLE table; this node has ${ram_gb}GiB"
      return 1
    fi
  fi
  local need=$(( weight + VLLM_RUNTIME_HEADROOM_GB ))
  if (( weight > 0 && vram > 0 && vram < need )); then
    echo "needs ~${need}GiB (weights ${weight} + runtime ${VLLM_RUNTIME_HEADROOM_GB}) but the card has ${vram}GiB"
    return 1
  fi
  return 0
}

# vLLM base image: the latest release, multi-arch (amd64, arm64).
# install-vllm.sh pins the resolved digest per node as VLLM_BASE_DIGEST.
VLLM_IMAGE_LATEST="vllm/vllm-openai:latest"

vllm_default_image() {
  case "$(detect_arch)" in
    arm64|amd64) echo "$VLLM_IMAGE_LATEST" ;;
    *)           echo "" ;;
  esac
}

# Registry digest of a local image; empty for a locally built tag
image_base_digest() {
  docker image inspect "$1" --format '{{if .RepoDigests}}{{index .RepoDigests 0}}{{end}}' 2>/dev/null
}

# Declared prices, USD per 1M tokens; or_refresh_prices overlays the live
# catalogue. Local models are 0.
declare -A MODEL_PRICE_IN_PM=(
  [gpt-6-astra]=10  [gpt-6.1-sol]=2  [gpt-6.1-terra]=1
  [gpt-6-luna]=0.1  [gpt-5.4-nano]=0.2  [gpt-5.3-codex]=1.75
  [claude-fable-5.2]=8  [claude-opus-5.5]=4  [claude-sonnet-5.5]=2
  [claude-haiku-5.5]=0.1  [gemini-4-argon]=2  [gemini-4-carbon]=2
  [gemini-4-hydrogen]=1.5  [gemini-3.8-flash]=0.825  [gemini-3.5-flash-lite]=0.3  [gemma-4-31b-it]=0.2
  [grok-4.7]=2  [sonar]=1  [sonar-pro]=3
  [nemotron-3-ultra-550b-a55b]=0.5  [nemotron-3.5-super-vl]=0.15  [nemotron-3.5-lightning]=0.07
  [deepseek-v4-pro-0813]=0.111  [deepseek-v4.1-flash]=0.3  [glm-5.3]=0.0699
  [glm-5.3-flash]=0.15  [kimi-k3.1-pro]=0.2403  [qwen4-max]=1.425
  [qwen3.8-max-0902]=2  [qwen3-coder]=0.3  [seed-2-1-turbo]=0.5
  [seed-2.0-lite]=0.25  [hy4-preview]=0.834  [mimo-v2.6-flash]=0.14
  [minimax-m3]=0.3
  [gemma-4-26b-a4b-it]=0.09
  [gemma-3-27b-it]=0.08
  [gemma-3-12b-it]=0.05
  [gemma-3-4b-it]=0.05
  [gpt-oss-120b]=0.15
  [gpt-oss-20b]=0.018
  [llama-3.3-70b-instruct]=0.71
  [llama-4-maverick]=0.35
  [llama-4-scout]=0.1
  [mistral-large-4-0]=0.7
  [mistral-medium-3-5]=1.5
  [mistral-small-2603]=0.15
  [ministral-14b-2512]=0.2
  [ministral-8b-2512]=0.15
  [ministral-3b-2512]=0.1
  [codestral-2508]=0.3
  [devstral-2512]=0.4
  [mistral-nemo]=0.029
  [qwen3.6-35b-a3b]=0.1
  [qwen3.5-9b]=0.17
  [qwen3.5-35b-a3b]=0.14
  [qwen3-vl-235b-a22b-instruct]=0.21
  [kimi-k3]=0.192
)
declare -A MODEL_PRICE_OUT_PM=(
  [gpt-6-astra]=50  [gpt-6.1-sol]=10  [gpt-6.1-terra]=5
  [gpt-6-luna]=0.5  [gpt-5.4-nano]=1.25  [gpt-5.3-codex]=14
  [claude-fable-5.2]=40  [claude-opus-5.5]=20  [claude-sonnet-5.5]=10
  [claude-haiku-5.5]=0.5  [gemini-4-argon]=10  [gemini-4-carbon]=10
  [gemini-4-hydrogen]=7.5  [gemini-3.8-flash]=4.125  [gemini-3.5-flash-lite]=2.5  [gemma-4-31b-it]=0.4
  [grok-4.7]=6  [sonar]=1  [sonar-pro]=15
  [nemotron-3-ultra-550b-a55b]=2.2  [nemotron-3.5-super-vl]=0.5  [nemotron-3.5-lightning]=0.2
  [deepseek-v4-pro-0813]=4.44  [deepseek-v4.1-flash]=1.2  [glm-5.3]=1.4
  [glm-5.3-flash]=0.5  [kimi-k3.1-pro]=8.4023  [qwen4-max]=4.275
  [qwen3.8-max-0902]=6  [qwen3-coder]=1  [seed-2-1-turbo]=2.5
  [seed-2.0-lite]=2  [hy4-preview]=2.501  [mimo-v2.6-flash]=0.28
  [minimax-m3]=1.2
  [gemma-4-26b-a4b-it]=0.3
  [gemma-3-27b-it]=0.35
  [gemma-3-12b-it]=0.15
  [gemma-3-4b-it]=0.1
  [gpt-oss-120b]=0.6
  [gpt-oss-20b]=0.09
  [llama-3.3-70b-instruct]=0.71
  [llama-4-maverick]=1.0
  [llama-4-scout]=0.3
  [mistral-large-4-0]=2.1
  [mistral-medium-3-5]=7.5
  [mistral-small-2603]=0.6
  [ministral-14b-2512]=0.2
  [ministral-8b-2512]=0.15
  [ministral-3b-2512]=0.1
  [codestral-2508]=0.9
  [devstral-2512]=2.0
  [mistral-nemo]=0.03
  [qwen3.6-35b-a3b]=0.9
  [qwen3.5-9b]=0.25
  [qwen3.5-35b-a3b]=1.0
  [qwen3-vl-235b-a22b-instruct]=1.9
  [kimi-k3]=16.0
)

# OpenRouter twins of local models, USD per 1M tokens,
# keyed by slug; or_price prefers the live figure
declare -A OR_TWIN_PRICE_IN_PM=(
  [qwen/qwen3.8-27b]=0.42  [qwen/qwen3.8-flash]=0.15
)
declare -A OR_TWIN_PRICE_OUT_PM=(
  [qwen/qwen3.8-27b]=3.00  [qwen/qwen3.8-flash]=0.47
)

per_token_cost() { awk -v v="$1" 'BEGIN { printf "%.10f", v/1000000 }'; }

has_openrouter() { [[ -n "$(env_get OPENROUTER_API_KEY)" ]]; }

# OpenRouter catalogue, fetched once per run
__OR_CATALOGUE_CACHE=""

or_catalogue() {
  has_openrouter || return 1
  command -v jq &>/dev/null || return 1
  if [[ -z "$__OR_CATALOGUE_CACHE" || ! -s "$__OR_CATALOGUE_CACHE" ]]; then
    local key tmp; key="$(env_get OPENROUTER_API_KEY)"
    tmp="$(mktemp -t or-catalogue.XXXXXX)" || return 1
    if ! curl -sf --max-time 20 https://openrouter.ai/api/v1/models \
              -H "Authorization: Bearer ${key}" -o "$tmp" 2>/dev/null; then
      rm -f "$tmp"; return 1
    fi
    __OR_CATALOGUE_CACHE="$tmp"
  fi
  cat "$__OR_CATALOGUE_CACHE"
}

# model_info lines for an OpenRouter slug that takes an effort: LiteLLM's
# supports_reasoning (without it drop_params discards reasoning_effort) and the
# levels KloudChat's /llm proxy and model listing read. Nothing when the
# catalogue is unreachable or the model takes no effort.
#   or_reasoning_info <slug>
or_reasoning_info() {
  local slug="$1"
  or_catalogue 2>/dev/null \
    | jq -r --arg s "$slug" '
        first(.data[] | select(.id == $s))
        | select(.supported_parameters // [] | index("reasoning_effort"))
        | (.reasoning // {}) as $r
        | "      supports_reasoning: true",
          (($r.supported_efforts // []) | select(length > 0)
            | "      kchat_reasoning_efforts: [" + (map(tojson) | join(", ")) + "]"),
          ($r.default_effort // empty | "      kchat_reasoning_default: " + tojson),
          "      kchat_reasoning_mandatory: " + (($r.mandatory // false) | tostring)
      ' 2>/dev/null || true
}

# model_info lines from the catalogue for an OpenRouter slug: context window,
# vision input, and function calling only when the catalogue says the model
# takes no tools. Nothing when the catalogue is unreachable or the slug unknown.
or_capability_info() {
  local slug="$1"
  or_catalogue 2>/dev/null \
    | jq -r --arg s "$slug" '
        first(.data[] | select(.id == $s))
        | (if .context_length then "      max_input_tokens: \(.context_length)" else empty end),
          (if (.architecture.input_modalities // [] | index("image")) then "      supports_vision: true" else empty end),
          (if (.supported_parameters // [] | index("tools")) then empty else "      supports_function_calling: false" end)
      ' 2>/dev/null || true
}

# Live price or the declared fallback, USD per 1M tokens
#   or_price <slug> in|out [fallback]
or_price() {
  local slug="$1" field="$2" fallback="${3:-}" key live
  if [[ "$field" == "in" ]]; then
    key="prompt";     fallback="${fallback:-${OR_TWIN_PRICE_IN_PM[$slug]:-0}}"
  else
    key="completion"; fallback="${fallback:-${OR_TWIN_PRICE_OUT_PM[$slug]:-0}}"
  fi
  live="$(or_catalogue 2>/dev/null \
          | jq -r --arg s "$slug" --arg k "$key" '
              .data[] | select(.id == $s) | (.pricing[$k] // empty | tonumber * 1000000)
            ' 2>/dev/null | head -1)"
  [[ -n "$live" && "$live" != "null" ]] && echo "$live" || echo "$fallback"
}

# Live prices over the declared tables, in place; sets OR_PRICE_TOTAL and
# OR_PRICE_MOVED. Run in the current shell, never in a command substitution.
or_refresh_prices() {
  has_openrouter || return 0
  local live; live="$(or_catalogue 2>/dev/null)" || return 0
  [[ -n "$live" ]] || return 0

  local prov disp var m slug pair moved=0 total=0
  for prov in "${OR_PROVIDERS[@]}"; do
    disp="${prov%%:*}"; var="${prov##*:}_MODELS[@]"
    for m in "${!var}"; do
      slug="${disp}/${m}"
      pair="$(jq -r --arg s "$slug" '
                .data[] | select(.id == $s)
                | "\((.pricing.prompt // "0" | tonumber) * 1000000) \((.pricing.completion // "0" | tonumber) * 1000000)"
              ' <<<"$live" 2>/dev/null | head -1)"
      [[ -n "$pair" ]] || continue
      total=$((total+1))
      local lin="${pair%% *}" lout="${pair##* }"
      # Numeric compare ("2" == "2.0")
      if ! awk -v a="$lin" -v b="${MODEL_PRICE_IN_PM[$m]:-0}" \
               -v c="$lout" -v d="${MODEL_PRICE_OUT_PM[$m]:-0}" \
              'BEGIN { exit !(a-b < 0.0005 && b-a < 0.0005 && c-d < 0.0005 && d-c < 0.0005) }'; then
        moved=$((moved+1))
      fi
      MODEL_PRICE_IN_PM[$m]="$lin"
      MODEL_PRICE_OUT_PM[$m]="$lout"
    done
  done
  OR_PRICE_TOTAL=$total; OR_PRICE_MOVED=$moved
}

# OpenRouter's free chat models, one slug per line: zero price both ways, text
# output, `:free` suffix, guardrail models excluded
or_free_models() {
  local m
  while IFS= read -r m; do
    [[ -n "$m" ]] && ! or_model_excluded "$m" && echo "$m"
  done < <(or_free_models_all)
  return 0
}

# The same before the provider and exclusion filters
or_free_models_all() {
  has_openrouter || return 0
  command -v jq &>/dev/null || return 0
  or_catalogue 2>/dev/null \
    | jq -r '.data[]
        | select((.pricing.prompt // "0" | tonumber) == 0)
        | select((.pricing.completion // "0" | tonumber) == 0)
        | select(.architecture.output_modalities // [] | index("text"))
        | select(.id | endswith(":free"))
        | select(.id | test("guard|safety|safeguard|moderation") | not)
        | .id' 2>/dev/null | sort
}

# Which OpenRouter models the generator leaves out. Two filters, both read from
# the shell environment first and then .env:
#   OR_PROVIDER_ALLOW  providers to register (empty = the built-in default list,
#                      `*` = every provider); applies to commercial, image,
#                      audio and free models alike. Local routes are not touched.
#   OR_MODEL_EXCLUDE   comma-separated items taken out of what is allowed: a
#                      `provider/` prefix, a `provider/model`, or a glob such as
#                      `liquid/lfm-*:free`. Empty = nothing excluded.
or_provider_allowed() {  # $1=provider
  local spec="${OR_PROVIDER_ALLOW-$(env_get OR_PROVIDER_ALLOW)}" p
  spec="${spec//[[:space:]]/}"
  spec="${spec:-$OR_PROVIDER_ALLOW_DEFAULT}"
  [[ "$spec" == '*' ]] && return 0
  local IFS=','
  for p in $spec; do [[ "$p" == "$1" ]] && return 0; done
  return 1
}

__or_filter_excluded() {  # $1=provider/model slug (free slugs keep their :free suffix)
  local slug="$1" spec="${OR_MODEL_EXCLUDE-$(env_get OR_MODEL_EXCLUDE)}" pat
  or_provider_allowed "${slug%%/*}" || return 0
  [[ -n "$spec" ]] || return 1
  local IFS=','
  for pat in $spec; do
    pat="${pat//[[:space:]]/}"
    [[ -n "$pat" ]] || continue
    [[ "$pat" == */ ]] && pat="${pat}*"
    # shellcheck disable=SC2053
    if [[ "$slug" == $pat || "${slug%:free}" == $pat ]]; then return 0; fi
  done
  return 1
}

# ── Run policy: which OpenRouter endpoints a model may run on ───────────────────
# All off by default (nothing changes). Shell environment first, then .env:
#   OR_OPEN_WEIGHT_POLICY  strict: an open-weight model (OR_OPEN_WEIGHT) runs only on
#                          endpoints that disclose their quantization AND are ZDR, any such
#                          endpoint (cheapest first with :floor); none → the model is left out
#   OR_OPEN_WEIGHT         glob list of the open-weight models (default below)
#   OR_PROVIDER_PIN        `pattern=Provider[|Provider]`, comma-separated: run on that provider
#                          only (no fallback); for an open-weight model it must be eligible
#   OR_ZDR_ONLY=1          every OpenRouter request carries provider.zdr; a model with no ZDR
#                          endpoint is left out
# Endpoint facts come from /api/v1/models/<id>/endpoints and /api/v1/endpoints/zdr (public).
OR_OPEN_WEIGHT_DEFAULT="openai/gpt-oss-*,google/gemma-*,meta-llama/*,nvidia/nemotron-*,deepseek/*,z-ai/*,moonshotai/*,qwen/qwen3-coder*,qwen/qwen3.5-*,qwen/qwen3.6-*,qwen/qwen3-vl-*,mistralai/mistral-nemo,mistralai/ministral-*,mistralai/mistral-small-*"

__or_setting() {  # $1=NAME: shell environment first, then .env
  local n="$1"
  if [[ -n "${!n+set}" ]]; then echo "${!n}"; else env_get "$n"; fi
}

__or_policy_active() {
  [[ "$(__or_setting OR_OPEN_WEIGHT_POLICY)" == strict \
     || "$(__or_setting OR_ZDR_ONLY)" == 1 || -n "$(__or_setting OR_PROVIDER_PIN)" ]]
}

# Per-run scratch directory for fetched endpoint data and decisions
__or_work() {
  if [[ -z "${KC_OR_WORK:-}" || ! -d "${KC_OR_WORK}" ]]; then
    KC_OR_WORK="$(mktemp -d -t kc-or-work.XXXXXX)"; export KC_OR_WORK
    [[ -z "$(trap -p EXIT)" ]] && trap 'rm -rf "${KC_OR_WORK:-}"' EXIT
  fi
  echo "$KC_OR_WORK"
}

# Public OpenRouter GET into a file, cached per run; rc 1 when unreachable
__or_fetch() {  # $1=api path  $2=cache file
  [[ -s "$2" ]] && return 0
  [[ -e "$2.fail" ]] && return 1
  mkdir -p "$(dirname "$2")"
  if curl -sf --max-time 20 "https://openrouter.ai/api/v1/$1" -o "$2" 2>/dev/null && [[ -s "$2" ]]; then
    return 0
  fi
  rm -f "$2"; : > "$2.fail"; return 1
}

or_zdr_file() { local w; w="$(__or_work)"; __or_fetch endpoints/zdr "$w/zdr.json" && echo "$w/zdr.json"; }
or_endpoints_file() {  # $1=slug (a :free suffix is dropped)
  local w slug="${1%:free}"; w="$(__or_work)"
  __or_fetch "models/${slug}/endpoints" "$w/ep/${slug//\//__}.json" && echo "$w/ep/${slug//\//__}.json"
}

# `pattern=Provider|Provider` entries of OR_PROVIDER_PIN that match $1; first match wins
__or_pin_for() {
  local slug="$1" spec pat val IFS=','
  spec="$(__or_setting OR_PROVIDER_PIN)"
  for pat in $spec; do
    pat="${pat#"${pat%%[![:space:]]*}"}"
    [[ "$pat" == *=* ]] || continue
    val="${pat#*=}"; pat="${pat%%=*}"; pat="${pat//[[:space:]]/}"
    # shellcheck disable=SC2053
    if [[ "$slug" == $pat || "${slug%:free}" == $pat ]]; then echo "${val%"${val##*[![:space:]]}"}"; return 0; fi
  done
  return 0
}

__or_is_open_weight() {
  local slug="${1%:free}" spec pat IFS=','
  [[ "$(__or_setting OR_OPEN_WEIGHT_POLICY)" == strict ]] || return 1
  spec="$(__or_setting OR_OPEN_WEIGHT)"; spec="${spec:-$OR_OPEN_WEIGHT_DEFAULT}"
  for pat in $spec; do
    pat="${pat//[[:space:]]/}"
    # shellcheck disable=SC2053
    [[ -n "$pat" && "$slug" == $pat ]] && return 0
  done
  return 1
}

# Decision JSON for one model (see scripts/or_policy.py), cached per run. Warnings print once.
or_decision() {  # $1=provider/model slug
  local slug="$1" w f ep zdr
  __or_policy_active || { echo '{"policy":"none","exclude":false,"provider":null}'; return 0; }
  w="$(__or_work)"; f="$w/dec/${slug//\//__}.json"
  if [[ ! -s "$f" ]]; then
    mkdir -p "$w/dec"
    local args=(decide --slug "$slug" --pin "$(__or_pin_for "$slug")")
    __or_is_open_weight "$slug" && args+=(--open-weight)
    [[ "$(__or_setting OR_ZDR_ONLY)" == 1 ]] && args+=(--zdr-only)
    local q; q="$(__or_setting OR_OPEN_WEIGHT_QUANTIZATIONS)"
    [[ -n "$q" ]] && args+=(--quantizations "$q")
    ep="$(or_endpoints_file "$slug" || true)"; zdr="$(or_zdr_file || true)"
    [[ -n "$ep" && -n "$zdr" ]] && args+=(--endpoints "$ep" --zdr "$zdr")
    python3 "${__PROJECT_DIR}/scripts/or_policy.py" "${args[@]}" > "$f.tmp" && mv "$f.tmp" "$f" \
      || { rm -f "$f.tmp"; echo '{"policy":"none","exclude":false,"provider":null}'; return 0; }
    jq -r '.warnings[]?' "$f" 2>/dev/null | while IFS= read -r line; do warn "$line"; done
    if [[ "$(jq -r .exclude "$f")" == true ]]; then
      echo "$slug ($(jq -r .reason "$f"))" >> "$w/excluded.txt"
    fi
  fi
  cat "$f"
}

# True when the run policy leaves the model out of the registry
or_policy_excluded() {
  [[ "$(or_decision "$1" | jq -r '.exclude // false' 2>/dev/null)" == true ]]
}

# Whether the OpenRouter name carries a model the policy or the filters leave out
or_model_excluded() {  # $1=provider/model slug (free slugs keep their :free suffix)
  __or_filter_excluded "$1" && return 0
  or_policy_excluded "$1"
}

# litellm_params lines for a model: `extra_body.provider` when the policy sets one
or_provider_params() {  # $1=slug
  local p; p="$(or_decision "$1" | jq -c '.provider // empty' 2>/dev/null)"
  [[ -n "$p" ]] && echo "      extra_body: {\"provider\": ${p}}"
  return 0
}

# Route suffix: a pinned model has its provider order, so no :floor sort on top
or_route_variant() {  # $1=slug  $2=default suffix
  if [[ -n "$(or_decision "$1" | jq -r '.pinned // empty' 2>/dev/null)" ]]; then return 0; fi
  echo -n "$2"
}

# model_info lines for a model: what KloudChat shows in the picker
or_policy_info() {  # $1=slug
  or_decision "$1" | jq -r '
    select(.policy != "none")
    | "      kchat_run_policy: " + (.policy | tojson),
      (if .pinned then "      kchat_provider_pinned: " + (.pinned | tojson) else empty end),
      (if .quantization then "      kchat_quantization: " + (.quantization | tojson) else empty end),
      (if .zdr != null then "      kchat_zdr: " + (.zdr | tostring) else empty end)' 2>/dev/null || true
}

# One slug per line: what the run policy takes out of the registered models
or_policy_excluded_list() {
  local m prov var
  has_openrouter || return 0
  __or_policy_active || return 0
  for prov in "${OR_PROVIDERS[@]}"; do
    var="${prov##*:}_MODELS[@]"
    for m in "${!var}"; do
      __or_filter_excluded "${prov%%:*}/$m" && continue
      or_policy_excluded "${prov%%:*}/$m" && echo "${prov%%:*}/$m"
    done
  done
  for m in "${OR_IMAGE_MODELS[@]}" "${OR_AUDIO_MODELS[@]}"; do
    __or_filter_excluded "$m" && continue
    or_policy_excluded "$m" && echo "$m"
  done
  return 0
}

# Every OpenRouter model the two filters remove, one slug per line
or_excluded_list() {
  local m prov var
  has_openrouter || return 0
  for prov in "${OR_PROVIDERS[@]}"; do
    var="${prov##*:}_MODELS[@]"
    for m in "${!var}"; do
      __or_filter_excluded "${prov%%:*}/$m" && echo "${prov%%:*}/$m"
    done
  done
  for m in "${OR_IMAGE_MODELS[@]}" "${OR_AUDIO_MODELS[@]}"; do
    __or_filter_excluded "$m" && echo "$m"
  done
  while IFS= read -r m; do
    [[ -n "$m" ]] && __or_filter_excluded "$m" && echo "$m"
  done < <(or_free_models_all)
  return 0
}

# Declared prices against the live catalogue; one line per mismatch, rc 1 if any
or_price_drift() {
  has_openrouter || { echo "no OPENROUTER_API_KEY — nothing to check" >&2; return 0; }
  command -v jq &>/dev/null || { echo "jq is required" >&2; return 0; }
  local key; key="$(env_get OPENROUTER_API_KEY)"
  local live; live="$(curl -sf --max-time 20 https://openrouter.ai/api/v1/models \
                       -H "Authorization: Bearer ${key}" 2>/dev/null)" || {
    echo "could not reach the OpenRouter catalogue" >&2; return 0; }

  local drift=0 slug m prov declared_in declared_out actual
  for prov in "${OR_PROVIDERS[@]}"; do
    local disp="${prov%%:*}" var="${prov##*:}_MODELS[@]"
    for m in "${!var}"; do
      slug="${disp}/${m}"
      declared_in="${MODEL_PRICE_IN_PM[$m]:-}"
      declared_out="${MODEL_PRICE_OUT_PM[$m]:-}"
      actual="$(jq -r --arg s "$slug" '
        .data[] | select(.id == $s)
        | "\((.pricing.prompt // "0" | tonumber) * 1000000)\t\((.pricing.completion // "0" | tonumber) * 1000000)"
      ' <<<"$live" 2>/dev/null | head -1)"
      if [[ -z "$actual" ]]; then
        echo "GONE      ${slug} — declared but not in the catalogue; it will 404 on first call"
        drift=1; continue
      fi
      local ain="${actual%%$'\t'*}" aout="${actual##*$'\t'}"
      if ! awk -v a="$ain" -v b="$declared_in" -v c="$aout" -v e="$declared_out" \
              'BEGIN { exit !(a-b < 0.0005 && b-a < 0.0005 && c-e < 0.0005 && e-c < 0.0005) }'; then
        printf 'DRIFT     %-40s declared %s/%s  actual %s/%s\n' \
               "$slug" "$declared_in" "$declared_out" "$ain" "$aout"
        drift=1
      fi
    done
  done
  # Twins and the STT fallback
  for slug in "${!OR_TWIN_PRICE_IN_PM[@]}"; do
    actual="$(jq -r --arg s "$slug" '
      .data[] | select(.id == $s)
      | "\((.pricing.prompt // "0" | tonumber) * 1000000)\t\((.pricing.completion // "0" | tonumber) * 1000000)"
    ' <<<"$live" 2>/dev/null | head -1)"
    if [[ -z "$actual" ]]; then
      echo "GONE      ${slug} — a local model's OpenRouter twin is not in the catalogue"
      drift=1; continue
    fi
    local tin="${actual%%$'\t'*}" tout="${actual##*$'\t'}"
    if ! awk -v a="$tin" -v b="${OR_TWIN_PRICE_IN_PM[$slug]}" \
             -v c="$tout" -v e="${OR_TWIN_PRICE_OUT_PM[$slug]}" \
            'BEGIN { exit !(a-b < 0.0005 && b-a < 0.0005 && c-e < 0.0005 && e-c < 0.0005) }'; then
      printf 'DRIFT     %-40s declared %s/%s  actual %s/%s\n' \
             "$slug" "${OR_TWIN_PRICE_IN_PM[$slug]}" "${OR_TWIN_PRICE_OUT_PM[$slug]}" "$tin" "$tout"
      drift=1
    fi
  done

  # image_output per image token; audio/audio_output per audio token. Image-only
  # models appear only in the image listing
  local id declared actual_out actual_in images
  images="$(curl -sf --max-time 20 'https://openrouter.ai/api/v1/models?output_modalities=image' \
             -H "Authorization: Bearer ${key}" 2>/dev/null)" || images='{"data":[]}'
  for id in "${OR_IMAGE_MODELS[@]}"; do
    declared="${MODEL_IMAGE_OUT_COST[$id]:-}"
    actual_out="$(jq -r --arg s "$id" '.data[] | select(.id == $s) | .pricing.image_output // empty' \
                  <<<"$images" | head -1)"
    if [[ -z "$actual_out" ]]; then
      echo "GONE      ${id} — image model not in the catalogue"; drift=1
    elif ! awk -v a="$actual_out" -v b="$declared" 'BEGIN { d=a-b; if (d<0) d=-d; exit !(d < 1e-9) }'; then
      printf 'DRIFT     %-40s image_output declared %s  actual %s\n' "$id" "$declared" "$actual_out"
      drift=1
    fi
  done
  for id in "${OR_AUDIO_MODELS[@]}"; do
    # Per-clip price is stated in the description, not in `pricing`
    if [[ -n "${MODEL_AUDIO_PER_CALL[$id]:-}" ]]; then
      declared="${MODEL_AUDIO_PER_CALL[$id]}"
      actual_out="$(jq -r --arg s "$id" '.data[] | select(.id == $s) | .description' <<<"$live" \
                    | grep -oE '\$[0-9]+\.?[0-9]* per clip' | head -1 | tr -d '$' | sed 's/ per clip//')"
      if [[ -z "$actual_out" ]]; then
        echo "UNCHECKED ${id} — per-clip price is not stated in the catalogue"
      elif ! awk -v a="$actual_out" -v b="$declared" 'BEGIN { d=a-b; if (d<0) d=-d; exit !(d < 1e-9) }'; then
        printf 'DRIFT     %-40s per clip declared %s  actual %s\n' "$id" "$declared" "$actual_out"
        drift=1
      fi
      continue
    fi
    declared="${MODEL_AUDIO_IN_PM[$id]:-}"
    actual_in="$(jq -r --arg s "$id" '.data[] | select(.id == $s) | ((.pricing.audio // "0" | tonumber) * 1000000)' <<<"$live" | head -1)"
    if ! awk -v a="$actual_in" -v b="$declared" 'BEGIN { d=a-b; if (d<0) d=-d; exit !(d < 0.0005) }'; then
      printf 'DRIFT     %-40s audio-in declared %s  actual %s\n' "$id" "$declared" "$actual_in"
      drift=1
    fi
    declared="${MODEL_AUDIO_OUT_PM[$id]:-}"
    actual_out="$(jq -r --arg s "$id" '.data[] | select(.id == $s) | ((.pricing.audio_output // "0" | tonumber) * 1000000)' <<<"$live" | head -1)"
    if ! awk -v a="$actual_out" -v b="$declared" 'BEGIN { d=a-b; if (d<0) d=-d; exit !(d < 0.0005) }'; then
      printf 'DRIFT     %-40s audio-out declared %s  actual %s\n' "$id" "$declared" "$actual_out"
      drift=1
    fi
  done

  (( drift )) && return 1 || { echo "every declared price matches the catalogue"; return 0; }
}

__vllm_normalize_url() {
  local u="$1"
  u="$(echo "$u" | sed 's/^[[:space:]]*//;s/[[:space:]]*$//;s|/$||')"
  echo "$u"
}
__vllm_node_models() {
  local raw="$1" url probe
  url="$(__vllm_normalize_url "$raw")"
  probe="${url//host.docker.internal/localhost}"
  curl -sf --max-time 5 "${probe}/v1/models" 2>/dev/null | jq -r '.data[]?.id' 2>/dev/null || true
}
# "<URL>\t<served-model-name>" per line for every reachable URL in the csv,
# deduplicated
vllm_union_node_models() {
  local urls_csv="$1"
  [[ -n "$urls_csv" ]] || return 0
  local IFS=, u nu tmp
  declare -A __seen_url=()
  for u in $urls_csv; do
    nu="$(__vllm_normalize_url "$u")"
    [[ -n "$nu" ]] || continue
    [[ -n "${__seen_url[$nu]:-}" ]] && continue
    __seen_url[$nu]=1
    tmp="$(__vllm_node_models "$u")"
    if [[ -z "$tmp" ]]; then
      warn "vllm unreachable: $nu"
      continue
    fi
    while IFS= read -r m; do
      [[ -n "$m" ]] && printf '%s\t%s\n' "$nu" "$m"
    done <<<"$tmp"
  done
}

# State of one URL:
#   0 = ready   (/v1/models 200 with at least one model)
#   1 = loading (HTTP answers but not ready: 503, empty model list)
#   2 = dead    (TCP refused / DNS failure)
__vllm_one_state() {
  local raw="$1" probe code body
  probe="$(__vllm_normalize_url "$raw")"
  probe="${probe//host.docker.internal/localhost}"
  code="$(curl -s -o /dev/null -w '%{http_code}' --connect-timeout 3 --max-time 8 \
            "${probe}/v1/models" 2>/dev/null || true)"
  case "$code" in
    200)
      body="$(curl -sf --max-time 5 "${probe}/v1/models" 2>/dev/null \
              | jq -r '.data[]?.id' 2>/dev/null)"
      [[ -n "$body" ]] && return 0
      return 1
      ;;
    "" | 000) return 2 ;;
    *)        return 1 ;;
  esac
}

# One URL's self-reported max_model_len, 3 tries 2s apart; rc 1 on failure
vllm_discover_max_len() {
  local raw="$1" probe len
  probe="$(__vllm_normalize_url "$raw")"
  probe="${probe//host.docker.internal/localhost}"
  for _ in 1 2 3; do
    len="$(curl -sf --max-time 5 "${probe}/v1/models" 2>/dev/null \
            | jq -r '.data[0].max_model_len // empty' 2>/dev/null)"
    if [[ -n "$len" && "$len" =~ ^[0-9]+$ ]]; then
      echo "$len"; return 0
    fi
    sleep 2
  done
  return 1
}

# What a running vLLM says about itself, one probe each (empty when unreachable or unsaid)
__vllm_probe_url() {
  local probe; probe="$(__vllm_normalize_url "$1")"
  echo "${probe//host.docker.internal/localhost}"
}
vllm_discover_engine() {  # "vLLM 0.33.0" from /version
  local v; v="$(curl -sf --max-time 5 "$(__vllm_probe_url "$1")/version" 2>/dev/null \
                | jq -r '.version // empty' 2>/dev/null)"
  [[ "$v" =~ ^[0-9A-Za-z][0-9A-Za-z._+-]*$ ]] && echo "vLLM ${v#v}"
  return 0
}
vllm_discover_kv_cache() {  # cache_dtype label of vllm:cache_config_info on /metrics; `auto` = unsaid.
  # Only the exact label: mamba_cache_dtype and mamba_ssm_cache_dtype share the line.
  local v; v="$(curl -sf --max-time 5 "$(__vllm_probe_url "$1")/metrics" 2>/dev/null \
                | grep -m1 '^vllm:cache_config_info' \
                | grep -o '[{,]cache_dtype="[^"]*"' | head -1 | sed 's/^[{,]cache_dtype="\(.*\)"$/\1/')"
  [[ "$v" =~ ^[0-9A-Za-z_]+$ && "$v" != auto ]] && echo "$v"
  return 0
}

# URL host → ssh target: the matching user@host from VLLM_NODES, else the host
__vllm_ssh_target() {
  local host="$1" csv entry
  csv="$(env_get VLLM_NODES 2>/dev/null)"
  local IFS=,
  for entry in $csv; do
    entry="${entry// /}"
    [[ -n "$entry" && "${entry#*@}" == "$host" ]] && { echo "$entry"; return 0; }
  done
  echo "$host"
}

# Container publishing the URL's port:
#   0 = Up (including "health: starting")
#   1 = missing / exited / restart loop
#   2 = unknown (ssh or docker query failed)
# Local hosts (localhost, loopback, own IP) are queried without ssh
__vllm_container_state() {
  local raw="$1" url hostport host port target out h islocal=0 ip
  url="$(__vllm_normalize_url "$raw")"
  hostport="${url#http://}"; hostport="${hostport%%/*}"
  host="${hostport%%:*}"; port="${hostport##*:}"
  host="${host//host.docker.internal/localhost}"
  h="${host#*@}"
  [[ -z "$h" || "$h" == localhost || "$h" == 127.0.0.1 || "$h" == ::1 ]] && islocal=1
  for ip in $(hostname -I 2>/dev/null); do [[ "$h" == "$ip" ]] && islocal=1; done
  if (( islocal )); then
    out="$(sudo -n docker ps --filter "publish=${port}" --format '{{.Status}}' 2>/dev/null)" || return 2
  else
    target="$(__vllm_ssh_target "$host")"
    out="$(ssh -o StrictHostKeyChecking=no -o ConnectTimeout=5 -o LogLevel=ERROR "$target" \
            "sudo -n docker ps --filter publish=${port} --format '{{.Status}}'" 2>/dev/null)" || return 2
  fi
  [[ -z "$out" ]] && return 1
  [[ "$out" == Up* ]] && return 0
  return 1
}

# Wait until every URL in the csv is ready; progress on stderr
#   vllm_wait_until_ready "$URL_CSV" "label" [timeout=600] [interval=10] [dead_thresh=3]
#   rc 0 = all ready; 2 = a container was missing/exited dead_thresh times in a
#   row; 3 = timeout. vLLM listens only after the weights load, so a TCP
#   refusal is judged by container state.
vllm_wait_until_ready() {
  local urls_csv="$1" label="$2"
  local timeout="${3:-600}" interval="${4:-10}" dead_thresh="${5:-3}"
  [[ -n "$urls_csv" ]] || return 0
  local deadline; deadline=$(( $(date +%s) + timeout ))
  local IFS=, u nu state cstate
  declare -A __seen=()
  declare -A __dead=()
  for u in $urls_csv; do
    nu="$(__vllm_normalize_url "$u")"
    [[ -n "$nu" ]] || continue
    [[ -n "${__seen[$nu]:-}" ]] && continue
    __seen[$nu]=1
    while :; do
      state=0; __vllm_one_state "$u" || state=$?
      if (( state == 0 )); then
        echo "  [ready] ${label}: ${nu}" >&2
        break
      fi
      if (( state == 2 )); then
        cstate=0; __vllm_container_state "$u" || cstate=$?
        if (( cstate == 0 )); then
          __dead[$nu]=0
          echo "  [load]  ${label}: ${nu} container Up — loading model (port not yet open)" >&2
        elif (( cstate == 1 )); then
          __dead[$nu]=$(( ${__dead[$nu]:-0} + 1 ))
          if (( __dead[$nu] >= dead_thresh )); then
            echo "  [fail]  ${label}: ${nu} container not started/exited ${dead_thresh} times in a row — vLLM down (check docker logs on the GPU node)" >&2
            return 2
          fi
          echo "  [wait]  ${label}: ${nu} container missing/exited (${__dead[$nu]}/${dead_thresh})" >&2
        else
          echo "  [load]  ${label}: ${nu} container state query failed — keep waiting" >&2
        fi
      else
        __dead[$nu]=0
        echo "  [load]  ${label}: ${nu} loading model…" >&2
      fi
      if (( $(date +%s) >= deadline )); then
        echo "  [fail]  ${label}: ${nu} not ready within ${timeout}s" >&2
        return 3
      fi
      sleep "$interval"
    done
  done
  return 0
}

# scheduler/models.yaml, read once through the scheduler: CATALOG_IDS in file
# order, CATALOG_<field>[id] per model
CATALOG_IDS=()
declare -gA CATALOG_API=() CATALOG_PREFIX=() CATALOG_OR=() CATALOG_TIMEOUT=() \
  CATALOG_MAX_INFLIGHT=() CATALOG_PRESENCE=() CATALOG_TEMPERATURE=()
load_model_catalog() {
  (( ${#CATALOG_IDS[@]} )) && return 0
  local rows id api prefix or tmo cap pres temp
  rows="$(cd "$__PROJECT_DIR" && python3 -m scheduler catalog)" \
    || { err "scheduler/models.yaml unreadable — python3 with PyYAML required"; return 1; }
  while IFS=$'\t' read -r id api prefix or tmo cap pres temp; do
    [[ "$pres" == - ]] && pres=""
    [[ "$temp" == - ]] && temp=""
    [[ -n "$id" ]] || continue
    CATALOG_IDS+=("$id")
    CATALOG_API[$id]="$api"
    CATALOG_PREFIX[$id]="$prefix"
    CATALOG_OR[$id]="${or/#-/}"
    CATALOG_TIMEOUT[$id]="${tmo/#-/}"
    CATALOG_MAX_INFLIGHT[$id]="$cap"
    CATALOG_PRESENCE[$id]="$pres"
    CATALOG_TEMPERATURE[$id]="$temp"
  done <<<"$rows"
}

# Local models and where they are served, from VLLM_ENDPOINTS (hand-run; "id=url"
# as given, a bare URL by the names its /v1/models lists), VLLM_PLACED (the
# scheduler's, "id=url").
# LOCAL_IDS: catalogue order, then models only the endpoint lists name;
# LOCAL_URLS[id] a CSV; LOCAL_SERVED["id url"] vLLM's name for it on that server.
LOCAL_IDS=()
declare -gA LOCAL_URLS=() LOCAL_SERVED=()
__local_add() {  # $1=id  $2=url  $3=served name
  local id="$1" url
  url="$(__vllm_normalize_url "$2")"
  [[ -n "$id" && -n "$url" ]] || return 0
  [[ -n "${LOCAL_URLS[$id]+set}" ]] || LOCAL_URLS[$id]=""
  LOCAL_SERVED["$id $url"]="${3:-local/$id}"
  [[ ",${LOCAL_URLS[$id]}," == *",${url},"* ]] && return 0
  LOCAL_URLS[$id]="${LOCAL_URLS[$id]:+${LOCAL_URLS[$id]},}${url}"
}
# vLLM's name for model $1 on server $2
local_served_at() { echo "${LOCAL_SERVED["$1 $(__vllm_normalize_url "$2")"]:-local/$1}"; }
# Model id for a served name: without local/, else its last path segment
__local_id() { local n="${1#local/}"; echo "${n##*/}"; }
resolve_local_models() {
  load_model_catalog || return 1
  LOCAL_IDS=(); LOCAL_URLS=(); LOCAL_SERVED=()
  local item id url served names extra=() IFS=, listed
  listed="$(env_get VLLM_ENDPOINTS),$(env_get VLLM_PLACED)"
  for item in $listed; do
    item="${item//[[:space:]]/}"
    [[ -n "$item" ]] || continue
    if [[ "$item" == *=* ]]; then
      id="${item%%=*}"; id="${id#local/}"; url="${item#*=}"
      # The listed name when the server answers; local/<id> otherwise
      served="$(IFS=$'\n'; for n in $(__vllm_node_models "$url"); do
                  [[ "$(__local_id "$n")" == "$id" || "$(__local_id "${n,,}")" == "${id,,}" ]] && { echo "$n"; break; }
                done)"
      __local_add "$id" "$url" "$served"
      extra+=("$id")
    else
      names="$(__vllm_node_models "$item")"
      [[ -n "$names" ]] || { warn "vllm unreachable, skipped: ${item} (name it as id=url to register it anyway)"; continue; }
      while IFS= read -r served; do
        [[ -n "$served" ]] || continue
        id="$(__local_id "$served")"
        __local_add "$id" "$item" "$served"
        extra+=("$id")
      done <<<"$names"
    fi
  done
  for id in "${CATALOG_IDS[@]}"; do
    [[ -n "${LOCAL_URLS[$id]:-}" ]] && LOCAL_IDS+=("$id")
  done
  for id in "${extra[@]}"; do
    [[ -n "${CATALOG_API[$id]:-}" || " ${LOCAL_IDS[*]} " == *" ${id} "* ]] && continue
    LOCAL_IDS+=("$id")
  done
  return 0
}

# Route a local model registers under: its catalogue api, chat when unlisted
local_api() { echo "${CATALOG_API[$1]:-chat}"; }

# LiteLLM team allowlist: every chat model gen-litellm-config.sh registers
litellm_chat_models_csv() {
  resolve_local_models >/dev/null || return 1
  local out=() m
  if has_openrouter; then
    local prov var
    for prov in "${OR_PROVIDERS[@]}"; do
      var="${prov##*:}_MODELS[@]"
      for m in "${!var}"; do
        or_model_excluded "${prov%%:*}/$m" || out+=("${prov%%:*}/$m")
      done
    done
  fi
  # Same routes as gen-litellm-config.sh emit_brain
  for m in "${CATALOG_IDS[@]}"; do
    [[ "${CATALOG_API[$m]}" == chat ]] || continue
    if [[ -n "${LOCAL_URLS[$m]:-}" ]]; then
      out+=("local/$m" "strict-local/$m")
    elif has_openrouter && [[ -n "${CATALOG_OR[$m]}" ]]; then
      out+=("${CATALOG_OR[$m]}")
    fi
  done
  for m in "${LOCAL_IDS[@]}"; do
    [[ -z "${CATALOG_API[$m]:-}" ]] && out+=("local/$m" "strict-local/$m")
  done
  local IFS=,
  echo "${out[*]:-}"
}

LITELLM_MASTER_KEY="${LITELLM_MASTER_KEY:-$(env_get LITELLM_MASTER_KEY)}"
# LITELLM_URL: shell env, then .env, then the gateway; container hostnames
# rewritten for host-side calls
LITELLM_URL="${LITELLM_URL:-$(env_get LITELLM_URL)}"
__gateway_port="$(env_get GATEWAY_PORT)"
LITELLM_URL="${LITELLM_URL:-http://localhost:${__gateway_port:-8080}/litellm}"
LITELLM_URL="${LITELLM_URL//host.docker.internal/localhost}"
LITELLM_URL="${LITELLM_URL//\/\/litellm:/\/\/localhost:}"
DATA_DIR="${__PROJECT_DIR}/data/ledger"

__litellm_call() {
  local method="$1" endpoint="$2" payload="${3:-}"
  local args=(-s -w "\n%{http_code}" -X "$method" "${LITELLM_URL}${endpoint}"
              -H "Authorization: Bearer ${LITELLM_MASTER_KEY}")
  [[ -n "$payload" ]] && args+=(-H "Content-Type: application/json" -d "$payload")
  local resp; resp=$(curl "${args[@]}")
  local code; code=$(echo "$resp" | tail -1)
  # Body: every line but the status code (portable)
  local body; body=$(echo "$resp" | sed '$d')
  if (( code < 200 || code >= 300 )); then
    echo "ERROR [HTTP $code]: $body" >&2; return 1
  fi
  echo "$body"
}

litellm_post() { __litellm_call POST "$1" "$2"; }
litellm_get()  { __litellm_call GET  "$1"; }

team_id_by_alias() {
  mkdir -p "$DATA_DIR"
  local alias="$1" cache="${DATA_DIR}/teams.json" id
  if [[ -f "$cache" ]]; then
    id=$(jq -r --arg a "$alias" '.[] | select(.team_alias == $a) | .team_id' "$cache" 2>/dev/null || true)
    [[ -n "$id" ]] && { echo "$id"; return 0; }
  fi
  litellm_get "/team/list" | jq -r --arg a "$alias" '.[] | select(.team_alias == $a) | .team_id' 2>/dev/null || true
}

# ───────────────────────── multi-node dispatch ─────────────────────────

# Repository path on a remote node, relative to the login user's $HOME
KLOUDCHAT_REMOTE_DIR="${KLOUDCHAT_REMOTE_DIR:-KloudChat-LLM}"

# Whether a VLLM_NODES target is this host: localhost, loopback, $HOSTNAME, the
# short hostname, a local IPv4 address, or a name resolving to one
is_local_host() {
  local target="${1#*@}"
  [[ -z "$target" ]] && return 1
  case "$target" in localhost|127.0.0.1|::1) return 0 ;; esac
  local self_short="${HOSTNAME%%.*}"
  [[ "$target" == "$HOSTNAME" || "$target" == "$self_short" ]] && return 0
  local local_ips ip
  local_ips="$(hostname -I 2>/dev/null || true)"
  for ip in $local_ips; do
    [[ "$target" == "$ip" ]] && return 0
  done
  local target_ip
  target_ip="$(getent hosts "$target" 2>/dev/null | awk '{print $1; exit}')"
  if [[ -n "$target_ip" ]]; then
    for ip in $local_ips; do
      [[ "$target_ip" == "$ip" ]] && return 0
    done
  fi
  return 1
}

# CSV → one item per line, trimmed of whitespace and quotes
csv_split() {
  local IFS=, s
  for s in $1; do
    s="$(echo "$s" | sed 's/^[[:space:]]*//;s/[[:space:]]*$//;s/^"//;s/"$//')"
    [[ -n "$s" ]] && echo "$s"
  done
}

# Repo to a node, minus runtime data and .env (per node; rsync_push_env_if_absent)
rsync_push() {
  local host="$1"
  echo "  → rsync to ${host}:${KLOUDCHAT_REMOTE_DIR}/"
  rsync -az --delete \
    --exclude='.git/' \
    --exclude='.env' \
    --exclude='.env.bak.*' \
    --exclude='data/' \
    --exclude='whisper/.cache/' \
    --exclude='services/searxng/settings.yml' \
    --exclude='__pycache__/' \
    --exclude='*.pyc' \
    --exclude='node_modules/' \
    --exclude='.venv/' \
    --exclude='test-results/' \
    --exclude='playwright-report/' \
    --rsync-path="mkdir -p '${KLOUDCHAT_REMOTE_DIR}' && rsync" \
    "${__PROJECT_DIR}/" "${host}:${KLOUDCHAT_REMOTE_DIR}/"
}

# Keys a GPU node reads; the orchestrator's other secrets stay off the node
NODE_ENV_KEY_RE='^(VLLM_|HF_|KLOUDCHAT_IMAGE_|WHISPER_|GPU_EXPORTER_)[A-Z0-9_]*='

# Seed a node's .env only when absent; the scheduler owns it afterwards
rsync_push_env_if_absent() {
  local host="$1" tmp rc=0
  if ssh_run "$host" "test -f .env" 2>/dev/null; then
    echo "  → ${host}: .env exists, keeping node-local overrides"
    return 0
  fi
  echo "  → ${host}: seeding .env (none present; GPU keys only)"
  tmp="$(umask 077; mktemp)"
  grep -E "$NODE_ENV_KEY_RE" "${__PROJECT_DIR}/.env" > "$tmp" || true
  rsync -az --chmod=F600 "$tmp" "${host}:${KLOUDCHAT_REMOTE_DIR}/.env" || rc=$?
  rm -f "$tmp"
  return "$rc"
}

# ssh, cd into the repo, run; -n leaves the caller's stdin alone (while-read loops)
ssh_run() {
  local host="$1"; shift
  ssh -n -o StrictHostKeyChecking=accept-new -o ServerAliveInterval=30 "$host" \
    "set -e; cd '${KLOUDCHAT_REMOTE_DIR}' && $*"
}
