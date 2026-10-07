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
OPENAI_MODELS=(gpt-6-astra gpt-6.1-sol gpt-6-luna gpt-5.4-nano gpt-5.3-codex)
ANTHROPIC_MODELS=(claude-fable-5.1 claude-opus-5.5 claude-sonnet-5.5 claude-haiku-4.5)
GOOGLE_MODELS=(gemini-3.1-pro-preview gemini-3.8-flash gemini-3.5-flash-lite)
XAI_MODELS=(grok-4.7)
PERPLEXITY_MODELS=(sonar sonar-pro)
# Open-weight tier
TENCENT_MODELS=(hy4-preview)
DEEPSEEK_MODELS=(deepseek-v4-pro-0813 deepseek-v4.1-flash)
ZAI_MODELS=(glm-5.3 glm-5.3-flash)
XIAOMI_MODELS=(mimo-v2.6-flash)
MOONSHOTAI_MODELS=(kimi-k3)
# Qwen's hosted tier (not the local checkpoints)
QWEN_MODELS=(qwen3.8-max-0902 qwen3.8-flash qwen3-coder-plus)
MINIMAX_MODELS=(minimax-m3)
# "<OpenRouter provider>:<array prefix>" for each *_MODELS array above
OR_PROVIDERS=(openai:OPENAI anthropic:ANTHROPIC google:GOOGLE x-ai:XAI
              perplexity:PERPLEXITY tencent:TENCENT deepseek:DEEPSEEK z-ai:ZAI
              xiaomi:XIAOMI moonshotai:MOONSHOTAI qwen:QWEN minimax:MINIMAX)

# Image generation, cheapest first (picker default)
OR_IMAGE_MODELS=(
  openai/gpt-5-image-mini
  google/gemini-2.5-flash-image
  openai/gpt-5-image
  google/gemini-3-pro-image
)
# USD per image_output token; one picture is ~1290 tokens
declare -A MODEL_IMAGE_OUT_COST=(
  [openai/gpt-5-image-mini]=0.000008
  [google/gemini-2.5-flash-image]=0.00003
  [openai/gpt-5-image]=0.00004
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
  [openai/gpt-5-image-mini]=2.50
  [google/gemini-2.5-flash-image]=0.30
  [openai/gpt-5-image]=10.00
  [google/gemini-3-pro-image]=2.00
)

# RAG embedding fallback, registered with OPENAI_API_KEY
OPENAI_EMBED_CATALOG=(text-embedding-3-small)

# vLLM catalogue: download alias → HF repo
declare -A VLLM_MODELS=(
  # Chat, NVFP4
  [qwen3.8-27b-nvfp4]="unsloth/Qwen3.8-27B-NVFP4"
  # Quality and judging, MoE 10B active, NVFP4; a GB10 node to itself
  [qwen3.5-122b-nvfp4]="txn545/Qwen3.5-122B-A10B-NVFP4"
  # Coding, FP8
  [qwen3-coder-next]="Qwen/Qwen3-Coder-Next-FP8"
  # Retrieval embeddings and reranking, BF16
  [bge-m3]="BAAI/bge-m3"
  [bge-reranker-v2-m3]="BAAI/bge-reranker-v2-m3"
  # Transcription, FP16
  [whisper-large-v3]="openai/whisper-large-v3"
)
: "${VLLM_MODELS_ROOT:=/var/lib/vllm/models}"

# Checkpoint size on disk (GiB)
declare -A VLLM_MODEL_WEIGHT_GB=(
  [qwen3.8-27b-nvfp4]=22
  [qwen3.5-122b-nvfp4]=72
  [qwen3-coder-next]=75
  [bge-m3]=3
  [bge-reranker-v2-m3]=3
  [whisper-large-v3]=4
)
# GiB over the weights: activations plus KV for one request
VLLM_RUNTIME_HEADROOM_GB=6

# Download set for a node without an explicit model list
VLLM_PREFERRED_MODELS=(qwen3.8-27b-nvfp4)

# Usable-VRAM floor (GiB): the RTX 5090
VLLM_MIN_USABLE_VRAM_GB=32

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
  local need=$(( weight + VLLM_RUNTIME_HEADROOM_GB ))
  if (( weight > 0 && vram > 0 && vram < need )); then
    echo "needs ~${need}GiB (weights ${weight} + runtime ${VLLM_RUNTIME_HEADROOM_GB}) but the card has ${vram}GiB"
    return 1
  fi
  return 0
}

# vLLM base image per architecture; nightly, as the stable tags lack the
# hybrid-attention architectures. install-vllm.sh pins the resolved digest
# per node as VLLM_BASE_DIGEST.
VLLM_IMAGE_ARM64="vllm/vllm-openai:nightly-aarch64"
VLLM_IMAGE_AMD64="vllm/vllm-openai:cu129-nightly"

vllm_default_image() {
  case "$(detect_arch)" in
    arm64) echo "$VLLM_IMAGE_ARM64" ;;
    amd64) echo "$VLLM_IMAGE_AMD64" ;;
    *)     echo "" ;;
  esac
}

# Registry digest of a local image; empty for a locally built tag
image_base_digest() {
  docker image inspect "$1" --format '{{if .RepoDigests}}{{index .RepoDigests 0}}{{end}}' 2>/dev/null
}

# Declared prices, USD per 1M tokens; or_refresh_prices overlays the live
# catalogue. Local models are 0.
declare -A MODEL_PRICE_IN_PM=(
  [gpt-6-astra]=10       [gpt-6.1-sol]=2        [gpt-6-luna]=0.10      [gpt-5.4-nano]=0.20
  [gpt-5.3-codex]=1.75
  [claude-fable-5.1]=10  [claude-opus-5.5]=4    [claude-sonnet-5.5]=2  [claude-haiku-4.5]=1
  [gemini-3.1-pro-preview]=2   [gemini-3.8-flash]=0.75   [gemini-3.5-flash-lite]=0.30
  [grok-4.7]=2           [sonar]=1.00              [sonar-pro]=3.00
  [hy4-preview]=0.751    [deepseek-v4-pro-0813]=0.22  [deepseek-v4.1-flash]=0.30
  [glm-5.3]=1.40         [glm-5.3-flash]=0.15      [mimo-v2.6-flash]=0.14    [kimi-k3]=0.99
  [qwen3.8-max-0902]=2.00  [qwen3.8-flash]=0.15    [qwen3-coder-plus]=0.65
  [minimax-m3]=0.30
  [qwen3.8-27b]=0  [qwen3.5-122b]=0  [qwen3-coder-next]=0
  [text-embedding-3-small]=0.02
)
declare -A MODEL_PRICE_OUT_PM=(
  [gpt-6-astra]=50       [gpt-6.1-sol]=10       [gpt-6-luna]=0.50      [gpt-5.4-nano]=1.25
  [gpt-5.3-codex]=14.00
  [claude-fable-5.1]=50  [claude-opus-5.5]=20   [claude-sonnet-5.5]=10 [claude-haiku-4.5]=5
  [gemini-3.1-pro-preview]=12  [gemini-3.8-flash]=3.75   [gemini-3.5-flash-lite]=2.50
  [grok-4.7]=6           [sonar]=1.00              [sonar-pro]=15.00
  [hy4-preview]=2.25     [deepseek-v4-pro-0813]=4.20  [deepseek-v4.1-flash]=1.20
  [glm-5.3]=4.40         [glm-5.3-flash]=0.50      [mimo-v2.6-flash]=0.28    [kimi-k3]=13.00
  [qwen3.8-max-0902]=6.00  [qwen3.8-flash]=0.47    [qwen3-coder-plus]=3.25
  [minimax-m3]=1.20
  [qwen3.8-27b]=0  [qwen3.5-122b]=0  [qwen3-coder-next]=0
)

# OpenRouter twins of local models and the STT fallback, USD per 1M tokens,
# keyed by slug; or_price prefers the live figure
declare -A OR_TWIN_PRICE_IN_PM=(
  [qwen/qwen3.8-27b]=0.42  [qwen/qwen3.5-122b-a10b]=0.26
  [qwen/qwen3-coder-next]=0.12
  [mistralai/voxtral-small-24b-2507]=0.10
)
declare -A OR_TWIN_PRICE_OUT_PM=(
  [qwen/qwen3.8-27b]=3.00  [qwen/qwen3.5-122b-a10b]=2.08
  [qwen/qwen3-coder-next]=0.80
  [mistralai/voxtral-small-24b-2507]=0.30
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

  # image_output per image token; audio/audio_output per audio token
  local id declared actual_out actual_in
  for id in "${OR_IMAGE_MODELS[@]}"; do
    declared="${MODEL_IMAGE_OUT_COST[$id]:-}"
    actual_out="$(jq -r --arg s "$id" '.data[] | select(.id == $s) | .pricing.image_output // empty' <<<"$live" | head -1)"
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

# Whether any URL in the csv answers /v1/models with a model
vllm_any_url_alive() {
  local csv="$1" u
  [[ -n "$csv" ]] || return 1
  local IFS=,
  for u in $csv; do
    [[ -n "$u" ]] || continue
    [[ -n "$(__vllm_node_models "$u")" ]] && return 0
  done
  return 1
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

# URL host → ssh target: the matching user@host from NODES_VLLM, else the host
__vllm_ssh_target() {
  local host="$1" csv entry
  csv="$(env_get NODES_VLLM 2>/dev/null)"
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

# LiteLLM team allowlist: every chat model gen-litellm-config.sh registers
litellm_chat_models_csv() {
  local vllm_chat_url; vllm_chat_url="$(env_get VLLM_QWEN27B_URL 2>/dev/null || true)"
  local vllm_judge_url; vllm_judge_url="$(env_get VLLM_QWEN122B_URL 2>/dev/null || true)"
  local vllm_coder_url; vllm_coder_url="$(env_get VLLM_CODERNEXT_URL 2>/dev/null || true)"
  local out=() m
  if has_openrouter; then
    for m in "${OPENAI_MODELS[@]}";     do out+=("openai/$m");     done
    for m in "${ANTHROPIC_MODELS[@]}";  do out+=("anthropic/$m");  done
    for m in "${GOOGLE_MODELS[@]}";     do out+=("google/$m");     done
    for m in "${DEEPSEEK_MODELS[@]}";   do out+=("deepseek/$m");   done
    for m in "${XAI_MODELS[@]}";        do out+=("x-ai/$m");       done
    for m in "${PERPLEXITY_MODELS[@]}"; do out+=("perplexity/$m"); done
    for m in "${TENCENT_MODELS[@]}";    do out+=("tencent/$m");    done
    for m in "${ZAI_MODELS[@]}";        do out+=("z-ai/$m");       done
    for m in "${XIAOMI_MODELS[@]}";     do out+=("xiaomi/$m");     done
    for m in "${MOONSHOTAI_MODELS[@]}"; do out+=("moonshotai/$m"); done
    for m in "${QWEN_MODELS[@]}";       do out+=("qwen/$m");       done
    for m in "${MINIMAX_MODELS[@]}";    do out+=("minimax/$m");    done
  fi
  # Same routes as gen-litellm-config.sh emit_brain
  if [[ -n "$vllm_chat_url" ]]; then
    out+=("local/qwen3.8-27b" "strict-local/qwen3.8-27b")
  elif has_openrouter; then
    out+=("qwen/qwen3.8-27b")
  fi
  if [[ -n "$vllm_judge_url" ]]; then
    out+=("local/qwen3.5-122b" "strict-local/qwen3.5-122b")
  elif has_openrouter; then
    out+=("qwen/qwen3.5-122b-a10b")
  fi
  if [[ -n "$vllm_coder_url" ]]; then
    out+=("local/qwen3-coder-next" "strict-local/qwen3-coder-next")
  elif has_openrouter; then
    out+=("qwen/qwen3-coder-next")
  fi
  if has_openrouter; then
    for m in "${OPENAI_EMBED_CATALOG[@]}"; do out+=("$m"); done
  fi
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

# Whether a NODES_VLLM target is this host: localhost, loopback, $HOSTNAME, the
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

# Seed a node's .env only when absent; the scheduler owns it afterwards
rsync_push_env_if_absent() {
  local host="$1"
  if ssh_run "$host" "test -f .env" 2>/dev/null; then
    echo "  → ${host}: .env exists, keeping node-local overrides"
    return 0
  fi
  echo "  → ${host}: seeding .env (none present)"
  rsync_push_file "$host" ".env"
}

rsync_push_file() {
  local host="$1" path="$2"
  echo "  → rsync ${path} → ${host}:${KLOUDCHAT_REMOTE_DIR}/${path}"
  rsync -az "${__PROJECT_DIR}/${path}" "${host}:${KLOUDCHAT_REMOTE_DIR}/${path}"
}

# ssh, cd into the repo, run; -n leaves the caller's stdin alone (while-read loops)
ssh_run() {
  local host="$1"; shift
  ssh -n -o StrictHostKeyChecking=accept-new -o ServerAliveInterval=30 "$host" \
    "set -e; cd '${KLOUDCHAT_REMOTE_DIR}' && $*"
}
