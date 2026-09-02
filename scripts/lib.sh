#!/usr/bin/env bash

[[ -n "${__KC_LIB_SH:-}" ]] && return 0
__KC_LIB_SH=1

__R='\033[0;31m'; __G='\033[0;32m'; __Y='\033[1;33m'
__B='\033[1;34m'; __N='\033[0m'

hdr()  { echo; echo -e "${__B}━━━ $* ━━━${__N}"; }
# Diagnostics on stderr: the config generators build YAML through `$(...)`.
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
    # Trailing newline guard.
    [[ -s "$file" && -n "$(tail -c 1 "$file")" ]] && printf '\n' >> "$file"
    printf '%s=%s\n' "$key" "$val" >> "$file"
  fi
}

# Write-permission check before a tmp+mv regeneration (root-owned files from a
# sudo run or a container write).
#   $1 = target file
#   $2 = need_read (default 1; 0 for a full regeneration)
assert_regen_writable() {
  local f="$1" need_read="${2:-1}" d owner me; d="$(dirname "$f")"; me="$(id -un)"
  if [[ "$need_read" == 1 && -e "$f" && ! -r "$f" ]]; then
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

# ~/.local/bin on PATH: uv and the HuggingFace CLI live there, and a
# non-interactive ssh session does not include it.
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
    *"RTX 4090"*) echo rtx4090 ;;
    *)            echo nvidia-other ;;
  esac
}

get_free_disk_gb() {
  df -BG "${1:-.}" 2>/dev/null | awk 'NR==2 {gsub("G",""); print $4; exit}'
}

# Compute capability of GPU 0 ("12.0"). Empty when the driver cannot answer;
# callers fall back to the GPU class.
gpu_compute_cap() {
  has_nvidia_gpu || { echo ""; return; }
  nvidia-smi --query-gpu=compute_cap --format=csv,noheader 2>/dev/null \
    | head -1 | tr -dc '0-9.'
}

# OS share on a unified-memory node. Mirrors
# scheduler/inventory.py::_UNIFIED_RESERVE_BYTES; the two must agree.
UNIFIED_RESERVE_GB=12

# Usable GPU memory in GiB, 0 without a GPU. Unified-memory cards report [N/A]
# to nvidia-smi and fall back to MemTotal minus the OS share.
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

# Weight dtype support. NVFP4 needs Blackwell (cc >= 10.0), FP8 needs Ada/Hopper
# (cc >= 8.9). Unknown cards pass: the engine is the better judge.
gpu_supports_quant() {
  local quant="$1" cap need
  cap="$(gpu_compute_cap)"
  if [[ -z "$cap" ]]; then
    case "$(detect_gpu_class)" in
      gb10|pro6000|pro5000|rtx5090) cap=12.0 ;;
      rtx4090)                      cap=8.9 ;;
      *)                            return 0 ;;
    esac
  fi
  case "$quant" in
    nvfp4)     need=10.0 ;;
    fp8)       need=8.9 ;;
    awq|gptq)  need=7.5 ;;
    *)         need=8.0 ;;
  esac
  awk -v c="$cap" -v n="$need" 'BEGIN { exit !(c + 0 >= n + 0) }'
}

# Commercial catalogue, served through OpenRouter; skipped without an OR key.
# One array per provider; gen-litellm-config.sh registers from these.
# Ids and prices: verify against https://openrouter.ai/api/v1/models.
# Order within a provider = picker order (flagship first).
OPENAI_MODELS=(gpt-5.6-sol gpt-5.6-terra gpt-5.6-luna gpt-5-nano gpt-5.3-codex)
ANTHROPIC_MODELS=(claude-fable-5 claude-opus-5 claude-sonnet-5 claude-haiku-4.5)
GOOGLE_MODELS=(gemini-3.1-pro-preview gemini-3.7-flash gemini-3.1-flash-lite)
XAI_MODELS=(grok-4.6)
PERPLEXITY_MODELS=(sonar sonar-pro)
# Open-weight tier (295B to 2.8T). Display name = OpenRouter id.
TENCENT_MODELS=(hy3)
DEEPSEEK_MODELS=(deepseek-v4-pro deepseek-v4-flash)
ZAI_MODELS=(glm-5.3)
XIAOMI_MODELS=(mimo-v2.5)
MOONSHOTAI_MODELS=(kimi-k3)
# Qwen's hosted tier, distinct from the locally served qwen3.x checkpoints.
QWEN_MODELS=(qwen3.8-max qwen3.7-flash qwen3-coder-plus)
MINIMAX_MODELS=(minimax-m3)

# Image generation, cheapest first (picker default). Prices per `image_output`
# token; one picture is ~1290 tokens.
OR_IMAGE_MODELS=(
  openai/gpt-5-image-mini
  google/gemini-2.5-flash-image
  openai/gpt-5-image
  google/gemini-3-pro-image
)
declare -A MODEL_IMAGE_OUT_COST=(
  [openai/gpt-5-image-mini]=0.000008
  [google/gemini-2.5-flash-image]=0.00003
  [openai/gpt-5-image]=0.00004
  [google/gemini-3-pro-image]=0.00012
)
# Audio generation through chat/completions (streaming required). gpt-audio:
# speech, per token; lyria: music, per clip.
OR_AUDIO_MODELS=(
  openai/gpt-audio-mini
  openai/gpt-audio
  google/lyria-3-clip-preview
)
# Audio-token rates (OpenRouter prices text and audio separately; the audio
# rate over-charges the text portion, the safe direction).
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
# Per-clip billing. The catalogue states this figure in the model description
# rather than in `pricing`; `or_price_drift` reads it from there.
declare -A MODEL_AUDIO_PER_CALL=(
  [google/lyria-3-clip-preview]=0.04
)

declare -A MODEL_IMAGE_IN_PM=(
  [openai/gpt-5-image-mini]=2.50
  [google/gemini-2.5-flash-image]=0.30
  [openai/gpt-5-image]=10.00
  [google/gemini-3-pro-image]=2.00
)

# RAG embedding fallback, registered when an OpenAI-compatible key is present.
OPENAI_EMBED_CATALOG=(text-embedding-3-small)

# vLLM catalogue: download alias → HF repo.
declare -A VLLM_MODELS=(
  # Chat, NVFP4 (default)
  [qwen3.6-35b-nvfp4]="unsloth/Qwen3.6-35B-A3B-NVFP4"
  # Chat, FP8 (engines below vLLM 0.24)
  [qwen3.6-35b]="Qwen/Qwen3.6-35B-A3B"
  # Chat, AWQ int4 (cards without FP4, cc >= 7.5). Point VLLM_QWEN35B_DIR at it;
  # the served entry in models.yaml is unchanged.
  [qwen3.6-35b-awq]="QuantTrio/Qwen3.6-35B-A3B-AWQ"
  # Top chat: 10B active, 78 GiB NVFP4, needs a card to itself
  [qwen3.5-122b-a10b]="Qwen/Qwen3.5-122B-A10B-NVFP4"
  # Coding, FP8
  [qwen3-coder-next]="Qwen/Qwen3-Coder-Next-FP8"
  [qwen3-coder-30b]="Qwen/Qwen3-Coder-30B-A3B-Instruct-FP8"
  # Dense chat, NVFP4
  [qwen3.6-27b]="Qwen/Qwen3.6-27B"
  # Retrieval embeddings and reranking, BF16
  [bge-m3]="BAAI/bge-m3"
  [bge-reranker-v2-m3]="BAAI/bge-reranker-v2-m3"
  # Transcription, FP16 (any card, any architecture)
  [whisper-large-v3]="openai/whisper-large-v3"
)
: "${VLLM_MODELS_ROOT:=/var/lib/vllm/models}"

# Per-checkpoint demands, used by download-vllm-models.sh to refuse weights
# the node cannot serve.
#   WEIGHT_GB  checkpoint size on disk
#   QUANT      weight dtype → required compute capability (gpu_supports_quant)
declare -A VLLM_MODEL_WEIGHT_GB=(
  [qwen3.6-35b-nvfp4]=21
  [qwen3.6-35b]=35
  [qwen3.5-122b-a10b]=78
  [qwen3.6-35b-awq]=26
  [qwen3-coder-next]=75
  [qwen3-coder-30b]=33
  [qwen3.6-27b]=21
  [bge-m3]=3
  [bge-reranker-v2-m3]=3
  [whisper-large-v3]=4
)
declare -A VLLM_MODEL_QUANT=(
  [qwen3.6-35b-nvfp4]=nvfp4
  [qwen3.6-35b]=fp8
  [qwen3.5-122b-a10b]=nvfp4
  [qwen3.6-35b-awq]=awq
  [qwen3-coder-next]=fp8
  [qwen3-coder-30b]=fp8
  [qwen3.6-27b]=nvfp4
  [bge-m3]=bf16
  [bge-reranker-v2-m3]=bf16
  [whisper-large-v3]=fp16
)
# Runtime headroom over the weights: activation buffers plus KV for one request.
VLLM_RUNTIME_HEADROOM_GB=6

# Recommended set for a node without an explicit model list. The 122B is a
# placement decision (78 GiB displaces everything else), so it is not here.
VLLM_PREFERRED_MODELS=(qwen3.6-35b-nvfp4)

# Usable-VRAM floor: the smallest chat build (26 GB int4) plus headroom.
VLLM_MIN_USABLE_VRAM_GB=32

# Why this node cannot serve alias $1: prints a reason and returns 1, or returns
# 0 silently. Alias validity is the caller's check.
vllm_model_unservable_reason() {
  local alias="$1" quant="${VLLM_MODEL_QUANT[$1]:-}" weight="${VLLM_MODEL_WEIGHT_GB[$1]:-0}"
  local vram; vram="$(gpu_usable_vram_gb)"
  if ! has_nvidia_gpu; then
    echo "no NVIDIA GPU on this node"; return 1
  fi
  # Size before capability: the more useful reason on a small card.
  if (( vram > 0 && vram < VLLM_MIN_USABLE_VRAM_GB )); then
    echo "the card has ${vram}GiB usable; this catalogue needs ${VLLM_MIN_USABLE_VRAM_GB}GiB before anything places with room to run"
    return 1
  fi
  if [[ -n "$quant" ]] && ! gpu_supports_quant "$quant"; then
    echo "$(get_gpu_name) cannot execute ${quant} weights (compute capability $(gpu_compute_cap))"
    return 1
  fi
  local need=$(( weight + VLLM_RUNTIME_HEADROOM_GB ))
  if (( weight > 0 && vram > 0 && vram < need )); then
    echo "needs ~${need}GiB (weights ${weight} + runtime ${VLLM_RUNTIME_HEADROOM_GB}) but the card has ${vram}GiB"
    return 1
  fi
  return 0
}

# vLLM base image per architecture. Nightly: the stable tags lag behind the
# hybrid GDN and MoE architectures. install-vllm.sh records the digest the tag
# resolved to as VLLM_BASE_DIGEST, which is the per-node pin.
VLLM_IMAGE_ARM64="vllm/vllm-openai:nightly-aarch64"
VLLM_IMAGE_AMD64="vllm/vllm-openai:cu129-nightly"

vllm_default_image() {
  case "$(detect_arch)" in
    arm64) echo "$VLLM_IMAGE_ARM64" ;;
    amd64) echo "$VLLM_IMAGE_AMD64" ;;
    *)     echo "" ;;
  esac
}

# Registry digest of a local image; empty for a locally built tag.
image_base_digest() {
  docker image inspect "$1" --format '{{if .RepoDigests}}{{index .RepoDigests 0}}{{end}}' 2>/dev/null
}

# Declared prices for LiteLLM spend tracking (USD per 1M tokens); the live
# catalogue overrides them at generation time (or_refresh_prices). Local models
# are free; their OpenRouter twins are priced below.
declare -A MODEL_PRICE_IN_PM=(
  [gpt-5.6-sol]=2.50     [gpt-5.6-terra]=2      [gpt-5.6-luna]=0.20    [gpt-5-nano]=0.05
  [gpt-5.3-codex]=1.75
  [claude-fable-5]=10    [claude-opus-5]=5      [claude-sonnet-5]=2    [claude-haiku-4.5]=1
  [gemini-3.1-pro-preview]=2   [gemini-3.7-flash]=0.375  [gemini-3.1-flash-lite]=0.25
  [grok-4.6]=2           [sonar]=1.00              [sonar-pro]=3.00
  [hy3]=0.132            [deepseek-v4-pro]=1.44    [deepseek-v4-flash]=0.0886
  [glm-5.3]=1.40         [mimo-v2.5]=0.14          [kimi-k3]=3.00
  [qwen3.8-max]=2.00     [qwen3.7-flash]=0.03      [qwen3-coder-plus]=0.65
  [minimax-m3]=0.30
  [qwen3.6-35b]=0    [qwen3.5-122b-a10b]=0    [qwen3-coder-30b]=0    [qwen3.6-27b]=0
  [text-embedding-3-small]=0.02
)
declare -A MODEL_PRICE_OUT_PM=(
  [gpt-5.6-sol]=15       [gpt-5.6-terra]=12     [gpt-5.6-luna]=1.20    [gpt-5-nano]=0.40
  [gpt-5.3-codex]=14.00
  [claude-fable-5]=50    [claude-opus-5]=25     [claude-sonnet-5]=10   [claude-haiku-4.5]=5
  [gemini-3.1-pro-preview]=12  [gemini-3.7-flash]=1.875  [gemini-3.1-flash-lite]=1.50
  [grok-4.6]=6           [sonar]=1.00              [sonar-pro]=15.00
  [hy3]=0.528            [deepseek-v4-pro]=2.88    [deepseek-v4-flash]=0.1772
  [glm-5.3]=4.40         [mimo-v2.5]=0.28          [kimi-k3]=15.00
  [qwen3.8-max]=6.00     [qwen3.7-flash]=0.13      [qwen3-coder-plus]=3.25
  [minimax-m3]=1.20
  [qwen3.6-35b]=0    [qwen3.5-122b-a10b]=0    [qwen3-coder-30b]=0    [qwen3.6-27b]=0
)

# Declared prices for the OpenRouter twins of local models and the STT
# fallback, keyed by OpenRouter slug. `or_price` prefers the live figure.
declare -A OR_TWIN_PRICE_IN_PM=(
  [qwen/qwen3.6-35b-a3b]=0.14              [qwen/qwen3.5-122b-a10b]=0.26
  [qwen/qwen3-coder-30b-a3b-instruct]=0.07 [qwen/qwen3.6-27b]=0.60
  [mistralai/voxtral-small-24b-2507]=0.10
)
declare -A OR_TWIN_PRICE_OUT_PM=(
  [qwen/qwen3.6-35b-a3b]=1.00              [qwen/qwen3.5-122b-a10b]=2.08
  [qwen/qwen3-coder-30b-a3b-instruct]=0.28 [qwen/qwen3.6-27b]=3.60
  [mistralai/voxtral-small-24b-2507]=0.30
)

per_token_cost() { awk -v v="$1" 'BEGIN { printf "%.10f", v/1000000 }'; }

has_openrouter() { [[ -n "$(env_get OPENROUTER_API_KEY)" ]]; }

# OpenRouter catalogue, fetched once per run (free-model list, price refresh,
# drift check).
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

# One model's live price, or the declared fallback. USD per 1M tokens.
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

# Overlay the live prices onto the declared tables, in place. Prints
# "<total> <moved>".
or_refresh_prices() {
  has_openrouter || return 0
  local live; live="$(or_catalogue 2>/dev/null)" || return 0
  [[ -n "$live" ]] || return 0

  local prov disp var m slug pair moved=0 total=0
  for prov in openai:OPENAI anthropic:ANTHROPIC google:GOOGLE x-ai:XAI \
              perplexity:PERPLEXITY tencent:TENCENT deepseek:DEEPSEEK z-ai:ZAI \
              xiaomi:XIAOMI moonshotai:MOONSHOTAI qwen:QWEN minimax:MINIMAX; do
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
  echo "${total} ${moved}"
}

# Free chat models OpenRouter currently offers, one slug per line: zero price
# both ways, text output, `:free` suffix, guardrail models excluded.
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

# Declared prices against the live catalogue. One line per mismatch; returns 1
# if any were found.
or_price_drift() {
  has_openrouter || { echo "no OPENROUTER_API_KEY — nothing to check" >&2; return 0; }
  command -v jq &>/dev/null || { echo "jq is required" >&2; return 0; }
  local key; key="$(env_get OPENROUTER_API_KEY)"
  local live; live="$(curl -sf --max-time 20 https://openrouter.ai/api/v1/models \
                       -H "Authorization: Bearer ${key}" 2>/dev/null)" || {
    echo "could not reach the OpenRouter catalogue" >&2; return 0; }

  local drift=0 slug m prov declared_in declared_out actual
  for prov in openai:OPENAI anthropic:ANTHROPIC google:GOOGLE x-ai:XAI \
              perplexity:PERPLEXITY tencent:TENCENT deepseek:DEEPSEEK z-ai:ZAI \
              xiaomi:XIAOMI moonshotai:MOONSHOTAI qwen:QWEN minimax:MINIMAX; do
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
  local slug
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

  # Image and audio: image_output per image token, audio/audio_output per
  # audio token.
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
# deduplicated.
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
      warn "vllm unreachable: $nu" >&2
      continue
    fi
    while IFS= read -r m; do
      [[ -n "$m" ]] && printf '%s\t%s\n' "$nu" "$m"
    done <<<"$tmp"
  done
}

# Whether any URL in the csv is a vLLM that answers. A URL only records that a
# backend was placed, not that it serves.
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

# One URL's self-reported max_model_len, 3 tries at 2s. Empty with rc!=0 on
# failure.
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

# URL host → ssh target: the matching user@host from NODES_VLLM, else the host.
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
# Local hosts (localhost, loopback, own IP) are queried without ssh.
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

# Wait until every URL in the csv is ready.
#   vllm_wait_until_ready "$URL_CSV" "label" [timeout=600] [interval=10] [dead_thresh=3]
#   rc 0 = all ready; 2 = a container was missing/exited dead_thresh times in a
#   row; 3 = timeout. vLLM listens only after the weights load, so a TCP refusal
#   is judged by container state, not elapsed time. Progress on stderr.
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

# LiteLLM team allowlist: every chat model gen-litellm-config.sh registers.
litellm_chat_models_csv() {
  local vllm_chat_url; vllm_chat_url="$(env_get VLLM_QWEN35B_URL 2>/dev/null || true)"
  local vllm_big_url; vllm_big_url="$(env_get VLLM_QWEN122B_URL 2>/dev/null || true)"
  local vllm_coder_url; vllm_coder_url="$(env_get VLLM_CODER30B_URL 2>/dev/null || true)"
  local vllm_dense_url; vllm_dense_url="$(env_get VLLM_QWEN27B_URL 2>/dev/null || true)"
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
  # Same shape as emit_brain: local/ and strict-local/ aliases over a vLLM URL,
  # the OpenRouter slug without one.
  if [[ -n "$vllm_chat_url" ]]; then
    out+=("local/qwen3.6-35b" "strict-local/qwen3.6-35b")
  elif has_openrouter; then
    out+=("qwen/qwen3.6-35b-a3b")
  fi
  if [[ -n "$vllm_big_url" ]]; then
    out+=("local/qwen3.5-122b-a10b" "strict-local/qwen3.5-122b-a10b")
  elif has_openrouter; then
    out+=("qwen/qwen3.5-122b-a10b")
  fi
  if [[ -n "$vllm_coder_url" ]]; then
    out+=("local/qwen3-coder-30b" "strict-local/qwen3-coder-30b")
  elif has_openrouter; then
    out+=("qwen/qwen3-coder-30b-a3b-instruct")
  fi
  if [[ -n "$vllm_dense_url" ]]; then
    out+=("local/qwen3.6-27b" "strict-local/qwen3.6-27b")
  elif has_openrouter; then
    out+=("qwen/qwen3.6-27b")
  fi
  if has_openrouter; then
    for m in "${OPENAI_EMBED_CATALOG[@]}"; do out+=("$m"); done
  fi
  local IFS=,
  echo "${out[*]:-}"
}

LITELLM_MASTER_KEY="${LITELLM_MASTER_KEY:-$(env_get LITELLM_MASTER_KEY)}"
# LITELLM_URL: shell env, then .env, then localhost:8000. Container-side
# hostnames are rewritten for host-side calls.
LITELLM_URL="${LITELLM_URL:-$(env_get LITELLM_URL)}"
LITELLM_URL="${LITELLM_URL:-http://localhost:8000}"
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
  # `sed '$d'` rather than the GNU-only `head -n -1`
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

# Repository path on a remote node, relative to the login user's $HOME.
KLOUDCHAT_REMOTE_DIR="${KLOUDCHAT_REMOTE_DIR:-KloudChat-LLM}"

# Whether a NODES_VLLM target is this host: localhost, loopback, $HOSTNAME, the
# short hostname, local IPv4 addresses, or DNS resolution to one of them.
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

# CSV → one item per line, trimmed of whitespace and quotes.
csv_split() {
  local IFS=, s
  for s in $1; do
    s="$(echo "$s" | sed 's/^[[:space:]]*//;s/[[:space:]]*$//;s/^"//;s/"$//')"
    [[ -n "$s" ]] && echo "$s"
  done
}

# Push the repo to a node, excluding runtime data. `.env` is excluded too: the
# scheduler writes per-node values there (see rsync_push_env_if_absent).
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

# Seed a node's .env only when it has none; after that the scheduler owns it.
# The test path is relative: ssh_run has already cd'd into KLOUDCHAT_REMOTE_DIR.
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

# ssh + cd into the repo + command. -n keeps the caller's stdin (while-read
# loops over the node list).
ssh_run() {
  local host="$1"; shift
  ssh -n -o StrictHostKeyChecking=accept-new -o ServerAliveInterval=30 "$host" \
    "set -e; cd '${KLOUDCHAT_REMOTE_DIR}' && $*"
}
