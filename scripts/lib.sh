#!/usr/bin/env bash

[[ -n "${__KC_LIB_SH:-}" ]] && return 0
__KC_LIB_SH=1

__R='\033[0;31m'; __G='\033[0;32m'; __Y='\033[1;33m'
__B='\033[1;34m'; __N='\033[0m'

hdr()  { echo; echo -e "${__B}━━━ $* ━━━${__N}"; }
# Diagnostics go to stderr: colour escapes would corrupt YAML built through
# `$(...)` capture in the config generators.
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
    # Newline first: a file without a trailing one concatenates the new key onto
    # the last line, producing a single variable with a garbage value.
    [[ -s "$file" && -n "$(tail -c 1 "$file")" ]] && printf '\n' >> "$file"
    printf '%s=%s\n' "$key" "$val" >> "$file"
  fi
}

# Write-permission check before a tmp+mv regeneration. A file left root-owned by
# a sudo run or a container write otherwise surfaces as a misleading "missing
# marker" error further down.
#   $1 = target file
#   $2 = need_read (default 1; 0 for a full regeneration)
# The parent directory is always checked, since mv writes there.
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

# Backend URLs are not derived from the node list: vLLM URLs come from the
# placement result, transcription URLs from a live probe. A node list would put
# backend-less arm64 nodes into WHISPER_URLS, and a non-empty value suppresses
# the OpenRouter STT fallback. Single-host wiring is written by install-vllm.sh.

has_nvidia_gpu() { command -v nvidia-smi &>/dev/null && nvidia-smi -L &>/dev/null; }

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
    nvfp4) need=10.0 ;;
    fp8)   need=8.9 ;;
    *)     need=8.0 ;;
  esac
  awk -v c="$cap" -v n="$need" 'BEGIN { exit !(c + 0 >= n + 0) }'
}

# Commercial catalogue, served through OpenRouter and skipped without an OR key.
# One array per provider group; gen-litellm-config.sh generates from these.
#
# Two tiers: frontier models users pick explicitly, and an open-weight tier at
# roughly a tenth of the price, all far too large to self-host.
#
# Ids and prices must be verified against https://openrouter.ai/api/v1/models.
# An id taken from prose rather than the catalogue 404s on first call.
OPENAI_MODELS=(gpt-5.6-sol gpt-5.6-terra gpt-5.6-luna gpt-5-nano)
ANTHROPIC_MODELS=(claude-opus-5 claude-sonnet-5 claude-haiku-4.5)
GOOGLE_MODELS=(gemini-3.1-pro-preview gemini-3.6-flash gemini-3.1-flash-lite)
XAI_MODELS=(grok-4.5)
PERPLEXITY_MODELS=(sonar)
# Open-weight tier, 295B to 2.8T. Display name equals the OpenRouter id, so the
# picker and the route cannot drift apart.
TENCENT_MODELS=(hy3)
DEEPSEEK_MODELS=(deepseek-v4-pro deepseek-v4-flash)
ZAI_MODELS=(glm-5.2)
XIAOMI_MODELS=(mimo-v2.5)
MOONSHOTAI_MODELS=(kimi-k3)

# Image generation, cheapest first: the picker defaults to the first entry and
# the price spread is a hundredfold. Prices are per `image_output` token, and one
# picture is ~1290 tokens.
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
# Audio generation through chat/completions. Both require a streaming request.
# gpt-audio is speech billed per token; lyria is music billed per clip.
OR_AUDIO_MODELS=(
  openai/gpt-audio
  google/lyria-3-clip-preview
)
declare -A MODEL_AUDIO_OUT_PM=(
  [openai/gpt-audio]=80.00
  [google/lyria-3-clip-preview]=0.00
)
declare -A MODEL_AUDIO_IN_PM=(
  [openai/gpt-audio]=40.00
  [google/lyria-3-clip-preview]=0.00
)
# Per-clip billing, which is why the per-token figures above are zero.
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

# vLLM catalogue (alias to HF repo). The chat model is NVFP4-only, so an
# FP4-incapable card cannot serve this lineup.
declare -A VLLM_MODELS=(
  [qwen3.6-35b-nvfp4]="unsloth/Qwen3.6-35B-A3B-NVFP4"
  # FP8 fallback for engines below vLLM 0.24: the NVFP4 build fails load_weights
  # there with a packed/unpacked shape mismatch.
  [qwen3.6-35b]="Qwen/Qwen3.6-35B-A3B"
  [glm-4.7-flash]="unsloth/GLM-4.7-Flash-NVFP4"
  # Retrieval embeddings. BF16 — at 2.2 GiB quantising buys nothing, and shifted
  # numerics mean a rebuilt index.
  [bge-m3]="BAAI/bge-m3"
)
: "${VLLM_MODELS_ROOT:=/var/lib/vllm/models}"

# Per-checkpoint demands on the card, used by download-vllm-models.sh to refuse
# weights the node could never serve.
#
#   WEIGHT_GB  measured checkpoint size on disk
#   QUANT      weight dtype → the compute capability it needs
declare -A VLLM_MODEL_WEIGHT_GB=(
  [qwen3.6-35b-nvfp4]=21
  [qwen3.6-35b]=35
  [glm-4.7-flash]=20
  [bge-m3]=3
)
declare -A VLLM_MODEL_QUANT=(
  [qwen3.6-35b-nvfp4]=nvfp4
  [qwen3.6-35b]=fp8
  [glm-4.7-flash]=nvfp4
  # BF16 — every card that can run the lineup can run this.
  [bge-m3]=bf16
)
# Runtime headroom on top of the weights: activation buffers plus enough KV to
# admit one request. A card that fits only the weights cannot start the engine.
VLLM_RUNTIME_HEADROOM_GB=6

# Fill order for the recommended set. The fp8 build is excluded: a fallback for
# older engines, requested by name.
VLLM_PREFERRED_MODELS=(qwen3.6-35b-nvfp4 glm-4.7-flash)

# Why this node cannot serve $1. Prints a reason and returns 1, or returns 0
# silently. Alias validity is the caller's check.
vllm_model_unservable_reason() {
  local alias="$1" quant="${VLLM_MODEL_QUANT[$1]:-}" weight="${VLLM_MODEL_WEIGHT_GB[$1]:-0}"
  local vram; vram="$(gpu_usable_vram_gb)"
  if ! has_nvidia_gpu; then
    echo "no NVIDIA GPU on this node"; return 1
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

# vLLM image per architecture. Nightly builds, for the hybrid GDN and MoE
# architectures the stable tags lag behind.
vllm_default_image() {
  case "$(detect_arch)" in
    arm64) echo "vllm/vllm-openai:nightly-aarch64" ;;
    amd64) echo "vllm/vllm-openai:cu129-nightly" ;;
    *)     echo "" ;;
  esac
}

# Unit prices for LiteLLM spend tracking (USD per 1M tokens). Local models are
# free; a fallback is billed at the twin's price by emit_or_fallback, since
# LiteLLM charges for the deployment that actually served.
#
# These drive user credit deduction. Verify against
# `curl https://openrouter.ai/api/v1/models`, never a blog post.
declare -A MODEL_PRICE_IN_PM=(
  [gpt-5.6-sol]=5        [gpt-5.6-terra]=1      [gpt-5.6-luna]=0.10    [gpt-5-nano]=0.05
  [claude-opus-5]=5      [claude-sonnet-5]=2    [claude-haiku-4.5]=1
  [gemini-3.1-pro-preview]=2   [gemini-3.6-flash]=1.50   [gemini-3.1-flash-lite]=0.25
  [grok-4.5]=2           [sonar]=1.00
  [hy3]=0.132            [deepseek-v4-pro]=0.435   [deepseek-v4-flash]=0.14
  [glm-5.2]=0.07         [mimo-v2.5]=0.14          [kimi-k3]=3.00
  # Local models are free. Their twins are priced in gen-litellm-config.sh.
  [qwen3.6-35b]=0    [glm-4.7-flash]=0
  [text-embedding-3-small]=0.02
)
declare -A MODEL_PRICE_OUT_PM=(
  [gpt-5.6-sol]=30       [gpt-5.6-terra]=6      [gpt-5.6-luna]=0.60    [gpt-5-nano]=0.40
  [claude-opus-5]=25     [claude-sonnet-5]=10   [claude-haiku-4.5]=5
  [gemini-3.1-pro-preview]=12  [gemini-3.6-flash]=7.50   [gemini-3.1-flash-lite]=1.50
  [grok-4.5]=6           [sonar]=1.00
  [hy3]=0.528            [deepseek-v4-pro]=0.87    [deepseek-v4-flash]=0.28
  [glm-5.2]=0.22         [mimo-v2.5]=0.28          [kimi-k3]=15.00
  [qwen3.6-35b]=0    [glm-4.7-flash]=0
)

per_token_cost() { awk -v v="$1" 'BEGIN { printf "%.10f", v/1000000 }'; }

has_openrouter() { [[ -n "$(env_get OPENROUTER_API_KEY)" ]]; }

# Free chat models OpenRouter currently offers, one slug per line. Queried rather
# than hard-coded: the free tier changes often, and a model that disappeared
# would still show in the picker and 404 on call. A failed query emits nothing.
#
# Filter: zero price both ways, text output, `:free` suffix. Previews without
# that suffix include non-chat models such as music and image generation.
#
# Guardrail models are excluded: they emit text but classify their input, so one
# picked as a chat partner returns a verdict instead of an answer.
or_free_models() {
  has_openrouter || return 0
  command -v jq &>/dev/null || return 0
  local key; key="$(env_get OPENROUTER_API_KEY)"
  curl -sf --max-time 15 https://openrouter.ai/api/v1/models \
       -H "Authorization: Bearer ${key}" 2>/dev/null \
    | jq -r '.data[]
        | select((.pricing.prompt // "0" | tonumber) == 0)
        | select((.pricing.completion // "0" | tonumber) == 0)
        | select(.architecture.output_modalities // [] | index("text"))
        | select(.id | endswith(":free"))
        | select(.id | test("guard|safety|safeguard|moderation") | not)
        | .id' 2>/dev/null | sort
}

# Output: "<URL>\t<served-model-name>" per line, from the URL csv the caller passes.
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
# URL dedup: duplicate deployments amplify LiteLLM retries onto one dead host.
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

# State of a single URL — the unit decision for the readiness wait.
#   0 = ready   (/v1/models 200 + at least 1 model)
#   1 = loading (TCP responds but HTTP not ready — model loading / 503 / empty model list)
#   2 = dead    (TCP refused / DNS failure — container not started)
# --connect-timeout separates TCP from HTTP time, distinguished by curl exit code.
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
    "" | 000) return 2 ;;   # curl failure (TCP / DNS) — dead
    *)        return 1 ;;   # 503/504/5xx etc. — loading
  esac
}

# One URL's self-reported max_model_len, retried 3 times at 2s.
# stdout: integer. Empty with rc!=0 on failure; the caller picks the fallback.
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

# Wait until all URLs in the CSV are ready. Re-evaluate state every interval.
#   - one URL's container is exited/missing dead_thresh times in a row → rc=2 (not-started fast-fail)
#   - some URL fails to become ready within timeout → rc=3
#   - all ready → rc=0
# vLLM listens only after the model loads, so TCP refused is normal for the first
# minutes of a cold start. Refusals are therefore judged by container state rather
# than elapsed time: Up means loading, exited or missing means dead.
# Progress goes to stderr.
# Usage: vllm_wait_until_ready "$URL_CSV" "label" [timeout=600] [interval=10] [dead_thresh=3]
# url's host → ssh target. Match user@host from the NODES_VLLM csv, else use host as-is.
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

# Container liveness signal (time-independent). Is the container publishing url's port Up?
#   0 = alive (Up, including "health: starting" = loading)
#   1 = dead  (no container on that port / exited / restart loop)
#   2 = unknown (ssh/docker query failed → defer judgment, wait until deadline)
# The host comes from the URL, so localhost, loopback and own IP are compared
# here to decide whether SSH can be bypassed.
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
        # TCP refused: judge by container liveness, not elapsed time. Up means
        # still loading; exited or missing fails fast after dead_thresh in a row.
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
          # cstate 2: state query failed. Defer judgement until the deadline.
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

# LiteLLM team allowlist. Must match what gen-litellm-config.sh emits.
litellm_chat_models_csv() {
  local vllm_chat_url; vllm_chat_url="$(env_get VLLM_QWEN35B_URL 2>/dev/null || true)"
  local vllm_fast_url; vllm_fast_url="$(env_get VLLM_GLMFLASH_URL 2>/dev/null || true)"
  local out=() m
  if has_openrouter; then
    for m in "${OPENAI_MODELS[@]}";     do out+=("openai/$m");     done
    for m in "${ANTHROPIC_MODELS[@]}";  do out+=("anthropic/$m");  done
    for m in "${GOOGLE_MODELS[@]}";     do out+=("google/$m");     done
    for m in "${DEEPSEEK_MODELS[@]}";   do out+=("deepseek/$m");   done
    for m in "${XAI_MODELS[@]}";        do out+=("x-ai/$m");       done
    for m in "${PERPLEXITY_MODELS[@]}"; do out+=("perplexity/$m"); done
    for m in "${META_MODELS[@]}";       do out+=("meta/$m");       done
    for m in "${QWEN_MODELS[@]}";       do out+=("qwen/$m");       done
  fi
  # Same condition as emit_brain (local URL or OR key): a mismatch shows the
  # agents in the picker while direct calls fail as model-not-allowed.
  { [[ -n "$vllm_chat_url" ]] || has_openrouter; } && out+=("local/qwen3.6-35b")
  { [[ -n "$vllm_fast_url" ]] || has_openrouter; } && out+=("local/glm-4.7-flash")
  if has_openrouter; then
    for m in "${OPENAI_EMBED_CATALOG[@]}"; do out+=("$m"); done
  fi
  local IFS=,
  echo "${out[*]:-}"
}

LITELLM_MASTER_KEY="${LITELLM_MASTER_KEY:-$(env_get LITELLM_MASTER_KEY)}"
# LITELLM_URL precedence: shell env, .env, then localhost:8000. Container-side
# hostnames in .env are normalised so host-side calls reach the published port.
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
  local body; body=$(echo "$resp" | head -n -1)
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
# NODE_* / NODES_* in .env decide whether a step runs locally or over SSH.
# is_local_host compares localhost, own IP, own hostname and DNS resolution.

# Repository path on a remote node, under the login user's $HOME.
KLOUDCHAT_REMOTE_DIR="${KLOUDCHAT_REMOTE_DIR:-KloudChat-LLM}"

# Is this host the current node? Matches localhost, loopback, $HOSTNAME, the
# short hostname, local IPv4 addresses and the target's DNS resolution.
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
  # Hostname targets are resolved and matched by IP. A missing getent is harmless.
  local target_ip
  target_ip="$(getent hosts "$target" 2>/dev/null | awk '{print $1; exit}')"
  if [[ -n "$target_ip" ]]; then
    for ip in $local_ips; do
      [[ "$target_ip" == "$ip" ]] && return 0
    done
  fi
  return 1
}

# is_local_node NODE_LITELLM — variable name in, is_local_host on its host.
# An empty value counts as local.
is_local_node() {
  local var="$1" host
  host="$(env_get "$var")"
  [[ -z "$host" ]] && return 0
  is_local_host "$host"
}

# CSV to newline-delimited, trimming whitespace and quotes.
csv_split() {
  local IFS=, s
  for s in $1; do
    s="$(echo "$s" | sed 's/^[[:space:]]*//;s/[[:space:]]*$//;s/^"//;s/"$//')"
    [[ -n "$s" ]] && echo "$s"
  done
}

# Push the repo to a remote, excluding runtime data (DB volumes, caches, logs).
#
# `.env` is excluded deliberately: `scheduler apply` writes per-node overrides
# there, and syncing the orchestrator's copy over them restores local defaults —
# which can put two models' gpu_util on one card. New nodes are seeded by
# rsync_push_env_if_absent; after that the applier owns the file.
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

# Seed a node's .env only when it has none, so applier-written placement values
# survive. Paired with rsync_push, which skips .env.
rsync_push_env_if_absent() {
  local host="$1"
  if ssh_run "$host" "test -f '${KLOUDCHAT_REMOTE_DIR}/.env'" 2>/dev/null; then
    echo "  → ${host}: .env exists, keeping node-local overrides"
    return 0
  fi
  echo "  → ${host}: seeding .env (none present)"
  rsync_push_file "$host" ".env"
}

# Push a single local file to a remote.
rsync_push_file() {
  local host="$1" path="$2"
  echo "  → rsync ${path} → ${host}:${KLOUDCHAT_REMOTE_DIR}/${path}"
  rsync -az "${__PROJECT_DIR}/${path}" "${host}:${KLOUDCHAT_REMOTE_DIR}/${path}"
}

# ssh + cd repo + command. -n protects the caller's stdin, which a while-read
# loop would otherwise lose after the first node.
ssh_run() {
  local host="$1"; shift
  ssh -n -o StrictHostKeyChecking=accept-new -o ServerAliveInterval=30 "$host" \
    "set -e; cd '${KLOUDCHAT_REMOTE_DIR}' && $*"
}

# docker_on_node NODE_LITELLM exec -i kloudchat-litellm-db psql ...
# Local runs docker directly, remote goes over ssh with stdin passed through.
# Arguments are quoted with printf %q for the remote shell.
docker_on_node() {
  local var="$1"; shift
  if is_local_node "$var"; then
    docker "$@"
    return
  fi
  local host; host="$(env_get "$var")"
  local q=() a
  for a in "$@"; do q+=("$(printf '%q' "$a")"); done
  ssh -o StrictHostKeyChecking=accept-new -o ServerAliveInterval=30 "$host" \
    "docker ${q[*]}"
}
