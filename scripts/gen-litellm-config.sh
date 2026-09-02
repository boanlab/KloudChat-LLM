#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
ENV_FILE="${KLOUDCHAT_ENV_FILE:-${PROJECT_DIR}/.env}"
CONFIG_FILE="${KLOUDCHAT_LITELLM_CONFIG_FILE:-${PROJECT_DIR}/services/litellm/config.yaml}"
CONFIG_EXAMPLE="${KLOUDCHAT_LITELLM_CONFIG_EXAMPLE:-${PROJECT_DIR}/services/litellm/config.yaml.example}"
source "${SCRIPT_DIR}/lib.sh"

MARKER_START='# >>> KLOUDCHAT_AUTOGEN_START'
MARKER_END='# <<< KLOUDCHAT_AUTOGEN_END'
# router_settings.fallbacks: generated too, under its own marker pair.
FB_START='# >>> KLOUDCHAT_FALLBACKS_START'
FB_END='# <<< KLOUDCHAT_FALLBACKS_END'

DRY_RUN=0
for arg in "$@"; do
  case "$arg" in
    --dry-run) DRY_RUN=1 ;;
    # Read-only: declared prices against the live catalogue
    --check-prices) or_price_drift; exit $? ;;
    -h|--help) echo "Usage: $(basename "$0") [--dry-run] [--check-prices]"; exit 0 ;;
    *)         err "Unknown: $arg"; exit 2 ;;
  esac
done

# config.yaml is gitignored; seeded from the example on first run.
if [[ ! -f "$CONFIG_FILE" ]]; then
  [[ -f "$CONFIG_EXAMPLE" ]] || { err "neither $CONFIG_FILE nor $CONFIG_EXAMPLE exists."; exit 1; }
  cp "$CONFIG_EXAMPLE" "$CONFIG_FILE"
  info "$CONFIG_FILE ← $CONFIG_EXAMPLE"
fi
assert_regen_writable "$CONFIG_FILE" || exit 1
grep -qF "$MARKER_START" "$CONFIG_FILE" && grep -qF "$MARKER_END" "$CONFIG_FILE" \
  || { err "AUTOGEN markers missing: $CONFIG_FILE"; exit 1; }
grep -qF "$FB_START" "$CONFIG_FILE" && grep -qF "$FB_END" "$CONFIG_FILE" \
  || { err "FALLBACKS markers missing: $CONFIG_FILE — add them under router_settings"; exit 1; }

# Context when a backend's max_model_len cannot be discovered from /v1/models.
CTX_FALLBACK=32768

# Per-model request timeout (s): a stuck-request backstop, generous because
# deep research runs for minutes. Load is the concurrency gate's job.
declare -A MODEL_TIMEOUT=( [qwen3.6-35b]=900 [qwen3.5-122b-a10b]=1800 [qwen3-coder-30b]=900 [qwen3.6-27b]=600 )

# OpenRouter provider-routing variant for chat routes: ":floor" (cheapest
# provider), ":nitro" (throughput), "" (OpenRouter default). KC_OR_VARIANT.
OR_VARIANT="${KC_OR_VARIANT-:floor}"

# Custom model_info fields KloudChat reads from /model/info. The id alone is
# not a trust boundary: local/* can spill to OpenRouter under load.
emit_kchat_boundary() {  # $1=self_hosted|hybrid|external  $2=strict  $3=privacy_only
  echo "      kchat_data_boundary: $1"
  echo "      kchat_strict_local: $2"
  echo "      kchat_privacy_only: $3"
}

# Free models. The :free suffix stays in the name; price 0.
emit_or_free() {
  local slug="$1"
  echo "  - model_name: ${slug}"
  echo "    litellm_params:"
  echo "      model: openrouter/${slug}"
  echo "      api_key: os.environ/OPENROUTER_API_KEY"
  echo "    model_info:"
  echo "      input_cost_per_token: 0.0"
  echo "      output_cost_per_token: 0.0"
  emit_kchat_boundary external false false
}

# Commercial model: name `<prov>/<id>`, route `openrouter/<route_prov>/<id>`.
# route_prov (5th arg) when OpenRouter's slug provider differs from the display one.
emit_commercial_or() {
  local prov="$1" id="$2" in_pm="$3" out_pm="$4" route_prov="${5:-$1}"
  has_openrouter || return 0
  echo "  - model_name: ${prov}/${id}"
  echo "    litellm_params:"
  echo "      model: openrouter/${route_prov}/${id}${OR_VARIANT}"
  echo "      api_key: os.environ/OPENROUTER_API_KEY"
  echo "    model_info:"
  echo "      input_cost_per_token: $(per_token_cost "$in_pm")"
  echo "      output_cost_per_token: $(per_token_cost "$out_pm")"
  emit_kchat_boundary external false false
}

# OpenRouter twin of a local vLLM model, the router_settings.fallbacks target.
# Hidden from the picker; emitted only when the local primary is deployed.
emit_or_fallback() {
  local local_url="$1" or_slug="$2" in_pm="$3" out_pm="$4"
  has_openrouter || return 0
  [[ -n "$local_url" ]] || return 0
  echo "  - model_name: ${or_slug}"
  echo "    litellm_params:"
  echo "      model: openrouter/${or_slug}${OR_VARIANT}"
  echo "      api_key: os.environ/OPENROUTER_API_KEY"
  echo "    model_info:"
  echo "      input_cost_per_token: $(per_token_cost "$in_pm")"
  echo "      output_cost_per_token: $(per_token_cost "$out_pm")"
  emit_kchat_boundary external false false
  echo "      kchat_hidden: true"
}

# max_input_tokens = ctx minus headroom for the tools schema and chat-template
# wrapper, which enable_pre_call_checks does not count. KC_PRE_CALL_HEADROOM.
__declared_max_input_tokens() {
  local ctx="$1" headroom="${KC_PRE_CALL_HEADROOM:-4096}"
  local v=$(( ctx - headroom ))
  (( v < 1024 )) && v=$(( ctx > 2048 ? ctx / 2 : ctx ))
  echo "$v"
}

# Image models. `output_cost_per_token` carries the per-image-token price.
emit_or_image() {
  local id="$1" out_per_token="$2" in_pm="$3"
  has_openrouter || return 0
  echo "  - model_name: ${id}"
  echo "    litellm_params:"
  echo "      model: openrouter/${id}"
  echo "      api_key: os.environ/OPENROUTER_API_KEY"
  echo "    model_info:"
  echo "      mode: image_generation"
  echo "      input_cost_per_token: $(per_token_cost "$in_pm")"
  echo "      output_cost_per_token: ${out_per_token}"
  emit_kchat_boundary external false false
}

# STT through OpenRouter: a plain chat deployment called with an audio content
# part (OpenRouter has no transcription endpoint). Hidden from the picker.
emit_or_stt() {
  local id="$1" in_pm="$2" out_pm="$3"
  has_openrouter || return 0
  echo "  - model_name: ${id}"
  echo "    litellm_params:"
  echo "      model: openrouter/${id}"
  echo "      api_key: os.environ/OPENROUTER_API_KEY"
  echo "    model_info:"
  echo "      input_cost_per_token: $(per_token_cost "$in_pm")"
  echo "      output_cost_per_token: $(per_token_cost "$out_pm")"
  emit_kchat_boundary external false false
  echo "      kchat_hidden: true"
}

# Audio models (`mode: audio_speech`). Per-clip models carry
# output_cost_per_request.
emit_or_audio() {
  local id="$1" in_pm="$2" out_pm="$3" per_call="$4"
  has_openrouter || return 0
  echo "  - model_name: ${id}"
  echo "    litellm_params:"
  echo "      model: openrouter/${id}"
  echo "      api_key: os.environ/OPENROUTER_API_KEY"
  echo "    model_info:"
  echo "      mode: audio_speech"
  echo "      input_cost_per_token: $(per_token_cost "$in_pm")"
  echo "      output_cost_per_token: $(per_token_cost "$out_pm")"
  [[ -n "$per_call" ]] && echo "      output_cost_per_request: ${per_call}"
  emit_kchat_boundary external false false
}

# Local embedding deployment, only when the scheduler placed one.
emit_vllm_embed() {
  local m="$1" url_csv="$2" urls
  [[ -n "$url_csv" ]] || return 0
  urls="$(__vllm_resolved_urls "local/${m}" "$url_csv")"
  while IFS= read -r url; do
    [[ -n "$url" ]] || continue
    echo "  - model_name: local/${m}"
    echo "    litellm_params:"
    echo "      model: hosted_vllm/local/${m}"
    echo "      api_base: ${url%/}/v1"
    echo "    model_info:"
    # `mode: embedding`: not a picker surface
    echo "      mode: embedding"
    echo "      input_cost_per_token: 0.0000000000"
    echo "      output_cost_per_token: 0.0000000000"
    emit_kchat_boundary self_hosted false false
  done <<< "$urls"
}

# Local reranker, proxied as /v1/rerank through LiteLLM's hosted_vllm path.
emit_vllm_rerank() {
  local m="$1" url_csv="$2" urls
  [[ -n "$url_csv" ]] || return 0
  urls="$(__vllm_resolved_urls "local/${m}" "$url_csv")"
  while IFS= read -r url; do
    [[ -n "$url" ]] || continue
    echo "  - model_name: local/${m}"
    echo "    litellm_params:"
    echo "      model: hosted_vllm/local/${m}"
    echo "      api_base: ${url%/}/v1"
    echo "    model_info:"
    echo "      mode: rerank"
    echo "      input_cost_per_token: 0.0000000000"
    echo "      output_cost_per_token: 0.0000000000"
    emit_kchat_boundary self_hosted false false
  done <<< "$urls"
}

# OpenAI embedding fallback, only with OPENAI_API_KEY (OpenRouter serves no
# embedding models).
emit_openai_embed() {
  local m="$1" in_pm
  [[ -n "$(env_get OPENAI_API_KEY)" ]] || return 0
  in_pm="${MODEL_PRICE_IN_PM[$m]:-}"
  echo "  - model_name: ${m}"
  echo "    model_info:"
  echo "      mode: embedding"
  [[ -n "$in_pm" ]] && echo "      input_cost_per_token: $(per_token_cost "$in_pm")"
  emit_kchat_boundary external false false
  echo "    litellm_params:"
  echo "      model: openai/${m}"
  echo "      api_key: os.environ/OPENAI_API_KEY"
}

# URLs from the csv that serve this model_name; the whole csv when none answer
# yet (LiteLLM's cooldown recovers once they do). Deduplicated.
__vllm_resolved_urls() {
  local want="$1" url_csv="$2" discovered
  discovered="$(vllm_union_node_models "$url_csv" \
    | awk -F'\t' -v w="$want" '$2==w {print $1}' \
    | awk '!seen[$0]++')"
  if [[ -n "$discovered" ]]; then echo "$discovered"; return 0; fi
  local IFS=, u
  for u in $url_csv; do
    u="$(__vllm_normalize_url "$u")"
    [[ -n "$u" ]] && echo "$u"
  done | awk '!seen[$0]++'
}

emit_vllm_chat_entry() {
  local m="$1" url="$2" ctx="$3" alias="$4" boundary="$5" strict="$6" privacy_only="$7"
  local in_pm out_pm tmo
  in_pm="${MODEL_PRICE_IN_PM[$m]:-}"
  out_pm="${MODEL_PRICE_OUT_PM[$m]:-}"
  echo "  - model_name: ${alias}"
  echo "    litellm_params:"
  # strict-local/* is an alias over the same deployment; the served id stays local/<m>
  echo "      model: hosted_vllm/local/${m}"
  echo "      api_base: ${url%/}/v1"
  tmo="${MODEL_TIMEOUT[$m]:-}"
  [[ -n "$tmo" ]] && echo "      timeout: ${tmo}"
  echo "    model_info:"
  # Native tool calls (--enable-auto-tool-choice); without these flags the
  # client falls back to ReAct text.
  echo "      supports_function_calling: true"
  echo "      supports_tool_choice: true"
  echo "      max_input_tokens: $(__declared_max_input_tokens "$ctx")"
  if [[ -n "$in_pm" && -n "$out_pm" ]]; then
    echo "      input_cost_per_token: $(per_token_cost "$in_pm")"
    echo "      output_cost_per_token: $(per_token_cost "$out_pm")"
  fi
  emit_kchat_boundary "$boundary" "$strict" "$privacy_only"
}

emit_vllm_chat() {
  local m="$1" url_csv="$2" urls ctx_fallback regular_boundary
  [[ -n "$url_csv" ]] || return 0
  regular_boundary=self_hosted
  has_openrouter && regular_boundary=hybrid
  # Context per deployment from /v1/models (the scheduler sets it per node).
  ctx_fallback="$CTX_FALLBACK"
  urls="$(__vllm_resolved_urls "local/${m}" "$url_csv")"
  while IFS= read -r url; do
    [[ -n "$url" ]] || continue
    local ctx
    if ! ctx="$(vllm_discover_max_len "$url")"; then
      warn "${url} max_model_len discovery failed — fallback ${ctx_fallback}"
      ctx="$ctx_fallback"
    fi
    emit_vllm_chat_entry "$m" "$url" "$ctx" "local/${m}" "$regular_boundary" false false
    # Privacy-only alias: no fallback twin, the gate rejects overload instead.
    emit_vllm_chat_entry "$m" "$url" "$ctx" "strict-local/${m}" self_hosted true true
  done <<<"$urls"
}

# A local model's OpenRouter route when no vLLM serves it: a commercial entry
# under its own slug, plus the tool flags LiteLLM's model map lacks for it.
emit_or_brain() {  # $1=or-slug  $2=in_pm  $3=out_pm
  echo "  - model_name: $1"
  echo "    litellm_params:"
  echo "      model: openrouter/$1${OR_VARIANT}"
  echo "      api_key: os.environ/OPENROUTER_API_KEY"
  echo "    model_info:"
  echo "      supports_function_calling: true"
  echo "      supports_tool_choice: true"
  echo "      input_cost_per_token: $(per_token_cost "$2")"
  echo "      output_cost_per_token: $(per_token_cost "$3")"
  emit_kchat_boundary external false false
}

# Local model registration: local/ and strict-local/ over a vLLM URL, the
# OpenRouter slug without one (local/* is never boundary external).
emit_brain() {  # $1=local-model  $2=url_csv  $3=or-slug  $4=or_in_pm  $5=or_out_pm
  if [[ -n "$2" ]]; then
    emit_vllm_chat "$1" "$2"
  elif has_openrouter; then
    # Exclusive with emit_or_fallback (same model_name, URL set)
    emit_or_brain "$3" "$4" "$5"
  fi
}

# Live prices over the declared tables before anything is emitted.
PRICE_REFRESH="$(or_refresh_prices || true)"
if [[ -n "$PRICE_REFRESH" ]]; then
  read -r PRICED MOVED <<<"$PRICE_REFRESH"
  if (( MOVED > 0 )); then
    info "prices: ${PRICED} read from the catalogue, ${MOVED} differ from the declared fallback"
  else
    info "prices: ${PRICED} read from the catalogue, all matching the declared fallback"
  fi
else
  warn "prices: could not reach the catalogue — using the declared fallbacks"
fi

SECTION=$(
  echo "  ${MARKER_START}"
  # --- local (vLLM), or the OpenRouter slug where nothing is deployed ---
  emit_brain "qwen3.6-35b"   "$(env_get VLLM_QWEN35B_URL)"    "qwen/qwen3.6-35b-a3b" "$(or_price qwen/qwen3.6-35b-a3b in)" "$(or_price qwen/qwen3.6-35b-a3b out)"
  emit_brain "qwen3.5-122b-a10b" "$(env_get VLLM_QWEN122B_URL)" "qwen/qwen3.5-122b-a10b" "$(or_price qwen/qwen3.5-122b-a10b in)" "$(or_price qwen/qwen3.5-122b-a10b out)"
  emit_brain "qwen3-coder-next" "$(env_get VLLM_CODERNEXT_URL)" "qwen/qwen3-coder-next" "$(or_price qwen/qwen3-coder-next in)" "$(or_price qwen/qwen3-coder-next out)"
  emit_brain "qwen3-coder-30b" "$(env_get VLLM_CODER30B_URL)" "qwen/qwen3-coder-30b-a3b-instruct" "$(or_price qwen/qwen3-coder-30b-a3b-instruct in)" "$(or_price qwen/qwen3-coder-30b-a3b-instruct out)"
  emit_brain "qwen3.6-27b" "$(env_get VLLM_QWEN27B_URL)" "qwen/qwen3.6-27b" "$(or_price qwen/qwen3.6-27b in)" "$(or_price qwen/qwen3.6-27b out)"
  # Retrieval: local when placed, the OpenAI catalogue below as fallback.
  emit_vllm_embed "bge-m3" "$(env_get VLLM_BGEM3_URL)"
  emit_vllm_rerank "bge-reranker-v2-m3" "$(env_get VLLM_RERANK_URL)"
  # STT fallback whenever no local whisper answers (a placed backend may be down).
  vllm_any_url_alive "$(env_get WHISPER_URLS)" || emit_or_stt "${STT_OR_MODEL:-mistralai/voxtral-small-24b-2507}" "$(or_price "${STT_OR_MODEL:-mistralai/voxtral-small-24b-2507}" in)" "$(or_price "${STT_OR_MODEL:-mistralai/voxtral-small-24b-2507}" out)"
  # --- openai (commercial + embed fallback) ---
  for m in "${OPENAI_MODELS[@]}";        do emit_commercial_or openai "$m" "${MODEL_PRICE_IN_PM[$m]}" "${MODEL_PRICE_OUT_PM[$m]}"; done
  for m in "${OPENAI_EMBED_CATALOG[@]}"; do emit_openai_embed "$m"; done
  # --- anthropic ---
  for m in "${ANTHROPIC_MODELS[@]}"; do emit_commercial_or anthropic "$m" "${MODEL_PRICE_IN_PM[$m]}" "${MODEL_PRICE_OUT_PM[$m]}"; done
  # --- google ---
  for m in "${GOOGLE_MODELS[@]}";    do emit_commercial_or google    "$m" "${MODEL_PRICE_IN_PM[$m]}" "${MODEL_PRICE_OUT_PM[$m]}"; done
  # --- x-ai / perplexity ---
  for m in "${XAI_MODELS[@]}";        do emit_commercial_or x-ai       "$m" "${MODEL_PRICE_IN_PM[$m]}" "${MODEL_PRICE_OUT_PM[$m]}"; done
  for m in "${PERPLEXITY_MODELS[@]}"; do emit_commercial_or perplexity "$m" "${MODEL_PRICE_IN_PM[$m]}" "${MODEL_PRICE_OUT_PM[$m]}"; done
  # --- open-weight tier ---
  for m in "${TENCENT_MODELS[@]}";    do emit_commercial_or tencent    "$m" "${MODEL_PRICE_IN_PM[$m]}" "${MODEL_PRICE_OUT_PM[$m]}"; done
  for m in "${DEEPSEEK_MODELS[@]}";   do emit_commercial_or deepseek   "$m" "${MODEL_PRICE_IN_PM[$m]}" "${MODEL_PRICE_OUT_PM[$m]}"; done
  for m in "${ZAI_MODELS[@]}";        do emit_commercial_or z-ai       "$m" "${MODEL_PRICE_IN_PM[$m]}" "${MODEL_PRICE_OUT_PM[$m]}"; done
  for m in "${XIAOMI_MODELS[@]}";     do emit_commercial_or xiaomi     "$m" "${MODEL_PRICE_IN_PM[$m]}" "${MODEL_PRICE_OUT_PM[$m]}"; done
  for m in "${MOONSHOTAI_MODELS[@]}"; do emit_commercial_or moonshotai "$m" "${MODEL_PRICE_IN_PM[$m]}" "${MODEL_PRICE_OUT_PM[$m]}"; done
  for m in "${QWEN_MODELS[@]}";       do emit_commercial_or qwen       "$m" "${MODEL_PRICE_IN_PM[$m]}" "${MODEL_PRICE_OUT_PM[$m]}"; done
  for m in "${MINIMAX_MODELS[@]}";    do emit_commercial_or minimax    "$m" "${MODEL_PRICE_IN_PM[$m]}" "${MODEL_PRICE_OUT_PM[$m]}"; done
  # --- free tier, from the live catalogue ---
  free_count=0
  while IFS= read -r free_slug; do
    [[ -n "$free_slug" ]] || continue
    emit_or_free "$free_slug"
    free_count=$((free_count+1))
  done < <(or_free_models)
  if (( free_count > 0 )); then info "registered ${free_count} free models"; fi
  # --- image generation (chat/completions with modalities ["image","text"]) ---
  for m in "${OR_IMAGE_MODELS[@]}"; do
    emit_or_image "$m" "${MODEL_IMAGE_OUT_COST[$m]}" "${MODEL_IMAGE_IN_PM[$m]}"
  done
  # --- audio generation (speech + music), same transport ---
  for m in "${OR_AUDIO_MODELS[@]}"; do
    emit_or_audio "$m" "${MODEL_AUDIO_IN_PM[$m]}" "${MODEL_AUDIO_OUT_PM[$m]}" "${MODEL_AUDIO_PER_CALL[$m]:-}"
  done
  # --- OpenRouter twins of the deployed local models (fallback targets, hidden) ---
  emit_or_fallback "$(env_get VLLM_QWEN35B_URL)"  "qwen/qwen3.6-35b-a3b" "$(or_price qwen/qwen3.6-35b-a3b in)" "$(or_price qwen/qwen3.6-35b-a3b out)"
  emit_or_fallback "$(env_get VLLM_QWEN122B_URL)" "qwen/qwen3.5-122b-a10b" "$(or_price qwen/qwen3.5-122b-a10b in)" "$(or_price qwen/qwen3.5-122b-a10b out)"
  emit_or_fallback "$(env_get VLLM_CODER30B_URL)" "qwen/qwen3-coder-30b-a3b-instruct" "$(or_price qwen/qwen3-coder-30b-a3b-instruct in)" "$(or_price qwen/qwen3-coder-30b-a3b-instruct out)"
  emit_or_fallback "$(env_get VLLM_QWEN27B_URL)" "qwen/qwen3.6-27b" "$(or_price qwen/qwen3.6-27b in)" "$(or_price qwen/qwen3.6-27b out)"
  echo "  ${MARKER_END}"
)

# router_settings.fallbacks: one line per deployed local model, same condition
# as emit_or_fallback.
fb_line() {  # $1=local-model  $2=url_csv  $3=or-slug
  has_openrouter || return 0
  [[ -n "$2" ]] || return 0
  echo "    - {\"local/$1\": [\"$3\"]}"
}
FALLBACKS=$(
  echo "  ${FB_START}"
  echo "  fallbacks:"
  fb_line "qwen3.6-35b"   "$(env_get VLLM_QWEN35B_URL)"  "qwen/qwen3.6-35b-a3b"
  fb_line "qwen3.5-122b-a10b" "$(env_get VLLM_QWEN122B_URL)" "qwen/qwen3.5-122b-a10b"
  fb_line "qwen3-coder-30b" "$(env_get VLLM_CODER30B_URL)" "qwen/qwen3-coder-30b-a3b-instruct"
  fb_line "qwen3.6-27b" "$(env_get VLLM_QWEN27B_URL)" "qwen/qwen3.6-27b"
  echo "  ${FB_END}"
)

if (( DRY_RUN )); then echo "$SECTION"; echo "$FALLBACKS"; exit 0; fi

tmp="$(mktemp)"; trap 'rm -f "$tmp"' EXIT
KC_SECTION="$SECTION" KC_FALLBACKS="$FALLBACKS" python3 - "$CONFIG_FILE" "$tmp" <<'PY'
import os, sys, pathlib

def splice(src, start, end, body, what):
    i, j = src.find(start), src.find(end)
    if i == -1 or j == -1 or j < i:
        sys.exit(f"error: {what} markers missing or reversed")
    ls = src.rfind("\n", 0, i) + 1
    le = src.find("\n", j)
    le = len(src) if le == -1 else le
    return src[:ls] + body + src[le:]

src = pathlib.Path(sys.argv[1]).read_text()
src = splice(src, "# >>> KLOUDCHAT_AUTOGEN_START", "# <<< KLOUDCHAT_AUTOGEN_END",
             os.environ["KC_SECTION"], "AUTOGEN")
src = splice(src, "# >>> KLOUDCHAT_FALLBACKS_START", "# <<< KLOUDCHAT_FALLBACKS_END",
             os.environ["KC_FALLBACKS"], "FALLBACKS")

# general_settings.store_prompts_in_spend_logs: false, set textually so operator
# comments and passthrough routes survive.
lines = src.splitlines(keepends=True)
general = next((i for i, line in enumerate(lines) if line.rstrip() == "general_settings:"), None)
if general is None:
    sys.exit("error: general_settings missing from LiteLLM config")
end = len(lines)
for i in range(general + 1, len(lines)):
    stripped = lines[i].strip()
    if stripped and not lines[i].startswith((" ", "\t", "#")):
        end = i
        break
setting = None
for i in range(general + 1, end):
    if lines[i].lstrip().startswith("store_prompts_in_spend_logs:"):
        setting = i
        break
if setting is not None:
    indent = lines[setting][:len(lines[setting]) - len(lines[setting].lstrip())]
    newline = "\n" if lines[setting].endswith("\n") else ""
    lines[setting] = f"{indent}store_prompts_in_spend_logs: false{newline}"
else:
    insert_at = general + 1
    for i in range(general + 1, end):
        if lines[i].strip().startswith(("master_key:", "store_model_in_db:")):
            insert_at = i + 1
    lines.insert(insert_at, "  store_prompts_in_spend_logs: false\n")
src = "".join(lines)
pathlib.Path(sys.argv[2]).write_text(src)
PY
mv "$tmp" "$CONFIG_FILE"; trap - EXIT

n=$(echo "$SECTION" | grep -c '^  - model_name:' || true)
ok "$CONFIG_FILE — $n models"
info "keys: openrouter=$(has_openrouter && echo y || echo n)"
