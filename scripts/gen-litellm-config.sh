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
# router_settings.fallbacks lives under a different top-level key than model_list,
# so it needs its own marker pair — it is generated, never hand-kept. A fallbacks
# entry naming a model the cluster does not serve is not harmless: the models it
# DOES serve then have no failover at all, and nothing says so.
FB_START='# >>> KLOUDCHAT_FALLBACKS_START'
FB_END='# <<< KLOUDCHAT_FALLBACKS_END'

DRY_RUN=0
for arg in "$@"; do
  case "$arg" in
    --dry-run) DRY_RUN=1 ;;
    # Read-only: compares the declared prices against OpenRouter's live
    # catalogue and writes nothing. A wrong price breaks no request, so it is
    # only ever noticed in a billing report — check it when the catalogue moves.
    --check-prices) or_price_drift; exit $? ;;
    -h|--help) echo "Usage: $(basename "$0") [--dry-run] [--check-prices]"; exit 0 ;;
    *)         err "Unknown: $arg"; exit 2 ;;
  esac
done

# services/litellm/config.yaml = gitignored — per-deployment only. If absent,
# initialize from the example (static sections like router_settings /
# general_settings + an empty AUTOGEN block).
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

# ctx fallback for emit when vLLM max_model_len discovery fails. Normally each
# deployment uses max_model_len from `/v1/models` directly.
CTX_FALLBACK=32768

# Per-model request timeout (s), a stuck-request backstop. Latency under load is
# the concurrency gate's job, so these stay generous; deep research runs for
# minutes on qwen3.6-35b. Note that max_parallel_requests only soft-queues — the
# gate's pre-call rewrite is what reroutes.
# The 122B decodes at roughly a third of the 35B's rate, so the same generous
# backstop is proportionally tighter for it.
declare -A MODEL_TIMEOUT=( [qwen3.6-35b]=900 [glm-4.7-flash]=300 [qwen3.5-122b-a10b]=1800 [gemma-4-26b-a4b]=600 [qwen3-coder-30b]=900 [qwen3.6-27b]=600 )

# OpenRouter provider routing variant appended to chat/agent model routes.
# ":floor" = route to the lowest-price provider for that model (cost optimization).
# Other options: ":nitro" (throughput), "" (OpenRouter default). NOT applied to
# embeddings. Operator-tunable via KC_OR_VARIANT (set empty to disable).
OR_VARIANT="${KC_OR_VARIANT-:floor}"

# KloudChat reads these custom model_info fields from LiteLLM's /model/info.
# A model id is never itself a trust boundary: local/* can be backed directly by
# OpenRouter, and the normal local alias can spill there under load. Keep the
# current deployment topology explicit on every generated entry.
emit_kchat_boundary() {  # $1=self_hosted|hybrid|external  $2=strict  $3=privacy_only
  echo "      kchat_data_boundary: $1"
  echo "      kchat_strict_local: $2"
  echo "      kchat_privacy_only: $3"
}

# canonical name=`<prov>/<id>`, route=`openrouter/<prov>/<id>`. Not registered if there's no OR key.
# Free models. The :free suffix is kept in the name so the picker distinguishes
# them from paid ones, and a price of 0 means no credit is deducted.
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

emit_commercial_or() {
  # route_prov (5th, default=prov): split out when OR's actual slug provider
  # differs from the display one.
  # e.g. display model_name=meta/llama-4-maverick, route=openrouter/meta-llama/llama-4-maverick
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

# OpenRouter twin of a local vLLM model. router_settings.fallbacks routes
# local/<name> here when the node errors or cools down. Hidden from the picker,
# and emitted only when a local primary is deployed.
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
  # Router-only. Without this the twin lands in the picker under the same vendor
  # and name as the local model it backs — two identical rows where one is free
  # and the other bills OpenRouter, with nothing on screen to tell them apart.
  echo "      kchat_hidden: true"
}

# Declared limit, reduced by a fixed headroom. enable_pre_call_checks counts only
# messages, excluding the tools schema and chat-template wrapper — 1.5K to 4K for
# tool-heavy agents, enough to overflow a small-context node after wrapping.
# Tunable through KC_PRE_CALL_HEADROOM.
__declared_max_input_tokens() {
  local ctx="$1" headroom="${KC_PRE_CALL_HEADROOM:-4096}"
  local v=$(( ctx - headroom ))
  # Floor for very small ctx — a too-small declared value is meaningless → if
  # under 1024, fall back to half of ctx.
  (( v < 1024 )) && v=$(( ctx > 2048 ? ctx / 2 : ctx ))
  echo "$v"
}

# RAG embedding fallback. OpenRouter serves no embedding models, so this needs a
# direct provider key (OPENAI_API_KEY) and is emitted only when one exists.
# Without it there is no embedding deployment and RAG is off.
# Image models. `output_cost_per_token` carries the image-output price, which
# every one of these charges as completion tokens.
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

# STT through OpenRouter, for deployments with no transcription backend.
# OpenRouter has no transcription endpoint, so this is a plain chat deployment
# called with an audio content part: no `mode: audio_speech`, which would put it
# on the a/v surface, and `kchat_hidden` to keep it out of the picker.
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

# Audio models. `mode: audio_speech` puts them on the a/v surface and nowhere
# else. Lyria charges a flat price per clip rather than per token, so it carries
# `output_cost_per_second`-style flat billing through `cost_per_request`.
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

# Local embedding deployment, registered only when the scheduler placed one:
# an api_base pointing at nothing would fail every index write instead of
# falling back.
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
    # `mode: embedding` keeps it out of KloudChat's model picker, which maps
    # only chat/completion/image_generation/audio_speech to a surface.
    echo "      mode: embedding"
    echo "      input_cost_per_token: 0.0000000000"
    echo "      output_cost_per_token: 0.0000000000"
    emit_kchat_boundary self_hosted false false
  done <<< "$urls"
}

# Local reranker. Second stage over what the vector search returns: an embedding
# compares question and passage in one shared space, a reranker reads the pair
# together, which is why 2.2 GiB of it beats a much larger embedding model on the
# same shelf. LiteLLM proxies /v1/rerank to vLLM through its hosted_vllm rerank
# path, so index-shim keeps talking to one gateway.
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
    # Like `embedding`, `rerank` is not a surface KloudChat's picker maps, which
    # is what keeps it out of the model list.
    echo "      mode: rerank"
    echo "      input_cost_per_token: 0.0000000000"
    echo "      output_cost_per_token: 0.0000000000"
    emit_kchat_boundary self_hosted false false
  done <<< "$urls"
}

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

# Nodes from the URL csv that serve this model_name. Zero discovered falls back
# to the csv as-is, which covers vLLM startup: LiteLLM's cooldown recovers once
# the node answers. Both paths dedup, to keep duplicate deployments out.
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
  # The backend's served id remains local/<model>. strict-local/* is a LiteLLM
  # alias over that exact deployment, never a separately hosted or remote model.
  echo "      model: hosted_vllm/local/${m}"
  echo "      api_base: ${url%/}/v1"
  tmo="${MODEL_TIMEOUT[$m]:-}"
  [[ -n "$tmo" ]] && echo "      timeout: ${tmo}"
  echo "    model_info:"
  # vLLM supports native function calling via --enable-auto-tool-choice +
  # tool-call-parser. Exposing this flag is what makes the client
  # actually execute tools via structured tool_calls instead of the ReAct
  # (action/action_input) text fallback.
  echo "      supports_function_calling: true"
  echo "      supports_tool_choice: true"
  # Conservative limit: the router reserves a buffer for the tools schema. With
  # different ctx per node it selects a deployment by input token count.
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
  # Fallback chain: (1) max_model_len from vLLM /v1/models — the scheduler may set
  # different values per node, so query per deployment. (2) CTX_FALLBACK on failure
  # (a safety net when the backend just came up or gen-litellm-config runs standalone).
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
    # Privacy-only alias over the same vLLM deployment. It is deliberately not
    # named in router_settings.fallbacks; the concurrency gate rejects overload
    # rather than rewriting this alias to an external twin.
    emit_vllm_chat_entry "$m" "$url" "$ctx" "strict-local/${m}" self_hosted true true
  done <<<"$urls"
}

# The brain's OpenRouter route where no vLLM serves it: an ordinary commercial
# registration under the model's own slug, plus the tool-capability flags. The
# frontier catalogue gets those from LiteLLM's model map, which does not
# reliably carry an entry for a self-hostable open-weight slug — and without
# them the picker drops tools for a model that has them.
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

# Brain registration. The local/ prefix states where a request starts, not which
# weights answer it: a local deployment that spills to OpenRouter under load or
# on failure still starts local, so local/<m> is honest there. With no vLLM URL
# nothing about the route is local, so the model registers under its own
# OpenRouter slug rather than borrowing the name. That keeps one invariant worth
# having: local/* is never kchat_data_boundary external.
#
# Consequence for a GPU-less install: surfaces that name local/<m> no longer
# resolve, and must pick from the catalogue instead (docs/models.md).
emit_brain() {  # $1=local-model  $2=url_csv  $3=or-slug  $4=or_in_pm  $5=or_out_pm
  if [[ -n "$2" ]]; then
    emit_vllm_chat "$1" "$2"
  elif has_openrouter; then
    # Exclusive with emit_or_fallback, which emits this same model_name only
    # when the URL *is* set. Both firing would put two deployments under one
    # name and split ordinary traffic onto the paid twin.
    emit_or_brain "$3" "$4" "$5"
  fi
}

# Registration order = local → openai → anthropic → google (by provider group)
SECTION=$(
  echo "  ${MARKER_START}"
  # --- local (vLLM) ---
  # Chat models: the local vLLM when one exists, otherwise the same model under
  # its OpenRouter slug — the local/ alias is not created without a deployment
  # behind it. qwen3.6-35b covers chat, vision, deep research and coding;
  # glm-4.7-flash is the cheap-decode floor.
  emit_brain "qwen3.6-35b"   "$(env_get VLLM_QWEN35B_URL)"    "qwen/qwen3.6-35b-a3b" 0.14 1.00
  emit_brain "glm-4.7-flash" "$(env_get VLLM_GLMFLASH_URL)"   "z-ai/glm-4.7-flash"  0.06 0.40
  emit_brain "qwen3.5-122b-a10b" "$(env_get VLLM_QWEN122B_URL)" "qwen/qwen3.5-122b-a10b" 0.26 2.08
  emit_brain "gemma-4-26b-a4b" "$(env_get VLLM_GEMMA26B_URL)" "google/gemma-4-26b-a4b-it" 0.07 0.34
  emit_brain "qwen3-coder-30b" "$(env_get VLLM_CODER30B_URL)" "qwen/qwen3-coder-30b-a3b-instruct" 0.07 0.28
  emit_brain "qwen3.6-27b" "$(env_get VLLM_QWEN27B_URL)" "qwen/qwen3.6-27b" 0.30 2.00
  # Embeddings for RAG. Local when placed; the OpenAI catalogue below is the
  # fallback, and with neither there is no /embeddings route and retrieval
  # falls back to lexical matching in KloudChat.
  emit_vllm_embed "bge-m3" "$(env_get VLLM_BGEM3_URL)"
  emit_vllm_rerank "bge-reranker-v2-m3" "$(env_get VLLM_RERANK_URL)"
  # STT twin — registered whenever no local whisper is deployed, matching the
  # scheduler's delegation. Its absence is what hides the microphone.
  [[ -z "$(env_get WHISPER_URLS)" ]] && emit_or_stt "${STT_OR_MODEL:-mistralai/voxtral-small-24b-2507}" 0.10 0.30
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
  # --- open-weight tier: near-frontier quality at ~1/10 the frontier price, and all
  #     too large to self-host on RTX-class cards (see lib.sh catalog comment). ---
  for m in "${TENCENT_MODELS[@]}";    do emit_commercial_or tencent    "$m" "${MODEL_PRICE_IN_PM[$m]}" "${MODEL_PRICE_OUT_PM[$m]}"; done
  for m in "${DEEPSEEK_MODELS[@]}";   do emit_commercial_or deepseek   "$m" "${MODEL_PRICE_IN_PM[$m]}" "${MODEL_PRICE_OUT_PM[$m]}"; done
  for m in "${ZAI_MODELS[@]}";        do emit_commercial_or z-ai       "$m" "${MODEL_PRICE_IN_PM[$m]}" "${MODEL_PRICE_OUT_PM[$m]}"; done
  for m in "${XIAOMI_MODELS[@]}";     do emit_commercial_or xiaomi     "$m" "${MODEL_PRICE_IN_PM[$m]}" "${MODEL_PRICE_OUT_PM[$m]}"; done
  for m in "${MOONSHOTAI_MODELS[@]}"; do emit_commercial_or moonshotai "$m" "${MODEL_PRICE_IN_PM[$m]}" "${MODEL_PRICE_OUT_PM[$m]}"; done
  # Qwen's hosted tier and MiniMax. Distinct from the qwen3.x checkpoints served
  # locally — same vendor, different weights and a context these cards cannot hold.
  for m in "${QWEN_MODELS[@]}";       do emit_commercial_or qwen       "$m" "${MODEL_PRICE_IN_PM[$m]}" "${MODEL_PRICE_OUT_PM[$m]}"; done
  for m in "${MINIMAX_MODELS[@]}";    do emit_commercial_or minimax    "$m" "${MODEL_PRICE_IN_PM[$m]}" "${MODEL_PRICE_OUT_PM[$m]}"; done
  # --- free tier: whatever OpenRouter gives away at generation time ---
  free_count=0
  while IFS= read -r free_slug; do
    [[ -n "$free_slug" ]] || continue
    emit_or_free "$free_slug"
    free_count=$((free_count+1))
  done < <(or_free_models)
  if (( free_count > 0 )); then info "registered ${free_count} free models"; fi
  # --- image generation. Served through chat/completions with
  #     `modalities: ["image","text"]`, so these register as ordinary chat
  #     deployments and the UI marks them image-only. Billing is per output
  #     token, ~1290 per picture, cheapest first because the picker defaults to
  #     the first entry.
  for m in "${OR_IMAGE_MODELS[@]}"; do
    emit_or_image "$m" "${MODEL_IMAGE_OUT_COST[$m]}" "${MODEL_IMAGE_IN_PM[$m]}"
  done
  # --- audio generation (speech + music). Same chat/completions transport.
  for m in "${OR_AUDIO_MODELS[@]}"; do
    emit_or_audio "$m" "${MODEL_AUDIO_IN_PM[$m]}" "${MODEL_AUDIO_OUT_PM[$m]}" "${MODEL_AUDIO_PER_CALL[$m]:-}"
  done
  # --- others: local vLLM → OR same-model fallback twin (see router_settings.fallbacks, not shown in UI) ---
  emit_or_fallback "$(env_get VLLM_QWEN35B_URL)"  "qwen/qwen3.6-35b-a3b" 0.14 1.00
  emit_or_fallback "$(env_get VLLM_GLMFLASH_URL)" "z-ai/glm-4.7-flash" 0.06 0.40
  emit_or_fallback "$(env_get VLLM_QWEN122B_URL)" "qwen/qwen3.5-122b-a10b" 0.26 2.08
  emit_or_fallback "$(env_get VLLM_GEMMA26B_URL)" "google/gemma-4-26b-a4b-it" 0.07 0.34
  emit_or_fallback "$(env_get VLLM_CODER30B_URL)" "qwen/qwen3-coder-30b-a3b-instruct" 0.07 0.28
  emit_or_fallback "$(env_get VLLM_QWEN27B_URL)" "qwen/qwen3.6-27b" 0.30 2.00
  echo "  ${MARKER_END}"
)

# Fallbacks: one line per local model that has BOTH a local deployment and an OR
# twin emitted above. Same condition as emit_or_fallback, so the two can't drift.
fb_line() {  # $1=local-model  $2=url_csv  $3=or-slug
  has_openrouter || return 0
  [[ -n "$2" ]] || return 0
  echo "    - {\"local/$1\": [\"$3\"]}"
}
FALLBACKS=$(
  echo "  ${FB_START}"
  echo "  fallbacks:"
  fb_line "qwen3.6-35b"   "$(env_get VLLM_QWEN35B_URL)"  "qwen/qwen3.6-35b-a3b"
  fb_line "glm-4.7-flash" "$(env_get VLLM_GLMFLASH_URL)" "z-ai/glm-4.7-flash"
  fb_line "qwen3.5-122b-a10b" "$(env_get VLLM_QWEN122B_URL)" "qwen/qwen3.5-122b-a10b"
  fb_line "gemma-4-26b-a4b" "$(env_get VLLM_GEMMA26B_URL)" "google/gemma-4-26b-a4b-it"
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

# config.yaml predates the privacy-safe default on existing installations. Do a
# narrow textual migration so regeneration disables prompt/response persistence
# without reserialising or discarding operator comments and passthrough routes.
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
