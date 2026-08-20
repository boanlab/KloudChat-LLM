#!/usr/bin/env bash
# Usage: setup.sh <role> [options]
#
# Prerequisite: create .env with ./scripts/gen-env.sh and fill in the keys.
# At least one of OPENROUTER_API_KEY or a reachable vLLM node is required.
#
# Roles
#   all                     install GPU nodes -> place models -> start stack -> print URLs
#   vllm                    run install-vllm.sh on every node in NODES_VLLM
#                           (vLLM plus transcription — a GPU node has one role)
#   scheduler <subcommand>  forwarded to python3 -m scheduler {inventory|plan|apply}
#   up                      start the backend stack (gateway, tools, LiteLLM)
#   urls                    print the addresses for the UI admin screen
#   stop | start            stop / resume containers, data preserved
#   clean                   DESTRUCTIVE — remove containers and runtime data
#
# Environment
#   KLOUDCHAT_SKIP_SCHEDULER=1   skip placement in `all` (you manage VLLM_*_URL)
#   KLOUDCHAT_REMOTE_DIR         repository path on remote nodes (default: KloudChat-LLM)
#   YES=1                        skip the clean confirmation
#   KLOUDCHAT_DISPATCHED=1       internal — marks a worker that arrived over SSH
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
cd "$PROJECT_DIR"
source "${SCRIPT_DIR}/lib.sh"

GATEWAY_PORT="$(env_get GATEWAY_PORT)"; GATEWAY_PORT="${GATEWAY_PORT:-8080}"

usage() { sed -n '2,/^[^#]/p' "$0" | sed -n 's/^# \{0,1\}//p'; }

# ───────────────────────── shared steps ─────────────────────────

step_env_check() {
  hdr "0. Environment"
  require_supported_platform
  ok "OS / ARCH: $(detect_os) / $(detect_arch)"
  command -v docker &>/dev/null || { err "docker not found. curl -fsSL https://get.docker.com | sh"; exit 1; }
  docker compose version &>/dev/null || { err "docker compose v2 is required"; exit 1; }
  ok "docker $(docker --version | awk '{print $3}' | tr -d ,)"
  local free; free="$(get_free_disk_gb)"
  (( free < 20 )) && warn "${free}GB free disk — may be too little to build images" || ok "${free}GB free disk"
}

step_env_validate() {
  hdr "1. .env"
  [[ -f .env ]] || { err ".env not found — run ./scripts/gen-env.sh first"; exit 1; }

  local vllm_urls; vllm_urls="$(vllm_urls_csv)"
  if has_openrouter; then
    ok "OPENROUTER_API_KEY is set"
  elif [[ -n "$vllm_urls" ]]; then
    ok "vLLM nodes present (local only, no OpenRouter)"
  else
    err "neither OPENROUTER_API_KEY nor a vLLM node — one of them is required"; exit 1
  fi

  local required=(LITELLM_MASTER_KEY LITELLM_DB_PASSWORD CODE_INTERPRETER_API_KEY CODE_INTERPRETER_MINIO_PASSWORD SCRAPER_API_KEY)
  # The index database only exists when its profile is on, but then its password
  # is as required as the others — compose refuses to start index-db without it.
  [[ ",$(env_get COMPOSE_PROFILES)," == *",index,"* ]] && required+=(INDEX_DB_PASSWORD)

  local missing=()
  local key
  for key in "${required[@]}"; do
    [[ -z "$(env_get "$key")" || "$(env_get "$key")" == change-me-* ]] && missing+=("$key")
  done
  (( ${#missing[@]} )) && { err "unfilled secrets: ${missing[*]} — ./scripts/gen-env.sh --force"; exit 1; }
  ok "secrets are set"
}

# Every vLLM URL recorded in .env. The scheduler writes these as {env_prefix}_URL.
vllm_urls_csv() {
  awk -F= '/^VLLM_[A-Z0-9_]+_URL=/ && $2 != "" { print $2 }' .env 2>/dev/null | paste -sd, -
}

# vLLM takes minutes to load weights and run torch.compile. Generating the config
# before that makes the context discovery in gen-litellm-config.sh fail, falling
# back to a conservative 32K — so a model serving 256K would be registered at 32K.
step_wait_vllm() {
  local csv; csv="$(vllm_urls_csv)"
  [[ -n "$csv" ]] || return 0
  hdr "2a. Waiting for vLLM"
  local timeout="${KLOUDCHAT_VLLM_WAIT_TIMEOUT:-1200}"
  local interval="${KLOUDCHAT_VLLM_WAIT_INTERVAL:-10}"
  local rc=0
  vllm_wait_until_ready "$csv" "vLLM" "$timeout" "$interval" 3 || rc=$?
  case "$rc" in
    0) ok "vLLM ready" ;;
    2) warn "no vLLM container came up — those models will go to OpenRouter" ;;
    3) warn "vLLM was not ready within ${timeout}s — contexts will be registered at the default" ;;
    *) warn "vLLM readiness unknown (rc=$rc)" ;;
  esac
  return 0
}

# A GPU node also runs the transcription backend. Without the shim in front of it,
# /tools/stt is the one capability left dead. This is the only place that touches
# the profile list, and it does nothing if the profile is already there.
step_enable_stt_profile() {
  [[ -n "$(env_get WHISPER_URLS)" ]] || return 0
  local profiles; profiles="$(env_get COMPOSE_PROFILES)"
  profiles="${profiles:-tools,models}"
  [[ ",${profiles}," == *",whisper,"* ]] && return 0
  env_set COMPOSE_PROFILES "${profiles},whisper"
  echo "  [profile] COMPOSE_PROFILES=${profiles},whisper (transcription shim)"
}

step_gen_configs() {
  hdr "2. Generating configuration"
  "${SCRIPT_DIR}/gen-searxng-config.sh"
  "${SCRIPT_DIR}/gen-litellm-config.sh"
}

step_compose_up() {
  hdr "3. Starting the stack"
  local profiles; profiles="$(env_get COMPOSE_PROFILES)"
  echo "  profiles: ${profiles:-tools,models}"
  docker compose up -d --build
  ok "containers started"
}

step_wait_gateway() {
  hdr "4. Checking the gateway"
  local max=120 step=5 elapsed code
  for elapsed in $(seq 0 "$step" "$max"); do
    code=$(curl -s -o /dev/null -w '%{http_code}' --max-time 3 "http://localhost:${GATEWAY_PORT}/health" 2>/dev/null || echo 000)
    printf "\r    [%3ds] gateway: %-6s" "$elapsed" "$code"
    [[ "$code" == "200" ]] && { echo; ok "gateway responding"; return 0; }
    sleep "$step"
  done
  echo; warn "gateway did not respond within ${max}s — docker compose logs gateway"
  return 1
}

# The gateway can answer while the services behind it are still starting. Printing
# the URL table in that window shows everything as "not started", which reads as a
# broken deployment when it is a healthy one.
step_wait_services() {
  hdr "5. Waiting for services"
  local max="${KLOUDCHAT_SERVICE_WAIT:-180}" step=5 elapsed pending
  for elapsed in $(seq 0 "$step" "$max"); do
    pending=0
    local probe
    for probe in /litellm/health/liveliness /tools/search/healthz /tools/fetch/health /tools/exec/health; do
      local code; code=$(curl -s -o /dev/null -w '%{http_code}' --max-time 3 "http://localhost:${GATEWAY_PORT}${probe}" 2>/dev/null || echo 000)
      [[ "$code" == "502" || "$code" == "503" || "$code" == "000" ]] && pending=$((pending+1))
    done
    printf "\r    [%3ds] capabilities not ready: %d " "$elapsed" "$pending"
    (( pending == 0 )) && { echo; ok "all capabilities responding"; return 0; }
    sleep "$step"
  done
  echo; warn "some capabilities did not respond within ${max}s — see the table below"
  return 0
}

# ───────────────────────── integration URLs ─────────────────────────

# The addresses to paste into the UI admin screen (Settings → System →
# Integrations). Per-capability status is printed alongside so a broken wire is
# found here rather than in the UI.
role_urls() {
  local host; host="$(hostname -I 2>/dev/null | awk '{print $1}')"
  host="${host:-localhost}"
  local base="http://${host}:${GATEWAY_PORT}"

  hdr "Addresses for the UI admin screen"
  echo
  printf "  %-14s %s\n" "Backend URL" "$base"
  echo "  (enter this one address under 'Integrations' and use auto-fill for the rest)"
  echo
  printf "  %-46s %-18s %s\n" "URL" "CAPABILITY" "STATUS"
  local name path probe
  for spec in \
    "LiteLLM:/litellm:/litellm/health/liveliness" \
    "Web search:/tools/search:/tools/search/healthz" \
    "Document fetch:/tools/fetch:/tools/fetch/health" \
    "Code execution:/tools/exec:/tools/exec/health" \
    "Deep research:/tools/research:/tools/research/mcp" \
    "Transcription:/tools/stt:/tools/stt/health"
  do
    name="${spec%%:*}"; local rest="${spec#*:}"
    path="${rest%%:*}"; probe="${rest#*:}"
    local code; code=$(curl -s -o /dev/null -w '%{http_code}' --max-time 3 "${base}${probe}" 2>/dev/null || echo 000)
    local status
    case "$code" in
      # 4xx still means we reached it — MCP endpoints answer GET with 405
      200|202|400|401|405|406) status="connected" ;;
      502|503)                 status="not started" ;;
      000)                     status="no gateway response" ;;
      *)                       status="http ${code}" ;;
    esac
    printf "  %-46s %-18s %s\n" "${base}${path}" "$name" "$status"
  done
  echo
  echo "  'not started' means that service's container is not running — check COMPOSE_PROFILES"
  echo "  The LiteLLM master key is LITELLM_MASTER_KEY in .env (enter it in the UI as well)"
}

# ───────────────────────── node installation ─────────────────────────

dispatch_csv() {
  local role="$1"; shift
  local var="NODES_${role^^}"
  local csv; csv="$(env_get "$var")"
  if [[ -z "$csv" ]]; then
    hdr "[${role}] localhost (${var} is empty — running on this host)"
    "${SCRIPT_DIR}/install-${role}.sh" "$@"
    return $?
  fi
  local host n=0 fail=0
  while IFS= read -r host; do
    n=$((n+1))
    hdr "[${role}] ${host}"
    if is_local_host "$host"; then
      "${SCRIPT_DIR}/install-${role}.sh" "$@" \
        || { warn "install-${role}.sh failed (local): $host"; fail=$((fail+1)); }
    else
      if ! rsync_push "$host"; then warn "rsync failed: $host"; fail=$((fail+1)); continue; fi
      rsync_push_env_if_absent "$host"
      local qargs="" a
      for a in "$@"; do qargs+=" $(printf '%q' "$a")"; done
      if ! ssh_run "$host" "KLOUDCHAT_DISPATCHED=1 ./scripts/setup.sh ${role}${qargs}"; then
        warn "setup.sh ${role} failed: $host"; fail=$((fail+1))
      fi
    fi
  done < <(csv_split "$csv")
  echo
  ok "${role}: $((n-fail))/${n} succeeded, ${fail} failed"
  (( fail > 0 )) && return 1 || return 0
}

# ───────────────────────── scheduler ─────────────────────────

# PyYAML is the placement step's only dependency. Install it through apt when
# available, otherwise tell the operator what to run.
ensure_scheduler_deps() {
  python3 -c "import yaml" 2>/dev/null && return 0
  if [[ "${KLOUDCHAT_SCHEDULER_NO_AUTOINSTALL:-0}" == "1" ]] || ! command -v apt-get &>/dev/null; then
    err "PyYAML is missing — sudo apt install python3-yaml (or pip install pyyaml)"
    return 1
  fi
  hdr "Installing scheduler dependencies (apt): python3-yaml"
  sudo apt-get install -y python3-yaml || { err "installation failed"; return 1; }
  ok "installed"
}

run_scheduler() {
  local sub="${1:-}"; shift || true
  [[ -n "$sub" ]] || { err "a scheduler subcommand is required: inventory | plan | apply"; return 2; }
  command -v python3 &>/dev/null || { err "python3 not found"; return 1; }
  ensure_scheduler_deps || return 1
  hdr "[scheduler ${sub}]"
  python3 -m scheduler "$sub" "$@"
}

# ───────────────────────── roles ─────────────────────────

role_up() {
  step_env_check
  step_env_validate
  step_gen_configs
  step_compose_up
  step_wait_gateway || true
  step_wait_services
  role_urls
}

role_all() {
  step_env_check
  step_env_validate

  # 1) Install GPU nodes — vLLM and transcription are one role. A failure here
  #    does not stop the rest: a partially installed cluster still serves.
  if [[ -n "$(env_get NODES_VLLM)" ]]; then
    dispatch_csv vllm || true
  else
    warn "NODES_VLLM is empty — skipping GPU node installation"
  fi

  # 2) Placement, when there are nodes and it was not skipped. Writes VLLM_*_URL
  #    (and WHISPER_URLS) into .env.
  if [[ "${KLOUDCHAT_SKIP_SCHEDULER:-0}" != "1" && -n "$(env_get NODES_VLLM)" ]]; then
    run_scheduler apply -y || warn "placement failed — using the VLLM_*_URL values already in .env"
  fi

  # 3) Transcription shim — enabled once placement has found a live backend.
  step_enable_stt_profile

  step_wait_vllm
  step_gen_configs
  step_compose_up
  step_wait_gateway || true
  step_wait_services
  role_urls
}

role_stack_stop() {
  hdr "stop — stopping containers (data preserved)"
  docker compose stop 2>&1 | sed 's/^/  /' || true
  ok "stopped — resume with: ./scripts/setup.sh start"
}

role_stack_start() {
  hdr "start — resuming containers"
  docker compose start 2>&1 | sed 's/^/  /' || true
  step_wait_gateway || true
}

role_clean() {
  hdr "clean — DESTRUCTIVE: removes containers and runtime data"
  echo "  Target: docker compose down + ./data (litellm postgres, code-interpreter redis/minio)"
  echo "  This cannot be undone."
  if [[ "${YES:-0}" != "1" ]]; then
    local answer
    read -r -p "  Type 'yes' to continue: " answer || answer=""
    [[ "$answer" == "yes" ]] || { echo "  cancelled"; return 1; }
  fi
  docker compose down -v --remove-orphans 2>&1 | sed 's/^/  /' || true
  if [[ -d ./data ]]; then
    rm -rf ./data && ok "./data removed" || warn "could not remove ./data — sudo may be required"
  fi
  ok "clean complete"
}

# ───────────────────────── entry point ─────────────────────────

main() {
  local role="${1:-}"; shift || true
  case "$role" in
    all)       role_all ;;
    up)        role_up ;;
    urls)      role_urls ;;
    vllm)
      if [[ "${KLOUDCHAT_DISPATCHED:-0}" == "1" ]]; then "${SCRIPT_DIR}/install-vllm.sh" "$@"
      else dispatch_csv vllm "$@"; fi ;;
    scheduler) run_scheduler "$@" ;;
    stop)      role_stack_stop ;;
    start)     role_stack_start ;;
    clean)     role_clean ;;
    ""|-h|--help) usage ;;
    *)         err "unknown role: $role"; echo; usage; exit 2 ;;
  esac
}

main "$@"
