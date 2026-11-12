#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
source "${SCRIPT_DIR}/lib.sh"

ENV_FILE="${PROJECT_DIR}/.env"
ENV_EXAMPLE="${PROJECT_DIR}/.env.example"

usage() { echo "Usage: $(basename "$0") [--fill | --force]"; }

gen_secret() {
  if command -v openssl &>/dev/null; then openssl rand -hex "${1:-16}"
  else od -vN "${1:-16}" -An -tx1 /dev/urandom | tr -d ' \n'; fi
}

secret_for() {
  case "$1" in
    # LiteLLM key convention: sk- prefix
    LITELLM_MASTER_KEY) echo "sk-$(gen_secret 32)" ;;
    # Shared with KloudChat (KCHAT_TOOLS_GATEWAY_KEY); the gateway wants >= 32 chars
    TOOLS_GATEWAY_KEY)  gen_secret 32 ;;
    *)                  gen_secret 16 ;;
  esac
}

# Database a password key belongs to, when its data directory already exists
initialised_db() {
  case "$1" in
    LITELLM_DB_PASSWORD) [[ -d "${PROJECT_DIR}/data/litellm/postgres" ]] && echo "data/litellm/postgres" ;;
    INDEX_DB_PASSWORD)   [[ -d "${PROJECT_DIR}/data/index/postgres" ]] && echo "data/index/postgres" ;;
  esac
  return 0
}

MODE=create
for arg in "$@"; do
  case "$arg" in
    --fill)    MODE=fill ;;
    --force)   MODE=force ;;
    -h|--help) usage; exit 0 ;;
    *)         err "Unknown: $arg"; usage; exit 2 ;;
  esac
done

[[ -f "$ENV_EXAMPLE" ]] || { err "$ENV_EXAMPLE not found"; exit 1; }
# Secrets: owner only
umask 077

if [[ "$MODE" == fill ]]; then
  [[ -f "$ENV_FILE" ]] || { err ".env not found — run without --fill to create it"; exit 1; }
  chmod 600 "$ENV_FILE"
  filled=()
  while IFS= read -r key; do
    current="$(env_get "$key")"
    [[ -z "$current" || "$current" == change-me-* ]] || continue
    db="$(initialised_db "$key")"
    [[ -n "$db" ]] && warn "${key}: ${db} exists and keeps its old password — set the original value instead"
    env_set "$key" "$(secret_for "$key")"
    filled+=("$key")
  done < <(sed -nE 's/^([A-Za-z_][A-Za-z0-9_]*)=change-me-.*/\1/p' "$ENV_EXAMPLE")
  if (( ${#filled[@]} )); then
    ok "filled: ${filled[*]}"
  else
    ok "no secret left to fill"
  fi
  exit 0
fi

if [[ -f "$ENV_FILE" && "$MODE" == create ]]; then
  info ".env already exists — --fill fills empty secrets, --force overwrites."; exit 0
fi
if [[ -f "$ENV_FILE" ]]; then
  backup="${ENV_FILE}.bak.$(date +%Y%m%d%H%M%S)"
  cp -p "$ENV_FILE" "$backup"
  warn "previous .env kept at ${backup}"
fi

generated=()
while IFS= read -r line; do
  if [[ "$line" =~ ^([A-Za-z_][A-Za-z0-9_]*)=change-me- ]]; then
    key="${BASH_REMATCH[1]}"
    generated+=("$key")
    echo "${key}=$(secret_for "$key")"
  else
    echo "$line"
  fi
done < "$ENV_EXAMPLE" > "$ENV_FILE"
chmod 600 "$ENV_FILE"

ok ".env generated: ${ENV_FILE} (mode 600)"
printf "  %s\n" "${generated[@]}" | sort
for key in LITELLM_DB_PASSWORD INDEX_DB_PASSWORD; do
  db="$(initialised_db "$key")"
  [[ -n "$db" ]] && warn "${key}: ${db} keeps its old password — restore it from the backup or reset the database"
done
exit 0
