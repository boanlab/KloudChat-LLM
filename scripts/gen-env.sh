#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
source "${SCRIPT_DIR}/lib.sh"

ENV_FILE="${PROJECT_DIR}/.env"
ENV_EXAMPLE="${PROJECT_DIR}/.env.example"

usage() { echo "Usage: $(basename "$0") [--force]"; }

gen_secret() {
  if command -v openssl &>/dev/null; then openssl rand -hex "${1:-16}"
  else od -vN "${1:-16}" -An -tx1 /dev/urandom | tr -d ' \n'; fi
}

FORCE=0
for arg in "$@"; do
  case "$arg" in
    --force)   FORCE=1 ;;
    -h|--help) usage; exit 0 ;;
    *)         err "Unknown: $arg"; usage; exit 2 ;;
  esac
done

[[ -f "$ENV_EXAMPLE" ]] || { err "$ENV_EXAMPLE not found"; exit 1; }
if [[ -f "$ENV_FILE" && $FORCE -eq 0 ]]; then
  info ".env already exists — use --force to overwrite."; exit 0
fi

declare -A GENERATED
while IFS= read -r line; do
  if [[ "$line" =~ ^([A-Za-z_][A-Za-z0-9_]*)=change-me- ]]; then
    key="${BASH_REMATCH[1]}"
    case "$key" in
      # LiteLLM key convention — both master and virtual keys carry the sk- prefix
      LITELLM_MASTER_KEY)  secret="sk-$(gen_secret 32)" ;;
      *MASTER_KEY*|JWT_*)  secret="$(gen_secret 32)" ;;
      *)                   secret="$(gen_secret 16)" ;;
    esac
    GENERATED["$key"]="$secret"
    echo "${key}=${secret}"
  else
    echo "$line"
  fi
done < "$ENV_EXAMPLE" > "$ENV_FILE"

ok ".env generated: ${ENV_FILE}"
for key in "${!GENERATED[@]}"; do printf "  %-24s %s\n" "$key" "${GENERATED[$key]}"; done | sort
