#!/usr/bin/env bash
# Usage: gen-searxng-config.sh
#
# services/searxng/settings.yml from settings.yml.example, the __SEARXNG_SECRET_KEY__
# and __NAVER_*__ sentinels filled from .env (SearXNG takes no secret_key from the
# environment). Rewritten only when the result differs; SearXNG reads it at start,
# so setup.sh restarts the container after a rewrite.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# No lib.sh dependency
err()  { printf '\033[31m%s\033[0m\n' "$*" >&2; }
info() { printf '  %s\n' "$*"; }

PROJECT_DIR="$(dirname "$SCRIPT_DIR")"

ENV_FILE="${PROJECT_DIR}/.env"
CONFIG_FILE="${PROJECT_DIR}/services/searxng/settings.yml"
CONFIG_EXAMPLE="${PROJECT_DIR}/services/searxng/settings.yml.example"
SENTINEL='__SEARXNG_SECRET_KEY__'

for arg in "$@"; do
  case "$arg" in
    -h|--help) echo "Usage: $(basename "$0")"; exit 0 ;;
    *)         err "Unknown: $arg"; exit 2 ;;
  esac
done

[[ -f "$CONFIG_EXAMPLE" ]] || { err "$CONFIG_EXAMPLE not found."; exit 1; }
[[ -f "$ENV_FILE" ]] || { err "$ENV_FILE not found. Run ./scripts/gen-env.sh first."; exit 1; }
# Single-key extraction rather than `source` (.env values may contain spaces)
SEARXNG_SECRET_KEY="$(grep -E '^SEARXNG_SECRET_KEY=' "$ENV_FILE" | tail -n1 | cut -d= -f2-)"
[[ -n "$SEARXNG_SECRET_KEY" ]] || { err "SEARXNG_SECRET_KEY in .env is empty."; exit 1; }
[[ "$SEARXNG_SECRET_KEY" != change-me-* ]] || { err "SEARXNG_SECRET_KEY is still the placeholder — re-run gen-env.sh."; exit 1; }

tmp="$(mktemp)"
trap 'rm -f "$tmp"' EXIT
# NAVER credentials are optional: both set enables the naver engines, otherwise
# they are written disabled. `|| true`: an absent line is fine under pipefail.
NAVER_CLIENT_ID="$( (grep -E '^NAVER_CLIENT_ID=' "$ENV_FILE" || true) | tail -n1 | cut -d= -f2-)"
NAVER_CLIENT_SECRET="$( (grep -E '^NAVER_CLIENT_SECRET=' "$ENV_FILE" || true) | tail -n1 | cut -d= -f2-)"
if [[ -n "$NAVER_CLIENT_ID" && -n "$NAVER_CLIENT_SECRET" ]]; then
  NAVER_DISABLED=false
else
  NAVER_DISABLED=true
  NAVER_CLIENT_ID="unset"; NAVER_CLIENT_SECRET="unset"
fi
# The secret_key line only; the sentinel also appears in a comment
sed -e "/^[[:space:]]*secret_key:/ s|${SENTINEL}|${SEARXNG_SECRET_KEY}|" \
    -e "s|__NAVER_CLIENT_ID__|${NAVER_CLIENT_ID}|g" \
    -e "s|__NAVER_CLIENT_SECRET__|${NAVER_CLIENT_SECRET}|g" \
    -e "s|__NAVER_DISABLED__|${NAVER_DISABLED}|g" \
    "$CONFIG_EXAMPLE" > "$tmp"
if grep -q "__NAVER_" "$tmp"; then
  err "a __NAVER_*__ sentinel was not substituted — check settings.yml.example."
  exit 1
fi

if ! grep -qE "^[[:space:]]*secret_key: \"?${SEARXNG_SECRET_KEY}\"?[[:space:]]*$" "$tmp"; then
  err "secret_key was not substituted — check the ${SENTINEL} line in .example."
  exit 1
fi

# An unreadable existing file (root-owned after a sudo run) is replaced
if [[ -r "$CONFIG_FILE" ]] && cmp -s "$tmp" "$CONFIG_FILE"; then
  info "$CONFIG_FILE is up to date."
  exit 0
fi
dir="$(dirname "$CONFIG_FILE")"
if [[ ! -w "$dir" ]]; then
  err "Directory $dir not writable (owner: $(stat -c '%U:%G' "$dir" 2>/dev/null || echo '?'), current user: $(id -un)) → cannot update $CONFIG_FILE."
  err "  Fix: sudo chown $(id -u):$(id -g) \"$dir\""
  exit 1
fi
mv "$tmp" "$CONFIG_FILE"
trap - EXIT
info "$CONFIG_FILE ← $CONFIG_EXAMPLE (secret injected; restart searxng to apply)"
