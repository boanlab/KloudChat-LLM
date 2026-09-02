#!/usr/bin/env bash
# Usage: gen-searxng-config.sh [--force]
#
# services/searxng/settings.yml from settings.yml.example, with the
# __SEARXNG_SECRET_KEY__ sentinel replaced by .env's SEARXNG_SECRET_KEY
# (SearXNG takes no secret_key from the environment). An existing file is kept
# unless --force.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# Self-contained: no lib.sh dependency.
err()  { printf '\033[31m%s\033[0m\n' "$*" >&2; }
info() { printf '  %s\n' "$*"; }

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

PROJECT_DIR="$(dirname "$SCRIPT_DIR")"

ENV_FILE="${PROJECT_DIR}/.env"
CONFIG_FILE="${PROJECT_DIR}/services/searxng/settings.yml"
CONFIG_EXAMPLE="${PROJECT_DIR}/services/searxng/settings.yml.example"
SENTINEL='__SEARXNG_SECRET_KEY__'

FORCE=0
for arg in "$@"; do
  case "$arg" in
    --force)   FORCE=1 ;;
    -h|--help) echo "Usage: $(basename "$0") [--force]"; exit 0 ;;
    *)         err "Unknown: $arg"; exit 2 ;;
  esac
done

[[ -f "$CONFIG_EXAMPLE" ]] || { err "$CONFIG_EXAMPLE not found."; exit 1; }
if [[ -f "$CONFIG_FILE" && $FORCE -eq 0 ]]; then
  info "$CONFIG_FILE already exists — use --force to regenerate."
  exit 0
fi

[[ -f "$ENV_FILE" ]] || { err "$ENV_FILE not found. Run ./scripts/gen-env.sh first."; exit 1; }
# Full regeneration: directory write access only (the container, uid 977, may
# own the existing file).
assert_regen_writable "$CONFIG_FILE" 0 || exit 1
# Single-key extraction rather than `source` (.env values may contain spaces)
SEARXNG_SECRET_KEY="$(grep -E '^SEARXNG_SECRET_KEY=' "$ENV_FILE" | tail -n1 | cut -d= -f2-)"
[[ -n "$SEARXNG_SECRET_KEY" ]] || { err "SEARXNG_SECRET_KEY in .env is empty."; exit 1; }
[[ "$SEARXNG_SECRET_KEY" != change-me-* ]] || { err "SEARXNG_SECRET_KEY is still the placeholder — re-run gen-env.sh."; exit 1; }

tmp="$(mktemp)"
trap 'rm -f "$tmp"' EXIT
# The secret_key line only; the sentinel also appears in a comment.
sed "/^[[:space:]]*secret_key:/ s|${SENTINEL}|${SEARXNG_SECRET_KEY}|" "$CONFIG_EXAMPLE" > "$tmp"

if ! grep -qE "^[[:space:]]*secret_key: \"?${SEARXNG_SECRET_KEY}\"?[[:space:]]*$" "$tmp"; then
  err "secret_key was not substituted — check the ${SENTINEL} line in .example."
  exit 1
fi
mv "$tmp" "$CONFIG_FILE"
trap - EXIT
info "$CONFIG_FILE ← $CONFIG_EXAMPLE (secret injected)"
