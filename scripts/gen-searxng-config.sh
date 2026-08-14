#!/usr/bin/env bash
# Generates services/searxng/settings.yml from the .example template.
# Substitute the .example's __SEARXNG_SECRET_KEY__ sentinel with .env's SEARXNG_SECRET_KEY.
# SearXNG can't override secret_key via env var (settings_loader only recognizes
# SEARXNG_SETTINGS_PATH/SEARXNG_DISABLE_ETC_SETTINGS) → merge once on the host to
# produce settings.yml, then bind-mount.
#
# Idempotent skip if a live file exists — preserves user customizations (engines/UI etc.). Use --force to regenerate.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# lib.sh lives in the operator repo. These three are all this script used from
# it, and copying them is cheaper than depending on a sibling checkout — the UI
# plane has to come up on its own.
err()  { printf '\033[31m%s\033[0m\n' "$*" >&2; }
say()  { printf '  %s\n' "$*"; }
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

hdr()  { printf '\n\033[1m%s\033[0m\n' "$*"; }

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
# Once the container (uid 977) has run, searxng/ is owned by 977 → on --force
# regeneration the mv gets Permission denied. Since it's a full regeneration, no
# need to read the existing file (need_read=0); only verify directory write access.
assert_regen_writable "$CONFIG_FILE" 0 || exit 1
# Don't `source` — spaces in other values (Gmail app pw etc.) would be a syntax error. Extract the single key only.
SEARXNG_SECRET_KEY="$(grep -E '^SEARXNG_SECRET_KEY=' "$ENV_FILE" | tail -n1 | cut -d= -f2-)"
[[ -n "$SEARXNG_SECRET_KEY" ]] || { err "SEARXNG_SECRET_KEY in .env is empty."; exit 1; }
[[ "$SEARXNG_SECRET_KEY" != change-me-* ]] || { err "SEARXNG_SECRET_KEY is still the placeholder — re-run gen-env.sh."; exit 1; }

# sed delimiter = a char that almost never appears in the secret. Just in case, avoid |, /, # all.
# The secret is gen-env.sh's openssl rand -hex output so it's [0-9a-f] only — no collision.
tmp="$(mktemp)"
trap 'rm -f "$tmp"' EXIT
sed "s|${SENTINEL}|${SEARXNG_SECRET_KEY}|" "$CONFIG_EXAMPLE" > "$tmp"

if grep -qF "$SENTINEL" "$tmp"; then
  err "sentinel substitution failed — check the $SENTINEL format in .example."
  exit 1
fi
mv "$tmp" "$CONFIG_FILE"
trap - EXIT
info "$CONFIG_FILE ← $CONFIG_EXAMPLE (secret injected)"
