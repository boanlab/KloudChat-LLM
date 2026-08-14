#!/usr/bin/env bash
# Usage: tune-host.sh [--check]
#
# sysctl tuning for LLM-serving hosts. Model weight mmap occupies file cache by
# the GB → with the default swappiness=60, file cache is reclaimed during idle →
# cold start on the next inference. GB10 unified-memory nodes suffer especially
# heavy file-cache pressure.
#
#   --check   show current values only, without applying
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/lib.sh"

CHECK_ONLY=0
for arg in "$@"; do
  case "$arg" in
    --check)    CHECK_ONLY=1 ;;
    -h|--help)  grep -E '^# ' "$0" | sed 's/^# \{0,1\}//'; exit 0 ;;
    *)          err "Unknown: $arg"; exit 2 ;;
  esac
done

CONF=/etc/sysctl.d/99-kloudchat-tuning.conf

hdr "Host tuning (sysctl)"

CUR_SWAPPINESS="$(cat /proc/sys/vm/swappiness)"
echo "  current vm.swappiness = $CUR_SWAPPINESS"

if has_gb10; then
  info "GB10 unified memory detected — lowering swappiness recommended"
fi

if (( CHECK_ONLY )); then
  [[ -f "$CONF" ]] && { ok "$CONF exists"; cat "$CONF" | sed 's/^/    /'; } \
                   || warn "$CONF not found — not applied"
  exit 0
fi

[[ $EUID -ne 0 ]] && exec sudo "$0" "$@"

cat > "$CONF" <<'EOF'
# KloudChat host tuning — for LLM serving
# Model weights occupy file cache via mmap, so protect file cache over anon.
vm.swappiness = 10
EOF
ok "$CONF written"

sysctl --system >/dev/null
NEW="$(cat /proc/sys/vm/swappiness)"
if [[ "$NEW" == "10" ]]; then
  ok "vm.swappiness = $NEW (applied, persists across reboot)"
else
  err "Apply failed — current value $NEW. Another sysctl file may be overriding it"
  err "  check: sysctl --system 2>&1 | grep swappiness"
  exit 1
fi

echo
info "Existing swapped-out pages are not auto-swapped-in. To force-clear them:"
echo "    sudo swapoff -a && sudo swapon -a   # safe only when swap used is less than free RAM"
