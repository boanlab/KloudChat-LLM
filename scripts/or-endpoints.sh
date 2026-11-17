#!/usr/bin/env bash
# Which OpenRouter endpoints may an open-weight model run on? Read-only, needs no key.
#   or-endpoints.sh <model-glob>...   per model: provider, quantization, price, context, uptime, ZDR
#   or-endpoints.sh --summary         the open-weight models in the declared lists: eligible count
# --all also lists endpoints that are not eligible (undisclosed quantization or no ZDR).
# Eligible = discloses its quantization AND is a ZDR endpoint (OR_OPEN_WEIGHT_POLICY=strict).
# Pick one with OR_PROVIDER_PIN=<model>=<Provider>[|<Provider>] in .env.
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/lib.sh"

ALL=0 SUMMARY=0 PATTERNS=()
for arg in "$@"; do
  case "$arg" in
    --all) ALL=1 ;;
    --summary) SUMMARY=1 ;;
    -h|--help) sed -n 2,7p "${BASH_SOURCE[0]}" | sed 's/^# \{0,1\}//'; exit 0 ;;
    -*) err "Unknown: $arg"; exit 2 ;;
    *) PATTERNS+=("$arg") ;;
  esac
done
command -v jq >/dev/null || { err "jq is required"; exit 1; }
(( SUMMARY )) || (( ${#PATTERNS[@]} )) || { err "give a model glob (e.g. 'moonshotai/*') or --summary"; exit 2; }

W="$(mktemp -d -t kc-or-endpoints.XXXXXX)"; export KC_OR_WORK="$W"; trap 'rm -rf "$W"' EXIT
zdr="$(or_zdr_file)" || { err "could not read /api/v1/endpoints/zdr"; exit 1; }

# Slugs to show
slugs=()
if (( SUMMARY )); then
  for prov in "${OR_PROVIDERS[@]}"; do
    var="${prov##*:}_MODELS[@]"
    for m in "${!var}"; do
      slug="${prov%%:*}/$m"
      OR_OPEN_WEIGHT_POLICY=strict __or_is_open_weight "$slug" && slugs+=("$slug")
    done
  done
else
  __or_fetch models "$W/models.json" || { err "could not read /api/v1/models"; exit 1; }
  while IFS= read -r id; do
    for pat in "${PATTERNS[@]}"; do
      # shellcheck disable=SC2053
      [[ "$id" == $pat ]] && { slugs+=("$id"); break; }
    done
  done < <(jq -r '.data[].id' "$W/models.json" | grep -v ':free$' | sort)
fi

(( ${#slugs[@]} )) || { warn "no model matches"; exit 0; }
flag=(); (( ALL )) && flag=(--all)
for slug in "${slugs[@]}"; do
  ep="$(or_endpoints_file "$slug" || true)"
  if [[ -z "$ep" ]]; then
    (( SUMMARY )) && printf '%-48s %s\n' "$slug" "unavailable"
    continue
  fi
  if (( SUMMARY )); then
    n="$(python3 "${SCRIPT_DIR}/or_policy.py" table --endpoints "$ep" --zdr "$zdr" | grep -vc '^provider\|^(' || true)"
    printf '%-48s %s eligible%s\n' "$slug" "$n" "$([[ "$n" == 0 ]] && echo ' -> left out by strict policy')"
  else
    echo "== $slug"
    python3 "${SCRIPT_DIR}/or_policy.py" table --endpoints "$ep" --zdr "$zdr" "${flag[@]}"
    echo
  fi
done
