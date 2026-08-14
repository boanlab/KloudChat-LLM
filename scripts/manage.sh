#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/lib.sh"

usage() {
  cat <<'EOF'
Usage: manage.sh <resource> <action> [opts]

team   create  --alias <n> [--budget --duration --tpm --rpm --models a,b,c]
       list / delete --id <team_id>
       sync                                                ← re-sync every team's model allowlist after a catalog change

user   list / delete --id <email>                           ← LiteLLM-side users
       usage  [--user <email>]                             ← per-user usage (spend) vs monthly budget
       topup  --user <email> --amount <N>                  ← temporary monthly-limit raise ($N; spend preserved → accurate stats, auto-reverts next month)

       Accounts themselves are created in kchat (signup → admin approval); kchat's API
       provisions the matching LiteLLM user + per-user key. These commands are the
       LiteLLM-side view/ops for them.

key    issue   --user <email> [--team] [--alias] [--budget]
       issue   --service <name> [--budget]                  ← service-account
       list   [--user <email>]                             ← from LiteLLM, first 20 chars of each key only
       show   [--user <email>]                             ← full plaintext keys from the local ledger
       revoke --key <sk-...>

EOF
  exit 1
}

require_arg() { [[ -n "$2" ]] || { err "$1 required"; exit 1; }; }
require_email() {
  [[ "$2" =~ ^[^@]+@[^@]+\.[^@]+$ ]] || { err "$1 must be email: $2"; exit 1; }
}
# Safe exit under set -u when a flag's value is missing. Usage: `--foo) need_val "$@"; foo="$2"; shift 2 ;;`
need_val() { [[ -n "${2:-}" ]] || { err "$1 requires a value"; exit 1; }; }

cmd_team_create() {
  local alias="" budget=9999 duration=1mo tpm=100000 rpm=500
  local models; models="$(litellm_chat_models_csv)"
  while [[ $# -gt 0 ]]; do
    case "$1" in
      --alias)    need_val "$@"; alias="$2";    shift 2 ;;
      --budget)   need_val "$@"; budget="$2";   shift 2 ;;
      --duration) need_val "$@"; duration="$2"; shift 2 ;;
      --tpm)      need_val "$@"; tpm="$2";      shift 2 ;;
      --rpm)      need_val "$@"; rpm="$2";      shift 2 ;;
      --models)   need_val "$@"; models="$2";   shift 2 ;;
      *) err "Unknown: $1"; exit 1 ;;
    esac
  done
  require_arg --alias "$alias"
  [[ -z "$models" ]] && { err "litellm chat model list is empty — needs OPENROUTER_API_KEY or a vLLM node (or specify directly with --models)"; exit 1; }

  local existing; existing=$(team_id_by_alias "$alias")
  if [[ -n "$existing" ]]; then
    echo "team exists: $alias (id=$existing)"; echo "$existing"; return 0
  fi

  local models_json; models_json=$(echo "$models" | tr ',' '\n' | jq -R . | jq -s .)
  local payload; payload=$(jq -n \
    --arg a "$alias" --argjson b "$budget" --arg d "$duration" \
    --argjson tpm "$tpm" --argjson rpm "$rpm" --argjson m "$models_json" \
    '{team_alias:$a, max_budget:$b, budget_duration:$d, tpm_limit:$tpm, rpm_limit:$rpm, models:$m}')
  local result; result=$(litellm_post "/team/new" "$payload")
  local team_id; team_id=$(echo "$result" | jq -r '.team_id')

  mkdir -p "${DATA_DIR}"
  local cache="${DATA_DIR}/teams.json"
  if [[ -f "$cache" ]]; then
    if jq --argjson r "$result" '. += [$r]' "$cache" > "${cache}.tmp"; then
      mv "${cache}.tmp" "$cache"
    else
      rm -f "${cache}.tmp"
      warn "failed to update teams.json — jq error"
    fi
  else
    echo "[$result]" > "$cache"
  fi
  echo "team created: $alias (id=$team_id)"
  echo "$team_id"
}

cmd_team_list() {
  litellm_get "/team/list" | jq -r '.[] | "\(.team_id)\t\(.team_alias)\tbudget:\(.max_budget)$\tmodels:\(.models // [] | join(","))"'
}

cmd_team_delete() {
  local id=""
  while [[ $# -gt 0 ]]; do case "$1" in --id) need_val "$@"; id="$2"; shift 2 ;; *) shift ;; esac; done
  require_arg --id "$id"
  litellm_post "/team/delete" "{\"team_ids\":[\"$id\"]}" | jq .
}

# Sync every team's model allowlist to the current catalog. If not run after a
# lib.sh catalog change, existing teams reject new models with 401.
cmd_team_sync() {
  local models; models="$(litellm_chat_models_csv)"
  [[ -z "$models" ]] && { err "0 litellm chat models — run gen-litellm-config + restart first"; exit 1; }
  local models_json; models_json=$(echo "$models" | tr ',' '\n' | jq -R . | jq -s .)
  local teams; teams=$(litellm_get "/team/list")
  local n=0 n_ok=0 n_fail=0
  while IFS=$'\t' read -r tid talias; do
    [[ -z "$tid" ]] && continue
    n=$((n+1))
    local payload; payload=$(jq -n --arg id "$tid" --argjson m "$models_json" '{team_id:$id, models:$m}')
    if litellm_post "/team/update" "$payload" >/dev/null 2>&1; then
      n_ok=$((n_ok+1)); ok "$talias ($tid)"
    else
      n_fail=$((n_fail+1)); err "$talias ($tid)"
    fi
  done < <(echo "$teams" | jq -r '.[] | "\(.team_id)\t\(.team_alias)"')
  echo "team/sync: $n_ok/$n updated, $n_fail failed (models: $(echo "$models" | tr ',' '\n' | wc -l))"
}

cmd_user_list() {
  litellm_get "/user/list" | jq -r '
    (if type == "array" then . else (.users // .data // []) end)[]
    | "\(.user_id)\t\(.user_role)\tspend:\(.spend // 0)$"'
}

# Per-user usage (this month's spend) vs monthly budget (max_budget). --user for a single user.
# spend resets every budget_duration (1mo) — the RESET column = next reset date.
cmd_user_usage() {
  local user_id=""
  while [[ $# -gt 0 ]]; do case "$1" in --user) need_val "$@"; user_id="$2"; shift 2 ;; *) shift ;; esac; done
  reconcile_topups
  local out; out="$( { printf 'USER\tSPEND\tBUDGET\tUSED\tRESET\n'
    litellm_get "/user/list" | jq -r --arg u "$user_id" '
      (if type == "array" then . else (.users // .data // []) end)[]
      | select($u == "" or .user_id == $u)
      | [ .user_id,
          "$" + ((.spend // 0)|(.*100|round/100)|tostring),
          (if .max_budget == null then "unlimited" else "$" + (.max_budget|tostring) end),
          (if (.max_budget // 0) > 0 then (((.spend // 0)/.max_budget*100)|floor|tostring)+"%" else "-" end),
          ((.budget_reset_at // "-")[0:10]) ] | @tsv'; } )"
  # Fall back to raw TSV if column is missing (rare) — a fallback on the right side of a pipe can't receive input → use a branch.
  if command -v column >/dev/null 2>&1; then printf '%s\n' "$out" | column -t -s "$(printf '\t')"
  else printf '%s\n' "$out"; fi
}

# Auto-restore expired topups (temporary budget raises). When expires_at (the budget_reset_at at
# raise time) has passed = a budget_duration reset happened in between → restore to original_budget.
# Runs lazily on user usage/list/topup entry (no-op if the ledger is empty). For immediacy, use a start-of-month cron.
reconcile_topups() {
  local f="${DATA_DIR}/topups.json"
  [[ -s "$f" ]] || return 0
  local now; now=$(date +%s)
  local keep='[]' e uid orig exp exp_s changed=0
  while IFS= read -r e; do
    uid=$(jq -r '.user_id' <<<"$e"); orig=$(jq -r '.original_budget' <<<"$e"); exp=$(jq -r '.expires_at' <<<"$e")
    exp_s=$(date -d "$exp" +%s 2>/dev/null || echo 0)
    if (( exp_s > 0 && now >= exp_s )); then
      if [[ "$orig" =~ ^[0-9]+(\.[0-9]+)?$ ]]; then   # prevent max_budget:null (unlimited) if the ledger is corrupted
        litellm_post "/user/update" "$(jq -n --arg u "$uid" --argjson b "$orig" '{user_id:$u, max_budget:$b}')" >/dev/null 2>&1 \
          && info "topup expired → restored: $uid monthly limit \$$orig"
        changed=1
      else
        warn "invalid original_budget in topup ledger ('$orig') — keeping $uid entry, manual check needed"
        keep=$(jq -c --argjson x "$e" '. + [$x]' <<<"$keep")
      fi
    else
      keep=$(jq -c --argjson x "$e" '. + [$x]' <<<"$keep")
    fi
  done < <(jq -c '.[]' "$f" 2>/dev/null)
  if (( changed )); then echo "$keep" > "$f"; chmod 600 "$f"; fi
  return 0
}

# Temporary budget topup — raise the monthly limit (max_budget) by amount. spend (actual usage)
# is kept as-is so statistics stay accurate. Record original_budget and expiry (= current
# budget_reset_at) in the ledger (data/ledger/topups.json) → after the monthly reset
# reconcile_topups auto-restores the original limit (not a permanent raise). Re-topups in the same month accumulate; original keeps the first topup value.
cmd_user_topup() {
  local user_id="" amount=""
  while [[ $# -gt 0 ]]; do case "$1" in
    --user)   need_val "$@"; user_id="$2"; shift 2 ;;
    --amount) need_val "$@"; amount="$2"; shift 2 ;;
    *) shift ;; esac; done
  require_arg --user "$user_id"
  require_arg --amount "$amount"
  [[ "$amount" =~ ^[0-9]+(\.[0-9]+)?$ ]] || { err "--amount must be a positive number (in \$): $amount"; return 1; }
  reconcile_topups
  local uinfo; uinfo=$(litellm_get "/user/info?user_id=${user_id}") || { err "failed to look up user: $user_id"; return 1; }
  local curbud spend reset
  curbud=$(echo "$uinfo" | jq -r 'if .user_info.max_budget == null then "" else (.user_info.max_budget|tostring) end')
  [[ -z "$curbud" ]] && { err "this user has no monthly limit (max_budget) (unlimited) and cannot be topped up: $user_id"; return 1; }
  spend=$(echo "$uinfo" | jq -r '.user_info.spend // 0')
  reset=$(echo "$uinfo" | jq -r '.user_info.budget_reset_at // empty')
  local expires
  if [[ -n "$reset" && "$reset" != "null" ]]; then expires="$reset"
  else expires="$(date -d "$(date +%Y-%m-01) +1 month" +%Y-%m-%dT00:00:00Z)"; fi
  local f="${DATA_DIR}/topups.json"; mkdir -p "$DATA_DIR"; [[ -s "$f" ]] || echo '[]' > "$f"
  # original = the first limit of an in-progress topup (if any), otherwise the current limit.
  local original; original=$(jq -r --arg u "$user_id" 'first(.[] | select(.user_id==$u) | .original_budget) // empty' "$f")
  [[ -z "$original" || "$original" == "null" ]] && original="$curbud"
  local new; new=$(awk -v c="$curbud" -v a="$amount" 'BEGIN{printf "%.2f", c+a}')
  litellm_post "/user/update" "$(jq -n --arg u "$user_id" --argjson b "$new" '{user_id:$u, max_budget:$b}')" >/dev/null \
    || { err "top-up failed: $user_id"; return 1; }
  jq --arg u "$user_id" --argjson o "$original" --arg e "$expires" \
     'map(select(.user_id != $u)) + [{user_id:$u, original_budget:$o, expires_at:$e}]' "$f" > "$f.tmp" \
     && mv "$f.tmp" "$f" && chmod 600 "$f"
  ok "top-up: ${user_id} +\$${amount}  (monthly limit \$${curbud} → \$${new}; spend \$${spend} preserved; auto-reverts to \$${original} after the ${expires%%T*} reset)"
}

cmd_user_delete() {
  local id=""
  while [[ $# -gt 0 ]]; do case "$1" in --id) need_val "$@"; id="$2"; shift 2 ;; *) shift ;; esac; done
  require_arg --id "$id"
  litellm_post "/user/delete" "{\"user_ids\":[\"$id\"]}" | jq .
}

# Append the issued plaintext key to the data/ledger/keys.json ledger. LiteLLM stores only the
# hash → this is the sole source for re-checking the plaintext after issuance. data/ is
# gitignored, and the file is locked to 600.
record_issued_key() {
  local user_id="$1" key_alias="$2" team_id="$3" key="$4" budget="$5"
  mkdir -p "$DATA_DIR"
  local cache="${DATA_DIR}/keys.json"
  [[ -f "$cache" ]] || { echo '[]' > "$cache"; chmod 600 "$cache"; }
  local entry; entry=$(jq -n \
    --arg u "$user_id" --arg a "$key_alias" --arg t "$team_id" \
    --arg k "$key" --argjson b "$budget" --arg ts "$(date -u +%Y-%m-%dT%H:%M:%SZ)" \
    '{user_id:$u, key_alias:$a, team_id:$t, key:$k, max_budget:$b, issued_at:$ts}')
  if jq --argjson e "$entry" '. += [$e]' "$cache" > "${cache}.tmp"; then
    mv "${cache}.tmp" "$cache"; chmod 600 "$cache"
  else
    rm -f "${cache}.tmp"; warn "failed to write keys.json — jq error"
  fi
}

cmd_key_issue() {
  local user_id="" team_alias="default" key_alias="" budget=9999 service=""
  while [[ $# -gt 0 ]]; do
    case "$1" in
      --user)    need_val "$@"; user_id="$2";    shift 2 ;;
      --team)    need_val "$@"; team_alias="$2"; shift 2 ;;
      --alias)   need_val "$@"; key_alias="$2";  shift 2 ;;
      --budget)  need_val "$@"; budget="$2";     shift 2 ;;
      --service) need_val "$@"; service="$2";    shift 2 ;;
      *) err "Unknown: $1"; exit 1 ;;
    esac
  done

  if [[ -n "$service" ]]; then
    local alias="${service}-service-key"
    # LiteLLM stores only the key hash. The plaintext is hardcoded into .env at issue time.
    # An alias clash makes reissue return HTTP 400 → skip if the .env key is still valid.
    local payload; payload=$(jq -n --arg a "$alias" --argjson b "$budget" \
      '{key_alias:$a, max_budget:$b, budget_duration:"1mo", user_role:"internal_user"}')
    local result
    if ! result=$(litellm_post "/key/generate" "$payload" 2>&1); then
      if echo "$result" | grep -q 'already exists'; then
        # alias exists in LiteLLM but .env is out of sync — manual rotation
        # required (DB has no plaintext to recover).
        err "service key alias '$alias' already exists in LiteLLM."
        err "  → check with ./scripts/manage.sh key list, then rotate: key revoke --key <stale>"
        err "    or use a different alias: key issue --service $service --alias <new>"
        return 1
      fi
      err "$result"; return 1
    fi
    local key; key=$(echo "$result" | jq -r '.key')
    echo "service key: $service"; echo "KEY: $key"
    record_issued_key "" "$alias" "" "$key" "$budget"
    return 0
  fi

  require_arg "--user or --service" "$user_id"
  require_email --user "$user_id"

  local team_id=""
  if [[ -n "$team_alias" ]]; then
    team_id=$(team_id_by_alias "$team_alias")
    [[ -z "$team_id" ]] && { err "team not found: $team_alias"; exit 1; }
  fi
  [[ -z "$key_alias" ]] && key_alias="${user_id%%@*}-key"

  local payload; payload=$(jq -n \
    --arg u "$user_id" --arg t "$team_id" --arg a "$key_alias" --argjson b "$budget" \
    '{user_id:$u, team_id:$t, key_alias:$a, max_budget:$b, budget_duration:"1mo"}')
  local result; result=$(litellm_post "/key/generate" "$payload")
  local key; key=$(echo "$result" | jq -r '.key')
  echo "key: $key_alias"
  echo "KEY: $key"
  record_issued_key "$user_id" "$key_alias" "$team_id" "$key" "$budget"
}

cmd_key_list() {
  local user_id=""
  while [[ $# -gt 0 ]]; do case "$1" in --user) need_val "$@"; user_id="$2"; shift 2 ;; *) shift ;; esac; done
  # Without return_full_object=true, .keys[] are hash strings → object indexing fails.
  local ep="/key/list?return_full_object=true"; [[ -n "$user_id" ]] && ep+="&user_id=${user_id}"
  litellm_get "$ep" | jq -r '.keys[]
    | "\(.key_alias // "unnamed")\t\((.token // "?")[0:20])...\tuser:\(.user_id // "-")\tbudget:\(.max_budget)$\tspend:\(.spend // 0)$"'
}

# Show the plaintext keys stored in the ledger (data/ledger/keys.json). Filter with --user.
# LiteLLM's key list only knows the hash (first 20 chars only); this shows the full plaintext.
cmd_key_show() {
  local user_id=""
  while [[ $# -gt 0 ]]; do case "$1" in --user) need_val "$@"; user_id="$2"; shift 2 ;; *) shift ;; esac; done
  local cache="${DATA_DIR}/keys.json"
  [[ -f "$cache" ]] || { err "no stored keys ($cache not created) — recorded only after key issue"; return 1; }
  local rows; rows=$(jq -r --arg u "$user_id" \
    '[.[] | select($u == "" or .user_id == $u)][]
     | "\(.user_id // "-")\t\(.key_alias)\t\(.key)\tbudget:\(.max_budget)$\t\(.issued_at)"' "$cache")
  [[ -z "$rows" ]] && { err "no matching stored keys${user_id:+ (user=$user_id)}"; return 1; }
  echo "$rows"
}

cmd_key_revoke() {
  local key=""
  while [[ $# -gt 0 ]]; do case "$1" in --key) need_val "$@"; key="$2"; shift 2 ;; *) shift ;; esac; done
  require_arg --key "$key"
  litellm_post "/key/delete" "{\"keys\":[\"$key\"]}" | jq .
  # Remove from the ledger too — so a revoked key doesn't linger in key show.
  local cache="${DATA_DIR}/keys.json"
  if [[ -f "$cache" ]] && jq --arg k "$key" 'map(select(.key != $k))' "$cache" > "${cache}.tmp" 2>/dev/null; then
    mv "${cache}.tmp" "$cache"; chmod 600 "$cache"
  else
    rm -f "${cache}.tmp"
  fi
}

(( $# < 2 )) && usage
resource="$1"; action="$2"; shift 2
case "${resource}/${action}" in
  team/create)  cmd_team_create "$@" ;;
  team/list)    cmd_team_list ;;
  team/delete)  cmd_team_delete "$@" ;;
  team/sync)    cmd_team_sync ;;
  user/list)    cmd_user_list ;;
  user/usage)   cmd_user_usage "$@" ;;
  user/topup)   cmd_user_topup "$@" ;;
  user/delete)  cmd_user_delete "$@" ;;
  key/issue)    cmd_key_issue "$@" ;;
  key/list)     cmd_key_list "$@" ;;
  key/show)     cmd_key_show "$@" ;;
  key/revoke)   cmd_key_revoke "$@" ;;
  *)            err "Unknown: ${resource} ${action}"; usage ;;
esac
