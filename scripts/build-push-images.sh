#!/usr/bin/env bash
# Usage: build-push-images.sh [--ns NS] [--tag TAG] [--no-push] [--push-only] [--multi-arch] [SERVICE...]
#
# Manual build + push of KloudChat's service images (<NS>/kloudchat-*). The
# usual path is .github/workflows/publish-images.yml; setup.sh pulls what is
# published. vLLM is built per node by install-vllm.sh.
#
#   SERVICE...      image short-names to build (default: all):
#                   crawl4ai-shim, search-shim, whisper-shim, code-interpreter, deep-research, index-shim
#   --no-push       build only
#   --push-only     push existing local images only
#   --multi-arch    linux/amd64,linux/arm64 via buildx (needs buildx + QEMU; always pushes)
#   --ns NS         namespace (default KLOUDCHAT_IMAGE_NS from .env, else boanlab)
#   --tag TAG       tag (default KLOUDCHAT_IMAGE_TAG from .env, else latest)
#
# Pushing needs `docker login`.
set -euo pipefail

__SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$__SCRIPT_DIR/lib.sh"
cd "$__SCRIPT_DIR/.."

# "short-name|dockerfile|context[|platform]". Image = <NS>/kloudchat-<short>:<TAG>,
# matching compose's image:. A platform field forces that platform.
BUILD_TABLE=(
  "crawl4ai-shim|services/crawl4ai-shim/Dockerfile|services/crawl4ai-shim"
  "search-shim|services/search-shim/Dockerfile|services/search-shim"
  "whisper-shim|services/whisper-shim/Dockerfile|services/whisper-shim"
  "code-interpreter|services/code-interpreter/Dockerfile|services/code-interpreter"
  "deep-research|services/deep-research/Dockerfile|services/deep-research"
  "index-shim|services/index-shim/Dockerfile|services/index-shim"
)

NS=""; TAG=""; DO_BUILD=1; DO_PUSH=1; MULTI=0; SELECTED=()
while [[ $# -gt 0 ]]; do
  case "$1" in
    --ns)         NS="${2:?--ns value required}"; shift 2 ;;
    --tag)        TAG="${2:?--tag value required}"; shift 2 ;;
    --no-push)    DO_PUSH=0; shift ;;
    --push-only)  DO_BUILD=0; shift ;;
    --multi-arch) MULTI=1; shift ;;
    -h|--help) sed -n '2,/^set -/p' "$0" | sed 's/^# \{0,1\}//;/^set -/d'; exit 0 ;;
    -*) err "unknown option: $1"; exit 2 ;;
    *) SELECTED+=("$1"); shift ;;
  esac
done

NS="${NS:-$(env_get KLOUDCHAT_IMAGE_NS 2>/dev/null || true)}"; NS="${NS:-boanlab}"
TAG="${TAG:-$(env_get KLOUDCHAT_IMAGE_TAG 2>/dev/null || true)}"; TAG="${TAG:-latest}"
img_of() { echo "${NS}/kloudchat-${1}:${TAG}"; }

if (( ${#SELECTED[@]} )); then
  _filtered=()
  for want in "${SELECTED[@]}"; do
    _hit=0
    for e in "${BUILD_TABLE[@]}"; do
      IFS='|' read -r s _ <<<"$e"
      [[ "$s" == "$want" ]] && { _filtered+=("$e"); _hit=1; break; }
    done
    (( _hit )) || { err "unknown image: '$want' (available: $(for e in "${BUILD_TABLE[@]}"; do IFS='|' read -r s _ <<<"$e"; printf '%s ' "$s"; done))"; exit 2; }
  done
  BUILD_TABLE=("${_filtered[@]}")
fi

hdr "KloudChat images ${NS}/kloudchat-*:${TAG}  (build=${DO_BUILD} push=${DO_PUSH} multi-arch=${MULTI})"
for e in "${BUILD_TABLE[@]}"; do IFS='|' read -r s _ _ <<<"$e"; echo "  $(img_of "$s")"; done

if (( MULTI )); then
  (( DO_PUSH )) || { err "--multi-arch requires push (multi-platform can't be loaded locally). can't be used with --no-push"; exit 2; }
  docker buildx version >/dev/null 2>&1 || { err "docker buildx required (multi-arch)"; exit 1; }
  if ! docker buildx inspect kloudchat-builder >/dev/null 2>&1; then
    info "creating buildx builder (docker-container driver)"
    docker buildx create --name kloudchat-builder --driver docker-container --bootstrap >/dev/null
  fi
  PLAT="linux/amd64,linux/arm64"
  for e in "${BUILD_TABLE[@]}"; do
    IFS='|' read -r short df ctx plat <<<"$e"; img="$(img_of "$short")"
    platforms="${plat:-$PLAT}"
    hdr "buildx ${img}  [${platforms}]"
    docker buildx build --builder kloudchat-builder --platform "$platforms" \
      -t "$img" -f "$df" --push "$ctx"
  done
  ok "multi-arch build+push done (${#BUILD_TABLE[@]})"
  exit 0
fi

if (( DO_BUILD )); then
  hdr "build (host arch)"
  host_plat="linux/$(detect_arch)"
  for e in "${BUILD_TABLE[@]}"; do
    IFS='|' read -r short df ctx plat <<<"$e"; img="$(img_of "$short")"
    if [[ -n "$plat" && "$plat" != *"$host_plat"* ]]; then
      warn "$short is ${plat}-only — can't build on host (${host_plat}), skipping (run on an amd64 node)"
      continue
    fi
    echo "  → build $img"
    docker build -t "$img" -f "$df" "$ctx"
  done
  ok "build done"
fi

if (( DO_PUSH )); then
  hdr "push → Docker Hub"
  docker info 2>/dev/null | grep -q "Username:" || warn "docker login not confirmed — if push fails, run 'docker login' first."
  host_plat="linux/$(detect_arch)"
  for e in "${BUILD_TABLE[@]}"; do
    IFS='|' read -r short _ _ plat <<<"$e"; img="$(img_of "$short")"
    if [[ -n "$plat" && "$plat" != *"$host_plat"* ]]; then
      warn "$short is ${plat}-only — no host (${host_plat}) build artifact, skipping push"
      continue
    fi
    echo "  → $img"
    docker push "$img" || { err "push failed: $img — check 'docker login' + ${NS} push permission"; exit 1; }
  done
  ok "push done"
fi
