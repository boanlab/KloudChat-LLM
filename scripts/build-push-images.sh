#!/usr/bin/env bash
# Usage: build-push-images.sh [--ns NS] [--tag TAG] [--no-push] [--push-only] [--multi-arch] [SERVICE...]
#
# Build KloudChat's own build images (boanlab/kloudchat-*) + push to Docker Hub.
# Compose files only pull these images (deploy-only) → build/publish is handled here.
# vLLM is excluded as it's an upstream image.
#
#   default         Build all images for the host architecture → push.
#   SERVICE...      Target only specific image short-names (multiple OK). All if omitted.
#                   Available: litellm
#                   amd64-only (only when explicitly selected, excluded from 'build all'): whisper
#
#                   Available: crawl4ai-shim, whisper-shim, code-interpreter,
#                   deep-research
#   --no-push       build only (local use).
#   --push-only     skip build, push local images only.
#   --multi-arch    Build linux/amd64,linux/arm64 simultaneously (buildx) → push. For mixed-node
#                   (amd64 + arm64 GB10) deployment. Needs buildx + QEMU, always pushes.
#   --ns NS         Override namespace (default KLOUDCHAT_IMAGE_NS=boanlab from .env).
#   --tag TAG       Override tag (default KLOUDCHAT_IMAGE_TAG=latest from .env).
#
# Prereq: to push to Docker Hub, run `docker login` first. Guidance shown on failure if no push permission.
# Local build only (no push): build-push-images.sh --no-push  (setup.sh always pulls → overwrites the local copy)
# Re-deploy multi-arch for a single image: build-push-images.sh --multi-arch whisper-shim
set -euo pipefail

__SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$__SCRIPT_DIR/lib.sh"
cd "$__SCRIPT_DIR/.."

# Build targets: "image short-name | dockerfile | build context | platform (optional)".
# Final image name = <NS>/kloudchat-<short>:<TAG>. Must map 1:1 with compose's image:.
# Empty platform field → host arch (default) / amd64+arm64 (--multi-arch). If set, force that.
BUILD_TABLE=(
  "crawl4ai-shim|services/crawl4ai-shim/Dockerfile|services/crawl4ai-shim"
  "whisper-shim|services/whisper-shim/Dockerfile|services/whisper-shim"
  "code-interpreter|services/code-interpreter/Dockerfile|services/code-interpreter"
  "deep-research|services/deep-research/Dockerfile|services/deep-research"
)

# GPU media backends (amd64-only — arm64 delegates STT to OpenRouter), built only on explicit
# selection. Force platform linux/amd64 (won't attempt arm64 even with
# --multi-arch).
MEDIA_TABLE=(
  "whisper|services/whisper/Dockerfile|services/whisper|linux/amd64"
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
    *) SELECTED+=("$1"); shift ;;   # specific image short-name
  esac
done

# NS/TAG: flag > .env > default.
NS="${NS:-$(env_get KLOUDCHAT_IMAGE_NS 2>/dev/null || true)}"; NS="${NS:-boanlab}"
TAG="${TAG:-$(env_get KLOUDCHAT_IMAGE_TAG 2>/dev/null || true)}"; TAG="${TAG:-latest}"
img_of() { echo "${NS}/kloudchat-${1}:${TAG}"; }

# If SERVICE args are given, narrow to those short-names (otherwise the whole
# BUILD_TABLE — excluding MEDIA). Explicit selection searches both BUILD_TABLE +
# MEDIA_TABLE. Nonexistent names are rejected.
if (( ${#SELECTED[@]} )); then
  _filtered=()
  for want in "${SELECTED[@]}"; do
    _hit=0
    for e in "${BUILD_TABLE[@]}" "${MEDIA_TABLE[@]}"; do
      IFS='|' read -r s _ <<<"$e"
      [[ "$s" == "$want" ]] && { _filtered+=("$e"); _hit=1; break; }
    done
    (( _hit )) || { err "unknown image: '$want' (available: $(for e in "${BUILD_TABLE[@]}" "${MEDIA_TABLE[@]}"; do IFS='|' read -r s _ <<<"$e"; printf '%s ' "$s"; done))"; exit 2; }
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
    platforms="${plat:-$PLAT}"   # if the entry specifies a platform (amd64-only media), only that.
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
    # Platform-forced (amd64-only media) entries are built only when the host arch matches.
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
    # Same guard as the build loop — platform-forced entries not buildable on the host arch are also skipped for push.
    if [[ -n "$plat" && "$plat" != *"$host_plat"* ]]; then
      warn "$short is ${plat}-only — no host (${host_plat}) build artifact, skipping push"
      continue
    fi
    echo "  → $img"
    docker push "$img" || { err "push failed: $img — check 'docker login' + ${NS} push permission"; exit 1; }
  done
  ok "push done"
fi
