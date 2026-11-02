#!/usr/bin/env bash
# Build locally using mirrored downloads. Does not publish, run eval, or select GPUs.
set -euo pipefail
repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
tag="${NEXTT_IMAGE_TAG:-nextt-runtime:cu128-candidate}"
builder_args=()
if [[ -n "${NEXTT_BUILDER:-}" ]]; then builder_args=(--builder "$NEXTT_BUILDER"); fi
build_args=()
for name in CUDA_IMAGE UV_IMAGE APT_MIRROR PYPI_INDEX_URL PYTHON_MIRROR; do
    if [[ -n "${!name:-}" ]]; then build_args+=(--build-arg "$name=${!name}"); fi
done
docker buildx build "${builder_args[@]}" --platform linux/amd64 --load \
    --progress plain --file "$repo_root/environments/eval/Dockerfile" \
    --build-arg "SOURCE_REVISION=$(git -C "$repo_root" rev-parse HEAD)" \
    "${build_args[@]}" --tag "$tag" "$repo_root"
docker image inspect "$tag" --format '{{.Id}}'
