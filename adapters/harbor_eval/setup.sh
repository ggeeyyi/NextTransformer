#!/usr/bin/env bash
# Isolated controller: use a caller-supplied Python 3.12 and pinned uv executable.
set -euo pipefail
: "${HARBOR_PYTHON:?Set the path to Python 3.12}"
: "${HARBOR_VENV:?Set an isolated controller environment path}"
: "${UV_BIN:?Set the path to uv 0.9.5}"
root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
"$HARBOR_PYTHON" -c 'import sys; assert sys.version_info[:2] == (3, 12)'
"$UV_BIN" venv --python "$HARBOR_PYTHON" "$HARBOR_VENV"
"$UV_BIN" pip sync --python "$HARBOR_VENV/bin/python" --require-hashes --only-binary :all: \
    --index-url "${PYPI_INDEX_URL:-https://repo.huaweicloud.com/repository/pypi/simple}" \
    "$root/adapters/harbor_eval/requirements.lock"
"$UV_BIN" pip check --python "$HARBOR_VENV/bin/python"
