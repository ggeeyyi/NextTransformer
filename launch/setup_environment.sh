#!/usr/bin/env bash
set -euo pipefail
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
if [[ " $* " == *" --help "* ]]; then
  echo 'Usage: setup_environment.sh --runtime-dir PATH [--site-config FILE] [--cpu] [--tracking] [--upstream] [runtime options]'
  exit 0
fi
source "$REPO_ROOT/launch/env.sh" "$@"
# An explicit index override wins; otherwise retain the caller's uv/pip settings.
INDEX_ARGS=()
PIP_INDEX_ARGS=()
if [[ -n "${NEXTT_INDEX_URL:-}" ]]; then
  unset UV_INDEX_URL UV_EXTRA_INDEX_URL UV_INDEX
  INDEX_ARGS=(--default-index "$NEXTT_INDEX_URL")
  PIP_INDEX_ARGS=(--index-url "$NEXTT_INDEX_URL")
fi
UV_BIN="$NANOCHAT_RUNTIME/tools/uv"
if [[ ! -x "$UV_BIN" ]]; then
  "${NEXTT_BOOTSTRAP_PYTHON:-python3}" -m pip install "${PIP_INDEX_ARGS[@]}" --target "$NANOCHAT_RUNTIME/tools/uv-bootstrap" 'uv==0.9.5'
  cp "$NANOCHAT_RUNTIME/tools/uv-bootstrap/bin/uv" "$UV_BIN"
fi
"$UV_BIN" python install "$NEXTT_PYTHON_VERSION"
MANAGED_PYTHON="$("$UV_BIN" python find --managed-python "$NEXTT_PYTHON_VERSION")"
if [[ ! -x "$NANOCHAT_VENV/bin/python" ]]; then
  "$UV_BIN" venv --python "$MANAGED_PYTHON" "$NANOCHAT_VENV"
else
  "$NANOCHAT_VENV/bin/python" - "$MANAGED_PYTHON" <<'PY'
import pathlib, sys
expected = pathlib.Path(sys.argv[1]).resolve()
actual = pathlib.Path(sys._base_executable).resolve()
if actual != expected:
    raise SystemExit(f"Existing venv uses {actual}; select a new --venv-dir for {expected}.")
PY
fi
EXTRAS=(--extra gpu)
for arg in "$@"; do
  case "$arg" in
    --cpu) EXTRAS=(--extra cpu) ;;
  esac
done
for arg in "$@"; do
  case "$arg" in
    --upstream) EXTRAS+=(--extra upstream) ;;
    --tracking) EXTRAS+=(--extra tracking) ;;
  esac
done
cd "$REPO_ROOT"
"$UV_BIN" sync --locked "${INDEX_ARGS[@]}" "${EXTRAS[@]}"
"$UV_BIN" pip freeze --python "$NANOCHAT_VENV/bin/python" > "$NANOCHAT_RUNTIME/environment.lock.txt"
