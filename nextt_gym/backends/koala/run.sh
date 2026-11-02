#!/usr/bin/env bash
# Koala pod entry for one gym run segment, the gym form of launch/koala/run.sh:
#
#   bash nextt_gym/backends/koala/run.sh <nextt_gym.remote run arguments>
#
# Runs from the repository unpacked out of the gym bundle on the pod-local disk. Like run.sh it
# installs the locked dependencies plus hash-pinned FLA into a virtual environment on /local-ssd,
# keeps every cache there, and then hands over to `nextt_gym.remote run`, which verifies the bundle
# again and launches the trainer. NEXTT_VENV overrides the environment directory.
set -euo pipefail
cd "$(dirname "$0")/../../.."

export UV_PROJECT_ENVIRONMENT="${NEXTT_VENV:-/local-ssd/nextt-venv}"
export UV_LINK_MODE=copy
export NANOCHAT_DTYPE=bfloat16 HF_HUB_OFFLINE=1 FLA_DISABLE_BACKEND_DISPATCH=1 PYTHONUNBUFFERED=1
export TRITON_CACHE_DIR="${TRITON_CACHE_DIR:-/local-ssd/triton-cache}"
export TORCHINDUCTOR_CACHE_DIR="${TORCHINDUCTOR_CACHE_DIR:-$TRITON_CACHE_DIR/inductor}"

uv sync --frozen --extra gpu --extra upstream
uv pip install --python "$UV_PROJECT_ENVIRONMENT/bin/python" --no-deps --require-hashes \
  -r requirements/linear.txt
mkdir -p "$TRITON_CACHE_DIR"
exec uv run --no-sync python -m nextt_gym.remote "$@"
