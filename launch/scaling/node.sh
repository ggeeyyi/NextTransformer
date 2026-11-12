#!/usr/bin/env bash
# Start this node's ranks of a multi-node scaling run (or a single-node run with NNODES=1).
# Required: NNODES, NODE_RANK, MASTER_ADDR, MASTER_PORT. Optional: GPUS (default 8), MICRO_BATCH (default 2),
# OUTPUT_DIR, and SCALING_ENV, a file of site-specific exports (network settings, W&B key) kept outside the repository.
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
if [[ " $* " == *" --help "* || $# == 0 ]]; then
  echo 'Usage: NNODES=N NODE_RANK=K MASTER_ADDR=HOST MASTER_PORT=PORT launch/scaling/node.sh CONFIG --site-config FILE [training options]'
  exit 0
fi
CONFIG="$1"; shift
source "$ROOT/launch/env.sh" "$@"
if [[ -n "${SCALING_ENV:-}" ]]; then source "$SCALING_ENV"; fi
: "${NNODES:?}" "${NODE_RANK:?}" "${MASTER_ADDR:?}" "${MASTER_PORT:?}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
cd "$ROOT"
OUTPUT="${OUTPUT_DIR:-$NANOCHAT_RUNTIME/runs/${NEXTTRANSFORMER_RUN_NAME:-scaling_$(basename "$CONFIG" .json)}}"
exec "$NANOCHAT_VENV/bin/python" -m torch.distributed.run --nnodes="$NNODES" --node_rank="$NODE_RANK" \
  --nproc_per_node="${GPUS:-8}" --master_addr="$MASTER_ADDR" --master_port="$MASTER_PORT" \
  -m nexttransformer.scaling.train --config "$CONFIG" --output "$OUTPUT" --micro-batch "${MICRO_BATCH:-2}" "$@"
