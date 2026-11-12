#!/usr/bin/env bash
# Time a grid of micro-batch, recomputation, compile and global-batch settings; prints one JSON line per setting.
# Run the same command on every node with the environment of launch/scaling/node.sh.
# Optional: MICRO_BATCHES ("2 4"), RECOMPUTE ("0"; blocks whose activations are recomputed, -1 for all),
# COMPILE ("off on"), BATCH_TOKENS ("2097152 4194304"), STEPS (20).
set -uo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
if [[ " $* " == *" --help "* || $# == 0 ]]; then
  echo 'Usage: NNODES=N NODE_RANK=K MASTER_ADDR=HOST MASTER_PORT=PORT launch/scaling/benchmark.sh CONFIG --site-config FILE [options]'
  exit 0
fi
CONFIG="$1"; shift
for batch in ${BATCH_TOKENS:-2097152 4194304}; do
  for micro in ${MICRO_BATCHES:-2 4}; do
    for recompute in ${RECOMPUTE:-0}; do
      for compile in ${COMPILE:-off on}; do
        flags=(--benchmark "${STEPS:-20}" --benchmark-batch-tokens "$batch" --recompute-layers "$recompute")
        [[ "$compile" == on ]] && flags+=(--compile)
        MICRO_BATCH="$micro" "$ROOT/launch/scaling/node.sh" "$CONFIG" "${flags[@]}" "$@" 2>&1 \
          | grep -E '^\{"event": "benchmark"|OutOfMemoryError: CUDA out of memory' | head -2 \
          || echo "{\"event\": \"benchmark\", \"global_batch_tokens\": $batch, \"micro_batch\": $micro, \"recompute_layers\": $recompute, \"compile\": \"$compile\", \"failed\": true}"
      done
    done
  done
done
