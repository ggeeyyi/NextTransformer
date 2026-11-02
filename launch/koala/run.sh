#!/usr/bin/env bash
# Koala H20 container entry: one reference model on one node, or one node of
# a two-node job (NEXTT_NODES=2, submitted with -n 2 -g 8).
#
#   bash launch/koala/run.sh --site-config SITE.json \
#       --config configs/linear/kda.json --output RUN_DIR --micro-batch N [nexttransformer.train options]
#
# Training is formal (--formal, cost admission enforced); pass --research to override.
#
# SITE.json is a machine file kept outside the code bundle: runtime_dir on the
# pod-local disk, resources/data_dir and an optional W&B netrc on persistent
# storage. The virtual environment and every cache stay on the pod-local disk.
# GPUS (default 8) sets the process count per node; formal runs use 8 or 16 ranks.
# With NEXTT_NODES > 1 every node runs this script; the nodes agree on ranks
# and the master address through a job-scoped directory on persistent storage
# (RUN_DIR/.nextt-rendezvous, or NEXTT_RENDEZVOUS_DIR), ordered by host name.
# NEXTT_TRITON_CACHE_ARCHIVE (optional) restores/saves the kernel caches (Triton,
# and the optimizer's Inductor cache inside it; one directory per local rank,
# nexttransformer.train) as one tar file per node, so every rank of a later segment
# reuses the kernel configurations it chose before.
set -euo pipefail
cd "$(dirname "$0")/../.."

export UV_PROJECT_ENVIRONMENT="${NEXTT_VENV:-/local-ssd/nextt-venv}"
export UV_LINK_MODE=copy
export NANOCHAT_DTYPE=bfloat16 HF_HUB_OFFLINE=1 FLA_DISABLE_BACKEND_DISPATCH=1 PYTHONUNBUFFERED=1
export TRITON_CACHE_DIR="${TRITON_CACHE_DIR:-/local-ssd/triton-cache}"
export TORCHINDUCTOR_CACHE_DIR="${TORCHINDUCTOR_CACHE_DIR:-$TRITON_CACHE_DIR/inductor}"  # archived with it
GPUS="${GPUS:-8}"
NODES="${NEXTT_NODES:-1}"

output=""
args=("$@")
for ((i = 0; i < ${#args[@]}; i++)); do
  [[ "${args[i]}" == "--output" ]] && output="${args[i+1]:-}"
done

# One installer before any rank starts; locked dependencies plus hash-pinned FLA.
uv sync --frozen --extra gpu --extra upstream --extra tracking
uv pip install --python "$UV_PROJECT_ENVIRONMENT/bin/python" --no-deps --require-hashes \
  -r requirements/linear.txt

launch=(--standalone --nproc-per-node="$GPUS")
node_rank=0
if (( NODES > 1 )); then
  # One rendezvous per launch; every node counts its own launches in the same order.
  count=$(( $(cat /local-ssd/nextt-launches 2>/dev/null || echo 0) + 1 )); echo "$count" > /local-ssd/nextt-launches
  [[ -n "$output" ]] || { echo "multi-node launch needs --output" >&2; exit 2; }
  rendezvous="${NEXTT_RENDEZVOUS_DIR:-$output/.nextt-rendezvous}/${KOALA_JOB_NAME:?}/$count"
  mkdir -p "$rendezvous"
  host=$(hostname)
  hostname -I | awk '{print $1}' > "$rendezvous/.$host" && mv "$rendezvous/.$host" "$rendezvous/$host"
  for ((i = 0; i < 180; i++)); do (( $(ls "$rendezvous" | wc -l) >= NODES )) && break; sleep 5; done
  hosts=($(ls "$rendezvous" | sort))
  (( ${#hosts[@]} == NODES )) || { echo "rendezvous: ${#hosts[@]} of $NODES nodes in $rendezvous" >&2; exit 1; }
  for ((i = 0; i < NODES; i++)); do [[ "${hosts[i]}" == "$host" ]] && node_rank=$i; done
  master=$(cat "$rendezvous/${hosts[0]}")
  echo "NEXTT_NODE rank $node_rank of $NODES, master $master, launch $count"
  launch=(--nnodes="$NODES" --node-rank="$node_rank" --master-addr="$master"
          --master-port=$((29500 + count)) --nproc-per-node="$GPUS")
fi

echo "$node_rank" > /local-ssd/nextt-node-rank  # for wrappers such as bounded_check.sh
mkdir -p "$TRITON_CACHE_DIR"
archive="${NEXTT_TRITON_CACHE_ARCHIVE:-}"
# Each node keeps its own autotune archive, so no two nodes write one file.
if [[ -n "$archive" && $node_rank -gt 0 ]]; then archive="${archive%.tar}.node$node_rank.tar"; fi
if [[ -n "$archive" && -f "$archive" ]]; then tar xf "$archive" -C "$TRITON_CACHE_DIR"; fi

set +e
uv run --no-sync python -m torch.distributed.run "${launch[@]}" -m nexttransformer.train --formal "$@"
code=$?
set -e

if [[ -n "$archive" ]]; then
  tar cf "$archive.tmp" -C "$TRITON_CACHE_DIR" . && mv "$archive.tmp" "$archive"
fi
if [[ -n "$output" && -f "$output/run_status.json" ]]; then
  # One machine-readable line for the local segment driver (chain.sh).
  echo "NEXTT_RUN_STATUS $(uv run --no-sync python -c 'import json,sys;s=json.load(open(sys.argv[1]));print(s["phase"],s["step"],s["final_status"])' "$output/run_status.json")"
fi
exit "$code"
