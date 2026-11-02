#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
if [[ " $* " == *" --help "* || $# == 0 ]]; then
  echo 'Usage: native_a800.sh|native_h200.sh dense|mor|loopformer|circle --site-config FILE [--micro-batch N] [--dry-run] [--preflight] [training options]'
  exit 0
fi
MODEL="$1"; shift
case "$MODEL" in dense|mor|loopformer|circle) ;; *) echo 'Unsupported native model' >&2; exit 2;; esac
source "$ROOT/launch/env.sh" "$@"
cd "$ROOT"
if [[ -z "${MICRO_BATCH:-}" ]]; then
  MICRO_BATCH="$("$NANOCHAT_VENV/bin/python" - "$ROOT/configs/native/hardware.json" "$NATIVE_HARDWARE" "$MODEL" <<'PY'
import sys,json
print(json.load(open(sys.argv[1]))[sys.argv[2]]['micro_batch'][sys.argv[3]])
PY
)"
fi
WORLD="${GPUS:-8}"
RUN_MODEL="$MODEL"
if [[ "$MODEL" == circle ]]; then RUN_MODEL=circle_sparse; fi
if [[ "$MODEL" == mor || "$MODEL" == loopformer ]]; then RUN_MODEL="${MODEL}_core393"; fi
OUTPUT="${OUTPUT_DIR:-$NANOCHAT_RUNTIME/runs/${NEXTT_RUN_NAME:-native_${RUN_MODEL}_${NATIVE_HARDWARE}_seed42}}"
ARGS=(--config "$ROOT/configs/native/$MODEL.json" --micro-batch "$MICRO_BATCH" --world-size "$WORLD" --output "$OUTPUT")
if [[ " $* " == *" --dry-run "* ]]; then
  exec "$NANOCHAT_VENV/bin/python" -m nexttransformer.native.train "${ARGS[@]}" "$@"
fi
if [[ " $* " == *" --preflight "* ]]; then
  EXTRA=()
  while [[ $# -gt 0 ]]; do
    case "$1" in
      --preflight) shift ;;
      --output) OUTPUT="$2"; shift 2 ;;
      --output=*) OUTPUT="${1#*=}"; shift ;;
      *) EXTRA+=("$1"); shift ;;
    esac
  done
  exec "$NANOCHAT_VENV/bin/python" -m nexttransformer.native.preflight --config "$ROOT/configs/native/$MODEL.json" --micro-batch "$MICRO_BATCH" --output "$OUTPUT/preflight.json" "${EXTRA[@]}"
fi
exec "$NANOCHAT_VENV/bin/python" -m torch.distributed.run --standalone --nnodes=1 --node_rank=0 --nproc_per_node="$WORLD" -m nexttransformer.native.train "${ARGS[@]}" "$@"
