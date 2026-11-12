#!/usr/bin/env bash
# Keep this node's share of a scaling run going: every launch resumes from the newest checkpoint complete on
# every node. Stops when the trainer records a terminal phase for this node; a non-finite loss or gradient is
# not retried (exit 3). Run one watchdog per node with the same environment as launch/scaling/node.sh.
# Optional: MAX_RESTARTS (default 20), RESTART_DELAY seconds (default 60).
set -uo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
if [[ " $* " == *" --help "* || $# == 0 ]]; then
  echo 'Usage: OUTPUT_DIR=DIR NNODES=N NODE_RANK=K MASTER_ADDR=HOST MASTER_PORT=PORT launch/scaling/watchdog.sh CONFIG --site-config FILE [training options]'
  exit 0
fi
CONFIG="$1"; shift
: "${OUTPUT_DIR:?set OUTPUT_DIR so every launch uses the same run directory}" "${NODE_RANK:?}"
STATUS="$OUTPUT_DIR/status.node$(printf %03d "$NODE_RANK").json"
failures=0
while true; do
  started="$(date +%s)"
  "$ROOT/launch/scaling/node.sh" "$CONFIG" --resume auto "$@"
  code=$?
  phase="$(python3 -c 'import json, os, sys
p, t = sys.argv[1], float(sys.argv[2])
print(json.load(open(p))["phase"] if os.path.exists(p) and os.path.getmtime(p) >= t else "")' "$STATUS" "$started")"
  case "$phase" in
    training_complete|smoke_complete|stopped) echo "watchdog: $phase"; exit 0 ;;
    nonfinite) echo "watchdog: non-finite loss or gradient; not restarting" >&2; exit 3 ;;
  esac
  failures=$((failures+1))
  if (( failures > ${MAX_RESTARTS:-20} )); then echo "watchdog: giving up after $failures failed launches" >&2; exit 1; fi
  echo "watchdog: launch exited with $code; restart $failures of ${MAX_RESTARTS:-20} in ${RESTART_DELAY:-60}s" >&2
  sleep "${RESTART_DELAY:-60}"
done
