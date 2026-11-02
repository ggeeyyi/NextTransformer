#!/usr/bin/env bash
# Local driver: one formal reference training as consecutive Koala H20 segments.
#
#   bash launch/koala/chain.sh JOB_PREFIX BUNDLE_URI -- <run.sh arguments>
#
# Every segment uses the same published code bundle and `--resume auto`, so the
# trainer continues from the newest complete checkpoint in --output. The next
# segment is submitted only after the previous job succeeded and reported
# phase=segment_complete. Any failure stops the chain with the job left in
# place for `koala get` / `koala log`; nothing is retried automatically.
# NEXTT_WALL_HOURS (default 45.5) ends each segment before the 48 h job limit.
# NEXTT_TRITON_CACHE_ARCHIVE (a persistent path, e.g. RUN_DIR/triton-autotune.tar)
# is forwarded to run.sh so later segments reuse the first segment's kernel
# configurations, as in the bounded segmented-resume check.
# NEXTT_NODES=2 submits two-node segments (-n 2 -g 8, 16 ranks; see run.sh).
set -euo pipefail
if [[ $# -lt 3 || "$3" != "--" ]]; then
  echo "usage: $0 JOB_PREFIX BUNDLE_URI -- <run.sh arguments>" >&2; exit 2
fi
prefix=$1 bundle=$2; shift 3
segments=${NEXTT_MAX_SEGMENTS:-8}
nodes=${NEXTT_NODES:-1}
archive=${NEXTT_TRITON_CACHE_ARCHIVE:+NEXTT_TRITON_CACHE_ARCHIVE=$(printf '%q' "$NEXTT_TRITON_CACHE_ARCHIVE") }
command="cd /data/work/run_codes && NEXTT_NODES=$nodes ${archive}bash launch/koala/run.sh $(printf '%q ' "$@")--resume auto --max-wall-hours ${NEXTT_WALL_HOURS:-45.5}"

for ((n = 1; n <= segments; n++)); do
  job=$(koala --cluster h20 submit -m normal -n "$nodes" -g 8 --ssd 64 -q -y -j "$prefix" \
        --code-bundle "$bundle:/data/work/run_codes" -c "$command")
  echo "$(date -u +%FT%TZ) segment $n submitted: $job"
  if ! koala --cluster h20 wait "$job" --timeout 50h; then
    echo "segment $n did not succeed; inspect: koala --cluster h20 get $job && koala --cluster h20 log $job -n 200" >&2
    exit 1
  fi
  status=$(koala --cluster h20 log "$job" -n 50 | grep '^NEXTT_RUN_STATUS' | tail -1 || true)
  echo "$(date -u +%FT%TZ) segment $n finished: ${status:-no status line}"
  case "$status" in
    *" segment_complete "*) ;;
    *" training_complete "*) echo "training complete"; exit 0 ;;
    *) echo "unexpected status; stopping the chain" >&2; exit 1 ;;
  esac
done
echo "segment limit $segments reached without completion" >&2
exit 1
