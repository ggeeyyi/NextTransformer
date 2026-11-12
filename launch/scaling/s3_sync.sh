#!/usr/bin/env bash
# Mirror this node's complete checkpoints to object storage with s5cmd; node 0 also mirrors the run records.
# Remote retention matches the trainer: the newest KEEP_REMOTE rolling checkpoints plus every milestone.
# Required: OUTPUT_DIR, NODE_RANK, S3_PREFIX (s3://bucket/prefix/run). Optional: SYNC_INTERVAL seconds (600),
# KEEP_REMOTE (3), MILESTONE_EVERY steps (10000, as in the configuration), ONCE=1 for a single pass.
set -uo pipefail
: "${OUTPUT_DIR:?}" "${NODE_RANK:?}" "${S3_PREFIX:?}"
node="node_$(printf %03d "$NODE_RANK")"
while true; do
  for manifest in "$OUTPUT_DIR"/checkpoint_*/"$node"/complete.json; do
    [[ -e "$manifest" ]] || continue
    dir="$(dirname "$manifest")"; name="$(basename "$(dirname "$dir")")"
    [[ -e "$dir/.uploaded" ]] && continue
    if s5cmd cp "$dir/*" "$S3_PREFIX/$name/$node/"; then touch "$dir/.uploaded"; else echo "sync: $name upload failed" >&2; fi
  done
  if [[ "$NODE_RANK" == 0 ]]; then
    for pattern in '*.json' '*.jsonl'; do
      compgen -G "$OUTPUT_DIR/$pattern" > /dev/null && s5cmd cp "$OUTPUT_DIR/$pattern" "$S3_PREFIX/records/"
    done
  fi
  rolling=()
  while read -r name; do
    (( 10#${name#checkpoint_} % ${MILESTONE_EVERY:-10000} )) && rolling+=("$name")
  done < <(s5cmd ls "$S3_PREFIX/" 2>/dev/null | awk '$1 == "DIR" && $2 ~ /^checkpoint_[0-9]+\/$/ {sub("/$", "", $2); print $2}' | sort)
  for (( i = 0; i < ${#rolling[@]} - ${KEEP_REMOTE:-3}; i++ )); do
    s5cmd rm "$S3_PREFIX/${rolling[i]}/$node/*" || echo "sync: could not remove remote ${rolling[i]}" >&2
  done
  [[ "${ONCE:-0}" == 1 ]] && break
  sleep "${SYNC_INTERVAL:-600}"
done
