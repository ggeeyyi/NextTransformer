#!/usr/bin/env bash
# One-GPU debug check before the 8-GPU jobs (team rule: debug first, GPU utilization median >= 80%):
# - the formal trainer on real data, a short smoke of the Stage 3 plugin and of dense_full (micro batch 16), its
#   evaluations on the dev split (formal_dev.py; the validation shard is reserved for formal runs);
# - the gym trainer with the plugin as an imported method, full size on synthetic rows (micro batch 16).
# nvidia-smi samples utilization and memory every second; markers.txt brackets each part.
#
#   bash debug_check.sh OUT      # exits non-zero if any part failed
set -u
OUT="$1"; SITE=/cpfs/<user>/nextt/site.gym.json
DIR="$(cd "$(dirname "$0")" && pwd)"
mkdir -p "$OUT"
for f in formal_job.sh formal_dev.py gym_debug/run_spec.json gym_debug/submission/model.py; do
  [[ -s "$DIR/$f" ]] || { echo "debug_check: $DIR/$f missing" | tee -a "$OUT/markers.txt"; exit 2; }
done
nvidia-smi --query-gpu=timestamp,utilization.gpu,memory.used --format=csv,noheader,nounits -l 1 > "$OUT/gpu_util.csv" &
SMI=$!
trap 'kill "$SMI" 2>/dev/null' EXIT
FAILED=0
for config in configs/residual/attnres_full_canon_abcd.json configs/baseline/dense_full.json; do
  name="formal-$(basename "$config" .json)"
  echo "start $name $(date +%s)" >> "$OUT/markers.txt"
  bash "$DIR/formal_job.sh" 1 "$SITE" --config "$config" --output "$OUT/$name" --smoke-steps 12 \
    --smoke-accumulation 4 --micro-batch 16 --log-every 1 --wandb-mode disabled > "$OUT/$name.log" 2>&1
  code=$?; echo "end $name $code $(date +%s)" >> "$OUT/markers.txt"; [[ $code == 0 ]] || FAILED=1
done
echo "start gym-attnres_full_canon_abcd $(date +%s)" >> "$OUT/markers.txt"
( cd /local-ssd/nextt-code && NANOCHAT_DTYPE=bfloat16 NANOCHAT_RUNTIME=/local-ssd/nextt-runtime PYTHONUNBUFFERED=1 \
    /local-ssd/nextt-venv/bin/python -m nextt_gym.trainer \
    --submission "$DIR/gym_debug/submission" --run-spec "$DIR/gym_debug/run_spec.json" --output "$OUT/gym-attnres_full_canon_abcd" \
    --synthetic --smoke-rows 16 --log-every 1 ) > "$OUT/gym-attnres_full_canon_abcd.log" 2>&1
code=$?; echo "end gym-attnres_full_canon_abcd $code $(date +%s)" >> "$OUT/markers.txt"; [[ $code == 0 ]] || FAILED=1
echo "done failed=$FAILED" >> "$OUT/markers.txt"
exit "$FAILED"
