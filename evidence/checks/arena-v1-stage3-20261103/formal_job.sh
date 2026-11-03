#!/usr/bin/env bash
# Koala pod entry for a bounded smoke run of the formal trainer (arena v1 Stage 3: the debug check and G2b):
#
#   bash formal_job.sh NPROC SITE_JSON <nexttransformer.train arguments, with --smoke-steps>
#
# Copies the mounted code bundle to the pod-local disk and builds the locked environment there (as the gym's
# nextt_gym/backends/koala/run.sh). Checks the dev shard against its pinned SHA-256 and links the dev evaluation
# directory once, then runs the trainer through formal_dev.py, which evaluates the dev split instead of the reserved
# validation shard, with NPROC ranks on this node (as launch/train.sh starts it).
set -euo pipefail
NPROC="$1"; SITE="$2"; shift 2
SRC="$(cd "$(dirname "$0")" && pwd)"
CODE=/local-ssd/nextt-code
rm -rf "$CODE" && mkdir -p "$CODE" && cp -a "$SRC/." "$CODE/" && cd "$CODE"
export UV_PROJECT_ENVIRONMENT=/local-ssd/nextt-venv UV_LINK_MODE=copy
export NANOCHAT_DTYPE=bfloat16 HF_HUB_OFFLINE=1 FLA_DISABLE_BACKEND_DISPATCH=1 PYTHONUNBUFFERED=1
export TRITON_CACHE_DIR=/local-ssd/triton-cache TORCHINDUCTOR_CACHE_DIR=/local-ssd/triton-cache/inductor
uv sync --frozen --extra gpu --extra upstream
uv pip install --python "$UV_PROJECT_ENVIRONMENT/bin/python" --no-deps --require-hashes -r requirements/linear.txt
mkdir -p "$TRITON_CACHE_DIR"
PY="$UV_PROJECT_ENVIRONMENT/bin/python"
DEV="$("$PY" - "$SITE" /local-ssd/nextt-dev-eval <<'EOF'
import json, sys
from pathlib import Path
from nexttransformer.arena import DEV_SHARD
from nextt_gym.trainer import check_dev_shard, dev_directory
site = Path(sys.argv[1]); data = Path(json.loads(site.read_text())["data_dir"])
data = data if data.is_absolute() else site.parent / data
check_dev_shard(data, DEV_SHARD)
Path(sys.argv[2]).mkdir(parents=True, exist_ok=True)
print(dev_directory(sys.argv[2], data, DEV_SHARD, create=True))
EOF
)"
echo "formal_job: $(cat "$CODE/STAGED_COMMIT") on $(hostname), $NPROC ranks, dev evaluation from $DEV"
exec "$PY" -m torch.distributed.run --standalone --nnodes 1 --nproc-per-node "$NPROC" formal_dev.py "$DEV" \
  --site-config "$SITE" "$@"
