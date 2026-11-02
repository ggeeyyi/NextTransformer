#!/usr/bin/env bash
# Bounded 8- or 16-GPU acceptance for one admitted method (Koala Normal job, ~30 min):
# full-size, full-global-batch smoke updates with real data; a run segmented
# after its first update and resumed in a new process must equal the
# continuous run bitwise (losses and final weights); val/CORE smoke chain.
#
#   bash launch/koala/bounded_check.sh SITE.json METHOD OUT_ROOT [STEPS [nexttransformer.train options]]
#
# Smoke results are never formal results. OUT_ROOT must be a new directory.
# NEXTT_MICRO_BATCH (default 8) sets the per-GPU micro batch (accumulation follows).
# NEXTT_NODES=2 runs on both nodes of a -n 2 -g 8 job (see run.sh); node 0 writes the summary.
# NEXTT_LAYOUT_REFERENCE (optional) is an 8-GPU check of the same method and steps:
# every rank's data-stream state must then equal that of its stream in the reference.
set -euo pipefail
cd "$(dirname "$0")/../.."
site=$1 method=$2 root=$3 steps=${4:-12}
if (( ${NEXTT_NODES:-1} > 1 )); then
  # Every node runs this script and may create the root (and rendezvous directories): no run may exist yet.
  [[ ! -e "$root/continuous/provenance.json" && ! -e "$root/segmented/provenance.json" ]] \
    || { echo "$root already holds a check" >&2; exit 1; }
  mkdir -p "$root"
else
  mkdir -p "$(dirname "$root")" && mkdir "$root"  # the run directory itself must be new
fi
common=(--site-config "$site" --config "$(ls configs/*/"$method".json)" --micro-batch "${NEXTT_MICRO_BATCH:-8}"
        --smoke-steps "$steps" --rolling-every 4 --log-every 1 --smoke-core-per-task 32 "${@:5}")

# Each later process starts from empty local kernel caches (Triton, with the
# Inductor cache inside it) and restores only the archive, as a new pod of a
# later segment would.
export NEXTT_TRITON_CACHE_ARCHIVE="$root/triton-autotune.tar"
cache="${TRITON_CACHE_DIR:-/local-ssd/triton-cache}"
bash launch/koala/run.sh "${common[@]}" --output "$root/continuous"
rm -rf "$cache"
bash launch/koala/run.sh "${common[@]}" --output "$root/segmented" --max-wall-hours 1e-9
rm -rf "$cache"
bash launch/koala/run.sh "${common[@]}" --output "$root/segmented"
[[ "$(cat /local-ssd/nextt-node-rank)" == 0 ]] || exit 0

export UV_PROJECT_ENVIRONMENT="${NEXTT_VENV:-/local-ssd/nextt-venv}"  # the env run.sh built
uv run --no-sync python - "$root" "$steps" "$method" "${NEXTT_LAYOUT_REFERENCE:-}" <<'EOF'
import json, sys
from pathlib import Path
import torch
root, steps, method = Path(sys.argv[1]), int(sys.argv[2]), sys.argv[3]
def events(run):
    return [json.loads(l) for l in (root/run/"metrics.jsonl").read_text().splitlines()]
def train(run): return {e["step"]: e["loss"] for e in events(run) if e["event"] == "train"}
def weights(run):
    return torch.load(root/run/f"checkpoint_{steps:08d}"/"model.pt", map_location="cpu", weights_only=False)["model"]
a, b = weights("continuous"), weights("segmented")
same_weights = a.keys() == b.keys() and all(torch.equal(a[k], b[k]) for k in a)
timing = [e["step_seconds"] for e in events("continuous") if e["event"] == "train" and e["step"] > 2]
first = [e for e in events("continuous") if e["event"] == "train"]
status = {run: json.loads((root/run/"run_status.json").read_text()) for run in ("continuous", "segmented")}
summary = dict(method=method, steps=steps, losses_equal=train("continuous") == train("segmented"),
               weights_equal=same_weights, resumed=(root/"segmented"/"resume_00000001.json").exists(),
               median_step_seconds=sorted(timing)[len(timing)//2] if timing else None,
               mfu_theory=[e["mfu_theory"] for e in first][-1], peak_memory_bytes=max(e["peak_memory_bytes"] for e in first),
               losses=train("continuous"), phases={k: v["phase"] for k, v in status.items()},
               evaluations=[dict(event=e["event"], step=e["step"], **{k: e[k] for k in ("val_bpb", "core_metric") if k in e})
                            for e in events("continuous") if e["event"] in ("validation", "core")])
summary["passed"] = summary["losses_equal"] and summary["weights_equal"] and summary["resumed"]
if sys.argv[4]:
    reference = Path(sys.argv[4])
    def streams(run):
        files = sorted((run/"continuous"/f"checkpoint_{steps:08d}").glob("rank_*.pt"))
        return [torch.load(f, map_location="cpu", weights_only=False)["data"] for f in files]
    mine, theirs = streams(root), streams(reference)
    shares = len(mine)//len(theirs)
    other = {e["step"]: e["loss"] for e in [json.loads(l) for l in
                            (reference/"continuous"/"metrics.jsonl").read_text().splitlines()] if e["event"] == "train"}
    summary["layout_reference"] = dict(
        root=str(reference), world=len(mine), reference_world=len(theirs),
        stream_states_equal=len(mine) == shares*len(theirs) and all(s == theirs[r//shares] for r, s in enumerate(mine)),
        reference_losses=other, max_loss_relative_difference=max(abs(summary["losses"][s]/other[s]-1) for s in other))
    summary["passed"] = summary["passed"] and summary["layout_reference"]["stream_states_equal"]
(root/"bounded_check.json").write_text(json.dumps(summary, indent=2)+"\n")
print("NEXTT_BOUNDED_CHECK", json.dumps({k: summary[k] for k in ("passed", "losses_equal", "weights_equal", "median_step_seconds", "mfu_theory")}
                                         | {k: v for k, v in summary.get("layout_reference", {}).items() if k in ("stream_states_equal", "max_loss_relative_difference")}))
EOF
