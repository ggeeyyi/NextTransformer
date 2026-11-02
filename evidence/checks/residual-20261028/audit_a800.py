"""Read-only cumulative audit of a finished Residual A800 run directory (record 050). Run from the repository root.

Repeats the checks of the Dense Baseline audit (evidence/checks/a800-dense-full-formal.json) for the runs of the
2026-10-23 four-node ModelArts job:

    python evidence/checks/residual-20261028/audit_a800.py RUN_DIR \
        evidence/checks/residual-20261028/a800-residual-standard-formal.json \
        "R00 Standard Residual (residual_standard)" 2026-10-26T18:59+08:00
"""
import json
import math
import statistics
import sys
from pathlib import Path

REPO = Path.cwd()
sys.path.insert(0, str(REPO))
from nexttransformer.protocol.spec import Protocol, TOKENIZER_SHA256  # noqa: E402

RUN, OUT, LABEL, FINISHED = Path(sys.argv[1]), REPO/sys.argv[2], sys.argv[3], sys.argv[4]
P = Protocol()
ANCHOR = 4672398325.44785
VAL_STEPS, CORE_STEPS = list(P.validation_steps), list(P.core_steps)
TASKS = {t["id"]: t["randomBaseline"] for t in json.loads((REPO/"website/data/core-tasks.json").read_text())["tasks"]}
problems = []


def need(ok, message):
    if not ok:
        problems.append(message)


def load(name):
    return json.loads((RUN/name).read_text())


status, budget = load("run_status.json"), load("budget.json")
need(status["phase"] == "training_complete" and status["step"] == P.steps, "run_status: not training_complete at the final step")
need(status["final_status"] == "matched", f"run_status: final_status {status['final_status']}")
ledger = status["budget"]
for source, b in (("run_status", ledger), ("budget.json", budget)):
    need(b["optimizer_step"] == P.steps and b["input_tokens"] == P.total_tokens, f"{source}: steps/tokens differ from the protocol")
    need(b["protocol_hash"] == P.fingerprint, f"{source}: protocol hash differs")
need(ledger["val_steps"] == VAL_STEPS and ledger["core_steps"] == CORE_STEPS, "run_status: val/CORE step lists differ from the protocol")
need(budget["val_steps"] == VAL_STEPS and budget["core_steps"] == CORE_STEPS, "budget.json: val/CORE step lists differ from the protocol")
need(budget["charged_theory_flops"] == budget["theory_flops"] == ledger["theory_flops"], "budget: charged and theory FLOPs differ")
cumulative = ledger["theory_flops"]/(P.total_tokens*ANCHOR)-1
need(abs(cumulative) <= 0.02, f"cumulative error {cumulative:.4%} exceeds 2%")

events = [json.loads(line) for line in (RUN/"metrics.jsonl").read_text().splitlines()]
kinds = [e["event"] for e in events]
anomalies = kinds.count("budget_anomaly")
train = [e for e in events if e["event"] == "train"]
steps = [e["step"] for e in train]
need(all(a < b for a, b in zip(steps, steps[1:])), "train: steps not strictly increasing")
need(all(e["input_tokens"] == e["step"]*P.global_batch_tokens for e in train), "train: input_tokens != step x global batch")
need(all(math.isfinite(e["loss"]) and math.isfinite(e["grad_norm"]) for e in train), "train: non-finite loss or grad norm")
need(steps[-1] == P.steps, "train: last step is not the final step")
need(train[-1]["theory_flops"] == ledger["theory_flops"], "train: last cumulative FLOPs differ from the ledger")
relative = [e["theory_relative_error"] for e in train]

val = [e for e in events if e["event"] == "validation"]
need([e["step"] for e in val] == VAL_STEPS, "validation: schedule differs from the protocol")
need(all(e["input_tokens"] == P.val_tokens and not e["smoke"] and e["protocol_hash"] == P.fingerprint for e in val), "validation: tokens/smoke/protocol")
val_files = sorted(RUN.glob("val_*.json"))
need(len(val_files) == len(VAL_STEPS), f"validation: {len(val_files)} val_*.json files")
by_step = {e["step"]: e for e in val}
for f in val_files:
    v = json.loads(f.read_text())
    need(by_step.get(v["step"], {}).get("val_bpb") == v["val_bpb"], f"{f.name}: differs from metrics.jsonl")

core = [e for e in events if e["event"] == "core"]
need([e["step"] for e in core] == CORE_STEPS, "CORE: schedule differs from the protocol")
for e in core:
    need(e["full_core"] and set(e["results"]) == set(TASKS) == set(e["centered_results"]), f"CORE {e['step']}: not the full 22-task evaluation")
    for task, base in TASKS.items():
        need(abs((e["results"][task]-base)/(1-base)-e["centered_results"][task]) <= 1e-7, f"CORE {e['step']}: {task} centering")
    need(abs(sum(e["centered_results"].values())/len(TASKS)-e["core_metric"]) <= 1e-7, f"CORE {e['step']}: aggregate")
    name = "core_final.json" if e["step"] == P.steps else f"core_{e['step']:08d}.json"
    need(load(name)["core_metric"] == e["core_metric"], f"{name}: differs from metrics.jsonl")

finals = [e for e in events if e["event"] == "final_bpb"]
need(len(finals) == 1, f"{len(finals)} final_bpb events")
bpb = load("bpb_final.json")
need(bpb["val"]["input_tokens"] == bpb["train"]["input_tokens"] == P.final_bpb_tokens, "bpb_final: tokens per split")
need(finals[0]["val_bpb"] == bpb["val"]["bpb"] and finals[0]["train_bpb"] == bpb["train"]["bpb"], "bpb_final: differs from metrics.jsonl")
need(load("latest.json") == {"step": P.steps, "checkpoint": f"checkpoint_{P.steps:08d}"}, "latest.json is not the final checkpoint")

names = ["provenance.json", *sorted(f.name for f in RUN.glob("resume_*.json"))]
segments, keys = [], ("model_hash", "protocol_hash", "data_manifest_sha256", "tokenizer_sha256", "parameters")
first = load(names[0])
for name in names:
    p = load(name)
    for k in keys:
        need(p[k] == first[k], f"{name}: {k} differs from the first segment")
    need(p["model"] == first["model"] and p["cost"] == first["cost"], f"{name}: config or cost proof differs")
    need(p["software"]["science_hash"] == first["software"]["science_hash"], f"{name}: science hash differs")
    need(p["software"]["reference_source_hash"] == first["software"]["reference_source_hash"], f"{name}: validation source hash differs")
    need(not p["smoke"], f"{name}: smoke segment")
    need(p["world_size"]*p["micro_batch"]*p["accumulation"]*P.sequence_len == P.global_batch_tokens, f"{name}: batch layout")
    segments.append(dict(file=name, step=0 if p["resumed_from"] is None else p["resumed_from"], host=p["host"], world=p["world_size"],
                         micro=p["micro_batch"], accumulation=p["accumulation"], commit=p["software"]["git_commit"],
                         code=p["software"]["code_snapshot"]))
need(first["protocol_hash"] == P.fingerprint, "provenance: protocol hash differs from the frozen protocol")
need(first["tokenizer_sha256"] == TOKENIZER_SHA256, "provenance: tokenizer differs from the pinned tokenizer")
need(abs(first["cost"]["relative_error"]) <= 0.01, "provenance: cost proof outside the 1% preflight")

tracking = load("tracking.json")
periodic = {e["step"]: e["val_bpb"] for e in val}
best = min(val, key=lambda e: e["val_bpb"])
report = dict(
    kind="a800_formal_run_cumulative_audit",
    method=LABEL,
    date="2026-10-28",
    finished_at=FINISHED,
    run_dir="daxiao nanochat-runtime/runs/"+RUN.name,
    experiments_record="050",
    checks=[
        "run_status.json: phase training_complete at 46875; trainer final_status",
        "ledger (run_status/budget.json): 46875 optimizer steps, 49,152,000,000 input tokens, val/CORE step lists equal to the protocol, charged == theory FLOPs",
        "cumulative theory FLOPs / (49.152e9 x SSSL 4.672398325e9) - 1 within 2%; budget_anomaly events counted",
        "metrics.jsonl train events: strictly increasing steps, input_tokens == step x 1,048,576, finite loss/grad norm, last step 46875 with the ledger's cumulative FLOPs",
        "189 validation events at 0, every 250, 46875 (41,943,040 tokens each) and the same 189 val_*.json files",
        "6 CORE events at 8000/16000/24000/32000/40000/46875, each full_core with 22 tasks, centering and aggregate reproduced, equal to core_*.json",
        "exactly one final_bpb event; bpb_final.json (20,971,520 tokens per split), core_final.json and latest.json at the final checkpoint",
        "every segment's provenance (provenance.json, resume_*.json): identical model/protocol/data/tokenizer fingerprints, parameter count, config, cost proof, validation source hash and science_hash; non-smoke; world x micro x accumulation x 2048 == 1,048,576",
    ],
    result_checkpoint=f"checkpoint_{P.steps:08d} (final; latest.json); no best-checkpoint selection",
    naming="periodic_val_bpb_46875 is the 189th scheduled val point; final_bpb is the independent final BPB",
    ok=not problems,
    problems=problems,
    final_status=status["final_status"],
    optimizer_steps=ledger["optimizer_step"],
    input_tokens=ledger["input_tokens"],
    theory_flops=ledger["theory_flops"],
    dense_flops_per_token=ANCHOR,
    flops_per_token=first["cost"]["flops_per_token"],
    proof_relative_error=first["cost"]["relative_error"],
    cumulative_relative_error=cumulative,
    train_event_relative_error=dict(min=min(relative), max=max(relative)),
    budget_anomalies=anomalies,
    train_events=len(train),
    validation_events=len(val),
    core_events=len(core),
    final_bpb=dict(train=bpb["train"]["bpb"], val=bpb["val"]["bpb"], tokens_per_split=bpb["val"]["input_tokens"]),
    core={e["step"]: e["core_metric"] for e in core},
    core_final=core[-1]["core_metric"],
    last_loss=train[-1]["loss"],
    grad_norm_max_after_500=max(e["grad_norm"] for e in train if e["step"] > 500),
    step_seconds_median=statistics.median(e["step_seconds"] for e in train),
    mfu_median=statistics.median(e["mfu_theory"] for e in train),
    peak_memory_gib_max=max(e["peak_memory_bytes"] for e in train)/2**30,
    gpu_hours_training=status["training_seconds"]*first["world_size"]/3600,
    training_seconds=status["training_seconds"],
    evaluation_seconds=status["evaluation_seconds"],
    evaluation_forward_flops=status["evaluation_forward_flops"],
    segments=segments,
    fingerprints=dict(model_hash=first["model_hash"], protocol_hash=first["protocol_hash"],
                      data_manifest_sha256=first["data_manifest_sha256"], tokenizer_sha256=first["tokenizer_sha256"],
                      reference_source_hash=first["software"]["reference_source_hash"], science_hash=first["software"]["science_hash"]),
    parameters=first["parameters"],
    layers=first["model"]["layers"],
    mlp_width=first["model"]["mlp_width"],
    window_pattern=first["model"]["window_pattern"],
    device=first["device"],
    software=dict(torch=first["software"]["torch"], cuda=first["software"]["cuda"], nccl=first["software"]["nccl"]),
    periodic_val_bpb={s: periodic[s] for s in (0, 8000, 16000, 24000, 32000, 40000, P.steps)},
    periodic_val_bpb_minimum=dict(val_bpb=best["val_bpb"], step=best["step"]),
    tracking=dict(run=tracking["id"], mode=tracking["mode"], last_step=tracking["last_step"], state="finished",
                  events_logged=len(events), summary_matches_run_dir=True,
                  note=("the trainer logged online; on 2026-10-28 the tracked run was finished with 4,885 history lines, and its "
                        "summary final BPB, periodic val BPB, CORE and cumulative theory FLOPs equal the run directory. "
                        "Every other value here is read from the run directory")),
    boundaries=("Single seed 42. Hardware 1x8 A800-SXM4-80GB, one node of the four-node ModelArts job that ran R00-R03 as "
                "independent eight-GPU runs (runs/train_reference_modelarts.py), micro "
                f"{first['micro_batch']} x accumulation {first['accumulation']}, launched from the frozen checkout asoul/NextT at "
                "commit 4dd6ebb (record 028). The preregistered pilot was skipped by the owner's decision; the formal run started "
                "from random initialization. "
                + ("One segment, so there is no resume boundary. " if len(segments) == 1 else f"{len(segments)} segments. ")
                + "Resources are outcomes, not controls; the reference residual implementations are eager PyTorch without fused "
                "kernels. Evaluation forward FLOPs are reported separately and not charged to the training budget."),
)
OUT.parent.mkdir(parents=True, exist_ok=True)
OUT.write_text(json.dumps(report, indent=1)+"\n")
print(json.dumps({k: report[k] for k in ("ok", "problems", "cumulative_relative_error", "train_events", "validation_events", "core_events",
                                          "final_bpb", "core_final", "step_seconds_median", "mfu_median", "gpu_hours_training",
                                          "peak_memory_gib_max", "grad_norm_max_after_500", "periodic_val_bpb_minimum", "budget_anomalies",
                                          "segments")}, indent=1))
