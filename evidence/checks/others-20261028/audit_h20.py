"""Read-only audit of the frozen Others H20 runs; print one JSON report per run.

Run with Python 3.10+ where the completed run directories are accessible:
    python audit_h20.py --runs RUNS_DIRECTORY

Derived from fe8089e:docs/others/audit_h20_run.py. The added CORE baselines
are frozen from main 1a47b0b:website/data/core-tasks.json. Reads metrics,
status, budget, provenance, final BPB and checkpoint metadata; never writes
to a run directory or loads checkpoint tensors. Branch record numbers are
explained in sources.json.
"""
import argparse
import json
import math
import statistics
from datetime import datetime
from pathlib import Path


LABEL = {"mla_qknorm": "O03′ mla_qknorm", "mla_nope_qknorm": "O03-N′ mla_nope_qknorm"}
STEPS, BATCH, VAL_TOKENS, FINAL_TOKENS = 46875, 1_048_576, 41_943_040, 20_971_520
TOKENS = STEPS*BATCH
VAL_PLAN = list(range(0, STEPS, 250)) + [STEPS]
CORE_PLAN = [8000, 16000, 24000, 32000, 40000, STEPS]
FIXED = ("model_hash", "protocol_hash", "data_manifest_sha256", "tokenizer_sha256", "parameters", "config", "cost")
SOFTWARE = ("science_hash", "reference_source_hash", "torch", "cuda", "nccl", "attention")


def finite(x): return isinstance(x, (int, float)) and math.isfinite(x)


def audit(method, record, runs):
    d = Path(runs)/f"{method}_h20_seed42"
    if not (d/"run_status.json").exists(): raise SystemExit(f"{d}: no run_status.json; the run has not finished")
    st = json.loads((d/"run_status.json").read_text())
    ledger, dense = st["budget"], json.loads((d/"budget.json").read_text())["dense_flops_per_token"]
    ev = [json.loads(l) for l in (d/"metrics.jsonl").read_text().splitlines() if l.strip()]
    train = [e for e in ev if e["event"] == "train"]
    vals = [e for e in ev if e["event"] == "validation"]
    cores = [e for e in ev if e["event"] == "core"]
    finals = [e for e in ev if e["event"] == "final_bpb"]
    bpb = json.loads((d/"bpb_final.json").read_text())
    names = ["provenance.json"] + sorted(p.name for p in d.glob("resume_*.json"))
    prov = [(n, json.loads((d/n).read_text())) for n in names]
    steps = [e["step"] for e in train]
    problems = []

    def need(ok, what):
        if not ok: problems.append(what)
        return ok

    checks = [
        need(st["phase"] == "training_complete" and st["step"] == STEPS and st["final_status"] == "matched",
             "run_status not training_complete/matched"),
        need(ledger["optimizer_step"] == STEPS and ledger["input_tokens"] == TOKENS and ledger["val_steps"] == VAL_PLAN
             and ledger["core_steps"] == CORE_PLAN, "ledger steps/tokens/event lists differ from the protocol"),
        need(abs(ledger["theory_flops"]/(TOKENS*dense) - 1) <= .02, "cumulative compute outside 2%"),
        need(steps == sorted(set(steps)) and steps[-1] == STEPS and all(e["input_tokens"] == e["step"]*BATCH for e in train)
             and all(finite(e["loss"]) and finite(e["grad_norm"]) for e in train), "train events"),
        need([e["step"] for e in vals] == VAL_PLAN and all(e["input_tokens"] == VAL_TOKENS and not e["smoke"] for e in vals),
             "validation events"),
        need([e["step"] for e in cores] == CORE_PLAN and all(e["full_core"] and len(e["centered_results"]) == 22 for e in cores),
             "CORE events"),
        need(len(finals) == 1 and all(bpb[s]["input_tokens"] == FINAL_TOKENS for s in ("train", "val"))
             and (d/f"checkpoint_{STEPS:08d}"/"complete.json").exists(), "final BPB / final checkpoint"),
        need(all(len({json.dumps(p.get(k), sort_keys=True) for _, p in prov}) == 1 for k in FIXED)
             and all(len({p["software"].get(k) for _, p in prov}) == 1 for k in SOFTWARE)
             and all(not p["smoke"] and p["world_size"]*p["micro_batch"]*p["accumulation"]*2048 == BATCH
                     and p["compile"] == prov[0][1]["compile"] for _, p in prov), "segment provenance"),
    ]
    rel = [e["theory_relative_error"] for e in train]
    after = [e for e in train if e["step"] > 500]
    grad_max = max(after, key=lambda e: e["grad_norm"])
    running, jumps = float("inf"), []
    for e in train:
        if e["step"] > 1000 and e["loss"] - running > 1.0: jumps.append((e["step"], e["loss"]))
        running = min(running, e["loss"])
    valmap = {e["step"]: e["val_bpb"] for e in vals}
    rises = sum(1 for a, b in zip(vals, vals[1:]) if b["val_bpb"] > a["val_bpb"] + .005)
    worst_val = max((e for e in vals if e["step"] >= 1000), key=lambda e: e["val_bpb"])
    first = prov[0][1]
    out = dict(
        kind="h20_formal_run_cumulative_audit", method=f"{LABEL[method]} ({method})",
        date=datetime.fromtimestamp((d/"run_status.json").stat().st_mtime).strftime("%Y-%m-%d"),
        run_dir=f"cpfs nextt/runs/others/{d.name}", experiments_record=record,
        checks=["run_status.json: phase training_complete at 46875; trainer status matched",
                "ledger (run_status.json budget): 46875 optimizer steps, 49,152,000,000 input tokens, val/CORE step lists equal to the protocol",
                "cumulative theory FLOPs / (49.152e9 x Dense 4.672398325e9) - 1 within 2%; budget_anomaly events counted",
                "metrics.jsonl train events: strictly increasing steps, input_tokens == step x 1,048,576, finite loss/grad norm, last step 46875",
                "189 validation events at 0, every 250, 46875 (41,943,040 tokens each, non-smoke)",
                "6 CORE events at 8000/16000/24000/32000/40000/46875, each full_core with 22 tasks",
                "final BPB for train and val (20,971,520 tokens per split, bpb_final.json); final checkpoint_00046875 complete",
                "every segment's provenance (provenance.json, resume_*.json): identical model/protocol/data/tokenizer/config/cost fingerprints, "
                "science and source hashes, torch/CUDA/NCCL and attention backend; non-smoke; world x micro x accumulation x 2048 == 1,048,576; same compile setting"],
        result_checkpoint="checkpoint_00046875 (final; latest.json); no best-checkpoint selection",
        naming="periodic_val_bpb_46875 is the 189th scheduled val point; final_bpb is the independent final BPB",
        ok=all(checks), problems=problems, final_status=st["final_status"], optimizer_steps=ledger["optimizer_step"],
        input_tokens=ledger["input_tokens"], theory_flops=ledger["theory_flops"], dense_flops_per_token=dense,
        flops_per_token=first["cost"]["flops_per_token"], proof_relative_error=first["cost"]["relative_error"],
        cumulative_relative_error=ledger["theory_flops"]/(TOKENS*dense) - 1,
        train_event_relative_error=dict(min=min(rel), max=max(rel)),
        budget_anomalies=sum(1 for e in ev if e["event"] == "budget_anomaly"),
        train_events=len(train), validation_events=len(vals), core_events=len(cores),
        final_bpb=dict(train=bpb["train"]["bpb"], val=bpb["val"]["bpb"], tokens_per_split=FINAL_TOKENS),
        core={str(e["step"]): e["core_metric"] for e in cores}, core_final=cores[-1]["core_metric"],
        core_tasks_centered={str(e["step"]): e["centered_results"] for e in cores},
        last_loss=train[-1]["loss"], grad_norm_max_after_500=grad_max["grad_norm"], grad_norm_max_step=grad_max["step"],
        stability=dict(loss_jumps_over_1_above_running_min_after_1000=len(jumps),
                       largest_loss_after_1000=max((e for e in train if e["step"] > 1000), key=lambda e: e["loss"])["loss"],
                       validation_rises_over_0p005=rises, worst_val_bpb_after_1000=dict(step=worst_val["step"], val_bpb=worst_val["val_bpb"])),
        step_seconds_median=statistics.median(e["step_seconds"] for e in train if e["step"] > 10),
        mfu_median=statistics.median(e["mfu_theory"] for e in train if e["step"] > 10),
        gpu_hours_training=train[-1]["gpu_hours"], training_seconds=st["training_seconds"],
        evaluation_seconds=st["evaluation_seconds"], evaluation_forward_flops=st["evaluation_forward_flops"],
        segment_provenance=[dict(file=n, start_step=0 if n == "provenance.json" else int(n[7:15]), job=p.get("koala_job"),
                                 world=p["world_size"], micro=p["micro_batch"], accumulation=p["accumulation"],
                                 device=p["device"], attention=p["software"]["attention"], compile=p["compile"],
                                 code=p["software"].get("code_snapshot"), commit=p["software"].get("git_commit")) for n, p in prov],
        fingerprints={k: first[k] for k in ("model_hash", "protocol_hash", "data_manifest_sha256", "tokenizer_sha256")}
                     | {k: first["software"][k] for k in SOFTWARE},
        parameters=first["parameters"], devices=sorted({p["device"] for _, p in prov}),
        periodic_val_bpb={str(s): valmap[s] for s in (0, 8000, 16000, 24000, 32000, 40000, STEPS)},
        periodic_val_bpb_minimum=min(({"val_bpb": e["val_bpb"], "step": e["step"]} for e in vals), key=lambda x: x["val_bpb"]),
    )
    out["boundaries"] = (
        f"Single seed 42, trained from update 0 on Koala H20 (record 041). {first['world_size']} ranks "
        f"({first['world_size']//8} nodes x 8 H20), registered eight-stream layout, micro {first['micro_batch']} x accumulation "
        f"{first['accumulation']}; {len(prov)-1} resume point(s). Attention backend {first['software']['attention']}, "
        f"{'compiled' if first['compile'] else 'eager'}. Stabilized row registered after the first attempt diverged; "
        "preregistered pilot P1-P5 passed (record 039). The stopped H800 formal run of this row is an unfinished attempt, not a result.")
    return out



TASKS = {'hellaswag_zeroshot': 0.25, 'jeopardy': 0.0, 'bigbench_qa_wikidata': 0.0, 'arc_easy': 0.25, 'arc_challenge': 0.25, 'copa': 0.5, 'commonsense_qa': 0.2, 'piqa': 0.5, 'openbook_qa': 0.25, 'lambada_openai': 0.0, 'hellaswag': 0.25, 'winograd': 0.5, 'winogrande': 0.5, 'bigbench_dyck_languages': 0.0, 'agi_eval_lsat_ar': 0.2, 'bigbench_cs_algorithms': 0.0, 'bigbench_operators': 0.0, 'bigbench_repeat_copy_logic': 0.0, 'squad': 0.0, 'coqa': 0.0, 'boolq': 0.62, 'bigbench_language_identification': 0.091}
ap = argparse.ArgumentParser(description=__doc__)
ap.add_argument("--runs", required=True, type=Path)
args = ap.parse_args()
for method in ("mla_qknorm", "mla_nope_qknorm"):
    root = args.runs
    d = root / f"{method}_h20_seed42"
    a = audit(method, "041", root)
    events = [json.loads(line) for line in (d/"metrics.jsonl").read_text().splitlines() if line.strip()]
    cores = [e for e in events if e["event"] == "core"]
    errors = []
    for e in cores:
        if set(e["results"]) != set(TASKS) or set(e["centered_results"]) != set(TASKS):
            errors.append(f"CORE task set at {e['step']}")
            continue
        for task, baseline in TASKS.items():
            actual = e["centered_results"][task]
            expected = (e["results"][task] - baseline)/(1-baseline)
            if not finite(actual) or abs(actual-expected) > 1e-7: errors.append(f"CORE centering {task} at {e['step']}")
        if abs(sum(e["centered_results"].values())/22 - e["core_metric"]) > 1e-7:
            errors.append(f"CORE aggregate at {e['step']}")
    bpb = json.loads((d/"bpb_final.json").read_text())
    final = [e for e in events if e["event"] == "final_bpb"][-1]
    for split in ("train", "val"):
        if not finite(bpb[split]["bpb"]) or bpb[split]["bpb"] != final[f"{split}_bpb"]: errors.append(f"final {split} BPB disagreement")
    p = json.loads((d/"provenance.json").read_text())
    if p["tokenizer_sha256"] != "06258dc0044b6198c6fa8ba75c5b3bbaeaeab44f4804b464d9ce5f5e3797ec54": errors.append("pinned tokenizer mismatch")
    if p["protocol_hash"] != "f8cd1b56fe75a6cf1fcc76aaa9ddd15c0ed054268071b67739a677476fd291dc": errors.append("frozen protocol mismatch")
    if p["software"].get("deterministic") is not True: errors.append("strict determinism not recorded")
    checkpoint = d/"checkpoint_00046875"
    payloads = {f.name: f.stat().st_size for f in checkpoint.iterdir() if f.is_file()}
    a["checks"].extend(["All six CORE events: exact canonical 22-task set, centering and aggregate reproduced to 1e-7", "Independent final BPB files agree with metrics; pinned tokenizer and protocol; deterministic algorithms recorded"])
    a["problems"].extend(errors)
    a["ok"] = a["ok"] and not errors
    a["deterministic"] = p["software"].get("deterministic")
    a["checkpoint_payload_sizes"] = payloads
    a["checkpoint_verification"] = "Completion marker and file inventory read; tensor payloads were not loaded or rehashed."
    a["core_tasks_raw"] = {str(e["step"]): e["results"] for e in cores}
    a["checks"] = [s.replace("Dense 4.672398325e9", "SSSL 4.672398325e9") for s in a["checks"]]
    print("NEXTT_AUDIT " + json.dumps({"method": method, "audit": a}))
