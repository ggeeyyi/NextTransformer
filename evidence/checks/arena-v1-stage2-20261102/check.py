"""Recompute the arena v1 Stage 2 run checks from the files beside this script; print the result as JSON.

Each scale-M run must train the first 2,000 updates of the formal 46,875-update schedule at seed 42 + its offset,
evaluating the dev split only, every 256 updates and at update 2,000. Its logged updates must carry continuous tokens
and the cumulative cost of the audited per-update cost plus the evaluations before them; it must settle exactly at
that cost, stay within its reservation and within 1% of the anchor cost, and report the dev BPB of its update-2,000
evaluation. A cancelled run is checked up to its last logged update.
Usage: python check.py  (from the repository's environment)"""
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
TOKENS_PER_UPDATE = 1_048_576
ANCHOR = 4.67239832544785e9                       # FLOPs per input token (PROTOCOL.md)
RUNS = dict(run_000038=("dense_full", 0), run_000039=("attnres", 0), run_000040=("dense_full", 1),
            run_000041=("attnres_gate", 0), run_000042=("attnres_canon", 0), run_000043=("attnres_full", 0),
            run_000044=("attnres_deep", 0), run_000045=("attnres_ve_all", 0), run_000046=("dense_full_gate", 0),
            run_000047=("attnres_moe", 0), run_000048=("attnres_gate_canon", 0), run_000049=("attnres_full_gate", 0),
            run_000050=("attnres_full_canon", 0), run_000051=("attnres_moe16", 0), run_000052=("attnres_canon_abcd", 0),
            run_000053=("attnres_moe16", 0), run_000054=("attnres_full_canon_abcd", 0))


def read(run, name):
    path = HERE / "runs" / run / name
    if not path.exists(): return None
    return [json.loads(l) for l in path.read_text().splitlines()] if name.endswith(".jsonl") else json.loads(path.read_text())


def check(run, method, offset):
    spec, chain = read(run, "run_spec.json"), read(run, "chain.json")
    problems = []
    if (spec["scale"], spec["steps"], spec["schedule_steps"], spec["val_every"]) != ("M", 2000, 46875, 256):
        problems.append("not the 2,000-update M prefix of the formal schedule")
    if spec["seed"] != 42 + offset or spec["seed_offset"] != offset: problems.append("seed")
    if spec["official_evaluated"] or spec["dev_shard"] != "shard_06541.parquet": problems.append("evaluation split")
    out = dict(method=method, seed=spec["seed"], state=chain["state"], segments=len(chain["segments"]),
               restarts=len(chain.get("restarts", [])), jobs=[s["job"] for s in chain["segments"]])
    events = read(run, "metrics.jsonl")
    if events is None:                           # cancelled before admission: nothing ran
        if chain["state"] != "cancelled": problems.append("no metrics for a run that was not cancelled")
        return dict(out, problems=problems, passed=not problems)
    budget, status, prov = read(run, "budget.json"), read(run, "run_status.json"), read(run, "provenance.json")
    if len(spec["stage_costs"]) != 1: problems.append("more than one stage")
    cost, forward = int(next(iter(spec["stage_costs"].values()))), spec["forward_flops_per_token"]
    train = [e for e in events if e["event"] == "train"]
    evals = [e for e in events if e["event"] == "validation"]
    last = train[-1]["step"]
    if any(e.get("split") != "dev" for e in evals): problems.append("a non-dev evaluation")
    steps = [e["step"] for e in train]
    if steps != sorted(set(steps)) or steps[0] != 1: problems.append("logged updates")
    for e in train:
        before = sum(int(forward * v["input_tokens"]) for v in evals if v["step"] < e["step"])
        if e["input_tokens"] != e["step"] * TOKENS_PER_UPDATE: problems.append(f"tokens at step {e['step']}")
        if int(e["consumed_flops"]) != cost * e["step"] + before: problems.append(f"cost at step {e['step']}")
    eval_steps = sorted(v["step"] for v in evals)
    if eval_steps != [s for s in [*range(0, 2000, 256), 2000] if s <= last]: problems.append("evaluation steps")
    train_flops, eval_flops = cost * budget["step"], sum(int(forward * v["input_tokens"]) for v in evals)
    settled = int(budget["consumed_flops"])
    if not (int(budget["consumed_train_flops"]) == train_flops and int(budget["consumed_eval_flops"]) == eval_flops
            and settled == train_flops + eval_flops <= int(spec["reserved_flops"])):
        problems.append("settlement")
    ratio = cost / (ANCHOR * TOKENS_PER_UPDATE)
    if abs(ratio - 1) > 0.01: problems.append("per-update cost outside 1% of the anchor")
    dev = {v["step"]: v["val_bpb"] for v in evals}
    peak = max(e["peak_memory_bytes"] for e in train)
    out.update(micro_batch=prov["micro_batch"], accumulation=prov["accumulation"], parameters=prov["parameters"],
               last_update=last, train_flops=train_flops, cost_vs_anchor=ratio, settled_flops=settled,
               reserved_flops=int(spec["reserved_flops"]), gym_gpu_hours=budget["gpu_hours"],
               peak_memory_gib=round(peak / 2**30, 1),
               mean_step_seconds=round(sum(e["step_seconds"] for e in train[1:]) / max(len(train) - 1, 1), 3),
               dev_bpb=dev)
    if chain["state"] == "succeeded":
        if last != 2000 or budget["step"] != 2000: problems.append("did not reach update 2,000")
        if status["final"]["val_bpb_dev"] != dev.get(2000): problems.append("final dev BPB is not the update-2,000 evaluation")
        out["training_seconds"] = status["training_seconds"]
    elif chain["state"] != "cancelled": problems.append(f"chain state {chain['state']}")
    return dict(out, problems=problems, passed=not problems)


if __name__ == "__main__":
    runs = {r: check(r, *m) for r, m in RUNS.items() if (HERE / "runs" / r).exists()}
    print(json.dumps(dict(runs=runs, all_passed=all(r["passed"] for r in runs.values()),
                          settled_flops_total=sum(r.get("settled_flops", 0) for r in runs.values()),
                          gym_gpu_hours_total=sum(r.get("gym_gpu_hours", 0) for r in runs.values())), indent=1))
