"""Recompute the arena v1 Stage 1 run checks from the files beside this script; print the result as JSON.

Each scale-C run must train 512 updates on its own 512-update schedule at seed 42 + its offset; its logged updates
must carry continuous tokens and the cumulative cost of the audited per-update costs plus the evaluations before them;
it must settle exactly at that cost, stay within its reservation and within 1% of the anchor cost on average, and
report the dev BPB of its final evaluation. `huginn` samples its depth stage per update: the sequence comes from its
Gym submission (`submissions/huginn/`, a copy of attempt att_000020) and is checked against the logged stages.
Usage: python check.py  (from the repository's environment)"""
import json
from pathlib import Path

from nextt_gym.contract import load_submission

HERE = Path(__file__).resolve().parent
TOKENS_PER_UPDATE = 1_048_576
ANCHOR = 4.67239832544785e9                       # FLOPs per input token (PROTOCOL.md)
RUNS = dict(run_000025=("dense_full", 0), run_000026=("gdn", 0), run_000027=("kda", 0), run_000028=("dense_full", 1),
            run_000029=("dense_full", 2), run_000030=("huginn", 0), run_000031=("ut_act", 0), run_000032=("deeploop", 0),
            run_000033=("fixed_recursive", 0), run_000034=("attnres", 0), run_000035=("residual_standard", 0),
            run_000036=("retnet", 0), run_000037=("elu_qknorm", 0))


def read(run, name):
    path = HERE / "runs" / run / name
    return [json.loads(l) for l in path.read_text().splitlines()] if name.endswith(".jsonl") else json.loads(path.read_text())


def update_stages(spec, method):
    """The stage of updates 1..512, in order."""
    if len(spec["stage_costs"]) == 1: return [next(iter(spec["stage_costs"]))] * spec["steps"]
    sub = load_submission(HERE / "submissions" / method)
    return [sub.stage_for_step(k, spec["schedule_steps"]) for k in range(spec["steps"])]


def check(run, method, offset):
    spec, budget, status, events = (read(run, n) for n in ("run_spec.json", "budget.json", "run_status.json", "metrics.jsonl"))
    chain, prov = read(run, "chain.json"), read(run, "provenance.json")
    stages, forward = update_stages(spec, method), spec["forward_flops_per_token"]
    costs = [int(spec["stage_costs"][s]) for s in stages]
    cumulative = [0]
    for c in costs: cumulative.append(cumulative[-1] + c)
    train = [e for e in events if e["event"] == "train"]
    evals = [e for e in events if e["event"] == "validation"]
    problems = []
    if (spec["scale"], spec["steps"], spec["schedule_steps"]) != ("C", 512, 512): problems.append("not a 512-update C schedule")
    if spec["seed"] != 42 + offset or spec["seed_offset"] != offset: problems.append("seed")
    if spec["official_evaluated"] or any(e.get("split") != "dev" for e in evals): problems.append("a non-dev evaluation")
    steps = [e["step"] for e in train]
    if steps != sorted(set(steps)) or steps[0] != 1 or steps[-1] != 512: problems.append("logged updates")
    if any(stages[e["step"] - 1] != e["stage"] for e in train): problems.append("logged stages differ from the schedule")
    for e in train:
        before = sum(int(forward * v["input_tokens"]) for v in evals if v["step"] < e["step"])
        if e["input_tokens"] != e["step"] * TOKENS_PER_UPDATE: problems.append(f"tokens at step {e['step']}")
        if int(e["consumed_flops"]) != cumulative[e["step"]] + before: problems.append(f"cost at step {e['step']}")
    train_flops, eval_flops = cumulative[-1], sum(int(forward * v["input_tokens"]) for v in evals)
    settled = int(budget["consumed_flops"])
    if not (int(budget["consumed_train_flops"]) == train_flops and int(budget["consumed_eval_flops"]) == eval_flops
            and settled == train_flops + eval_flops <= int(spec["reserved_flops"])):
        problems.append("settlement")
    ratio = train_flops / len(costs) / (ANCHOR * TOKENS_PER_UPDATE)
    if abs(ratio - 1) > 0.01: problems.append("per-update cost outside 1% of the anchor")
    final = {e["step"]: e["val_bpb"] for e in evals}
    if status["final"]["val_bpb_dev"] != final.get(512): problems.append("final dev BPB is not the update-512 evaluation")
    if chain["state"] != "succeeded" or chain["step"] != 512: problems.append("chain did not succeed at 512")
    return dict(method=method, seed=spec["seed"], micro_batch=prov["micro_batch"], accumulation=prov["accumulation"],
                parameters=prov["parameters"], segments=len(chain["segments"]), restarts=len(chain.get("restarts", [])),
                stage_updates={k: stages.count(k) for k in spec["stage_costs"]},
                train_flops=train_flops, cost_vs_anchor=ratio, settled_flops=settled,
                reserved_flops=int(spec["reserved_flops"]), gym_gpu_hours=budget["gpu_hours"],
                training_seconds=status["training_seconds"], dev_bpb=final, problems=problems, passed=not problems)


if __name__ == "__main__":
    runs = {r: check(r, *m) for r, m in RUNS.items()}
    print(json.dumps(dict(runs=runs, all_passed=all(r["passed"] for r in runs.values()),
                          settled_flops_total=sum(r["settled_flops"] for r in runs.values()),
                          gym_gpu_hours_total=sum(r["gym_gpu_hours"] for r in runs.values())), indent=1))
