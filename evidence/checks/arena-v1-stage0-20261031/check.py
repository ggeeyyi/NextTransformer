"""Recompute the arena v1 Stage 0 checks A1-A3 from the files beside this script; print the result as JSON.

A1, A2 and A3 are preregistered in docs/plans/ARENA_V1_RESEARCH.md (Stage 0). Usage: python check.py
"""
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
TOKENS_PER_UPDATE = 1_048_576
METHODS = {"run_000021": "gdn", "run_000022": "dense_full", "run_000023": "kda", "run_000024": "dense_full"}
FORMAL_UPDATES = 46_875


def audited_per_update(method):
    """The formal run's theoretical FLOPs per update: step 1 of the H20 run, or the A800 Dense audit's total / updates."""
    if method == "dense_full":
        return json.loads((HERE.parent / "a800-dense-full-formal.json").read_text())["theory_flops"] // FORMAL_UPDATES
    head = json.loads((HERE / "formal" / f"{method}_h20_seed42_head.json").read_text())
    return next(e["theory_flops"] for e in head if e["event"] == "train" and e["step"] == 1)


def read(run, name):
    path = HERE / "runs" / run / name
    return [json.loads(l) for l in path.read_text().splitlines()] if name.endswith(".jsonl") else json.loads(path.read_text())


def a1(run, method):
    """|step-1 loss difference| <= 1e-3 and mean |difference| over logged updates 300-510 <= 0.015."""
    formal = {e["step"]: e["lm_loss"] for e in json.loads((HERE / "formal" / f"{method}_h20_seed42_head.json").read_text())
              if e["event"] == "train"}
    gym = {e["step"]: e["loss"] for e in read(run, "metrics.jsonl") if e["event"] == "train"}
    common = sorted(set(formal) & set(gym)); window = [s for s in common if 300 <= s <= 510]
    step1, mean = abs(formal[1] - gym[1]), sum(abs(formal[s] - gym[s]) for s in window) / len(window)
    return dict(common_steps=len(common), step1_abs_diff=step1, window_steps=len(window), window_mean_abs_diff=mean,
                max_abs_diff=max(abs(formal[s] - gym[s]) for s in common), passed=step1 <= 1e-3 and mean <= 0.015)


def a3(run, method):
    """Settled FLOPs = steps x audited per-update cost + sum over evaluations of int(forward cost x tokens)."""
    spec, budget, events = read(run, "run_spec.json"), read(run, "budget.json"), read(run, "metrics.jsonl")
    per_update = int(spec["stage_costs"]["joint"])
    evaluations = [e["input_tokens"] for e in events if e["event"] == "validation"]
    train, evaluation = spec["steps"] * per_update, sum(int(spec["forward_flops_per_token"] * t) for t in evaluations)
    settled = int(budget["consumed_flops"])
    audited = audited_per_update(method)
    return dict(per_update_flops=per_update, audited_per_update_flops=audited, train_flops=train, eval_tokens=sum(evaluations), eval_flops=evaluation,
                settled_flops=settled, reserved_flops=int(spec["reserved_flops"]),
                settled_over_reserved=settled / int(spec["reserved_flops"]), gym_gpu_hours=budget["gpu_hours"],
                passed=(per_update == audited and int(budget["consumed_train_flops"]) == train and int(budget["consumed_eval_flops"]) == evaluation
                        and settled == train + evaluation <= int(spec["reserved_flops"])
                        and budget["eval_tokens"] == sum(evaluations)))


def a2(run):
    """Every logged update and its token count and cumulative cost continue across the segment boundaries: planned
    segment ends, or restarts from the latest checkpoint after a segment ended without the trainer's final status."""
    spec, events, chain = read(run, "run_spec.json"), read(run, "metrics.jsonl"), read(run, "chain.json")
    per_update, forward = int(spec["stage_costs"]["joint"]), spec["forward_flops_per_token"]
    train = [e for e in events if e["event"] == "train"]
    steps = [e["step"] for e in train]
    problems = []
    if steps != sorted(set(steps)): problems.append("logged steps repeat or go backwards")
    for e in train:
        before = sum(int(forward * v["input_tokens"]) for v in events if v["event"] == "validation" and v["step"] < e["step"])
        if e["input_tokens"] != e["step"] * TOKENS_PER_UPDATE: problems.append(f"tokens at step {e['step']}")
        if int(e["consumed_flops"]) != e["step"] * per_update + before: problems.append(f"cost at step {e['step']}")
    restarts, boundaries = {r["segment"]: r for r in chain.get("restarts", [])}, []
    for s in chain["segments"][:-1]:
        if s["status"][0] == "segment_complete": boundaries.append(s["status"][1])
        elif s["n"] in restarts and restarts[s["n"]]["resumed_from"] == s.get("checkpoint"): boundaries.append(s["checkpoint"])
        else: problems.append(f"segment {s['n']} neither ended cleanly nor restarted from its checkpoint")
    jobs = [s["job"] for s in chain["segments"]]
    if len(set(jobs)) != len(jobs) or len(jobs) < 2: problems.append("segments are not separate jobs")
    losses = {e["step"]: e["loss"] for e in train}
    around = {b: {s: losses[s] for s in sorted(losses) if b - 20 <= s <= b + 20} for b in boundaries}
    return dict(segments=len(jobs), boundaries=boundaries, restarts=list(restarts.values()), final_step=chain["step"],
                logged_updates=len(steps), losses_around_boundaries=around, problems=problems,
                passed=not problems and chain["state"] == "succeeded" and chain["step"] == spec["steps"])


def replay(run):
    """The events a stopped segment logged after its checkpoint (archived by the trainer when it resumed) against the
    same events after the restart: input tokens and cumulative cost must be equal; value differences are reported."""
    restart = read(run, "chain.json")["restarts"][0]
    final = {(e["event"], e["step"]): e for e in read(run, "metrics.jsonl")}
    rows, problems = [], []
    for e in read(run, "replayed_metrics.jsonl"):
        after, key = final.get((e["event"], e["step"])), "loss" if e["event"] == "train" else "val_bpb"
        if not restart["resumed_from"] < e["step"] <= restart["failed_at_step"]: problems.append(f"step {e['step']} outside the replay")
        if after is None: problems.append(f"{e['event']} {e['step']} missing after the restart"); continue
        if e["input_tokens"] != after["input_tokens"] or e.get("consumed_flops") != after.get("consumed_flops"):
            problems.append(f"tokens or cost differ at {e['event']} {e['step']}")
        rows.append(dict(event=e["event"], step=e["step"], before=e[key], after=after[key], abs_diff=abs(e[key] - after[key])))
    return dict(resumed_from=restart["resumed_from"], failed_at_step=restart["failed_at_step"], events=rows,
                problems=problems, passed=bool(rows) and not problems)


def agreement(run, reference):
    """Two runs of one method: training loss at every shared logged update and dev BPB at every evaluation."""
    def values(r, event, key): return {e["step"]: e[key] for e in read(r, "metrics.jsonl") if e["event"] == event}
    loss, ref = values(run, "train", "loss"), values(reference, "train", "loss")
    common = sorted(set(loss) & set(ref)); window = [s for s in common if 300 <= s <= 510]
    bpb, ref_bpb = values(run, "validation", "val_bpb"), values(reference, "validation", "val_bpb")
    return dict(reference=reference, common_steps=len(common), step1_abs_diff=abs(loss[1] - ref[1]),
                window_mean_abs_diff=sum(abs(loss[s] - ref[s]) for s in window) / len(window),
                max_abs_diff=max(abs(loss[s] - ref[s]) for s in common),
                dev_bpb={s: dict(run=bpb[s], reference=ref_bpb[s], diff=bpb[s] - ref_bpb[s]) for s in sorted(bpb)})


if __name__ == "__main__":
    print(json.dumps(dict(
        A1={METHODS[r]: a1(r, METHODS[r]) for r in ("run_000021", "run_000023")},
        A2={"run_000022": a2("run_000022"),
            "run_000024": dict(a2("run_000024"), replay=replay("run_000024"), agreement=agreement("run_000024", "run_000022"))},
        A3={r: dict(method=METHODS[r], **a3(r, METHODS[r])) for r in METHODS},
    ), indent=1))
