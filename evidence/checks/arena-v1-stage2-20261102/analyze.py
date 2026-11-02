"""The preregistered Stage 2 analysis (plan, Stage 2 amendments before round 1), from the runs beside this script.

- Screen: reject a candidate whose dev BPB at update 512 exceeds the dense_full anchor's (run_000038) by more than 0.030.
- Noise: delta is the gap at update 2,000 between the two dense_full runs (run_000038, run_000040 with offset 1).
- Decision: a candidate beats the best anchor if its dev BPB at update 2,000 is lower than the lower of the two
  anchors' (dense_full run_000038, attnres run_000039) by more than max(2 delta, 0.001).
- Stop after 3 rounds, at the compute cap, or after 2 rounds without a candidate beating the best anchor.
Usage: python analyze.py  (prints JSON)"""
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
SCREEN = 0.030
ANCHORS = dict(dense_full="run_000038", attnres="run_000039")
REPEAT = "run_000040"
# run -> (candidate, round, the Gym attempt's parent, its primary change)
CANDIDATES = dict(
    run_000041=("attnres_gate", 1, "attnres", "head-wise sigmoid gate on the attention output"),
    run_000042=("attnres_canon", 1, "attnres", "Canon-A/C causal depthwise convolutions on sublayer inputs"),
    run_000043=("attnres_full", 1, "attnres", "Full AttnRes: attnres_groups = layers (8 -> 23)"),
    run_000044=("attnres_deep", 1, "attnres", "27 layers, FFN 4608 (calibrated)"),
    run_000045=("attnres_ve_all", 1, "attnres", "value embeddings on every layer"),
    run_000046=("dense_full_gate", 1, "dense_full", "head-wise gate on dense_full"),
    run_000047=("attnres_moe", 2, "attnres", "8-expert top-2 MoE FFN"),
    run_000048=("attnres_gate_canon", 2, "attnres", "gate of attnres_gate added to attnres_canon"),
    run_000049=("attnres_full_gate", 2, "attnres", "gate of attnres_gate added to attnres_full"),
    run_000050=("attnres_full_canon", 2, "attnres", "Canon-A/C of attnres_canon added to attnres_full"),
    run_000051=("attnres_moe16", 2, "attnres", "16-expert top-4 MoE FFN (micro batch 16; out of memory)"),
    run_000052=("attnres_canon_abcd", 2, "attnres", "Canon-B on q, k, v and Canon-D on the MLP hidden added to attnres_canon"),
    run_000053=("attnres_moe16", 2, "attnres", "16-expert top-4 MoE FFN (relaunch at micro batch 8)"),
    run_000054=("attnres_full_canon_abcd", 3, "attnres_canon_abcd", "Full AttnRes added to attnres_canon_abcd"),
)
CANDIDATES = {r: v for r, v in CANDIDATES.items() if (HERE / "runs" / r).exists()}


def dev(run):
    path = HERE / "runs" / run / "metrics.jsonl"
    if not path.exists(): return {}
    return {e["step"]: e["val_bpb"] for e in map(json.loads, path.read_text().splitlines()) if e["event"] == "validation"}


def state(run):
    return json.loads((HERE / "runs" / run / "chain.json").read_text())["state"]


curves = {r: dev(r) for r in [*ANCHORS.values(), REPEAT, *CANDIDATES]}
anchor_512 = curves[ANCHORS["dense_full"]][512]
delta = abs(curves[ANCHORS["dense_full"]][2000] - curves[REPEAT][2000])
margin = max(2 * delta, 0.001)
best_name, best_run = min(ANCHORS.items(), key=lambda kv: curves[kv[1]][2000])
best = curves[best_run][2000]
threshold = best - margin

rows = {}
for run, (name, rnd, parent, change) in CANDIDATES.items():
    c, st = curves[run], state(run)
    row = dict(candidate=name, round=rnd, parent=parent, change=change, state=st, dev_512=c.get(512), dev_2000=c.get(2000))
    if 512 in c:
        row["gap_512_vs_dense_full"] = c[512] - anchor_512
        row["screen"] = "rejected" if row["gap_512_vs_dense_full"] > SCREEN else "continues"
    else:
        row["screen"] = f"ended {st} before update 512"
    if 2000 in c:
        row["vs_best_anchor_2000"] = c[2000] - best
        row["beats_best_anchor"] = c[2000] < threshold
    rows[run] = row

finished = {r: v for r, v in rows.items() if v["dev_2000"] is not None}
ranking = sorted(finished, key=lambda r: finished[r]["dev_2000"])
rounds = {}
for rnd in sorted({v["round"] for v in rows.values()}):
    members = [r for r, v in rows.items() if v["round"] == rnd]
    rounds[rnd] = dict(runs=members, complete=all(rows[r]["state"] in ("succeeded", "cancelled", "failed") for r in members),
                       winners=[r for r in members if rows[r].get("beats_best_anchor")])

print(json.dumps(dict(
    anchors={k: dict(run=v, dev_512=curves[v][512], dev_2000=curves[v][2000]) for k, v in ANCHORS.items()},
    repeat=dict(run=REPEAT, dev_512=curves[REPEAT][512], dev_2000=curves[REPEAT][2000]),
    delta=delta, margin=margin, best_anchor=best_name, threshold=threshold, screen_threshold=SCREEN,
    candidates=rows, rounds=rounds,
    ranking=[dict(run=r, candidate=finished[r]["candidate"], dev_2000=finished[r]["dev_2000"],
                  vs_best_anchor=finished[r]["vs_best_anchor_2000"], beats=finished[r]["beats_best_anchor"]) for r in ranking],
), indent=1))
