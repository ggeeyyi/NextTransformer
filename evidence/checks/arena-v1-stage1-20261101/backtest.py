"""Prefix backtest on published curves: rank agreement of validation BPB at update k with the final BPB.
Usage: backtest.py  (reads results/runs/*.json; prints tables; writes backtest.json beside this script)."""
import itertools, json, glob, os
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]
runs = {Path(p).stem: json.load(open(p)) for p in glob.glob(str(REPO / "results/runs/*.json"))}
# The finished runs published when Stage 1 ran (2026-11-01). Runs published later stay out, so that this script
# reproduces backtest.json.
PUBLISHED_AT_STAGE1 = [
    "attnres", "deeploop", "deltanet", "dense_full", "dsa", "elu_qknorm", "fixed_recursive", "gdn", "gla",
    "gqa", "huginn", "kda", "mla_nope_qknorm", "mla_qknorm", "moba", "mqa", "nsa", "ouro", "qsa",
    "residual_standard", "retnet", "seer", "ut_act",
]
final = {m: r["finalBpb"] for m, r in runs.items()
         if m in PUBLISHED_AT_STAGE1 and r.get("finalBpb") and r["status"] == "finished"}
curve = {m: {p["step"]: p["value"] for p in runs[m]["val"]} for m in final}
SETS = {
    "plan nine": ["dense_full", "attnres", "residual_standard", "kda", "gdn", "retnet", "elu_qknorm", "fixed_recursive", "huginn"],
    "eleven (decision rows)": ["dense_full", "attnres", "residual_standard", "kda", "gdn", "retnet", "elu_qknorm",
                               "fixed_recursive", "huginn", "ut_act", "deeploop"],
    "13 trainable": ["dense_full", "attnres", "residual_standard", "kda", "gdn", "retnet", "elu_qknorm", "fixed_recursive",
                     "huginn", "gla", "deltanet", "ut_act", "deeploop"],
    f"all {len(final)}": sorted(final),
}
GAP = 0.005


def ranks(xs):
    order = sorted(range(len(xs)), key=lambda i: xs[i]); r = [0] * len(xs)
    for k, i in enumerate(order): r[i] = k
    return r


def spearman(a, b):
    ra, rb = ranks(a), ranks(b); n = len(a)
    return 1 - 6 * sum((x - y) ** 2 for x, y in zip(ra, rb)) / (n * (n * n - 1))


def score(methods, step):
    x, y = [curve[m][step] for m in methods], [final[m] for m in methods]
    pairs = [(i, j) for i, j in itertools.combinations(range(len(methods)), 2) if abs(y[i] - y[j]) > GAP]
    right = sum((x[i] < x[j]) == (y[i] < y[j]) for i, j in pairs)
    return dict(spearman=spearman(x, y), pair_accuracy=right / len(pairs), pairs=len(pairs),
                wrong=[(methods[i], methods[j]) for i, j in pairs if (x[i] < x[j]) != (y[i] < y[j])])


STEPS = [250, 500, 750, 1000, 1500, 2000, 3000, 4000, 6000, 8000, 12000, 16500, 25000, 35000, 46875]
out = {}
for name, methods in SETS.items():
    print(f"\n{name} ({len(methods)} rows): update  Spearman  pair-acc (pairs with final gap > {GAP})")
    out[name] = {}
    for s in STEPS:
        if not all(s in curve[m] for m in methods): continue
        r = score(methods, s); out[name][s] = r
        print(f"  {s:6d}  {r['spearman']:.3f}  {r['pair_accuracy']:.3f} ({r['pairs']})  wrong: {r['wrong'][:6]}")
json.dump(dict(final=final, sets=SETS, scores=out), open(HERE / "backtest.json", "w"), indent=1)
