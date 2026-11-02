"""Stage 1 screen comparison under the Stage 2 reject rule (gap to the dense_full anchor above a threshold).

For each proxy (C at 512, published prefix at 500 and at 2,000): every row's gap to dense_full, the smallest threshold
that keeps every row within 0.005 of the best final, and the rows that threshold rejects. Prints JSON.
Usage: screen.py [ANALYSIS_JSON=analysis.json beside this script]"""
import json, sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]
GAP = 0.005

c = json.loads(Path(sys.argv[1] if len(sys.argv) > 1 else HERE / "analysis.json").read_text())["c_dev_bpb_512"]
runs = {m: json.loads((REPO / "results/runs" / f"{m}.json").read_text()) for m in c}
final = {m: r["finalBpb"] for m, r in runs.items()}
curve = {m: {p["step"]: p["value"] for p in r["val"]} for m, r in runs.items()}
proxies = dict(C_512=c, prefix_500={m: curve[m][500] for m in c}, prefix_2000={m: curve[m][2000] for m in c})
best = min(final.values())
top = sorted(m for m in c if final[m] - best <= GAP)
out = dict(top_rows=top, proxies={})
for name, proxy in proxies.items():
    gap = {m: proxy[m] - proxy["dense_full"] for m in c}
    threshold = max(gap[m] for m in top)
    kept = sorted((m for m in c if gap[m] <= threshold), key=gap.get)
    out["proxies"][name] = dict(gap_to_dense_full={m: round(gap[m], 4) for m in sorted(c, key=gap.get)},
                                threshold_keeping_top_rows=round(threshold, 4), kept=kept,
                                rejected=[m for m in sorted(c, key=final.get) if m not in kept],
                                worst_final_kept=max(kept, key=final.get))
print(json.dumps(out, indent=1))
