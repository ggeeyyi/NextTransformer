"""Stage 1 (D1) analysis, as preregistered in docs/plans/ARENA_V1_RESEARCH.md (Stage 1 amendments).

Reads the scale-C dev BPB at update 512 from the run files beside this script (or from the Gym store, or a JSON file
{method: bpb, "dense_full@1": ..., ...}) and compares its ranking with the published final BPB, next to the published
prefix at update 500.
Usage: analyze.py [RUNS_DIR | GYM_DB | RESULTS.json] > analysis.json"""
import itertools, json, random, sqlite3, statistics, sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]
RUNS = dict(run_000025="dense_full", run_000026="gdn", run_000027="kda", run_000028="dense_full@1",
            run_000029="dense_full@2", run_000030="huginn", run_000031="ut_act", run_000032="deeploop",
            run_000033="fixed_recursive", run_000034="attnres", run_000035="residual_standard", run_000036="retnet",
            run_000037="elu_qknorm")
NINE = ["dense_full", "attnres", "residual_standard", "kda", "gdn", "retnet", "elu_qknorm", "fixed_recursive", "huginn"]
ELEVEN = NINE + ["ut_act", "deeploop"]   # gla and deltanet dropped before launch (plan, second amendment)
GAP, RANKS = 0.005, dict(spearman=0.85, pair_accuracy=0.9)


def published():
    final, prefix = {}, {}
    for m in ELEVEN:
        r = json.loads((REPO / "results/runs" / f"{m}.json").read_text())
        final[m], prefix[m] = r["finalBpb"], {p["step"]: p["value"] for p in r["val"]}[500]
    return final, prefix


def measured(source):
    """{method: dev BPB at 512 of the seed-0 C run} and {seed offset: dense_full BPB}."""
    if Path(source).is_dir():
        values = {RUNS[r]: json.loads((Path(source) / r / "run_status.json").read_text())["final"]["val_bpb_dev"] for r in RUNS}
    elif source.endswith(".json"): values = json.loads(Path(source).read_text())
    else:
        db = sqlite3.connect(f"file:{source}?mode=ro", uri=True)
        rows = db.execute("SELECT a.title, r.seed_offset, r.final FROM runs r JOIN attempts a ON a.id = r.attempt_id "
                          "WHERE r.scale = 'C' AND r.status = 'succeeded' AND r.origin = 'agent'").fetchall()
        values = {(t if s == 0 else f"{t}@{s}"): json.loads(f)["val_bpb_dev"] for t, s, f in rows}
    seeds = {0: values["dense_full"], **{int(k.split("@")[1]): v for k, v in values.items() if k.startswith("dense_full@")}} \
        if "dense_full" in values else {}
    return {m: v for m, v in values.items() if "@" not in m}, seeds


def ranks(xs):
    order = sorted(range(len(xs)), key=lambda i: xs[i]); r = [0.0] * len(xs)
    for k, i in enumerate(order): r[i] = k
    return r


def spearman(a, b):
    ra, rb = ranks(a), ranks(b); ma, mb = statistics.mean(ra), statistics.mean(rb)
    num = sum((x - ma) * (y - mb) for x, y in zip(ra, rb))
    den = (sum((x - ma) ** 2 for x in ra) * sum((y - mb) ** 2 for y in rb)) ** 0.5
    return num / den if den else float("nan")


def pair_accuracy(proxy, final):
    pairs = [(i, j) for i, j in itertools.combinations(range(len(final)), 2) if abs(final[i] - final[j]) > GAP]
    if not pairs: return float("nan"), 0, []
    wrong = [(i, j) for i, j in pairs if (proxy[i] < proxy[j]) != (final[i] < final[j])]
    return 1 - len(wrong) / len(pairs), len(pairs), wrong


def score(methods, proxy, final):
    x, y = [proxy[m] for m in methods], [final[m] for m in methods]
    acc, n, wrong = pair_accuracy(x, y)
    return dict(rows=len(methods), spearman=spearman(x, y), pair_accuracy=acc, pairs=n,
                wrong=[(methods[i], methods[j]) for i, j in wrong])


def bootstrap(methods, c, prefix, final, draws=10000, seed=0):
    """90% intervals over rows resampled with replacement (duplicates tie in both rankings), and the share of draws
    in which C's statistic exceeds the prefix's on the same rows."""
    rng, out = random.Random(seed), dict(spearman=[], pair_accuracy=[], c_beats_prefix=dict(spearman=0, pair_accuracy=0))
    for _ in range(draws):
        sample = [rng.choice(methods) for _ in methods]
        if len(set(sample)) < 4: continue
        y = [final[m] for m in sample]
        sc, sp = spearman([c[m] for m in sample], y), spearman([prefix[m] for m in sample], y)
        ac, ap = pair_accuracy([c[m] for m in sample], y)[0], pair_accuracy([prefix[m] for m in sample], y)[0]
        out["spearman"].append(sc); out["pair_accuracy"].append(ac)
        out["c_beats_prefix"]["spearman"] += sc > sp; out["c_beats_prefix"]["pair_accuracy"] += ac > ap
    n = len(out["spearman"])
    q = lambda xs, p: sorted(v for v in xs if v == v)[int(p * (len([v for v in xs if v == v]) - 1))]
    return dict(draws=n, spearman_90=[q(out["spearman"], .05), q(out["spearman"], .95)],
                pair_accuracy_90=[q(out["pair_accuracy"], .05), q(out["pair_accuracy"], .95)],
                c_beats_prefix={k: v / n for k, v in out["c_beats_prefix"].items()})


def main(source):
    final, prefix = published()
    c, seeds = measured(source)
    where = Path(source).resolve()
    report = dict(source=str(where.relative_to(REPO)) if where.is_relative_to(REPO) else source, c_dev_bpb_512=c,
                  dense_full_seeds=seeds)
    if len(seeds) > 1:
        report["noise"] = dict(spread=max(seeds.values()) - min(seeds.values()), sd=statistics.stdev(seeds.values()))
    for name, rows in (("eleven", ELEVEN), ("nine", NINE)):
        have = [m for m in rows if m in c]
        entry = dict(missing=[m for m in rows if m not in c], c=score(have, c, final), prefix_500=score(have, prefix, final))
        if len(have) >= 4: entry["bootstrap"] = bootstrap(have, c, prefix, final)
        entry["c_ranks"] = not entry["missing"] and all(entry["c"][k] >= v for k, v in RANKS.items())
        report[name] = entry
    # Screen: the rows within GAP of the best final, and how far behind the C leader each sits.
    have = [m for m in ELEVEN if m in c]
    if have:
        best = min(final[m] for m in have); lead = min(c[m] for m in have)
        report["screen"] = dict(top_rows={m: dict(final=final[m], c_behind_leader=c[m] - lead)
                                          for m in have if final[m] - best <= GAP},
                                c_order=sorted(have, key=c.get), final_order=sorted(have, key=final.get))
    print(json.dumps(report, indent=1))


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else str(HERE / "runs"))
