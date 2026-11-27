"""Stage 3 GPU comparisons, from metrics.jsonl files copied beside this script (results/<name>/metrics.jsonl).

- G2b: the same registered method under the gym trainer (an S run) and the formal trainer (a 512-update smoke):
  training loss at every shared update; Stage 0's A1 limits (step-1 |difference| <= 1e-3, mean |difference| over
  updates 300-510 <= 0.015) as the yardstick.
- New code base: dense_full's dev BPB at update 512 against the Stage 0 S runs on the old code (0.887830, 0.887221;
  two equal S runs differed by about 0.0006 there).

    python compare.py pairs.json > compare.json       # [[gym_name, formal_name], ...]
"""
import json
from pathlib import Path
import statistics
import sys

HERE = Path(__file__).resolve().parent / "runs"
STAGE0_DENSE_S = {"run_000022": 0.887830, "run_000024": 0.887221}


def events(name):
    return [json.loads(line) for line in (HERE / name / "metrics.jsonl").read_text().splitlines() if line.strip()]


def losses(name):
    return {e["step"]: e.get("lm_loss", e["loss"]) for e in events(name) if e["event"] == "train"}


def seconds(name):
    times = [e["step_seconds"] for e in events(name) if e["event"] == "train" and e.get("step_seconds") and e["step"] > 20]
    return statistics.median(times) if times else None


def g2b(gym, formal, steps=512):
    """Both runs must log every update 1..steps (--log-every 1); otherwise the comparison is reported incomplete."""
    a, b = losses(gym), losses(formal)
    expected = set(range(1, steps + 1))
    missing = dict(gym=sorted(expected - set(a))[:10], formal=sorted(expected - set(b))[:10])
    shared = sorted(expected & set(a) & set(b))
    diff = {s: abs(a[s] - b[s]) for s in shared}
    late = [diff[s] for s in range(300, 511) if s in diff]
    complete = not missing["gym"] and not missing["formal"]
    return dict(gym=gym, formal=formal, complete=complete, missing=missing, shared_updates=len(shared),
                step1_difference=diff.get(1), mean_abs_difference_300_510=statistics.mean(late) if late else None,
                max_abs_difference=max(diff.values()) if diff else None,
                final_loss=dict(gym=a.get(steps), formal=b.get(steps)),
                passes_stage0_limits=(diff[1] <= 1e-3 and statistics.mean(late) <= 0.015) if complete else None,
                seconds_per_update=dict(gym=seconds(gym), formal=seconds(formal)))


def dev_at(name, step=512):
    return next((e["val_bpb"] for e in events(name) if e["event"] == "validation" and e.get("split") == "dev"
                 and e["step"] == step), None)


if __name__ == "__main__":
    pairs = json.loads(Path(sys.argv[1]).read_text())
    out = dict(g2b=[g2b(g, f) for g, f in pairs])
    dense = next((g for g, _ in pairs if "dense_full" in g), None)
    if dense:
        value = dev_at(dense)
        out["new_code_base"] = dict(run=dense, dev_bpb_512=value, stage0=STAGE0_DENSE_S,
                                    differences={k: None if value is None else round(value - v, 6)
                                                 for k, v in STAGE0_DENSE_S.items()})
    print(json.dumps(out, indent=1))
