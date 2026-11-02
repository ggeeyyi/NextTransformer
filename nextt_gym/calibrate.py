"""Bring a submission's training FLOPs/token to the anchor by adjusting declared tunable integers.

Cost-only, like nexttransformer/protocol/calibrate.py: no data or scores are used. Fields are
tried in the order the agent lists them; each search assumes cost is monotone
in the field over its declared [min, max] grid and keeps the value closest to
the anchor. Every candidate is fully re-audited.
"""
from .audit import audit_submission
from .errors import GymError
from .task import FLOPS_TOLERANCE


def _grid(rule, current):
    lo, hi, step = int(rule["min"]), int(rule["max"]), int(rule.get("step", 1))
    base = current if (current - lo) % step == 0 else lo
    return [v for v in range(lo, hi + 1) if (v - base) % step == 0]


def calibrate(sub, knobs, *, target=0.0, scale="L", max_audits=40):
    """Return (new_submission, changes, audit). Raises CALIBRATION_FAILED when no value fits."""
    unknown = [k for k in knobs if k not in sub.tunable]
    if unknown:
        raise GymError("INVALID_ARGUMENT", f"fields {unknown} are not declared tunable in config.json",
                       details=dict(tunable=sorted(sub.tunable)))
    audits = 0
    cache = {}

    def error_of(candidate):
        nonlocal audits
        key = tuple(sorted(candidate.model_config.items()))
        if key not in cache:
            audits += 1
            if audits > max_audits: raise GymError("CALIBRATION_FAILED", "audit budget exhausted")
            cache[key] = audit_submission(candidate)
        report = cache[key]
        if report["unknown_ops"]:
            raise GymError("AUDIT_UNKNOWN_OP", "cannot calibrate: uncounted operators",
                           details=dict(ops=report["unknown_ops"]))
        return report["scales"][scale]["relative_error_vs_anchor"] - target, report

    current, changes = sub, {}
    err, report = error_of(current)
    for field in knobs:
        if abs(err + target) <= FLOPS_TOLERANCE and abs(err) < 1e-3: break
        grid = _grid(sub.tunable[field], current.model_config[field])
        value = current.model_config[field]
        if value not in grid: grid = sorted(set(grid) | {value})
        i = grid.index(value)
        best = (abs(err), value, current, report)
        lo, hi = (i, len(grid) - 1) if err < 0 else (0, i)
        # Binary search for the sign change of the error inside [lo, hi].
        while hi - lo > 1:
            mid = (lo + hi) // 2
            cand = current.with_model_config(**{field: grid[mid]})
            e, r = error_of(cand)
            if abs(e) < best[0]: best = (abs(e), grid[mid], cand, r)
            if (e < 0) == (err < 0): lo, hi = (mid, hi) if err < 0 else (lo, mid)
            else: lo, hi = (lo, mid) if err < 0 else (mid, hi)
        for j in {lo, hi} - {i}:
            cand = current.with_model_config(**{field: grid[j]})
            e, r = error_of(cand)
            if abs(e) < best[0]: best = (abs(e), grid[j], cand, r)
        _, chosen, current, report = best
        if chosen != value: changes[field] = {"from": value, "to": chosen}
        err = report["scales"][scale]["relative_error_vs_anchor"] - target
    if not report["within_tolerance"]:
        raise GymError("CALIBRATION_FAILED", "no value of the tunable fields brings every scale within tolerance",
                       details=dict(best_relative_error={k: v["relative_error_vs_anchor"]
                                                         for k, v in report["scales"].items()}, changes=changes))
    report.pop("stage_records", None)
    return current, changes, report


def main(argv=None):
    """python -m nextt_gym.calibrate DIR --knobs a,b --out result.json (runs submitted code: use a subprocess)."""
    import argparse, json
    from pathlib import Path
    from .contract import load_submission
    parser = argparse.ArgumentParser(description=main.__doc__)
    parser.add_argument("submission"); parser.add_argument("--knobs", required=True)
    parser.add_argument("--target", type=float, default=0.0); parser.add_argument("--out", required=True)
    args = parser.parse_args(argv)
    sub = load_submission(args.submission)
    try:
        tuned, changes, report = calibrate(sub, args.knobs.split(","), target=args.target)
        result = dict(ok=True, changes=changes, config=tuned.config, audit=report)
    except GymError as exc:
        result = dict(ok=False, error=exc.to_dict())
    Path(args.out).write_text(json.dumps(result, default=str))


if __name__ == "__main__":
    main()
