"""Audit a finished formal MoE run, export its record, and check published records.

    python -m moe.audit runs/moe-d768-<method>                         # writes moe/evidence/audits/<track>/<method>.json
    python -m moe.audit runs/moe-d768-<method> --export --name "<display name>" \\
        --throughput throughput-d768.json                              # writes moe/results/<track>/<method>.json
    python -m moe.audit --check                                         # validates every published record

The audit re-reads what the trainer wrote and checks it against the protocol: formal mode in every segment,
unchanged protocol/track/model fingerprints, every validation event, every MoE-CORE event with all 14 entries,
the final full validation BPB and MoE-CORE, the ledger (all updates, all tokens, charged FLOPs within
2% of the track anchor) and a throughput measurement. It never records host names, paths or devices.
"""
import argparse
import json
import math
from pathlib import Path
import sys

from moe import compute
from moe.budgets import run_protocol
from moe.evaluation.recall import DATASETS as RECALL
from moe.evaluation.ruler import SCORED_LENGTHS
from moe.evaluation.score import ENTRIES
from moe.evaluation.short import TASKS as SHORT
from moe.spec import PROTOCOL, TRACKS, VALIDATION_SOURCES

HERE = Path(__file__).resolve().parent
AUDITS = HERE / "evidence" / "audits"
RESULTS = HERE / "results"


def events(run_dir):
    """Last event of each (kind, step); a resumed run may repeat one."""
    out = {}
    for line in (Path(run_dir) / "metrics.jsonl").read_text().splitlines():
        e = json.loads(line)
        out[(e["event"], e.get("step"))] = e
    return out


def segments(run_dir):
    """run.json of every segment (the trainer keeps the latest; resumed segments archive earlier ones)."""
    paths = sorted(Path(run_dir).glob("run*.json"))
    return [json.loads(p.read_text()) for p in paths]


def complete(e):
    """Problems of one MoE-CORE event: every short task in full, every long task on all its examples."""
    problems = []
    short, ruler, recall, score = (e.get(k, {}) for k in ("short", "ruler", "recall", "score"))
    if not short.get("full") or set(short.get("centered", {})) != set(SHORT):
        problems.append("the short tasks are not complete")
    if any(str(n) not in ruler for n in SCORED_LENGTHS) or set(recall) != set(RECALL):
        problems.append("the long tasks are not complete")
    if tuple(score.get("centered", {})) != ENTRIES:
        problems.append("MoE-CORE was not computed on all 14 entries")
    return problems


def audit(run_dir):
    run_dir = Path(run_dir)
    problems = []
    segs = segments(run_dir)
    if not segs:
        return ["run.json is missing"], {}
    run = segs[-1]
    track = TRACKS.get(run.get("track"))
    if track is None:
        return [f"unknown track {run.get('track')}"], {}
    protocol = run_protocol(track.name, run.get("protocol_hash"))
    for s in segs:
        if not s.get("formal"):
            problems.append("a training segment ran in research mode")
        if s.get("protocol_hash") != protocol.fingerprint:
            problems.append("protocol fingerprint differs from the current protocol")
        if s.get("track_hash") != track.fingerprint:
            problems.append("track fingerprint differs")
        if s.get("model_hash") != run.get("model_hash") or s.get("data_manifest_sha256") != run.get("data_manifest_sha256"):
            problems.append("segments disagree on model or data")
    ev = events(run_dir)
    for step in track.validation_steps:
        if ("val", step) not in ev and ("eval", step) not in ev:
            problems.append(f"missing validation event at step {step}")
    for step in track.core_steps:
        e = ev.get(("eval", step))
        if e is None:
            problems.append(f"missing MoE-CORE event at step {step}")
            continue
        problems += [f"step {step}: {p}" for p in complete(e)]
    final = ev.get(("final", track.updates))
    if final is None:
        problems.append("missing final evaluation (full validation BPB and MoE-CORE)")
    else:
        problems += [f"final: {p}" for p in complete(final)]
    ledger = json.loads((run_dir / "ledger.json").read_text()) if (run_dir / "ledger.json").exists() else {}
    if ledger.get("updates") != track.updates or ledger.get("input_tokens") != track.total_tokens:
        problems.append("ledger does not cover the full token budget")
    try:
        anchor = compute.anchor(track.name) * track.total_tokens
        gap = ledger.get("charged_flops", 0) / anchor - 1
        if abs(gap) > PROTOCOL.cumulative_tolerance:
            problems.append(f"cumulative compute {gap:+.2%} is outside the 2% window")
    except FileNotFoundError:
        gap = None
        problems.append("no baseline anchor for the track")
    throughput = [e for (kind, _), e in ev.items() if kind == "throughput"]
    if not throughput:
        problems.append("no throughput event")
    summary = dict(method=run.get("method"), track=track.name, model_hash=run.get("model_hash"),
                   protocol_hash=protocol.fingerprint, track_hash=track.fingerprint,
                   data_manifest_sha256=run.get("data_manifest_sha256"), segments=len(segs),
                   world_sizes=sorted({s.get("world_size") for s in segs}), cumulative_gap=gap,
                   updates=ledger.get("updates"), input_tokens=ledger.get("input_tokens"),
                   problems=problems, passed=not problems)
    return problems, summary


def export(run_dir, name, throughput_file=None, note=None):
    problems, summary = audit(run_dir)
    if problems:
        raise ValueError("the audit failed:\n" + "\n".join(problems))
    run = segments(run_dir)[-1]
    track = TRACKS[run["track"]]
    protocol = run_protocol(track.name, run["protocol_hash"])
    ev = events(run_dir)
    val = [[step, (ev.get(("eval", step)) or ev[("val", step)])["val"]["macro_bpb"]] for step in track.validation_steps]
    evals = []
    for step in track.core_steps:
        e = ev[("eval", step)]
        evals.append(dict(step=step, moe_core=e["score"]["moe_core"], short=e["score"]["short"],
                          long=e["score"]["long"], centered=e["score"]["centered"],
                          ruler_probe=e["score"]["ruler_probe"]))
    final = ev[("final", track.updates)]
    own = [e for (kind, _), e in ev.items() if kind == "throughput"][-1]
    record = dict(protocol=protocol.version, protocol_hash=protocol.fingerprint, track=track.name,
                  method=run["method"], name=name, model=run["model"], model_hash=run["model_hash"],
                  parameters=run["parameters"], flops_per_token=run["flops_per_token"],
                  training_flops=run["flops_per_token"] * track.total_tokens, tokens=track.total_tokens,
                  validation=val, evaluations=evals, final=dict(
                      val_macro_bpb=final["val"]["macro_bpb"], val_sources=final["val"]["sources"],
                      moe_core=final["score"]["moe_core"], short=final["score"]["short"],
                      long=final["score"]["long"], centered=final["score"]["centered"]),
                  throughput=dict(training=dict(tokens_per_second=own["tokens_per_second"],
                                                tokens_per_second_per_device=own["tokens_per_second_per_device"],
                                                mfu=own.get("mfu"), offload_activations=run.get("offload_activations", False))),
                  audit=f"moe/evidence/audits/{track.name}/{run['method']}.json", note=note)
    if throughput_file:
        harness = json.loads(Path(throughput_file).read_text())
        mine = next(r for r in harness["results"] if r["method"] == run["method"])
        record["throughput"]["harness"] = {k: mine[k] for k in (
            "tokens_per_second", "tokens_per_second_per_device", "mfu", "relative_throughput", "peak_memory_gib",
            "world_size", "micro_batch")}
        record["throughput"]["harness"]["offload_activations"] = mine.get("offload_activations", False)
    path = RESULTS / track.name / f"{run['method']}.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(record, indent=2) + "\n")
    return path


def numeric(value):
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)


def published_scores(event):
    centered = event.get("centered", {})
    if set(centered) != set(ENTRIES) or not all(numeric(v) for v in centered.values()):
        return False
    groups = {"moe_core": ENTRIES, "short": SHORT, "long": [k for k in ENTRIES if k not in SHORT]}
    return all(numeric(event.get(key)) and math.isclose(event[key], sum(centered[k] for k in tasks) / len(tasks),
                                                        rel_tol=1e-6, abs_tol=1e-6)
               for key, tasks in groups.items())


def check_record(path):
    """Validate published evidence without changing the serialized protocol of early formal runs."""
    r = json.loads(path.read_text())
    method, track_name = path.stem, path.parent.name
    track = TRACKS.get(track_name)
    if track is None or (r.get("track"), r.get("method")) != (track_name, method):
        return ["file name disagrees with track/method"]
    problems = []
    protocol = run_protocol(track_name, r.get("protocol_hash"))
    if (r.get("protocol"), r.get("protocol_hash")) != (protocol.version, protocol.fingerprint):
        problems.append("protocol fingerprint differs")
    suffix = Path(track_name) / path.name
    config = json.loads((HERE / "configs" / suffix).read_text())
    cost = json.loads((HERE / "evidence/costs" / suffix).read_text())
    expected_cost = (Path("moe/evidence/costs") / suffix).as_posix()
    if (config["calibration"]["status"] != "admitted" or config["calibration"]["cost_record"] != expected_cost
            or cost.get("admitted") is not True or (cost.get("track"), cost.get("method")) != (track_name, method)
            or config["track_hash"] != track.fingerprint or config["model_hash"] != r.get("model_hash")
            or cost["model_hash"] != config["model_hash"] or config["model"] != r.get("model")):
        problems.append("record disagrees with the admitted configuration or cost proof")
    expected_audit = (Path("moe/evidence/audits") / suffix).as_posix()
    audit = json.loads((AUDITS / suffix).read_text())
    if (r.get("audit") != expected_audit or audit.get("passed") is not True or audit.get("problems") != []
            or any(audit.get(key) != r.get(key) for key in ("track", "method", "model_hash", "protocol_hash"))
            or audit.get("track_hash") != track.fingerprint or audit.get("updates") != track.updates
            or audit.get("input_tokens") != track.total_tokens or r.get("tokens") != track.total_tokens):
        problems.append("no matching passing audit for the complete budget")
    evaluations = r.get("evaluations", [])
    if [e.get("step") for e in evaluations] != list(track.core_steps) or not all(published_scores(e) for e in evaluations):
        problems.append("scheduled MoE-CORE evaluations differ")
    validation = r.get("validation", [])
    if [s for s, _ in validation] != list(track.validation_steps) or not all(numeric(v) for _, v in validation):
        problems.append("periodic validation schedule or scores differ")
    final = r.get("final", {})
    sources = final.get("val_sources", {})
    if (not published_scores(final) or set(sources) != set(VALIDATION_SOURCES)
            or not all(numeric(v.get("bpb")) for v in sources.values()) or not numeric(final.get("val_macro_bpb"))
            or not math.isclose(final["val_macro_bpb"], sum(v["bpb"] for v in sources.values())/len(VALIDATION_SOURCES),
                                rel_tol=1e-6, abs_tol=1e-6)):
        problems.append("invalid final MoE-CORE or full macro BPB")
    return problems


def check():
    problems = []
    for path in sorted(RESULTS.glob("*/*.json")):
        try:
            errors = check_record(path)
        except (OSError, TypeError, ValueError, KeyError, AttributeError) as error:
            errors = [str(error)]
        problems.extend(f"{path.relative_to(HERE.parent)}: {error}" for error in errors)
    return problems


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("run", nargs="?", type=Path)
    p.add_argument("--export", action="store_true")
    p.add_argument("--name")
    p.add_argument("--throughput", type=Path, help="moe.throughput output measured on the same node")
    p.add_argument("--note")
    p.add_argument("--check", action="store_true")
    args = p.parse_args(argv)
    if args.check:
        problems = check()
        print("\n".join(problems) if problems else f"{len(list(RESULTS.glob('*/*.json')))} MoE records are valid")
        raise SystemExit(1 if problems else 0)
    if args.run is None:
        p.error("pass a run directory or --check")
    problems, summary = audit(args.run)
    if summary:
        path = AUDITS / summary["track"] / f"{summary['method']}.json"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(summary, indent=2) + "\n")
        print(f"audit written to {path.relative_to(HERE.parent)}")
    print("\n".join(problems) if problems else "audit passed")
    if problems:
        raise SystemExit(1)
    if args.export:
        if not args.name:
            p.error("--export needs --name")
        print(f"record written to {export(args.run, args.name, args.throughput, args.note).relative_to(HERE.parent)}")


if __name__ == "__main__":
    sys.exit(main())
