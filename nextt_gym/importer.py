"""Import the upstream registered methods and completed formal results into the archive.

Imported attempts reuse the upstream cost proofs (no re-audit) and are tagged
`upstream`. Published run records (results/runs/) of the reference-v1 cohort for
these configured methods that finished all 46,875 updates become succeeded L-scale
runs with official-split BPB and CORE. Records without a configured method are skipped.
`final.accepted` says whether upstream accepted the record as complete (all
evaluations and a matched cumulative audit). Imported runs are not charged to the
gym ledger. Other cohorts use older compute accounting and are skipped.
Idempotent.
"""
import json
from pathlib import Path

from nexttransformer.models.registry import FAMILY

from . import upstream_model
from .task import ANCHOR_FLOPS_PER_TOKEN, FLOPS_TOLERANCE, SCALES

RESULT_COHORT = "reference-v1"     # the `cohort` of published records in results/runs/ that use this ledger
UNTRAINABLE = {"ouro": "Ouro's gate-only optimizer transition is not supported by the gym trainer"}


def family(method):
    return FAMILY.get(method, "baseline")


def _imported(archive):
    """method -> attempt_id for upstream attempts already in the archive."""
    out = {}
    for row in archive.store.all("SELECT id, title, tags FROM attempts WHERE origin='imported'"):
        if "upstream" in json.loads(row["tags"]): out[row["title"]] = row["id"]
    return out


def config_paths(repo_root):
    """configs/<family>/<method>.json of the registered reference methods (not the native track)."""
    return sorted(p for p in (Path(repo_root) / "configs").glob("*/*.json") if p.parent.name != "native")


def proof_audit(path, raw, repo_root):
    """The audit of an upstream method from its cost proof: stage costs and every scale's FLOPs per token."""
    calibration = raw["calibration"]
    proof = json.loads((path.parent / calibration["cost_record"]).read_text())
    stages = [dict(name=s["stage"] if s.get("depth") is None else f"depth{s['depth']}",
                   optimizer_steps=s["optimizer_steps"], total=s["cost"]["total"],
                   **{ph: s["cost"]["phases"][ph] for ph in ("forward", "backward", "optimizer")})
              for s in proof["stages"]]
    audit = dict(source="upstream cost proof", cost_record=str((path.parent / calibration["cost_record"]).resolve()
                                                               .relative_to(Path(repo_root).resolve())),
                 formula_version=calibration["formula_version"], unknown_ops=[], parameters=proof["parameters"],
                 flops_per_token=calibration["flops_per_token"], stages=stages,
                 scales=_scale_costs(raw["model"], stages))
    audit["within_tolerance"] = all(v["within_tolerance"] for v in audit["scales"].values())
    joint = next((st for st in stages if st["name"] == "joint"), stages[0])
    audit["forward_flops_per_token"] = joint["forward"] / (SCALES["L"].input_tokens // SCALES["L"].steps)
    return audit


def import_upstream(archive, repo_root):
    """Returns the created attempts, the existing ones whose audit gained a scale added since their import
    (`refreshed`, recomputed from the same proof), and the imported runs."""
    repo_root = Path(repo_root)
    existing = _imported(archive)
    created, refreshed, runs = [], [], []
    for path in config_paths(repo_root):
        raw = json.loads(path.read_text())
        method = raw["model"]["method"]
        if method in existing:
            stored = (archive.get_attempt(existing[method], include=("audit",)).get("audit") or {}).get("scales") or {}
            if set(SCALES) - set(stored):
                audit = proof_audit(path, raw, repo_root)
                archive.record_verification(existing[method], dict(status="passed", levels=[], audit=audit,
                                                                   parameters=audit["parameters"]))
                refreshed.append(existing[method])
            continue
        tags = [family(method), "upstream"] + (["gym-untrainable"] if method in UNTRAINABLE else [])
        files = {"model.py": upstream_model.MODEL_PY,
                 "config.json": json.dumps(upstream_model.submission_config(raw["model"]), indent=1) + "\n"}
        relative = path.relative_to(repo_root).as_posix()
        hypothesis = f"Upstream registered method {method} ({relative}); see the method sources (read_doc \"sources\")."
        if method in UNTRAINABLE: hypothesis += " Not trainable in the gym: " + UNTRAINABLE[method] + "."
        attempt_id = archive.put_attempt(files, title=method, hypothesis=hypothesis, tags=tags, origin="imported")
        audit = proof_audit(path, raw, repo_root)
        archive.record_verification(attempt_id, dict(status="passed", levels=[], audit=audit,
                                                     parameters=audit["parameters"]))
        existing[method] = attempt_id
        created.append(attempt_id)
    for path in sorted((repo_root / "results/runs").glob("*.json")):
        record = json.loads(path.read_text())
        if record.get("cohort") != RESULT_COHORT: continue
        method = record["methodId"]
        if method not in existing or record.get("status") != "finished" or record.get("finalBpb") is None: continue
        if record["step"] != SCALES["L"].steps: continue
        attempt_id = existing[method]
        source = path.relative_to(repo_root).as_posix()
        if any(r["extra"].get("source") == source for r in archive.list_runs(attempt_id=attempt_id, origin="imported")):
            continue
        runs.append(_import_record(archive, attempt_id, record, source))
    return dict(attempts=created, refreshed=refreshed, runs=runs)


def _scale_costs(model_config, stages):
    """Per-scale training FLOPs from the proof's per-update stage costs and each scale's stage schedule."""
    costs = {st["name"]: int(st["total"]) for st in stages}
    out = {}
    for name, scale in SCALES.items():
        schedule = {}
        for step in range(scale.steps):
            stage = upstream_model.stage_for_step(model_config, step, scale.schedule_steps)
            schedule[stage] = schedule.get(stage, 0) + 1
        total = sum(n * costs[st] for st, n in schedule.items())
        per_token = total / scale.input_tokens
        error = per_token / ANCHOR_FLOPS_PER_TOKEN - 1
        out[name] = dict(stage_steps=schedule, total_training_flops=float(total), total_training_flops_exact=str(total),
                         flops_per_token=per_token, relative_error_vs_anchor=error,
                         within_tolerance=abs(error) <= FLOPS_TOLERANCE)
    for name, scale in SCALES.items():
        if scale.prefix_of:
            out[name].update(prefix_of=scale.prefix_of, within_tolerance=out[scale.prefix_of]["within_tolerance"])
    return out


def _import_record(archive, attempt_id, record, source):
    scale = SCALES["L"]
    compute = record.get("trainingCompute") or {}
    core = record.get("core") or []
    run_id = archive.create_run(attempt_id, "L", gpu_type=record.get("hardware"), origin="imported",
                                status="succeeded", extra=dict(source=source, record_id=record["id"],
                                                               name=record.get("name"), url=record.get("url"),
                                                               accepted=bool(record.get("complete"))))
    final = dict(val_bpb_dev=None, val_bpb_official=record["finalBpb"], core=core[-1]["value"] if core else None,
                 accepted=bool(record.get("complete")),
                 cumulative_relative_error=None if not compute.get("averageGflopsPerInputToken") else
                 compute["averageGflopsPerInputToken"] * 1e9 / ANCHOR_FLOPS_PER_TOKEN - 1,
                 curve_points=len(record.get("val") or []))
    consumed = int(compute.get("trainingFlopsEstimate") or 0)
    archive.update_run(run_id, consumed_flops=consumed,
                       progress=dict(step=record["step"], total_steps=scale.steps,
                                     input_tokens=int(compute.get("inputTokens") or scale.input_tokens)),
                       latest=dict(val_bpb_official=record["finalBpb"], at_step=record["step"]), final=final)
    tokens = scale.input_tokens // scale.steps
    points = [dict(metric="val_bpb", split="official", step=int(p["step"]), input_tokens=int(p["step"]) * tokens,
                   value=p["value"]) for p in record.get("val") or []]
    points += [dict(metric="core", split="official", step=int(p["step"]), input_tokens=int(p["step"]) * tokens,
                    value=p["value"]) for p in core]
    archive.add_curve_points(run_id, points)
    return run_id
