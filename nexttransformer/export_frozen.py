"""Export the frozen A01-A03 run-directory schema with its original accounting."""
import argparse
from datetime import datetime
import hashlib
import json
import math
from pathlib import Path

from nexttransformer.export_run import P, REPO, RESULTS, record

COHORT = "xiangyue-20261019"
ACCOUNTING = "reference-aten-math-v2-all-backends-replay"
BASE_COMMIT = "f41786282acf98f54292774972f6e3509ddbeb51"
NOTE = "Frozen all-backends-replay accounting; original cumulative FLOPs and source fingerprints are retained."


def check_audit(run, audit):
    """Check a record against a concise cumulative audit, as for other runs."""
    c, p, f = run["trainingCompute"], run["proof"], audit["fingerprints"]
    pairs = [(run["id"], audit["run_id"]), (run["methodId"], audit["method"]),
             (run["step"], audit["optimizer_steps"]), (c["inputTokens"], audit["input_tokens"]),
             (c["trainingFlopsEstimate"], audit["theory_flops"]), (c["accounting"], ACCOUNTING),
             (audit["formula_version"], ACCOUNTING), (run["reportedAudit"], audit["final_status"]),
             (run["val"][-1]["value"], audit["periodic_final_val_bpb"]), (run["core"][-1]["value"], audit["core"]),
             (run["finalBpb"], audit["final_bpb"]["val"]), (run["finalTrainBpb"], audit["final_bpb"]["train"]),
             (p["protocol_hash"], f["protocol"]), (p["model_hash"], f["model"]),
             (p["src_head"], f["git_commit"]), (p["science_hash"], f["cost_source"]),
             (p["source_snapshot_sha256"], f["source"]), (p["tokenizer_sha256"], f["tokenizer"]),
             (p["data_manifest_sha256"], f["data_manifest"])]
    if (not audit["ok"] or audit["problems"] or audit["final_status"] != "matched" or any(a != b for a, b in pairs)
            or run.get("baselineAcceptance")
            or [x["step"] for x in run["val"]] != list(P.validation_steps)
            or [x["step"] for x in run["core"]] != list(P.core_steps)
            or audit["validation_events"] != 189 or audit["core_events"] != 6
            or not math.isclose(c["averageGflopsPerInputToken"] * 1e9, audit["flops_per_token"], rel_tol=1e-12)):
        raise ValueError("historical run differs from its cumulative audit")


def export(directory, audit_path):
    path = REPO / audit_path
    audit = json.loads(path.read_text())
    directory = Path(directory)
    metrics = (directory/"metrics.jsonl").read_bytes()
    if hashlib.sha256(metrics).hexdigest() != audit["metrics_sha256"]:
        raise ValueError("metrics differ from the final audited archive")
    events = [json.loads(line) for line in metrics.splitlines()]
    p = json.loads((directory/"provenance.json").read_text())
    status = json.loads((directory/"run_status.json").read_text())
    identity = p["identity"]
    val = [e for e in events if e["event"] == "validation"]
    if any(e["smoke"] or e["input_tokens"] != P.val_tokens or e["protocol_hash"] != P.fingerprint for e in val):
        raise ValueError("periodic validation differs from the full protocol")
    raw = dict(run_dir=directory.name, val=[(e["step"], e["bpb"]) for e in val],
               core=[(e["step"], dict(core_metric=e["core_metric"], results=e["results"],
                                      centered=e["centered_results"], full_core=e["full_core"])) for e in events if e["event"] == "core"],
               final_bpb=next(e for e in events if e["event"] == "final_bpb"),
               last=next(e for e in reversed(events) if e["event"] == "train"), status=status["budget_status"],
               core_final_mtime=datetime.fromisoformat(audit["verification"]["accepted_at"]).timestamp(),
               provenance=[dict(protocol_hash=identity["protocol_hash"], tokenizer_sha256=identity["tokenizer_sha256"],
                                model_hash=identity["model_hash"], data_manifest_sha256=identity["data_manifest_sha256"],
                                software=dict(science_hash=p["cost_proof"]["source_hash"]), cost=p["calibration"])])
    method = p["model"]["method"]
    name = {"deltanet": "A01 DeltaNet", "gla": "A03 GLA"}[method]
    run, _ = record(raw, method, name, "8 × H200", audit_path, BASE_COMMIT, COHORT)
    run.update(comparisonNote=NOTE, finalTrainBpb=raw["final_bpb"]["train_bpb"],
               auditSha256=hashlib.sha256(path.read_bytes()).hexdigest())
    run["trainingCompute"]["accounting"] = ACCOUNTING
    run["proof"].update(src_head=p["source"]["commit"], reconstruction_base_commit=BASE_COMMIT,
                        source_snapshot_sha256=p["source"]["sha256"], cost_formula_version=ACCOUNTING,
                        cost_anchor_gflops_per_token=p["calibration"]["dense_flops_per_token"]/1e9,
                        backend=identity["backend"], source_manifest="evidence/checks/xiangyue-a01-a03-20261027/source-manifest.json")
    check_audit(run, audit)
    return run


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_dir", type=Path)
    parser.add_argument("--audit", required=True)
    args = parser.parse_args()
    run = export(args.run_dir, args.audit)
    target = RESULTS / "runs" / f"{run['methodId']}.json"
    target.write_text(json.dumps(run, indent=1, ensure_ascii=False) + "\n")
    print(f"wrote {target.relative_to(REPO)}")


if __name__ == "__main__":
    main()
