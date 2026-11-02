"""Export run directories of the frozen Others branch's eight-rank trainer (scripts/reference_train.py).

That trainer logs the same validation, CORE and train events as nexttransformer.train, but writes one
final_bpb event per split, `status` instead of `final_status` in run_status.json, no core_final.json,
and a flat provenance without the `software` / `cost` sections. This adapter maps those files onto the
standard record and checks it against the run's cumulative audit:

    python -m nexttransformer.export_frozen_8rank RUN_DIR --method mqa --name "MQA" --hardware "H800/H200 ×8" \\
        --audit evidence/checks/others-20261028/h800-mqa-formal.json --commit SHA --note NOTE
"""
import argparse
import json
import os
from pathlib import Path

from nexttransformer.export_run import REPO, record

IDENTITY = ("protocol_hash", "model_hash", "tokenizer_sha256", "data_manifest_sha256", "science_hash")


def read_run(path):
    run = Path(path)
    events = [json.loads(line) for line in (run/"metrics.jsonl").read_text().splitlines() if line.strip()]
    names = ["provenance.json", *sorted(p.name for p in run.glob("resume_*.json"))]
    segments = [json.loads((run/name).read_text()) for name in names]
    if any(s["smoke"] or any(s[k] != segments[0][k] for k in IDENTITY) for s in segments):
        raise ValueError(f"{run.name}: a segment is a smoke run or differs in protocol, model, data, tokenizer or source")
    p = segments[0]
    finals = {e["split"]: e["bpb"] for e in events if e["event"] == "final_bpb"}
    status = run/"run_status.json"
    return dict(
        run_dir=run.name, val=[(e["step"], e["val_bpb"]) for e in events if e["event"] == "validation"],
        core=[(e["step"], dict(core_metric=e["core_metric"], results=e["results"], centered=e["centered_results"],
                               full_core=e["full_core"])) for e in events if e["event"] == "core"],
        final_bpb=dict(val_bpb=finals["val"], train_bpb=finals["train"]),
        last=next(e for e in reversed(events) if e["event"] == "train"),
        status=json.loads(status.read_text())["status"], core_final_mtime=os.path.getmtime(status),
        provenance=[dict(p, software=dict(science_hash=p["science_hash"], native_attention=p["native_attention"],
                                          sparse_backend=p["sparse_backend"]),
                         cost=dict(flops_per_token=p["cost_proof_flops_per_token"],
                                   relative_error=p["cost_proof_flops_per_token"]/p["dense_flops_per_token"]-1))])


def check_audit(run, audit):
    """The record must reproduce the run's cumulative audit (status, final BPB, CORE, compute, fingerprints)."""
    pairs = [(run["reportedAudit"], audit["final_status"]), (run["finalBpb"], audit["final_bpb"]["val"]),
             (run["core"][-1]["value"], audit["core_final"]),
             (run["trainingCompute"]["trainingFlopsEstimate"], audit["theory_flops"]),
             (run["proof"]["model_hash"], audit["fingerprints"]["model_hash"]),
             (run["proof"]["science_hash"], audit["fingerprints"]["science_hash"])]
    if not audit["ok"] or audit["problems"] or any(a != b for a, b in pairs):
        raise ValueError(f"{run['id']}: record differs from its cumulative audit")


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("run_dir", type=Path)
    for flag in ("--method", "--name", "--hardware", "--audit", "--commit"):
        ap.add_argument(flag, required=True)
    ap.add_argument("--note", help="provenance note shown with the run")
    args = ap.parse_args(argv)
    run, path = record(read_run(args.run_dir), args.method, args.name, args.hardware, args.audit, args.commit, note=args.note)
    check_audit(run, json.loads((REPO/args.audit).read_text()))
    if path.exists() and json.loads(path.read_text())["id"] != run["id"]:
        raise SystemExit(f"{path.relative_to(REPO)} holds another run")
    path.write_text(json.dumps(run, indent=1, ensure_ascii=False)+"\n")
    print(f"wrote {path.relative_to(REPO)}: {len(run['val'])} validations, {len(run['core'])} CORE events")


if __name__ == "__main__":
    main()
