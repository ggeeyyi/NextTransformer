"""The run audit accepts a complete formal run directory and names what is wrong with an incomplete one."""
import json

from nexttransformer.audit_run import SSSL_ANCHOR, audit
from nexttransformer.export_run import RESULTS
from nexttransformer.protocol.spec import Protocol, TOKENIZER_SHA256

P = Protocol()


def write(path, value):
    path.write_text(json.dumps(value))


def run_directory(root):
    """A synthetic finished run with the published Dense Baseline's evaluation results."""
    r = json.loads((RESULTS/"runs/dense_full.json").read_text())
    flops = r["trainingCompute"]["trainingFlopsEstimate"]
    ledger = dict(optimizer_step=P.steps, input_tokens=P.total_tokens, protocol_hash=P.fingerprint,
                  val_steps=list(P.validation_steps), core_steps=list(P.core_steps), theory_flops=flops)
    write(root/"run_status.json", dict(phase="training_complete", step=P.steps, final_status="matched", budget=ledger,
                                       evaluation_forward_flops=1.0))
    write(root/"budget.json", dict(ledger, charged_theory_flops=flops))
    cost = dict(flops_per_token=flops/P.total_tokens, relative_error=flops/P.total_tokens/SSSL_ANCHOR-1,
                dense_flops_per_token=SSSL_ANCHOR)
    write(root/"provenance.json", dict(model_hash="m"*64, protocol_hash=P.fingerprint, data_manifest_sha256="d"*64,
                                       tokenizer_sha256=TOKENIZER_SHA256, parameters=1, model={"layers": 23}, cost=cost,
                                       software=dict(science_hash="s"*64, reference_source_hash="r"*64, torch="2.9.1"),
                                       smoke=False, admission=dict(mode="formal"), world_size=8, micro_batch=8,
                                       accumulation=8, resumed_from=None))
    events = [dict(event="train", step=s, input_tokens=s*P.global_batch_tokens, loss=2.5, grad_norm=1.0,
                   theory_flops=flops*s/P.steps, theory_relative_error=cost["relative_error"])
              for s in range(250, P.steps+1, 250)] + [dict(event="train", step=P.steps, input_tokens=P.total_tokens, loss=2.2,
                                                         grad_norm=1.0, theory_flops=flops, theory_relative_error=0.0)]
    events = sorted({e["step"]: e for e in events}.values(), key=lambda e: e["step"])
    for point in r["val"]:
        events.append(dict(event="validation", step=point["step"], val_bpb=point["value"], input_tokens=P.val_tokens,
                           smoke=False, protocol_hash=P.fingerprint))
        write(root/f"val_{point['step']:08d}.json", dict(step=point["step"], val_bpb=point["value"]))
    for e, c in zip(r["taskEvaluations"], r["core"]):
        events.append(dict(event="core", step=c["step"], core_metric=c["value"], results=e["raw"],
                           centered_results=e["centered"], full_core=True))
        name = "core_final.json" if c["step"] == P.steps else f"core_{c['step']:08d}.json"
        write(root/name, dict(core_metric=c["value"]))
    events.append(dict(event="final_bpb", val_bpb=r["finalBpb"], train_bpb=r["finalBpb"]+.002))
    write(root/"bpb_final.json", dict(val=dict(bpb=r["finalBpb"], input_tokens=P.final_bpb_tokens),
                                      train=dict(bpb=r["finalBpb"]+.002, input_tokens=P.final_bpb_tokens)))
    write(root/"latest.json", dict(step=P.steps, checkpoint=f"checkpoint_{P.steps:08d}"))
    (root/"metrics.jsonl").write_text("\n".join(map(json.dumps, events))+"\n")
    return r


def test_a_complete_run_passes_and_the_report_names_nothing_local(tmp_path):
    r = run_directory(tmp_path)
    report, problems = audit(tmp_path, "dense_full")
    assert problems == [] and report["ok"]
    assert (report["validation_events"], report["core_events"]) == (189, 6)
    assert report["core_final"] == r["core"][-1]["value"] and report["final_bpb"]["val"] == r["finalBpb"]
    text = json.dumps(report)
    assert str(tmp_path) not in text and "host" not in text and "device" not in text


def test_registered_execution_layouts_hold_the_global_batch(tmp_path):
    """Every multiple of eight ranks drives the same eight streams, so the global batch is the invariant."""
    run_directory(tmp_path)
    provenance = json.loads((tmp_path/"provenance.json").read_text())
    for world, micro, accumulation in ((8, 8, 8), (16, 8, 4), (32, 8, 2), (64, 8, 1)):
        write(tmp_path/"provenance.json", dict(provenance, world_size=world, micro_batch=micro,
                                               accumulation=accumulation))
        assert audit(tmp_path, "dense_full")[1] == []
    # An unregistered layout is rejected even when it keeps the global batch.
    write(tmp_path/"provenance.json", dict(provenance, world_size=128, micro_batch=4, accumulation=1))
    assert any("unregistered world size 128" in p for p in audit(tmp_path, "dense_full")[1])


def test_an_incomplete_run_is_reported(tmp_path):
    run_directory(tmp_path)
    (tmp_path/"val_00046875.json").unlink()
    provenance = json.loads((tmp_path/"provenance.json").read_text())
    write(tmp_path/"resume_00020000.json", dict(provenance, admission=dict(mode="research"), resumed_from="ckpt"))
    report, problems = audit(tmp_path, "dense_full")
    assert not report["ok"]
    assert any("val_*.json files" in p for p in problems)
    assert any("research-mode segment" in p for p in problems)
    assert report["segments"][1]["step"] == 20000


def test_frozen_mhc_resume_must_match_both_registered_proofs(tmp_path, monkeypatch):
    import hashlib
    from nexttransformer import audit_run

    archive = tmp_path/"run"
    archive.mkdir()
    run_directory(archive)
    registration_path = "evidence/registrations/mhc-sinkhorn.json"
    reg = json.loads((audit_run.REPO/registration_path).read_text())
    boundary = reg["checkpoint"]
    old, new = (reg[side]["cost"]["total_training_flops"] // P.steps for side in ("from", "to"))
    metrics = archive/"metrics.jsonl"
    events = [json.loads(line) for line in metrics.read_text().splitlines()]
    for e in events:
        if e["event"] == "train":
            e["theory_flops"] = old*min(e["step"], boundary) + new*max(e["step"]-boundary, 0)
    metrics.write_text("\n".join(map(json.dumps, events))+"\n")
    reg["metrics_sha256"] = hashlib.sha256(metrics.read_bytes()).hexdigest()
    (tmp_path/registration_path).parent.mkdir(parents=True)
    write(tmp_path/registration_path, reg)
    (tmp_path/"website").symlink_to(audit_run.REPO/"website", target_is_directory=True)
    monkeypatch.setattr(audit_run, "REPO", tmp_path)

    first = json.loads((archive/"provenance.json").read_text())
    first.update(model={"method": "mhc"}, model_hash=reg["model_hash"], train={"seed": 42}, protocol={"seed": 42})
    for step in (0, boundary, 35000, 45000):
        side = reg["from" if step == 0 else "to"]
        p = dict(first, software=side["software"], cost=side["cost"],
                 resumed_from=f"checkpoint_{step:08d}" if step else None)
        if step == boundary:
            p["source_transition"] = {"changed": {k: [reg["from"]["software"][k], reg["to"]["software"][k]]
                                      for k in ("science_hash", "native_source_sha256", "reference_source_hash")}}
        write(archive/(f"resume_{step:08d}.json" if step else "provenance.json"), p)
    for name in ("run_status.json", "budget.json"):
        f = archive/name
        obj = json.loads(f.read_text())
        ledger = obj.get("budget", obj)
        ledger["theory_flops"] = old*boundary + new*(P.steps-boundary)
        if name == "budget.json":
            ledger["charged_theory_flops"] = ledger["theory_flops"]
        write(f, obj)
    assert audit(archive, "mhc")[1] == []
    path = archive/"resume_00035000.json"
    p = json.loads(path.read_text())
    p["software"]["native_source_sha256"] = "0"*64
    write(path, p)
    assert any("native_source_sha256" in problem for problem in audit(archive, "mhc")[1])
