"""Every published record under results/ is valid."""
import json
import shutil

import pytest

from nexttransformer.export_run import P, RESULTS, check, record


def raw_from_record(path):
    r = json.loads(path.read_text()); p = r["proof"]
    core = [(e["step"], dict(core_metric=c["value"], results=e["raw"], centered=e["centered"], full_core=True))
            for e, c in zip(r["taskEvaluations"], r["core"])]
    return dict(run_dir=r["id"], val=[(x["step"], x["value"]) for x in r["val"]], core=core, final_bpb=dict(val_bpb=r["finalBpb"]),
                last=dict(step=P.steps, input_tokens=P.total_tokens, theory_flops=r["trainingCompute"]["trainingFlopsEstimate"]),
                status="matched", core_final_mtime=r["lastRecordedAt"],
                provenance=[dict(tokenizer_sha256=p["tokenizer_sha256"], protocol_hash=p["protocol_hash"], model_hash=p["model_hash"],
                                 data_manifest_sha256=p["data_manifest_sha256"], software=dict(science_hash=p["science_hash"]),
                                 cost=dict(flops_per_token=p["cost_proof_gflops_per_token"]*1e9,
                                           relative_error=p["cost_proof_vs_dense_pct"]/100))])


def test_results_records_are_valid():
    assert check() == []


def test_export_records_a_provenance_note_only_when_given():
    raw = raw_from_record(RESULTS/"runs/dense_full.json")
    plain, _ = record(raw, "gqa", "GQA", "H800 ×8", commit="a"*40)
    noted, path = record(raw, "gqa", "GQA", "H800 ×8", commit="a"*40, note="Trained from a frozen branch.")
    assert "comparisonNote" not in plain and noted["comparisonNote"] == "Trained from a frozen branch."
    assert path == RESULTS/"runs/gqa.json" and noted["proof"]["src_head"] == "a"*40


def test_export_refuses_a_run_with_a_research_mode_segment():
    raw = raw_from_record(RESULTS/"runs/dense_full.json")
    first = raw["provenance"][0]
    formal = dict(raw, provenance=[dict(first, admission=dict(mode="formal", findings=[]))])
    assert record(formal, "gqa", "GQA", "H800 ×8", commit="a"*40)[0]["complete"]
    research = dict(raw, provenance=[*formal["provenance"], dict(first, admission=dict(mode="research", findings=[]))])
    with pytest.raises(ValueError, match="research mode"):
        record(research, "gqa", "GQA", "H800 ×8", commit="a"*40)


def test_runs_share_one_flat_directory(tmp_path):
    root = tmp_path/"results"
    shutil.copytree(RESULTS, root)
    (root/"runs/early").mkdir()
    assert any("results/runs/ is flat" in p for p in check(root))


def test_baseline_acceptance_must_match_one_complete_dense_run(tmp_path):
    root = tmp_path/"results"
    shutil.copytree(RESULTS, root)
    dense = next(p for p in root.glob("runs/*.json") if json.loads(p.read_text()).get("baselineAcceptance"))
    record = json.loads(dense.read_text())
    acceptance = dict(record["baselineAcceptance"])
    record["baselineAcceptance"]["sourceCommit"] = "0"*40
    dense.write_text(json.dumps(record))
    other = root/"runs/gdn.json"
    gdn = json.loads(other.read_text())
    gdn["baselineAcceptance"] = acceptance
    other.write_text(json.dumps(gdn))
    problems = check(root)
    assert any("differs from the run's provenance" in p for p in problems)
    assert any("needs a complete, matched dense_full" in p for p in problems)
    assert any("more than one accepted Dense baseline" in p for p in problems)


def test_frozen_eight_rank_run_directory_reproduces_its_record_and_audit(tmp_path):
    import pytest
    from nexttransformer.export_frozen_8rank import check_audit, read_run
    from nexttransformer.export_run import REPO
    r = json.loads((RESULTS/"runs/mqa.json").read_text()); p = r["proof"]
    events = [dict(event="validation", step=x["step"], val_bpb=x["value"]) for x in r["val"]]
    events += [dict(event="core", step=c["step"], core_metric=c["value"], results=e["raw"], centered_results=e["centered"], full_core=True)
               for e, c in zip(r["taskEvaluations"], r["core"])]
    events += [dict(event="final_bpb", split="train", bpb=r["finalBpb"]+.002), dict(event="final_bpb", split="val", bpb=r["finalBpb"]),
               dict(event="train", step=P.steps, input_tokens=P.total_tokens, theory_flops=r["trainingCompute"]["trainingFlopsEstimate"])]
    (tmp_path/"metrics.jsonl").write_text("\n".join(map(json.dumps, events))+"\n")
    flat = dict(protocol_hash=p["protocol_hash"], model_hash=p["model_hash"], tokenizer_sha256=p["tokenizer_sha256"],
                data_manifest_sha256=p["data_manifest_sha256"], science_hash=p["science_hash"], smoke=False,
                native_attention=p["native_attention"], sparse_backend=None, dense_flops_per_token=4672398325.44785,
                cost_proof_flops_per_token=p["cost_proof_gflops_per_token"]*1e9)
    for name in ("provenance.json", "resume_00005000.json"):
        (tmp_path/name).write_text(json.dumps(flat))
    (tmp_path/"run_status.json").write_text(json.dumps(dict(status="matched")))
    run, path = record(read_run(tmp_path), "mqa", "MQA", r["hardware"], commit=p["src_head"])
    assert path == RESULTS/"runs/mqa.json"
    assert (run["val"], run["core"], run["taskEvaluations"], run["finalBpb"]) == (r["val"], r["core"], r["taskEvaluations"], r["finalBpb"])
    assert {k: v for k, v in run["proof"].items() if not k.startswith("cost_")} == {k: v for k, v in p.items() if not k.startswith("cost_")}
    assert run["proof"]["cost_proof_vs_dense_pct"] == pytest.approx(p["cost_proof_vs_dense_pct"])
    check_audit(run, json.loads((REPO/"evidence/checks/others-20261028/h800-mqa-formal.json").read_text()))
    (tmp_path/"resume_00005000.json").write_text(json.dumps(dict(flat, model_hash="0"*64)))
    with pytest.raises(ValueError, match="segment"):
        read_run(tmp_path)
