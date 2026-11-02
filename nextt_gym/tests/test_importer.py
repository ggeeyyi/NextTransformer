from pathlib import Path

import json

from nextt_gym.importer import RESULT_COHORT, config_paths, import_upstream
from nextt_gym.task import SCALES

REPO = Path(__file__).resolve().parents[2]


def test_import_upstream_idempotent(archive, ledger):
    out = import_upstream(archive, REPO)
    configs = config_paths(REPO)
    methods = {json.loads(p.read_text())["model"]["method"] for p in configs}
    records = [json.loads(p.read_text()) for p in sorted((REPO / "results/runs").glob("*.json"))]
    finished = [r for r in records if r["methodId"] in methods and r.get("cohort") == RESULT_COHORT
                and r["status"] == "finished" and r["step"] == SCALES["L"].steps
                and r.get("finalBpb") is not None]
    assert finished, "no published records of the imported cohort"
    assert len(out["attempts"]) == len(configs) and len(out["runs"]) == len(finished)
    again = import_upstream(archive, REPO)
    assert again == dict(attempts=[], refreshed=[], runs=[])
    items = archive.list_attempts(limit=100)["items"]
    assert all(i["status"] == "verified" and i["origin"] == "imported" for i in items)
    titles = {i["title"]: i for i in items}
    assert "gym-untrainable" in titles["ouro"]["tags"] and "upstream" in titles["dense"]["tags"]
    assert titles["kda"]["tags"][0] == "linear" and titles["hc"]["tags"][0] == "residual"
    dense = archive.get_attempt(titles["dense"]["attempt_id"], include=("audit", "files"))
    assert dense["audit"]["scales"]["L"]["relative_error_vs_anchor"] == 0.0
    for title in ("dense", "dsa", "huginn", "deltanet_hybrid", "gdn_hybrid", "gdn_hybrid_swa",
                  "kda_hybrid", "qsa_hybrid", "moba_hybrid"):
        audit = archive.get_attempt(titles[title]["attempt_id"], include=("audit",))["audit"]
        assert set(audit["scales"]) == set(SCALES) and audit["forward_flops_per_token"] > 0
        proof_total = audit["scales"]["L"]["total_training_flops_exact"]
        assert int(proof_total) == sum(n * next(int(s["total"]) for s in audit["stages"] if s["name"] == st)
                                       for st, n in audit["scales"]["L"]["stage_steps"].items())
        if title == "qsa_hybrid":
            assert audit["scales"]["S"]["stage_steps"] == {"burnin": 512}
            assert audit["scales"]["C"]["stage_steps"] == {"burnin": 6, "joint": 506}   # scaled to 512 updates
            assert audit["scales"]["L"]["stage_steps"] == {"burnin": 512, "joint": 46363}
    assert "nextt_gym.upstream_model" in dense["file_contents"]["model.py"]
    board = archive.leaderboard("L", split="official")
    accepted = [e for e in board["entries"] if e["accepted"]]
    expected = sorted((r for r in finished if r.get("complete")), key=lambda r: r["finalBpb"])
    assert [e["title"] for e in accepted] == [r["methodId"] for r in expected]
    assert [e["val_bpb"] for e in accepted] == [r["finalBpb"] for r in expected]
    assert archive.leaderboard("L")["entries"] == []     # dev split unknown for imported runs
    kda_run = archive.get_run(next(e for e in accepted if e["title"] == "kda")["run_id"])
    assert kda_run["gpu_type"].startswith("H20") and kda_run["final"]["core"] > 0.3 and kda_run["origin"] == "imported"
    assert kda_run["consumed"]["flops_exact"] == "230642969256506765625"
    curve = archive.get_curve(kda_run["run_id"], "val_bpb", split="official")
    assert curve["points"][-1]["step"] == 46875
    assert ledger.snapshot()["used_flops"] == 0          # imported runs are not gym spending


def test_import_upstream_skips_results_without_config(archive, ledger, tmp_path):
    # Published historical results can precede the current model port and admission.
    record = json.loads((REPO / "results/runs/dense_full.json").read_text())
    record.update(id="historical_run", methodId="historical_method")
    repo = tmp_path / "repo"
    results = repo / "results/runs"
    results.mkdir(parents=True)
    (results / "historical.json").write_text(json.dumps(record))

    assert import_upstream(archive, repo) == dict(attempts=[], refreshed=[], runs=[])
    assert archive.list_attempts()["items"] == []
    assert archive.list_runs() == []
    assert ledger.snapshot()["used_flops"] == 0


def test_import_upstream_adds_new_scales_to_existing_attempts(archive, ledger):
    import_upstream(archive, REPO)
    attempt = next(i for i in archive.list_attempts(limit=100)["items"] if i["title"] == "kda")
    audit = archive.get_attempt(attempt["attempt_id"], include=("audit",))["audit"]
    older = dict(audit, scales={k: v for k, v in audit["scales"].items() if k != "C"})   # imported before C existed
    archive.record_verification(attempt["attempt_id"], dict(status="passed", levels=[], audit=older))
    out = import_upstream(archive, REPO)
    assert out == dict(attempts=[], refreshed=[attempt["attempt_id"]], runs=[])
    assert archive.get_attempt(attempt["attempt_id"], include=("audit",))["audit"]["scales"] == audit["scales"]
