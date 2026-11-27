import json

import pytest

from nextt_gym.errors import GymError

MODEL = "import torch\n\ndef build(config):\n    return torch.nn.Linear(2, 2)\n"
CONFIG = json.dumps(dict(model=dict(width=8), model_dim=8))


def files(model=MODEL, config=CONFIG):
    return {"model.py": model, "config.json": config}


def test_put_get_immutable(archive):
    a = archive.put_attempt(files(), title="t", hypothesis="h", tags=["x"])
    assert a == "att_000001"
    got = archive.get_attempt(a, include=("files", "lineage"))
    assert got["status"] == "submitted" and got["files"] == ["config.json", "model.py"]
    assert got["file_contents"]["model.py"] == MODEL and got["lineage"] == dict(ancestors=[], children=[])
    assert (archive.attempt_dir(a) / "model.py").read_text() == MODEL
    with pytest.raises(GymError):
        archive.put_attempt({"model.py": MODEL}, title="t", hypothesis="h")
    with pytest.raises(GymError):
        archive.put_attempt(files(), title="t", hypothesis="h", parent_ids=["att_999999"])


def test_idempotent_put(archive):
    a = archive.put_attempt(files(), title="t", hypothesis="h", idempotency_key="k1")
    assert archive.put_attempt(files(), title="t", hypothesis="h", idempotency_key="k1") == a
    with pytest.raises(GymError) as exc:
        archive.put_attempt(files(model=MODEL + "#x\n"), title="t", hypothesis="h", idempotency_key="k1")
    assert exc.value.code == "IDEMPOTENCY_CONFLICT"
    assert len(archive.list_attempts()["items"]) == 1


def test_lineage_and_listing(archive):
    a = archive.put_attempt(files(), title="a", hypothesis="h", tags=["root"])
    b = archive.put_attempt(files(MODEL + "\nX = 1\n"), title="b", hypothesis="h", parent_ids=[a])
    c = archive.put_attempt(files(MODEL + "\nY = 2\n"), title="c", hypothesis="h", parent_ids=[b])
    assert archive.ancestors(c) == [b, a] and archive.descendants(a) == [b, c]
    assert [i["attempt_id"] for i in archive.list_attempts(descendant_of=a)["items"]] == [c, b]
    assert [i["attempt_id"] for i in archive.list_attempts(ancestor_of=c)["items"]] == [b, a]
    assert [i["attempt_id"] for i in archive.list_attempts(tags=["root"])["items"]] == [a]
    page = archive.list_attempts(limit=2)
    assert len(page["items"]) == 2 and page["next_cursor"] == "2"
    assert len(archive.list_attempts(limit=2, cursor=page["next_cursor"])["items"]) == 1


def test_verification_and_similarity(archive, store):
    a = archive.put_attempt(files(), title="a", hypothesis="h")
    vid = archive.create_verification(a)
    assert archive.get_attempt(a)["status"] == "verifying"
    audit = dict(flops_per_token=4.7e9, scales=dict(L=dict(flops_per_token=4.7e9)), stage_records={"x": 1},
                 parameters=10, within_tolerance=True, unknown_ops=[])
    archive.record_verification(a, dict(status="passed", levels=[], audit=audit, parameters=10), vid)
    got = archive.get_attempt(a, include=("audit",))
    assert got["status"] == "verified" and "stage_records" not in got["audit"] and got["parameters"] == 10
    assert archive.get_verification(vid)["status"] == "passed"
    assert any(e["type"] == "verification.finished" for e in store.events_since(0)["events"])
    sims = archive.similar(files())
    assert sims[0] == (a, 1.0)
    other = archive.similar(files(model="import math\n\ndef build(c):\n    return math.pi * 3 + c['q']\n"))
    assert other[0][1] < 0.5


def test_runs_curves_leaderboard(archive, store):
    imp = archive.put_attempt(files(), title="imp", hypothesis="h", origin="imported")
    anchor = archive.put_attempt(files(MODEL + "#a\nA=1\n"), title="anchor", hypothesis="h", tags=["anchor"])
    mine = archive.put_attempt(files(MODEL + "\nB = 2\n"), title="mine", hypothesis="h")
    for attempt, value in ((imp, 0.90), (anchor, 0.95), (mine, 0.88), (mine, 0.87)):
        run = archive.create_run(attempt, "L", gpu_type="A800", reserved_flops=10 ** 20)
        archive.update_run(run, status="running")
        archive.update_run(run, status="succeeded", consumed_flops=10 ** 20,
                           final=dict(val_bpb_dev=value, val_bpb_official=value + 0.01))
    lb = archive.leaderboard("L")
    assert lb["anchor_val_bpb"] == 0.95
    top = lb["entries"][0]
    assert top["attempt_id"] == mine and top["val_bpb"] == 0.87 and top["seeds"] == 2 and top["rank"] == 1
    assert top["delta_vs_anchor"] == pytest.approx(-0.08) and top["delta_vs_best_imported"] == pytest.approx(-0.03)
    assert [e["attempt_id"] for e in archive.leaderboard("L", include_imported=False)["entries"]] == [mine, anchor]
    assert archive.leaderboard("L", split="official")["entries"][0]["val_bpb"] == pytest.approx(0.88)
    assert archive.get_attempt(mine)["best"]["L"]["val_bpb"] == 0.87
    run = archive.list_runs(attempt_id=mine)[0]
    assert run["started_at"] and run["ended_at"] and run["consumed"]["flops_exact"] == str(10 ** 20)
    archive.add_curve_points(run["run_id"], [dict(metric="val_bpb", split="dev", step=s, input_tokens=s, value=1 / (s + 1))
                                             for s in range(500)])
    curve = archive.get_curve(run["run_id"], "val_bpb", max_points=50)
    assert len(curve["points"]) == 50 and curve["points"][-1]["step"] == 499
    kinds = [e["type"] for e in store.events_since(0)["events"]]
    assert kinds.count("run.finished") == 4 and kinds.count("run.status") == 4


def test_official_results_are_sealed_until_revealed(archive):
    imported = archive.put_attempt(files(), title="imp", hypothesis="h", origin="imported")
    mine = archive.put_attempt(files(MODEL + "\nB = 2\n"), title="mine", hypothesis="h")
    for attempt, origin in ((imported, "imported"), (mine, "agent")):
        run = archive.create_run(attempt, "L", origin=origin)
        archive.update_run(run, status="succeeded", final=dict(val_bpb_dev=0.9, val_bpb_official=0.8))
    sealed = archive.list_runs(attempt_id=mine)[0]
    assert archive.sealed(sealed) and archive.sealed(archive.list_runs(attempt_id=imported)[0])
    titles = lambda **kw: [e["title"] for e in archive.leaderboard("L", split="official", **kw)["entries"]]
    assert titles(seal=True) == [] and sorted(titles()) == ["imp", "mine"]
    assert archive.get_attempt(mine, seal=True)["best_official"] == {}
    s_run = archive.create_run(mine, "S")
    assert not archive.sealed(archive.get_run(s_run))       # S never evaluates the official split
    archive.reveal(mine, "entered through ARENA.md")
    assert titles(seal=True) == ["mine"]
    assert archive.get_attempt(mine, seal=True)["best_official"]["L"]["val_bpb"] == 0.8


def test_notes(archive):
    a = archive.put_attempt(files(), title="a", hypothesis="h")
    assert archive.add_note(a, "learned x", tags_add=["insight"]) == dict(attempt_id=a, notes_count=1)
    got = archive.get_attempt(a)
    assert got["notes"][0]["text"] == "learned x" and "insight" in got["tags"]
    with pytest.raises(GymError):
        archive.add_note(a, "x" * 5000)
