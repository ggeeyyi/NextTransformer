"""The whole loop through GymService: import, submit, verify, calibrate, quote, launch, wait, settle."""
import json
from pathlib import Path
import shutil
import time

import pytest

from nextt_gym import service as service_module
from nextt_gym.errors import GymError
from nextt_gym.service import GymService

ROOT = Path(__file__).resolve().parents[2]


@pytest.fixture
def gym(tmp_path, monkeypatch, upstream_gym):
    clone, home = upstream_gym
    shutil.copytree(clone, tmp_path / "clone", symlinks=True)
    shutil.copytree(home, tmp_path / "home")
    monkeypatch.setattr(service_module, "REPO_ROOT", tmp_path / "clone")
    (tmp_path / "backends.toml").write_text(
        '[cpu-smoke]\nkind = "local"\ngpu_type = "CPU"\nallowed_world_sizes = [1]\nsynthetic = true\n'
        f'smoke_steps = 4\ntrainer_args = ["--smoke-rows", "2"]\nenv = {{ NANOCHAT_RUNTIME = "{tmp_path}/runtime" }}\n')
    return GymService(home=tmp_path / "home", backends_path=tmp_path / "backends.toml")


def test_agents_see_sealed_official_results_and_l_keeps_the_seed(tmp_path):
    gym = GymService(home=tmp_path / "home", backends_path=tmp_path / "none.toml")
    files = {n: (ROOT / "nextt_gym/templates" / n).read_text() for n in ("model.py", "config.json")}
    attempt = gym.archive.put_attempt(files, title="mine", hypothesis="h")
    cursor = gym.store.last_event_seq()
    run_id = gym.archive.create_run(attempt, "L")
    gym.archive.update_run(run_id, status="succeeded", final=dict(val_bpb_dev=0.9, val_bpb_official=0.8, core=0.3))
    gym.archive.add_curve_points(run_id, [dict(metric="val_bpb", split="official", step=0, value=1.0)])
    run = gym.get_run(run_id)
    assert run["final"] == dict(val_bpb_dev=0.9) and run["sealed"]
    with pytest.raises(GymError) as denied:
        gym.get_run_curve(run_id, split="official")
    assert denied.value.code == "POLICY_DENIED"
    assert gym.get_leaderboard("L", "official")["entries"] == []
    assert gym.list_attempts()["items"][0]["best_official"] == {}
    finished = gym.wait_events(cursor, ["run.finished"], 0)["events"][0]
    assert finished["data"]["final"] == dict(val_bpb_dev=0.9)

    gym.archive.reveal(attempt, "entered through ARENA.md")
    assert gym.get_run(run_id)["final"]["val_bpb_official"] == 0.8 and "sealed" not in gym.get_run(run_id)
    assert [e["attempt_id"] for e in gym.get_leaderboard("L", "official")["entries"]] == [attempt]

    expires = "2999-01-01T00:00:00+00:00"
    gym.store.execute("INSERT INTO quotes(id, seq, created_at, expires_at, payload) VALUES (?, ?, ?, ?, ?)",
                      ("q_000001", 1, expires, expires, json.dumps(dict(attempt_id=attempt, scale="L", feasible=True))))
    with pytest.raises(GymError) as seeded:
        gym.launch_run("q_000001", "seeded", seed_offset=1)
    assert seeded.value.code == "POLICY_DENIED" and "seed" in seeded.value.message


def test_full_loop(gym):
    assert gym.get_leaderboard("L", "official")["entries"] == []       # imported official results are sealed
    assert gym.get_leaderboard("L")["entries_per_split"]["official"] == 0
    board = gym.archive.leaderboard("L", split="official")["entries"]   # the operator's view
    accepted = [e for e in board if e["accepted"]]
    assert accepted and "kda" in [e["title"] for e in accepted]
    assert [e["val_bpb"] for e in accepted] == sorted(e["val_bpb"] for e in accepted)
    with pytest.raises(GymError) as hidden:
        gym.get_leaderboard("S", "official")
    assert hidden.value.code == "POLICY_DENIED"

    files = {n: (ROOT / "nextt_gym/templates" / n).read_text() for n in ("model.py", "config.json")}
    submitted = gym.submit_attempt(files, title="plain", hypothesis="full attention", idempotency_key="a", wait=True)
    assert gym.submit_attempt(files, title="plain", hypothesis="full attention", idempotency_key="a") == submitted
    assert gym.get_verification(submitted["verification_id"])["status"] == "failed"     # +4.1% FLOPs
    with pytest.raises(GymError) as duplicate:
        gym.submit_attempt(files, title="again", hypothesis="same", idempotency_key="b")
    assert duplicate.value.code == "DUPLICATE_ATTEMPT"

    tuned = gym.calibrate(submitted["attempt_id"], ["mlp_width"])
    assert tuned["audit"]["within_tolerance"]
    assert tuned["status"] == "verified" and tuned["verification_id"]
    quote = gym.quote_run(tuned["attempt_id"], "S", "cpu-smoke")
    per_step = int(next(s["total"] for s in tuned["audit"]["stages"] if s["name"] == "joint"))
    assert quote["feasible"] and 4 * per_step <= int(quote["reserve_flops_exact"]) < 4 * per_step * 1.05
    run = gym.launch_run(quote["quote_id"], "launch-1")
    assert gym.launch_run(quote["quote_id"], "launch-1") == run
    assert gym.get_budget()["reserved_flops"] == pytest.approx(quote["reserve_flops"])

    deadline, cursor, finished = time.time() + 600, run["events_cursor"], None
    while time.time() < deadline and not finished:
        events = gym.wait_events(cursor, ["run.finished"], 30)
        cursor, finished = events["next_cursor"], events["events"]
    state = gym.get_run(run["run_id"])
    assert state["status"] == "succeeded", gym.get_run_logs(run["run_id"], 50)
    assert "handle" not in state
    consumed = int(state["consumed"]["flops_exact"])
    assert 4 * per_step <= consumed < 4 * per_step * 1.01
    budget = gym.get_budget()
    assert budget["reserved_flops"] == 0 and budget["used_flops"] == pytest.approx(consumed)
    assert gym.get_leaderboard("S")["entries"][0]["attempt_id"] == tuned["attempt_id"]
    curve = gym.get_run_curve(run["run_id"], "val_bpb")["points"]
    assert curve and all(isinstance(p["value"], float) for p in curve)
    assert curve[0]["input_tokens"] == 0 and curve[-1]["input_tokens"] > 0
    compared = gym.compare_attempts([tuned["attempt_id"], submitted["attempt_id"]], "S")
    assert compared["series"][0]["split"] == "dev" and compared["series"][0]["points"] == curve


def test_imported_official_results_are_sealed(gym):
    run = next(r for r in gym.archive.list_runs(origin="imported") if (r["final"] or {}).get("core"))
    seen = gym.get_run(run["run_id"])
    assert seen["sealed"] and "val_bpb_official" not in (seen["final"] or {}) and "core" not in (seen["final"] or {})
    assert "val_bpb_official" not in (seen["latest"] or {})
    with pytest.raises(GymError) as denied:
        gym.get_run_curve(run["run_id"], split="official")
    assert denied.value.code == "POLICY_DENIED"
    assert all(item["best_official"] == {} for item in gym.list_attempts(limit=100)["items"])
    gym.archive.reveal(run["attempt_id"], "operator check")
    assert gym.get_run(run["run_id"])["final"]["val_bpb_official"] == run["final"]["val_bpb_official"]


def test_agent_documents_carry_no_published_result(tmp_path):
    from nextt_gym import mcp_server
    gym = GymService(home=tmp_path / "home", backends_path=tmp_path / "none.toml")
    values = set()
    for path in (ROOT / "results/runs").glob("*.json"):
        record = json.loads(path.read_text())
        for value in [record.get("finalBpb")] + [p["value"] for p in record.get("core") or []] + \
                [p["value"] for p in record.get("val") or []][-3:]:
            if isinstance(value, float): values |= {f"{value:.4f}", f"{int(value * 1e4) / 1e4:.4f}"}
    assert len(values) > 40
    texts = {doc: mcp_server.upstream_doc(doc) for doc in mcp_server.UPSTREAM_DOCS}
    texts.update(spec=mcp_server.spec_text(), failures=gym.knowledge_failures())
    import re
    leaks = {doc: sorted(v for v in values if re.search(r"(?<!\d)" + re.escape(v), text)) for doc, text in texts.items()}
    assert not any(leaks.values()), leaks


def test_backend_restarts_reach_the_run_and_the_events(tmp_path):
    gym = GymService(home=tmp_path / "home", backends_path=tmp_path / "none.toml")
    restart = dict(segment=1, job_state="Stopped", failed_at_step=330, resumed_from=250)

    class Stub:
        def status(self, handle): return dict(state="running", step=330, restarts=[restart])
        def fetch(self, handle, dest): return dest
    gym.backends["stub"] = Stub()
    files = {n: (ROOT / "nextt_gym/templates" / n).read_text() for n in ("model.py", "config.json")}
    attempt = gym.archive.put_attempt(files, title="mine", hypothesis="h")
    run_id = gym.archive.create_run(attempt, "S", backend_id="stub")
    gym.ledger.reserve(run_id, 10 ** 18, scale="S")
    gym.archive.update_run(run_id, status="running", handle="h")
    cursor = gym.store.last_event_seq()
    for _ in range(2): gym._refresh(gym.archive.get_run(run_id))
    assert gym.get_run(run_id)["extra"]["restarts"] == [restart]
    events = gym.store.events_since(cursor, types=["run.restarted"])["events"]
    assert [e["data"] for e in events] == [restart]


def test_each_dev_evaluation_is_announced_once(tmp_path):
    gym = GymService(home=tmp_path / "home", backends_path=tmp_path / "none.toml")
    train = lambda step: dict(event="train", step=step, input_tokens=step * 1048576, lm_loss=3.0)
    dev = lambda step: dict(event="validation", split="dev", step=step, val_bpb=1.0 + 1 / (step + 1))
    head = [dev(0), train(10), train(128), dev(128), train(130)]
    # Polls see the metrics file grow, rewind to the checkpoint at 250 after a restart, and replay.
    polls = [head[:2], head, head + [train(256)], head + [train(256), dev(256), train(300)], head,
             head + [train(256), dev(256), train(300), train(512)], head + [train(256), dev(256), train(512), dev(512)]]
    shown = []

    class Stub:
        def status(self, handle): return dict(state="running")
        def fetch(self, handle, dest):
            Path(dest).mkdir(parents=True, exist_ok=True)
            (Path(dest) / "metrics.jsonl").write_text("".join(json.dumps(e) + "\n" for e in shown))
            return dest
    gym.backends["stub"] = Stub()
    files = {n: (ROOT / "nextt_gym/templates" / n).read_text() for n in ("model.py", "config.json")}
    run_id = gym.archive.create_run(gym.archive.put_attempt(files, title="mine", hypothesis="h"), "S", backend_id="stub")
    gym.ledger.reserve(run_id, 10 ** 18, scale="S")
    gym.archive.update_run(run_id, status="running", handle="h")
    cursor = gym.store.last_event_seq()
    for events in polls + polls[-1:]:
        shown[:] = events
        gym._refresh(gym.archive.get_run(run_id))
    announced = gym.store.events_since(cursor, types=["run.eval"])["events"]
    assert [(e["data"]["step"], e["data"]["val_bpb_dev"]) for e in announced] == [(s, dev(s)["val_bpb"])
                                                                                 for s in (0, 128, 256, 512)]
