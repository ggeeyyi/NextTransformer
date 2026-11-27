"""Research metering (G4) and the disclosure card drafted from it (AR-01, G1)."""
import io
import json
from pathlib import Path
import tarfile

from nextt_gym.cli import main as cli
from nextt_gym.disclosure import draft
from nextt_gym.service import GymService

ROOT = Path(__file__).resolve().parents[2]
TEMPLATES = {n: (ROOT / "nextt_gym/templates" / n).read_text() for n in ("model.py", "config.json")}
OPERATOR = dict(
    name="Attempt B", statusNote="",
    researcher=dict(people=[dict(name="Operator", role="Approved the plan and the compute")],
                    llms=[dict(name="Claude", version="test", role="Researcher")],
                    involvement="L2", involvementNote="The operator approved each stage's plan."),
    novelty=dict(closestPrior=["Block AttnRes"], statement="A test entry."))


class Stub:
    """A backend whose status carries the platform's GPU-hours."""
    def __init__(self, state="running", platform=3.5): self.state, self.platform = state, platform
    def status(self, handle): return dict(state=self.state, step=7, consumed_flops="1000", platform_gpu_hours=self.platform)
    def fetch(self, handle, dest): Path(dest).mkdir(parents=True, exist_ok=True); return dest
    def logs(self, handle, tail=200): return []


def attempt(gym, title, origin="agent"):
    files = {**TEMPLATES, "config.json": TEMPLATES["config.json"] + " " * len(title)}
    return gym.archive.put_attempt(files, title=title, hypothesis="h", origin=origin)


def run(gym, attempt_id, *, steps, platform, status="succeeded", commit=None, backend="stub", synthetic=False):
    run_id = gym.archive.create_run(attempt_id, "M", backend_id=backend, gpu_type="H20", world_size=8)
    gym.ledger.reserve(run_id, 10 ** 18, scale="M")
    gym.archive.update_run(run_id, status=status, progress=dict(step=steps), platform_gpu_hours=platform, gpu_hours=1.0)
    gym.ledger.settle(run_id, 10 ** 15, gpu_hours=1.0, platform_gpu_hours=platform)
    directory = Path(gym.home) / "runs" / run_id; directory.mkdir(parents=True, exist_ok=True)
    (directory / "run_spec.json").write_text(json.dumps(dict(synthetic=synthetic)))
    if commit:
        with tarfile.open(directory / "bundle-x.tar.gz", "w:gz") as tar:
            data = json.dumps(dict(commit=commit, submission_sha256="0" * 64)).encode()
            info = tarfile.TarInfo("BUNDLE.json"); info.size = len(data); tar.addfile(info, io.BytesIO(data))
    return run_id


def test_platform_hours_reach_the_run_and_the_ledger(tmp_path):
    gym = GymService(home=tmp_path / "home", backends_path=tmp_path / "none.toml")
    gym.backends["stub"] = Stub()
    run_id = gym.archive.create_run(attempt(gym, "a"), "S", backend_id="stub", gpu_type="H20", world_size=8)
    gym.ledger.reserve(run_id, 10 ** 18, scale="S", backend_id="stub", gpu_type="H20")
    gym.archive.update_run(run_id, status="running", handle="h")
    gym.poll(force=True)
    assert gym.get_run(run_id)["consumed"]["platform_gpu_hours"] == 3.5
    gym.backends["stub"] = Stub(platform=9.0)                     # a provisional reading above the final one
    gym.poll(force=True)
    gym.backends["stub"] = Stub(state="succeeded", platform=4.25)
    gym.poll(force=True)
    run = gym.get_run(run_id)
    assert run["status"] == "succeeded" and run["consumed"]["platform_gpu_hours"] == 4.25     # the final one wins
    assert run["consumed"]["wall_hours"] >= 0
    budget = gym.get_budget()
    assert budget["metering"]["platform_gpu_hours"] == 4.25 and budget["metering"]["runs_without_platform_hours"] == 0
    assert budget["gpu_hours"] == [dict(backend_id="stub", gpu_type="H20", hours=0.0, platform_hours=4.25)]
    assert budget["metering"]["llm"] == dict(tokens=None, cost_usd=None, note=None)


def test_a_cancelled_run_takes_its_final_metering_later(tmp_path):
    """cancel_run settles at once; the backend measures the stopped job later, and poll() records that."""
    gym = GymService(home=tmp_path / "home", backends_path=tmp_path / "none.toml")
    stub = Stub(platform=0.25)
    stub.cancel = lambda handle: None
    gym.backends["stub"] = stub
    run_id = gym.archive.create_run(attempt(gym, "a"), "S", backend_id="stub", gpu_type="H20", world_size=8)
    gym.ledger.reserve(run_id, 10 ** 18, scale="S", backend_id="stub", gpu_type="H20")
    gym.archive.update_run(run_id, status="running", handle="h")
    gym.cancel_run(run_id, "stop")
    assert gym.get_run(run_id)["extra"]["metering_pending"]
    stub.platform = 8.0; gym.poll(force=True)                     # the backend has not finished yet
    assert gym.get_run(run_id)["consumed"]["platform_gpu_hours"] == 0.25
    stub.state = "cancelled"; gym.poll(force=True)
    run = gym.get_run(run_id)
    assert run["consumed"]["platform_gpu_hours"] == 8.0 and "metering_pending" not in run["extra"]
    assert gym.get_budget()["metering"]["platform_gpu_hours"] == 8.0


def test_disclosure_card_from_the_archive(tmp_path):
    gym = GymService(home=tmp_path / "home", backends_path=tmp_path / "none.toml")
    anchor = attempt(gym, "dense_full", origin="imported")
    a, b, c, d = attempt(gym, "a"), attempt(gym, "b"), attempt(gym, "c"), attempt(gym, "d")
    run(gym, anchor, steps=2000, platform=27.5, commit="e06e4e654cc6cda24fbe9aeff08dafbf7752caec")
    run(gym, a, steps=2000, platform=31.0)
    run(gym, b, steps=2000, platform=40.0)
    run(gym, c, steps=0, platform=0.0, status="cancelled")       # never trained: not a candidate
    run(gym, d, steps=6, platform=0.5, synthetic=True)           # a smoke test: not a candidate, its hours count
    card, metering, problems = draft(gym, b, method="attnres_full_canon_abcd", entry="entry-b", operator=OPERATOR)
    assert problems == []
    research = card["research"]
    assert research["candidatesTrained"] == 2 and metering["designs_trained"] == sorted([a, b])
    assert research["gpuHours"] == 27.5 + 31.0 + 40.0 + 0.5
    assert research["baseCommit"] == "e06e4e654cc6cda24fbe9aeff08dafbf7752caec"
    assert research["selectionSignal"] == ["dev-bpb"] and research["hardware"] == "8 x H20 per run"
    assert "1 of them on synthetic rows" in research["computeNote"]
    assert "LLM tokens and API cost: not metered" in research["computeNote"]
    assert card["name"] == "Attempt B" and card["researcher"]["framework"]["name"] == "NextT-Gym"
    assert card["researcher"]["involvement"] == "L2" and len(metering["runs"]) == 5

    e = attempt(gym, "e")
    run(gym, e, steps=100, platform=None)                         # no platform record: the total is unknown
    card, metering, problems = draft(gym, b, method="attnres_full_canon_abcd", entry="entry-b", operator=OPERATOR)
    assert problems == [] and card["research"]["gpuHours"] is None and metering["measured_gpu_hours"] == 99.0
    assert "no total because 1 runs have none" in card["research"]["computeNote"]

    cli(["--home", str(tmp_path / "home"), "--backends", str(tmp_path / "none.toml"), "llm-usage",
         "--tokens", "123456", "--cost-usd", "7.5", "--note", "operator estimate"])
    (tmp_path / "operator.json").write_text(json.dumps(OPERATOR))
    cli(["--home", str(tmp_path / "home"), "--backends", str(tmp_path / "none.toml"), "disclose", b,
         "--method", "attnres_full_canon_abcd", "--entry", "entry-b", "--operator", str(tmp_path / "operator.json"),
         "--out", str(tmp_path / "card.json"), "--metering", str(tmp_path / "metering.json")])
    written = json.loads((tmp_path / "card.json").read_text())
    assert "123456 tokens, $7.5 (operator estimate)" in written["research"]["computeNote"]
    assert json.loads((tmp_path / "metering.json").read_text())["llm"]["tokens"] == 123456
