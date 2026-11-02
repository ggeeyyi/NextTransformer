"""Koala backend against a strict stand-in for the `koala` CLI (_fake_koala.py), its pod entry with a stub `uv`,
and the shell reader."""
import json
import os
from pathlib import Path
import re
import shutil
import stat
import subprocess
import sys
import time

import pytest

from nextt_gym.backends import SSHBackend, reader
from nextt_gym.backends.koala import Koala, KoalaBackend, chain

REPO = Path(__file__).resolve().parents[2]


@pytest.fixture
def fake_koala(tmp_path, monkeypatch):
    """`koala` on PATH is the stand-in of _fake_koala.py; every test fails on a command line the real CLI rejects."""
    bin_dir = tmp_path / "bin"; bin_dir.mkdir()
    path = bin_dir / "koala"
    path.write_text(f"#!{sys.executable}\nimport sys\nsys.path.insert(0, {str(REPO)!r})\n"
                    "from nextt_gym.tests._fake_koala import main\nsys.exit(main())\n")
    path.chmod(path.stat().st_mode | stat.S_IEXEC)
    monkeypatch.setenv("PATH", f"{bin_dir}{os.pathsep}{os.environ['PATH']}")
    monkeypatch.setenv("FAKE_KOALA_STATE", str(tmp_path / "koala"))
    yield tmp_path / "koala"
    violations = tmp_path / "koala" / "violations.jsonl"
    assert not violations.exists(), violations.read_text()


def calls(state):
    return [json.loads(line) for line in (state / "calls.jsonl").read_text().splitlines()]


def backend(tmp_path, **extra):
    return KoalaBackend("koala-test", dict(gpu_type="H20", allowed_world_sizes=[8], workdir=str(tmp_path / "cpfs"),
                                           reader="bash -c {script}", fetch_every_s=0, **extra))


def launch(tmp_path, koala_backend, run_id="run_000007"):
    run_dir = tmp_path / "home" / "runs" / run_id
    run_dir.mkdir(parents=True)
    spec_path = run_dir / "run_spec.json"; spec_path.write_text(json.dumps(dict(run_id=run_id, scale="S")))
    bundle = run_dir / "bundle-abc.tar.gz"; bundle.write_bytes(b"bundle")
    handle = koala_backend.submit(run_dir / "backend", bundle, "0" * 64, spec_path, world_size=8)
    return handle


def wait_final(koala_backend, handle, timeout=60):
    deadline = time.time() + timeout
    while time.time() < deadline:
        status = koala_backend.status(handle)
        if status["state"] in ("succeeded", "failed", "cancelled"): return status
        time.sleep(0.2)
    raise AssertionError(f"driver did not finish: {status}")


def test_reader_mounted_directory(tmp_path):
    (tmp_path / "remote" / "run").mkdir(parents=True)
    for name, text in (("run/a.json", "1"), ("run/val_1_dev.json", "2"), ("run/val_2_dev.json", "3"), ("top.json", "4")):
        (tmp_path / "remote" / name).write_text(text)
    names = reader.fetch("bash -c {script}", tmp_path / "remote", ["run/a.json", "run/val_*.json", "top.json",
                                                                  "run/missing.json", "run/core_*.json"], tmp_path / "out")
    assert sorted(names) == ["a.json", "top.json", "val_1_dev.json", "val_2_dev.json"]
    assert (tmp_path / "out" / "val_2_dev.json").read_text() == "3"
    assert reader.fetch("bash -c {script}", tmp_path / "absent", ["a.json"], tmp_path / "out2") == []
    with pytest.raises(ValueError): reader.read_script(tmp_path, ["../secret"])
    with pytest.raises(ValueError): reader.read_script(tmp_path, ["a.json; rm -rf /"])
    with pytest.raises(RuntimeError): reader.fetch("false {script}", tmp_path / "remote", ["top.json"], tmp_path / "o")


def test_koala_run_segments_and_fetch(tmp_path, fake_koala, monkeypatch):
    monkeypatch.setenv("FAKE_PHASES", "segment_complete,training_complete")
    koala_backend = backend(tmp_path, trainer_args=["--micro-batch", "16"])
    handle = launch(tmp_path, koala_backend)
    status = wait_final(koala_backend, handle)
    assert status["state"] == "succeeded" and status["phase"] == "training_complete" and len(status["segments"]) == 2
    plan = json.loads((Path(handle) / "koala_plan.json").read_text())
    # The remote directory comes from the run spec (not the local `backend` directory) plus a spec hash.
    assert Path(plan["remote_dir"]).name.startswith("run_000007-") and plan["job_prefix"] == "nextt-gym-run-000007"
    assert "--code-dir /local-ssd/nextt-gym-code" in plan["command"] and "-- --micro-batch 16" in plan["command"]
    assert "rm -f" in plan["command"] and "bash /local-ssd/nextt-gym-boot/nextt_gym/backends/koala/run.sh run " in \
        plan["command"]
    publish = [c for c in calls(fake_koala) if c[0] == "code"]
    # Publishes only the previewed bundle, and the job mounts the URI the publish returned.
    previewed = Path(plan["bundle_uri"]).name.split(".")[0]
    assert "--dry-run" in publish[0] and publish[1][publish[1].index("--expect-sha256") + 1] == previewed
    fetched = koala_backend.fetch(handle, tmp_path / "fetched")
    assert json.loads((fetched / "run_status.json").read_text())["phase"] == "training_complete"
    assert {"budget.json", "metrics.jsonl", "remote_status.json", "val_00000512_dev.json"} <= {
        p.name for p in fetched.iterdir()}


def test_koala_cancel_stops_the_job(tmp_path, fake_koala, monkeypatch):
    monkeypatch.setenv("FAKE_PHASES", "hang")
    koala_backend = backend(tmp_path)
    handle = launch(tmp_path, koala_backend)
    deadline = time.time() + 30
    while not koala_backend.status(handle)["segments"] and time.time() < deadline: time.sleep(0.2)
    koala_backend.cancel(handle)
    assert wait_final(koala_backend, handle)["state"] == "cancelled"
    assert any(c[0] == "stop" for c in calls(fake_koala))
    assert len([c for c in calls(fake_koala) if c[0] == "submit"]) == 1


@pytest.mark.parametrize("job_state", ["Queuing", "Updating"])
def test_koala_queued_job_reports_queued(tmp_path, fake_koala, monkeypatch, job_state):
    """Koala admits a job (Queuing), then schedules and pulls the image (Updating); the run is queued until Running."""
    monkeypatch.setenv("FAKE_PHASES", "hang"); monkeypatch.setenv("FAKE_JOB_STATE", job_state)
    koala_backend = backend(tmp_path)
    handle = launch(tmp_path, koala_backend)
    deadline = time.time() + 30
    while time.time() < deadline and not (koala_backend.status(handle)["segments"] or [{}])[-1].get("job_state"):
        time.sleep(0.2)
    assert koala_backend.status(handle)["state"] == "queued"
    koala_backend.cancel(handle)
    assert wait_final(koala_backend, handle)["state"] == "cancelled"


def test_koala_driver_reattaches(tmp_path, fake_koala):
    koala_backend = backend(tmp_path)
    run_dir = tmp_path / "home" / "runs" / "run_000009" / "backend"; run_dir.mkdir(parents=True)
    remote = tmp_path / "cpfs" / "runs" / "run_000009-x"
    (remote / "run").mkdir(parents=True)
    (remote / "run" / "run_status.json").write_text(json.dumps(dict(phase="training_complete", step=512,
                                                                     consumed_flops="7")))
    (run_dir / "koala_plan.json").write_text(json.dumps(dict(
        cluster="h20", bundle_uri="oss://x", code_dir="/data/work/run_codes", command="true", remote_dir=str(remote),
        reader=koala_backend.reader, nodes=1, gpus=8, flags=[], max_segments=8, job_prefix="nextt-gym-run-000009")))
    job = "tester-nextt-gym-run-000009-0000000001-normal-20261102-000001"
    (fake_koala / "jobs").mkdir(parents=True)
    (fake_koala / "jobs" / f"{job}.json").write_text(json.dumps(dict(state="Succeeded")))
    (run_dir / "chain.json").write_text(json.dumps(dict(state="running", segments=[dict(n=1, job=job)])))
    chain(run_dir)
    record = json.loads((run_dir / "chain.json").read_text())
    assert record["state"] == "succeeded" and record["consumed_flops"] == "7" and len(record["segments"]) == 1
    assert not any(c[0] == "submit" for c in calls(fake_koala))


@pytest.mark.parametrize("phases, max_restarts, state, submits", [
    ("crash,training_complete", 2, "succeeded", 2),     # restarts from the checkpoint at 250
    ("crash,crash,training_complete", 2, "failed", 2),  # never twice from the same checkpoint
    ("crash,training_complete", 0, "failed", 1),        # restarts are off by default
    ("crash-early,training_complete", 2, "failed", 1),  # no checkpoint to restart from
])
def test_koala_failed_segment_restarts_from_the_checkpoint(tmp_path, fake_koala, monkeypatch, phases, max_restarts,
                                                           state, submits):
    monkeypatch.setenv("FAKE_PHASES", phases)
    koala_backend = backend(tmp_path, max_restarts=max_restarts)
    handle = launch(tmp_path, koala_backend)
    status = wait_final(koala_backend, handle)
    assert status["state"] == state and len([c for c in calls(fake_koala) if c[0] == "submit"]) == submits
    expected = [dict(segment=1, job_state="Failed", failed_at_step=300, resumed_from=250)] if submits == 2 else []
    assert status["restarts"] == expected
    assert koala_backend.describe().get("max_restarts") == (max_restarts or None)


@pytest.mark.parametrize("ended", ["Failed", "Deleting", "Stopped"])
def test_koala_restarts_after_every_ended_state(tmp_path, fake_koala, monkeypatch, ended):
    """A segment ends Failed (a crash), Deleting (after `koala stop`, as on H20) or Stopped (the CLI's record after a
    stop); the chain restarts from the checkpoint after any of them."""
    monkeypatch.setenv("FAKE_PHASES", "crash,training_complete"); monkeypatch.setenv("FAKE_CRASH_STATE", ended)
    koala_backend = backend(tmp_path, max_restarts=1)
    status = wait_final(koala_backend, launch(tmp_path, koala_backend))
    assert status["state"] == "succeeded"
    assert status["restarts"] == [dict(segment=1, job_state=ended, failed_at_step=300, resumed_from=250)]


def test_koala_backend_follows_the_backend_interface(tmp_path, fake_koala, monkeypatch):
    """describe() and status() carry what every backend reports (base.Backend, base.summarize), plus the segment
    limits of a Koala job; logs() is the last segment's job log."""
    from nextt_gym.backends.base import STATES, summarize
    monkeypatch.setenv("FAKE_PHASES", "segment_complete,training_complete")
    koala_backend = backend(tmp_path)
    described = koala_backend.describe()
    assert {"backend_id", "kind", "gpu_type", "allowed_world_sizes", "status"} <= set(described)
    assert described["max_wall_hours_per_segment"] == 45.5 and described["max_segments"] == 8
    handle = launch(tmp_path, koala_backend)
    assert koala_backend.status(handle)["state"] in STATES
    status = wait_final(koala_backend, handle)
    common = {"state", *summarize(dict(step=0), dict(phase="x"))}
    assert common <= set(status), common - set(status)
    assert status["step"] == 512 and status["gpu_hours"] == 1.0 and status["final"] == {"val_bpb_dev": 0.9}
    assert koala_backend.logs(handle) == ["NEXTT_RUN_STATUS training_complete 512 1000"]


def test_koala_default_reader_runs_through_koala_exec(tmp_path, fake_koala):
    """Without a `reader`, result files come back through `koala exec --cpu -q -- bash -c` on a debug pod."""
    koala_backend = KoalaBackend("koala-test", dict(gpu_type="H20", allowed_world_sizes=[8],
                                                    workdir=str(tmp_path / "cpfs"), fetch_every_s=0))
    handle = launch(tmp_path, koala_backend)
    assert wait_final(koala_backend, handle)["state"] == "succeeded"
    fetched = koala_backend.fetch(handle, tmp_path / "fetched")
    assert json.loads((fetched / "run_status.json").read_text())["phase"] == "training_complete"
    execs = [c for c in calls(fake_koala) if c[0] == "exec"]
    assert execs and all(c[1:6] == ["--cpu", "-q", "--", "bash", "-c"] for c in execs)


def test_koala_unknown_job_has_no_state(fake_koala):
    """`koala get` of a job Koala no longer knows exits 1; the gym reads that as an unknown state, not as an end."""
    gone = "tester-gone-0000000000-normal-20261102-000000"
    assert Koala("h20").state(gone) is None and Koala("h20").stop(gone) == 1 and Koala("h20").log(gone) == []


def test_the_fake_rejects_what_koala_rejects():
    from nextt_gym.tests._fake_koala import Rejected, parse
    for argv in (["submit", "-m", "normal", "-g", "16", "-c", "x"], ["submit", "-m", "normal", "-g", "8"],
                 ["submit", "-m", "debug", "-c", "x"], ["submit", "-m", "normal", "-c", "x", "--s3-log", "s3://x"],
                 ["submit", "-m", "normal", "-c", "x", "--code", "a:b", "--code-bundle", "c:d"],
                 ["submit", "-m", "normal", "-c", "x", "--cluster", "h20"], ["--cluster", "mars", "get", "j"],
                 ["get", "a", "b"], ["exec", "pod"], ["stop"], ["wait"], ["log", "j", "--since", "1h"],
                 ["code", "publish", "--force"]):
        with pytest.raises(Rejected): parse(argv)
    assert parse(["--cluster", "h20", "exec", "--cpu", "-q", "--", "bash", "-c", "true"])[4] == ["bash", "-c", "true"]


@pytest.mark.skipif(not shutil.which("koala"), reason="no koala CLI installed")
def test_fake_grammar_matches_the_installed_cli():
    """The fake accepts exactly the options of the installed CLI, with the same value arity. A Koala release that
    changes them fails here until _fake_koala.GRAMMAR (and the backend, if it uses them) follow."""
    from nextt_gym.tests._fake_koala import GRAMMAR
    for command, grammar in GRAMMAR.items():
        text = subprocess.run(["koala", *command.split(), "--help"], capture_output=True, text=True,
                              timeout=60).stdout
        flags, options = set(), set()
        for line in text.splitlines():
            head = re.match(r"\s+(-\S.*?)(?:\s{2,}|$)", line)
            for part in (head.group(1).split(", ") if head else []):
                name, *metavar = part.split()
                (options if metavar else flags).add(name)
        assert flags - {"-h", "--help"} == set(grammar["flags"]), command
        assert options == set(grammar["options"]), command


def test_only_restartable_backends_accept_max_restarts(tmp_path):
    from nextt_gym.backends import LocalBackend
    with pytest.raises(ValueError, match="max_restarts"): LocalBackend("local", dict(max_restarts=1))
    assert LocalBackend("local", {}).max_restarts == 0


def test_blank_job_state_does_not_end_the_wait(monkeypatch):
    """Koala reports a blank status while it admits a queued job (H20, 2026-10-30); the job has not ended."""
    from nextt_gym.backends.koala.chain import UNKNOWN_POLLS, wait_until_ended
    monkeypatch.setattr(time, "sleep", lambda seconds: None)

    class Stub:
        def __init__(self, states): self.states = list(states); self.waits = 0
        def state(self, job): return self.states.pop(0)
        def wait(self, job): self.waits += 1

    seen = []
    stub = Stub(["Queuing", "", "", "Running", None, "Failed"])
    assert wait_until_ended(stub, "job", seen.append) == "Failed"
    assert seen == ["Queuing", "Running", "Failed"] and stub.waits == 5
    assert wait_until_ended(Stub(["Running", "Stopped"]), "job") == "Stopped"
    assert wait_until_ended(Stub([""] * (UNKNOWN_POLLS + 1) + ["Running"]), "job") is None   # bounded


def test_ssh_remote_directory_uses_the_run_id(tmp_path, monkeypatch):
    commands = []
    ssh = SSHBackend("bare", dict(hosts=["h"], workdir="/w", allowed_world_sizes=[8], check_occupancy=False))
    monkeypatch.setattr(ssh, "_ssh", lambda host, command, **kw: commands.append(command))
    monkeypatch.setattr("nextt_gym.backends.ssh.subprocess.run", lambda *a, **kw: None)
    spec_path = tmp_path / "run_spec.json"; spec_path.write_text(json.dumps(dict(run_id="run_000003")))
    handle = json.loads(ssh.submit(tmp_path / "run_000003" / "backend", tmp_path / "b.tar.gz", "0", spec_path,
                                   world_size=8))
    assert handle["dir"] == "/w/runs/run_000003"


def test_koala_operations_and_command_variants(tmp_path):
    assert Koala("h20").exec_template() == "koala --cluster h20 exec --cpu -q -- bash -c {script}"
    assert Koala("h20").exec_template("cpfs-reader", cpu=False) == "koala --cluster h20 exec -q cpfs-reader -- bash -c {script}"
    assert backend(tmp_path).reader == "bash -c {script}"
    assert KoalaBackend("k", dict(workdir="/w", reader_target="pod-x")).reader.endswith("-q pod-x -- bash -c {script}")
    custom = backend(tmp_path, env_setup="source /venv/bin/activate", python="python")
    command = custom.segment_command("/w/runs/r", "b.tar.gz", "0", "s.json", 8, False, None)
    assert "cd /local-ssd/nextt-gym-boot && source /venv/bin/activate && python -m nextt_gym.remote run " in command
    assert "run.sh" not in command


def test_run_sh_hands_over_to_remote(tmp_path):
    """run.sh builds the environment from the unpacked repository, then execs nextt_gym.remote with its arguments."""
    bin_dir = tmp_path / "bin"; bin_dir.mkdir()
    uv = bin_dir / "uv"
    uv.write_text('#!/bin/sh\necho "uv $* | $(pwd) | $UV_PROJECT_ENVIRONMENT | $NANOCHAT_DTYPE"\n')
    uv.chmod(uv.stat().st_mode | stat.S_IEXEC)
    env = dict(os.environ, PATH=f"{bin_dir}{os.pathsep}{os.environ['PATH']}", NEXTT_VENV=str(tmp_path / "venv"),
               TRITON_CACHE_DIR=str(tmp_path / "triton"))
    out = subprocess.run(["bash", str(REPO / "nextt_gym/backends/koala/run.sh"), "run", "--output", "/o"], env=env,
                         capture_output=True, text=True, check=True).stdout.splitlines()
    assert [line.split(" | ")[0] for line in out] == [
        "uv sync --frozen --extra gpu --extra upstream",
        f"uv pip install --python {tmp_path}/venv/bin/python --no-deps --require-hashes -r requirements/linear.txt",
        "uv run --no-sync python -m nextt_gym.remote run --output /o"]
    assert all(line.split(" | ")[1:] == [str(REPO), str(tmp_path / "venv"), "bfloat16"] for line in out)


def test_remote_replaces_a_stale_code_dir_in_place(tmp_path, monkeypatch):
    """`--code-dir` may sit on another filesystem than OUTPUT (pod-local disk vs CPFS): a stale copy is
    renamed beside the code directory, never across devices."""
    from nextt_gym import bundle, remote
    code, out = tmp_path / "local" / "code", tmp_path / "cpfs" / "out"
    code.mkdir(parents=True); (code / ".bundle_sha256").write_text("old\n")
    monkeypatch.setattr(bundle, "unpack_bundle", lambda path, sha, dest: Path(dest).mkdir())
    monkeypatch.setattr(remote, "preflight", lambda *a: dict(problems=[]))
    monkeypatch.setattr(remote.subprocess, "call", lambda *a, **kw: 0)
    args = remote.argparse.Namespace(output=str(out), code_dir=str(code), sha256="new", bundle="b", nproc=1,
                                     synthetic=True, nnodes=1, run_spec=str(tmp_path / "s.json"), resume="auto",
                                     site_config=None, max_wall_hours=0., model_overrides=None, trainer_args=[])
    assert remote.run(args) == 0
    assert [p.name.split("-")[0] for p in (tmp_path / "local").glob("code.replaced-*")] == ["code.replaced"]
    assert not list(out.glob("code*"))
