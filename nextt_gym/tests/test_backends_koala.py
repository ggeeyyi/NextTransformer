"""Koala backend against a fake `koala` CLI, its pod entry with a stub `uv`, and the shell reader."""
import json
import os
from pathlib import Path
import stat
import subprocess
import sys
import time

import pytest

from nextt_gym.backends import SSHBackend, reader
from nextt_gym.backends.koala import Koala, KoalaBackend, chain

REPO = Path(__file__).resolve().parents[2]

# A stand-in for the koala CLI: publishes, runs each submitted segment instantly by writing the
# trainer's status files into the remote run directory, and logs every call. A `crash` segment fails
# after update 300 with a complete checkpoint at 250 (`crash-early`: before any checkpoint).
FAKE = r'''#!PYTHON
import json, os, shlex, sys
from pathlib import Path
state = Path(os.environ["FAKE_KOALA_STATE"]); state.mkdir(exist_ok=True)
argv = sys.argv[1:]
if argv[:1] == ["--cluster"]: argv = argv[2:]
with open(state / "calls.jsonl", "a") as log: log.write(json.dumps(argv) + "\n")
cmd = argv[0] if argv else "--help"
if cmd == "code":
    files = len(list(Path(argv[2]).iterdir()))
    print(json.dumps(dict(files=files, bundle_sha256="f" * 64, bundle_uri="oss://fake/" + "f" * 64 + ".tar.gz",
                          published="--dry-run" not in argv)))
elif cmd == "submit":
    n = len(list(state.glob("job-*"))) + 1
    job = argv[argv.index("-j") + 1] + f"-{n}-normal"
    (state / f"job-{n}").write_text(job)
    words = shlex.split(argv[argv.index("-c") + 1].replace("&&", " ").replace(";", " "))
    out = Path(words[words.index("--output") + 1])
    phases = os.environ.get("FAKE_PHASES", "training_complete").split(",")
    phase = phases[min(n, len(phases)) - 1]
    if phase.startswith("crash"):
        (out / "run").mkdir(parents=True, exist_ok=True)
        (out / "run" / "run_status.json").unlink(missing_ok=True)
        (out / "run" / "budget.json").write_text(json.dumps(dict(step=300, consumed_flops="300")))
        if phase == "crash": (out / "run" / "latest.json").write_text(json.dumps(dict(step=250)))
        (state / ("crashed-" + job)).write_text("1")
    elif phase != "hang":
        (out / "run").mkdir(parents=True, exist_ok=True)
        status = dict(phase=phase, step=256 * n, consumed_flops=str(1000 * n), final={"val_bpb_dev": 0.9})
        (out / "run" / "run_status.json").write_text(json.dumps(status))
        (out / "run" / "budget.json").write_text(json.dumps(dict(step=256 * n, consumed_flops=str(1000 * n))))
        (out / "run" / "metrics.jsonl").write_text('{"event": "train", "step": 1, "loss": 10.0}\n')
        (out / "run" / f"val_{256 * n:08d}_dev.json").write_text("{}")
        (out / "remote_status.json").write_text(json.dumps(dict(phase="exited", returncode=0)))
    print(job)
elif cmd == "get":
    stopped = (state / ("stopped-" + argv[1])).exists() or (state / ("crashed-" + argv[1])).exists()
    hang = "hang" in os.environ.get("FAKE_PHASES", "")
    active = os.environ.get("FAKE_JOB_STATE", "Running")
    print(json.dumps(dict(status="Failed" if stopped else active if hang else "Succeeded")))
elif cmd == "wait":
    import time; time.sleep(0.2)
elif cmd == "stop":
    (state / ("stopped-" + argv[-1])).write_text("1")
elif cmd == "log":
    if not (state / ("crashed-" + argv[1])).exists(): print("NEXTT_RUN_STATUS training_complete 512 1000")
'''


@pytest.fixture
def fake_koala(tmp_path, monkeypatch):
    bin_dir = tmp_path / "bin"; bin_dir.mkdir()
    path = bin_dir / "koala"
    path.write_text(FAKE.replace("PYTHON", sys.executable))
    path.chmod(path.stat().st_mode | stat.S_IEXEC)
    monkeypatch.setenv("PATH", f"{bin_dir}{os.pathsep}{os.environ['PATH']}")
    monkeypatch.setenv("FAKE_KOALA_STATE", str(tmp_path / "koala"))
    return tmp_path / "koala"


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
    assert "--dry-run" in publish[0] and publish[1][publish[1].index("--expect-sha256") + 1] == "f" * 64
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


def test_koala_queued_job_reports_queued(tmp_path, fake_koala, monkeypatch):
    monkeypatch.setenv("FAKE_PHASES", "hang"); monkeypatch.setenv("FAKE_JOB_STATE", "Queuing")
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
    (run_dir / "chain.json").write_text(json.dumps(dict(state="running", segments=[dict(n=1, job="old-job")])))
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
