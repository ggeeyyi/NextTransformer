"""Local tests of koala_gym.py with a fake Koala CLI (CPU, synthetic fixtures).

Run with the Gym environment:
  PYTHONPATH=gym/src:. gym/.venv/bin/python -m pytest -q gym/koala/tests
Test directories live outside /tmp: the worker sandbox mounts a fresh /tmp over anything there.
"""

import base64
import io
import json
import os
import runpy
import shutil
import socket
import subprocess
import sys
import threading
import time
import uuid
from pathlib import Path

import pytest

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
sys.path[:0] = [str(HERE.parent), str(ROOT / "gym" / "src"), str(ROOT)]

import koala_gym  # noqa: E402
from nexttransformer_gym.backend import NativeBackend, Scheduler  # noqa: E402
from nexttransformer_gym.common import GymError  # noqa: E402
from nexttransformer_gym.evidence import snapshot, verify_receipt  # noqa: E402
from nexttransformer_gym.final import get_final_run, launch_final_run  # noqa: E402
from nexttransformer_gym.service import Service  # noqa: E402

BASE = Path(
    os.environ.get("KOALA_GYM_TEST_ROOT", Path.home() / ".local/share/koala-gym-tests")
)


def candidate_bytes():
    blob = io.BytesIO()
    runpy.run_path(str(ROOT / "gym/examples/make_candidate.py"))["package"](
        blob, "square"
    )
    return blob.getvalue()


def fake_s5cmd(base):
    """A wrapper script that runs fake_s5cmd.py on BASE/s3."""
    script = base / "s5cmd"
    script.write_text(
        f'#!/bin/sh\nexec {sys.executable} {HERE / "fake_s5cmd.py"} {base / "s3"} "$@"\n'
    )
    script.chmod(0o755)
    return str(script)


@pytest.fixture(params=["cpfs"])
def gym(request):
    base = BASE / uuid.uuid4().hex[:10]
    home = base / "home"
    home.mkdir(parents=True)
    config = {
        "home": str(home),
        "koala": str(HERE / "fake_koala.py"),
        "staging_dir": str(base / "staging"),
        "remote_root": str(base / "cpfs"),
        "code_project": "kgym-test",
        "job_prefix": "kgym-test",
        "code_mount": "/koala-test/run_codes",
        "pod_work_dir": str(base / "local-ssd"),
        "pod_python": sys.executable,
        "poll_seconds": 0.3,
        "relay_margin_seconds": 10,
        "gpu_memory_total_bytes": 8 << 30,
        "gpu_memory_limit_bytes": 4 << 30,
        "runtime": koala_gym.runtime_of(sys.executable),
    }
    if request.param == "s3":
        # A cluster without shared storage: job files in an object store, a pod-local mirror.
        config.update(
            remote_root="s3://bucket/harbor-gym",
            pod_remote_dir=str(base / "local-ssd" / "harbor-gym"),
            s5cmd=fake_s5cmd(base),
            final={"nodes": 1, "gpus_per_node": 1},
        )
    path = base / "koala.json"
    path.write_text(json.dumps(config))
    profiles = json.loads((ROOT / "gym/examples/profiles.fixture.json").read_text())
    for profile in profiles.values():
        profile["timeout_sec"] = 600
    backend = NativeBackend(
        profiles, base / "state", ROOT, execution_config=cpu_settings(path)
    )
    service = Service(base / "state" / "gym.sqlite3", backend)
    credentials = service.create_session(10**12, "koala-test")

    def call(operation, **params):
        return service.call(credentials["agent_token"], operation, params)

    candidate = call(
        "submit_candidate",
        package_base64=base64.b64encode(candidate_bytes()).decode(),
        idempotency_key="first",
    )
    yield dict(
        base=base,
        home=home,
        path=path,
        config=config,
        service=service,
        call=call,
        candidate=candidate,
        credentials=credentials,
    )
    for job in (home / ".fake-koala" / "jobs").glob("*"):
        subprocess.run(
            [str(HERE / "fake_koala.py"), "stop", "-y", job.name],
            env=dict(os.environ, HOME=str(home)),
        )
    shutil.rmtree(base, ignore_errors=True)


def cpu_settings(path, **values):
    """The adapter's execution configuration with no cards: the fixture's CPU research profiles."""
    settings = koala_gym.execution_settings(
        dict(koala_gym.load_config(path), **values), path
    )
    settings["executor"]["devices"] = []
    return settings


def fake_settings(gym, **values):
    root = gym["home"] / ".fake-koala"
    root.mkdir(exist_ok=True)
    (root / "settings.json").write_text(json.dumps(values))


def run(gym, **params):
    call = gym["call"]
    quote = call(
        "quote_experiment",
        candidate_id=gym["candidate"]["id"],
        profile_id="research-fixture",
        **params,
    )
    job = call("run_experiment", quote_id=quote["id"], idempotency_key=quote["id"])
    Scheduler(gym["service"]).work_once()
    return quote, call("get_job", job_id=job["id"])


def diagnostics(gym, job):
    return (
        gym["service"].backend.state_dir / (job["id"] + ".diagnostic.log")
    ).read_text()


def journal(gym):
    path = gym["service"].backend.state_dir / "koala-jobs.jsonl"
    return (
        [json.loads(line) for line in path.read_text().splitlines()]
        if path.exists()
        else []
    )


def test_research_run_through_koala_charges_the_quoted_bound(gym):
    before = gym["call"]("get_budget")
    quote, job = run(gym, target_update=2)
    assert job["state"] == "succeeded", diagnostics(gym, job)
    result = gym["call"]("get_result", job_id=job["id"])
    assert result["step"] == 2 and result["dev_bpb"] > 0 and result["checkpoint_id"]
    after = gym["call"]("get_budget")
    assert int(before["available"]) - int(after["available"]) == int(quote["bound"])
    assert after["reserved"] == "0"
    pointer = json.loads(
        gym["service"].backend.checkpoint_path(result["checkpoint_id"]).read_text()
    )
    assert (
        pointer["schema"] == "koala_gym.checkpoint.v1"
        and Path(pointer["remote_path"]).is_file()
    )
    assert koala_gym.sha256(pointer["remote_path"]) == pointer["sha256"]
    entries = journal(gym)
    assert [e["event"] for e in entries] == ["submitted", "running", "ended"]
    # Pod time and GPU-hours from the pod's own clock, for disclosure.
    ended = entries[-1]
    assert (
        ended["pod_started"] == entries[1]["pod_started"] and ended["pod_seconds"] > 0
    )
    assert abs(ended["gpu_hours"] - ended["pod_seconds"] / 3600) <= 1e-4


@pytest.mark.parametrize("gym", ["cpfs", "s3"], indirect=True)
def test_continuation_from_a_koala_checkpoint_matches_an_uninterrupted_run(gym):
    _, first = run(gym, target_update=1)
    assert first["state"] == "succeeded", diagnostics(gym, first)
    checkpoint = gym["call"]("get_result", job_id=first["id"])["checkpoint_id"]
    _, resumed = run(gym, target_update=2, checkpoint_id=checkpoint)
    assert resumed["state"] == "succeeded", diagnostics(gym, resumed)
    _, whole = run(gym, target_update=2)
    a = gym["call"]("get_result", job_id=resumed["id"])
    b = gym["call"]("get_result", job_id=whole["id"])
    assert a["dev_bpb"] == b["dev_bpb"]
    if gym["config"]["remote_root"].startswith("s3://"):
        # The checkpoint, result and logs live in the object store; the pods' local mirrors are disposable.
        pointer = json.loads(
            gym["service"].backend.checkpoint_path(checkpoint).read_text()
        )
        stored = gym["base"] / "s3" / pointer["remote_path"][len("s3://") :]
        assert (
            pointer["remote_path"].startswith("s3://bucket/harbor-gym/")
            and koala_gym.sha256(stored) == pointer["sha256"]
        )
        assert (stored.parent / "result.json").is_file() and list(
            stored.parent.glob("started-*")
        )
        assert all(
            e["remote"].startswith("s3://")
            for e in journal(gym)
            if e["event"] == "submitted"
        )


def test_an_object_store_research_job_stages_only_its_data_and_never_the_validation_shard(
    tmp_path,
):
    bucket = tmp_path / "s3" / "bucket"
    for name in (
        "shard_00000.parquet",
        "shard_00001.parquet",
        "shard_06541.parquet",
        "shard_06542.parquet",
        ".downloads/rev/manifest.json",
    ):
        (bucket / "climbmix" / name).parent.mkdir(parents=True, exist_ok=True)
        (bucket / "climbmix" / name).write_text(name)
    for name in (
        "tokenizer/tokenizer.pkl",
        "tokenizer/token_bytes.pt",
        "native_manifest.json",
    ):
        (bucket / "resources" / name).parent.mkdir(parents=True, exist_ok=True)
        (bucket / "resources" / name).write_text(name)
    data_dir, resources = tmp_path / "pod" / "data", tmp_path / "pod" / "resources"
    payload = {
        "s5cmd": fake_s5cmd(tmp_path),
        "data_stage": [
            [str(data_dir), "s3://bucket/climbmix"],
            [str(resources), "s3://bucket/resources"],
        ],
    }
    data = {
        "kind": "climbmix",
        "train_paths": [str(data_dir / "shard_00000.parquet")],
        "dev_path": str(data_dir / "shard_06541.parquet"),
        "tokenizer": str(resources / "tokenizer"),
        "token_bytes": str(resources / "tokenizer" / "token_bytes.pt"),
    }
    request = {"quote": {"operation": "train"}, "profile": {"data": data}}
    koala_gym.stage_inputs(payload, request)
    staged = sorted(
        str(p.relative_to(tmp_path / "pod"))
        for p in (tmp_path / "pod").rglob("*")
        if p.is_file()
    )
    assert staged == [
        "data/shard_00000.parquet",
        "data/shard_06541.parquet",
        "resources/tokenizer/token_bytes.pt",
        "resources/tokenizer/tokenizer.pkl",
    ]
    assert all(p.stat().st_mode & 0o004 for p in (tmp_path / "pod").rglob("*"))
    for reserved in (
        dict(data, dev_path=str(data_dir / "shard_06542.parquet")),
        dict(data, train_paths=[str(data_dir / "shard_06542.parquet")]),
    ):
        with pytest.raises(koala_gym.PodError) as error:
            koala_gym.stage_inputs(
                payload,
                {"quote": {"operation": "train"}, "profile": {"data": reserved}},
            )
        assert error.value.code == "denied_access"
    assert not (data_dir / "shard_06542.parquet").exists()
    with pytest.raises(koala_gym.PodError):  # data outside every staged prefix
        koala_gym.stage_inputs(
            payload,
            {
                "quote": {"operation": "train"},
                "profile": {
                    "data": dict(data, dev_path="/elsewhere/shard_06541.parquet")
                },
            },
        )
    # A trusted final run mirrors whole prefixes, the validation shard and hidden manifests included.
    koala_gym.stage_inputs(payload, {"quote": {"operation": "official"}, "profile": {}})
    assert (data_dir / "shard_06542.parquet").is_file() and (
        data_dir / ".downloads/rev/manifest.json"
    ).is_file()
    assert (resources / "native_manifest.json").is_file()


def test_an_object_store_remote_root_allows_only_single_node_final_runs(tmp_path):
    config = tmp_path / "koala.json"
    value = {
        "home": str(tmp_path),
        "staging_dir": str(tmp_path / "staging"),
        "remote_root": "s3://bucket/gym",
        "code_project": "p",
        "job_prefix": "p",
        "gpu_memory_total_bytes": 8 << 30,
        "gpu_memory_limit_bytes": 4 << 30,
        "final": {"nodes": 4, "gpus_per_node": 8},
    }
    config.write_text(json.dumps(value))
    with pytest.raises(SystemExit, match="multi-node final run"):
        koala_gym.load_config(config)
    config.write_text(json.dumps(dict(value, final={"nodes": 1, "gpus_per_node": 8})))
    assert (
        koala_gym.local_root(koala_gym.load_config(config)) == "/local-ssd/harbor-gym"
    )


def test_checks_finalize_and_final_run_through_koala(gym):
    call, service, credentials = gym["call"], gym["service"], gym["credentials"]
    job = call(
        "validate_candidate",
        candidate_id=gym["candidate"]["id"],
        profile_id="final-fixture",
        idempotency_key="checks",
    )
    Scheduler(service).work_once()
    state = call("get_job", job_id=job["id"])
    assert state["state"] == "succeeded", diagnostics(gym, state)
    assert call("get_result", job_id=job["id"])["admitted"]
    submission = call(
        "finalize",
        candidate_id=gym["candidate"]["id"],
        profile_id="final-fixture",
        idempotency_key="final",
    )
    verifier = {
        "session_id": credentials["session_id"],
        "token": credentials["verifier_token"],
    }
    receipt = {
        "session_id": credentials["session_id"],
        "submission_id": submission["id"],
        "candidate_hash": gym["candidate"]["hash"],
    }
    assert verify_receipt(receipt, snapshot(service, verifier), verifier)["valid"]
    final = launch_final_run(
        service, credentials["session_id"], submission["id"], 10**12, "dispatch"
    )
    assert Scheduler(service).work_once()
    result = get_final_run(service, credentials["session_id"], final["id"])
    assert (
        result["state"] == "succeeded"
        and result["result"]["official_compatible"] is False
    )


def test_cancel_reaches_a_queued_koala_job_and_charges_nothing(gym):
    fake_settings(gym, queue_seconds=60)
    call = gym["call"]
    before = call("get_budget")
    quote = call(
        "quote_experiment",
        candidate_id=gym["candidate"]["id"],
        profile_id="research-fixture",
        target_update=2,
    )
    job = call("run_experiment", quote_id=quote["id"], idempotency_key="queued")
    worker = threading.Thread(target=Scheduler(gym["service"]).work_once)
    worker.start()
    deadline = time.time() + 60
    while not any(e["event"] == "submitted" for e in journal(gym)):
        assert time.time() < deadline
        time.sleep(0.2)
    call("cancel_job", job_id=job["id"], idempotency_key="stop")
    worker.join(60)
    assert call("get_job", job_id=job["id"])["state"] == "cancelled"
    after = call("get_budget")
    assert after["available"] == before["available"] and after["reserved"] == "0"
    assert journal(gym)[-1]["reason"] == "cancel_job"


def test_a_runtime_that_differs_from_the_probe_fails_before_work_without_a_charge(gym):
    config = dict(gym["config"])
    config["runtime"] = dict(config["runtime"], python=[3, 0, 0])
    gym["path"].write_text(json.dumps(config))
    before = gym["call"]("get_budget")
    _, job = run(gym, target_update=1)
    assert job["state"] == "failed" and job["error_code"] == "resource_unavailable"
    assert gym["call"]("get_budget")["available"] == before["available"]
    assert "differs from the recorded" in diagnostics(gym, job)


def test_a_failed_submission_charges_nothing(gym):
    fake_settings(gym, fail_submit=True)
    before = gym["call"]("get_budget")
    _, job = run(gym, target_update=1)
    assert job["state"] == "failed" and job["error_code"] == "resource_unavailable"
    assert gym["call"]("get_budget")["available"] == before["available"]


def test_a_full_koala_quota_is_waited_out_and_then_the_job_runs(gym):
    # Two refusals and one revoked (non-atomic quota check) submission, then a normal submission.
    fake_settings(gym, quota_full_submits=2, quota_race_submits=1)
    config = dict(gym["config"], quota_wait_seconds=0.2)
    gym["path"].write_text(json.dumps(config))
    before = gym["call"]("get_budget")
    quote, job = run(gym, target_update=1)
    assert job["state"] == "succeeded"
    assert gym["call"]("get_budget")["spent"] == str(
        int(before["spent"]) + int(quote["bound"])
    )
    log = diagnostics(gym, job)
    assert (
        log.count("Koala quota is full; retrying") == 3
        and "submitted after waiting" in log
    )
    revoked = (gym["home"] / ".fake-koala" / "stopped-unknown.txt").read_text().split()
    assert len(revoked) == 1 and "-normal-" in revoked[0]
    assert [e["event"] for e in journal(gym)][:1] == ["submitted"]


def test_a_quota_full_until_the_profile_timeout_fails_without_a_charge(gym):
    fake_settings(gym, quota_full_submits=10**6)
    # The relay deadline is the profile timeout minus relay_margin_seconds (at least 60 s).
    config = dict(gym["config"], quota_wait_seconds=30, relay_margin_seconds=10**6)
    gym["path"].write_text(json.dumps(config))
    before = gym["call"]("get_budget")
    _, job = run(gym, target_update=1)
    assert job["state"] == "failed" and job["error_code"] == "resource_unavailable"
    assert gym["call"]("get_budget")["available"] == before["available"]
    assert "stayed full" in diagnostics(gym, job) and not journal(gym)


def test_a_cancel_while_waiting_for_quota_charges_nothing(gym):
    fake_settings(gym, quota_full_submits=10**6)
    config = dict(gym["config"], quota_wait_seconds=0.2)
    gym["path"].write_text(json.dumps(config))
    call = gym["call"]
    before = call("get_budget")
    quote = call(
        "quote_experiment",
        candidate_id=gym["candidate"]["id"],
        profile_id="research-fixture",
        target_update=1,
    )
    job = call("run_experiment", quote_id=quote["id"], idempotency_key="quota-wait")
    worker = threading.Thread(target=Scheduler(gym["service"]).work_once)
    worker.start()
    deadline = time.time() + 60
    # The Gym writes the diagnostic log when the worker exits; the fake's refusal count shows the wait.
    refusals = gym["home"] / ".fake-koala" / "quota_full_submits.count"
    while not refusals.exists():
        assert time.time() < deadline
        time.sleep(0.2)
    call("cancel_job", job_id=job["id"], idempotency_key="stop")
    worker.join(60)
    assert call("get_job", job_id=job["id"])["state"] == "cancelled"
    after = call("get_budget")
    assert after["available"] == before["available"] and after["reserved"] == "0"
    assert "cancelled while waiting" in diagnostics(gym, job) and not journal(gym)


def test_a_job_lost_after_its_worker_started_keeps_the_reservation(gym):
    call = gym["call"]
    quote = call(
        "quote_experiment",
        candidate_id=gym["candidate"]["id"],
        profile_id="research-fixture",
        target_update=8,
    )
    job = call("run_experiment", quote_id=quote["id"], idempotency_key="lost")
    worker = threading.Thread(target=Scheduler(gym["service"]).work_once)
    worker.start()
    jobs = gym["home"] / ".fake-koala" / "jobs"
    deadline = time.time() + 120
    while True:
        logs = list(jobs.glob("*/node-0/log")) if jobs.exists() else []
        if logs and '"kind":"ready"' in logs[0].read_text():
            break
        assert time.time() < deadline
        time.sleep(0.05)
    name = logs[0].parent.parent.name
    os.killpg(int((jobs / name / "node-0" / "pid").read_text()), 9)
    worker.join(120)
    state = call("get_job", job_id=job["id"])
    assert state["state"] == "reconciliation"
    assert call("get_budget")["reserved"] == quote["bound"]
    assert journal(gym)[-1]["event"] == "unknown"


def free_port():
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def test_final_rank_spawn_splits_across_nodes_and_meets_over_tcp(tmp_path):
    remote = BASE / ("multinode-" + uuid.uuid4().hex[:8])
    (remote / "official").mkdir(parents=True)
    output = remote / "out"
    output.mkdir()
    port = free_port()
    paths = [str(HERE.parent), str(HERE), str(ROOT / "gym" / "src"), str(ROOT)]

    def topo(node):
        return {
            "node": node,
            "nodes": 2,
            "gpus_per_node": 1,
            "master": "127.0.0.1",
            "port": port,
            "remote": str(remote),
            "smoke": None,
            "spawn_wait_seconds": 120,
        }

    driver = (
        "import sys; sys.path[:0] = %r\n"
        "import koala_gym, multinode_helper, torch.multiprocessing as mp\n"
        "koala_gym.install_official(%r, driver=True)\n"
        "mp.spawn(multinode_helper.ranks, args=(2, {}, %r), nprocs=2, join=True)\n"
    ) % (paths, topo(0), str(output))
    satellite = (
        "import sys; sys.path[:0] = %r\nimport koala_gym\nsys.exit(koala_gym.pod_satellite())\n"
        % (paths,)
    )
    env = dict(os.environ, PYTHONPATH=os.pathsep.join(paths))
    node1 = subprocess.Popen(
        [sys.executable, "-c", satellite],
        env=dict(env, KOALA_GYM_TOPOLOGY=json.dumps(topo(1))),
    )
    node0 = subprocess.run([sys.executable, "-c", driver], env=env, timeout=180)
    assert node0.returncode == 0 and node1.wait(60) == 0
    ranks = {k: json.loads((output / f"rank-{k}.json").read_text()) for k in (0, 1)}
    assert all(r["sum"] == 3.0 for r in ranks.values())
    assert ranks[0]["local_rank"] == "0" and ranks[1]["local_rank"] == "0"
    assert ranks[1]["rank"] == 1
    shutil.rmtree(remote, ignore_errors=True)


def test_koala_credentials_reach_only_the_relay(gym, monkeypatch):
    monkeypatch.setenv("OSS_ACCESS_KEY_ID", "test-id")
    backend = gym["service"].backend
    profile = backend.profiles["research-fixture"]
    assert "OSS_ACCESS_KEY_ID" not in backend.environment(profile)
    env, identity = backend.worker_launch({"mode": "execute"}, profile)
    assert env["OSS_ACCESS_KEY_ID"] == "test-id" and identity == {}
    audit_env, _ = backend.worker_launch({"mode": "inspect", "profile": profile})
    assert "OSS_ACCESS_KEY_ID" not in audit_env
    inspect = backend.worker_command({"mode": "inspect", "profile": profile})
    assert inspect[0].endswith("bwrap") and inspect[-2:] == [
        "-m",
        "nexttransformer_gym.worker",
    ]
    assert backend.worker_command({"mode": "execute"})[1:3] == [
        str(HERE.parent / "koala_gym.py"),
        "relay",
    ]


def test_a_worker_python_without_the_pods_torch_build_is_refused(gym):
    config = dict(koala_gym.load_config(gym["path"]))
    config["runtime"] = dict(
        config["runtime"],
        dependencies=dict(config["runtime"]["dependencies"], torch="2.9.1"),
    )
    with pytest.raises(SystemExit, match="set worker_python"):
        koala_gym.check_worker_python(config, sys.executable)


def test_a_pod_without_its_isolation_prerequisites_fails_before_work_without_a_charge(
    gym,
):
    # The landlock backend needs a root controller; these fake pods run as an ordinary user.
    gym["path"].write_text(json.dumps(dict(gym["config"], pod_isolation="landlock")))
    before = gym["call"]("get_budget")
    _, job = run(gym, target_update=1)
    assert job["state"] == "failed" and job["error_code"] == "resource_unavailable"
    assert gym["call"]("get_budget")["available"] == before["available"]
    assert "root controller" in diagnostics(gym, job)


def test_local_workers_use_the_local_sandbox_and_the_gym_reports_the_pods(
    gym, tmp_path
):
    config = koala_gym.load_config(gym["path"])
    config.update(pod_isolation="landlock", local_isolation="bwrap")
    path = koala_gym.execution_config(config, gym["path"])
    assert path.name == f"execution.{gym['path'].stem}.json"
    written = json.loads(path.read_text())
    assert written["isolation"] == "bwrap"
    assert written["executor"]["isolation"] == "landlock"
    profiles = json.loads((ROOT / "gym/examples/profiles.fixture.json").read_text())
    settings = dict(written, executor=dict(written["executor"], devices=[]))
    # The pods' landlock is checked in the pods, never on this ordinary-user controller.
    backend = NativeBackend(
        profiles, tmp_path / "state", ROOT, execution_config=settings
    )
    assert backend.worker_command({"mode": "inspect", "profile": {}})[0].endswith(
        "bwrap"
    )
    official = backend.worker_command(
        {"mode": "inspect", "quote": {"operation": "official"}}
    )
    assert official[-2:] == ["-m", "nexttransformer_gym.worker"]
    assert backend.worker_command({"mode": "execute"})[-2] == "relay"
    shown = backend.resources()["execution_config"]
    assert shown["executor"]["isolation"] == "landlock"
    assert "command" not in shown["executor"] and "environment" not in shown["executor"]


def test_a_landlock_controller_is_refused(gym):
    gym["path"].write_text(json.dumps(dict(gym["config"], local_isolation="landlock")))
    with pytest.raises(SystemExit, match="local_isolation"):
        koala_gym.load_config(gym["path"])


def test_the_probe_reports_reserved_paths_the_worker_reaches(tmp_path):
    allowed, workspace = tmp_path / "allowed.txt", tmp_path / "workspace"
    allowed.write_text("x")
    workspace.mkdir()
    locked_dir, locked_file, open_file = (
        tmp_path / "runs",
        tmp_path / "core.zip",
        tmp_path / "shard.parquet",
    )
    locked_dir.mkdir()
    (locked_dir / "val.json").write_text("{}")
    locked_file.write_text("x")
    open_file.write_text("x")
    locked_dir.chmod(0o300)  # traverse and write, but no listing
    locked_file.chmod(0o200)
    checks = {"reserved": [str(locked_dir), str(locked_file), str(open_file)]}
    spec = {
        "allowed": [str(allowed)],
        "workspace": str(workspace),
        "forbidden": [],
        "reserved": checks["reserved"],
        "outside": [],
        "network": [],
        "code_file": str(allowed),
        "cuda": False,
        "daemon_beat": str(workspace / "beat"),
        "daemon_stop": str(workspace / "stop"),
    }
    (workspace / "stop").write_text("stop")
    try:
        run = subprocess.run(
            [sys.executable, "-c", koala_gym.PROBE_SCRIPT, json.dumps(spec)],
            capture_output=True,
            text=True,
            timeout=60,
        )
    finally:
        locked_dir.chmod(0o700)
        locked_file.chmod(0o600)
    line = [x for x in run.stdout.splitlines() if x.startswith("PROBE_RESULT ")][-1]
    report = json.loads(line[len("PROBE_RESULT ") :])
    if os.geteuid() != 0:
        assert koala_gym.reserved_reached(checks, report) == [str(open_file)]
    # A probe that left no answer counts every reserved path as reached.
    assert koala_gym.reserved_reached(checks, {}) == sorted(checks["reserved"])


def write_journal(state, *entries):
    state.mkdir(parents=True, exist_ok=True)
    with open(state / "koala-jobs.jsonl", "w") as handle:
        for entry in entries:
            handle.write(json.dumps(entry) + "\n")


def test_jobs_totals_gpu_hours_and_stops_only_orphans(tmp_path, monkeypatch, capsys):
    config = tmp_path / "koala.json"
    config.write_text(
        json.dumps(
            {
                "home": str(tmp_path),
                "staging_dir": str(tmp_path / "staging"),
                "remote_root": str(tmp_path / "cpfs"),
                "code_project": "p",
                "job_prefix": "p",
                "gpu_memory_total_bytes": 8 << 30,
                "gpu_memory_limit_bytes": 4 << 30,
            }
        )
    )
    state = tmp_path / "state"
    write_journal(
        state,
        {
            "time": "t1",
            "job": "done-job",
            "quote_id": "q1",
            "operation": "train",
            "event": "submitted",
        },
        {
            "time": "t2",
            "job": "done-job",
            "quote_id": "q1",
            "event": "ended",
            "outcome": "succeeded",
            "gpu_hours": 0.5,
        },
        {
            "time": "t3",
            "job": "orphan-queued",
            "quote_id": "q2",
            "operation": "train",
            "event": "submitted",
        },
        {
            "time": "t4",
            "job": "orphan-ended",
            "quote_id": "q3",
            "operation": "train",
            "event": "submitted",
        },
        {
            "time": "t5",
            "job": "orphan-running",
            "quote_id": "q5",
            "operation": "train",
            "event": "submitted",
            "nodes": 2,
            "gpus_per_node": 8,
        },
        {
            "time": "t6",
            "job": "orphan-running",
            "quote_id": "q5",
            "event": "running",
            "pod_started_t": time.time() - 3600,
        },
        {
            "time": "t7",
            "job": "live-job",
            "quote_id": "q4",
            "operation": "train",
            "event": "submitted",
        },
    )
    status = {
        "orphan-queued": "Queuing",
        "orphan-ended": "Failed",
        "orphan-running": "Running",
        "live-job": "Running",
    }
    stopped = []
    monkeypatch.setattr(koala_gym.Koala, "status", lambda self, job: status[job])
    monkeypatch.setattr(
        koala_gym.Koala, "stop", lambda self, job: stopped.append(job) or 0
    )
    monkeypatch.setattr(
        koala_gym,
        "gym_jobs",
        lambda state: {
            "q2": ("gym-job-2", "session-1", "queued"),
            "q3": ("gym-job-3", "session-1", "running"),
        },
    )
    monkeypatch.setattr(koala_gym, "live_relays", lambda state: [])
    assert koala_gym.main(["--koala", str(config), "jobs", "--state", str(state)]) == 0
    assert (
        stopped == []
        and "5 Koala jobs, 4 open, 0.5 GPU-hours recorded" in capsys.readouterr().out
    )
    # A live relay drives the newest open job; only the older ones are orphans.
    monkeypatch.setattr(koala_gym, "live_relays", lambda state: [4242])
    assert (
        koala_gym.main(
            ["--koala", str(config), "jobs", "--state", str(state), "--stop-orphans"]
        )
        == 0
    )
    out = capsys.readouterr().out
    assert (
        stopped == ["orphan-queued", "orphan-running"]
        and "driven by live relay [4242]" in out
    )
    events = [
        json.loads(line)
        for line in (state / "koala-jobs.jsonl").read_text().splitlines()
    ]
    assert [(e["job"], e["event"]) for e in events[7:]] == [
        ("orphan-queued", "stopped"),
        ("orphan-ended", "unknown"),
        ("orphan-running", "stopped"),
    ]
    # A pod that reported running is charged from its start to the stop, on every GPU; one that never did is not.
    assert "gpu_hours" not in events[7] and abs(events[9]["gpu_hours"] - 16.0) < 0.05
    assert (
        "--session-id session-1 --job-id gym-job-2" in out
        and "--job-id gym-job-3" in out
    )
    assert koala_gym.main(["--koala", str(config), "jobs", "--state", str(state)]) == 0
    assert "5 Koala jobs, 1 open" in capsys.readouterr().out


def write_config(gym, **values):
    path = gym["base"] / f"koala-{uuid.uuid4().hex[:6]}.json"
    path.write_text(json.dumps({**gym["config"], **values}))
    return path


def test_research_gpus_and_microbatch_reach_the_gym_execution_configuration(gym):
    from nexttransformer_gym.gpu import configuration

    config = koala_gym.load_config(
        write_config(
            gym,
            research_gpus=2,
            research_microbatch_size=4,
            research_dtype="bfloat16",
            research_compile=True,
        )
    )
    settings = configuration(koala_gym.execution_settings(config, gym["path"]))
    assert len(set(settings["gpu_ids"])) == 2
    assert settings["research_microbatch_size"] == 4
    assert settings["research_dtype"] == "bfloat16"
    assert settings["research_compile"] is True
    assert settings["isolation"] == config["local_isolation"]
    assert settings["executor"]["isolation"] == config["pod_isolation"]
    assert settings["executor"]["runtime"] == config["runtime"]
    assert {d["name"] for d in settings["executor"]["devices"]} == {config["gpu_name"]}
    payload, _ = koala_gym.build_job(
        config,
        {
            "quote": {"id": "q", "operation": "train", "bound": "1"},
            "profile": {"device": "cuda:0"},
            "gpu": {"id": "GPU-v0", "memory_limit_bytes": 1},
        },
        str(gym["base"] / "remote"),
    )
    env = koala_gym.worker_env(
        payload, gym["base"] / "code", gym["base"] / "work", [{"id": "GPU-a"}]
    )
    assert env["CUDA_VISIBLE_DEVICES"] == "GPU-a"
    assert env["NANOCHAT_DTYPE"] == "bfloat16"
    assert env["TORCHDYNAMO_DISABLE"] == "0"
    assert env["FLA_DISABLE_BACKEND_DISPATCH"] == "1"
    with pytest.raises(SystemExit, match="research_gpus"):
        koala_gym.load_config(write_config(gym, virtual_gpus=2))
    with pytest.raises(SystemExit, match="positive integer"):
        koala_gym.load_config(write_config(gym, research_microbatch_size=0))
    with pytest.raises(SystemExit, match="research_dtype"):
        koala_gym.load_config(write_config(gym, research_dtype="float16"))
    with pytest.raises(SystemExit, match="research_compile"):
        koala_gym.load_config(write_config(gym, research_compile="yes"))


def test_a_research_job_asks_koala_for_the_gym_allocations_gpus(gym, tmp_path):
    group = [{"id": f"GPU-v{i}", "memory_limit_bytes": 1} for i in range(4)]
    request = {
        "quote": {"id": "q", "operation": "train", "bound": "1"},
        "profile": {"device": "cuda:0"},
    }
    sizes = []
    for allocation in ({**group[0], "devices": group}, group[0], None):
        payload, _ = koala_gym.build_job(
            koala_gym.load_config(gym["path"]),
            {**request, "gpu": allocation},
            str(tmp_path),
        )
        sizes.append((payload["nodes"], payload["gpus_per_node"]))
    assert sizes == [(1, 4), (1, 1), (1, 1)]


def test_a_two_gpu_research_job_reaches_koala_as_one_two_gpu_job(gym):
    """CPU only: the Gym dispatches GPU work only for a CUDA runtime, so the configuration records one; the
    CPU pod then fails its runtime check before work, and nothing is charged."""
    path = write_config(
        gym,
        runtime=dict(gym["config"]["runtime"], cuda_version="12.8"),
        research_gpus=2,
        research_microbatch_size=1,
        virtual_gpu_offset=4000 + int(uuid.uuid4().hex[:4], 16),
    )
    config = koala_gym.load_config(path)
    profiles = json.loads((ROOT / "gym/examples/profiles.fixture.json").read_text())
    for profile in profiles.values():
        profile["timeout_sec"] = 600
    backend = NativeBackend(
        profiles,
        gym["base"] / "state-2gpu",
        ROOT,
        execution_config=koala_gym.execution_settings(config, path),
    )
    service = Service(gym["base"] / "state-2gpu" / "gym.sqlite3", backend)
    credentials = service.create_session(10**12, "koala-test-2gpu")

    def call(operation, **params):
        return service.call(credentials["agent_token"], operation, params)

    research = call("get_resources")["profiles"]
    assert {p["id"]: p["supported_device_counts"] for p in research}[
        "research-fixture"
    ] == [2]
    candidate = call(
        "submit_candidate",
        package_base64=base64.b64encode(candidate_bytes()).decode(),
        idempotency_key="first",
    )
    quote = call(
        "quote_experiment",
        candidate_id=candidate["id"],
        profile_id="research-fixture",
        target_update=1,
    )
    before = call("get_budget")
    job = call("run_experiment", quote_id=quote["id"], idempotency_key=quote["id"])
    Scheduler(service).work_once()
    job = call("get_job", job_id=job["id"])
    specs = [
        json.loads((path / "spec.json").read_text())
        for path in (gym["home"] / ".fake-koala" / "jobs").glob("*")
    ]
    assert [(spec["nodes"], spec["gpus"]) for spec in specs] == [(1, 2)]
    assert job["state"] == "failed" and job["error_code"] == "resource_unavailable"
    assert call("get_budget")["available"] == before["available"]


def test_cancelling_a_started_long_training_job_refunds_unexecuted_steps(gym):
    import io
    import zipfile

    gym["service"].backend.profiles["research-fixture"]["max_updates"] = 60
    with zipfile.ZipFile(io.BytesIO(candidate_bytes())) as original:
        files = {name: original.read(name) for name in original.namelist()}
    files["plugin/research_model/model.py"] = (
        b"import time\nfrom nexttransformer.models.lm import ReferenceLM\n"
        b"class ResearchModel(ReferenceLM):\n"
        b"    def _readout(self, x, backout):\n"
        b"        if self.training:\n"
        b"            time.sleep(0.2)\n"
        b"        return super()._readout(x, backout) * self.config.option('output_scale')\n"
    )
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as archive:
        for name, value in files.items():
            archive.writestr(name, value)

    call = gym["call"]
    slow = call(
        "submit_candidate",
        package_base64=base64.b64encode(buf.getvalue()).decode(),
        idempotency_key="slow-cand",
    )
    before = call("get_budget")
    quote = call(
        "quote_experiment",
        candidate_id=slow["id"],
        profile_id="research-fixture",
        target_update=60,
    )
    job = call("run_experiment", quote_id=quote["id"], idempotency_key="long-run")
    worker = threading.Thread(target=Scheduler(gym["service"]).work_once)
    worker.start()
    deadline = time.time() + 60
    while not any(e["event"] == "running" for e in journal(gym)):
        assert time.time() < deadline
        time.sleep(0.1)
    call("cancel_job", job_id=job["id"], idempotency_key="cancel-midway")
    worker.join(60)
    assert call("get_job", job_id=job["id"])["state"] == "cancelled"
    after = call("get_budget")
    assert after["reserved"] == "0"
    spent_delta = int(after["spent"]) - int(before["spent"])
    assert 0 < spent_delta < int(quote["bound"]) // 2
    assert int(before["available"]) - int(after["available"]) == spent_delta


# ------------------------------------------------------------------------------------- multi-node research


def test_a_research_group_larger_than_a_node_spans_whole_nodes(gym, tmp_path):
    config = koala_gym.load_config(
        write_config(
            gym, research_gpus=4, research_gpus_per_node=2, pod_isolation="none"
        )
    )
    group = [{"id": f"GPU-v{i}", "memory_limit_bytes": 1} for i in range(4)]
    request = {
        "quote": {"id": "q", "operation": "train", "bound": "1"},
        "profile": {"device": "cuda:0"},
    }
    shapes = []
    for allocation in (
        {**group[0], "devices": group},
        {**group[0], "devices": group[:2]},
        group[0],
    ):
        payload, _ = koala_gym.build_job(
            config, {**request, "gpu": allocation}, str(tmp_path)
        )
        shapes.append((payload["nodes"], payload["gpus_per_node"]))
    assert shapes == [(2, 2), (1, 2), (1, 1)]
    assert payload["mesh_timeout_seconds"] == 3600
    with pytest.raises(GymError, match="does not fill"):
        koala_gym.research_topology(config, {**group[0], "devices": group[:3]})


@pytest.mark.parametrize(
    "values, message",
    [
        ({"research_gpus": 3}, "multiple"),
        ({"pod_isolation": "landlock"}, "pod_isolation 'none'"),
        (
            {"remote_root": "s3://bucket/x", "final": {"nodes": 1, "gpus_per_node": 8}},
            "shared remote_root",
        ),
    ],
)
def test_multi_node_research_needs_whole_nodes_shared_storage_and_no_netns_sandbox(
    gym, values, message
):
    base = {"research_gpus": 4, "research_gpus_per_node": 2, "pod_isolation": "none"}
    with pytest.raises(SystemExit, match=message):
        koala_gym.load_config(write_config(gym, **{**base, **values}))


# --------------------------------------------------------------------------------------- the Gym's interface


def gym_imports(function):
    """The Gym modules a top-level function of koala_gym.py imports or runs (`-m`)."""
    import ast

    tree = ast.parse((HERE.parent / "koala_gym.py").read_text())
    node = next(
        n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == function
    )
    found = set()
    for item in ast.walk(node):
        if isinstance(item, ast.ImportFrom) and (item.module or "").startswith(
            "nexttransformer_gym"
        ):
            module = item.module
            found |= (
                {f"{module}.{a.name}" for a in item.names}
                if module == "nexttransformer_gym"
                else {module}
            )
        if isinstance(item, ast.Constant) and str(item.value).startswith(
            "nexttransformer_gym."
        ):
            found.add(item.value)
    return found


PUBLIC = {
    "nexttransformer_gym.executor",
    "nexttransformer_gym.nodes",
    "nexttransformer_gym.remote",
    "nexttransformer_gym.cli",
}
# The official final-run path still adapts the Gym's official pipeline in-process.
OFFICIAL = {"install_official", "rank_entry", "pod_official", "smoke_invoke"}


def test_the_adapter_uses_only_the_gyms_public_interface_outside_the_official_path():
    import ast

    tree = ast.parse((HERE.parent / "koala_gym.py").read_text())
    functions = [n.name for n in tree.body if isinstance(n, ast.FunctionDef)]
    for name in functions:
        used = gym_imports(name)
        if name in OFFICIAL:
            continue
        assert used <= PUBLIC, (name, used - PUBLIC)
    source = (HERE.parent / "koala_gym.py").read_text()
    for private in ("NativeBackend", "install_controller", "RuntimeQuery", "sqlite3"):
        assert private not in source
    # The Gym's multi-node executor runs the nodes and the relay; this file is its Koala transport only.
    for generic in ("def drive(", "def research_node(", "def report(", "Charges(", "Channel("):
        assert generic not in source


def test_agents_see_neither_koala_nor_the_relay(gym):
    resources = gym["call"]("get_resources")
    text = json.dumps(resources)
    assert "koala" not in text.lower() and "relay" not in text
    assert '"backend": "operator_executor"' in text
