"""The multi-node executor (`nodes`): local and ssh launchers through the Gym, and the node-side rules.

CPU processes on this machine stand in for the nodes: a CPU job spanning nodes meets over gloo. A fake ssh runs the
remote command here, so the ssh launcher runs end to end too."""

import base64
import io
import json
import subprocess
import threading
import time
from pathlib import Path

import pytest
from nexttransformer_gym import mesh, nodes, remote
from nexttransformer_gym.backend import NativeBackend, Scheduler
from nexttransformer_gym.common import GymError
from nexttransformer_gym.gpu import configuration
from nexttransformer_gym.service import Service
from test_gym import ISOLATION, ROOT, candidate_bytes


def write(path, value):
    path.write_text(json.dumps(value))
    return path


def fake_ssh(tmp_path):
    """An ssh that runs the remote command on this machine (the destination is ignored)."""
    script = tmp_path / "ssh"
    script.write_text('#!/bin/sh\nshift\nexec /bin/sh -c "$*"\n')
    script.chmod(0o755)
    return str(script)


def node_config(tmp_path, **values):
    return write(
        tmp_path / "nodes.json",
        {
            "launcher": "local",
            "nodes": 2,
            "cpu_nodes": 2,
            "job_root": str(tmp_path / "jobs"),
            "isolation": "none",
            "network": "deny",
            "master_port": 0,
            "poll_seconds": 0.2,
            **values,
        },
    )


def gym(tmp_path, config_path, profiles=None):
    execution = nodes.configure(
        config_path, {"isolation": ISOLATION, "gpu_memory_limit_bytes": 4 << 30}
    )
    profiles = profiles or json.loads(
        (ROOT / "gym/examples/profiles.fixture.json").read_text()
    )
    backend = NativeBackend(
        profiles, tmp_path / "state", ROOT, execution_config=execution
    )
    service = Service(tmp_path / "state" / "gym.sqlite3", backend)
    credentials = service.create_session(10**12, "nodes")

    def call(operation, **params):
        return service.call(credentials["agent_token"], operation, params)

    candidate = call(
        "submit_candidate",
        package_base64=base64.b64encode(candidate_bytes()).decode(),
        idempotency_key="first",
    )
    return service, call, candidate


def train(service, call, candidate, target, checkpoint_id=None):
    quote = call(
        "quote_experiment",
        candidate_id=candidate["id"],
        profile_id="research-fixture",
        target_update=target,
        **({"checkpoint_id": checkpoint_id} if checkpoint_id else {}),
    )
    job = call("run_experiment", quote_id=quote["id"], idempotency_key=quote["id"])
    assert Scheduler(service).work_once()
    state = call("get_job", job_id=job["id"])
    log = service.backend.state_dir / (job["id"] + ".diagnostic.log")
    assert state["state"] == "succeeded", log.read_text() if log.exists() else state
    return quote, call("get_result", job_id=job["id"])


def journal(service):
    path = service.backend.state_dir / "nodes-jobs.jsonl"
    return [json.loads(line) for line in path.read_text().splitlines()]


def test_a_two_node_job_trains_as_one_node_and_continues_from_its_checkpoint(tmp_path):
    (tmp_path / "one").mkdir()
    single = gym(tmp_path / "one", node_config(tmp_path / "one", cpu_nodes=1, nodes=1))
    _, alone = train(*single, 3)

    (tmp_path / "two").mkdir()
    service, call, candidate = gym(tmp_path / "two", node_config(tmp_path / "two"))
    quote, result = train(service, call, candidate, 3)
    placement = result["execution"]
    assert placement["nodes"] == 2 and placement["total_device_count"] == 2
    assert placement["cross_node_resyncs"] == 0
    assert result["dev_bpb"] == pytest.approx(alone["dev_bpb"], rel=1e-4)
    budget = call("get_budget")
    assert budget["spent"] == quote["bound"] and budget["reserved"] == "0"
    # The Gym keeps a pointer; node 0 alone wrote the checkpoint in the shared job directory.
    events = journal(service)
    assert [e["event"] for e in events] == ["submitted", "running", "ended"]
    assert events[-1]["outcome"] == "succeeded"
    job_dir = Path(events[0]["remote"])
    assert (job_dir / "checkpoint.pt").exists()
    assert not list((job_dir / "work" / "node-1").rglob("checkpoint.pt")) or all(
        "follower" in str(p)
        for p in (job_dir / "work" / "node-1").rglob("checkpoint.pt")
    )
    resumed_quote, resumed = train(service, call, candidate, 5, result["checkpoint_id"])
    assert resumed["step"] == 5
    assert call("get_budget")["spent"] == str(
        int(quote["bound"]) + int(resumed_quote["bound"])
    )


def test_the_ssh_launcher_runs_each_node_through_ssh(tmp_path):
    config = node_config(
        tmp_path,
        launcher="ssh",
        hosts=[
            {"ssh": "node-a", "address": "127.0.0.1"},
            {"ssh": "user@node-b", "address": "127.0.0.1"},
        ],
        ssh=[fake_ssh(tmp_path)],
        master_port=nodes.free_ports(),
    )
    service, call, candidate = gym(tmp_path, config)
    _, result = train(service, call, candidate, 2)
    assert result["execution"]["nodes"] == 2
    assert journal(service)[-1]["outcome"] == "succeeded"


def test_cancel_job_stops_every_node_and_charges_only_the_started_window(tmp_path):
    profiles = json.loads((ROOT / "gym/examples/profiles.fixture.json").read_text())
    profiles["research-fixture"].update(max_updates=40000, timeout_sec=600)
    service, call, candidate = gym(tmp_path, node_config(tmp_path), profiles)
    quote = call(
        "quote_experiment",
        candidate_id=candidate["id"],
        profile_id="research-fixture",
        target_update=40000,
    )
    job = call("run_experiment", quote_id=quote["id"], idempotency_key="long")
    worker = threading.Thread(target=Scheduler(service).work_once)
    worker.start()
    path = service.backend.state_dir / "nodes-jobs.jsonl"
    deadline = time.time() + 120
    while not (path.exists() and "running" in path.read_text()):
        assert time.time() < deadline and worker.is_alive()
        time.sleep(0.2)
    time.sleep(2)
    call("cancel_job", job_id=job["id"], idempotency_key="stop")
    worker.join(120)
    assert call("get_job", job_id=job["id"])["state"] == "cancelled"
    budget = call("get_budget")
    assert budget["reserved"] == "0" and 0 < int(budget["spent"]) < int(quote["bound"])
    events = journal(service)
    assert events[-1]["event"] == "stopped" and events[-1]["reason"] == "cancel_job"
    job_dir = Path(events[0]["remote"])
    assert (job_dir / "stop").exists()
    deadline = time.time() + 30
    while (
        subprocess.run(["pgrep", "-f", str(job_dir)], capture_output=True).returncode
        == 0
    ):
        assert time.time() < deadline, "node processes outlived the job"
        time.sleep(0.5)


def test_final_training_never_runs_on_the_nodes(tmp_path):
    launcher = nodes.launcher(nodes.load(node_config(tmp_path)))
    request = {
        "quote": {"id": "q", "operation": "official", "bound": "1"},
        "profile": {"device": "cuda:0"},
        "execution": {},
    }
    with pytest.raises(GymError, match="dispatched separately"):
        launcher.plan(request, "tag")


@pytest.mark.parametrize(
    "values, message",
    [
        ({"isolation": "bwrap"}, "isolation none"),
        ({"launcher": "slurm"}, "local or ssh"),
        ({"launcher": "ssh", "hosts": []}, "needs hosts"),
        ({"extra": 1}, "unknown fields"),
        ({"cpu_nodes": 3}, "cpu_nodes exceeds"),
        (
            {"launcher": "ssh", "hosts": ["a"], "cpu_nodes": 1, "master_port": 0},
            "local nodes only",
        ),
    ],
)
def test_a_nodes_configuration_is_checked(tmp_path, values, message):
    with pytest.raises(SystemExit, match=message):
        nodes.load(node_config(tmp_path, **values))


def test_configure_records_the_runtime_and_names_the_relay(tmp_path):
    path = node_config(tmp_path)
    value = nodes.configure(path, {"isolation": ISOLATION})
    executor = value["executor"]
    assert executor["command"][-3:] == ["nexttransformer_gym.nodes", "relay", str(path)]
    assert executor["runtime"] == remote.runtime()
    assert (executor["isolation"], executor["network"]) == ("none", "deny")
    assert configuration(value)["executor"]["devices"] == executor["devices"]


# ------------------------------------------------------------------------------------------- node rules


def test_exact_cards_are_this_nodes_own_and_idle(monkeypatch):
    from nexttransformer_gym import gpu

    cards = [
        {"id": f"GPU-{i}", "uuid": f"GPU-{i}", "index": i, "compute_process_count": 0}
        for i in range(3)
    ]
    monkeypatch.setattr(gpu, "discover", lambda: {"status": "ok", "gpus": cards})
    payload = {"device_ids": "exact"}
    chosen = nodes.node_cards(payload, [{"id": "GPU-2"}, {"id": "GPU-0"}])
    assert [c["id"] for c in chosen] == ["GPU-2", "GPU-0"]
    with pytest.raises(GymError, match="no GPU GPU-9"):
        nodes.node_cards(payload, [{"id": "GPU-9"}])
    cards[1]["compute_process_count"] = 1
    with pytest.raises(GymError, match="runs another compute process"):
        nodes.node_cards(payload, [{"id": "GPU-1"}])


def test_a_node_maps_a_gpu_group_onto_its_own_gpus_in_order():
    virtual = [
        {"id": f"GPU-virtual-{i}", "uuid": f"GPU-virtual-{i}", "memory_limit_bytes": 7}
        for i in range(2)
    ]
    cards = [
        {"id": "GPU-a", "uuid": "GPU-a", "index": 0, "device_minor": 3},
        {"id": "GPU-b", "uuid": "GPU-b", "index": 1, "device_minor": 5},
    ]
    record, visible = nodes.assign_gpus({**virtual[0], "devices": virtual}, cards)
    assert visible == "GPU-a,GPU-b" and record["id"] == "GPU-a"
    assert [
        (d["uuid"], d["device_minor"], d["memory_limit_bytes"])
        for d in record["devices"]
    ] == [("GPU-a", 3, 7), ("GPU-b", 5, 7)]
    single, visible = nodes.assign_gpus(virtual[0], cards[:1])
    assert visible == "GPU-a" and "devices" not in single
    with pytest.raises(GymError, match="1 GPUs for an allocation of 2"):
        nodes.assign_gpus({**virtual[0], "devices": virtual}, cards[:1])


def test_a_group_larger_than_a_node_spans_whole_nodes():
    assert nodes.research_topology(4, 2) == (2, 2)
    assert nodes.research_topology(2, 2) == (1, 2)
    assert nodes.research_topology(1, 8) == (1, 1)
    with pytest.raises(GymError, match="does not fill"):
        nodes.research_topology(3, 2)


def research_payload(base, node_count=2, per=1):
    shared = base / "shared" / "tag" / "q"
    shared.mkdir(parents=True)
    group = [
        {"id": f"GPU-v{i}", "uuid": f"GPU-v{i}", "memory_limit_bytes": 7}
        for i in range(node_count * per)
    ]
    return {
        "request": {
            "quote": {"id": "q", "operation": "train", "bound": "100"},
            "profile": {"device": "cuda:0"},
            "gpu": {**group[0], "devices": group},
            "output": str(shared / "checkpoint.pt"),
            "platform_root": str(shared / "code"),
        },
        "remote": str(shared),
        "remote_root": str(base / "shared"),
        "nodes": node_count,
        "gpus_per_node": per,
        "credit": "100",
        "isolation": "none",
        "network": "deny",
        "master_port": 29531,
        "mesh_timeout_seconds": 60,
        "rendezvous_seconds": 60,
    }


class FakeCheck:
    """A mesh check process: finishes with `code`, or (code None) runs until killed."""

    def __init__(self, code):
        self.code, self.killed = code, False
        self.stdout = io.StringIO(
            'NCCL INFO ring\n{"node": 0, "bus_gbit_per_s": 1.0}\n'
        )

    @property
    def returncode(self):
        return -9 if self.killed else self.code

    def poll(self):
        return self.returncode

    def kill(self):
        self.killed = True

    def wait(self):
        return self.returncode


def fake_node(monkeypatch, outcome, mesh_code=0):
    """Stand-ins for the node's GPUs, worker identity, mesh check and worker."""
    seen = {}
    monkeypatch.setattr(
        nodes,
        "node_cards",
        lambda payload, wanted: [
            {"id": f"GPU-p{i}", "uuid": f"GPU-p{i}", "index": i, "device_minor": i}
            for i in range(len(wanted))
        ],
    )
    monkeypatch.setattr(remote, "worker_identity", lambda request: {})
    monkeypatch.setattr(nodes, "PEER_GRACE_SECONDS", 0)
    real_popen = subprocess.Popen

    def popen(command, *args, **options):
        if command[-2:] == ["nexttransformer_gym.mesh", "check"]:
            seen["mesh"] = json.loads(options["env"][mesh.ENV])
            seen["check"] = FakeCheck(mesh_code)
            return seen["check"]
        return real_popen(command, *args, **options)

    monkeypatch.setattr(nodes.subprocess, "Popen", popen)

    def drive(command, env, request, credit, emit, cwd, stderr_path, **options):
        seen.update(command=command, env=env, request=request)
        return outcome(request)

    monkeypatch.setattr(nodes, "drive", drive)
    return seen


def node_events(capsys):
    return [
        json.loads(line[len(nodes.MARK) :])
        for line in capsys.readouterr().out.splitlines()
        if line.startswith(nodes.MARK)
    ]


def test_each_node_trains_its_share_and_only_node_0_reports_and_writes(
    tmp_path, monkeypatch, capsys
):
    payload = research_payload(tmp_path)
    shared = Path(payload["remote"])

    def succeeded(request):
        Path(request["output"]).parent.mkdir(parents=True, exist_ok=True)
        Path(request["output"]).write_bytes(b"weights")
        return {"outcome": "succeeded", "result": {"step": 1}, "spent": 100}

    seen = fake_node(monkeypatch, succeeded)
    (shared / "research").mkdir()
    (shared / "research" / "done").write_text("succeeded")  # lets node 1 leave at once
    work = tmp_path / "node1"
    copy = json.loads(json.dumps(payload))
    assert (
        nodes.run(copy, 1, "10.0.0.1", nodes.Emitter(), work / "code", work, shared)
        == 0
    )
    assert seen["mesh"] == {
        "node": 1,
        "nodes": 2,
        "master": "10.0.0.1",
        "port": 29532,
        "timeout": 60,
        "network": "deny",
    }
    assert json.loads(seen["env"][mesh.ENV])["port"] == 29531
    assert seen["env"]["CUDA_VISIBLE_DEVICES"] == "GPU-p0"
    assert seen["command"][-3:] == ["-m", "nexttransformer_gym.mesh", "worker"]
    # Node 1's card of the group, on this node's own GPU; its worker never writes the job's checkpoint.
    assert [d["uuid"] for d in seen["request"]["gpu"]["devices"]] == ["GPU-p0"]
    assert seen["request"]["output"] != payload["request"]["output"]
    assert not (shared / "checkpoint.pt").exists()
    assert {e["kind"] for e in node_events(capsys)} == {"satellite"}

    (shared / "research" / "done").unlink()
    work = tmp_path / "node0"
    assert (
        nodes.run(
            json.loads(json.dumps(payload)),
            0,
            "10.0.0.1",
            nodes.Emitter(),
            work / "code",
            work,
            shared,
        )
        == 0
    )
    events = node_events(capsys)
    kinds = [e["kind"] for e in events]
    assert kinds.index("mesh") < kinds.index("ready") and kinds[-1] == "done"
    assert events[-1]["outcome"] == "succeeded"
    assert events[-1]["checkpoint"]["path"] == payload["request"]["output"]
    assert (shared / "research" / "done").read_text() == "succeeded"


def test_node_0_settles_a_peer_failure_as_a_failed_job(tmp_path, monkeypatch, capsys):
    payload = research_payload(tmp_path)
    shared = Path(payload["remote"])
    (shared / "research").mkdir()
    (shared / "research" / "failed-1").write_text("worker exited 1")
    fake_node(
        monkeypatch,
        lambda request: {"outcome": "unknown", "detail": "exited -9", "spent": 40},
    )
    work = tmp_path / "node0"
    assert (
        nodes.run(payload, 0, "10.0.0.1", nodes.Emitter(), work / "code", work, shared)
        == 0
    )
    done = node_events(capsys)[-1]
    assert done["kind"] == "done" and done["outcome"] == "failed"
    assert done["error_code"] == "infrastructure_failure"
    assert "failed-1" in done["detail"] and done["spent"] == "40"
    assert (shared / "research" / "failed-0").exists()


def test_a_mesh_that_does_not_form_fails_before_work_without_a_charge(
    tmp_path, monkeypatch, capsys
):
    payload = research_payload(tmp_path)
    shared = Path(payload["remote"])
    seen = fake_node(monkeypatch, lambda request: pytest.fail("worker ran"), 1)
    work = tmp_path / "node0"
    assert (
        nodes.run(payload, 0, "10.0.0.1", nodes.Emitter(), work / "code", work, shared)
        == 1
    )
    events = node_events(capsys)
    assert "ready" not in [e["kind"] for e in events]
    assert events[-1]["kind"] == "done" and events[-1]["outcome"] == "pre_failure"
    assert "command" not in seen and (shared / "research" / "failed-0").exists()


def test_another_node_waits_for_node_0_after_its_own_failure(
    tmp_path, monkeypatch, capsys
):
    """On some clusters a job ends when any node does: a failed node stays up until node 0 has logged the outcome."""
    payload = research_payload(tmp_path)
    shared = Path(payload["remote"])
    (shared / "research").mkdir()
    (shared / "research" / "done").write_text("failed")
    fake_node(
        monkeypatch,
        lambda request: {"outcome": "unknown", "detail": "exited 1", "spent": 9},
    )
    work = tmp_path / "node1"
    assert (
        nodes.run(payload, 1, "10.0.0.1", nodes.Emitter(), work / "code", work, shared)
        == 1
    )
    assert (shared / "research" / "failed-1").exists()
    assert {e["kind"] for e in node_events(capsys)} == {"satellite"}


def test_a_mesh_check_stops_when_another_node_failed_or_the_job_was_stopped(
    tmp_path, monkeypatch, capsys
):
    payload = research_payload(tmp_path)
    shared = Path(payload["remote"])
    (shared / "research").mkdir()
    (shared / "research" / "failed-1").write_text("environment build failed")
    seen = fake_node(monkeypatch, lambda request: pytest.fail("worker ran"), None)
    work = tmp_path / "node0"
    assert (
        nodes.run(payload, 0, "10.0.0.1", nodes.Emitter(), work / "code", work, shared)
        == 1
    )
    assert seen["check"].killed
    done = node_events(capsys)[-1]
    assert done["kind"] == "done" and done["outcome"] == "pre_failure"
    assert "failed-1" in done["detail"]
    assert (shared / "research" / "done").read_text() == "pre_failure"

    payload = research_payload(tmp_path / "again")
    shared = Path(payload["remote"])
    (shared / "stop").write_text("")
    seen = fake_node(monkeypatch, lambda request: pytest.fail("worker ran"), None)
    work = tmp_path / "again" / "node0"
    assert (
        nodes.run(payload, 0, "10.0.0.1", nodes.Emitter(), work / "code", work, shared)
        == 1
    )
    assert seen["check"].killed
    assert "the job was stopped" in node_events(capsys)[-1]["detail"]


def test_a_node_whose_runtime_differs_fails_before_work(tmp_path, capsys):
    payload = research_payload(tmp_path)
    payload["runtime"] = dict(remote.runtime(), python=[3, 0, 0])
    payload["work_root"] = str(tmp_path / "work")
    path = write(tmp_path / "job.json", payload)
    assert nodes.node_main(path, 1, "10.0.0.1") == 1
    done = node_events(capsys)[-1]
    assert (
        done["outcome"] == "pre_failure"
        and "differs from the recorded" in done["detail"]
    )
    # The other nodes stop waiting for this one.
    assert (Path(payload["remote"]) / "research" / "failed-1").exists()


def test_a_worker_identity_can_reach_its_workspace_without_listing_the_tree(tmp_path):
    root = tmp_path / "remote"
    workspace = root / "tag" / "quote"
    workspace.mkdir(parents=True)
    outside = tmp_path / "other"
    outside.mkdir()
    for path in (tmp_path, outside, root, root / "tag", workspace):
        path.chmod(0o700)  # what a root node's umask 0077 leaves
    nodes.reach(workspace, root)
    for path in (root, root / "tag", workspace):
        assert path.stat().st_mode & 0o777 == 0o701  # traverse only: no listing
    assert (
        outside.stat().st_mode & 0o777 == 0o700
        and tmp_path.stat().st_mode & 0o777 == 0o700
    )


def test_the_identity_cleanup_runs_only_on_a_root_node_and_never_for_root(monkeypatch):
    calls = []
    monkeypatch.setattr(nodes.subprocess, "run", lambda *a, **k: calls.append(k))
    nobody = {"user": 65534, "group": 65534, "extra_groups": []}
    monkeypatch.setattr(nodes.os, "geteuid", lambda: 1000)
    nodes.stop_identity(nobody)
    assert calls == []  # an ordinary runner: no identity switch, nothing to kill
    monkeypatch.setattr(nodes.os, "geteuid", lambda: 0)
    nodes.stop_identity({})
    nodes.stop_identity({"user": 0, "group": 0})
    assert calls == []
    nodes.stop_identity(nobody)
    assert len(calls) == 2 and all(
        c["user"] == 65534 and c["extra_groups"] == [] for c in calls
    )


def test_usage_counts_every_gpu_of_every_node():
    assert nodes.usage(None, 10.0, 1, 8) == {}
    held = nodes.usage(1000.0, 1000.0 + 1800, 2, 8)
    assert held["pod_seconds"] == 1800.0 and held["gpu_hours"] == 8.0
    assert held["pod_started"] == "1970-01-01T00:16:40Z"


def test_a_group_reports_every_nodes_cards_in_global_order():
    from nexttransformer_gym.worker import across_nodes

    def record(node):
        devices = [
            {
                "gpu_id": f"GPU-{node}{i}",
                "peak_memory_allocated_bytes": 10 * node + i,
                "peak_memory_reserved_bytes": 0,
            }
            for i in range(2)
        ]
        return {
            "device": "cuda:0",
            "gpu_id": devices[0]["gpu_id"],
            "gpu_ids": [d["gpu_id"] for d in devices],
            "device_count": 2,
            "devices": devices,
            "gpu_name": "NVIDIA H20-3e",
            "peak_memory_allocated_bytes": 10 * node,
        }

    merged = across_nodes([record(0), record(1)])
    assert merged["gpu_ids"] == ["GPU-00", "GPU-01", "GPU-10", "GPU-11"]
    assert merged["device_count"] == 4 and merged["nodes"] == 2
    assert [d["node"] for d in merged["devices"]] == [0, 0, 1, 1]
    assert [d["peak_memory_allocated_bytes"] for d in merged["devices"]] == [
        0,
        1,
        10,
        11,
    ]
    # Node 0's first card stays the primary device.
    assert merged["gpu_id"] == "GPU-00" and merged["peak_memory_allocated_bytes"] == 0
    assert across_nodes([{"device": "cpu"}, {"device": "cpu"}]) == {
        "device": "cpu",
        "nodes": 2,
    }
