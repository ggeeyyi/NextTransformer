"""Operator executors: the execution configuration's executor block, the worker protocol an executor speaks, and
the remote-side functions (runtime, confinement, multi-node groups) it uses instead of Gym internals."""

import base64
import json
import os
import socket
import subprocess
import sys
import threading
import time
from pathlib import Path

import pytest
from nexttransformer_gym import accounting, executor, mesh, remote
from nexttransformer_gym.backend import NativeBackend, Scheduler
from nexttransformer_gym.common import GymError
from nexttransformer_gym.gpu import configuration
from nexttransformer_gym.service import Service
from torch import multiprocessing

from test_gym import ISOLATION, ROOT, candidate_bytes

UUIDS = [f"GPU-0000000{i}-0000-0000-0000-000000000000" for i in range(2)]
RUNTIME = {
    "dependencies": {"torch": "2.10.0"},
    "python": [3, 12, 3],
    "cuda_version": "12.8",
}

# A fake executor that uses only the public interface: it runs every job "elsewhere" by writing its result.
FAKE_EXECUTOR = r"""
import json, os, sys, time
from pathlib import Path
from nexttransformer_gym.executor import Channel, Charges

channel = Channel()
request = channel.request()
quote = request["quote"]
log = Path(os.environ["FAKE_LOG"])
log.write_text(json.dumps({
    "cwd": os.getcwd(),
    "site_token": os.environ.get("SITE_TOKEN"),
    "other": os.environ.get("OTHER_SECRET"),
    "checkpoint": Path(request["checkpoint"]).read_text() if request.get("checkpoint") else None,
}))
if os.environ.get("FAKE_MODE") == "wait-for-cancel":
    while not channel.poll():
        time.sleep(0.05)
    channel.cancelled()
    sys.exit(0)
charges = Charges(channel, quote, window=2, label="fake")
if not charges.advance(0):
    channel.cancelled()
    sys.exit(0)
start = int(quote.get("start_update") or 0)
for step in range(start + 1, int(quote["target_update"]) + 1):
    channel.progress(step)
    if not charges.advance(step - start):
        channel.cancelled()
        sys.exit(0)
Path(request["output"]).write_text(json.dumps({"weights": "elsewhere", "step": quote["target_update"]}))
if not charges.settle_full():
    channel.cancelled()
    sys.exit(0)
channel.result({"step": quote["target_update"], "dev_bpb": 1.5, "curves": [],
                "candidate_hash": quote["candidate_hash"], "profile_hash": quote["profile_hash"],
                "runtime_revision": quote["runtime_revision"], "official_compatible": False})
"""


def block(**values):
    return {
        "command": ["/bin/true"],
        "runtime": RUNTIME,
        "devices": [
            {"id": uid, "name": "NVIDIA H20-3e", "memory_total_bytes": 8 << 30}
            for uid in UUIDS
        ],
        **values,
    }


def test_an_executor_block_declares_the_allocation_and_is_checked():
    plain = configuration({"isolation": "none"})
    assert "executor" not in plain
    settings = configuration(
        {"isolation": "none", "gpu_memory_limit_bytes": 4 << 30, "executor": block()}
    )
    assert settings["gpu_ids"] == UUIDS
    assert settings["executor"]["isolation"] == "bwrap"
    assert settings["executor"]["network"] == "deny"
    assert settings["executor"]["environment"] == []
    for bad in (
        {"executor": block(), "gpu_ids": UUIDS[::-1]},
        {"executor": block(shell="x")},
        {"executor": block(command=[])},
        {"executor": block(environment=["NOT A NAME"])},
        {"executor": block(runtime={"python": [3]})},
        {"executor": block(isolation="docker")},
        {
            "executor": block(
                devices=[{"id": "gpu0", "name": "x", "memory_total_bytes": 8 << 30}]
            )
        },
        {
            "executor": block(),
            "gpu_memory_limit_bytes": 8 << 30,
        },  # no room for the budget
    ):
        with pytest.raises(GymError):
            configuration(
                {"isolation": "none", "gpu_memory_limit_bytes": 4 << 30, **bad}
            )


def executor_backend(
    tmp_path, command=None, environment=(), devices=False, python=None, **options
):
    profiles = json.loads((ROOT / "gym/examples/profiles.fixture.json").read_text())
    value = block(
        command=command or ["/bin/true"],
        environment=list(environment),
        runtime=remote.runtime() if not devices else RUNTIME,
    )
    if not devices:
        value["devices"] = []
    config = {
        "isolation": ISOLATION,
        "gpu_memory_limit_bytes": 4 << 30,
        "executor": value,
        **options,
    }
    return NativeBackend(
        profiles, tmp_path, ROOT, python=python, execution_config=config
    )


def test_agents_and_the_revision_see_neither_the_executor_command_nor_its_environment(
    tmp_path,
):
    # The worker interpreter is never queried: the executor declares its runtime.
    first = executor_backend(tmp_path / "a", devices=True, python="/nonexistent/python")
    second = executor_backend(
        tmp_path / "b",
        command=["/opt/other"],
        environment=["SITE_TOKEN"],
        devices=True,
        python="/nonexistent/python",
    )
    assert first.revision == second.revision
    assert first.worker_runtime == RUNTIME
    service = Service(tmp_path / "a" / "gym.sqlite3", first)
    credentials = service.create_session(10**12, "executor-view")
    resources = service.call(credentials["agent_token"], "get_resources", {})
    text = json.dumps(resources)
    assert "/bin/true" not in text and "command" not in text
    node = resources["servers"][0] if "servers" in resources else resources
    assert '"backend": "operator_executor"' in text
    assert "NVIDIA H20-3e" in text and "Koala" not in text
    assert node is not None


def gym_session(tmp_path, monkeypatch, mode=None, environment=("SITE_TOKEN",)):
    script = tmp_path / "fake_executor.py"
    script.write_text(FAKE_EXECUTOR)
    log = tmp_path / "executor.json"
    monkeypatch.setenv("SITE_TOKEN", "for-the-executor")
    monkeypatch.setenv("OTHER_SECRET", "for-nobody")
    backend = executor_backend(
        tmp_path / "state",
        command=[sys.executable, str(script)],
        environment=[*environment, "FAKE_LOG", "FAKE_MODE"],
    )
    monkeypatch.setenv("FAKE_LOG", str(log))
    if mode:
        monkeypatch.setenv("FAKE_MODE", mode)
    service = Service(tmp_path / "state" / "gym.sqlite3", backend)
    credentials = service.create_session(10**12, "executor-trial")

    def call(operation, **params):
        return service.call(credentials["agent_token"], operation, params)

    candidate = call(
        "submit_candidate",
        package_base64=base64.b64encode(candidate_bytes()).decode(),
        idempotency_key="first",
    )
    return service, call, candidate, log


def test_an_executor_runs_scheduled_jobs_with_only_its_named_variables(
    tmp_path, monkeypatch
):
    service, call, candidate, log = gym_session(tmp_path, monkeypatch)
    quote = call(
        "quote_experiment",
        candidate_id=candidate["id"],
        profile_id="research-fixture",
        target_update=5,
    )
    job = call("run_experiment", quote_id=quote["id"], idempotency_key="run")
    assert Scheduler(service).work_once()
    state = call("get_job", job_id=job["id"])
    assert state["state"] == "succeeded", (
        service.backend.state_dir / (job["id"] + ".diagnostic.log")
    ).read_text()
    seen = json.loads(log.read_text())
    assert seen["cwd"] == str(service.backend.state_dir)
    assert seen["site_token"] == "for-the-executor" and seen["other"] is None
    # Windowed billing settles exactly the bound.
    budget = call("get_budget")
    assert int(budget["spent"]) == int(quote["bound"]) and budget["reserved"] == "0"
    result = call("get_result", job_id=job["id"])
    # The checkpoint artifact is the executor's own reference, handed back on continuation.
    resumed = call(
        "quote_experiment",
        candidate_id=candidate["id"],
        profile_id="research-fixture",
        target_update=8,
        checkpoint_id=result["checkpoint_id"],
    )
    job = call("run_experiment", quote_id=resumed["id"], idempotency_key="resume")
    assert Scheduler(service).work_once()
    assert call("get_job", job_id=job["id"])["state"] == "succeeded"
    assert json.loads(json.loads(log.read_text())["checkpoint"]) == {
        "weights": "elsewhere",
        "step": 5,
    }


def test_an_executor_learns_of_cancel_job_by_polling(tmp_path, monkeypatch):
    service, call, candidate, _ = gym_session(tmp_path, monkeypatch, "wait-for-cancel")
    quote = call(
        "quote_experiment",
        candidate_id=candidate["id"],
        profile_id="research-fixture",
        target_update=3,
    )
    job = call("run_experiment", quote_id=quote["id"], idempotency_key="run")
    worker = threading.Thread(target=Scheduler(service).work_once)
    worker.start()
    deadline = time.time() + 30
    while call("get_job", job_id=job["id"])["state"] != "running":
        assert time.time() < deadline
        time.sleep(0.05)
    call("cancel_job", job_id=job["id"], idempotency_key="stop")
    worker.join(30)
    assert call("get_job", job_id=job["id"])["state"] == "cancelled"
    budget = call("get_budget")
    assert budget["spent"] == "0" and budget["reserved"] == "0"
    assert [j["state"] for j in executor.jobs(service.backend.state_dir)] == [
        "cancelled"
    ]


def test_static_audits_stay_local_and_use_the_configured_isolation(tmp_path):
    backend = executor_backend(tmp_path, command=["/opt/site/run"])
    profile = backend.profiles["research-fixture"]
    assert backend.worker_command({"mode": "execute"}) == ["/opt/site/run"]
    audit = backend.worker_command({"mode": "inspect", "profile": profile})
    assert audit[-2:] == ["-m", "nexttransformer_gym.worker"]
    assert audit[0] != "/opt/site/run"


class Recorder:
    def __init__(self, cancel_at=None):
        self.log, self.cancel_at = [], cancel_at

    def intent(self, amount, label):
        self.log.append(("intent", label, int(amount)))
        return len(self.log) != self.cancel_at

    def charge(self, amount, label):
        self.log.append(("charge", label, int(amount)))


REPORT = {
    "initialization": "100",
    "forward": "5",
    "backward": "10",
    "optimizer": "1",
    "research_forward": "6",
    "research_backward": "12",
    "research_reduction": "2",
    "evaluation": "3",
    "research_evaluation": "4",
}


def test_windowed_charges_bill_ahead_and_settle_exactly_the_bound():
    assert accounting.update_cost(REPORT) == 21
    bound = accounting.training_bound(REPORT, 40, 2)
    assert bound == NativeBackend.bound(REPORT, "train", 40, {"dev_batches": 2})
    quote = {
        "operation": "train",
        "bound": str(bound),
        "start_update": 0,
        "target_update": 40,
        "cost_report": REPORT,
    }
    channel = Recorder()
    charges = executor.Charges(channel, quote, window=16)
    # The window slides with the observed progress: the start covers 16 updates, 10 done covers 26.
    assert charges.advance(0) and charges.advance(0)
    assert channel.log == [("intent", "executor-1", 100 + 16 * 21)]
    assert charges.advance(10)
    assert channel.log[1:] == [
        ("charge", "executor-1", 100 + 16 * 21),
        ("intent", "executor-2", 10 * 21),
    ]
    assert charges.advance(30) and channel.log[-1][2] == bound - (100 + 26 * 21)
    assert charges.settle_full()
    assert sum(a for kind, _, a in channel.log if kind == "charge") == bound
    # A stop settles only the started window; a short job is billed its bound at once.
    stopped = Recorder()
    early = executor.Charges(stopped, quote, window=16)
    early.advance(0)
    early.settle_current()
    assert sum(a for kind, _, a in stopped.log if kind == "charge") == 100 + 16 * 21
    short = Recorder()
    executor.Charges(short, {**quote, "target_update": 3}, window=16).advance(0)
    assert short.log == [("intent", "executor-1", bound)]
    refused = Recorder(cancel_at=1)
    assert not executor.Charges(refused, quote, window=16).advance(0)


def test_remote_workers_run_under_the_gyms_confinement():
    cpu = {"profile": {"device": "cpu"}}
    cuda = {"profile": {"device": "cuda:0"}}
    worker = ["py", "-m", "nexttransformer_gym.worker"]
    assert remote.worker_command("py", "/code", cpu, "none", "allow") == worker
    assert remote.worker_command("py", "/code", cpu, "none", "deny") == [
        "py",
        "-m",
        "nexttransformer_gym.netdeny",
        "--unix",
        "--",
        *worker,
    ]
    assert "--unix" not in remote.worker_command("py", "/code", cuda, "none", "deny")
    probe = remote.worker_command("py", "/code", cpu, "none", "deny", ["py", "-c", "x"])
    assert probe[-3:] == ["py", "-c", "x"]
    with pytest.raises(GymError):
        remote.worker_command("py", "/code", cpu, "none", "maybe")
    env = remote.worker_environment(
        "/code",
        "/work/home",
        {"device": "cuda:0"},
        {"research_dtype": "bfloat16", "research_compile": True},
        [{"id": UUIDS[0]}, {"id": UUIDS[1]}],
    )
    assert env["CUDA_VISIBLE_DEVICES"] == ",".join(UUIDS)
    assert env["NANOCHAT_DTYPE"] == "bfloat16" and env["TORCHDYNAMO_DISABLE"] == "0"
    assert env["HOME"] == "/work/home" and env["PYTHONPATH"].endswith("/code")
    with pytest.raises(GymError):
        remote.worker_environment("/code", "/h", {"device": "cuda:0"}, {}, None)
    assert remote.worker_identity({"quote": {"operation": "train"}}) == (
        {"user": 65534, "group": 65534, "extra_groups": []} if os.geteuid() == 0 else {}
    )


def test_the_runtime_record_is_the_one_the_backend_binds(tmp_path):
    profiles = json.loads((ROOT / "gym/examples/profiles.fixture.json").read_text())
    backend = NativeBackend(
        profiles, tmp_path, ROOT, execution_config={"isolation": ISOLATION}
    )
    assert remote.runtime() == backend.worker_runtime
    assert remote.valid_runtime(backend.worker_runtime)
    shown = subprocess.run(
        [sys.executable, "-m", "nexttransformer_gym.remote", "runtime"],
        capture_output=True,
        text=True,
        check=True,
        env=dict(
            os.environ, PYTHONPATH=os.pathsep.join([str(ROOT / "gym/src"), str(ROOT)])
        ),
    )
    assert json.loads(shown.stdout) == backend.worker_runtime


PROBE_SOCKETS = """
import os, socket
def t(family):
    try:
        socket.socket(family).close()
        return "ok"
    except OSError as error:
        return error.errno
read, write = os.pipe()
os.write(write, b"x")
print(t(socket.AF_INET), t(socket.AF_INET6), t(socket.AF_UNIX), os.read(read, 1) == b"x")
"""


@pytest.mark.parametrize(
    "device, expected", [("cuda:0", "13 13 ok True"), ("cpu", "13 13 13 True")]
)
def test_an_unsandboxed_worker_with_network_deny_opens_no_network_sockets(
    device, expected
):
    command = remote.worker_command(
        sys.executable,
        ROOT,
        {"profile": {"device": device}},
        "none",
        "deny",
        [sys.executable, "-c", PROBE_SOCKETS],
    )
    run = subprocess.run(
        command,
        capture_output=True,
        text=True,
        env=dict(os.environ, PYTHONPATH=str(ROOT / "gym/src")),
    )
    assert run.returncode == 0, run.stderr
    assert run.stdout.split() == expected.split()


PROBE_THREAD_SOCKETS = """
import socket, sys, threading
from nexttransformer_gym.netdeny import deny_sockets
ready, go, result = threading.Event(), threading.Event(), {}
def early():
    ready.set()
    go.wait()
    try:
        socket.socket(socket.AF_INET).close()
        result["early"] = "ok"
    except OSError as error:
        result["early"] = error.errno
thread = threading.Thread(target=early)
thread.start()
ready.wait()
listener = socket.socket()
listener.bind(("127.0.0.1", 0))
listener.listen()
client = socket.create_connection(listener.getsockname())
server, _ = listener.accept()
deny_sockets([socket.AF_INET, socket.AF_INET6], all_threads=sys.argv[1] == "all")
go.set()
thread.join()
client.sendall(b"x")
print(result["early"], server.recv(1) == b"x")
"""


@pytest.mark.parametrize("scope, expected", [("all", "13 True"), ("thread", "ok True")])
def test_the_all_thread_socket_filter_binds_threads_started_before_it(scope, expected):
    """A process group starts threads before the filter; only the all-thread filter binds them, and connections
    opened before it keep working."""
    run = subprocess.run(
        [sys.executable, "-c", PROBE_THREAD_SOCKETS, scope],
        capture_output=True,
        text=True,
        env=dict(os.environ, PYTHONPATH=str(ROOT / "gym/src")),
    )
    assert run.returncode == 0, run.stderr
    assert run.stdout.split() == expected.split()


def free_port():
    with socket.socket() as handle:
        handle.bind(("127.0.0.1", 0))
        return handle.getsockname()[1]


def mesh_node(rank, port, out):
    spec = mesh.spec(
        {
            "node": rank,
            "nodes": 2,
            "master": "127.0.0.1",
            "port": port,
            "timeout": 60,
            "network": "deny",
            "backend": "gloo",
        }
    )
    torch, dist = mesh.prepare(spec, sizes=(1, 1 << 10))
    value = torch.ones(4)
    dist.all_reduce(value)  # the group's connections survive the filter
    try:
        socket.socket(socket.AF_INET).close()
        opened = "ok"
    except OSError as error:
        opened = error.errno
    Path(out, f"node-{rank}.json").write_text(
        json.dumps({"sum": float(value[0]), "socket": opened})
    )
    dist.destroy_process_group()


def test_a_mesh_forms_before_the_network_filter_and_keeps_its_connections(tmp_path):
    multiprocessing.spawn(mesh_node, args=(free_port(), str(tmp_path)), nprocs=2)
    for rank in range(2):
        seen = json.loads((tmp_path / f"node-{rank}.json").read_text())
        assert seen == {"sum": 2.0, "socket": 13}


def test_a_mesh_description_is_checked():
    good = {
        "node": 1,
        "nodes": 2,
        "master": "h",
        "port": 1,
        "timeout": 5,
        "network": "deny",
    }
    assert mesh.spec(json.dumps(good))["backend"] == "nccl"
    for bad in (
        {**good, "node": 2},
        {**good, "nodes": 1},
        {**good, "network": "open"},
        {**good, "extra": 1},
        {k: v for k, v in good.items() if k != "master"},
    ):
        with pytest.raises(ValueError):
            mesh.spec(bad)
