"""Multi-node execution: an operator executor that runs each scheduled job on one or more machines (nodes).

The Gym runs its numerical jobs on its own machine unless the execution configuration names an operator executor
(DESIGN.md, "Operator executors"). This module is that executor for a group of machines. A launcher says how the
nodes of a job start and how node 0's output comes back:

* `local`: node processes on this machine (`nodes` of them);
* `ssh`: node k on `hosts[k]`, over ssh;
* a cluster plugin subclasses `Launcher` and runs `relay(launcher)` as its own executor command (gym/koala does).

Configuration (`nodes.json`, defaults in `DEFAULTS`):

    {"launcher": "ssh", "hosts": ["gpu-a", "gpu-b"], "gpus_per_node": 8,
     "python": "/opt/gym/.venv/bin/python", "job_root": "/shared/gym-jobs",
     "isolation": "none", "network": "deny", "environment": {"NCCL_SOCKET_IFNAME": "eth0"}}

`job_root` is a directory at the same path on this machine and on every node (a shared filesystem): each job gets a
directory there with its description, a copy of the platform, its checkpoint and logs. `python` is the numerical
interpreter on every node (the controller's own interpreter by default). A host is an ssh destination, or
{"ssh": destination, "address": the address other nodes reach it at}.

    python -m nexttransformer_gym.nodes configure nodes.json --out execution.json [--execution base.json]

records the nodes' runtime and GPUs and writes the Gym execution configuration whose executor runs
`python -m nexttransformer_gym.nodes relay nodes.json`; `base.json` supplies its other fields (memory budget,
microbatch, precision, the static audits' isolation). Start the Gym with it as with any execution configuration.

A job. The relay (the executor command; it runs in the Gym state as the controller's user) reads the Gym request,
writes the job description and the platform into the job directory, starts the nodes and speaks the worker protocol
for them: it bills ahead of node 0's observed progress (`executor.Charges`), polls for cancel_job, stops the nodes
when the Gym cancels or the profile timeout passes, and returns node 0's result and, for training, a pointer to the
job's checkpoint (the Gym retains the pointer; a later job continues from the file it names). A research group larger
than `gpus_per_node` cards spans whole nodes, node k taking cards k*gpus_per_node.. of the group; CPU jobs (fixtures)
span `cpu_nodes` nodes over gloo. Final training is dispatched separately and never runs here.

Test status. The GPU path of this executor has run only through the Koala launcher (gym/koala): one node, and 4
nodes x 8 H20 for one research job. The `local` and `ssh` launchers have been tested only with CPU nodes (gloo) on
one machine, `ssh` with a stand-in that runs the command locally. With GPUs on bare-metal machines, without Koala,
they are untested: real ssh and shared storage, the cards' mapping by identity and the busy-card refusal, and NCCL
between machines. Check a bounded job on such machines before relying on it.

A node (`python -m nexttransformer_gym.nodes node job.json --node k --master address`) checks its interpreter
against the declared runtime and the input checkpoint against its hash, takes its share of the group's cards (a card
that runs another compute process refuses the job before work) and runs the Gym worker with the Gym's own identity,
environment and confinement (`remote`). On a group of nodes each node first checks the mesh (`mesh check`, before
node 0 reports ready, so a mesh that does not form charges nothing) and then runs the mesh worker. Node 0
acknowledges the worker's operations within the job's reservation, reports, and alone writes the checkpoint; the other
nodes leave markers (`research/`) and stop their worker when a peer fails or node 0 is done. Every node stops its
worker when the relay leaves a `stop` marker in the job directory.
"""

import hashlib
import json
import os
import shlex
import shutil
import signal
import socket
import subprocess
import sys
import threading
import time
from pathlib import Path

from . import remote as gym_remote
from .common import GymError

MARK = "GYM_NODE "
GYM = "GYM_EVENT "
CHUNK = 3000
UNKNOWN_EXIT = 3
PEER_GRACE_SECONDS = 60
PLATFORM_FILES = ("pyproject.toml", "uv.lock", "README.md", ".python-version")
PLATFORM_DIRS = ("nexttransformer", "nanochat")
# Caches and work files never ship (a code bundle counts its files against the publisher's preview).
IGNORE = shutil.ignore_patterns(
    "__pycache__",
    "*.pyc",
    ".git",
    ".venv",
    "node_modules",
    ".agent-work",
    ".ruff_cache",
    ".pytest_cache",
    ".mypy_cache",
)
# Library and interconnect settings a node passes from its own environment to its worker.
PASSTHROUGH = (
    "LD_LIBRARY_PATH",
    "NCCL_SOCKET_IFNAME",
    "NCCL_IB_HCA",
    "NCCL_IB_GID_INDEX",
    "NCCL_IB_DISABLE",
    "NCCL_NET_GDR_LEVEL",
    "NCCL_DEBUG",
    "GLOO_SOCKET_IFNAME",
)
DEFAULTS = {
    "schema_version": "gym.nodes.v1",
    "launcher": "local",
    # local: node processes on this machine.
    "nodes": 1,
    # ssh: node k runs on hosts[k].
    "hosts": [],
    "ssh": ["ssh", "-o", "BatchMode=yes"],
    "gpus_per_node": 8,
    # Nodes a CPU job spans (gloo); more than one only to exercise the multi-node path without GPUs.
    "cpu_nodes": 1,
    "python": None,
    "job_root": None,
    # Node-local scratch (environment, temporary files); by default inside the job directory.
    "work_root": None,
    "environment": {},
    # How nodes confine candidate code (written into the executor block by `configure`). A job spanning nodes
    # needs "none": the sandboxes give the worker a network namespace without the other nodes.
    "isolation": "bwrap",
    "network": "deny",
    # Node 0 serves the mesh on this port and checks it on the next one; 0 picks free ports (local only).
    "master_port": 29531,
    "mesh_timeout_seconds": 3600,
    "rendezvous_seconds": 1800,
    "poll_seconds": 2,
    "margin_seconds": 60,
    "window": 16,
}


def sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def utc(seconds=None):
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(seconds))


def is_cuda(request):
    return str(request["profile"].get("device", "cpu")).startswith("cuda")


def allocated_devices(allocation):
    """The cards of a Gym GPU allocation: one card, or a group (`devices`); no allocation is one empty card."""
    if not allocation:
        return [{}]
    return list(allocation.get("devices") or [allocation])


def research_topology(count, per):
    """(nodes, GPUs per node) of a group of `count` cards: more than `per` cards span whole nodes."""
    if count <= per:
        return 1, count
    if count % per:
        raise GymError(
            "resource_unavailable",
            f"a group of {count} GPUs does not fill nodes of {per}",
        )
    return count // per, per


# --------------------------------------------------------------------------------------------- events


class Emitter:
    """One numbered event per stdout line; node 0's lines are what the relay reads."""

    def __init__(self, seq=0, mark=MARK):
        self.seq, self.mark = seq, mark

    def __call__(self, kind, **value):
        self.seq += 1
        event = dict(seq=self.seq, kind=kind, t=round(time.time(), 3), **value)
        print(
            self.mark + json.dumps(event, allow_nan=False, separators=(",", ":")),
            flush=True,
        )

    def blob(self, name, value):
        text = json.dumps(value, allow_nan=False, separators=(",", ":"))
        parts = [text[i : i + CHUNK] for i in range(0, len(text), CHUNK)] or [""]
        for i, part in enumerate(parts):
            self("chunk", name=name, i=i, data=part)
        return len(parts)


def events(lines, mark=MARK):
    found = []
    for line in lines or ():
        index = line.find(mark)
        if index < 0:
            continue
        try:
            found.append(json.loads(line[index + len(mark) :]))
        except ValueError:
            continue
    return found


# ----------------------------------------------------------------------------------------------- relay


class Busy(RuntimeError):
    """The launcher's capacity is in use for now (a cluster quota, for example); the relay retries."""


class Launcher:
    """How the nodes of a job start and how node 0's output returns. A subclass implements the transport; `relay`
    does everything else."""

    name = "nodes"
    # The prefix of node 0's event lines, the relay's record in the Gym state, and the Gym-retained pointer's schema.
    mark = MARK
    journal_file = "nodes-jobs.jsonl"
    checkpoint_schema = "nexttransformer_gym.nodes.checkpoint.v1"
    # The relay ends this long before the profile timeout; it reads node 0's output this often; it retries a busy
    # launcher this often; it bills this many updates ahead of the observed progress.
    margin_seconds = 60
    poll_seconds = 2.0
    busy_seconds = 120.0
    window = 16

    def plan(self, request, tag):
        """(payload, snapshot): the job description the nodes read, for the Gym `request` (`tag` names the Gym
        state), and a directory to ship with it or None. GymError for a job this executor cannot run."""
        raise NotImplementedError

    def prepare(self, payload, request, snapshot):
        """Put the job where its nodes read it, once; returns a record of it (journalled, passed to `start`)."""
        raise NotImplementedError

    def start(self, record, nodes, gpus):
        """Start the job's nodes; returns the job's id. Raises Busy while the capacity is in use."""
        raise NotImplementedError

    def lines(self, job):
        """Node 0's output so far, or None when it cannot be read now."""
        raise NotImplementedError

    def ended(self, job):
        """A status once every node of the job has ended, else None."""
        raise NotImplementedError

    def stop(self, job):
        """Stop every node of the job."""
        raise NotImplementedError


def checkpoint_input(request, schema):
    """The input checkpoint of a continuation: the file the Gym-retained pointer names, and its hash."""
    if not request.get("checkpoint"):
        return None
    pointer = json.loads(Path(request["checkpoint"]).read_text())
    if pointer.get("schema") != schema:
        raise GymError(
            "incompatible_checkpoint",
            "the retained checkpoint is not this executor's checkpoint pointer",
        )
    return {"path": pointer["remote_path"], "sha256": pointer["sha256"]}


def usage(first, last, nodes, gpus):
    """Node time from the first node event to `last` (node clock), and the GPU-hours it held."""
    if first is None or last is None:
        return {}
    seconds = max(0.0, float(last) - float(first))
    return {
        "pod_started": utc(first),
        "pod_seconds": round(seconds, 1),
        "gpu_hours": round(seconds * int(nodes) * int(gpus) / 3600, 4),
    }


def relay(launcher):
    """The executor command: one Gym job on the launcher's nodes, over the Gym's worker channel."""
    from .executor import Channel, Charges

    channel, state = Channel(), Path.cwd()
    request = channel.request()
    quote, profile = request["quote"], request["profile"]
    operation = quote["operation"]
    name = launcher.name

    def note(message):
        print(f"{name} relay: {message}", file=sys.stderr, flush=True)

    def journal(**entry):
        with open(state / launcher.journal_file, "a") as handle:
            handle.write(json.dumps({"time": utc(), **entry}) + "\n")

    def fail(code, detail=None):
        channel.failed(code)
        if detail:
            note(f"{code}: {detail}")

    tag = hashlib.sha256(str(state).encode()).hexdigest()[:12]
    deadline = time.monotonic() + max(
        60, int(profile["timeout_sec"]) - int(launcher.margin_seconds)
    )
    try:
        payload, snapshot = launcher.plan(request, tag)
    except Exception as error:
        fail(
            getattr(error, "code", "resource_unavailable"),
            f"cannot describe the {name} job: {error}",
        )
        return 0
    nodes, gpus = payload["nodes"], payload["gpus_per_node"]

    class Stopped(Exception):
        pass

    def stopped(signum, frame):
        raise Stopped()

    # No job exists until start returns, so a SIGTERM here ends the relay with a certain failure.
    signal.signal(signal.SIGTERM, stopped)
    try:
        record = launcher.prepare(payload, request, snapshot)
        waited = 0.0
        while True:
            try:
                job = launcher.start(record, nodes, gpus)
                break
            except Busy as error:
                pause = float(launcher.busy_seconds)
                if channel.poll():
                    note(f"cancelled while waiting {waited:.0f} s for {name} quota")
                    channel.cancelled()
                    return 0
                if time.monotonic() + pause > deadline:
                    raise RuntimeError(
                        f"the {name} quota stayed full for {waited:.0f} s, until the profile "
                        f"timeout: {error}"
                    ) from None
                note(f"{name} quota is full; retrying in {pause:.0f} s: {error}")
                time.sleep(pause)
                waited += pause
    except Stopped:
        fail(
            "resource_unavailable",
            f"the relay was terminated before a {name} job was submitted",
        )
        return 0
    except Exception as error:
        fail("resource_unavailable", f"{name} submission failed: {error}")
        return 0
    signal.signal(signal.SIGTERM, signal.SIG_DFL)
    if waited:
        note(f"submitted after waiting {waited:.0f} s for {name} quota")
    journal(
        quote_id=quote["id"],
        operation=operation,
        job=job,
        **record,
        nodes=nodes,
        gpus_per_node=gpus,
        event="submitted",
    )
    note(f"submitted {job} ({nodes}x{gpus} GPU) for {operation} quote {quote['id']}")

    class Terminated(Exception):
        pass

    def terminate(signum, frame):
        raise Terminated()

    signal.signal(signal.SIGTERM, terminate)
    charges = Charges(channel, quote, window=launcher.window)
    started, seen, chunks = False, 0, {}
    # Node clock: the first event seen (boot) and the latest. The relay reads the output from the start, so it
    # sees the boot event before a long run scrolls it away.
    span = {"first": None, "last": None}

    def held(end=None):
        return usage(span["first"], span["last"] if end is None else end, nodes, gpus)

    def unknown(reason, end=None):
        journal(
            quote_id=quote["id"], job=job, event="unknown", reason=reason, **held(end)
        )
        note(
            f"outcome unknown ({reason}); the reservation stays for reconcile; {name} job {job}"
        )
        return UNKNOWN_EXIT

    def stop(reason):
        launcher.stop(job)
        journal(
            quote_id=quote["id"],
            job=job,
            event="stopped",
            reason=reason,
            **held(time.time() if started else None),
        )

    def finish(done):
        outcome = done.get("outcome")
        journal(
            quote_id=quote["id"],
            job=job,
            event="ended",
            outcome=outcome,
            error_code=done.get("error_code"),
            spent=done.get("spent"),
            **held(done.get("t")),
        )
        if outcome == "pre_failure":
            fail(done.get("error_code", "resource_unavailable"), done.get("detail"))
            return 0
        if not started:
            return unknown("a node reported an outcome without starting its worker")
        if outcome == "succeeded":
            parts = chunks.get("result", {})
            if len(parts) != done.get("result_chunks"):
                return unknown("the result was not fully logged")
            result = json.loads("".join(parts[i] for i in range(len(parts))))
            if operation == "train":
                saved = done.get("checkpoint") or {}
                pointer = {
                    "schema": launcher.checkpoint_schema,
                    "job": job,
                    "remote_path": saved.get("path"),
                    "sha256": saved.get("sha256"),
                    "size_bytes": saved.get("size_bytes"),
                    "step": result.get("step"),
                    "candidate_hash": quote.get("candidate_hash"),
                    "profile_hash": quote.get("profile_hash"),
                }
                Path(request["output"]).write_text(json.dumps(pointer, indent=1))
            if not charges.settle_full():
                channel.cancelled()
                return 0
            channel.result(result)
            return 0
        if outcome == "failed":
            charges.settle_current()
            fail(done.get("error_code", "infrastructure_failure"), done.get("detail"))
            return 0
        return unknown(done.get("detail") or f"node outcome {outcome}")

    def read():
        """Handle node 0's new events; returns an exit code once the job is settled."""
        nonlocal started, seen
        lines = launcher.lines(job)
        for event in events(lines, launcher.mark):
            if event.get("seq", 0) <= seen:
                continue
            seen = event["seq"]
            kind = event.get("kind")
            if isinstance(event.get("t"), (int, float)):
                first, last = span["first"], span["last"]
                span["first"] = event["t"] if first is None else min(first, event["t"])
                span["last"] = event["t"] if last is None else max(last, event["t"])
            if kind == "ready" and not started:
                if not charges.advance(0):
                    stop("cancelled before work")
                    channel.cancelled()
                    return 0
                started = True
                journal(
                    quote_id=quote["id"],
                    job=job,
                    event="running",
                    **(
                        {
                            "pod_started": utc(span["first"]),
                            "pod_started_t": round(span["first"], 3),
                        }
                        if span["first"] is not None
                        else {}
                    ),
                )
            elif kind == "runtime":
                channel.runtime(event["value"])
            elif kind == "progress":
                step = int(event["step"])
                channel.progress(step)
                if started and not charges.advance(step - charges.start_update):
                    stop("cancel_job")
                    channel.cancelled()
                    return 0
            elif kind == "chunk":
                chunks.setdefault(event["name"], {})[event["i"]] = event["data"]
            elif kind == "restarted":
                launcher.stop(job)
                return unknown("a node restarted after it had begun")
            elif kind == "done":
                return finish(event)
        return None if lines is not None else "unreadable"

    try:
        while True:
            outcome = read()
            if isinstance(outcome, int):
                return outcome
            if channel.poll():
                stop("cancel_job")
                if started:
                    charges.settle_current()
                channel.cancelled()
                return 0
            status = launcher.ended(job)
            if status:
                outcome = read()
                if isinstance(outcome, int):
                    return outcome
                if not started and outcome is None and seen > 0:
                    fail(
                        "resource_unavailable",
                        f"{name} job {job} ended ({status}) before its worker started",
                    )
                    journal(
                        quote_id=quote["id"],
                        job=job,
                        event="ended",
                        outcome="pre_failure",
                        status=status,
                        **held(),
                    )
                    return 0
                return unknown(f"{name} job ended ({status}) without a logged outcome")
            if time.monotonic() > deadline:
                stop("profile timeout")
                if started:
                    charges.settle_current()
                fail(
                    "infrastructure_failure",
                    f"{name} job {job} exceeded the profile timeout",
                )
                return 0
            time.sleep(float(launcher.poll_seconds))
    except Terminated:
        launcher.stop(job)
        return unknown(
            f"the relay was terminated; the {name} job was stopped",
            end=time.time() if span["first"] is not None else None,
        )


# ------------------------------------------------------------------------------------------------ nodes


def worker_env(payload, code, work, devices=None):
    """The Gym's worker environment on a node, plus the node's own library and interconnect settings."""
    (work / "home").mkdir(parents=True, exist_ok=True)
    env = gym_remote.worker_environment(
        code,
        work / "home",
        payload.get("request", {}).get("profile"),
        {
            "research_dtype": payload.get("research_dtype", "float32"),
            "research_compile": payload.get("research_compile", False),
        },
        devices,
    )
    for key in PASSTHROUGH:
        if key in os.environ:
            env[key] = os.environ[key]
    return env


def local_gpus(count):
    """This machine's GPUs in index order, as the Gym's device records."""
    try:
        return gym_remote.local_devices(count)
    except Exception as error:
        raise GymError("resource_unavailable", str(error)) from None


def node_cards(payload, wanted):
    """This node's GPUs for `wanted` (its share of the job's cards), in order.

    With `device_ids` "exact" the cards are this machine's own, by identity, and a card that runs another compute
    process refuses the job before work. Otherwise (a cluster that names cards virtually) the node takes all of its
    GPUs in index order."""
    if payload.get("device_ids") != "exact":
        return local_gpus(len(wanted))
    from . import gpu

    seen = gpu.discover()
    if seen.get("status") != "ok":
        raise GymError(
            "resource_unavailable",
            f"this node cannot observe its GPUs: {seen.get('reason')}",
        )
    found = {card["id"]: card for card in seen["gpus"]}
    missing = [card.get("id") for card in wanted if card.get("id") not in found]
    if missing:
        raise GymError("resource_unavailable", f"this node has no GPU {missing[0]}")
    cards = [found[card["id"]] for card in wanted]
    busy = [card["id"] for card in cards if card.get("compute_process_count")]
    if busy:
        raise GymError(
            "resource_unavailable", f"GPU {busy[0]} runs another compute process"
        )
    return cards


def assign_gpus(allocation, cards):
    """Replace an allocation's cards with this node's cards, in order, keeping each memory limit.

    Returns the Gym's `gpu` record (one card, or the first card with `devices` for a group) and the
    CUDA_VISIBLE_DEVICES value; the worker's cuda:i is the group's i-th card."""
    wanted = allocated_devices(allocation)
    if len(cards) != len(wanted):
        raise GymError(
            "resource_unavailable",
            f"the node has {len(cards)} GPUs for an allocation of {len(wanted)}",
        )
    devices = [
        dict(card, memory_limit_bytes=virtual.get("memory_limit_bytes"))
        for card, virtual in zip(cards, wanted)
    ]
    record = (
        dict(devices[0], devices=devices)
        if (allocation or {}).get("devices")
        else devices[0]
    )
    return record, ",".join(card["uuid"] for card in devices)


def grant(path, identity):
    """Hand a directory the node (root) created to the worker identity; on storage that refuses ownership changes,
    make it world-writable with the sticky bit instead."""
    path.mkdir(parents=True, exist_ok=True)
    try:
        os.chown(path, identity["user"], identity["group"])
    except PermissionError:
        os.chmod(path, 0o1777)


def reach(path, root):
    """Let other users traverse, not list, every directory from root down to path.

    A root node may create directories with umask 0077, so a worker running as nobody could not reach a workspace
    it owns. Directories outside root are left alone."""
    path, root = Path(path), Path(root)
    for directory in [path, *path.parents]:
        if directory != root and root not in directory.parents:
            continue
        mode = directory.stat().st_mode
        if not mode & 0o001:
            os.chmod(directory, (mode & 0o7777) | 0o001)


def stop_identity(identity):
    """Kill every process still running as the worker identity, after the worker itself has exited.

    Without a PID namespace a worker's detached children outlive it; they must not touch its outputs while the node
    hashes and reports them. A child running as that identity sends kill(-1), which reaches exactly the processes
    the identity may signal. Only a root node does this, and never for root."""
    if not identity or os.geteuid() != 0 or not identity.get("user"):
        return
    for _ in range(
        2
    ):  # a second pass catches children forked while the first was signalling
        subprocess.run(
            [
                sys.executable,
                "-c",
                "import os, signal\ntry:\n    os.kill(-1, signal.SIGKILL)\n"
                "except ProcessLookupError:\n    pass",
            ],
            user=identity["user"],
            group=identity["group"],
            extra_groups=[],
            check=False,
            timeout=60,
        )


def drive(
    command,
    env,
    request,
    credit,
    emit,
    cwd,
    stderr_path,
    progress_every=60.0,
    identity=None,
    started=None,
):
    """Run one Gym worker and acknowledge its operations within the job's reservation."""
    with open(stderr_path, "a") as stderr:
        process = subprocess.Popen(
            command,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=stderr,
            text=True,
            bufsize=1,
            env=env,
            cwd=cwd,
            **(identity or {}),
        )
        if started is not None:
            started(process)

        def send(value):
            process.stdin.write(json.dumps(value) + "\n")
            process.stdin.flush()

        send(request)
        spent, pending, result, failure, refused = 0, None, None, None, False
        step, shown, last = None, None, 0.0
        for line in process.stdout:
            if not line.startswith(GYM):
                continue
            event = json.loads(line[len(GYM) :])
            kind = event.get("kind")
            if kind == "intent":
                amount = int(event["amount"])
                if pending is not None or amount < 0 or spent + amount > credit:
                    refused = True
                    send({"cancel": True})
                    continue
                pending = event
                send({"ack": event["label"]})
            elif kind == "charge":
                if pending is None or (event["label"], event["amount"]) != (
                    pending["label"],
                    pending["amount"],
                ):
                    process.kill()
                    process.wait()
                    return {
                        "outcome": "unknown",
                        "detail": "worker charge without a matching intent",
                        "spent": spent,
                    }
                spent += int(event["amount"])
                pending = None
                send({"ack": event["label"]})
            elif kind == "runtime":
                emit("runtime", value=event["value"])
            elif kind == "progress":
                step = event["step"]
                if time.time() - last >= progress_every:
                    emit("progress", step=step)
                    shown, last = step, time.time()
            elif kind == "result":
                result = event["value"]
            elif kind == "failed":
                failure = event.get("error_code", "infrastructure_failure")
            elif kind == "cancelled":
                refused = True
        code = process.wait()
    if step is not None and step != shown:
        emit("progress", step=step)
    if code == 0 and pending is None:
        if failure:
            return {"outcome": "failed", "error_code": failure, "spent": spent}
        if refused:
            return {
                "outcome": "failed",
                "error_code": "insufficient_budget",
                "spent": spent,
            }
        if result is not None:
            return {"outcome": "succeeded", "result": result, "spent": spent}
    return {
        "outcome": "unknown",
        "detail": f"worker exited {code} without a settled outcome",
        "spent": spent,
    }


def explain(outcome, path):
    """Add the worker's stderr tail to an unsuccessful outcome."""
    if outcome["outcome"] != "succeeded" and path.exists():
        tail = path.read_text(errors="replace")[-1500:]
        outcome["detail"] = "\n".join(filter(None, [outcome.get("detail"), tail]))[
            -3000:
        ]


def keep_log(path, remote, label):
    """Copy a node-local log next to the job's shared files: node-local disk may go with the node."""
    if path.exists():
        target = remote / "logs" / f"{label}-{path.name}"
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(path, target)


def report(emit, payload, outcome, remote, publish=None):
    """Node 0's outcome as its last events: the result in chunks, then `done` with the checkpoint's identity.

    `publish(checkpoint)` (a launcher without shared storage) uploads the job's files before `done` and returns the
    checkpoint record as later jobs find it; it is called with None, best effort, for an unsuccessful job."""
    request = payload["request"]
    if outcome["outcome"] == "succeeded":
        result = outcome["result"]
        (remote / "result.json").write_text(json.dumps(result, indent=1))
        checkpoint = None
        if request["quote"]["operation"] == "train":
            path = Path(request["output"])
            checkpoint = {
                "path": str(path),
                "sha256": sha256(path),
                "size_bytes": path.stat().st_size,
            }
        if publish is not None:
            try:
                checkpoint = publish(checkpoint)
            except Exception as error:
                emit(
                    "done",
                    outcome="unknown",
                    detail=f"the job's files were not uploaded: {error}"[-1500:],
                    spent=str(outcome["spent"]),
                )
                return 1
        parts = emit.blob("result", result)
        emit(
            "done",
            outcome="succeeded",
            result_chunks=parts,
            checkpoint=checkpoint,
            spent=str(outcome["spent"]),
        )
        return 0
    if publish is not None:
        try:  # logs of an unsuccessful job, best effort
            publish(None)
        except Exception:
            pass
    emit(
        "done",
        outcome=outcome["outcome"],
        error_code=outcome.get("error_code"),
        detail=outcome.get("detail"),
        spent=str(outcome.get("spent", 0)),
    )
    return 0 if outcome["outcome"] == "failed" else 1


def check_runtime(payload):
    """Before any work: this node's interpreter is the declared runtime."""
    found = gym_remote.runtime(sys.executable)
    if found != payload["runtime"]:
        raise GymError(
            "resource_unavailable",
            f"node runtime {found} differs from the recorded {payload['runtime']}",
        )


def check_checkpoint(payload, node):
    """Before any work: node 0 finds the input checkpoint intact (every node reads the same file)."""
    checkpoint = payload.get("checkpoint")
    if checkpoint and node == 0 and sha256(checkpoint["path"]) != checkpoint["sha256"]:
        raise GymError(
            "incompatible_checkpoint",
            "the stored checkpoint differs from its recorded hash",
        )


def pre_failure(payload, node, emit, error):
    """A node that cannot start: a marker for the other nodes of a group, and `done` (charged nothing)."""
    if int(payload["nodes"]) > 1:
        handoff = Path(payload["remote"]) / "research"
        handoff.mkdir(parents=True, exist_ok=True)
        (handoff / f"failed-{node}").write_text(repr(error)[-1500:])
    emit(
        "done",
        outcome="pre_failure",
        error_code=getattr(error, "code", "resource_unavailable"),
        detail=(str(error) if hasattr(error, "code") else repr(error))[-1500:],
    )
    return 1


def run(payload, node, master, emit, code, work, remote, publish=None):
    """One node of a job, after its checks: the Gym worker on one node, or this node's part of a group."""
    (work / "tmp").mkdir(parents=True, exist_ok=True)
    if int(payload["nodes"]) > 1:
        return research_node(payload, node, master, emit, code, work, remote, publish)
    if node != 0:
        return 0
    return single_node(payload, emit, code, work, remote, publish)


def stop_requested(remote):
    return (Path(remote) / "stop").exists()


def single_node(payload, emit, code, work, remote, publish=None):
    """A job on one node: the Gym worker under the Gym's confinement."""
    request = payload["request"]
    try:
        # The Gym's own rule: a root runner runs every non-official worker as nobody.
        identity = gym_remote.worker_identity(request)
        devices = None
        if is_cuda(request):
            wanted = allocated_devices(request.get("gpu"))
            request["gpu"], _ = assign_gpus(
                request.get("gpu"), node_cards(payload, wanted)
            )
            devices = allocated_devices(request["gpu"])
        env = worker_env(payload, code, work, devices)
        workspace = Path(request["output"]).parent
        workspace.mkdir(parents=True, exist_ok=True)
        if identity:
            for path in (work / "tmp", work / "home", workspace):
                grant(path, identity)
            reach(workspace, payload.get("remote_root") or remote.parent.parent)
            reach(work / "tmp", work)
        if payload["isolation"] != "none":
            gym_remote.preflight(payload["isolation"])
        worker = gym_remote.worker_command(
            sys.executable,
            code,
            request,
            payload["isolation"],
            payload.get("network", "deny"),
        )
    except Exception as error:
        emit(
            "done",
            outcome="pre_failure",
            error_code=getattr(error, "code", "resource_unavailable"),
            detail=repr(error)[-1500:],
        )
        return 1
    emit(
        "ready",
        isolation=payload["isolation"],
        uid=identity.get("user", os.geteuid()),
        network="allowed"
        if payload["isolation"] == "none" and payload.get("network") == "allow"
        else "denied",
    )
    stop = threading.Event()

    def watch(process):
        """Stop the worker when the relay leaves its stop marker."""

        def loop():
            while not stop.wait(2):
                if stop_requested(remote):
                    process.kill()
                    stop_identity(identity)
                    return

        threading.Thread(target=loop, daemon=True).start()

    outcome = drive(
        worker,
        env,
        request,
        int(payload["credit"]),
        emit,
        work / "tmp",
        work / "worker.stderr",
        progress_every=15.0,
        identity=identity,
        started=watch,
    )
    stop.set()
    stop_identity(identity)
    explain(outcome, work / "worker.stderr")
    keep_log(work / "worker.stderr", remote, "node-0")
    return report(emit, payload, outcome, remote, publish)


def mesh_spec(payload, node, master, port):
    """The trusted description of a node's place in the job, for its research processes (never candidate input)."""
    value = {
        "node": node,
        "nodes": int(payload["nodes"]),
        "master": master,
        "port": port,
        "timeout": int(payload.get("mesh_timeout_seconds", 3600)),
        "network": payload.get("network", "deny"),
    }
    if not is_cuda(payload["request"]):
        value["backend"] = "gloo"
    return json.dumps(value)


def mesh_result(stdout):
    """The mesh check's JSON line; NCCL may log around it (NCCL_DEBUG)."""
    for line in reversed(stdout.splitlines()):
        if line.startswith("{"):
            try:
                return json.loads(line)
            except ValueError:
                continue
    return None


def research_node(payload, node, master, emit, code, work, remote, publish=None):
    """One node of a multi-node research job: its share of the GPU group, a worker in the nodes' group."""
    from . import mesh

    request = payload["request"]
    nodes, per = int(payload["nodes"]), int(payload["gpus_per_node"])
    lead = node == 0
    handoff = remote / "research"
    handoff.mkdir(parents=True, exist_ok=True)

    def mark(name, text=""):
        tmp = handoff / f".{name}-{os.getpid()}"
        tmp.write_text(text)
        tmp.replace(handoff / name)

    def peers_failed():
        return sorted(
            p.name for p in handoff.glob("failed-*") if p.name != f"failed-{node}"
        )

    def await_lead(seconds):
        """Another node stays up until node 0 has logged the outcome: on some clusters a job ends when any of its
        nodes does, and node 0 must still report (a pre-failure, a failure, or the result)."""
        deadline = time.time() + seconds
        while (
            not (handoff / "done").exists()
            and not stop_requested(remote)
            and time.time() < deadline
        ):
            time.sleep(5)

    # Only node 0's output reaches the relay; the others record their events for people reading the nodes' logs.
    report_to = (
        emit
        if lead
        else lambda kind, **fields: emit("satellite", node=node, event=kind, **fields)
    )
    identity = {}
    try:
        if payload["isolation"] != "none":
            raise GymError(
                "resource_unavailable",
                "a job spanning nodes needs isolation none: the sandboxes give the worker a network namespace "
                "without the other nodes",
            )
        identity = gym_remote.worker_identity(request)
        devices = None
        if is_cuda(request):
            wanted = allocated_devices(request.get("gpu"))
            if len(wanted) != nodes * per:
                raise GymError(
                    "resource_unavailable",
                    f"a group of {len(wanted)} GPUs for {nodes} nodes of {per}",
                )
            share = wanted[node * per : (node + 1) * per]
            request["gpu"], _ = assign_gpus(
                dict(request["gpu"], devices=share), node_cards(payload, share)
            )
            devices = allocated_devices(request["gpu"])
        env = worker_env(payload, code, work, devices)
        env.update({k: v for k, v in os.environ.items() if k.startswith("NCCL_")})
        if not lead:  # node 0 alone writes the job's checkpoint
            request["output"] = str(work / "tmp" / "follower" / "checkpoint.pt")
        workspace = Path(request["output"]).parent
        workspace.mkdir(parents=True, exist_ok=True)
        if identity:
            for path in (work / "tmp", work / "home", workspace):
                grant(path, identity)
            reach(workspace, payload.get("remote_root") or remote.parent.parent)
            reach(work / "tmp", work)
        port = int(payload["master_port"])
        # Before node 0 reports ready (a broken mesh charges nothing): the nodes meet as the worker identity. A
        # node that failed before the check leaves a marker, and the others stop waiting for it.
        check = subprocess.Popen(
            [sys.executable, "-m", "nexttransformer_gym.mesh", "check"],
            env=dict(env, **{mesh.ENV: mesh_spec(payload, node, master, port + 1)}),
            cwd=work / "tmp",
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            **identity,
        )
        output = []
        reader = threading.Thread(
            target=lambda: output.append(check.stdout.read()), daemon=True
        )
        reader.start()
        deadline = time.time() + float(payload["rendezvous_seconds"]) + 600
        while check.poll() is None:
            failed = peers_failed()
            if failed or stop_requested(remote) or time.time() > deadline:
                check.kill()
                check.wait()
                raise GymError(
                    "resource_unavailable",
                    "the mesh check stopped: "
                    + (
                        ", ".join(failed)
                        or (
                            "the job was stopped"
                            if stop_requested(remote)
                            else "the nodes did not meet in time"
                        )
                    ),
                )
            time.sleep(2)
        reader.join(60)
        text = "".join(output)
        (work / "mesh.log").write_text(text)
        keep_log(work / "mesh.log", remote, f"node-{node}")
        if check.returncode:
            raise GymError(
                "resource_unavailable",
                "the nodes did not form a process group: " + text[-1200:],
            )
        report_to("mesh", value=mesh_result(text))
        env[mesh.ENV] = mesh_spec(payload, node, master, port)
        worker = [sys.executable, "-m", "nexttransformer_gym.mesh", "worker"]
    except Exception as error:
        mark(f"failed-{node}", repr(error)[-1500:])
        if not lead:
            report_to("pre_failure", detail=repr(error)[-1500:])
            await_lead(float(payload["rendezvous_seconds"]) + 900)
            return 1
        emit(
            "done",
            outcome="pre_failure",
            error_code=getattr(error, "code", "resource_unavailable"),
            detail=repr(error)[-1500:],
        )
        mark("done", "pre_failure")
        return 1
    if lead:
        emit(
            "ready",
            isolation=payload["isolation"],
            uid=identity.get("user", os.geteuid()),
            network="allowed" if payload.get("network") == "allow" else "denied",
            nodes=nodes,
            gpus_per_node=per,
        )
    stop, stopped_by = threading.Event(), {}

    def watch(process):
        """Stop this node's worker when a peer failed (after a grace period), the relay stopped the job, or, on
        another node, node 0 ended."""

        def loop():
            first = None
            while not stop.wait(5):
                failed = peers_failed()
                ended = not lead and (handoff / "done").exists()
                if failed and first is None:
                    first = time.time()
                if stop_requested(remote):
                    stopped_by["reason"] = "the job was stopped"
                elif ended:
                    stopped_by["reason"] = "node 0 ended"
                elif failed and time.time() - first >= PEER_GRACE_SECONDS:
                    stopped_by["reason"] = ", ".join(failed)
                else:
                    continue
                process.kill()
                stop_identity(identity)
                return

        threading.Thread(target=loop, daemon=True).start()

    outcome = drive(
        worker,
        env,
        request,
        int(payload["credit"]),
        report_to,
        work / "tmp",
        work / "worker.stderr",
        progress_every=15.0 if lead else 300.0,
        identity=identity,
        started=watch,
    )
    stop.set()
    stop_identity(identity)
    if outcome["outcome"] != "succeeded":
        mark(f"failed-{node}", outcome.get("detail") or outcome["outcome"])
    if lead and outcome["outcome"] == "unknown":
        # A peer's failure ends this worker at its next collective; that is a known failure, not a lost job.
        deadline = time.time() + PEER_GRACE_SECONDS
        while not peers_failed() and not stopped_by and time.time() < deadline:
            time.sleep(5)
        failed = peers_failed()
        if failed or stopped_by:
            outcome = {
                "outcome": "failed",
                "error_code": "infrastructure_failure",
                "spent": outcome["spent"],
                "detail": "another node failed: "
                + (", ".join(failed) or stopped_by.get("reason", "")),
            }
    explain(outcome, work / "worker.stderr")
    keep_log(work / "worker.stderr", remote, f"node-{node}")
    if not lead:
        report_to("worker", outcome=outcome["outcome"], detail=outcome.get("detail"))
        # Node 0 stops its own worker PEER_GRACE_SECONDS after this node's failure marker, or its worker fails at
        # the next collective (within the mesh timeout); either way it then reports and leaves the done marker.
        await_lead(int(payload.get("mesh_timeout_seconds", 3600)) + 900)
        return 0 if outcome["outcome"] == "succeeded" else 1
    code_ = report(emit, payload, outcome, remote, publish)
    mark("done", outcome["outcome"])
    return code_


def node_main(job_path, node, master):
    """A node of a `local` or `ssh` job: verify, then run."""
    payload = json.loads(Path(job_path).read_text())
    node = int(node)
    emit = Emitter(mark=payload.get("mark", MARK))
    remote, code = Path(payload["remote"]), Path(payload["request"]["platform_root"])
    work = Path(payload["work_root"]) / f"node-{node}"
    emit("boot", node=node, host=socket.gethostname())
    try:
        check_runtime(payload)
        check_checkpoint(payload, node)
    except Exception as error:
        return pre_failure(payload, node, emit, error)
    return run(payload, node, master, emit, code, work, remote)


# ------------------------------------------------------------------------------------ local and ssh


def stage_code(platform, target):
    """The platform and this Gym package, as a job's nodes import them."""
    for directory in PLATFORM_DIRS:
        shutil.copytree(Path(platform) / directory, target / directory, ignore=IGNORE)
    shutil.copytree(
        Path(__file__).resolve().parent,
        target / "gym" / "src" / "nexttransformer_gym",
        ignore=IGNORE,
    )
    for file in PLATFORM_FILES:
        if (Path(platform) / file).is_file():
            shutil.copy2(Path(platform) / file, target / file)


def free_ports():
    """Two consecutive free ports on this machine (the mesh and its check)."""
    for _ in range(50):
        with socket.socket() as first:
            first.bind(("127.0.0.1", 0))
            port = first.getsockname()[1]
            if port >= 65535:
                continue
            try:
                with socket.socket() as second:
                    second.bind(("127.0.0.1", port + 1))
            except OSError:
                continue
            return port
    raise GymError("resource_unavailable", "no free ports for the mesh")


def load(path):
    """A nodes configuration, checked and with defaults."""
    raw = json.loads(Path(path).read_text())
    unknown = set(raw) - set(DEFAULTS)
    if unknown:
        raise SystemExit(f"nodes: unknown fields {sorted(unknown)}")
    config = {**DEFAULTS, **raw}
    if config["schema_version"] != DEFAULTS["schema_version"]:
        raise SystemExit("nodes: schema_version must be gym.nodes.v1")
    if config["launcher"] not in ("local", "ssh"):
        raise SystemExit("nodes: launcher is local or ssh")
    if not config["job_root"]:
        raise SystemExit("nodes: job_root (a directory every node sees) is required")
    for key in ("nodes", "gpus_per_node", "cpu_nodes"):
        if type(config[key]) is not int or config[key] < 1:
            raise SystemExit(f"nodes: {key} must be a positive integer")
    hosts = config["hosts"]
    if config["launcher"] == "ssh":
        if not hosts or not all(
            isinstance(h, str)
            or (isinstance(h, dict) and set(h) <= {"ssh", "address"} and "ssh" in h)
            for h in hosts
        ):
            raise SystemExit("nodes: ssh needs hosts (destinations or {ssh, address})")
        config["nodes"] = len(hosts)
    if not (
        isinstance(config["environment"], dict)
        and all(
            isinstance(k, str) and isinstance(v, str)
            for k, v in config["environment"].items()
        )
    ):
        raise SystemExit("nodes: environment maps names to strings")
    if config["isolation"] not in ("bwrap", "landlock", "none") or config[
        "network"
    ] not in ("deny", "allow"):
        raise SystemExit(
            "nodes: isolation is bwrap, landlock or none; network deny or allow"
        )
    if (config["nodes"] > 1 or config["cpu_nodes"] > 1) and config[
        "isolation"
    ] != "none":
        raise SystemExit(
            "nodes: a job spanning nodes needs isolation none (the sandboxes give the worker a network namespace "
            "without the other nodes); network deny still refuses new sockets once the nodes have met"
        )
    if config["cpu_nodes"] > config["nodes"]:
        raise SystemExit("nodes: cpu_nodes exceeds the nodes")
    if config["master_port"] == 0 and config["launcher"] != "local":
        raise SystemExit("nodes: master_port 0 (free ports) works for local nodes only")
    config["python"] = config["python"] or sys.executable
    return config


class ProcessLauncher(Launcher):
    """Nodes as processes this relay starts, writing their output into the job directory."""

    def __init__(self, config):
        self.config = config
        self.root = Path(config["job_root"])
        self.margin_seconds = int(config["margin_seconds"])
        self.poll_seconds = float(config["poll_seconds"])
        self.window = int(config["window"])
        self.running = {}

    # Transport: a subclass says where node k runs and how a command reaches it.
    def destinations(self, request, nodes, per):
        raise NotImplementedError

    def wrap(self, destination, argv, env, code):
        """(command, Popen options) that run `argv` on a node with `env` added, in `code` when given."""
        raise NotImplementedError

    def topology(self, request):
        if not is_cuda(request):
            return int(self.config["cpu_nodes"]), 0
        count = len(allocated_devices(request.get("gpu")))
        return research_topology(count, int(self.config["gpus_per_node"]))

    def plan(self, request, tag):
        quote = request["quote"]
        if quote["operation"] == "official":
            raise GymError(
                "resource_unavailable",
                "this executor runs research and checks; final training is dispatched separately",
            )
        execution = request["execution"]
        remote = self.root / tag / quote["id"]
        nodes, per = self.topology(request)
        if nodes > int(self.config["nodes"]):
            raise GymError(
                "resource_unavailable",
                f"the job needs {nodes} nodes; there are {self.config['nodes']}",
            )
        job_request = json.loads(json.dumps(request))
        job_request.update(
            platform_root=str(remote / "code"), output=str(remote / "checkpoint.pt")
        )
        checkpoint = checkpoint_input(request, self.checkpoint_schema)
        if checkpoint:
            job_request["checkpoint"] = checkpoint["path"]
        work_root = (
            Path(self.config["work_root"]) / tag / quote["id"]
            if self.config["work_root"]
            else remote / "work"
        )
        destinations = self.destinations(request, nodes, per)
        port = int(self.config["master_port"]) or free_ports()
        payload = {
            "schema": "nexttransformer_gym.nodes.job.v1",
            "quote_id": quote["id"],
            "request": job_request,
            "remote": str(remote),
            "remote_root": str(self.root),
            "work_root": str(work_root),
            "runtime": execution["runtime"],
            "checkpoint": checkpoint,
            "credit": str(int(quote["bound"])),
            "nodes": nodes,
            "gpus_per_node": per,
            "research_dtype": execution.get("research_dtype", "float32"),
            "research_compile": execution.get("research_compile", False),
            "isolation": execution["isolation"],
            "network": execution["network"],
            "master_port": port,
            "mesh_timeout_seconds": int(self.config["mesh_timeout_seconds"]),
            "rendezvous_seconds": int(self.config["rendezvous_seconds"]),
            "device_ids": "exact",
            "mark": self.mark,
            "destinations": destinations,
        }
        return payload, None

    def prepare(self, payload, request, snapshot):
        remote = Path(payload["remote"])
        if remote.exists():
            raise GymError("resource_unavailable", f"the job directory {remote} exists")
        (remote / "logs").mkdir(parents=True)
        stage_code(request["platform_root"], remote / "code")
        (remote / "job.json").write_text(json.dumps(payload, indent=1))
        return {"remote": str(remote)}

    def start(self, record, nodes, gpus):
        remote = Path(record["remote"])
        payload = json.loads((remote / "job.json").read_text())
        code = remote / "code"
        destinations = payload["destinations"]
        master = destinations[0]["address"]
        env = {
            **self.config["environment"],
            "PYTHONPATH": os.pathsep.join([str(code / "gym" / "src"), str(code)]),
        }
        processes = []
        for node in range(nodes):
            argv = [
                self.config["python"],
                "-m",
                "nexttransformer_gym.nodes",
                "node",
                str(remote / "job.json"),
                "--node",
                str(node),
                "--master",
                master,
            ]
            command, options = self.wrap(destinations[node], argv, env, code)
            with open(remote / "logs" / f"node-{node}.out", "w") as out:
                processes.append(
                    subprocess.Popen(
                        command,
                        stdin=subprocess.DEVNULL,
                        stdout=out,
                        stderr=subprocess.STDOUT,
                        start_new_session=True,
                        **options,
                    )
                )
        job = payload["quote_id"]
        self.running[job] = (remote, processes)
        return job

    def lines(self, job):
        remote, _ = self.running[job]
        try:
            return (
                (remote / "logs" / "node-0.out")
                .read_text(errors="replace")
                .splitlines()
            )
        except OSError:
            return None

    def ended(self, job):
        _, processes = self.running[job]
        codes = [p.poll() for p in processes]
        return None if None in codes else f"exited {codes}"

    def stop(self, job):
        remote, processes = self.running[job]
        (remote / "stop").write_text("")
        for process in processes:
            if process.poll() is None:
                self.signal(process, signal.SIGTERM)
        deadline = time.time() + 30
        while time.time() < deadline and any(p.poll() is None for p in processes):
            time.sleep(0.2)
        for process in processes:
            if process.poll() is None:
                self.signal(process, signal.SIGKILL)
                process.wait()

    def signal(self, process, signum):
        try:
            os.killpg(process.pid, signum)
        except OSError:
            pass

    def run_on(self, destination, argv, timeout=300):
        """Run a probe command on one node; its stdout."""
        command, options = self.wrap(destination, argv, {}, None)
        return subprocess.run(
            command,
            capture_output=True,
            text=True,
            timeout=timeout,
            check=True,
            **options,
        ).stdout


class LocalLauncher(ProcessLauncher):
    """Every node on this machine. Tested with CPU nodes only; untested with GPUs (see "Test status")."""

    name = "local"

    def destinations(self, request, nodes, per):
        return [{"host": "local", "address": "127.0.0.1"}] * nodes

    def wrap(self, destination, argv, env, code):
        base = {k: v for k, v in os.environ.items() if k != "CUDA_VISIBLE_DEVICES"}
        return argv, {"env": {**base, **env}, "cwd": str(code) if code else None}

    def all_destinations(self):
        return [{"host": "local", "address": "127.0.0.1"}]


class SshLauncher(ProcessLauncher):
    """Node k on hosts[k], over ssh; the job directory must be at the same path on every host.

    Tested only with CPU nodes and a stand-in ssh on one machine; untested on bare-metal GPU machines (see
    "Test status")."""

    name = "ssh"

    def __init__(self, config):
        super().__init__(config)
        self.cards = {}

    def all_destinations(self):
        found = []
        for host in self.config["hosts"]:
            host = {"ssh": host} if isinstance(host, str) else dict(host)
            host.setdefault("address", host["ssh"].rsplit("@", 1)[-1])
            found.append({"host": host["ssh"], "address": host["address"]})
        return found

    def host_cards(self, destination):
        """The GPU identities of one host, in index order (asked once per relay)."""
        if destination["host"] not in self.cards:
            text = self.run_on(
                destination,
                ["nvidia-smi", "--query-gpu=uuid", "--format=csv,noheader"],
                timeout=60,
            )
            self.cards[destination["host"]] = [
                line.strip() for line in text.splitlines() if line.strip()
            ]
        return self.cards[destination["host"]]

    def destinations(self, request, nodes, per):
        hosts = self.all_destinations()
        if not is_cuda(request):
            return hosts[:nodes]
        wanted = [card.get("id") for card in allocated_devices(request.get("gpu"))]
        chosen = []
        for node in range(nodes):
            share = wanted[node * per : (node + 1) * per]
            owner = next(
                (h for h in hosts if set(share) <= set(self.host_cards(h))), None
            )
            if owner is None:
                raise GymError(
                    "resource_unavailable",
                    f"no host has every card of node {node}'s share {share}",
                )
            chosen.append(owner)
        return chosen

    def wrap(self, destination, argv, env, code):
        assignments = [f"{k}={v}" for k, v in env.items()]
        line = shlex.join(argv)
        if assignments:
            line = "exec env " + shlex.join(assignments) + " " + line
        if code:
            line = f"cd {shlex.quote(str(code))} && {line}"
        return [*self.config["ssh"], destination["host"], line], {}

    def signal(self, process, signum):
        # The ssh client: the node stops its worker itself when it sees the stop marker.
        try:
            process.send_signal(signum)
        except OSError:
            pass


def launcher(config):
    return (LocalLauncher if config["launcher"] == "local" else SshLauncher)(config)


def configure(config_path, base=None):
    """The Gym execution configuration for a nodes configuration: the nodes' runtime and GPUs, and this module's
    relay as the operator executor."""
    config = load(config_path)
    nodes = launcher(config)
    hosts = nodes.all_destinations()
    runtimes = []
    for host in hosts:
        runtimes.append(
            json.loads(
                nodes.run_on(host, [config["python"], "-c", gym_remote.RUNTIME_SCRIPT])
            )
        )
    if any(r != runtimes[0] for r in runtimes):
        raise SystemExit(f"nodes: the hosts' runtimes differ: {runtimes}")
    per = int(config["gpus_per_node"])
    devices = []
    for host in hosts:
        try:
            text = nodes.run_on(
                host,
                [
                    "nvidia-smi",
                    "--query-gpu=uuid,name,memory.total",
                    "--format=csv,noheader,nounits",
                ],
                timeout=60,
            )
        except (OSError, subprocess.SubprocessError):
            text = ""
        rows = [
            [part.strip() for part in line.split(",")]
            for line in text.splitlines()
            if line.strip()
        ]
        count = per * (int(config["nodes"]) if config["launcher"] == "local" else 1)
        if rows and len(rows) < count:
            raise SystemExit(
                f"nodes: {host['host']} has {len(rows)} GPUs, fewer than {count}"
            )
        devices += [
            {
                "id": uuid,
                "name": name,
                "memory_total_bytes": int(float(total)) * 1024**2,
            }
            for uuid, name, total in rows[:count]
        ]
    if devices and len(devices) != per * int(config["nodes"]):
        raise SystemExit("nodes: some hosts have no GPUs")
    if devices:
        print(
            f"nodes: WARNING: GPU jobs on the {config['launcher']} launcher are untested on bare-metal machines "
            "(only CPU nodes and the Koala launcher's GPUs were tested); check a bounded job first.",
            file=sys.stderr,
        )
    value = dict(base or {})
    value.setdefault("schema_version", "gym.execution.v1")
    value["max_concurrent_jobs"] = 1
    value["executor"] = {
        "command": [
            sys.executable,
            "-m",
            "nexttransformer_gym.nodes",
            "relay",
            str(Path(config_path).resolve()),
        ],
        "devices": devices,
        "runtime": runtimes[0],
        "isolation": config["isolation"],
        "network": config["network"],
    }
    return value


def main(argv=None):
    import argparse

    parser = argparse.ArgumentParser(prog="python -m nexttransformer_gym.nodes")
    commands = parser.add_subparsers(dest="command", required=True)
    setup = commands.add_parser(
        "configure", help="write the Gym execution configuration"
    )
    setup.add_argument("config", type=Path)
    setup.add_argument(
        "--execution", type=Path, help="the configuration's other fields"
    )
    setup.add_argument("--out", type=Path, required=True)
    run_relay = commands.add_parser(
        "relay", help="the executor command the Gym starts per job"
    )
    run_relay.add_argument("config", type=Path)
    run_node = commands.add_parser("node", help="one node of a job")
    run_node.add_argument("job", type=Path)
    run_node.add_argument("--node", type=int, required=True)
    run_node.add_argument("--master", required=True)
    args = parser.parse_args(argv)
    if args.command == "configure":
        base = json.loads(args.execution.read_text()) if args.execution else None
        value = configure(args.config, base)
        args.out.write_text(json.dumps(value, indent=2) + "\n")
        print(
            f"{args.out}: {len(value['executor']['devices'])} GPUs, "
            f"isolation {value['executor']['isolation']}",
            file=sys.stderr,
        )
        return 0
    if args.command == "relay":
        return relay(launcher(load(args.config)))
    return node_main(args.job, args.node, args.master)


if __name__ == "__main__":
    sys.exit(main())
