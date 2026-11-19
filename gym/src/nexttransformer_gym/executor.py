"""Operator executors: the Gym's numerical jobs run somewhere other than the controller.

The Gym's controller, API, budget and evidence code stay on this machine. An operator who runs the numerical work
elsewhere (a cluster scheduler, another host) declares an executor in the execution configuration:

    "executor": {
        "command": ["/usr/bin/python3", "/opt/site/executor.py", "run"],
        "environment": ["SITE_TOKEN"],
        "devices": [{"id": "GPU-<uuid>", "name": "NVIDIA H20-3e", "memory_total_bytes": 150754820096}, ...],
        "runtime": {"dependencies": {...}, "python": [3, 12, 3], "cuda_version": "12.8"},
        "isolation": "none",
        "network": "deny"
    }

* `command` replaces the local worker for every job the scheduler executes (research training, checks and final
  runs). It starts in the Gym's state directory and speaks the worker protocol on stdin/stdout, as worker.py does
  (`Channel` is that side of it). Static audits (cost inspection, calibration) still run locally, under the
  configuration's own `isolation`.
* `environment` names controller variables (credentials, for example) the executor process receives; no worker
  and no static audit sees them. The executor runs as the controller's user.
* `devices` are the cards one job may use, in order: the Gym leases them as it leases local cards (one group for
  multi-GPU research) and never inspects them; `gpu_ids` defaults to their ids. The names are what agents see.
* `runtime` is the remote numerical interpreter's record (`remote.runtime`): the revision binds it, and the remote
  side refuses work when its interpreter differs.
* `isolation` and `network` state how the remote side confines candidate code (`remote.worker_command`); agents
  see them, and `none` prints the Gym's unisolated warning at startup.

Agents see the configuration without `command` and `environment` (`public`), and the revision ignores them too.

The worker protocol. The Gym writes the request as one JSON line, then answers each event:
    GYM_EVENT {"kind": "intent", "label": L, "amount": A}  -> {"ack": L}, or {"cancel": true} to stop before work
    GYM_EVENT {"kind": "charge", "label": L, "amount": A}  -> {"ack": L}   (settles the open intent)
    GYM_EVENT {"kind": "poll"}                             -> {"cancel": true} or {"continue": true}
    GYM_EVENT {"kind": "runtime" | "progress" | "result" | "failed" | "cancelled", ...}   (no answer)
At most one intent is open; a charge repeats it exactly; the job ends with exactly one result, failed or cancelled
event and exit status 0. Anything else leaves the reservation for the operator's `reconcile`. A train job's
`output` file is the job's checkpoint artifact; the Gym retains and hashes it and hands it back as `checkpoint`
when a later job continues from it, so an executor may store there a reference to weights it keeps elsewhere. The
request's `execution` holds what the remote side needs to run the job as this Gym would: the declared `runtime`,
`isolation` and `network`, and the research precision and compiler setting (`research_dtype`, `research_compile`).

`nodes` is such an executor for a group of machines (local processes, ssh, or a cluster plugin's launcher).
"""

import json
import re
import sqlite3
import sys
from pathlib import Path

from .common import require

EVENT = "GYM_EVENT "
FIELDS = ("command", "environment", "devices", "runtime", "isolation", "network")
HIDDEN = ("command", "environment")
NAME = re.compile(r"[A-Za-z_][A-Za-z0-9_]*\Z")


def settings(value, memory_limit):
    """The executor block of an execution configuration, checked and complete."""
    from .gpu import UUID
    from .isolation import BACKENDS
    from .remote import valid_runtime

    require(
        isinstance(value, dict)
        and not set(value) - set(FIELDS)
        and {"command", "runtime"} <= set(value),
        "invalid_request",
        "An executor needs a command and a runtime, and has only: "
        + ", ".join(FIELDS)
        + ".",
    )
    command = value["command"]
    require(
        isinstance(command, list)
        and command
        and all(isinstance(a, str) and a for a in command),
        "invalid_request",
        "executor.command must be a non-empty argument list.",
    )
    environment = value.get("environment", [])
    require(
        isinstance(environment, list)
        and all(isinstance(n, str) and NAME.fullmatch(n) for n in environment),
        "invalid_request",
        "executor.environment lists variable names.",
    )
    devices = value.get("devices", [])
    require(
        isinstance(devices, list)
        and all(
            isinstance(d, dict)
            and set(d) == {"id", "name", "memory_total_bytes"}
            and isinstance(d["id"], str)
            and UUID.fullmatch(d["id"])
            and isinstance(d["name"], str)
            and 0 < len(d["name"]) <= 128
            and type(d["memory_total_bytes"]) is int
            and d["memory_total_bytes"] >= memory_limit + 256 * 1024**2
            for d in devices
        )
        and len({d["id"] for d in devices}) == len(devices),
        "invalid_request",
        "executor.devices are distinct {id: NVIDIA-style GPU UUID, name, memory_total_bytes} records with room "
        "for the memory budget.",
    )
    require(
        valid_runtime(value["runtime"]),
        "invalid_request",
        "executor.runtime must be a runtime record (python -m nexttransformer_gym.remote runtime).",
    )
    isolation = value.get("isolation", "bwrap")
    network = value.get("network", "deny")
    require(
        isolation in BACKENDS and network in ("deny", "allow"),
        "invalid_request",
        "executor.isolation is one of "
        + ", ".join(BACKENDS)
        + "; network is deny or allow.",
    )
    return {
        "command": list(command),
        "environment": list(environment),
        "devices": [dict(d) for d in devices],
        "runtime": value["runtime"],
        "isolation": isolation,
        "network": network,
    }


def public(config):
    """An execution configuration as agents (and the revision) see it: without the executor's command and
    environment."""
    if not config.get("executor"):
        return config
    return {
        **config,
        "executor": {k: v for k, v in config["executor"].items() if k not in HIDDEN},
    }


def inventory(config):
    """The executor's declared cards, in the Gym's discovery format; they are never observed locally, so each is
    idle unless this Gym leases it."""
    cards = []
    for index, device in enumerate(config["executor"]["devices"]):
        cards.append(
            {
                "id": device["id"],
                "uuid": device["id"],
                "index": index,
                "device_minor": index,
                "name": device["name"],
                "memory_total_bytes": device["memory_total_bytes"],
                "memory_free_bytes": device["memory_total_bytes"],
                "utilization_percent": 0,
                "process_count": 0,
                "compute_process_count": 0,
                "graphics_process_count": 0,
            }
        )
    return {"status": "ok", "gpus": cards}


class Channel:
    """The worker side of the worker protocol, over this process's stdin and stdout."""

    def __init__(self, stdin=None, stdout=None):
        self.stdin = stdin or sys.stdin
        self.stdout = stdout or sys.stdout

    def request(self):
        return json.loads(self.stdin.readline())

    def emit(self, value):
        self.stdout.write(EVENT + json.dumps(value, allow_nan=False) + "\n")
        self.stdout.flush()

    def _reply(self):
        line = self.stdin.readline()
        if not line:
            raise RuntimeError("the Gym scheduler closed the worker protocol")
        return json.loads(line)

    def intent(self, amount, label):
        """Announce an operation of `amount` before its work; False when the Gym cancels instead."""
        self.emit({"kind": "intent", "label": label, "amount": str(int(amount))})
        reply = self._reply()
        if reply.get("cancel"):
            return False
        if reply != {"ack": label}:
            raise RuntimeError("accounting acknowledgement mismatch")
        return True

    def charge(self, amount, label):
        self.emit({"kind": "charge", "label": label, "amount": str(int(amount))})
        if self._reply() != {"ack": label}:
            raise RuntimeError("charge acknowledgement mismatch")

    def poll(self):
        """Whether the Gym asks this job to stop (cancel_job)."""
        self.emit({"kind": "poll"})
        reply = self._reply()
        if reply.get("cancel"):
            return True
        if reply != {"continue": True}:
            raise RuntimeError("poll reply mismatch")
        return False

    def runtime(self, value):
        self.emit({"kind": "runtime", "value": value})

    def progress(self, step):
        self.emit({"kind": "progress", "step": int(step)})

    def result(self, value):
        self.emit({"kind": "result", "value": value})

    def failed(self, code):
        self.emit(
            {
                "kind": "failed",
                "error_code": code,
                "accounting": "conservative_operation_bound",
            }
        )

    def cancelled(self):
        self.emit({"kind": "cancelled"})


class Charges:
    """Ahead-of-work billing for an executor that learns of a job's progress only after the work happened.

    The job's bound is the most it may cost. A training job of more than `window` updates is billed in windows: the
    initialization and the first `window` updates when it starts, then `window` updates ahead of the progress the
    executor observes, so stopping it (cancel_job, a failure) leaves the updates that never started unbilled. Any
    other job is billed its whole bound when it starts. A successful job settles exactly its bound."""

    def __init__(self, channel, quote, window=16, label="executor"):
        from .accounting import initialization_cost, update_cost

        self.channel, self.label = channel, label
        self.bound = int(quote["bound"])
        self.start_update = int(quote.get("start_update") or 0)
        self.total = max(1, int(quote.get("target_update") or 1) - self.start_update)
        self.window = int(window)
        report = quote.get("cost_report") or {}
        self.windowed = (
            quote.get("operation") == "train"
            and bool(report)
            and self.total > self.window
        )
        if self.windowed:
            self.initialization, self.update = (
                initialization_cost(report),
                update_cost(report),
            )
        self.settled, self.open, self.count = 0, None, 0

    def _covered(self, completed):
        if not self.windowed or completed >= self.total:
            return self.bound
        horizon = min(self.total, max(0, completed) + self.window)
        if horizon >= self.total:
            return self.bound
        return min(self.bound, self.initialization + horizon * self.update)

    def _next_label(self):
        self.count += 1
        return f"{self.label}-{self.count}"

    def advance(self, completed=0):
        """Cover the work up to `completed` updates plus one window; False when the Gym cancels instead."""
        target = self._covered(completed)
        if target <= self.settled + (self.open[1] if self.open else 0):
            return True
        self.settle_current()
        label, amount = self._next_label(), target - self.settled
        if not self.channel.intent(amount, label):
            return False
        self.open = (label, amount)
        return True

    def settle_current(self):
        """Charge the open window: its work may have started."""
        if self.open is not None:
            label, amount = self.open
            self.channel.charge(amount, label)
            self.settled += amount
            self.open = None

    def settle_full(self):
        """A finished job: charge exactly the bound; False when the Gym cancels the remainder."""
        self.settle_current()
        if self.settled < self.bound:
            label, amount = self._next_label(), self.bound - self.settled
            if not self.channel.intent(amount, label):
                return False
            self.channel.charge(amount, label)
            self.settled += amount
        return True


def jobs(state_dir):
    """The job records of a Gym state directory, read-only, for operator tools."""
    database = Path(state_dir) / "gym.sqlite3"
    try:
        connection = sqlite3.connect(f"file:{database}?mode=ro", uri=True, timeout=5)
        try:
            rows = connection.execute(
                "SELECT id, owner, value FROM objects WHERE kind='job' ORDER BY rowid"
            ).fetchall()
        finally:
            connection.close()
    except sqlite3.Error:
        return []
    found = []
    for job_id, owner, value in rows:
        record = json.loads(value)
        found.append(
            {
                "id": job_id,
                "session_id": owner,
                "quote_id": record.get("quote_id"),
                "operation": record.get("operation"),
                "state": record.get("state"),
                "cancel_requested": bool(record.get("cancel_requested")),
            }
        )
    return found


def main(argv=None):
    import argparse

    parser = argparse.ArgumentParser(prog="python -m nexttransformer_gym.executor")
    commands = parser.add_subparsers(dest="command", required=True)
    listing = commands.add_parser("jobs", help="list a Gym state's job records")
    listing.add_argument("--state", type=Path, required=True)
    args = parser.parse_args(argv)
    for record in jobs(args.state):
        print(json.dumps(record))
    return 0


if __name__ == "__main__":
    sys.exit(main())
