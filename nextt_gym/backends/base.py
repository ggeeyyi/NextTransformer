"""Execution backend interface: how a run reaches a cluster, and how its state comes back.

Every backend ends in the same remote command (`python -m nextt_gym.remote run`);
backends only differ in shipping the bundle, starting processes, polling and
fetching files. Handles are opaque strings that survive service restarts.

Restarts (config `max_restarts`, default 0; only where `supports_restarts`): a run whose process ends without
the trainer's final status (killed, evicted, crashed) starts again with `--resume auto` from its latest complete
checkpoint, at most once from each checkpoint and `max_restarts` times per run. The trainer restores the
cumulative cost from the checkpoint, so the replayed updates stay outside the FLOPs ledger; status() reports
each restart under `restarts`.
"""
from abc import ABC, abstractmethod
import json

STATES = ("queued", "running", "succeeded", "failed", "cancelled")
# Files the service reads back from a run directory (trainer outputs, design §11.9).
RESULT_FILES = ("provenance.json", "metrics.jsonl", "budget.json", "run_status.json", "remote_status.json",
                "latest.json")


def state_from(run_status, remote_status, alive):
    """Map trainer/remote status files and process liveness to a backend state."""
    phase = (run_status or {}).get("phase")
    if alive:
        return "running" if remote_status else "queued"
    if phase in ("training_complete", "smoke_complete"): return "succeeded"
    if phase == "segment_complete": return "running"          # the driver starts the next segment
    if phase == "cancelled": return "cancelled"
    return "failed"


class Backend(ABC):
    kind = "base"
    supports_restarts = False

    def __init__(self, backend_id, config):
        self.backend_id, self.config = backend_id, config
        self.gpu_type = config.get("gpu_type", "unknown")
        self.gpus_per_node = int(config.get("gpus_per_node", 8))
        self.allowed_world_sizes = [int(w) for w in config.get("allowed_world_sizes", [8])]
        self.max_wall_hours_per_segment = config.get("max_wall_hours_per_segment")
        self.max_restarts = int(config.get("max_restarts", 0))
        if self.max_restarts and not self.supports_restarts:
            raise ValueError(f"{backend_id}: {self.kind} backends do not restart failed runs (max_restarts)")

    def describe(self):
        """The Backend object of the design doc (appendix A)."""
        out = dict(backend_id=self.backend_id, kind=self.kind, gpu_type=self.gpu_type,
                   allowed_world_sizes=self.allowed_world_sizes, status=self.availability())
        if self.max_wall_hours_per_segment: out["max_wall_hours_per_segment"] = self.max_wall_hours_per_segment
        if self.max_restarts: out["max_restarts"] = self.max_restarts
        if self.config.get("queue_estimate_minutes") is not None:
            out["queue_estimate_minutes"] = self.config["queue_estimate_minutes"]
        return out

    def availability(self):
        return "available"

    def check_world_size(self, world_size):
        if world_size not in self.allowed_world_sizes:
            raise ValueError(f"{self.backend_id} allows world sizes {self.allowed_world_sizes}")

    @abstractmethod
    def submit(self, run_dir_local, bundle, sha256, run_spec_path, *, world_size, synthetic=False,
               model_overrides=None):
        """Start a run; returns a handle. `run_dir_local` is the service's directory for this run."""

    @abstractmethod
    def status(self, handle):
        """dict(state, phase?, step?, consumed_flops?, gpu_hours?, restarts?)."""

    @abstractmethod
    def logs(self, handle, tail=200):
        """Last `tail` log lines."""

    @abstractmethod
    def cancel(self, handle):
        """Stop the run; checkpoints stay where they are."""

    @abstractmethod
    def fetch(self, handle, dest):
        """Copy RESULT_FILES and val_*.json into dest; returns dest."""


def summarize(budget, run_status):
    out = {}
    if budget: out.update(step=budget.get("step"), consumed_flops=budget.get("consumed_flops"),
                          gpu_hours=budget.get("gpu_hours"))
    if run_status: out.update(phase=run_status.get("phase"), final=run_status.get("final"))
    return out


def read_json(text):
    try: return json.loads(text) if text and text.strip() else None
    except json.JSONDecodeError: return None
