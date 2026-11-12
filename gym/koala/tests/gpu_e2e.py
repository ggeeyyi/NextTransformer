"""GPU end-to-end checks of koala_gym.py on Koala, driven through the Gym's own service API.

  gym/.venv/bin/python gym/koala/tests/gpu_e2e.py single --koala .agent-work/koala.json
  gym/.venv/bin/python gym/koala/tests/gpu_e2e.py multi --koala .agent-work/koala.smoke.json

single: research runs on one Koala GPU (synthetic fixture, continuation from a Koala checkpoint, the
        final-fixture checks) and a short ClimbMix run reading CPFS data.
multi:  an admissible native plugin is calibrated and frozen (prepare-final), checked on one GPU at
        formal dimensions, finalized, verified and dispatched as a final run over several Koala nodes
        with official_smoke (bounded; never a result).
"""

import argparse
import base64
import io
import json
import sys
import time
import zipfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
sys.path[:0] = [str(HERE.parent), str(ROOT / "gym" / "src"), str(ROOT)]

import koala_gym  # noqa: E402
from nexttransformer_gym.backend import NativeBackend, Scheduler  # noqa: E402
from nexttransformer_gym.evidence import snapshot, verify_receipt  # noqa: E402
from nexttransformer_gym.final import get_final_run, launch_final_run  # noqa: E402
from nexttransformer_gym.prepare import prepare_snapshot  # noqa: E402
from nexttransformer_gym.service import Service  # noqa: E402

# Pod paths of the data: CPFS on H20; on H200 (object storage) the pod-local paths that data_stage maps to S3.
import os  # noqa: E402

RESOURCES = os.environ.get(
    "KOALA_E2E_RESOURCES", "/cpfs/<user>/nextt/resources/native-d24"
)
DATA = os.environ.get(
    "KOALA_E2E_DATA", "/cpfs/<user>/nextt/datasets/climbmix-400b-shuffle"
)
CLIMBMIX = {
    "kind": "climbmix",
    "tokenizer": f"{RESOURCES}/tokenizer",
    "train_paths": [f"{DATA}/shard_00000.parquet"],
    "dev_path": f"{DATA}/shard_06541.parquet",
    "token_bytes": f"{RESOURCES}/tokenizer/token_bytes.pt",
}


def log(message, **values):
    print(
        time.strftime("%H:%M:%SZ", time.gmtime()),
        message,
        json.dumps(values, default=str) if values else "",
        flush=True,
    )


class Session:
    def __init__(self, config_path, profiles, state):
        self.config = koala_gym.load_config(config_path)
        koala_gym.install_controller(self.config, config_path)
        self.state = state
        self.execution = {
            "gpu_ids": koala_gym.virtual_ids(self.config),
            "gpu_memory_limit_bytes": self.config["gpu_memory_limit_bytes"],
            "max_concurrent_jobs": 1,
        }
        self.open(profiles)
        self.credentials = self.service.create_session(10**21, "koala-gpu-e2e")

    def open(self, profiles):
        self.profiles = profiles
        python = koala_gym.check_worker_python(
            self.config, self.config["worker_python"] or sys.executable
        )
        self.backend = NativeBackend(
            profiles, self.state, ROOT, python=python, execution_config=self.execution
        )
        self.service = Service(self.state / "gym.sqlite3", self.backend)

    def call(self, operation, **params):
        return self.service.call(self.credentials["agent_token"], operation, params)

    def job(self, job, label):
        started = time.time()
        Scheduler(self.service).work_once()
        state = self.call("get_job", job_id=job["id"])
        record = {
            "label": label,
            "state": state["state"],
            "error_code": state.get("error_code"),
            "minutes": round((time.time() - started) / 60, 1),
        }
        if state["state"] == "succeeded":
            record["result"] = self.call("get_result", job_id=job["id"])
        else:
            path = self.backend.state_dir / (job["id"] + ".diagnostic.log")
            record["diagnostics"] = path.read_text()[-3000:] if path.exists() else None
        record["budget"] = self.call("get_budget")
        log(
            label,
            state=record["state"],
            error=record["error_code"],
            minutes=record["minutes"],
        )
        return record

    def train(self, candidate, profile, label, **params):
        quote = self.call(
            "quote_experiment",
            candidate_id=candidate["id"],
            profile_id=profile,
            **params,
        )
        job = self.call(
            "run_experiment", quote_id=quote["id"], idempotency_key=quote["id"]
        )
        record = self.job(job, label)
        record["bound"] = quote["bound"]
        return record


def example_candidate():
    blob = io.BytesIO()
    import runpy

    runpy.run_path(str(ROOT / "gym/examples/make_candidate.py"))["package"](
        blob, "square"
    )
    return blob.getvalue()


def probe_candidate():
    """A native plugin with the anchor architecture: it passes formal admission unchanged."""
    files = {
        "proposal.json": json.dumps(
            {
                "schema_version": "gym.candidate.v1",
                "sdk_version": "reference_plugin_v1",
                "method": "koala_probe",
                "parent_ids": [],
                "hypothesis": "Koala integration check with the dense anchor shape.",
            }
        ),
        "architecture.json": json.dumps(
            {"layers": 24, "mlp_width": 6144, "window_pattern": "SSSL"}
        ),
        "plugin/koala_probe/method.json": json.dumps(
            {
                "name": "koala_probe",
                "family": "others",
                "options": {},
                "shell": {"window_pattern": "SSSL"},
            }
        ),
        "plugin/koala_probe/__init__.py": (
            "from nexttransformer.models.lm import ReferenceLM\n\n\n"
            "class KoalaProbe(ReferenceLM):\n    pass\n\n\nmodel = KoalaProbe\n"
        ),
    }
    blob = io.BytesIO()
    with zipfile.ZipFile(blob, "w") as archive:
        for name, source in files.items():
            archive.writestr(name, source)
    return blob.getvalue()


def fixture_profiles(timeout):
    profiles = json.loads((ROOT / "gym/examples/profiles.fixture.json").read_text())
    for profile in profiles.values():
        profile["timeout_sec"] = timeout
    return profiles


def climbmix_profile(timeout):
    return {
        "id": "research-climbmix",
        "kind": "research",
        "resource_profile_id": "koala-climbmix-small-v1",
        "width": 256,
        "heads": 4,
        "sequence_len": 512,
        "vocab_size": 32768,
        "batch_size": 8,
        "max_updates": 20,
        "dev_batches": 4,
        "timeout_sec": timeout,
        "device": "cpu",
        "formal_dimensions": False,
        "official_compatibility_verified": False,
        "data": CLIMBMIX,
        "memory_limit_bytes": 8 << 30,
        "checkpoint_quota_bytes": 4 << 30,
    }


def single(args):
    profiles = fixture_profiles(args.timeout)
    profiles["research-climbmix"] = climbmix_profile(args.timeout)
    session = Session(args.koala, profiles, args.state)
    candidate = session.call(
        "submit_candidate",
        package_base64=base64.b64encode(example_candidate()).decode(),
        idempotency_key="example",
    )
    summary = {"resources": session.call("get_resources"), "steps": []}
    first = session.train(
        candidate, "research-fixture", "research-fixture to 4", target_update=4
    )
    summary["steps"].append(first)
    if first["state"] == "succeeded":
        summary["steps"].append(
            session.train(
                candidate,
                "research-fixture",
                "resume 4 -> 8",
                target_update=8,
                checkpoint_id=first["result"]["checkpoint_id"],
            )
        )
    checks = session.call(
        "validate_candidate",
        candidate_id=candidate["id"],
        profile_id="final-fixture",
        idempotency_key="checks",
    )
    summary["steps"].append(session.job(checks, "final-fixture checks"))
    summary["steps"].append(
        session.train(
            candidate, "research-climbmix", "climbmix to 20", target_update=20
        )
    )
    return summary


def resume(args):
    """A research run to 4 updates and its continuation to 8 from the Koala checkpoint, in its own session."""
    session = Session(args.koala, fixture_profiles(args.timeout), args.state)
    candidate = session.call(
        "submit_candidate",
        package_base64=base64.b64encode(example_candidate()).decode(),
        idempotency_key="example",
    )
    first = session.train(
        candidate, "research-fixture", "research-fixture to 4", target_update=4
    )
    steps = [first]
    if first["state"] == "succeeded":
        steps.append(
            session.train(
                candidate,
                "research-fixture",
                "resume 4 -> 8",
                target_update=8,
                checkpoint_id=first["result"]["checkpoint_id"],
            )
        )
    return {"steps": steps}


def checks(args):
    """The final-fixture checks alone (one GPU job), in its own session."""
    session = Session(args.koala, fixture_profiles(args.timeout), args.state)
    candidate = session.call(
        "submit_candidate",
        package_base64=base64.b64encode(example_candidate()).decode(),
        idempotency_key="example",
    )
    job = session.call(
        "validate_candidate",
        candidate_id=candidate["id"],
        profile_id="final-fixture",
        idempotency_key="checks",
    )
    return {"steps": [session.job(job, "final-fixture checks")]}


def climbmix(args):
    """A short ClimbMix run reading CPFS data (one GPU job), in its own session."""
    profiles = fixture_profiles(args.timeout)
    profiles["research-climbmix"] = climbmix_profile(args.timeout)
    session = Session(args.koala, profiles, args.state)
    candidate = session.call(
        "submit_candidate",
        package_base64=base64.b64encode(example_candidate()).decode(),
        idempotency_key="example",
    )
    return {
        "steps": [
            session.train(
                candidate, "research-climbmix", "climbmix to 20", target_update=20
            )
        ]
    }


def multi(args):
    profiles = fixture_profiles(args.timeout)
    session = Session(args.koala, profiles, args.state)
    candidate = session.call(
        "submit_candidate",
        package_base64=base64.b64encode(probe_candidate()).decode(),
        idempotency_key="probe",
    )
    summary = {
        "candidate": {k: candidate[k] for k in ("id", "hash", "method")},
        "steps": [],
    }
    destination = args.state.parent / "snapshot"
    log("prepare-final (calibration and admission on this machine)")
    plan = prepare_snapshot(
        session.service,
        session.credentials["session_id"],
        candidate["id"],
        destination,
        RESOURCES,
    )
    summary["plan"] = {
        k: plan[k] for k in ("config", "method", "initialization_flops", "resources")
    }
    final = {
        "id": "final-climbmix",
        "kind": "final",
        "resource_profile_id": "koala-final-v1",
        "width": 1536,
        "heads": 12,
        "sequence_len": 2048,
        "vocab_size": 32768,
        "batch_size": args.batch,
        "max_updates": 46875,
        "dev_batches": 1,
        "timeout_sec": args.timeout,
        "device": "cuda",
        "formal_dimensions": True,
        "official_compatibility_verified": True,
        "data": CLIMBMIX,
        "memory_limit_bytes": session.config["gpu_memory_limit_bytes"],
        "checkpoint_quota_bytes": 8 << 30,
        "official_plan": plan,
        "cuda_visible_devices": list(range(8)),
    }
    session.open({**profiles, "final-climbmix": final})
    checks = session.call(
        "validate_candidate",
        candidate_id=candidate["id"],
        profile_id="final-climbmix",
        idempotency_key="final-checks",
    )
    record = session.job(checks, "final-climbmix checks (1 GPU, formal dimensions)")
    summary["steps"].append(record)
    if record["state"] != "succeeded" or not record["result"].get("admitted"):
        return summary
    submission = session.call(
        "finalize",
        candidate_id=candidate["id"],
        profile_id="final-climbmix",
        idempotency_key="finalize",
    )
    verifier = {
        "session_id": session.credentials["session_id"],
        "token": session.credentials["verifier_token"],
    }
    receipt = {
        "session_id": session.credentials["session_id"],
        "submission_id": submission["id"],
        "candidate_hash": candidate["hash"],
    }
    summary["verified"] = verify_receipt(
        receipt, snapshot(session.service, verifier), verifier
    )["valid"]
    run = launch_final_run(
        session.service,
        session.credentials["session_id"],
        submission["id"],
        10**20,
        "dispatch",
    )
    started = time.time()
    Scheduler(session.service).work_once()
    final_run = get_final_run(
        session.service, session.credentials["session_id"], run["id"]
    )
    log(
        "final run",
        state=final_run.get("state"),
        minutes=round((time.time() - started) / 60, 1),
    )
    summary["final_run"] = final_run
    path = session.backend.state_dir / (final_run.get("job_id", "") + ".diagnostic.log")
    if path.is_file():
        summary["final_diagnostics"] = path.read_text()[-3000:]
    return summary


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "scenario", choices=("single", "multi", "resume", "checks", "climbmix")
    )
    parser.add_argument("--koala", type=Path, required=True)
    parser.add_argument("--state", type=Path)
    # The Gym's worker deadline starts when the scheduler hands the job to the relay, so on Koala the
    # profile timeout covers the queue wait as well as the run; H20 queues have exceeded 3 h.
    parser.add_argument("--timeout", type=int, default=48 * 3600)
    parser.add_argument("--batch", type=int, default=4)
    args = parser.parse_args()
    base = (
        Path.home()
        / ".local/share/koala-gym/e2e"
        / f"{args.scenario}-{time.strftime('%Y%m%d-%H%M%S')}"
    )
    args.state = args.state or base / "state"
    args.state.mkdir(parents=True, exist_ok=True)
    log(f"{args.scenario} e2e; state {args.state}")
    summary = {
        "single": single,
        "multi": multi,
        "resume": resume,
        "checks": checks,
        "climbmix": climbmix,
    }[args.scenario](args)
    out = args.state.parent / "summary.json"
    out.write_text(json.dumps(summary, indent=1, default=str))
    log(f"summary written to {out}")


if __name__ == "__main__":
    main()
