"""H200 checks of koala_gym.py (no CPFS there): the GPU-hour journal of a research run, and the orphan stop.

  gym/.venv/bin/python gym/koala/tests/h200_e2e.py --koala .agent-work/koala.h200-single.json

1. A fixture research run to 4 updates and the final-fixture checks: the journal must hold submitted, running
   and ended with GPU-hours for each.
2. A long fixture research run whose relay is killed once the pod reports running: `jobs --stop-orphans` must
   stop the Koala job, journal the stop with GPU-hours, and print the reconcile command for the Gym job.
"""

import argparse
import base64
import json
import os
import signal
import subprocess
import sys
import threading
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from gpu_e2e import Session, example_candidate, fixture_profiles, log, koala_gym  # noqa: E402
from nexttransformer_gym.backend import Scheduler  # noqa: E402


def journal(state):
    path = state / "koala-jobs.jsonl"
    return (
        [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
        if path.exists()
        else []
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--koala", type=Path, required=True)
    parser.add_argument(
        "--long-updates", type=int, default=46875
    )  # the Gym's schedule horizon
    args = parser.parse_args()
    base = (
        Path.home()
        / ".local/share/koala-gym/e2e"
        / f"h200-{time.strftime('%Y%m%d-%H%M%S')}"
    )
    state = base / "state"
    state.mkdir(parents=True)
    log(f"h200 e2e; state {state}")
    profiles = fixture_profiles(6 * 3600)
    # Larger than the fixture so the pod still trains when the relay is killed.
    profiles["research-long"] = dict(
        profiles["research-fixture"],
        id="research-long",
        max_updates=args.long_updates,
        width=128,
        sequence_len=128,
        batch_size=4,
    )
    session = Session(args.koala, profiles, state)
    candidate = session.call(
        "submit_candidate",
        package_base64=base64.b64encode(example_candidate()).decode(),
        idempotency_key="example",
    )
    summary = {"state": str(state), "steps": []}

    # 1. A research run that ends normally.
    summary["steps"].append(
        session.train(candidate, "research-fixture", "research 0 -> 4", target_update=4)
    )
    if summary["steps"][-1]["state"] != "succeeded":
        raise SystemExit(
            f"the research run did not succeed: {json.dumps(summary['steps'][-1], default=str)[-3000:]}"
        )
    # The final-fixture checks: candidate code on one GPU under the same pod backend.
    checks = session.call(
        "validate_candidate",
        candidate_id=candidate["id"],
        profile_id="final-fixture",
        idempotency_key="checks",
    )
    summary["steps"].append(session.job(checks, "final-fixture checks"))
    rows = journal(state)
    summary["journal_run"] = rows
    log(
        "journal after the run",
        events=[r["event"] for r in rows],
        gpu_hours=[r.get("gpu_hours") for r in rows if r["event"] == "ended"],
    )

    # 2. An orphan: the relay dies while its pod trains.
    quote = session.call(
        "quote_experiment",
        candidate_id=candidate["id"],
        profile_id="research-long",
        target_update=args.long_updates,
    )
    job = session.call(
        "run_experiment", quote_id=quote["id"], idempotency_key=quote["id"]
    )
    worker = threading.Thread(
        target=lambda: Scheduler(session.service).work_once(), daemon=True
    )
    worker.start()
    deadline = time.time() + 6 * 3600
    while time.time() < deadline:
        running = [
            r
            for r in journal(state)
            if r.get("quote_id") == quote["id"] and r["event"] == "running"
        ]
        if running or not worker.is_alive():
            break
        time.sleep(10)
    relays = koala_gym.live_relays(state)
    log("pod running; killing the relay", relays=relays, running=running)
    for pid in relays:
        os.kill(pid, signal.SIGKILL)
    worker.join(timeout=900)
    summary["gym_job_after_kill"] = session.call("get_job", job_id=job["id"])
    log("gym job after the relay died", state=summary["gym_job_after_kill"]["state"])
    koala_job = next(
        (r["job"] for r in journal(state) if r.get("quote_id") == quote["id"]), None
    )
    if koala_job is None:
        raise SystemExit(
            f"no Koala job was submitted for the orphan run: {summary['gym_job_after_kill']}"
        )
    config = koala_gym.load_config(args.koala)
    summary["koala_status_before"] = koala_gym.Koala(config).status(koala_job)
    command = [
        sys.executable,
        koala_gym.__file__,
        "--koala",
        str(args.koala),
        "jobs",
        "--state",
        str(state),
    ]
    listing = subprocess.run(
        command + ["--stop-orphans"], capture_output=True, text=True
    )
    summary["stop_orphans"] = {
        "exit": listing.returncode,
        "out": listing.stdout,
        "err": listing.stderr[-2000:],
    }
    print(listing.stdout, listing.stderr[-2000:], flush=True)
    time.sleep(60)
    summary["koala_status_after"] = koala_gym.Koala(config).status(koala_job)
    summary["journal_orphan"] = [
        r for r in journal(state) if r.get("quote_id") == quote["id"]
    ]
    again = subprocess.run(command, capture_output=True, text=True)
    summary["listing_after"] = again.stdout
    print(again.stdout, flush=True)
    log(
        "orphan",
        koala_job=koala_job,
        before=summary["koala_status_before"],
        after=summary["koala_status_after"],
        events=[(r["event"], r.get("gpu_hours")) for r in summary["journal_orphan"]],
    )
    out = base / "summary.json"
    out.write_text(json.dumps(summary, indent=1, default=str))
    log(f"summary written to {out}")


if __name__ == "__main__":
    main()
