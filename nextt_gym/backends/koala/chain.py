"""Local segment driver for one gym run, the Python form of launch/koala/chain.sh:

    python -m nextt_gym.backends.koala chain RUN_DIR

RUN_DIR holds `koala_plan.json`, written by KoalaBackend.submit. Each segment is one Koala job with
the same code bundle and `--resume auto`; the next one is submitted only after the previous one ended
with phase `segment_complete`, or after a failed one that may restart (`restart_from`). The driver records
the chain in `chain.json` and holds `driver.lock` while it runs. Restarting the driver is safe: a second
driver exits, and a new one re-attaches to the unfinished job instead of submitting another. A cancelled run
(the `cancelled` marker) ends the chain.
"""
import fcntl
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import time

from .. import reader
from .cli import ACTIVE_STATES, ENDED_STATES, Koala

FINAL = ("succeeded", "failed", "cancelled")
UNKNOWN_POLLS = 60        # unknown job states in a row (each after `koala wait`, 30 s apart) before a job counts as ended


def write_json(path, payload):
    tmp = Path(str(path) + ".tmp"); tmp.write_text(json.dumps(payload, indent=1)); tmp.replace(path)


def read_chain(run_dir):
    path = Path(run_dir) / "chain.json"
    if not path.exists(): return {}
    try: return json.loads(path.read_text()) or {}
    except json.JSONDecodeError: return {}


def driver_running(run_dir):
    """True while a driver holds the chain lock (immune to pid reuse)."""
    with open(Path(run_dir) / "driver.lock", "a") as lock:
        try: fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError: return True
        fcntl.flock(lock, fcntl.LOCK_UN)
        return False


def start_driver(run_dir):
    """Start a detached driver for RUN_DIR; it outlives the gym process that started it."""
    driver = subprocess.Popen([sys.executable, "-m", "nextt_gym.backends.koala", "chain", str(run_dir)],
                              stdout=open(Path(run_dir) / "driver.log", "ab"), stderr=subprocess.STDOUT,
                              start_new_session=True,
                              env=dict(os.environ, PYTHONPATH=str(Path(__file__).resolve().parents[3])))
    (Path(run_dir) / "driver_pid").write_text(str(driver.pid))
    return driver.pid


def segment_status(plan, koala, job):
    """What the segment that just ended left: phase, step and consumed FLOPs from run_status.json (else the
    job log), the last logged step of budget.json and the latest complete checkpoint of latest.json."""
    found = {}
    try:
        with tempfile.TemporaryDirectory() as tmp:
            for name in reader.fetch(plan["reader"], plan["remote_dir"],
                                     ["run/run_status.json", "run/budget.json", "run/latest.json"], tmp):
                found[name] = json.loads((Path(tmp) / name).read_text())
    except Exception as exc:
        print(f"reader failed, falling back to the job log: {exc}", flush=True)
    status = found.get("run_status.json")
    if status:
        phase, step, consumed = status.get("phase"), status.get("step"), status.get("consumed_flops")
    else:
        lines = [l for l in koala.log(job) if l.startswith("NEXTT_RUN_STATUS")]
        fields = lines[-1].split() if lines else []
        phase, step, consumed = (fields[1], int(fields[2]), fields[3]) if len(fields) >= 4 else (None, None, None)
    return dict(phase=phase, step=step, consumed=consumed, logged_step=(found.get("budget.json") or {}).get("step"),
                checkpoint=(found.get("latest.json") or {}).get("step"))


def restart_from(plan, record, outcome):
    """The checkpoint step a failed segment restarts from, or None (backends/base.py). Only a segment that ended
    without the trainer's final status restarts: a trainer that wrote one (e.g. budget_exceeded) has decided."""
    restarts, checkpoint = record.get("restarts", []), outcome["checkpoint"]
    if outcome["phase"] is not None or not checkpoint or len(restarts) >= plan.get("max_restarts", 0): return None
    if any(r["resumed_from"] == checkpoint for r in restarts): return None   # once per checkpoint
    return checkpoint


def wait_until_ended(koala, job, observe=lambda state: None):
    """The job's final platform state; `observe` sees every state read. Only an ended state ends the
    wait: an unreadable, blank or unrecognized state is unknown, and UNKNOWN_POLLS of them in a row end
    it with None. `koala wait` also returns on timeouts and CLI errors."""
    unknown = 0
    while True:
        state = koala.state(job)
        if state: observe(state)
        if state in ENDED_STATES: return state
        if state in ACTIVE_STATES:
            unknown = 0
        else:
            unknown += 1
            if unknown > UNKNOWN_POLLS: return state or None
        koala.wait(job)
        if state not in ACTIVE_STATES: time.sleep(30)


def chain(run_dir):
    run_dir = Path(run_dir); plan = json.loads((run_dir / "koala_plan.json").read_text())
    lock = open(run_dir / "driver.lock", "a")
    try: fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError: return                      # another driver owns this chain
    koala = Koala(plan["cluster"])
    record = read_chain(run_dir) or dict(state="queued", segments=[])
    write = lambda: write_json(run_dir / "chain.json", record)
    cancelled = lambda: (run_dir / "cancelled").exists()
    while record["state"] not in FINAL:
        segment = record["segments"][-1] if record["segments"] else None
        if segment is None or segment.get("finished"):
            if cancelled(): record["state"] = "cancelled"; break
            if len(record["segments"]) >= plan["max_segments"]:
                record.update(state="failed", error="segment limit reached"); break
            try:
                job = koala.submit(plan["command"], bundle_uri=plan["bundle_uri"], code_dir=plan["code_dir"],
                                   job_prefix=plan["job_prefix"], nodes=plan["nodes"], gpus=plan["gpus"],
                                   flags=plan["flags"])
            except RuntimeError as exc:
                record.update(state="failed", error=str(exc)); break
            segment = dict(n=len(record["segments"]) + 1, job=job, submitted=time.time())
            record["segments"].append(segment); record["state"] = "running"; write()
            if cancelled(): koala.stop(job)
        def observe(state, segment=segment):
            if segment.get("job_state") != state: segment["job_state"] = state; write()
        job_state = wait_until_ended(koala, segment["job"], observe)
        outcome = segment_status(plan, koala, segment["job"])
        phase = outcome["phase"]
        segment.update(job_state=job_state, status=[phase, outcome["step"], outcome["consumed"]],
                       checkpoint=outcome["checkpoint"], finished=time.time())
        if phase: record.update(phase=phase, step=outcome["step"], consumed_flops=outcome["consumed"])
        if cancelled(): record["state"] = "cancelled"
        elif phase in ("training_complete", "smoke_complete"): record["state"] = "succeeded"
        elif phase == "segment_complete": pass
        elif (checkpoint := restart_from(plan, record, outcome)) is not None:
            record.setdefault("restarts", []).append(dict(segment=segment["n"], job_state=job_state,
                                                          failed_at_step=outcome["logged_step"],
                                                          resumed_from=checkpoint))
        else: record.update(state="failed", error=f"job {job_state}, phase {phase}")
        write()
    write()
