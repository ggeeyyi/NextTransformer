"""GymService: the 19 agent tools as plain Python methods (MCP and CLI both call these).

State lives in the SQLite store under NEXTT_GYM_HOME. Submitted code only ever
runs in subprocesses (verification, calibration) or on a backend (training).
Run state is pulled from backends lazily, at most every POLL_SECONDS, whenever a
tool reads runs or events, or by `python -m nextt_gym.cli monitor`.
"""
from datetime import datetime, timedelta, timezone
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import tempfile
import threading
import time

from .archive import Archive
from .backends import load_backends
from .contract import REQUIRED_FILES, OPTIONAL_FILES, load_submission
from .errors import GymError
from .ledger import FlopsLedger
from .store import Store, now
from .task import SCALES, TASK

REPO_ROOT = Path(__file__).resolve().parents[1]
POLL_SECONDS = 20
QUOTE_TTL_MINUTES = 30
VERIFY_TIMEOUT_S = 1800
PEAK_BF16 = {"H100": 989e12, "H200": 989e12, "H20": 148e12, "A800": 312e12, "A100": 312e12}
# Fraction of peak that the four formal H20 linear runs reached in theory FLOPs (~0.55-0.59).
ASSUMED_MFU = 0.5
SUBPROCESS_ENV = dict(CUDA_VISIBLE_DEVICES="", NANOCHAT_DTYPE="float32", TORCH_COMPILE_DISABLE="1")
TERMINAL = ("succeeded", "failed", "cancelled", "budget_exceeded")
SEALED_HINT = ("official BPB and CORE of gym and imported runs stay sealed until the operator reveals the attempt "
               "(cli reveal)")
SEALED_LOG = re.compile(r'val_bpb_official|"split": "official"|centered:')


def _sha(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, default=str).encode()).hexdigest()


def _python_module(module, *args, timeout):
    env = dict(os.environ, **SUBPROCESS_ENV)
    env["PYTHONPATH"] = str(REPO_ROOT) + os.pathsep + env.get("PYTHONPATH", "")
    return subprocess.run([sys.executable, "-m", module, *map(str, args)], cwd=REPO_ROOT, env=env,
                          capture_output=True, text=True, timeout=timeout)


def _run_verify(directory, levels="V0,V1,V2,V3"):
    with tempfile.TemporaryDirectory() as tmp:
        out = Path(tmp) / "result.json"
        try:
            proc = _python_module("nextt_gym.verify", directory, "--out", out, "--levels", levels,
                                  timeout=VERIFY_TIMEOUT_S)
        except subprocess.TimeoutExpired:
            return dict(status="failed", levels=[dict(level="V0-V3", status="failed", errors=[GymError(
                "SPEC_VIOLATION", f"verification exceeded {VERIFY_TIMEOUT_S} s").to_dict()])])
        if not out.exists():
            return dict(status="failed", levels=[dict(level="V0-V3", status="failed", errors=[GymError(
                "INTERNAL", "verifier crashed", details=dict(stderr=proc.stderr[-2000:])).to_dict()])])
        return json.loads(out.read_text())


class GymService:
    def __init__(self, home=None, backends_path=None):
        self.store = Store(home)
        self.archive, self.ledger = Archive(self.store), FlopsLedger(self.store)
        self.backends = load_backends(backends_path)
        self.home = self.store.home
        self._last_poll = 0.0
        self._lock = threading.Lock()

    # ---- helpers ------------------------------------------------------------------------
    def _backend(self, backend_id):
        if backend_id not in self.backends:
            raise GymError("NOT_FOUND", f"unknown backend {backend_id!r}", details=dict(available=sorted(self.backends)))
        return self.backends[backend_id]

    @staticmethod
    def _scale(scale):
        if scale not in SCALES: raise GymError("INVALID_ARGUMENT", f"scale must be one of {list(SCALES)}")
        return SCALES[scale]

    def _check_split(self, scale, split):
        if split not in ("dev", "official"): raise GymError("INVALID_ARGUMENT", "split is 'dev' or 'official'")
        if split == "official" and not self._scale(scale).official_evaluated:
            raise GymError("POLICY_DENIED", f"the official split is not evaluated at scale {scale}")

    def _files(self, files):
        if not isinstance(files, dict): raise GymError("INVALID_ARGUMENT", "files must map file names to text")
        allowed = set(REQUIRED_FILES + OPTIONAL_FILES)
        extra = sorted(set(files) - allowed)
        missing = sorted(set(REQUIRED_FILES) - set(files))
        if extra or missing:
            raise GymError("SPEC_VIOLATION", "submission files", details=dict(unexpected=extra, missing=missing))
        size = sum(len(v.encode() if isinstance(v, str) else v) for v in files.values())
        if size > TASK.policy.submission_max_bytes:
            raise GymError("PAYLOAD_TOO_LARGE", f"submission is {size} bytes",
                           details=dict(limit=TASK.policy.submission_max_bytes))
        return {k: v.encode() if isinstance(v, str) else v for k, v in files.items()}

    def _run_dir(self, run_id):
        path = self.home / "runs" / run_id
        path.mkdir(parents=True, exist_ok=True)
        return path

    # ---- observation ------------------------------------------------------------------
    def get_task(self):
        out = TASK.describe()
        out["policy"] = dict(require_s_before_l=TASK.policy.require_s_before_l,
                             seed_offsets_at_l=TASK.policy.seed_offsets_at_l,
                             max_concurrent_runs=TASK.policy.max_concurrent_runs,
                             max_run_share_of_cap=TASK.policy.max_run_share_of_cap,
                             estimate_cost_per_hour=TASK.policy.estimate_cost_per_hour,
                             submission_max_bytes=TASK.policy.submission_max_bytes)
        return out

    def get_budget(self):
        return self.ledger.snapshot()

    def get_leaderboard(self, scale, split="dev", limit=20, include_imported=True):
        self._check_split(scale, split)
        self.poll()
        board = self.archive.leaderboard(scale, split=split, limit=limit, include_imported=include_imported, seal=True)
        visible = ["dev", "official"] if self._scale(scale).official_evaluated else ["dev"]
        board["entries_per_split"] = {s: len(self.archive.leaderboard(scale, split=s, limit=10**6, seal=True)["entries"])
                                      for s in visible}
        notes = []
        if not board["entries"] and any(board["entries_per_split"].values()):
            notes.append(f"no entries on the {split} split; try " +
                         ", ".join(s for s, n in board["entries_per_split"].items() if n))
        if board["anchor_val_bpb"] is None:
            notes.append(f"no Dense anchor run at scale {scale} yet (admin: cli anchor); delta_vs_anchor is null")
        if split == "official":
            notes.append("imported runs carry `accepted`: false means upstream has not accepted the record as complete")
            notes.append(SEALED_HINT)
        board["notes"] = notes
        return board

    def list_attempts(self, **filters):
        return self.archive.list_attempts(**filters, seal=True)

    def get_attempt(self, attempt_id, include=()):
        attempt = self.archive.get_attempt(attempt_id, include=tuple(include), seal=True)
        if "runs" in include: attempt["run_details"] = [self._visible(r) for r in attempt.get("run_details", [])]
        return attempt

    def _visible(self, run):
        """Remove metrics the agent may not see: unevaluated splits and sealed official results."""
        scale = SCALES[run["scale"]]
        sealed = self.archive.sealed(run)
        run.pop("handle", None)
        run["extra"] = {k: v for k, v in (run.get("extra") or {}).items() if k not in ("run_dir", "url")}
        final = dict(run.get("final") or {})
        if sealed or not scale.official_evaluated: final.pop("val_bpb_official", None)
        if sealed or not scale.core: final.pop("core", None)
        run["final"] = final or None
        if sealed and run.get("latest"):
            run["latest"] = {k: v for k, v in run["latest"].items() if k not in ("val_bpb_official", "core")} or None
        if sealed: run["sealed"] = SEALED_HINT
        return run

    def get_run(self, run_id):
        self.poll(force=True, only=run_id)
        return self._visible(self.archive.get_run(run_id))

    def get_run_curve(self, run_id, metric="val_bpb", split="dev", max_points=200):
        run = self.archive.get_run(run_id)
        self._check_split(run["scale"], split)
        if split == "official" and self.archive.sealed(run): raise GymError("POLICY_DENIED", SEALED_HINT)
        if metric not in ("train_loss", "val_bpb"): raise GymError("INVALID_ARGUMENT", "metric is train_loss or val_bpb")
        self.poll(only=run_id)
        return self.archive.get_curve(run_id, metric, split=split, max_points=max_points)

    def get_run_logs(self, run_id, tail=200, grep=None):
        run = self.archive.get_run(run_id)
        tail = max(1, min(int(tail), 2000))
        if not run.get("handle"): return dict(run_id=run_id, lines=[], truncated=False)
        lines = self._backend(run["backend_id"]).logs(run["handle"], tail=tail if not grep else 2000)
        if self.archive.sealed(run): lines = [line for line in lines if not SEALED_LOG.search(line)]
        if grep: lines = [line for line in lines if grep in line][-tail:]
        redact = [(re.compile(re.escape(str(Path.home()))), "~"), (re.compile(r"/(home|data|persistent|mnt)\S*"), "<path>"),
                  (re.compile(r"(?i)(token|key|secret|password)=\S+"), r"\1=<redacted>")]
        for pattern, repl in redact: lines = [pattern.sub(repl, line) for line in lines]
        return dict(run_id=run_id, lines=lines, truncated=len(lines) >= tail)

    def compare_attempts(self, attempt_ids, scale, metric="val_bpb"):
        if not 2 <= len(attempt_ids) <= 8: raise GymError("INVALID_ARGUMENT", "compare 2 to 8 attempts")
        self._scale(scale)
        series, summary, first = [], [], None
        for attempt_id in attempt_ids:
            runs = [r for r in self.archive.list_runs(attempt_id=attempt_id, scale=scale) if r["status"] == "succeeded"]
            runs = runs or self.archive.list_runs(attempt_id=attempt_id, scale=scale)
            if not runs:
                series.append(dict(attempt_id=attempt_id, run_id=None, points=[])); continue
            run = runs[-1]
            has_dev = bool(self.archive.get_curve(run["run_id"], metric, split="dev", max_points=2)["points"])
            official = SCALES[scale].official_evaluated and not self.archive.sealed(run)
            split = "dev" if has_dev or not official else "official"
            points = self.archive.get_curve(run["run_id"], metric, split=split, max_points=200)["points"]
            final = points[-1]["value"] if points else None
            first = final if first is None else first
            series.append(dict(attempt_id=attempt_id, run_id=run["run_id"], split=split, points=points))
            summary.append(dict(attempt_id=attempt_id, final=final,
                                delta_vs_first=None if final is None or first is None else final - first))
        return dict(scale=scale, metric=metric, aligned_by="input_tokens", series=series, summary=summary)

    # ---- proposal and verification ----------------------------------------------------------
    def estimate_cost(self, files):
        files = self._files(files)
        self.ledger.check_rate("estimate_cost", TASK.policy.estimate_cost_per_hour)   # counts this use
        with tempfile.TemporaryDirectory() as tmp:
            for name, data in files.items(): (Path(tmp) / name).write_bytes(data)
            result = _run_verify(tmp)
        errors = [e for level in result["levels"] for e in level["errors"]]
        return dict(ok=result["status"] == "passed", audit=result.get("audit"), levels=result["levels"], errors=errors)

    def submit_attempt(self, files, *, title, hypothesis, parent_ids=(), tags=(), idempotency_key, wait=False):
        files = self._files(files)
        for parent in parent_ids: self.archive.get_attempt(parent)
        request = _sha(dict(files={k: hashlib.sha256(v).hexdigest() for k, v in files.items()}, title=title,
                            hypothesis=hypothesis, parent_ids=list(parent_ids), tags=list(tags)))
        cached = self.store.idempotent("submit", idempotency_key, request)
        if cached is not None: return cached
        config = json.loads(files["config.json"])
        similar = [(a, s) for a, s in self.archive.similar(files, k=3) if s >= TASK.policy.duplicate_similarity
                   and json.loads(self.archive.read_files(a)["config.json"]) == config]
        if similar:
            raise GymError("DUPLICATE_ATTEMPT", "an existing attempt is nearly identical",
                           details=dict(duplicate_of=similar[0][0], similarity=round(similar[0][1], 4)),
                           hint="change the architecture, or reuse the existing attempt")
        attempt_id = self.archive.put_attempt(files, title=title, hypothesis=hypothesis, parent_ids=list(parent_ids),
                                              tags=list(tags), origin="agent")
        cursor = self.store.last_event_seq()
        verification_id = self.archive.create_verification(attempt_id)
        self.ledger.record_check("verification")
        response = dict(attempt_id=attempt_id, status="verifying", verification_id=verification_id,
                        events_cursor=cursor)
        self.store.remember("submit", idempotency_key, request, response)
        job = lambda: self.archive.record_verification(attempt_id, _run_verify(self.archive.attempt_dir(attempt_id)),
                                                       verification_id)
        if wait: job()
        else: threading.Thread(target=job, daemon=True, name=f"verify-{attempt_id}").start()
        return response

    def get_verification(self, verification_id):
        return self.archive.get_verification(verification_id)

    def calibrate(self, attempt_id, knobs, target_relative_error=0.0):
        attempt = self.archive.get_attempt(attempt_id, include=("audit",))
        verification = self.archive.get_verification(attempt["verification_id"]) if attempt.get("verification_id") else None
        levels = {l["level"]: l for l in (verification or {}).get("levels", [])}
        if attempt["status"] == "verifying":
            raise GymError("NOT_VERIFIED", "verification is still running",
                           hint="wait_events(events_cursor, ['verification.finished']) and retry")
        if any(levels.get(l, {}).get("status") != "passed" for l in ("V0", "V1", "V2")):
            raise GymError("NOT_VERIFIED", "calibration needs an attempt that passed V0-V2")
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp) / "calibration.json"
            _python_module("nextt_gym.calibrate", self.archive.attempt_dir(attempt_id), "--knobs", ",".join(knobs),
                           "--target", target_relative_error, "--out", out, timeout=VERIFY_TIMEOUT_S)
            if not out.exists(): raise GymError("INTERNAL", "calibration crashed")
            result = json.loads(out.read_text())
        if not result["ok"]: raise GymError(result["error"]["code"], result["error"]["message"],
                                            details=result["error"].get("details"))
        files = self.archive.read_files(attempt_id)
        files["config.json"] = (json.dumps(result["config"], indent=2) + "\n").encode()
        new_id = self.archive.put_attempt(files, title=attempt["title"] + " (calibrated)", hypothesis=attempt["hypothesis"],
                                          parent_ids=[attempt_id], tags=list(attempt["tags"]) + ["calibrated"])
        verification_id = self.archive.create_verification(new_id)
        record = dict(status="passed", levels=[levels["V0"], levels["V1"], levels["V2"],
                                               dict(level="V3", status="passed", seconds=None, errors=[])],
                      audit=result["audit"], parameters=result["audit"]["parameters"])
        self.archive.record_verification(new_id, record, verification_id)
        self.ledger.record_check("verification")
        return dict(attempt_id=new_id, status="verified", verification_id=verification_id,
                    changed=result["changes"], audit=result["audit"])

    # ---- resources and execution ----------------------------------------------------------------
    def list_backends(self, gpu_type=None):
        return dict(backends=[b.describe() for b in self.backends.values() if gpu_type in (None, b.gpu_type)])

    def _trainable(self, attempt, scale):
        if attempt["status"] != "verified":
            raise GymError("NOT_VERIFIED", f"attempt {attempt['attempt_id']} is {attempt['status']}")
        if "gym-untrainable" in attempt["tags"]:
            raise GymError("POLICY_DENIED", "this upstream method cannot be trained by the gym trainer")
        audit = attempt.get("audit") or {}
        scale_audit = (audit.get("scales") or {}).get(scale)
        if scale_audit is None or not scale_audit.get("within_tolerance"):
            raise GymError("FLOPS_OUT_OF_TOLERANCE", f"attempt is not within the FLOPs tolerance at scale {scale}",
                           details=dict(audit=scale_audit))
        return audit

    def quote_run(self, attempt_id, scale, backend_id, world_size=None):
        from .runspec import planned_cost
        self._scale(scale)
        attempt = self.archive.get_attempt(attempt_id, include=("audit",))
        backend = self._backend(backend_id)
        blockers = []
        world_size = world_size or backend.allowed_world_sizes[0]
        if world_size not in backend.allowed_world_sizes:
            raise GymError("INVALID_WORLD_SIZE", f"{backend_id} allows {backend.allowed_world_sizes}")
        if backend.describe()["status"] == "offline":
            raise GymError("BACKEND_UNAVAILABLE", f"{backend_id} is offline")
        audit = self._trainable(attempt, scale)
        if scale == "L" and TASK.policy.require_s_before_l and not any(
                r["status"] == "succeeded" for r in self.archive.list_runs(attempt_id=attempt_id, scale="S")):
            blockers.append(GymError("POLICY_DENIED", "an L run requires a succeeded S run of this attempt")
                            .to_dict())
        sub = load_submission(self.archive.attempt_dir(attempt_id)) if attempt["origin"] == "agent" else \
            load_submission(self.archive.attempt_dir(attempt_id))
        plan = planned_cost(sub, audit, scale)
        if backend.config.get("synthetic"):   # smoke backend: a few steps of the formal per-step cost
            steps = int(backend.config.get("smoke_steps", 4))
            per_step = max(int(s["total"]) for s in audit["stages"])
            plan = dict(train=steps * per_step, eval=plan["eval"] * steps // SCALES[scale].steps + 1,
                        eval_tokens=plan["eval_tokens"])
        margin = int((plan["train"] + plan["eval"]) * TASK.policy.reserve_margin)
        reserve = plan["train"] + plan["eval"] + margin
        snapshot = self.ledger.snapshot()
        if reserve > int(snapshot["available_flops_exact"] if "available_flops_exact" in snapshot else snapshot["available_flops"]):
            blockers.append(GymError("INSUFFICIENT_BUDGET", "not enough FLOPs budget",
                                     details=dict(required_flops=reserve, available_flops=snapshot["available_flops"])).to_dict())
        throughput = backend.config.get("throughput_flops_per_gpu_s") or PEAK_BF16.get(backend.gpu_type, 150e12) * ASSUMED_MFU
        gpu_hours = (plan["train"] + plan["eval"]) / throughput / 3600
        wall = gpu_hours / max(world_size, 1)
        segment = backend.max_wall_hours_per_segment
        payload = dict(attempt_id=attempt_id, scale=scale, backend_id=backend_id, world_size=world_size,
                       reserve_flops=float(reserve), reserve_flops_exact=str(reserve),
                       breakdown=dict(train=float(plan["train"]), eval=float(plan["eval"]), margin=float(margin)),
                       est_gpu_hours=round(gpu_hours, 2), est_wall_hours=round(wall, 2),
                       est_segments=int(wall // segment) + 1 if segment else 1,
                       budget_after_reserve=snapshot["available_flops"] - reserve,
                       feasible=not blockers, blockers=blockers)
        expires = (datetime.now(timezone.utc) + timedelta(minutes=QUOTE_TTL_MINUTES)).isoformat(timespec="seconds")
        with self.store.transaction():
            seq, quote_id = self.store.next_id("quotes", "q")
            payload.update(quote_id=quote_id, expires_at=expires)
            self.store.execute("INSERT INTO quotes(id, seq, created_at, expires_at, payload) VALUES (?, ?, ?, ?, ?)",
                               (quote_id, seq, now(), expires, json.dumps(payload)))
        return payload

    def launch_run(self, quote_id, idempotency_key, seed_offset=0):
        from .bundle import make_bundle
        from .runspec import make_run_spec, write_run_spec
        request = _sha(dict(quote_id=quote_id, seed_offset=seed_offset))
        cached = self.store.idempotent("launch", idempotency_key, request)
        if cached is not None: return cached
        row = self.store.one("SELECT * FROM quotes WHERE id=?", (quote_id,))
        if row is None: raise GymError("NOT_FOUND", f"quote {quote_id} does not exist")
        quote = json.loads(row["payload"])
        if row["used"]: raise GymError("RUN_CONFLICT", "this quote was already used")
        if row["expires_at"] < datetime.now(timezone.utc).isoformat(timespec="seconds"):
            raise GymError("QUOTE_EXPIRED", "quote expired; request a new one")
        if not quote["feasible"]:
            first = quote["blockers"][0]
            raise GymError(first["code"], first["message"], details=first.get("details"))
        if quote["scale"] == "L" and seed_offset and not TASK.policy.seed_offsets_at_l:
            raise GymError("POLICY_DENIED", "L runs use the protocol seed; seed offsets are for S runs")
        active = [r for r in self.archive.list_runs() if r["status"] not in TERMINAL and r["origin"] == "agent"]
        if len(active) >= TASK.policy.max_concurrent_runs:
            raise GymError("POLICY_DENIED", f"at most {TASK.policy.max_concurrent_runs} concurrent runs")
        if any(r["attempt_id"] == quote["attempt_id"] and r["scale"] == quote["scale"] and r["seed_offset"] == seed_offset
               for r in active):
            raise GymError("RUN_CONFLICT", "the same attempt, scale and seed is already running")
        backend = self._backend(quote["backend_id"])
        attempt = self.archive.get_attempt(quote["attempt_id"], include=("audit",))
        audit = self._trainable(attempt, quote["scale"])
        reserve = int(quote["reserve_flops_exact"])
        run_id = self.archive.create_run(quote["attempt_id"], quote["scale"], backend_id=backend.backend_id,
                                         gpu_type=backend.gpu_type, world_size=quote["world_size"],
                                         reserved_flops=reserve, quote_id=quote_id, seed_offset=seed_offset)
        try:
            self.ledger.reserve(run_id, reserve, scale=quote["scale"], backend_id=backend.backend_id,
                                gpu_type=backend.gpu_type)
        except GymError as exc:
            self.archive.update_run(run_id, status="failed", failure=exc.to_dict()); raise
        self.store.execute("UPDATE quotes SET used=1 WHERE id=?", (quote_id,))
        try:
            run_dir = self._run_dir(run_id)
            directory = self.archive.attempt_dir(quote["attempt_id"])
            sub = load_submission(directory)
            synthetic = bool(backend.config.get("synthetic"))
            spec = make_run_spec(sub, audit, quote["scale"], run_id, quote["world_size"], reserve, seed_offset,
                                 synthetic=synthetic)
            if synthetic:  # smoke backend: few steps, debug-size model, frequent evaluation
                steps = int(backend.config.get("smoke_steps", 4))
                spec.update(steps=steps, val_every=max(1, steps // 2))
            write_run_spec(spec, run_dir / "run_spec.json")
            bundle, sha = make_bundle(REPO_ROOT, directory, run_dir)
            handle = backend.submit(run_dir / "backend", bundle, sha, run_dir / "run_spec.json",
                                    world_size=quote["world_size"], synthetic=synthetic,
                                    model_overrides=sub.debug_overrides if synthetic and not
                                    backend.config.get("smoke_full_size") else None)
        except Exception as exc:
            failure = exc.to_dict() if isinstance(exc, GymError) else GymError(
                "BACKEND_UNAVAILABLE", f"{type(exc).__name__}: {exc}").to_dict()
            self.archive.update_run(run_id, status="failed", failure=failure)
            self.ledger.settle(run_id, 0)
            raise GymError(failure["code"], failure["message"]) from exc
        cursor = self.store.last_event_seq()
        self.archive.update_run(run_id, status="queued", handle=handle)
        response = dict(run_id=run_id, status="queued", reserved_flops=float(reserve), events_cursor=cursor,
                        budget={k: v for k, v in self.ledger.snapshot().items() if k in ("available_flops", "reserved_flops")})
        self.store.remember("launch", idempotency_key, request, response)
        return response

    def cancel_run(self, run_id, reason):
        run = self.archive.get_run(run_id)
        if run["status"] in TERMINAL: raise GymError("ALREADY_TERMINAL", f"run {run_id} is {run['status']}")
        if run.get("handle"): self._backend(run["backend_id"]).cancel(run["handle"])
        self._refresh(run, terminal_status="cancelled", reason=reason)
        run = self.archive.get_run(run_id)
        settled = self.ledger.settle(run_id, int(run["consumed"]["flops_exact"]))
        return dict(run_id=run_id, status=run["status"], consumed=run["consumed"],
                    refunded_flops=float(settled["refunded"]))

    # ---- run monitoring ------------------------------------------------------------------------
    def poll(self, force=False, only=None):
        if not force and time.monotonic() - self._last_poll < POLL_SECONDS: return
        with self._lock:
            self._last_poll = time.monotonic()
            for run in self.archive.list_runs():
                if run["status"] in TERMINAL or run["origin"] != "agent" or not run.get("handle"): continue
                if only and run["run_id"] != only: continue
                try: self._refresh(run)
                except Exception as exc:   # a flaky backend must not break observation tools
                    self.archive.append_event("run.poll_error", dict(run_id=run["run_id"]), dict(error=str(exc)[:500]))

    def _refresh(self, run, terminal_status=None, reason=None):
        backend = self._backend(run["backend_id"])
        state = backend.status(run["handle"])
        local = self._run_dir(run["run_id"]) / "fetched"
        backend.fetch(run["handle"], local)
        budget = json.loads((local / "budget.json").read_text()) if (local / "budget.json").exists() else {}
        consumed = int(budget.get("consumed_flops") or state.get("consumed_flops") or 0)
        gpu_hours = float(budget.get("gpu_hours") or state.get("gpu_hours") or 0.0)
        charge = self.ledger.charge(run["run_id"], consumed, gpu_hours, run["gpu_type"],
                                    breakdown=dict(train=int(budget.get("consumed_train_flops") or 0),
                                                   eval=int(budget.get("consumed_eval_flops") or 0)))
        points, latest, evals = [], dict(run.get("latest") or {}), {}
        metrics = local / "metrics.jsonl"
        train_tokens = {0: 0}
        per_step_tokens = SCALES[run["scale"]].input_tokens // SCALES[run["scale"]].steps
        if metrics.exists():
            for line in metrics.read_text().splitlines():
                try: event = json.loads(line)
                except json.JSONDecodeError: continue
                if event.get("event") == "train" and event.get("input_tokens") is not None:
                    train_tokens[event["step"]] = event["input_tokens"]
            if len(train_tokens) > 1:
                per_step_tokens = max(v // k for k, v in train_tokens.items() if k)
            for line in metrics.read_text().splitlines():
                try: event = json.loads(line)
                except json.JSONDecodeError: continue
                if event.get("event") == "train":
                    points.append(dict(metric="train_loss", split="dev", step=event["step"],
                                       input_tokens=event.get("input_tokens"), value=event.get("lm_loss", event.get("loss"))))
                    latest.update(train_loss=event.get("lm_loss", event.get("loss")), at_step=event["step"],
                                  **{k: event[k] for k in ("step_seconds", "tokens_per_second") if event.get(k) is not None})
                    if event.get("peak_memory_bytes"): latest["peak_memory_gb"] = round(event["peak_memory_bytes"] / 2**30, 2)
                elif event.get("event") == "validation":
                    split = event.get("split", "dev")
                    trained = train_tokens.get(event["step"], event["step"] * per_step_tokens)
                    points.append(dict(metric="val_bpb", split=split, step=event["step"],
                                       input_tokens=trained, value=event.get("val_bpb", event.get("bpb"))))
                    if split == "dev":
                        latest.update(val_bpb_dev=event.get("val_bpb", event.get("bpb")), at_step=event["step"])
                        evals[event["step"]] = event.get("val_bpb", event.get("bpb"))
        # One run.eval per dev evaluation: the stored curve holds those already announced, including the ones a
        # restart replays.
        stored = {p["step"] for p in self.archive.get_curve(run["run_id"], "val_bpb", "dev", max_points=10**9)["points"]}
        if points: self.archive.add_curve_points(run["run_id"], points)
        progress = dict(step=int(budget.get("step") or state.get("step") or 0), total_steps=SCALES[run["scale"]].steps,
                        input_tokens=int(budget.get("input_tokens") or 0))
        status = terminal_status or {"queued": "queued", "running": "running", "succeeded": "succeeded",
                                     "failed": "failed", "cancelled": "cancelled"}[state["state"]]
        phase = state.get("phase")
        if phase == "budget_exceeded" or charge.get("over"): status = "budget_exceeded"
        fields = dict(progress=progress, latest=latest, gpu_hours=gpu_hours)
        for step in sorted(set(evals) - stored):
            self.archive.append_event("run.eval", dict(run_id=run["run_id"], attempt_id=run["attempt_id"]),
                                      dict(step=step, val_bpb_dev=evals[step]))
        restarts, known = state.get("restarts") or [], (run.get("extra") or {}).get("restarts") or []
        if len(restarts) > len(known):
            for restart in restarts[len(known):]:
                self.archive.append_event("run.restarted", dict(run_id=run["run_id"], attempt_id=run["attempt_id"]),
                                          restart)
            fields["extra"] = dict(run.get("extra") or {}, restarts=restarts)
        if status in TERMINAL:
            final = (state.get("final") or {})
            status_file = local / "run_status.json"
            if status_file.exists(): final = json.loads(status_file.read_text()).get("final") or final
            fields.update(final=final or None)
            if status == "failed":
                tail = [l for l in backend.logs(run["handle"], tail=400) if "Error" in l or "error:" in l][-5:]
                fields.update(failure=dict(code="INTERNAL", message=f"run ended in phase {phase}",
                                           details=dict(log_errors=tail)))
            if status == "cancelled": fields.update(failure=dict(code="ALREADY_TERMINAL", message=reason or "cancelled"))
        if status != run["status"]: fields["status"] = status
        self.archive.update_run(run["run_id"], consumed_flops=consumed, **fields)
        if status in TERMINAL and terminal_status is None:
            self.ledger.settle(run["run_id"], consumed, gpu_hours=gpu_hours)

    # ---- events and memory ----------------------------------------------------------------------
    def wait_events(self, cursor=None, types=None, timeout_s=30):
        deadline = time.monotonic() + max(0, min(int(timeout_s), 60))
        while True:
            self.poll(force=True)
            result = self.store.events_since(cursor, types=types)
            for event in result["events"]:
                final = (event["data"] or {}).get("final")
                if final and "run_id" in event["ref"] and self.archive.sealed(self.archive.get_run(event["ref"]["run_id"])):
                    event["data"]["final"] = {k: v for k, v in final.items() if k not in ("val_bpb_official", "core")}
            if result["events"] or time.monotonic() >= deadline: return result
            if cursor is None: cursor = result["next_cursor"]
            time.sleep(2)

    def annotate_attempt(self, attempt_id, text, tags_add=None):
        return self.archive.add_note(attempt_id, text, tags_add)

    # ---- read-only knowledge ----------------------------------------------------------------------
    def knowledge_failures(self):
        lines = ["# Known failure modes", "", "## Gym attempts that failed", ""]
        for item in self.archive.list_attempts(status=["rejected"], limit=50)["items"]:
            lines.append(f"- {item['attempt_id']} {item['title']}")
        for run in self.archive.list_runs():
            if run["status"] in ("failed", "budget_exceeded"):
                lines.append(f"- {run['run_id']} ({run['attempt_id']}, {run['scale']}): {run['status']} "
                             f"{(run.get('failure') or {}).get('message', '')}")
        return "\n".join(lines) + "\n"

    def upstream_method(self, method):
        from .importer import config_paths, family
        paths = [p for p in config_paths(REPO_ROOT) if p.stem == method]
        if not paths: raise GymError("NOT_FOUND", f"unknown upstream method {method!r}")
        path = paths[0]
        raw = json.loads(path.read_text())
        proof = path.parent / raw["calibration"]["cost_record"]
        # The model and plugin files the method's cost audit executed; older proofs fall back to its family module.
        audited = (json.loads(proof.read_text()).get("method_source") or {}).get("files", []) if proof.exists() else []
        files = [name for name in audited if name.startswith(("nexttransformer/models/", "nexttransformer/methods/"))
                 or not name.startswith(("nanochat/", "nexttransformer/"))]
        if not files:
            module = REPO_ROOT / "nexttransformer/models" / f"{family(method)}.py"
            files = ["nexttransformer/models/lm.py", "nexttransformer/models/backbone.py"] + \
                ([module.relative_to(REPO_ROOT).as_posix()] if module.exists() else [])
        out = [f"# Upstream method `{method}` (read-only)", "", "## Frozen configuration", "",
               "```json", json.dumps(raw, indent=1), "```", ""]
        if proof.exists():
            p = json.loads(proof.read_text())
            out += ["## Cost proof summary", "", "```json", json.dumps(dict(
                flops_per_token=p.get("flops_per_token"), total_training_flops=str(p.get("total_training_flops")),
                parameters=p.get("parameters"),
                stages=[dict(stage=s["stage"], depth=s["depth"], optimizer_steps=s["optimizer_steps"],
                             total=str(s["cost"]["total"])) for s in p["stages"]]), indent=1), "```", ""]
        for name in files:
            out += [f"## `{name}`", "", "```python", (REPO_ROOT / name).read_text(), "```", ""]
        return "\n".join(out)
