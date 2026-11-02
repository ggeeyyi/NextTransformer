"""Attempts (immutable architecture submissions), runs, curves, notes, leaderboards and events.

Attempt files are materialized to GYM_HOME/attempts/<id>/ and never change; an
edit is a new attempt whose `parent_ids` point at its sources.
"""
import base64
from datetime import datetime
import hashlib
import json
from pathlib import Path
import re
import shutil
import tempfile

from .contract import OPTIONAL_FILES, REQUIRED_FILES, files_sha256
from .errors import GymError
from .store import dumps, loads, now
from .task import SCALES

ATTEMPT_STATUS = ("submitted", "verifying", "verified", "rejected")
RUN_JSON = ("progress", "latest", "final", "failure", "extra")
RUN_FIELDS = ("status", "backend_id", "gpu_type", "world_size", "micro_batch", "quote_id", "seed_offset", "handle",
              "gpu_hours", "platform_gpu_hours", "started_at", "ended_at", *RUN_JSON)
TERMINAL = ("succeeded", "failed", "cancelled", "budget_exceeded")
NOTE_LIMIT = 4096
_TOKEN = re.compile(r"[A-Za-z_][A-Za-z0-9_]*|\d+(?:\.\d+)?|[^\s\w]")


def _text(value):
    return value.decode() if isinstance(value, (bytes, bytearray)) else str(value)


def _shingles(files, n=5):
    """Normalized token 5-grams over model.py (comments dropped) and canonical config.json."""
    source = re.sub(r"#[^\n]*", "", _text(files.get("model.py", "")))
    tokens = _TOKEN.findall(source)
    try:
        tokens += _TOKEN.findall(json.dumps(json.loads(_text(files.get("config.json", "{}"))), sort_keys=True))
    except json.JSONDecodeError:
        tokens += _TOKEN.findall(_text(files.get("config.json", "")))
    return {tuple(tokens[i:i + n]) for i in range(max(len(tokens) - n + 1, 1))}


def _summary(audit):
    """Audit report without the heavy per-operator stage records."""
    if not audit: return audit
    return {k: v for k, v in audit.items() if k != "stage_records"}


def _hours(row):
    """Wall-clock hours from creation (queueing included) to the end, or to now while the run is open."""
    end = row["ended_at"] or now()
    start, stop = (datetime.fromisoformat(v.replace("Z", "+00:00")) for v in (row["created_at"], end))
    return round((stop - start).total_seconds() / 3600, 4)


class Archive:
    def __init__(self, store):
        self.store = store

    # ---- attempts -----------------------------------------------------------------
    def attempt_dir(self, attempt_id):
        path = self.store.attempts_dir / attempt_id
        if not path.is_dir(): raise GymError("NOT_FOUND", f"attempt {attempt_id} does not exist")
        return path

    def put_attempt(self, files, *, title, hypothesis, parent_ids=(), tags=(), origin="agent", idempotency_key=None):
        names = set(files)
        allowed = set(REQUIRED_FILES + OPTIONAL_FILES)
        if not set(REQUIRED_FILES) <= names or not names <= allowed:
            raise GymError("SPEC_VIOLATION", "a submission needs model.py and config.json (optional: kernels.py, NOTES.md)",
                           details=dict(files=sorted(names)))
        parent_ids = list(parent_ids or ()); tags = list(tags or ())
        for parent in parent_ids:
            if self.store.one("SELECT 1 FROM attempts WHERE id=?", (parent,)) is None:
                raise GymError("NOT_FOUND", f"parent attempt {parent} does not exist")
        with self.store.transaction():
            tmp = Path(tempfile.mkdtemp(prefix=".incoming-", dir=self.store.attempts_dir))
            try:
                for name, content in files.items():
                    (tmp / name).write_bytes(content if isinstance(content, (bytes, bytearray)) else str(content).encode())
                sha = files_sha256(tmp)
                request = hashlib.sha256(dumps(dict(sha=sha, title=title, hypothesis=hypothesis, parent_ids=parent_ids,
                                                    tags=tags, origin=origin)).encode()).hexdigest()
                if idempotency_key:
                    previous = self.store.idempotent("put_attempt", idempotency_key, request)
                    if previous is not None:
                        shutil.rmtree(tmp); return previous["attempt_id"]
                seq, attempt_id = self.store.next_id("attempts", "att")
                tmp.rename(self.store.attempts_dir / attempt_id)
            except BaseException:
                shutil.rmtree(tmp, ignore_errors=True); raise
            self.store.execute(
                "INSERT INTO attempts(id, seq, created_at, origin, title, hypothesis, tags, parent_ids, files_sha256, "
                "files, status) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 'submitted')",
                (attempt_id, seq, now(), origin, title, hypothesis, dumps(tags), dumps(parent_ids), sha,
                 dumps(sorted(names))))
            if idempotency_key: self.store.remember("put_attempt", idempotency_key, request, dict(attempt_id=attempt_id))
        return attempt_id

    def _row(self, attempt_id):
        row = self.store.one("SELECT * FROM attempts WHERE id=?", (attempt_id,))
        if row is None: raise GymError("NOT_FOUND", f"attempt {attempt_id} does not exist")
        return row

    def read_files(self, attempt_id):
        directory = self.attempt_dir(attempt_id)
        out = {}
        for name in loads(self._row(attempt_id)["files"]):
            data = (directory / name).read_bytes()
            try: out[name] = data.decode()
            except UnicodeDecodeError: out[name] = "base64:" + base64.b64encode(data).decode()
        return out

    def set_status(self, attempt_id, status, verification_id=None):
        if status not in ATTEMPT_STATUS: raise GymError("INVALID_ARGUMENT", f"unknown attempt status {status!r}")
        self._row(attempt_id)
        self.store.execute("UPDATE attempts SET status=?, verification_id=COALESCE(?, verification_id) WHERE id=?",
                           (status, verification_id, attempt_id))

    def create_verification(self, attempt_id):
        """A queued verification; the attempt moves to `verifying`."""
        with self.store.transaction():
            self._row(attempt_id)
            seq, vid = self.store.next_id("verifications", "ver")
            self.store.execute("INSERT INTO verifications(id, seq, attempt_id, status, created_at) VALUES (?, ?, ?, 'queued', ?)",
                               (vid, seq, attempt_id, now()))
            self.set_status(attempt_id, "verifying", vid)
        return vid

    def record_verification(self, attempt_id, result, verification_id=None):
        """Store a verify.verify() result; the attempt becomes verified or rejected."""
        status = "passed" if result.get("status") == "passed" else "failed"
        with self.store.transaction():
            self._row(attempt_id)
            if verification_id is None:
                seq, verification_id = self.store.next_id("verifications", "ver")
                self.store.execute("INSERT INTO verifications(id, seq, attempt_id, status, created_at) VALUES (?, ?, ?, ?, ?)",
                                   (verification_id, seq, attempt_id, status, now()))
            stored = dict(result, audit=_summary(result.get("audit")))
            self.store.execute("UPDATE verifications SET status=?, finished_at=?, result=? WHERE id=?",
                               (status, now(), dumps(stored), verification_id))
            self.store.execute("UPDATE attempts SET status=?, verification_id=?, audit=?, parameters=? WHERE id=?",
                               ("verified" if status == "passed" else "rejected", verification_id,
                                dumps(_summary(result.get("audit"))), result.get("parameters")
                                or (result.get("audit") or {}).get("parameters"), attempt_id))
            self.store.append_event("verification.finished", dict(attempt_id=attempt_id, verification_id=verification_id),
                                    dict(status=status, errors=[e for lvl in result.get("levels", []) for e in lvl.get("errors", [])]))
        return verification_id

    def get_verification(self, verification_id):
        row = self.store.one("SELECT * FROM verifications WHERE id=?", (verification_id,))
        if row is None: raise GymError("NOT_FOUND", f"verification {verification_id} does not exist")
        result = loads(row["result"], {}) or {}
        return dict(verification_id=row["id"], attempt_id=row["attempt_id"], status=row["status"],
                    created_at=row["created_at"], finished_at=row["finished_at"],
                    levels=result.get("levels", []), audit=result.get("audit"), parameters=result.get("parameters"))

    # ---- sealed official results ----------------------------------------------------
    def reveal(self, attempt_id, reason):
        """Operator only: show the official BPB and CORE of this attempt's gym runs to agents."""
        self._row(attempt_id)
        self.store.execute("INSERT OR REPLACE INTO reveals(attempt_id, at, reason) VALUES (?, ?, ?)",
                           (attempt_id, now(), reason))
        self.store.append_event("attempt.revealed", dict(attempt_id=attempt_id), dict(reason=reason))

    def revealed(self, attempt_id):
        return self.store.one("SELECT 1 FROM reveals WHERE attempt_id=?", (attempt_id,)) is not None

    def sealed(self, run):
        """A run's official results stay hidden from agents until its attempt is revealed: gym runs and the
        imported upstream runs alike."""
        return SCALES[run["scale"]].official_evaluated and not self.revealed(run["attempt_id"])

    def _best(self, attempt_id, split, seal=False):
        best = {}
        for run in self.list_runs(attempt_id=attempt_id, status="succeeded"):
            if seal and split == "official" and self.sealed(run): continue
            value = (run["final"] or {}).get(f"val_bpb_{split}")
            if value is not None and (run["scale"] not in best or value < best[run["scale"]]["val_bpb"]):
                best[run["scale"]] = dict(run_id=run["run_id"], val_bpb=value)
        return best

    def _attempt(self, row, seal=False):
        notes = self.store.all("SELECT at, text FROM notes WHERE attempt_id=? ORDER BY id", (row["id"],))
        runs = [r["id"] for r in self.store.all("SELECT id FROM runs WHERE attempt_id=? ORDER BY seq", (row["id"],))]
        return dict(attempt_id=row["id"], parent_ids=loads(row["parent_ids"], []), created_at=row["created_at"],
                    origin=row["origin"], title=row["title"], hypothesis=row["hypothesis"], tags=loads(row["tags"], []),
                    files_sha256=row["files_sha256"], files=loads(row["files"], []), status=row["status"],
                    verification_id=row["verification_id"], audit=loads(row["audit"]), parameters=row["parameters"],
                    runs=runs, best=self._best(row["id"], "dev"), best_official=self._best(row["id"], "official", seal),
                    notes=[dict(at=n["at"], text=n["text"]) for n in notes])

    def get_attempt(self, attempt_id, include=(), seal=False):
        out = self._attempt(self._row(attempt_id), seal)
        if "files" in include: out["file_contents"] = self.read_files(attempt_id)
        if "runs" in include: out["run_details"] = self.list_runs(attempt_id=attempt_id)
        if "lineage" in include: out["lineage"] = dict(ancestors=self.ancestors(attempt_id), children=self.children(attempt_id))
        if "audit" not in include and out["audit"]:
            out["audit"] = {k: out["audit"][k] for k in ("flops_per_token", "scales", "parameters", "within_tolerance",
                                                         "unknown_ops") if k in out["audit"]}
        return out

    def children(self, attempt_id):
        return [r["id"] for r in self.store.all("SELECT id, parent_ids FROM attempts ORDER BY seq")
                if attempt_id in loads(r["parent_ids"], [])]

    def ancestors(self, attempt_id):
        seen, stack = [], list(loads(self._row(attempt_id)["parent_ids"], []))
        while stack:
            a = stack.pop(0)
            if a in seen: continue
            seen.append(a)
            row = self.store.one("SELECT parent_ids FROM attempts WHERE id=?", (a,))
            if row: stack += loads(row["parent_ids"], [])
        return seen

    def descendants(self, attempt_id):
        seen, stack = [], self.children(attempt_id)
        while stack:
            a = stack.pop(0)
            if a in seen: continue
            seen.append(a); stack += self.children(a)
        return seen

    def list_attempts(self, status=None, origin=None, tags=None, ancestor_of=None, descendant_of=None, sort="created",
                      limit=50, cursor=None, seal=False):
        rows = self.store.all("SELECT * FROM attempts ORDER BY seq")
        statuses = [status] if isinstance(status, str) else status
        keep = set(self.ancestors(ancestor_of)) if ancestor_of else None
        under = set(self.descendants(descendant_of)) if descendant_of else None
        items = []
        for row in rows:
            if statuses and row["status"] not in statuses: continue
            if origin and row["origin"] != origin: continue
            if tags and not set(tags) <= set(loads(row["tags"], [])): continue
            if keep is not None and row["id"] not in keep: continue
            if under is not None and row["id"] not in under: continue
            a = self._attempt(row, seal)
            audit = a["audit"] or {}
            fpt = audit.get("flops_per_token") or ((audit.get("scales") or {}).get("L") or {}).get("flops_per_token")
            items.append(dict(attempt_id=a["attempt_id"], title=a["title"], status=a["status"], origin=a["origin"],
                              parent_ids=a["parent_ids"], tags=a["tags"], best=a["best"],
                              best_official=a["best_official"], flops_per_token=fpt, created_at=a["created_at"]))
        if sort == "created":
            items.reverse()   # newest first
        elif sort in [f"best_{s}" for s in SCALES]:
            scale = sort[len("best_"):]
            items.sort(key=lambda a: (a["best"].get(scale, {}).get("val_bpb") is None,
                                      a["best"].get(scale, {}).get("val_bpb") or 0.0))
        else:
            raise GymError("INVALID_ARGUMENT", f"unknown sort {sort!r}",
                           details=dict(available=["created", *(f"best_{s}" for s in SCALES)]))
        start = int(cursor or 0)
        page = items[start:start + int(limit)]
        nxt = start + len(page)
        return dict(items=page, next_cursor=str(nxt) if nxt < len(items) else None)

    def similar(self, files, k=5, *, exclude=()):
        """Most similar existing attempts by token-shingle Jaccard over model.py + config.json."""
        mine = _shingles(files)
        scores = []
        for row in self.store.all("SELECT id FROM attempts ORDER BY seq"):
            if row["id"] in exclude: continue
            try: other = _shingles(self.read_files(row["id"]))
            except GymError: continue
            union = len(mine | other)
            scores.append((row["id"], len(mine & other) / union if union else 1.0))
        scores.sort(key=lambda s: -s[1])
        return scores[:k]

    def add_note(self, attempt_id, text, tags_add=None):
        if len(text.encode()) > NOTE_LIMIT: raise GymError("INVALID_ARGUMENT", f"notes are limited to {NOTE_LIMIT} bytes")
        with self.store.transaction():
            row = self._row(attempt_id)
            self.store.execute("INSERT INTO notes(attempt_id, at, text) VALUES (?, ?, ?)", (attempt_id, now(), text))
            if tags_add:
                tags = loads(row["tags"], []) + [t for t in tags_add if t not in loads(row["tags"], [])]
                self.store.execute("UPDATE attempts SET tags=? WHERE id=?", (dumps(tags), attempt_id))
            count = self.store.one("SELECT COUNT(*) FROM notes WHERE attempt_id=?", (attempt_id,))[0]
        return dict(attempt_id=attempt_id, notes_count=count)

    # ---- runs ---------------------------------------------------------------------
    def create_run(self, attempt_id, scale, *, backend_id=None, gpu_type=None, world_size=None, reserved_flops=0,
                   quote_id=None, seed_offset=0, origin="agent", status="reserved", extra=None):
        if scale not in SCALES: raise GymError("INVALID_ARGUMENT", f"unknown scale {scale!r}")
        with self.store.transaction():
            self._row(attempt_id)
            seq, run_id = self.store.next_id("runs", "run")
            self.store.execute(
                "INSERT INTO runs(id, seq, attempt_id, scale, origin, status, backend_id, gpu_type, world_size, quote_id, "
                "seed_offset, reserved_flops, progress, extra, created_at) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                (run_id, seq, attempt_id, scale, origin, status, backend_id, gpu_type, world_size, quote_id,
                 int(seed_offset), str(int(reserved_flops)),
                 dumps(dict(step=0, total_steps=SCALES[scale].steps, input_tokens=0)), dumps(extra or {}), now()))
        return run_id

    def update_run(self, run_id, *, consumed_flops=None, **fields):
        unknown = set(fields) - set(RUN_FIELDS)
        if unknown: raise GymError("INVALID_ARGUMENT", f"unknown run fields {sorted(unknown)}")
        with self.store.transaction():
            before = self._run_row(run_id)
            sets, params = [], []
            for key, value in fields.items():
                sets.append(f"{key}=?"); params.append(dumps(value) if key in RUN_JSON else value)
            if consumed_flops is not None:
                sets.append("consumed_flops=?"); params.append(str(int(consumed_flops)))
            if fields.get("status") in TERMINAL and "ended_at" not in fields:
                sets.append("ended_at=?"); params.append(now())
            if fields.get("status") == "running" and before["started_at"] is None and "started_at" not in fields:
                sets.append("started_at=?"); params.append(now())
            if sets:
                self.store.execute(f"UPDATE runs SET {', '.join(sets)} WHERE id=?", params + [run_id])
            if "status" in fields and fields["status"] != before["status"]:
                kind = "run.finished" if fields["status"] in TERMINAL else "run.status"
                self.store.append_event(kind, dict(run_id=run_id, attempt_id=before["attempt_id"]),
                                        dict(status=fields["status"], previous=before["status"],
                                             final=fields.get("final"), failure=fields.get("failure")))
        return self.get_run(run_id)

    def _run_row(self, run_id):
        row = self.store.one("SELECT * FROM runs WHERE id=?", (run_id,))
        if row is None: raise GymError("NOT_FOUND", f"run {run_id} does not exist")
        return row

    @staticmethod
    def _run(row):
        consumed = int(row["consumed_flops"])
        return dict(run_id=row["id"], attempt_id=row["attempt_id"], scale=row["scale"], origin=row["origin"],
                    backend_id=row["backend_id"], gpu_type=row["gpu_type"], world_size=row["world_size"],
                    micro_batch=row["micro_batch"], status=row["status"], quote_id=row["quote_id"],
                    seed_offset=row["seed_offset"], handle=row["handle"], progress=loads(row["progress"], {}),
                    consumed=dict(flops=float(consumed), flops_exact=str(consumed), gpu_hours=row["gpu_hours"],
                                  platform_gpu_hours=row["platform_gpu_hours"], wall_hours=_hours(row)),
                    reserved_flops=float(int(row["reserved_flops"])), reserved_flops_exact=row["reserved_flops"],
                    latest=loads(row["latest"], {}), final=loads(row["final"]), failure=loads(row["failure"]),
                    extra=loads(row["extra"], {}), created_at=row["created_at"], started_at=row["started_at"],
                    ended_at=row["ended_at"])

    def get_run(self, run_id):
        return self._run(self._run_row(run_id))

    def list_runs(self, attempt_id=None, status=None, scale=None, origin=None):
        sql, params = "SELECT * FROM runs WHERE 1=1", []
        for column, value in (("attempt_id", attempt_id), ("scale", scale), ("origin", origin)):
            if value is not None: sql += f" AND {column}=?"; params.append(value)
        if status is not None:
            statuses = [status] if isinstance(status, str) else list(status)
            sql += f" AND status IN ({','.join('?' * len(statuses))})"; params += statuses
        return [self._run(r) for r in self.store.all(sql + " ORDER BY seq", params)]

    # ---- curves -------------------------------------------------------------------
    def add_curve_points(self, run_id, points):
        """points: dicts with step, metric, split, value and optional input_tokens (upsert)."""
        self._run_row(run_id)
        with self.store.transaction():
            for p in points:
                self.store.execute("INSERT OR REPLACE INTO run_curves(run_id, metric, split, step, input_tokens, value) "
                                   "VALUES (?, ?, ?, ?, ?, ?)",
                                   (run_id, p["metric"], p.get("split", "dev"), int(p["step"]),
                                    p.get("input_tokens"), float(p["value"])))

    def get_curve(self, run_id, metric, split="dev", max_points=200):
        self._run_row(run_id)
        rows = self.store.all("SELECT step, input_tokens, value FROM run_curves WHERE run_id=? AND metric=? AND split=? "
                              "ORDER BY step", (run_id, metric, split))
        points = [dict(step=r["step"], input_tokens=r["input_tokens"], value=r["value"]) for r in rows]
        max_points = max(int(max_points), 2)
        if len(points) > max_points:
            stride = (len(points) - 1) / (max_points - 1)
            points = [points[round(i * stride)] for i in range(max_points)]
        return dict(run_id=run_id, metric=metric, split=split, points=points)

    # ---- leaderboard -------------------------------------------------------------
    def leaderboard(self, scale, split="dev", limit=20, include_imported=True, seal=False):
        if scale not in SCALES: raise GymError("INVALID_ARGUMENT", f"unknown scale {scale!r}")
        key = f"val_bpb_{split}"
        best, seeds = {}, {}
        for run in self.list_runs(scale=scale, status="succeeded"):
            value = (run["final"] or {}).get(key)
            if value is None or (seal and split == "official" and self.sealed(run)): continue
            a = run["attempt_id"]
            seeds[a] = seeds.get(a, 0) + 1
            if a not in best or value < best[a][0]: best[a] = (value, run)
        attempts = {a: self._attempt(self._row(a), seal) for a in best}
        anchors = [v for a, (v, _) in best.items() if "anchor" in attempts[a]["tags"]]
        anchor = min(anchors) if anchors else None
        imported = [v for a, (v, _) in best.items() if attempts[a]["origin"] == "imported"]
        best_imported = min(imported) if imported else None
        rows = []
        for a, (value, run) in best.items():
            info = attempts[a]
            if not include_imported and info["origin"] == "imported": continue
            audit = info["audit"] or {}
            fpt = ((audit.get("scales") or {}).get(scale) or {}).get("flops_per_token") or audit.get("flops_per_token")
            rows.append(dict(attempt_id=a, title=info["title"], origin=info["origin"], scale=scale, run_id=run["run_id"],
                             val_bpb=value, delta_vs_anchor=None if anchor is None else value - anchor,
                             delta_vs_best_imported=None if best_imported is None else value - best_imported,
                             flops_per_token=fpt, parameters=info["parameters"], gpu_type=run["gpu_type"],
                             seeds=seeds[a], accepted=(run["final"] or {}).get("accepted", True)))
        rows.sort(key=lambda r: r["val_bpb"])
        for rank, r in enumerate(rows, 1): r["rank"] = rank
        return dict(scale=scale, split=split, anchor_val_bpb=anchor, entries=rows[:int(limit)], updated_at=now())

    # ---- events -------------------------------------------------------------------
    def append_event(self, type, ref=None, data=None):
        return self.store.append_event(type, ref, data)

    def events_since(self, cursor=None, types=None, limit=100):
        return self.store.events_since(cursor, types, limit)
