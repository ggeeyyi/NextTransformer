"""SQLite persistence for the gym (attempts, runs, curves, events, ledger, quotes, idempotency).

Everything lives under GYM_HOME (env NEXTT_GYM_HOME, default ~/.nextt-gym):
`gym.db` plus materialized submission files in `attempts/<id>/`. FLOPs exceed
int64 (the default cap is ~2.3e21), so FLOP columns hold decimal TEXT and all
arithmetic is done with Python ints.
"""
from contextlib import contextmanager
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import sqlite3
import threading


def gym_home(path=None):
    return Path(path or os.environ.get("NEXTT_GYM_HOME") or Path.home() / ".nextt-gym").expanduser()


# Every event type the gym emits; wait_events filters on these (INTERFACE.md lists the same).
EVENT_TYPES = ("verification.finished", "attempt.revealed", "run.status", "run.eval", "run.restarted", "run.poll_error",
               "run.finished", "budget.low", "budget.settled")


def now():
    return datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z")


def dumps(value):
    return None if value is None else json.dumps(value, sort_keys=True, default=str)


def loads(text, default=None):
    return default if text is None else json.loads(text)


MIGRATIONS = [
    # version 1
    """
    CREATE TABLE attempts (
        id TEXT PRIMARY KEY, seq INTEGER UNIQUE NOT NULL, created_at TEXT NOT NULL,
        origin TEXT NOT NULL, title TEXT NOT NULL, hypothesis TEXT NOT NULL,
        tags TEXT NOT NULL, parent_ids TEXT NOT NULL, files_sha256 TEXT NOT NULL, files TEXT NOT NULL,
        status TEXT NOT NULL, verification_id TEXT, audit TEXT, parameters INTEGER);
    CREATE INDEX attempts_sha ON attempts(files_sha256);
    CREATE TABLE verifications (
        id TEXT PRIMARY KEY, seq INTEGER UNIQUE NOT NULL, attempt_id TEXT NOT NULL, status TEXT NOT NULL,
        created_at TEXT NOT NULL, finished_at TEXT, result TEXT);
    CREATE TABLE runs (
        id TEXT PRIMARY KEY, seq INTEGER UNIQUE NOT NULL, attempt_id TEXT NOT NULL, scale TEXT NOT NULL,
        origin TEXT NOT NULL, status TEXT NOT NULL, backend_id TEXT, gpu_type TEXT, world_size INTEGER,
        micro_batch INTEGER, quote_id TEXT, seed_offset INTEGER NOT NULL DEFAULT 0, handle TEXT,
        progress TEXT, consumed_flops TEXT NOT NULL DEFAULT '0', gpu_hours REAL NOT NULL DEFAULT 0,
        reserved_flops TEXT NOT NULL DEFAULT '0', latest TEXT, final TEXT, failure TEXT, extra TEXT,
        created_at TEXT NOT NULL, started_at TEXT, ended_at TEXT);
    CREATE INDEX runs_attempt ON runs(attempt_id);
    CREATE TABLE run_curves (
        run_id TEXT NOT NULL, metric TEXT NOT NULL, split TEXT NOT NULL, step INTEGER NOT NULL,
        input_tokens INTEGER, value REAL NOT NULL, PRIMARY KEY (run_id, metric, split, step));
    CREATE TABLE events (
        seq INTEGER PRIMARY KEY AUTOINCREMENT, at TEXT NOT NULL, type TEXT NOT NULL, ref TEXT, data TEXT);
    CREATE TABLE ledger_entries (
        entry_id TEXT PRIMARY KEY, kind TEXT NOT NULL, scale TEXT, state TEXT NOT NULL,
        reserved TEXT NOT NULL DEFAULT '0', consumed TEXT NOT NULL DEFAULT '0', breakdown TEXT,
        gpu_hours REAL NOT NULL DEFAULT 0, gpu_type TEXT, backend_id TEXT,
        created_at TEXT NOT NULL, updated_at TEXT NOT NULL);
    CREATE TABLE settings (key TEXT PRIMARY KEY, value TEXT NOT NULL);
    CREATE TABLE quotes (
        id TEXT PRIMARY KEY, seq INTEGER UNIQUE NOT NULL, created_at TEXT NOT NULL, expires_at TEXT NOT NULL,
        payload TEXT NOT NULL, used INTEGER NOT NULL DEFAULT 0);
    CREATE TABLE idempotency (
        key TEXT NOT NULL, scope TEXT NOT NULL, request_sha TEXT NOT NULL, response TEXT,
        created_at TEXT NOT NULL, PRIMARY KEY (scope, key));
    CREATE TABLE notes (
        id INTEGER PRIMARY KEY AUTOINCREMENT, attempt_id TEXT NOT NULL, at TEXT NOT NULL, text TEXT NOT NULL);
    CREATE TABLE counters (id INTEGER PRIMARY KEY AUTOINCREMENT, kind TEXT NOT NULL, at REAL NOT NULL);
    CREATE INDEX counters_kind ON counters(kind, at);
    """,
    # version 2: attempts whose sealed official results the operator has revealed
    """
    CREATE TABLE reveals (attempt_id TEXT PRIMARY KEY, at TEXT NOT NULL, reason TEXT NOT NULL);
    """,
    # version 3: platform GPU-hours (pod time x GPUs as the cluster records it, queueing excluded), null if unknown
    """
    ALTER TABLE runs ADD COLUMN platform_gpu_hours REAL;
    ALTER TABLE ledger_entries ADD COLUMN platform_gpu_hours REAL;
    """,
]


class Store:
    """One SQLite database; a re-entrant lock serializes writers inside the process."""

    def __init__(self, home=None):
        self.home = gym_home(home)
        self.home.mkdir(parents=True, exist_ok=True)
        (self.home / "attempts").mkdir(exist_ok=True)
        self.path = self.home / "gym.db"
        self.conn = sqlite3.connect(self.path, check_same_thread=False, isolation_level=None, timeout=30)
        self.conn.row_factory = sqlite3.Row
        self.conn.execute("PRAGMA journal_mode=WAL")
        self.conn.execute("PRAGMA foreign_keys=ON")
        self.lock = threading.RLock()
        self._depth = 0
        self.migrate()

    @property
    def attempts_dir(self):
        return self.home / "attempts"

    def migrate(self):
        with self.transaction():
            version = self.conn.execute("PRAGMA user_version").fetchone()[0]
            for number, script in enumerate(MIGRATIONS[version:], start=version + 1):
                for statement in script.split(";"):
                    if statement.strip(): self.conn.execute(statement)
                self.conn.execute(f"PRAGMA user_version={number}")

    @property
    def schema_version(self):
        return self.conn.execute("PRAGMA user_version").fetchone()[0]

    @contextmanager
    def transaction(self):
        """Nested-safe IMMEDIATE transaction (outermost level commits)."""
        with self.lock:
            outer = self._depth == 0
            if outer: self.conn.execute("BEGIN IMMEDIATE")
            self._depth += 1
            try:
                yield self.conn
            except BaseException:
                self._depth -= 1
                if outer: self.conn.execute("ROLLBACK")
                raise
            self._depth -= 1
            if outer: self.conn.execute("COMMIT")

    def execute(self, sql, params=()):
        with self.lock:
            return self.conn.execute(sql, params)

    def all(self, sql, params=()):
        with self.lock:
            return self.conn.execute(sql, params).fetchall()

    def one(self, sql, params=()):
        with self.lock:
            return self.conn.execute(sql, params).fetchone()

    def next_id(self, table, prefix, width=6):
        """Sequential human-readable ids: att_000001, run_000001, ..."""
        row = self.one(f"SELECT COALESCE(MAX(seq), 0) + 1 FROM {table}")
        return row[0], f"{prefix}_{row[0]:0{width}d}"

    # ---- settings ---------------------------------------------------------------
    def get_setting(self, key, default=None):
        row = self.one("SELECT value FROM settings WHERE key=?", (key,))
        return default if row is None else row[0]

    def set_setting(self, key, value):
        self.execute("INSERT INTO settings(key, value) VALUES (?, ?) ON CONFLICT(key) DO UPDATE SET value=excluded.value",
                     (key, str(value)))

    # ---- events -----------------------------------------------------------------
    def append_event(self, type, ref=None, data=None):
        if type not in EVENT_TYPES: raise ValueError(f"unknown event type {type!r}")
        cur = self.execute("INSERT INTO events(at, type, ref, data) VALUES (?, ?, ?, ?)",
                           (now(), type, dumps(ref or {}), dumps(data or {})))
        return cur.lastrowid

    def last_event_seq(self):
        return self.one("SELECT COALESCE(MAX(seq), 0) FROM events")[0]

    def events_since(self, cursor=None, types=None, limit=100):
        """Events with seq > cursor. cursor None = from now on (returns none, gives the current cursor)."""
        if cursor is None:
            return dict(events=[], next_cursor=self.last_event_seq())
        sql, params = "SELECT * FROM events WHERE seq > ?", [int(cursor)]
        if types:
            sql += f" AND type IN ({','.join('?' * len(types))})"; params += list(types)
        rows = self.all(sql + " ORDER BY seq LIMIT ?", params + [int(limit)])
        events = [dict(seq=r["seq"], at=r["at"], type=r["type"], ref=loads(r["ref"], {}), data=loads(r["data"], {}))
                  for r in rows]
        next_cursor = events[-1]["seq"] if events else int(cursor)
        if not events and types:
            next_cursor = max(int(cursor), self.last_event_seq())
        return dict(events=events, next_cursor=next_cursor)

    # ---- idempotency --------------------------------------------------------------
    def idempotent(self, scope, key, request_sha):
        """Previous response for (scope, key) or None. A different request with the same key conflicts."""
        from .errors import GymError
        row = self.one("SELECT request_sha, response FROM idempotency WHERE scope=? AND key=?", (scope, key))
        if row is None: return None
        if row["request_sha"] != request_sha:
            raise GymError("IDEMPOTENCY_CONFLICT", f"idempotency key {key!r} was used for a different request")
        return loads(row["response"])

    def remember(self, scope, key, request_sha, response):
        self.execute("INSERT OR REPLACE INTO idempotency(key, scope, request_sha, response, created_at) "
                     "VALUES (?, ?, ?, ?, ?)", (key, scope, request_sha, dumps(response), now()))

    def close(self):
        self.conn.close()
