"""Total research-compute ledger: reserve before a run, charge while it runs, settle at the end.

One entry per run (entry_id = run_id) or per direct charge (anchor runs, checks).
`used` sums consumed FLOPs over all entries; `reserved` is the unconsumed part
of open reservations; `available = cap - used - reserved`. Theoretical FLOPs are
the primary account (Python ints, stored as decimal text); GPU-hours are a
secondary account per GPU type.
"""
import time

from .errors import GymError
from .store import dumps, loads, now
from .task import SCALES, TASK

KINDS = ("train", "eval", "check", "anchor")
KEEP = object()   # settle(): keep the stored platform reading
LOW_FRACTION = 0.10


class FlopsLedger:
    def __init__(self, store):
        self.store = store

    # ---- cap ----------------------------------------------------------------------
    def cap(self):
        value = self.store.get_setting("cap_flops")
        return int(value) if value is not None else int(TASK.policy.default_cap_flops)

    def set_cap(self, flops):
        flops = int(flops)
        if flops <= 0: raise GymError("INVALID_ARGUMENT", "the cap must be positive")
        self.store.set_setting("cap_flops", flops)
        self.store.set_setting("budget_low_emitted", 0)

    # ---- totals -------------------------------------------------------------------
    def _entries(self):
        return self.store.all("SELECT * FROM ledger_entries")

    def totals(self):
        used = reserved = 0
        for e in self._entries():
            consumed = int(e["consumed"]); used += consumed
            if e["state"] == "open": reserved += max(int(e["reserved"]) - consumed, 0)
        cap = self.cap()
        return dict(cap=cap, used=used, reserved=reserved, available=cap - used - reserved)

    def _maybe_low(self):
        t = self.totals()
        low = t["available"] < LOW_FRACTION * t["cap"]
        emitted = self.store.get_setting("budget_low_emitted", "0") == "1"
        if low and not emitted:
            self.store.append_event("budget.low", {}, dict(available_flops=float(t["available"]),
                                                           available_flops_exact=str(t["available"]),
                                                           cap_flops=float(t["cap"])))
            self.store.set_setting("budget_low_emitted", 1)
        elif not low and emitted:
            self.store.set_setting("budget_low_emitted", 0)

    # ---- run lifecycle ------------------------------------------------------------
    def reserve(self, run_id, flops, *, kind="train", scale=None, backend_id=None, gpu_type=None):
        flops = int(flops)
        if flops <= 0: raise GymError("INVALID_ARGUMENT", "a reservation must be positive")
        if kind not in KINDS: raise GymError("INVALID_ARGUMENT", f"unknown ledger kind {kind!r}")
        with self.store.transaction():
            existing = self.store.one("SELECT reserved FROM ledger_entries WHERE entry_id=?", (run_id,))
            if existing is not None:
                if int(existing["reserved"]) == flops: return dict(reserved=flops, **self._public(self.totals()))
                raise GymError("RUN_CONFLICT", f"{run_id} already holds a different reservation")
            t = self.totals()
            share = TASK.policy.max_run_share_of_cap
            if flops > share * t["cap"]:
                raise GymError("POLICY_DENIED", f"one run may reserve at most {share:.0%} of the cap",
                               details=dict(required_flops=float(flops), limit_flops=float(share * t["cap"])))
            if flops > t["available"]:
                raise GymError("INSUFFICIENT_BUDGET", "not enough research compute left for this run",
                               details=dict(available_flops=float(t["available"]), required_flops=float(flops),
                                            available_flops_exact=str(t["available"]), required_flops_exact=str(flops)))
            ts = now()
            self.store.execute(
                "INSERT INTO ledger_entries(entry_id, kind, scale, state, reserved, consumed, gpu_type, backend_id, "
                "created_at, updated_at) VALUES (?, ?, ?, 'open', ?, '0', ?, ?, ?, ?)",
                (run_id, kind, scale, str(flops), gpu_type, backend_id, ts, ts))
            self._maybe_low()
            return dict(reserved=flops, **self._public(self.totals()))

    def _open(self, run_id):
        row = self.store.one("SELECT * FROM ledger_entries WHERE entry_id=?", (run_id,))
        if row is None: raise GymError("NOT_FOUND", f"no ledger entry for {run_id}")
        return row

    def charge(self, run_id, consumed_total, gpu_hours=0.0, gpu_type=None, *, breakdown=None, platform_gpu_hours=None):
        """Record the running total consumed by a run. Idempotent; totals never move backwards.

        `gpu_hours` is the trainer's own training and evaluation time; `platform_gpu_hours` is the pod time the
        cluster charged (setup, compilation and restarts included, queueing excluded), None when unknown. A running
        run's platform reading is provisional: the latest known one replaces the stored one.
        """
        consumed_total = int(consumed_total)
        with self.store.transaction():
            row = self._open(run_id)
            if row["state"] != "open": return dict(consumed=int(row["consumed"]), over=False)
            consumed = max(int(row["consumed"]), consumed_total)
            hours = max(float(row["gpu_hours"]), float(gpu_hours or 0))
            platform = row["platform_gpu_hours"] if platform_gpu_hours is None else float(platform_gpu_hours)
            self.store.execute("UPDATE ledger_entries SET consumed=?, gpu_hours=?, platform_gpu_hours=?, "
                               "gpu_type=COALESCE(?, gpu_type), breakdown=COALESCE(?, breakdown), updated_at=? "
                               "WHERE entry_id=?",
                               (str(consumed), hours, platform, gpu_type, dumps(_ints(breakdown)), now(), run_id))
            self._maybe_low()
            return dict(consumed=consumed, over=consumed > int(row["reserved"]))

    def settle(self, run_id, final_consumed=None, *, gpu_hours=None, breakdown=None, platform_gpu_hours=KEEP):
        """Close a run's entry. `platform_gpu_hours`, when given, is the final measurement and replaces any
        provisional reading, None (unknown) included; left out, the stored reading stays (cancel_run, before the
        backend's final metering)."""
        with self.store.transaction():
            row = self._open(run_id)
            reserved = int(row["reserved"])
            if row["state"] == "settled":
                consumed = int(row["consumed"])
                return dict(consumed=consumed, refunded=max(reserved - consumed, 0))
            consumed = int(row["consumed"]) if final_consumed is None else int(final_consumed)
            hours = float(row["gpu_hours"]) if gpu_hours is None else float(gpu_hours)
            platform = row["platform_gpu_hours"] if platform_gpu_hours is KEEP else platform_gpu_hours
            self.store.execute("UPDATE ledger_entries SET consumed=?, gpu_hours=?, platform_gpu_hours=?, "
                               "breakdown=COALESCE(?, breakdown), state='settled', updated_at=? WHERE entry_id=?",
                               (str(consumed), hours, platform, dumps(_ints(breakdown)), now(), run_id))
            refunded = max(reserved - consumed, 0)
            t = self.totals()
            self.store.append_event("budget.settled", dict(run_id=run_id),
                                    dict(consumed_flops=float(consumed), consumed_flops_exact=str(consumed),
                                         refunded_flops=float(refunded), available_flops=float(t["available"])))
            self._maybe_low()
            return dict(consumed=consumed, refunded=refunded)

    def record(self, kind, flops, *, entry_id=None, scale=None, gpu_hours=0.0, gpu_type=None, backend_id=None):
        """A direct, already-settled charge (e.g. an anchor run imported after the fact)."""
        if kind not in KINDS: raise GymError("INVALID_ARGUMENT", f"unknown ledger kind {kind!r}")
        ts = now()
        entry_id = entry_id or f"{kind}-{time.time_ns()}"
        with self.store.transaction():
            self.store.execute(
                "INSERT INTO ledger_entries(entry_id, kind, scale, state, reserved, consumed, gpu_hours, gpu_type, "
                "backend_id, created_at, updated_at) VALUES (?, ?, ?, 'settled', ?, ?, ?, ?, ?, ?, ?)",
                (entry_id, kind, scale, str(int(flops)), str(int(flops)), float(gpu_hours), gpu_type, backend_id, ts, ts))
            self._maybe_low()
        return entry_id

    def meter(self, run_id, platform_gpu_hours):
        """Set a run's platform GPU-hours after the fact (`cli meter`), in any state; the FLOPs stay as they are."""
        self._open(run_id)
        self.store.execute("UPDATE ledger_entries SET platform_gpu_hours=?, updated_at=? WHERE entry_id=?",
                           (platform_gpu_hours, now(), run_id))

    def llm_usage(self):
        """LLM tokens and API cost of the research, as the operator recorded them (`cli llm-usage`); null if not."""
        return loads(self.store.get_setting("llm_usage"), dict(tokens=None, cost_usd=None, note=None))

    def record_anchor(self, flops, **kwargs):
        return self.record("anchor", flops, **kwargs)

    def record_check(self, kind="verification"):
        """Count a free CPU check (no FLOPs charged)."""
        self.store.execute("INSERT INTO counters(kind, at) VALUES (?, ?)", (kind, time.time()))

    # ---- rate limits ----------------------------------------------------------------
    def check_rate(self, kind, per_hour):
        """Count one use of `kind`; raise RATE_LIMITED when the last hour already has `per_hour` uses."""
        t = time.time()
        with self.store.transaction():
            rows = self.store.all("SELECT at FROM counters WHERE kind=? AND at > ? ORDER BY at", (kind, t - 3600))
            if len(rows) >= per_hour:
                retry = max(1, int(rows[0]["at"] + 3600 - t) + 1)
                raise GymError("RATE_LIMITED", f"{kind} is limited to {per_hour} per hour",
                               details=dict(retry_after_s=retry))
            self.store.execute("INSERT INTO counters(kind, at) VALUES (?, ?)", (kind, t))

    def count(self, kind, seconds):
        return self.store.one("SELECT COUNT(*) FROM counters WHERE kind=? AND at > ?",
                              (kind, time.time() - seconds))[0]

    # ---- snapshot -----------------------------------------------------------------
    @staticmethod
    def _public(t):
        out = {}
        for name in ("cap", "used", "reserved", "available"):
            out[f"{name}_flops"] = float(t[name]); out[f"{name}_flops_exact"] = str(t[name])
        return out

    def snapshot(self):
        t = self.totals()
        by_kind = {k: 0 for k in KINDS}
        by_scale = {k: 0 for k in SCALES}
        hours, platform, unmetered = {}, {}, 0
        for e in self._entries():
            consumed = int(e["consumed"])
            parts = _ints(loads(e["breakdown"])) if e["breakdown"] else None
            if parts:
                for kind, value in parts.items(): by_kind[kind] = by_kind.get(kind, 0) + value
                by_kind[e["kind"]] += max(consumed - sum(parts.values()), 0)
            else:
                by_kind[e["kind"]] = by_kind.get(e["kind"], 0) + consumed
            if e["scale"] in by_scale: by_scale[e["scale"]] += consumed
            key = (e["backend_id"], e["gpu_type"])
            if e["gpu_hours"]:
                hours[key] = hours.get(key, 0.0) + float(e["gpu_hours"])
            if e["platform_gpu_hours"] is not None:
                platform[key] = platform.get(key, 0.0) + float(e["platform_gpu_hours"])
            elif e["kind"] == "train": unmetered += 1
        return dict(
            **self._public(t),
            by_kind={k: float(v) for k, v in by_kind.items()},
            by_scale={k: float(v) for k, v in by_scale.items()},
            gpu_hours=[dict(backend_id=b, gpu_type=g, hours=hours.get((b, g), 0.0), platform_hours=platform.get((b, g)))
                       for b, g in sorted(set(hours) | set(platform), key=str)],
            metering=dict(trainer_gpu_hours=round(sum(hours.values()), 4),
                          platform_gpu_hours=round(sum(platform.values()), 4),
                          runs_without_platform_hours=unmetered, llm=self.llm_usage()),
            counters=dict(verifications_today=self.count("verification", 86400),
                          estimate_cost_last_hour=self.count("estimate_cost", 3600),
                          estimate_cost_limit_per_hour=TASK.policy.estimate_cost_per_hour))


def _ints(breakdown):
    return None if breakdown is None else {k: int(v) for k, v in breakdown.items()}
