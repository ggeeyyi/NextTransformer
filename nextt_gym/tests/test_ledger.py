import pytest

from nextt_gym.errors import GymError
from nextt_gym.task import TASK

CAP = int(TASK.policy.default_cap_flops)


def test_default_cap_exceeds_int64(ledger):
    assert ledger.cap() == CAP and CAP > 2 ** 63
    snap = ledger.snapshot()
    assert snap["available_flops_exact"] == str(CAP) and snap["used_flops"] == 0


def test_reserve_charge_settle(ledger, store):
    r = CAP // 10
    out = ledger.reserve("run_1", r, scale="L", gpu_type="A800", backend_id="b")
    assert out["reserved_flops_exact"] == str(r) and out["available_flops_exact"] == str(CAP - r)
    ledger.charge("run_1", r // 4, 1.5, "A800")
    ledger.charge("run_1", r // 4, 1.5, "A800")           # idempotent running total
    ledger.charge("run_1", r // 8, 1.0, "A800")           # never moves backwards
    t = ledger.totals()
    assert t["used"] == r // 4 and t["reserved"] == r - r // 4 and t["available"] == CAP - r
    assert ledger.charge("run_1", r + 1)["over"] is True
    settled = ledger.settle("run_1", r // 2, breakdown=dict(train=r // 2 - 10, eval=10))
    assert settled == dict(consumed=r // 2, refunded=r - r // 2)
    assert ledger.settle("run_1")["consumed"] == r // 2            # settle is idempotent
    snap = ledger.snapshot()
    assert snap["used_flops_exact"] == str(r // 2) and snap["reserved_flops"] == 0
    assert snap["by_kind"]["eval"] == 10 and snap["by_scale"]["L"] == float(r // 2)
    assert snap["gpu_hours"] == [dict(backend_id="b", gpu_type="A800", hours=1.5)]
    assert [e["type"] for e in store.events_since(0)["events"]] == ["budget.settled"]


def test_insufficient_and_share_limits(ledger):
    ledger.set_cap(1000)
    with pytest.raises(GymError) as exc:
        ledger.reserve("big", 300)
    assert exc.value.code == "POLICY_DENIED"
    for i in range(4): ledger.reserve(f"r{i}", 250)
    with pytest.raises(GymError) as exc:
        ledger.reserve("r4", 1)
    assert exc.value.code == "INSUFFICIENT_BUDGET" and exc.value.details["available_flops"] == 0
    with pytest.raises(GymError) as exc:
        ledger.reserve("r0", 200)
    assert exc.value.code == "RUN_CONFLICT"
    ledger.reserve("r0", 250)                               # same reservation is idempotent


def test_budget_low_event_and_direct_records(ledger, store):
    ledger.set_cap(1000)
    ledger.record_anchor(950, scale="S", gpu_hours=2, gpu_type="H100")
    types = [e["type"] for e in store.events_since(0)["events"]]
    assert types == ["budget.low"]
    snap = ledger.snapshot()
    assert snap["by_kind"]["anchor"] == 950 and snap["available_flops"] == 50


def test_rate_limit(ledger):
    for _ in range(3): ledger.check_rate("estimate_cost", 3)
    with pytest.raises(GymError) as exc:
        ledger.check_rate("estimate_cost", 3)
    assert exc.value.code == "RATE_LIMITED" and exc.value.details["retry_after_s"] > 0
    ledger.record_check()
    assert ledger.snapshot()["counters"]["verifications_today"] == 1
