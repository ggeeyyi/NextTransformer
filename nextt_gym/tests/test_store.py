from nextt_gym.errors import GymError
from nextt_gym.store import MIGRATIONS, Store
import pytest


def test_schema_and_home(store, tmp_path):
    assert store.home == tmp_path / "home"
    assert (store.home / "gym.db").exists() and store.schema_version == len(MIGRATIONS) == 2
    assert store.one("PRAGMA journal_mode")[0] == "wal"
    # Reopening runs no migration twice.
    again = Store(store.home); assert again.schema_version == 2; again.close()


def test_big_flops_roundtrip(store):
    big = 10 * 4672398325 * 49152000000 * 3
    assert big > 2 ** 63
    store.set_setting("x", big)
    assert int(store.get_setting("x")) == big


def test_transaction_rollback(store):
    with pytest.raises(RuntimeError):
        with store.transaction():
            store.set_setting("a", 1)
            with store.transaction():
                store.set_setting("b", 2)
            raise RuntimeError
    assert store.get_setting("a") is None and store.get_setting("b") is None


def test_events_cursor(store):
    assert store.events_since() == dict(events=[], next_cursor=0)
    s1 = store.append_event("run.status", dict(run_id="r"), dict(status="queued"))
    s2 = store.append_event("budget.low", {}, {})
    out = store.events_since(0)
    assert [e["seq"] for e in out["events"]] == [s1, s2] and out["next_cursor"] == s2
    only = store.events_since(0, types=["budget.low"])
    assert [e["type"] for e in only["events"]] == ["budget.low"]
    assert store.events_since(s2)["events"] == []
    assert store.events_since()["next_cursor"] == s2


def test_idempotency(store):
    assert store.idempotent("scope", "k", "sha1") is None
    store.remember("scope", "k", "sha1", dict(id=1))
    assert store.idempotent("scope", "k", "sha1") == dict(id=1)
    with pytest.raises(GymError) as exc:
        store.idempotent("scope", "k", "sha2")
    assert exc.value.code == "IDEMPOTENCY_CONFLICT"
