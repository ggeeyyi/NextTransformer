import pytest

from nextt_gym.archive import Archive
from nextt_gym.ledger import FlopsLedger
from nextt_gym.store import Store


def pytest_configure(config):
    config.addinivalue_line("markers", "audit: gym audits that reproduce a representative set of upstream cost proofs")
    config.addinivalue_line("markers", "audit_full: gym audits of the methods outside the representative set")


@pytest.fixture
def store(tmp_path, monkeypatch):
    monkeypatch.setenv("NEXTT_GYM_HOME", str(tmp_path / "home"))
    s = Store()
    yield s
    s.close()


@pytest.fixture
def archive(store):
    return Archive(store)


@pytest.fixture
def ledger(store):
    return FlopsLedger(store)


@pytest.fixture(scope="session")
def upstream_gym(tmp_path_factory):
    """A committed clone and a gym home with the upstream methods imported, built once per process."""
    from nextt_gym import service as service_module
    from nextt_gym.importer import import_upstream
    from nextt_gym.service import GymService

    from .test_backends_local import committed_clone

    base = tmp_path_factory.mktemp("upstream")
    clone = committed_clone(base)
    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(service_module, "REPO_ROOT", clone)
        gym = GymService(home=base / "home", backends_path=base / "none.toml")
        import_upstream(gym.archive, clone)
        gym.store.close()          # checkpoints the WAL so the home directory can be copied
    return clone, base / "home"
