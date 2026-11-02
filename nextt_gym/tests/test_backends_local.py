"""End-to-end: bundle -> local backend -> remote -> trainer (synthetic, CPU, one rank)."""
import json
import os
from pathlib import Path
import shutil
import subprocess
import time

from nextt_gym.backends import LocalBackend, load_backends
from nextt_gym.bundle import make_bundle

from ._smoke import DEBUG, REPO, env, spec, submission


def committed_clone(tmp_path):
    """A clone of HEAD with the working copy of nextt_gym committed (bundles need committed code)."""
    clone = tmp_path / "clone"
    subprocess.run(["git", "clone", "-q", "--local", "--no-hardlinks", str(REPO), str(clone)], check=True)
    shutil.rmtree(clone / "nextt_gym")
    shutil.copytree(REPO / "nextt_gym", clone / "nextt_gym",
                    ignore=shutil.ignore_patterns("__pycache__", "backends.local.toml"))
    for args in (["add", "nextt_gym"], ["commit", "-qm", "test", "--allow-empty"]):
        subprocess.run(["git", "-C", str(clone), "-c", "user.name=t", "-c", "user.email=t@t", *args],
                       check=True, capture_output=True)
    return clone


def test_load_backends(tmp_path):
    path = tmp_path / "b.toml"
    path.write_text('[local]\nkind = "local"\ngpu_type = "CPU"\nallowed_world_sizes = [1, 8]\n'
                    '[a800]\nkind = "ssh"\nhosts = ["h"]\nworkdir = "/w"\ngpu_type = "A800"\n')
    backends = load_backends(path)
    assert set(backends) == {"local", "a800"}
    assert backends["local"].describe()["allowed_world_sizes"] == [1, 8]
    assert load_backends(tmp_path / "missing.toml") == {}


def test_local_backend_synthetic_run(tmp_path):
    clone = committed_clone(tmp_path)
    sub = submission(tmp_path)
    bundle, sha = make_bundle(clone, sub, tmp_path / "bundles")
    spec_path = spec(tmp_path, sub, steps=4, run_id="local_smoke")
    for key, value in env(tmp_path).items():
        if key != "PYTHONPATH": os.environ[key] = value
    backend = LocalBackend("local", dict(gpu_type="CPU", allowed_world_sizes=[1]))
    handle = backend.submit(tmp_path / "runs" / "local_smoke", bundle, sha, spec_path, world_size=1, synthetic=True,
                            model_overrides=DEBUG)
    deadline = time.time() + 600
    while time.time() < deadline:
        state = backend.status(handle)
        if state["state"] in ("succeeded", "failed", "cancelled"): break
        time.sleep(1)
    assert state["state"] == "succeeded", "\n".join(backend.logs(handle, 80))
    assert state["phase"] == "smoke_complete" and state["step"] == 4
    fetched = backend.fetch(handle, tmp_path / "fetched")
    status = json.loads((fetched / "run_status.json").read_text())
    remote = json.loads((fetched / "remote_status.json").read_text())
    assert status["phase"] == "smoke_complete" and remote["phase"] == "exited" and remote["returncode"] == 0
    # The trainer ran from the unpacked bundle, not from this checkout.
    provenance = json.loads((fetched / "provenance.json").read_text())
    assert Path(provenance["submission"]).parent.name == "code"
