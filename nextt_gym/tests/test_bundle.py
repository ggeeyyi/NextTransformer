from pathlib import Path
import subprocess

import pytest

from nextt_gym.bundle import make_bundle, unpack_bundle


def git(repo, *args):
    subprocess.run(["git", "-C", str(repo), "-c", "user.name=t", "-c", "user.email=t@t", *args],
                   check=True, capture_output=True)


def repo_with_gym(tmp_path):
    repo = tmp_path / "repo"; (repo / "nextt_gym").mkdir(parents=True)
    (repo / "nextt_gym" / "x.py").write_text("X = 1\n"); (repo / "README.md").write_text("r\n")
    git(repo, "init", "-q"); git(repo, "add", "."); git(repo, "commit", "-qm", "init")
    sub = tmp_path / "sub"; sub.mkdir()
    (sub / "model.py").write_text("def build(c): pass\n"); (sub / "config.json").write_text('{"model": {}, "model_dim": 8}')
    return repo, sub


def test_bundle_is_deterministic_and_unpacks(tmp_path):
    repo, sub = repo_with_gym(tmp_path)
    path, sha = make_bundle(repo, sub, tmp_path / "b1")
    path2, sha2 = make_bundle(repo, sub, tmp_path / "b2")
    assert sha == sha2 and path.name == path2.name
    dest = unpack_bundle(path, sha, tmp_path / "out")
    assert (dest / "nextt_gym" / "x.py").read_text() == "X = 1\n"
    assert (dest / "submission" / "model.py").exists() and (dest / "BUNDLE.json").exists()
    with pytest.raises(ValueError, match="checksum"):
        unpack_bundle(path, "0" * 64, tmp_path / "bad")


def test_uncommitted_gym_code_is_refused(tmp_path):
    repo, sub = repo_with_gym(tmp_path)
    (repo / "nextt_gym" / "x.py").write_text("X = 2\n")
    with pytest.raises(RuntimeError, match="uncommitted"):
        make_bundle(repo, sub, tmp_path / "b")
