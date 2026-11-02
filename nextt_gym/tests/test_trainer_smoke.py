import json
from pathlib import Path
import shutil

import torch

from ._smoke import COST, FORWARD, spec, submission, train


def events(output):
    return [json.loads(l) for l in (Path(output) / "metrics.jsonl").read_text().splitlines()]


def test_full_run_outputs_and_budget(tmp_path):
    sub = submission(tmp_path)
    out = tmp_path / "out"
    status = train(tmp_path, sub, spec(tmp_path, sub, steps=6), out)
    assert status["phase"] == "smoke_complete" and status["step"] == 6 and status["synthetic"]
    budget = json.loads((out / "budget.json").read_text())
    assert int(budget["consumed_train_flops"]) == 6 * COST
    evaluated = [e["input_tokens"] for e in events(out) if e["event"] == "validation"]
    assert budget["eval_tokens"] == sum(evaluated)
    assert int(budget["consumed_eval_flops"]) == sum(int(FORWARD * t) for t in evaluated)
    assert int(status["consumed_flops"]) == int(budget["consumed_train_flops"]) + int(budget["consumed_eval_flops"])
    kinds = [e["event"] for e in events(out)]
    assert kinds.count("train") == 6 and kinds[-1] == "final" and "validation" in kinds
    assert "val_bpb_dev" in status["final"]
    for name in ("provenance.json", "val_00000000_dev.json", "val_00000006_dev.json", "latest.json"):
        assert (out / name).exists(), name


def test_resume_matches_continuous_run(tmp_path):
    sub = submission(tmp_path)
    path = spec(tmp_path, sub, steps=6)
    full = tmp_path / "full"
    reference = train(tmp_path, sub, path, full)
    resumed = tmp_path / "resumed"
    shutil.copytree(full, resumed)
    for name in ("checkpoint_00000006", "run_status.json", "latest.json", "budget.json"):
        p = resumed / name
        shutil.rmtree(p) if p.is_dir() else p.unlink()
    status = train(tmp_path, sub, path, resumed)
    assert status["step"] == 6 and status["consumed_flops"] == reference["consumed_flops"]
    assert status["final"] == reference["final"]
    a = torch.load(full / "checkpoint_00000006" / "model.pt", weights_only=False)["model"]
    b = torch.load(resumed / "checkpoint_00000006" / "model.pt", weights_only=False)["model"]
    assert all(torch.equal(a[k], b[k]) for k in a)


def test_budget_exceeded_stops_before_the_update(tmp_path):
    sub = submission(tmp_path)
    out = tmp_path / "out"
    # Enough for the step-0 validation and three updates, not a fourth.
    status = train(tmp_path, sub, spec(tmp_path, sub, steps=6, reserved=3 * COST + 1000), out)
    assert status["phase"] == "budget_exceeded" and status["step"] == 3
    assert int(status["consumed_train_flops"]) == 3 * COST
    assert int(status["consumed_flops"]) <= 3 * COST + 1000
    assert any(e["event"] == "budget_exceeded" for e in events(out))


def test_dev_directory_links_are_made_by_one_rank(tmp_path):
    from nextt_gym.trainer import VALIDATION_SHARD, dev_directory
    data = tmp_path / "data"; data.mkdir()
    for name in ("shard_00001.parquet", VALIDATION_SHARD): (data / name).write_text(name)
    out = tmp_path / "out"; out.mkdir()
    assert dev_directory(out, data, "shard_00001.parquet", False) == out / "dev_data"
    assert not (out / "dev_data").exists()                    # other ranks wait at the caller's barrier
    for _ in range(2): directory = dev_directory(out, data, "shard_00001.parquet", True)
    assert sorted(p.name for p in directory.iterdir()) == sorted(["shard_00001.parquet", VALIDATION_SHARD])
    assert (directory / "shard_00001.parquet").read_text() == "shard_00001.parquet"


def test_dev_shard_must_match_the_pinned_sha256(tmp_path, monkeypatch):
    import hashlib
    import pytest
    from nextt_gym import trainer
    from nextt_gym.task import DEV_SHARD, DEV_SHARD_SHA256
    contract = Path(__file__).resolve().parents[2] / "evidence/checks/scoring-dense-full-a800-20261028/contract/reference.json"
    assert DEV_SHARD_SHA256 in contract.read_text()
    with pytest.raises(ValueError, match="missing"): trainer.check_dev_shard(tmp_path, DEV_SHARD)
    (tmp_path / DEV_SHARD).write_bytes(b"not the dev shard")
    with pytest.raises(ValueError, match="pinned"): trainer.check_dev_shard(tmp_path, DEV_SHARD)
    monkeypatch.setattr(trainer, "DEV_SHARD_SHA256", hashlib.sha256(b"not the dev shard").hexdigest())
    trainer.check_dev_shard(tmp_path, DEV_SHARD)
