import pytest

from nextt_gym.contract import load_submission, submission_from_hooks
from nextt_gym.runspec import CORE_TOKENS_ESTIMATE, make_run_spec, planned_cost, validate_run_spec, validation_steps
from nextt_gym.task import SCALES

from ._smoke import TEMPLATES

REPORT = dict(stages=[dict(name="joint", total=4899364746504805)], forward_flops_per_token=1.5e9)


def test_spec_fields_and_exact_strings():
    sub = load_submission(TEMPLATES)
    spec = make_run_spec(sub, REPORT, "L", "run_1", 8, 3 * 10 ** 20)
    assert spec["steps"] == SCALES["L"].steps and spec["world_size"] == 8
    assert spec["stage_costs"] == {"joint": "4899364746504805"}
    assert spec["reserved_flops"] == spec["stop_at_flops"] == str(3 * 10 ** 20)
    assert spec["official_evaluated"] and spec["core"] and spec["submission_sha256"]
    assert spec["dev_shard"] == "shard_06541.parquet" and spec["seed"] == 42


def test_world_size_and_budget_rules():
    sub = load_submission(TEMPLATES)
    with pytest.raises(ValueError, match="world_size"):
        make_run_spec(sub, REPORT, "S", "r", 4, 10 ** 20)
    make_run_spec(sub, REPORT, "S", "r", 1, 10 ** 20, synthetic=True)   # smoke may use any world size
    spec = make_run_spec(sub, REPORT, "S", "r", 8, 10 ** 20)
    spec["stop_at_flops"] = str(10 ** 21)
    with pytest.raises(ValueError, match="exceeds"):
        validate_run_spec(spec)
    spec["stop_at_flops"] = 5.0
    with pytest.raises(ValueError, match="exact decimal"):
        validate_run_spec(spec)


def test_missing_stage_cost_is_rejected():
    config = dict(model={}, model_dim=48)
    staged = submission_from_hooks(config, build=None,
                                   stage_for_step=lambda c, step, total: "burnin" if step < 10 else "joint")
    with pytest.raises(ValueError, match="burnin"):
        make_run_spec(staged, REPORT, "S", "r", 8, 10 ** 20)


def test_planned_cost():
    sub = load_submission(TEMPLATES)
    scale = SCALES["L"]
    plan = planned_cost(sub, REPORT, "L")
    assert plan["train"] == scale.steps * 4899364746504805
    tokens = 2 * (len(validation_steps(scale)) * scale.val_tokens + scale.final_val_tokens) + CORE_TOKENS_ESTIMATE
    assert plan["eval_tokens"] == tokens and plan["eval"] == int(1.5e9 * tokens)


def test_compressed_scale_runs_its_own_schedule():
    import torch
    from nexttransformer.native.optimizer import set_schedule
    from nextt_gym.trainer import Schedule
    sub = load_submission(TEMPLATES)
    c, s = (make_run_spec(sub, REPORT, name, "r", 8, 10 ** 20) for name in ("C", "S"))
    assert c["steps"] == s["steps"] == 512 and c["input_tokens"] == s["input_tokens"]
    assert c["schedule_steps"] == 512 and s["schedule_steps"] == SCALES["L"].steps
    optimizer = torch.optim.SGD([torch.zeros(1)], lr=1.0)
    optimizer.param_groups[0].update(initial_lr=1.0, kind="adamw")
    lr = {step: set_schedule(optimizer, step, Schedule(c["schedule_steps"])) for step in (39, 179, 180, 511)}
    assert lr[39] == lr[179] == 1.0 and lr[180] < 1.0 and abs(lr[511] - 0.05) < 0.01   # decays over the last 65%
    assert set_schedule(optimizer, 511, Schedule(s["schedule_steps"])) == 1.0           # S stops before L's decay
