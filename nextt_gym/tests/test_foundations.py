"""Foundations: role-based optimizer groups and the generic audit reproduce upstream exactly."""
import json
from pathlib import Path
import types

import pytest
import torch

from nextt_gym import upstream_model
from nextt_gym.audit import audit_submission
from nextt_gym.calibrate import calibrate
from nextt_gym.contract import Submission, load_submission
from nextt_gym.importer import config_paths
from nextt_gym.roles import parameter_groups
from nextt_gym.verify import static_check, verify
from nexttransformer.models.registry import METHODS
from nexttransformer.models.config import HYBRID, debug_config
from nexttransformer.training.optimizer import parameter_groups as upstream_groups
from nexttransformer.models.lm import build_model

ROOT = Path(__file__).resolve().parents[2]
HOOKS = types.SimpleNamespace(build=upstream_model.build, stage_for_step=upstream_model.stage_for_step,
                              set_stage=upstream_model.set_stage)


def upstream_submission(method):
    path = next(p for p in config_paths(ROOT) if p.stem == method)
    raw = json.loads(path.read_text())
    return Submission(None, HOOKS, upstream_model.submission_config(raw["model"])), (raw, path)


def _signature(groups):
    out = []
    for g in groups:
        keys = {k: g[k] for k in ("kind", "lr", "weight_decay") if k in g}
        keys["betas"] = tuple(g.get("betas", ()))
        out.append((tuple(sorted(keys.items())), frozenset(id(p) for p in g["params"])))
    return sorted(out, key=repr)


@pytest.mark.parametrize("method", METHODS)
def test_role_groups_match_upstream(method):
    model = build_model(debug_config(method), "cpu")
    model.set_training_stage(0)
    assert _signature(parameter_groups(model, model.config.width)) == _signature(upstream_groups(model))


AUDITED = ["dense", "dsa", "huginn", "deltanet", "kda", *sorted(HYBRID), "kda_no_ve", "mla_nope", "mha_no_ve"]
# One method per audit path: single stage, burn-in stages, recurrence depths, affine scan, chunked backend,
# sparse and linear hybrid layouts, MLA, no value embeddings. The rest repeat a covered path.
REPRESENTATIVE = {"dense", "dsa", "huginn", "deltanet", "kda", "moba_hybrid", "gdn_hybrid_swa", "mla_nope", "mha_no_ve"}


@pytest.mark.audit
@pytest.mark.parametrize("method", [m if m in REPRESENTATIVE else pytest.param(m, marks=pytest.mark.audit_full)
                                    for m in AUDITED])
def test_audit_reproduces_upstream_proof(method):
    sub, (raw, path) = upstream_submission(method)
    proof = json.loads((path.parent / raw["calibration"]["cost_record"]).read_text())
    report = audit_submission(sub, ["S", "L"])          # S is a prefix of L: no further stage audits
    assert report["scales"]["L"]["total_training_flops"] == proof["total_training_flops"]
    assert report["parameters"] == proof["parameters"]
    for stage in proof["stages"]:
        label = stage["stage"] if stage.get("depth") is None else f"depth{stage['depth']}"
        actual = report["stage_records"][label]
        for field in ("total", "phases", "operators", "unknown"):
            assert actual[field] == stage["cost"][field]
    if method == "dsa":                                  # the prefix scale is matched by its parent
        s = report["scales"]["S"]
        assert s["prefix_of"] == "L" and s["stage_steps"] == {"burnin": 512}
        assert s["relative_error_vs_anchor"] > 0.01 and s["within_tolerance"]   # dense burn-in costs more


@pytest.mark.parametrize("method", ["dsa", "qsa_hybrid"])
def test_stage_schedule_scales_fixed_stages(method):
    sub, _ = upstream_submission(method)
    assert sub.stage_schedule(46875) == {"burnin": 512, "joint": 46363}
    assert sub.stage_schedule(512, 46875) == {"burnin": 512}           # S: the 512-update prefix is all burn-in


@pytest.mark.parametrize("method", sorted(HYBRID))
def test_hybrid_submission_preserves_layout_and_training_stage(method):
    from nexttransformer.models.sparse import SparseMixer
    sub, (raw, _) = upstream_submission(method)
    assert upstream_model.model_config(raw["model"]).serialized() == raw["model"]
    assert sub.audit.mode == "direct" and not sub.audit.cross_check_lengths
    model = sub.instantiate("cpu", overrides=sub.debug_overrides)
    expected = model.config.layer_mixers
    assert tuple(block.attn.kind if hasattr(block.attn, "kind") else kind
                 for block, kind in zip(model.transformer.h, expected)) == expected
    assert sub.config["aux_loss"] == (method == "qsa_hybrid")
    for step in (0, 511, 512, 46874):
        stage = sub.stage_for_step(step, 46875)
        assert stage == ("burnin" if method == "qsa_hybrid" and step < 512 else "joint")
        sub.set_stage(model, stage, audit=False, step=step, total_steps=46875)
        for block in model.transformer.h:
            if isinstance(block.attn, SparseMixer) and block.attn.kind == "qsa":
                assert block.attn.in_burnin == (stage == "burnin")
    bad = json.loads(json.dumps(raw["model"]))
    bad["layer_mixers"][0] = "attention"
    with pytest.raises(ValueError, match="layer_mixers"):
        upstream_model.submission_config(bad)


def test_static_check_rejects_escape_hatches(tmp_path):
    bad = tmp_path / "model.py"
    bad.write_text("import os\nimport torch\ntorch.nn.functional.softmax = None\n"
                   "class X(torch.Tensor):\n    __torch_function__ = None\n"
                   "def build(c):\n    return open('/etc/passwd')\nprint('side effect')\n")
    problems = "\n".join(static_check(bad))
    for needle in ("import os", "modifying imported module", "__torch_function__", "open()", "module level"):
        assert needle in problems


def test_static_check_allows_upstream_model_blocks(tmp_path):
    ok = tmp_path / "model.py"
    ok.write_text("from nexttransformer.models.backbone import norm\nimport models\n")
    problems = static_check(ok)
    assert problems == ["model.py:2: import models is not allowed"]


def test_dev_shard_is_the_arena_dev_split():
    from nexttransformer.arena import DEV_SHARD
    from nextt_gym.task import TASK, SCALES
    assert TASK.dev_shard == DEV_SHARD
    assert SCALES["S"].origin == "gym" and not SCALES["S"].official_evaluated and SCALES["L"].official_evaluated
    assert SCALES["C"].origin == "gym" and not SCALES["C"].official_evaluated and SCALES["C"].prefix_of is None
    assert SCALES["M"].prefix_of == "L" and SCALES["M"].steps == 2000 and 512 % SCALES["M"].val_every == 0
    assert TASK.describe()["visible_splits"] == {"S": ["dev"], "C": ["dev"], "M": ["dev"], "L": ["dev"]}
    assert TASK.describe()["sealed"] == {"L": ["official", "core"]}


def test_trainer_stdout_keeps_official_results_sealed():
    from nextt_gym.trainer import sealed
    assert sealed(dict(event="validation", split="official", val_bpb=0.7))["val_bpb"] == "sealed"
    assert sealed(dict(event="validation", split="dev", val_bpb=0.7))["val_bpb"] == 0.7
    final = sealed(dict(event="final", val_bpb_dev=0.7, val_bpb_official=0.6, core=0.3))
    assert final == dict(event="final", val_bpb_dev=0.7, val_bpb_official="sealed", core="sealed")


def test_template_verifies_after_calibration(tmp_path):
    for name in ("model.py", "config.json"):
        (tmp_path / name).write_bytes((ROOT / "nextt_gym/templates" / name).read_bytes())
    result = verify(tmp_path)
    assert [lvl["status"] for lvl in result["levels"]] == ["passed", "passed", "passed", "failed"]
    assert result["levels"][-1]["errors"][0]["code"] == "FLOPS_OUT_OF_TOLERANCE"
    tuned, changes, report = calibrate(load_submission(tmp_path), ["mlp_width"])
    assert changes and report["within_tolerance"]


def test_causality_check_catches_leak(tmp_path):
    for name in ("model.py", "config.json"):
        (tmp_path / name).write_bytes((ROOT / "nextt_gym/templates" / name).read_bytes())
    source = (tmp_path / "model.py").read_text()
    # Look one token ahead: a non-causal shortcut must be rejected by V2.
    source = source.replace("x = rms_norm(self.transformer.wte(idx))",
                            "x = rms_norm(self.transformer.wte(idx) + self.transformer.wte(idx.roll(-1, 1)))")
    (tmp_path / "model.py").write_text(source)
    result = verify(tmp_path, ("V0", "V1", "V2"))
    codes = [e["code"] for lvl in result["levels"] for e in lvl["errors"]]
    assert "CAUSALITY_VIOLATION" in codes
