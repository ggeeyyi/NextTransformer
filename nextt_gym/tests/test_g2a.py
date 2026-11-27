"""G2a: the gym trainer and the formal trainer train a registered method identically.

Both trainers read the training rows with the same code (runtime.data.BestFitStream, protocol.admission.data_layout
and share_batches), compute the loss with the model's own forward, and clip with the same call; each builds the rest on
its own. Here, for representative methods (one per training path, and the two Stage 3 plugins), each side builds its
model, optimizer and schedule as its trainer does, the gym from the method's registered configuration as an imported
attempt (nextt_gym.trainer: Submission.instantiate, roles.build_optimizer, Schedule over the formal horizon for the
prefix scales S and M) and the formal side as nexttransformer.train does (build_model, training.optimizer,
NativeTrainConfig). From the same rows, three updates of two micro batches each must give bitwise equal weights.
The gym scales the embedding and unembedding learning rates by the submission's `model_dim`, the formal trainer by the
model's width: equal at the registered size (checked); the gym trainer follows a smoke override of the width
(model_dim_field), so the debug-size models here pass their own width.
The scale C has its own schedule by design and is not a G2 path. On GPU the same comparison is G2b.
"""
from dataclasses import replace
import json
from pathlib import Path

import pytest
import torch

from nextt_gym import upstream_model
from nextt_gym.contract import Submission
from nextt_gym.importer import config_paths
from nextt_gym.roles import build_optimizer as gym_optimizer
from nextt_gym.task import SCALES
from nextt_gym.trainer import Schedule
from nexttransformer.models.lm import build_model
from nexttransformer.models.native.config import NativeTrainConfig
from nexttransformer.native.optimizer import set_schedule
from nexttransformer.protocol.spec import Protocol
from nexttransformer.training.optimizer import build_optimizer as formal_optimizer
import types

ROOT = Path(__file__).resolve().parents[2]
HOOKS = types.SimpleNamespace(build=upstream_model.build, stage_for_step=upstream_model.stage_for_step,
                              set_stage=upstream_model.set_stage)
# One method per training path: Dense baseline, residual trunk, AttnRes, a looped recurrence, the plugins of Stage 3.
METHODS = ["dense_full", "residual_standard", "attnres", "fixed_recursive", "attnres_canon_abcd",
           "attnres_full_canon_abcd"]


def gym_side(method):
    path = next(p for p in config_paths(ROOT) if p.stem == method)
    sub = Submission(None, HOOKS, upstream_model.submission_config(json.loads(path.read_text())["model"]))
    model = sub.instantiate("cpu", overrides=sub.debug_overrides, seed=Protocol().seed)
    sub.set_stage(model, sub.stage_for_step(0, Protocol().steps), audit=False, step=0, total_steps=Protocol().steps)
    assert sub.model_dim == json.loads(path.read_text())["model"]["width"]      # the trained size: same LR scaling
    return sub, model, gym_optimizer(model, model.config.width), Schedule(Protocol().steps)


def formal_side(config):
    model = build_model(config, "cpu", seed=Protocol().seed)
    return model, formal_optimizer(model, NativeTrainConfig()), NativeTrainConfig()


def update(model, optimizer, schedule, step, rows, set_step, set_row_offset):
    """One update as both trainers make it: schedule, two micro batches, clip, optimizer step."""
    set_step(model, step)
    optimizer.zero_grad(set_to_none=True)
    set_schedule(optimizer, step, schedule)
    for micro, (x, y) in enumerate(rows):
        set_row_offset(model, step * 4 + micro * 2)
        (model(x, y) / len(rows)).backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), float("inf"), error_if_nonfinite=True)
    optimizer.step()


@pytest.mark.parametrize("method", METHODS)
def test_gym_and_formal_trainers_take_the_same_updates(method):
    sub, gym, gym_opt, gym_schedule = gym_side(method)
    formal, formal_opt, formal_schedule = formal_side(gym.config)
    state = lambda m: {n: t.clone() for n, t in m.state_dict().items()}
    start = state(gym)
    assert all(torch.equal(start[n], t) for n, t in formal.state_dict().items()), "initial weights differ"
    torch.manual_seed(0)
    c = gym.config
    rows = [[(torch.randint(0, c.vocab_size, (2, c.sequence_len)), torch.randint(0, c.vocab_size, (2, c.sequence_len)))
             for _ in range(2)] for _ in range(3)]
    for step in range(3):
        update(gym, gym_opt, gym_schedule, step, rows[step], sub.set_step, sub.set_row_offset)
        update(formal, formal_opt, formal_schedule, step, rows[step],
               lambda m, s: m.set_step(s), lambda m, r: m.set_row_offset(r))
        after = state(gym)
        different = [n for n, t in formal.state_dict().items() if not torch.equal(after[n], t)]
        assert not different, (step, different[:5])
    assert any(not torch.equal(start[n], t) for n, t in after.items())


# The horizon each scale must train under, stated here rather than read from nextt_gym.task: S and M are prefixes
# of the formal 46,875-update schedule, C compresses its own 512 updates, L is the formal schedule.
HORIZONS = dict(S=46_875, M=46_875, L=46_875, C=512)


@pytest.mark.parametrize("scale", sorted(HORIZONS))
def test_prefix_scales_follow_the_formal_schedule(scale):
    """S and M are prefixes of L: through the gym's own run spec (make_run_spec) and trainer (schedule_for), their
    learning rate, momentum and weight decay are the formal run's at every update, warmup and decay boundaries
    included. C compresses the schedule to its 512 updates by design and must differ."""
    from nextt_gym.runspec import make_run_spec
    from nextt_gym.trainer import schedule_for
    sub, _, _, _ = gym_side("dense_full")
    report = dict(stages=[dict(name="joint", total=1)], forward_flops_per_token=1.)
    gym, formal = schedule_for(make_run_spec(sub, report, scale, "s", 8, 10 ** 30)), NativeTrainConfig()
    groups = lambda: types.SimpleNamespace(param_groups=[dict(kind="muon", initial_lr=1., initial_weight_decay=1.),
                                                         dict(kind="adamw", initial_lr=1.)])
    steps = SCALES[scale].steps
    warmdown_start = formal.steps - round(formal.warmdown_ratio * formal.steps)
    checked = sorted({*range(0, steps, 7), 0, 38, 39, 40, 41, 399, 400, steps - 1, warmdown_start, warmdown_start + 1}
                     & set(range(steps)))
    same = []
    for step in checked:
        a, b = groups(), groups()
        same.append(set_schedule(a, step, gym) == set_schedule(b, step, formal) and a.param_groups == b.param_groups)
    assert gym.steps == HORIZONS[scale] and set(SCALES) == set(HORIZONS)
    assert all(same) if HORIZONS[scale] == formal.steps else not all(same)


# ---- both trainers end to end on the same rows -----------------------------------------------------------------
VOCAB, LENGTH = 64, 12


def rows(rank, start, size):
    """Deterministic rows keyed by (rank, row): the one source both trainers read here."""
    out = [torch.randint(VOCAB, (LENGTH + 1,), generator=torch.Generator().manual_seed(1_000_003 * rank + row))
           for row in range(start, start + size)]
    tokens = torch.stack(out)
    return tokens[:, :-1].contiguous(), tokens[:, 1:].contiguous()


class Rows:
    """BestFitStream's interface for the formal trainer, SyntheticStream's for the gym trainer."""
    def __init__(self, *args, **kwargs):
        self.rank = kwargs.get("rank", args[3] if len(args) > 4 else args[3] if len(args) == 4 else 0)
        self.rows = 0
    def batch(self, size, device):
        x, y = rows(self.rank, self.rows, size); self.rows += size
        return x.to(device), y.to(device)
    def batches(self, size, device):
        while True: yield self.batch(size, device)
    def state_dict(self): return dict(rows=self.rows)
    def load_state_dict(self, state): self.rows = state["rows"]


@pytest.mark.parametrize("method", ["dense_full", "attnres_full_canon_abcd"])
def test_both_trainers_end_to_end(method, tmp_path, monkeypatch):
    """`nexttransformer.train.main` (smoke, one micro batch of two rows per update) and `nextt_gym.trainer.main`
    (synthetic, two rows per update) on the same rows: equal losses at every update and equal final weights."""
    from nextt_gym import trainer as gym_trainer
    from nextt_gym.contract import load_submission
    from nextt_gym.runspec import make_run_spec, write_run_spec
    from nexttransformer import train
    torch.set_num_threads(1)
    for name in ("WORLD_SIZE", "RANK", "LOCAL_RANK"): monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv("NANOCHAT_RUNTIME", str(tmp_path / "runtime"))
    path = next(p for p in config_paths(ROOT) if p.stem == method)
    submission = tmp_path / "submission"; submission.mkdir()
    (submission / "model.py").write_text(upstream_model.MODEL_PY)
    (submission / "config.json").write_text(json.dumps(upstream_model.submission_config(
        json.loads(path.read_text())["model"])))
    sub = load_submission(submission)
    overrides = dict(sub.debug_overrides, vocab_size=VOCAB, sequence_len=LENGTH)
    config = upstream_model.model_config({**sub.config["model"], **overrides})

    # The formal trainer, as tests/test_train.py runs it: fakes for resources, tokenizer, data and evaluators.
    proof = dict(formula_version="test", flops_per_token=1e9, total_training_flops=1,
                 stages=[dict(stage="joint", depth=None, cost=dict(total=5e15, phases=dict(forward=2e15)))])
    monkeypatch.setattr(train, "review_config", lambda p: (config, proof, []))
    monkeypatch.setattr(train, "verify_resources", lambda r, d: (dict(tokenizer_sha256="t"), "manifest"))
    monkeypatch.setattr(train.RustBPETokenizer, "from_directory", staticmethod(lambda p: type("T", (), dict(
        get_vocab_size=lambda self: VOCAB))()))
    monkeypatch.setattr(train, "BestFitStream", Rows)
    monkeypatch.setattr(train, "evaluate_bpb", lambda *a, **k: dict(bpb=1.0, input_tokens=k["tokens"], split=k["split"],
                                                                    seconds=0.0, smoke=k["smoke"]))
    monkeypatch.setattr(train, "evaluate_core", lambda *a, **k: dict(core_metric=0.0, results={}, full_core=False))
    (tmp_path / "config.json").write_text(json.dumps(dict(calibration=dict(dense_flops_per_token=4.6e9))))
    formal_out = tmp_path / "formal"
    train.main(["--config", str(tmp_path / "config.json"), "--output", str(formal_out), "--runtime-dir",
                str(tmp_path / "runtime"), "--resources", str(tmp_path / "resources"), "--data-dir", str(tmp_path / "data"),
                "--micro-batch", "2", "--smoke-steps", "6", "--smoke-accumulation", "1", "--log-every", "1",
                "--wandb-mode", "disabled"])

    # The gym trainer: an S run (a prefix of the formal schedule) of the imported method, synthetic rows.
    monkeypatch.setattr(gym_trainer, "SyntheticStream", Rows)
    report = dict(stages=[dict(name="joint", total=5 * 10 ** 15)], forward_flops_per_token=0.5)
    spec = make_run_spec(sub, report, "S", "g2a", 1, 10 ** 18, synthetic=True)
    spec["steps"] = 6
    write_run_spec(spec, tmp_path / "spec.json")
    gym_out = tmp_path / "gym"
    gym_trainer.main(["--submission", str(submission), "--run-spec", str(tmp_path / "spec.json"), "--output", str(gym_out),
                      "--synthetic", "--model-overrides", json.dumps(overrides), "--smoke-rows", "2", "--log-every", "1"])

    def losses(output, key):
        events = [json.loads(line) for line in (output / "metrics.jsonl").read_text().splitlines()]
        return {e["step"]: e[key] for e in events if e["event"] == "train"}
    assert losses(formal_out, "lm_loss") == losses(gym_out, "loss") and len(losses(gym_out, "loss")) == 6
    load = lambda output: torch.load(output / "checkpoint_00000006" / "model.pt", map_location="cpu", weights_only=False)["model"]
    formal, gym = load(formal_out), load(gym_out)
    assert formal.keys() == gym.keys() and all(torch.equal(formal[n], gym[n]) for n in formal)
