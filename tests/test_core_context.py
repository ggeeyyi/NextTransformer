"""CORE owns its input limit independently of training and BPB."""
import argparse
from types import SimpleNamespace

import pytest
import torch

from nanochat.core_eval import evaluate_example
from nexttransformer.evaluation import core, formal, native
from nexttransformer.evaluation.context import (
    add_core_max_tokens, check_full_context, core_context, resume_core_limit, validate_submission,
)
from nexttransformer.models import build_model
from nexttransformer.models.config import debug_config
from nexttransformer.models.native.config import NativeModelConfig


class CharacterTokenizer:
    def get_bos_token_id(self):
        return 0

    def __call__(self, prompts, *, prepend):
        return [[prepend, *(1 + ord(char) % 50 for char in prompt)] for prompt in prompts]


@pytest.mark.parametrize("method", ["dense", "attnres", "native"])
def test_full_and_capped_inputs_through_upstream_scorer(method, monkeypatch, tmp_path):
    if method == "native":
        config = NativeModelConfig(width=32, heads=2, mlp_width=128, vocab_size=64,
                                   sequence_len=16, dense_layers=6, pre_layers=1, core_layers=2,
                                   post_layers=1, fourier_dim=16)
        evaluator, smoke = native, {"max_per_task": 1}
    else:
        config = debug_config(method, width=32, sequence_len=16)
        evaluator, smoke = formal, {"smoke_max_per_task": 1}
    model = build_model(config).eval()
    monkeypatch.setenv("NANOCHAT_BASE_DIR", str(tmp_path))
    monkeypatch.setattr(evaluator, "verify_core", lambda _: None)
    inputs = []
    task = dict(task_type="multiple_choice", continuation_delimiter=" ", num_fewshot=0)
    data = [dict(query="Hi", choices=["A", "B"], gold=0),
            dict(query="Z"*24, choices=["A", "B"], gold=0)]

    def score(wrapped, tokenizer, device, *, max_per_task):
        wrapped.register_forward_pre_hook(lambda module, args: inputs.append(args[0].clone()))
        return {"results": [evaluate_example(i, wrapped, tokenizer, data, device, task) for i in range(2)]}

    monkeypatch.setattr(core, "upstream_core", score)
    tokenizer = CharacterTokenizer()
    device = torch.device("cpu")
    full = evaluator.evaluate_core(model, tokenizer, tmp_path, device, **smoke)
    capped = evaluator.evaluate_core(model, tokenizer, tmp_path, device, core_max_tokens=16, **smoke)
    extended = evaluator.evaluate_core(model, tokenizer, tmp_path, device, core_max_tokens=20, **smoke)
    short, long = inputs[:2]
    assert long.shape[1] == 27 and torch.all(long[:, 0] == tokenizer.get_bos_token_id())
    for other in (inputs[2], inputs[4]):
        assert torch.equal(short, other)
    assert torch.equal(inputs[3], long[:, -16:]) and torch.equal(inputs[5], long[:, -20:])
    assert full["coreContext"] == core_context(16, None, 27)
    assert capped["coreContext"] == core_context(16, 16, 16)
    assert extended["coreContext"] == core_context(16, 20, 20)
    view = core.inference_view(model, None)
    with torch.no_grad():
        torch.testing.assert_close(view(short), model(short), rtol=0, atol=0)
    if hasattr(model, "_window"):
        assert view._window(0, 2) == model._window(0, 2)
    assert model.config is config and model.config.sequence_len == 16
    with pytest.raises(ValueError, match="context"):
        model(long)


def test_cli_and_resume_length_policy():
    parser = argparse.ArgumentParser()
    add_core_max_tokens(parser)
    assert parser.parse_args([]).core_max_tokens is None
    assert parser.parse_args(["--core-max-tokens", "4096"]).core_max_tokens == 4096
    for value in ("0", "-1", "1.5"):
        with pytest.raises(SystemExit):
            parser.parse_args(["--core-max-tokens", value])
    assert resume_core_limit({}, None, 2048) == 2048
    assert resume_core_limit({"core_max_tokens": None}, None, 2048) is None
    assert resume_core_limit({"core_max_tokens": 4096}, None, 2048) == 4096
    with pytest.raises(ValueError, match="checkpoint"):
        resume_core_limit({"core_max_tokens": None}, 2048, 2048)


def test_full_bundle_answer_and_learned_position_limits():
    fixed = SimpleNamespace(options={"rel_extent": 2048})
    check_full_context(fixed, 2048)
    with pytest.raises(ValueError, match="learned REL table"):
        check_full_context(fixed, None)
    check_full_context(SimpleNamespace(), 225)
    with pytest.raises(ValueError, match="preceding token"):
        check_full_context(SimpleNamespace(), 224)


@pytest.mark.parametrize("cap,actual", [(None, 5835), (225, 225), (2048, 2048), (4096, 4096), (8192, 5835)])
def test_published_lengths(cap, actual):
    context = core_context(2048, cap, actual)
    validate_submission(context)
    for bad in ({}, dict(context, maxTokens=False), dict(context, truncation="unknown"),
                dict(context, maxEvaluatedTokens=actual-1), dict(context, trainingTokens=1024)):
        with pytest.raises(ValueError, match="CORE"):
            validate_submission(bad)
