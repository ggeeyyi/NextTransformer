"""Independent boundary, causal and state-dict checks for the evaluation adapter."""
import importlib.util
from pathlib import Path

import pytest
import torch

from nexttransformer.methods.inkling_rel import RelativeSelfAttention
from nexttransformer.models.backbone import norm
from nexttransformer.models.config import debug_config
from nexttransformer.models import build_model
from nexttransformer.evaluation.core import inference_view

PATH = Path(__file__).resolve().parents[1] / "evidence/evaluations/xsa-rel-full-context-20261123/rel_full_context.py"
spec = importlib.util.spec_from_file_location("rel_full_context", PATH)
adapter = importlib.util.module_from_spec(spec)
spec.loader.exec_module(adapter)


@torch.no_grad()
def test_boundary_uses_zero_bias_only_for_out_of_bank_distances():
    torch.manual_seed(42)
    config = debug_config("inkling_rel")
    attn = RelativeSelfAttention(config, 0).double().eval()
    extent = attn.rel_extent
    state = {k: v.clone() for k, v in attn.state_dict().items()}
    x = torch.randn(1, extent + 3, config.width, dtype=torch.float64)
    prefix = attn(x[:, :extent], None, None, (-1, 0), None)
    attn.__class__ = adapter.FullContextRelativeAttention
    torch.testing.assert_close(attn(x[:, :extent], None, None, (-1, 0), None), prefix, atol=0, rtol=0)
    actual = attn(x, None, None, (-1, 0), None)
    t, h, d = x.size(1), attn.n_head, attn.head_dim
    q, k = [norm(p(x).view(1, t, h, d)).transpose(1, 2) * 1.2 for p in (attn.c_q, attn.c_k)]
    v = attn.c_v(x).view(1, t, h, d).transpose(1, 2)
    r = attn.c_r(x).view(1, t, h, attn.d_rel)
    bias = torch.zeros(1, h, t, t, dtype=x.dtype)
    for i in range(t):
        for j in range(i + 1):
            if i-j < extent:
                bias[:, :, i, j] = r[:, i] @ attn.rel_logits_proj.proj[:, i-j]
    distance = torch.arange(t)[:, None] - torch.arange(t)[None, :]
    scores = (q @ k.transpose(-1, -2)) * d**-0.5 + bias
    weights = scores.masked_fill(distance < 0, -torch.inf).softmax(-1)
    expected = attn.c_proj((weights @ v).transpose(1, 2).reshape(1, t, config.width))
    torch.testing.assert_close(actual, expected, atol=1e-12, rtol=1e-12)
    assert bias[:, :, extent, 0].eq(0).all()
    assert bias[:, :, -1, -2].abs().sum() > 0
    changed = x.clone(); changed[:, extent:] += 10
    torch.testing.assert_close(attn(changed, None, None, (-1, 0), None)[:, :extent], actual[:, :extent], atol=0, rtol=0)
    for name, value in attn.state_dict().items():
        torch.testing.assert_close(value, state[name], atol=0, rtol=0)


def test_extended_path_rejects_training():
    config = debug_config("inkling_rel")
    attn = adapter.FullContextRelativeAttention(config, 0)
    with pytest.raises(ValueError, match="evaluation-only"):
        attn(torch.randn(1, attn.rel_extent + 1, config.width), None, None, (-1, 0), None)


@torch.no_grad()
def test_full_model_view_preserves_short_logits_and_configuration():
    config = debug_config("inkling_rel", width=32, sequence_len=32)
    model = build_model(config).eval()
    short = torch.randint(config.vocab_size, (1, config.sequence_len))
    expected = model(short)
    state = {k: v.clone() for k, v in model.state_dict().items()}
    adapter.enable_full_context(model)
    view = inference_view(model, None)
    torch.testing.assert_close(view(short), expected, atol=0, rtol=0)
    long = torch.randint(config.vocab_size, (1, config.option("rel_extent") + 1))
    logits = view(long)
    assert logits.shape == (*long.shape, config.vocab_size) and torch.isfinite(logits).all()
    assert model.config is config and model.config.sequence_len == 32
    for name, value in model.state_dict().items():
        torch.testing.assert_close(value, state[name], atol=0, rtol=0)
