"""Training-kernel parity, including nonzero taps and all first-order gradients."""
from dataclasses import replace
from unittest.mock import patch

import pytest
import torch

from nexttransformer.methods.msar import ops
from nexttransformer.models.config import debug_config
from nexttransformer.models.lm import build_model

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA kernel parity")


def close_bf16(actual, expected, relative=.008, norm_atol=0.):
    a, b = actual.float(), expected.float()
    assert torch.isfinite(a).all() and torch.isfinite(b).all()
    error = (a-b).norm() / b.norm().clamp_min(1e-8)
    assert (a-b).norm() <= relative*b.norm()+norm_atol, float(error)
    assert (a-b).abs().max() <= relative*4*b.abs().max().clamp_min(1e-8)+norm_atol+1e-7


@pytest.mark.parametrize("length,width", [(1, 48), (7, 48), (33, 96), (2048, 1536)])
def test_temporal_training_kernel(length, width):
    torch.manual_seed(47)
    module = ops.TemporalAugmentation(width).cuda()
    with torch.no_grad():
        for w in module.taps:
            w.normal_(std=.02)
    x = torch.randn(2, length, width, device="cuda", dtype=torch.bfloat16, requires_grad=True)
    actual = module(x)
    with patch.object(ops, "_cuda_kernel_supported", return_value=False):
        expected = module(x)
    close_bf16(actual, expected)
    grad = torch.randn_like(actual)
    ga = torch.autograd.grad(actual, (x, *module.taps), grad)
    gb = torch.autograd.grad(expected, (x, *module.taps), grad)
    for a, b in zip(ga, gb):
        close_bf16(a, b)


@pytest.mark.parametrize("queries,depth,width", [(1, 1, 48), (5, 3, 96), (6, 8, 1536)])
def test_completed_value_training_kernel(queries, depth, width):
    from nexttransformer.methods.msar.kernels import completed_values
    torch.manual_seed(51)
    sources = [torch.randn(2, 7, 4, width, device="cuda", dtype=torch.bfloat16,
                           requires_grad=True) for _ in range(depth)]
    weights = torch.randn(queries, depth, 2, 7, 4, device="cuda").softmax(1).bfloat16().requires_grad_()
    actual = completed_values(weights, sources)
    expected = weights[:, 0, ..., None]*sources[0]
    for c in range(1, depth):
        expected = expected+weights[:, c, ..., None]*sources[c]
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    grad = torch.randn_like(actual)
    ga = torch.autograd.grad(actual, (weights, *sources), grad)
    gb = torch.autograd.grad(expected, (weights, *sources), grad)
    for a, b in zip(ga, gb):
        close_bf16(a, b)


def test_model_hidden_all_gradients():
    c = debug_config("msar", sequence_len=13)
    c = replace(c, layers=4)
    model = build_model(c, "cuda")
    with torch.no_grad():
        for block in model.transformer.h:
            block.attn.c_proj.weight.normal_(std=.03)
            block.mlp.c_proj.weight.normal_(std=.03)
        for name, p in model.named_parameters():
            if name.startswith(("residual_maps.", "msar_initial.", "msar_writes.")):
                p.add_(torch.randn_like(p)*.1)
    tokens = torch.randint(c.vocab_size, (2, 13), device="cuda")
    actual = model._ordinary_hidden(tokens)[0]
    with patch.object(ops, "_cuda_kernel_supported", return_value=False):
        expected = model._ordinary_hidden(tokens)[0]
    close_bf16(actual, expected, .025)
    params = [p for p in model.parameters() if p.requires_grad and p is not model.lm_head.weight]
    grad = torch.randn_like(actual)
    ga = torch.autograd.grad(actual, params, grad)
    gb = torch.autograd.grad(expected, params, grad)
    for a, b in zip(ga, gb):
        # Small scalar VJPs can be cancellation-dominated. Bound their absolute
        # norm too, and independently bound the relative error of the whole VJP.
        close_bf16(a, b, .05, norm_atol=.02)
    error2 = sum((a.float()-b.float()).square().sum() for a, b in zip(ga, gb))
    reference2 = sum(b.float().square().sum() for b in gb)
    assert torch.sqrt(error2/reference2) < .015
