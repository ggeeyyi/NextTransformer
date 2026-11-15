import pytest
import torch

from nexttransformer.models.config import debug_config
from nexttransformer.scaling.reference_blocks import KdaDsaGroup, RoutedFFN


def test_routing_matches_weighted_experts_and_shared_path():
    torch.manual_seed(42)
    config = debug_config("dsa")
    model = RoutedFFN(config, num_experts=3, top_k=2, expert_width=12)
    hidden = torch.randn(2, 3, config.width, requires_grad=True)
    logits, selected = model.router(hidden).topk(2)
    weights = torch.zeros_like(model.router(hidden)).scatter(-1, selected, logits.softmax(-1))
    expected = model.shared(hidden)
    for i, expert in enumerate(model.experts):
        expected = expected + expert(hidden) * weights[..., i, None]
    output = model(hidden)
    torch.testing.assert_close(output, expected)
    output.square().mean().backward()
    for grad in (hidden.grad, model.router.weight.grad, model.shared.c_fc.weight.grad):
        assert torch.isfinite(grad).all() and grad.abs().sum() > 0


@pytest.mark.parametrize("burnin", [True, False])
def test_hybrid_is_causal_and_trains_its_indexer(burnin):
    torch.manual_seed(42)
    config = debug_config("dsa")
    model = KdaDsaGroup(config)
    assert [m.kind for m in model.mixers] == ["kda", "kda", "kda", "dsa"]
    assert all(m.ve_gate is None for m in model.mixers)
    hidden = torch.randn(1, config.sequence_len, config.width)
    frequency = 10000 ** (-torch.arange(0, config.head_dim, 2) / config.head_dim)
    angle = torch.arange(config.sequence_len)[:, None] * frequency
    rotary = (angle.cos()[None, :, None], angle.sin()[None, :, None])
    step = config.burnin_steps - int(burnin)
    output, index_loss = model(hidden, rotary, step)
    assert model.mixers[-1].in_burnin is burnin
    (output.square().mean() + index_loss).backward()
    grad = model.mixers[-1].index_q.weight.grad
    assert torch.isfinite(grad).all() and grad.abs().sum() > 0
    for ffn in model.ffns:
        assert ffn.c_fc.weight.grad.abs().sum() > 0
    changed = hidden.clone()
    changed[:, -2:] += torch.randn_like(changed[:, -2:])
    with torch.no_grad():
        changed_output, _ = model(changed, rotary, step)
    torch.testing.assert_close(output[:, :-2], changed_output[:, :-2])
