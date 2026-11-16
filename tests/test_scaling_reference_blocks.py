import pytest
import torch

from nexttransformer.models import build_model
from nexttransformer.models.config import debug_config
from nexttransformer.models.lm import FlexibleMLP
from nexttransformer.scaling.reference_blocks import KdaDsaGroup, RoutedFFN, shared_expert_rule


def rule(config, **changes):
    return shared_expert_rule(config, **dict(num_experts=4, top_k=2, expert_width=12, **changes))


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


def test_only_the_bottom_layer_of_the_stack_keeps_a_dense_feed_forward():
    """The three expert families share one rule, so it is keyed on position in the whole stack."""
    config = debug_config("dsa")
    build = rule(config)
    assert isinstance(build(0), FlexibleMLP)
    assert all(isinstance(build(layer), RoutedFFN) for layer in (1, 2, 7))


@pytest.mark.parametrize("first_layer, dense_first", [(0, True), (4, False)])
def test_hybrid_group_routes_every_layer_above_the_bottom(first_layer, dense_first):
    torch.manual_seed(42)
    config = debug_config("dsa")
    model = KdaDsaGroup(config, rule(config), first_layer=first_layer)
    assert isinstance(model.ffns[0], FlexibleMLP if dense_first else RoutedFFN)
    assert all(isinstance(ffn, RoutedFFN) for ffn in model.ffns[1:])
    hidden = torch.randn(1, config.sequence_len, config.width)
    frequency = 10000 ** (-torch.arange(0, config.head_dim, 2) / config.head_dim)
    angle = torch.arange(config.sequence_len)[:, None] * frequency
    output, index_loss = model(hidden, (angle.cos()[None, :, None], angle.sin()[None, :, None]), 0)
    (output.square().mean() + index_loss).backward()
    routed = [ffn for ffn in model.ffns if isinstance(ffn, RoutedFFN)]
    assert all(torch.isfinite(ffn.router.weight.grad).all() for ffn in routed)
    assert all(ffn.shared.c_fc.weight.grad.abs().sum() > 0 for ffn in routed)


def test_looped_core_runs_one_set_of_experts_on_every_pass():
    """Looped stores its core once, so routed experts placed there are reused across depth."""
    torch.manual_seed(42)
    config = debug_config("fixed_recursive")
    model = build_model(config)
    build = rule(config)
    for layer, block in enumerate(model.transformer.h):
        block.mlp = build(layer)
    assert len(model.transformer.h) == config.n_layer < config.maximum_depth
    assert config.maximum_depth - config.n_layer == (config.loops - 1) * config.core_layers
    core = model.transformer.h[config.pre_layers]
    assert isinstance(core.mlp, RoutedFFN)
    assert len({id(block.mlp) for block in model.transformer.h}) == config.n_layer
    tokens = torch.randint(config.vocab_size, (2, config.sequence_len + 1))
    model(tokens[:, :-1], targets=tokens[:, 1:]).backward()
    for weight in (core.mlp.router.weight, core.mlp.shared.c_fc.weight):
        assert torch.isfinite(weight.grad).all() and weight.grad.abs().sum() > 0
