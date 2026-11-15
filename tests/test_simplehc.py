"""SimpleHC plugin: shared-gate group execution, initial gates, causality and parameter ownership."""
from dataclasses import replace

import pytest
import torch

from nexttransformer.methods.simplehc import group_sizes
from nexttransformer.models.config import debug_config
from nexttransformer.models.lm import build_model
from nexttransformer.training.optimizer import parameter_groups


def build(group=4, layers=3, seed=1, perturb=True):
    c = debug_config("simplehc")
    c = replace(c, layers=layers, options=dict(c.options, group=group))
    model = build_model(c, "cpu", seed=seed)
    if perturb:
        # Output projections start at zero, which would hide the scalar corrections; move every weight off its init.
        generator = torch.Generator().manual_seed(seed)
        with torch.no_grad():
            for p in model.parameters():
                p.add_(0.2 * torch.randn(p.shape, generator=generator, dtype=p.dtype))
    return model


def sequential_hidden(model, idx):
    """Sublayer-by-sublayer execution with the same group-entry gates, materializing every intermediate state."""
    c = model.config
    x = model._embed_reference(idx)
    streams = x[:, :, None].expand(-1, -1, c.option("streams"), -1)
    k = 0
    for gates in model.residual_maps:
        pre, post = gates(streams)
        state = streams
        for a in range(gates.size):
            h = torch.einsum("btn,btnd->btd", pre[:, :, a].to(state.dtype), state)
            out = model._sublayer(h, idx, k)
            state = state + post[:, :, a, :, None].to(out.dtype) * out[:, :, None]
            k += 1
        streams = state
    return streams.sum(2)


def test_group_sizes_keep_the_remainder_last():
    assert group_sizes(46, 4) == (4,) * 11 + (2,)
    assert group_sizes(6, 4) == (4, 2) and group_sizes(8, 4) == (4, 4) and group_sizes(3, 8) == (3,)


@pytest.mark.parametrize("group", [1, 2, 3, 4, 8])
def test_group_execution_equals_sequential_execution(group):
    model = build(group=group).eval()
    idx = torch.randint(model.config.vocab_size, (2, model.config.sequence_len), generator=torch.Generator().manual_seed(3))
    with torch.no_grad():
        grouped, backout = model._ordinary_hidden(idx)
        reference = sequential_hidden(model, idx)
    assert backout is None
    assert len(model.residual_maps) == len(group_sizes(2 * model.config.n_layer, group))
    # FP32 sums in a different order; dropping the corrections instead moves the result by about its own scale.
    torch.testing.assert_close(grouped, reference, rtol=1e-5, atol=2e-6 * float(reference.abs().max()))


def test_initial_gates_read_half_and_write_one():
    model = build(perturb=False)
    for gates in model.residual_maps:
        assert torch.all(gates.bias == 0) and torch.all(gates.alpha == 0.01)
        assert abs(float(gates.projection.detach().std()) - 0.006) < 0.002
    streams = torch.randn(2, 5, 4, model.config.width)
    pre, post = model.residual_maps[0](streams)
    torch.testing.assert_close(pre, torch.full_like(pre, .5), rtol=0, atol=1e-3)
    torch.testing.assert_close(post, torch.ones_like(post), rtol=0, atol=2e-3)


def test_causal_and_every_parameter_trains():
    model = build().train()
    model.set_training_stage(0)
    c = model.config
    idx = torch.randint(c.vocab_size, (2, c.sequence_len), generator=torch.Generator().manual_seed(5))
    with torch.no_grad():
        before = model(idx)
        changed = idx.clone(); changed[:, -1] = (changed[:, -1] + 1) % c.vocab_size
        after = model(changed)
    torch.testing.assert_close(before[:, :-1], after[:, :-1])
    loss = model(idx, idx)
    loss.backward()
    assert torch.isfinite(loss)
    missing = [name for name, p in model.named_parameters() if p.requires_grad and (p.grad is None or not p.grad.any())]
    assert not missing
    frozen = {name for name, p in model.named_parameters() if not p.requires_grad}
    assert frozen == {"resid_lambdas", "x0_lambdas", "backout_lambda"}


def test_gate_parameters_belong_to_adamw_without_decay():
    model = build(perturb=False)
    model.set_training_stage(0)
    owners = {name: group for group in parameter_groups(model) for name in group["parameter_names"]}
    gates = [name for name in owners if name.startswith("residual_maps.")]
    assert gates and all(owners[name]["kind"] == "adamw" and owners[name]["weight_decay"] == 0 for name in gates)


@pytest.mark.parametrize("option,value", [("group", 0), ("streams", 0), ("group", 2.5), ("init_std", 0), ("alpha_init", -1.)])
def test_invalid_options_are_rejected(option, value):
    c = debug_config("simplehc")
    with pytest.raises(ValueError, match=option):
        replace(c, options=dict(c.options, **{option: value}))
