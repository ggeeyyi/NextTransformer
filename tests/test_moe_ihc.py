"""The Hy4 identity, dynamic scalar gates, learned final read and MoE integration."""
from dataclasses import replace

import pytest
import torch

from moe import plugins
from moe.model import build_model


def test_mappings_match_the_per_stream_equations_and_gradients():
    torch.manual_seed(19)
    config = plugins.config_for("ihc", "d768-240tpp", debug=True)
    plugin = plugins.module("ihc")
    mapping = plugin.StreamMapping(config).double()
    mapping.reset_parameters()
    with torch.no_grad():
        mapping.scales.fill_(0.7)
    n = config.option("streams")
    streams = torch.randn(2, 3, n, config.width, dtype=torch.float64, requires_grad=True)
    update = torch.randn_like(streams[..., 0, :], requires_grad=True)
    x, post = mapping(streams)
    result = plugin.write(streams, update, post)

    flat = streams.flatten(-2)
    normed = flat * torch.rsqrt(flat.square().mean(-1, keepdim=True) + config.norm_eps)
    read_logits, write_logits = (normed @ mapping.projection).chunk(2, -1)
    pre = (read_logits * mapping.scales[0] + mapping.bias[:n]).sigmoid() + mapping.gate_eps
    post_ref = 2 * (write_logits * mapping.scales[1] + mapping.bias[n:]).sigmoid() + mapping.gate_eps
    expected_x = sum(pre[..., i, None] * streams[..., i, :] for i in range(n))
    expected_result = torch.stack([streams[..., i, :] + post_ref[..., i, None] * update for i in range(n)], -2)
    torch.testing.assert_close(x, expected_x)
    torch.testing.assert_close(result, expected_result)
    torch.testing.assert_close(plugin.write(streams, torch.zeros_like(update), post), streams, rtol=0, atol=0)
    variables = (streams, update, *mapping.parameters())
    actual = torch.autograd.grad(x.square().mean() + result.square().mean(), variables)
    expected = torch.autograd.grad(expected_x.square().mean() + expected_result.square().mean(), variables)
    for grad, ref in zip(actual, expected):
        torch.testing.assert_close(grad, ref)


def test_attention_ffn_and_final_read_use_their_own_current_states():
    torch.manual_seed(42)
    config = plugins.config_for("ihc", "d768-240tpp", debug=True)
    model = build_model(config, "cpu").eval()
    block = model.blocks[-1]
    states = {}
    def record(name):
        def hook(module, args):
            states[name] = args[0].detach().clone()
        return hook
    block.attn_mapping.register_forward_pre_hook(record("attn"))
    block.ffn_mapping.register_forward_pre_hook(record("ffn"))
    block.head_mapping.register_forward_pre_hook(record("head"))
    tokens = torch.randint(0, config.vocab_size, (config.qb_group_rows, 12))
    logits = model(tokens)
    assert logits.shape == (*tokens.shape, config.vocab_size)
    assert states["attn"].shape == (*tokens.shape, config.option("streams"), config.width)
    assert not torch.equal(states["attn"], states["ffn"])
    assert not torch.equal(states["ffn"], states["head"])
    assert sum(layer.head_mapping is not None for layer in model.blocks) == 1
    x = states["head"]
    final = block.head_mapping
    with torch.no_grad():
        final.scales.zero_()
        final.bias.copy_(torch.linspace(-2, 2, final.streams))
    got, post = final(x)
    assert post is None
    torch.testing.assert_close(got, ((final.bias.sigmoid() + final.gate_eps)[:, None] * x).sum(-2))
    model(tokens, tokens.roll(-1, dims=1)).backward()
    for layer in model.blocks:
        for mapping in (layer.attn_mapping, layer.ffn_mapping):
            assert torch.count_nonzero(mapping.projection.grad)
    assert torch.count_nonzero(final.bias.grad)


def test_init_and_config_validation():
    config = plugins.config_for("ihc", "d768-240tpp")
    plugin = plugins.module("ihc")
    mapping = plugin.StreamMapping(config)
    mapping.reset_parameters()
    n = config.option("streams")
    assert mapping.projection.shape == (n * config.width, 2 * n)
    assert torch.count_nonzero(mapping.projection)
    torch.testing.assert_close(mapping.bias[:n].sigmoid(), torch.full((n,), 1 / n))
    torch.testing.assert_close(2 * mapping.bias[n:].sigmoid(), torch.ones(n))
    for name, value in (("streams", 1), ("init_std", 0), ("gate_eps", float("nan"))):
        options = dict(config.options, **{name: value})
        with pytest.raises(ValueError, match=name):
            plugin.check(replace(config, options=tuple(options.items())))
