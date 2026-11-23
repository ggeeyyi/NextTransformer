"""The `canon_depth_lab` plugin: AttnRes-Canon with a headwise sigmoid gate on every attention output.

The shared per-method tests cover forward and backward, causality, gradients and the checkpoint round trip. Here:
without the gate the plugin is exactly `attnres_full_canon_abcd`; the gate scales each head's output before the
output projection; and the registered size is the frozen entry's.
"""
from dataclasses import replace

import torch

from nexttransformer.models.config import candidate_config, debug_config
from nexttransformer.models.lm import build_model
from nexttransformer.training.optimizer import parameter_groups


def ungated(config):
    return replace(config, options=dict(dict(config.options), head_gate=False))


def test_without_the_gate_is_attnres_canon():
    torch.manual_seed(0)
    reference = build_model(replace(debug_config("attnres_full_canon_abcd"), layers=3), "cpu", seed=3)
    for name, parameter in reference.named_parameters():
        if "canon" in name: torch.nn.init.normal_(parameter, std=0.1)  # exercise every Canon path
    config = ungated(replace(debug_config("canon_depth_lab"), layers=3))
    plugin = build_model(config, "cpu", seed=11)
    plugin.load_state_dict(reference.state_dict())  # the same parameter names and shapes
    x = torch.randint(0, config.vocab_size, (2, config.sequence_len))
    for model in (reference, plugin): model.eval()
    assert torch.equal(reference(x), plugin(x))


def test_the_gate_scales_each_head_before_the_output_projection():
    torch.manual_seed(0)
    config = debug_config("canon_depth_lab")
    gated, plain = build_model(config, "cpu", seed=1), build_model(ungated(config), "cpu", seed=2)
    plain.load_state_dict(gated.state_dict(), strict=False)
    x = torch.randn(2, config.sequence_len, config.width)
    t = config.sequence_len

    def attend(model, projection):
        attn = model.transformer.h[0].attn
        attn.c_proj.weight.data.copy_(projection)
        return attn(x, None, (model.cos[:, :t], model.sin[:, :t]), model._window(0, config.n_layer), None)

    gate = gated.transformer.h[0].attn.output_gate
    assert gate.weight.shape == (config.heads, config.width) and torch.count_nonzero(gate.weight) > 0
    head_dim = config.width // config.heads
    # With the output projection as the identity the module returns the gated heads themselves.
    heads = attend(plain, torch.eye(config.width)).view(2, t, config.heads, head_dim)
    expected = heads * torch.sigmoid(gate(x)).unsqueeze(-1)
    assert torch.allclose(attend(gated, torch.eye(config.width)), expected.view(2, t, -1), atol=1e-6)
    # A zero gate halves every head: sigmoid(0) = 1/2.
    gate.weight.data.zero_()
    assert torch.allclose(attend(gated, torch.eye(config.width)), heads.view(2, t, -1) / 2, atol=1e-6)


def test_registered_configuration_is_the_frozen_entry():
    config = candidate_config("canon_depth_lab")
    model = build_model(config, "meta")
    assert (config.layers, config.mlp_width, config.window_pattern) == (23, 6144, "L")
    assert config.option("head_gate") is True and config.option("source_granularity") == 2
    assert sum(p.numel() for p in model.parameters()) == 1_357_578_504
    gates = [block.attn.output_gate.weight for block in model.transformer.h]
    assert [tuple(gate.shape) for gate in gates] == [(12, 1536)] * 23
    # The gates train in the auxiliary AdamW group, never under Muon.
    kinds = {group["kind"] for group in parameter_groups(model) if any(p is gates[0] for p in group["params"])}
    assert kinds == {"adamw"}
