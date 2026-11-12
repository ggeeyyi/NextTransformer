"""Gated Attention plugins: gate position, gate shapes, initialization and optimizer ownership."""
from dataclasses import replace

import pytest
import torch

from nexttransformer.models.backbone import CausalSelfAttention
from nexttransformer.models.config import candidate_config, debug_config
from nexttransformer.models.lm import build_model
from nexttransformer.methods.gated_attention import GatedAttention
from nexttransformer.training.optimizer import parameter_groups

METHODS = ("gated_attention", "gated_attention_headwise")


def attention_inputs(model, config, seed=0):
    g = torch.Generator().manual_seed(seed)
    x = torch.randn(2, config.sequence_len, config.width, generator=g)
    cos, sin = model.cos[:, :config.sequence_len], model.sin[:, :config.sequence_len]
    return x, (cos, sin), (-1, 0)


@pytest.mark.parametrize("method", METHODS)
def test_gate_sits_on_the_sdpa_output_before_c_proj(method):
    config = debug_config(method)
    model = build_model(config, "cpu", seed=1)
    attn = model.transformer.h[0].attn  # value embeddings sit on the last layer and alternate back
    assert attn.ve_gate is None
    x, cos_sin, window = attention_inputs(model, config)
    with torch.no_grad():
        torch.nn.init.normal_(attn.c_proj.weight, std=0.1)  # the shell starts it at zero
        proj = attn.c_proj.weight.clone()
        # A zero gate is σ(0) = 1/2: exactly half of Dense's attention output.
        attn.output_gate.weight.zero_()
        dense = CausalSelfAttention.forward(attn, x, None, cos_sin, window, None)
        assert torch.equal(attn(x, None, cos_sin, window, None), dense / 2)
        # A random gate multiplies the SDPA output (c_proj = identity), then c_proj applies.
        torch.nn.init.normal_(attn.output_gate.weight)
        attn.c_proj.weight.copy_(torch.eye(config.width))
        y = CausalSelfAttention.forward(attn, x, None, cos_sin, window, None)
        gate = torch.sigmoid(x @ attn.output_gate.weight.T).view(2, config.sequence_len, attn.n_head, -1)
        expected = (y.view(2, config.sequence_len, attn.n_head, attn.head_dim) * gate).view_as(y) @ proj.T
        attn.c_proj.weight.copy_(proj)
        torch.testing.assert_close(attn(x, None, cos_sin, window, None), expected)


@pytest.mark.parametrize("method,outputs", [("gated_attention", 1536), ("gated_attention_headwise", 12)])
def test_gate_shape_at_the_formal_size(method, outputs):
    attn = GatedAttention(candidate_config(method), 0)
    assert tuple(attn.output_gate.weight.shape) == (outputs, 1536)


@pytest.mark.parametrize("method", METHODS)
def test_gate_takes_c_q_initialization_and_trains_with_adamw(method):
    config = debug_config(method)
    model = build_model(config, "cpu", seed=1)
    bound = 3**0.5 * config.width**-0.5
    for block in model.transformer.h:
        w = block.attn.output_gate.weight
        assert torch.count_nonzero(w) == w.numel() and w.abs().max() <= bound
    groups = parameter_groups(model)
    owner = {name: g for g in groups for name in g["parameter_names"]}
    gates = [name for name in owner if ".output_gate." in name]
    assert len(gates) == config.layers
    assert all(owner[name]["kind"] == "adamw" for name in gates)
    assert owner["transformer.h.0.attn.c_q.weight"]["kind"] == "muon"
    # The zero-initialized c_proj passes no gradient to the gate (nor to c_q/c_k/c_v) on the first update.
    with torch.no_grad():
        for block in model.transformer.h: torch.nn.init.normal_(block.attn.c_proj.weight, std=0.02)
    model.train()
    x = torch.randint(config.vocab_size, (2, config.sequence_len))
    model(x, x).backward()
    assert all(block.attn.output_gate.weight.grad.abs().sum() > 0 for block in model.transformer.h)


def test_granularity_is_checked():
    with pytest.raises(ValueError, match="granularity"):
        replace(candidate_config("gated_attention"), options={"granularity": "channelwise"})
