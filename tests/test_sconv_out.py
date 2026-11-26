"""Static four-site convolution: shell fidelity, placement, gradients, and cost."""
from dataclasses import replace

import pytest
import torch
from torch.nn import functional as F

from nexttransformer.methods.sconv_out import ShortConvolution
from nexttransformer.models.backbone import apply_rotary_emb, norm
from nexttransformer.models.config import candidate_config, debug_config
from nexttransformer.models.lm import build_model
from nexttransformer.protocol.compute import audit_graph
from nexttransformer.training.optimizer import parameter_groups


@pytest.fixture(autouse=True)
def reproducible():
    torch.manual_seed(42)
    torch.set_num_threads(1)


def oracle(x, weight):
    size = weight.shape[-1]
    result = []
    for t in range(x.shape[1]):
        value = x[:, t]
        for lag in range(min(size, t + 1)):
            value = value + x[:, t - lag] * weight[:, 0, size - 1 - lag]
        result.append(value)
    return torch.stack(result, dim=1)


@pytest.mark.parametrize("length", [1, 2, 9])
def test_filter_and_gradients_match_causal_sum(length):
    conv = ShortConvolution(3, 4).double()
    with torch.no_grad():
        conv.weight.normal_(std=.2)
    x = torch.randn(2, length, 3, dtype=torch.float64, requires_grad=True)
    actual, expected = conv(x), oracle(x, conv.weight)
    torch.testing.assert_close(actual, expected)
    probe = torch.randn_like(actual)
    ga = torch.autograd.grad((actual * probe).sum(), (x, conv.weight))
    ge = torch.autograd.grad((expected * probe).sum(), (x, conv.weight))
    for a, e in zip(ga, ge):
        torch.testing.assert_close(a, e)


def nonzero_branches(model):
    with torch.no_grad():
        for block in model.transformer.h:
            block.attn.c_proj.weight.normal_(std=.03)
            block.mlp.c_proj.weight.normal_(std=.03)


def test_identity_matches_dense_weights_outputs_and_shared_gradients():
    config = debug_config("sconv_out")
    model = build_model(config)
    control = build_model(replace(config, method="dense_full", options=()))
    shared = dict(control.named_parameters())
    for name, parameter in model.named_parameters():
        if "_sconv." in name:
            assert parameter.shape == (config.width, 1, 4)
            assert parameter.count_nonzero() == 0
        else:
            torch.testing.assert_close(parameter, shared[name], atol=0, rtol=0)
    nonzero_branches(model)
    control.load_state_dict({k: v for k, v in model.state_dict().items() if "_sconv." not in k})
    tokens = torch.randint(config.vocab_size, (2, config.sequence_len))
    torch.testing.assert_close(model(tokens), control(tokens), atol=0, rtol=0)
    model(tokens, tokens).backward()
    control(tokens, tokens).backward()
    for name, parameter in model.named_parameters():
        if "_sconv." not in name and parameter.grad is not None:
            torch.testing.assert_close(parameter.grad, shared[name].grad, atol=0, rtol=0)


def test_four_positions_against_explicit_attention_and_mlp():
    config = debug_config("sconv_out")
    model = build_model(config)
    nonzero_branches(model)
    with torch.no_grad():
        for module in model.modules():
            if isinstance(module, ShortConvolution):
                module.weight.normal_(std=.1)
    # Last layer has a value embedding, so this detects V placement relative to VE.
    attn = model.transformer.h[-1].attn
    x = torch.randn(2, 7, config.width)
    ve = torch.randn_like(x)
    angle = torch.randn(1, 7, 1, attn.head_dim // 2)
    cos, sin = angle.cos(), angle.sin()
    q = attn.c_q(x).view(2, 7, attn.n_head, attn.head_dim)
    k = oracle(attn.c_k(x), attn.k_sconv.weight).view_as(q)
    v = oracle(attn.c_v(x), attn.v_sconv.weight).view_as(q)
    v = v + (3 * torch.sigmoid(attn.ve_gate(x[..., :12]))).unsqueeze(-1) * ve.view_as(q)
    q, k = apply_rotary_emb(q, cos, sin), apply_rotary_emb(k, cos, sin)
    q, k = norm(q) * 1.2, norm(k) * 1.2
    attended = F.scaled_dot_product_attention(q.transpose(1, 2), k.transpose(1, 2),
                                              v.transpose(1, 2), is_causal=True)
    projected = attn.c_proj(attended.transpose(1, 2).contiguous().view_as(x))
    expected = oracle(projected, attn.out_sconv.weight)
    torch.testing.assert_close(attn(x, ve, (cos, sin), (-1, 0), None), expected)
    mlp = model.transformer.h[-1].mlp
    projected = mlp.c_proj(torch.relu(mlp.c_fc(x)).square())
    torch.testing.assert_close(mlp(x), oracle(projected, mlp.out_sconv.weight))


def test_model_causality_row_isolation_and_filter_gradients():
    config = debug_config("sconv_out")
    model = build_model(config)
    nonzero_branches(model)
    with torch.no_grad():
        for module in model.modules():
            if isinstance(module, ShortConvolution):
                module.weight.normal_(std=.05)
    tokens = torch.randint(config.vocab_size, (2, config.sequence_len))
    cut = config.sequence_len // 2
    changed = tokens.clone()
    changed[0, cut:] = (changed[0, cut:] + 1) % config.vocab_size
    changed[1] = (changed[1] + 1) % config.vocab_size
    torch.testing.assert_close(model(tokens)[0, :cut], model(changed)[0, :cut], atol=0, rtol=0)
    groups = parameter_groups(model)
    names = [name for group in groups for name in group["parameter_names"]]
    assert len(names) == len(set(names))
    owners = {name: group["kind"] for group in groups for name in group["parameter_names"]}
    model(tokens, tokens).backward()
    count = 0
    for name, parameter in model.named_parameters():
        if parameter.requires_grad:
            assert parameter.grad is not None and torch.isfinite(parameter.grad).all(), name
        if "_sconv." in name:
            assert owners[name] == "adamw"
            assert parameter.grad.abs().sum() > 0, name
            count += 1
    assert count == 4 * config.n_layer


def test_cost_counts_all_four_filters_in_every_training_phase():
    config = debug_config("sconv_out")
    actual = audit_graph(config)
    control = audit_graph(replace(config, method="dense_full", options=()))
    assert not actual["unknown"]
    assert actual["parameters"] - control["parameters"] == 4 * config.n_layer * config.width * 4
    for phase in ("forward", "backward", "optimizer"):
        assert actual["phases"][phase] > control["phases"][phase]


def test_invalid_options_and_cache_fail_explicitly():
    for size in (0, -1, True, 1.5):
        with pytest.raises(ValueError, match="kernel_size"):
            replace(candidate_config("sconv_out"), options={"kernel_size": size})
    with pytest.raises(ValueError, match="full causal attention"):
        replace(candidate_config("sconv_out"), window_pattern="SSSL")
    model = build_model(debug_config("sconv_out"))
    with pytest.raises(NotImplementedError, match="full-sequence"):
        model(torch.zeros(1, 2, dtype=torch.long), kv_cache=object())
