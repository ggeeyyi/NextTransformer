"""Dynamic K/V convolution: independent oracles, shell fidelity, and complete cost."""
from dataclasses import replace

import pytest
import torch
from torch.nn import functional as F

from nexttransformer.methods.sconv_dynamic_out import DynamicShortConvolution, ShortConvolution, ZeroLinear
from nexttransformer.methods.sconv_out import ShortConvolution as StaticControl
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
    config = debug_config("sconv_dynamic_out")
    model = build_model(config)
    control = build_model(replace(config, method="dense_full", options=()))
    shared = dict(control.named_parameters())
    for name, parameter in model.named_parameters():
        if "_sconv." in name:
            assert parameter.shape == ((config.heads * 4, config.width) if "weight_proj" in name else (config.width, 1, 4))
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
    config = debug_config("sconv_dynamic_out")
    model = build_model(config)
    nonzero_branches(model)
    with torch.no_grad():
        for module in model.modules():
            if isinstance(module, ShortConvolution):
                module.weight.normal_(std=.1)
            if isinstance(module, ZeroLinear):
                module.weight.normal_(std=.02)
    # Last layer has a value embedding, so this detects V placement relative to VE.
    attn = model.transformer.h[-1].attn
    x = torch.randn(2, 7, config.width)
    ve = torch.randn_like(x)
    angle = torch.randn(1, 7, 1, attn.head_dim // 2)
    cos, sin = angle.cos(), angle.sin()
    q = attn.c_q(x).view(2, 7, attn.n_head, attn.head_dim)
    k = dynamic_oracle(attn.c_k(x), x, attn.k_sconv).view_as(q)
    v = dynamic_oracle(attn.c_v(x), x, attn.v_sconv).view_as(q)
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
    config = debug_config("sconv_dynamic_out")
    model = build_model(config)
    nonzero_branches(model)
    with torch.no_grad():
        for module in model.modules():
            if isinstance(module, ShortConvolution):
                module.weight.normal_(std=.05)
            if isinstance(module, ZeroLinear):
                module.weight.normal_(std=.02)
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
            assert owners[name] == ("muon" if "weight_proj" in name else "adamw")
            assert parameter.grad.abs().sum() > 0, name
            count += 1
    assert count == 6 * config.n_layer


def test_cost_counts_all_four_filters_in_every_training_phase():
    config = debug_config("sconv_dynamic_out")
    actual = audit_graph(config)
    control = audit_graph(replace(config, method="dense_full", options=()))
    assert not actual["unknown"]
    assert actual["parameters"] - control["parameters"] == config.n_layer * (4 * config.width * 4 + 2 * config.width * config.heads * 4)
    for phase in ("forward", "backward", "optimizer"):
        assert actual["phases"][phase] > control["phases"][phase]


def test_invalid_options_and_cache_fail_explicitly():
    for size in (0, -1, True, 1.5):
        with pytest.raises(ValueError, match="kernel_size"):
            replace(candidate_config("sconv_dynamic_out"), options={"kernel_size": size})
    with pytest.raises(ValueError, match="full causal attention"):
        replace(candidate_config("sconv_dynamic_out"), window_pattern="SSSL")
    model = build_model(debug_config("sconv_dynamic_out"))
    with pytest.raises(NotImplementedError, match="full-sequence"):
        model(torch.zeros(1, 2, dtype=torch.long), kv_cache=object())


def dynamic_oracle(z, x, conv):
    """Explicit token/head/channel sum, independent of the vectorized implementation."""
    taps = F.linear(x, conv.weight_proj.weight).reshape(*x.shape[:2], conv.heads, conv.size)
    result = []
    head_width = z.shape[-1] // conv.heads
    for t in range(z.shape[1]):
        channels = []
        for channel in range(z.shape[-1]):
            value = z[:, t, channel]
            for lag in range(min(conv.size, t + 1)):
                tap = conv.weight[channel, 0, conv.size - 1 - lag] + taps[:, t, channel // head_width, lag]
                value = value + tap * z[:, t - lag, channel]
            channels.append(value)
        result.append(torch.stack(channels, dim=-1))
    return torch.stack(result, dim=1)


@pytest.mark.parametrize("length", [1, 2, 9])
def test_dynamic_forward_backward_independent_oracle(length):
    conv = DynamicShortConvolution(4, 4, 2).double()
    with torch.no_grad():
        conv.weight.normal_(std=.1)
        conv.weight_proj.weight.normal_(std=.1)
    z = torch.randn(2, length, 4, dtype=torch.float64, requires_grad=True)
    x = torch.randn_like(z, requires_grad=True)
    actual, expected = conv(z, x), dynamic_oracle(z, x, conv)
    torch.testing.assert_close(actual, expected, atol=1e-12, rtol=1e-12)
    probe = torch.randn_like(actual)
    params = (z, x, conv.weight, conv.weight_proj.weight)
    ga = torch.autograd.grad((actual * probe).sum(), params)
    ge = torch.autograd.grad((expected * probe).sum(), params)
    for a, e in zip(ga, ge):
        torch.testing.assert_close(a, e, atol=1e-12, rtol=1e-12)
    assert ga[-1].abs().sum() > 0


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64, torch.bfloat16])
def test_zero_generator_reduces_bitwise_to_static(dtype):
    dynamic = DynamicShortConvolution(4, 4, 2)
    static = StaticControl(4, 4)
    with torch.no_grad():
        dynamic.weight.normal_(std=.1)
        static.weight.copy_(dynamic.weight)
    z = torch.randn(2, 9, 4, dtype=dtype)
    x = torch.randn_like(z)
    torch.testing.assert_close(dynamic(z, x), static(z), atol=0, rtol=0)


def test_generator_uses_output_time_and_shares_only_within_head():
    conv = DynamicShortConvolution(4, 4, 2).double()
    with torch.no_grad():
        conv.weight.zero_()
        conv.weight_proj.weight.zero_()
        # Head zero, lag two, reads x[t,0]. Head one stays at identity.
        conv.weight_proj.weight[2, 0] = 1
    z = torch.ones(2, 6, 4, dtype=torch.float64)
    x = torch.zeros_like(z)
    x[0, 4, 0] = 3
    actual = conv(z, x)
    expected = z.clone()
    expected[0, 4, :2] += 3
    torch.testing.assert_close(actual, expected, atol=0, rtol=0)
    # A source-position impulse distinguishes lag orientation and zero padding.
    z.zero_()
    z[0, 2, :2] = 5
    expected = z.clone()
    expected[0, 4, :2] += 15
    torch.testing.assert_close(conv(z, x), expected, atol=0, rtol=0)


def test_generator_construction_does_not_consume_rng_and_matches_sout():
    before = torch.random.get_rng_state()
    ZeroLinear(16, 8, bias=False)
    torch.testing.assert_close(before, torch.random.get_rng_state(), atol=0, rtol=0)
    config = debug_config("sconv_dynamic_out")
    dynamic = build_model(config)
    static = build_model(replace(config, method="sconv_out"))
    shared = dict(static.named_parameters())
    for name, parameter in dynamic.named_parameters():
        if "weight_proj" not in name:
            torch.testing.assert_close(parameter, shared[name], atol=0, rtol=0)
    assert dynamic.transformer.h[0].attn.k_sconv.weight_proj.weight is not dynamic.transformer.h[0].attn.v_sconv.weight_proj.weight
    tokens = torch.randint(config.vocab_size, (2, config.sequence_len))
    torch.testing.assert_close(dynamic(tokens), static(tokens), atol=0, rtol=0)


def test_low_precision_shared_initialization_matches_sout():
    # Run a fresh process so imported COMPUTE_DTYPE is the formal BF16 choice.
    import os
    import subprocess
    import sys
    script = """
from dataclasses import replace
import torch
from nexttransformer.models.config import debug_config
from nexttransformer.models.lm import build_model
c = debug_config('sconv_dynamic_out')
a = build_model(c)
b = build_model(replace(c, method='sconv_out'))
for n, p in b.named_parameters():
    torch.testing.assert_close(p, dict(a.named_parameters())[n], atol=0, rtol=0)
"""
    subprocess.run([sys.executable, "-c", script], check=True,
                   env=dict(os.environ, NANOCHAT_DTYPE="bfloat16", CUDA_VISIBLE_DEVICES="", HF_HUB_OFFLINE="1"))


def test_generator_and_mixing_add_cost_in_all_phases():
    config = debug_config("sconv_dynamic_out")
    dynamic = audit_graph(config)
    static = audit_graph(replace(config, method="sconv_out"))
    assert dynamic["unknown"] == {}
    assert dynamic["parameters"] - static["parameters"] == 2 * config.n_layer * config.width * config.heads * 4
    for phase in ("forward", "backward", "optimizer"):
        assert dynamic["phases"][phase] > static["phases"][phase]
    assert any(k.startswith("forward/") and "mm" in k for k in dynamic["operators"])
    assert any(k.startswith("backward/") and "mul" in k for k in dynamic["operators"])
