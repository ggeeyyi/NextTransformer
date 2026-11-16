"""XSA's paper projector, attention gradients, cache alignment and cost accounting."""
from dataclasses import replace
import json

import pytest
import torch
from torch.nn import functional as F

from nanochat import flash_attention as attention_backend
from nanochat.engine import KVCache
from nanochat.flash_attention import HAS_FA3, flash_attn
from nexttransformer.methods.xsa import ExclusiveSelfAttention, exclusive_projection
from nexttransformer.models.backbone import GPTConfig
from nexttransformer.models.config import candidate_config, debug_config
from nexttransformer.models.lm import build_model
from nexttransformer.protocol.compute import audit_graph
from nexttransformer.training.optimizer import parameter_groups


@pytest.fixture(autouse=True)
def reproducible():
    torch.manual_seed(42)
    torch.set_num_threads(1)


def projector_oracle(y, v):
    """Explicit rank-one projector, independent of normalize/dot implementation."""
    outer = v.unsqueeze(-1) * v.unsqueeze(-2)
    denominator = v.square().sum(-1, keepdim=True).clamp_min(1e-24).unsqueeze(-1)
    projector = torch.eye(v.size(-1), dtype=v.dtype, device=v.device) - outer / denominator
    return (projector @ y.unsqueeze(-1)).squeeze(-1)


def attention_oracle(q, k, v):
    q, k, v = (x.transpose(1, 2) for x in (q, k, v))
    repeat = q.size(1) // k.size(1)
    k, v = (x.repeat_interleave(repeat, dim=1) for x in (k, v))
    scores = q @ k.transpose(-1, -2) / q.size(-1)**0.5
    mask = torch.ones(scores.shape[-2:], dtype=torch.bool, device=q.device).tril()
    return (scores.masked_fill(~mask, -torch.inf).softmax(-1) @ v).transpose(1, 2)


def rotary_tables(length, head_dim, dtype=torch.float64):
    angles = torch.arange(length, dtype=dtype)[:, None] * 10000**(
        -torch.arange(0, head_dim, 2, dtype=dtype) / head_dim)
    return angles.cos()[None, :, None], angles.sin()[None, :, None]


def mixer_oracle(attn, x, ve, cos_sin):
    b, t, _ = x.shape
    q = F.linear(x, attn.c_q.weight).view(b, t, attn.n_head, attn.head_dim)
    k = F.linear(x, attn.c_k.weight).view(b, t, attn.n_kv_head, attn.head_dim)
    v = F.linear(x, attn.c_v.weight).view_as(k)
    if ve is not None:
        gate = 3 * F.linear(x[..., :attn.ve_gate_channels], attn.ve_gate.weight).sigmoid()
        v = v + gate.unsqueeze(-1) * ve.view_as(v)
    cos, sin = cos_sin

    def rotate_and_normalize(a):
        first, second = a.chunk(2, dim=-1)
        rotated = torch.cat((first * cos + second * sin, second * cos - first * sin), -1)
        return rotated / (rotated.square().mean(-1, keepdim=True) + torch.finfo(a.dtype).eps).sqrt() * 1.2

    y = attention_oracle(rotate_and_normalize(q), rotate_and_normalize(k), v)
    effective_v = v.repeat_interleave(attn.n_head // attn.n_kv_head, dim=2)
    z = projector_oracle(y, effective_v)
    return F.linear(z.flatten(2), attn.c_proj.weight)


def test_projector_forward_backward_and_head_isolation():
    y, v = (torch.randn(1, 2, 2, 3, dtype=torch.float64, requires_grad=True) for _ in range(2))
    z = exclusive_projection(y, v)
    expected = projector_oracle(y, v)
    torch.testing.assert_close(z, expected, atol=1e-12, rtol=1e-12)
    torch.testing.assert_close((z * v).sum(-1), torch.zeros_like(z[..., 0]), atol=1e-12, rtol=0)
    weights = torch.randn_like(z)
    actual_grads = torch.autograd.grad((z * weights).sum(), (y, v), retain_graph=True)
    expected_grads = torch.autograd.grad((expected * weights).sum(), (y, v))
    for actual, want in zip(actual_grads, expected_grads):
        torch.testing.assert_close(actual, want, atol=1e-12, rtol=1e-12)
    assert torch.autograd.gradcheck(exclusive_projection, (y, v))
    changed = v.detach().clone()
    changed[:, :, 1] *= torch.tensor([2., -3., 0.5])
    torch.testing.assert_close(exclusive_projection(y, changed)[:, :, 0], z[:, :, 0], atol=0, rtol=0)


@pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
def test_zero_values_and_low_precision_projection(dtype):
    y = torch.randn(2, 3, 2, 8).to(dtype).requires_grad_()
    v = torch.randn_like(y).requires_grad_()
    with torch.no_grad():
        v[0, 0].zero_()
    z = exclusive_projection(y, v)
    expected = projector_oracle(y.float(), v.float()).to(dtype)
    assert z.dtype == dtype and torch.isfinite(z).all()
    torch.testing.assert_close(z, expected, atol=2e-3 if dtype == torch.bfloat16 else 5e-4, rtol=2e-3)
    torch.testing.assert_close(z[0, 0], y[0, 0], atol=0, rtol=0)
    z.float().square().sum().backward()
    assert torch.isfinite(y.grad).all() and torch.isfinite(v.grad).all()


def test_norm_epsilon_matches_paper_normalize_for_tiny_values():
    y = torch.tensor([[[[1., 2., -3.]]]], dtype=torch.float64, requires_grad=True)
    v = torch.tensor([[[[1e-14, -2e-14, 3e-14]]]], dtype=torch.float64, requires_grad=True)
    z = exclusive_projection(y, v)
    expected = projector_oracle(y, v)
    torch.testing.assert_close(z, expected, atol=1e-14, rtol=1e-14)
    actual_grads = torch.autograd.grad(z.sum(), (y, v))
    expected_grads = torch.autograd.grad(expected.sum(), (y, v))
    for actual, want in zip(actual_grads, expected_grads):
        torch.testing.assert_close(actual, want, atol=1e-14, rtol=1e-12)


@pytest.mark.parametrize("with_ve", [False, True])
@pytest.mark.parametrize("kv_heads", [1, 2])
def test_mixer_matches_independent_attention_and_parameter_gradients(with_ve, kv_heads):
    config = GPTConfig(n_embd=16, n_head=2, n_kv_head=kv_heads, n_layer=2, window_pattern="L")
    attn = ExclusiveSelfAttention(config, int(with_ve)).double()
    x = torch.randn(2, 5, 16, dtype=torch.float64, requires_grad=True)
    ve = torch.randn(2, 5, kv_heads * 8, dtype=torch.float64, requires_grad=True) if with_ve else None
    tables = rotary_tables(5, 8)
    # Direct module initialization gives c_proj nonzero weights, unlike the model's zero branch init.
    assert attn.c_proj.weight.count_nonzero() > 0
    actual = attn(x, ve, tables, (-1, 0), None)
    expected = mixer_oracle(attn, x, ve, tables)
    torch.testing.assert_close(actual, expected, atol=1e-10, rtol=1e-10)
    torch.testing.assert_close(actual[:, 0], torch.zeros_like(actual[:, 0]), atol=1e-12, rtol=0)
    inputs = (x,) + ((ve,) if with_ve else ()) + tuple(attn.parameters())
    weights = torch.randn_like(actual)
    actual_grads = torch.autograd.grad((actual * weights).sum(), inputs)
    expected_grads = torch.autograd.grad((expected * weights).sum(), inputs)
    for actual_grad, expected_grad in zip(actual_grads, expected_grads):
        torch.testing.assert_close(actual_grad, expected_grad, atol=1e-9, rtol=1e-9)
        assert actual_grad.abs().sum() > 0


def test_self_score_stays_in_softmax_and_context_is_projected():
    q = k = torch.zeros(1, 2, 1, 2, dtype=torch.float64)
    v = torch.tensor([[[[1., 0.]], [[1., 1.]]]], dtype=torch.float64)
    y = flash_attn.flash_attn_func(q, k, v, causal=True, window_size=(-1, 0))
    actual = exclusive_projection(y, v)
    torch.testing.assert_close(actual[0, 1, 0], torch.tensor([.25, -.25], dtype=v.dtype))
    # Excluding the self key would produce v[0] at position 1, a different operation.
    assert not torch.allclose(actual[:, 1], v[:, 0])


@pytest.mark.parametrize("with_ve", [False, True])
@pytest.mark.parametrize("kv_heads", [1, 2])
@pytest.mark.parametrize("chunks", [(4, 1, 1, 1), (2, 3, 2)])
def test_mixer_prefill_and_decode_match_full_sequence(with_ve, kv_heads, chunks):
    config = GPTConfig(n_embd=16, n_head=2, n_kv_head=kv_heads, n_layer=1, window_pattern="L")
    attn = ExclusiveSelfAttention(config, 0).eval()
    x = torch.randn(2, 7, 16)
    ve = torch.randn(2, 7, kv_heads * 8) if with_ve else None
    cos, sin = rotary_tables(7, 8, dtype=x.dtype)
    cache = KVCache(2, kv_heads, 7, 8, 1, "cpu", x.dtype)
    with torch.no_grad():
        full = attn(x, ve, (cos, sin), (-1, 0), None)
        outputs = []
        for length in chunks:
            start = cache.get_pos()
            end = start + length
            outputs.append(attn(x[:, start:end], ve[:, start:end] if with_ve else None,
                                (cos[:, start:end], sin[:, start:end]), (-1, 0), cache))
            assert cache.get_pos() == end
    torch.testing.assert_close(torch.cat(outputs, dim=1), full, atol=2e-6, rtol=2e-6)


def test_parameter_ownership_and_checkpoint_roundtrip(tmp_path):
    config = debug_config("xsa")
    model = build_model(config, "cpu", seed=42)
    dense = build_model(replace(config, method="dense_full"), "cpu", seed=42)
    assert {n: p.shape for n, p in model.named_parameters()} == {n: p.shape for n, p in dense.named_parameters()}
    owners = lambda m: {name: group["kind"] for group in parameter_groups(m) for name in group["parameter_names"]}
    assert owners(model) == owners(dense)
    with torch.no_grad():
        for block in model.transformer.h:
            block.attn.c_proj.weight.normal_(std=.03)
            block.mlp.c_proj.weight.normal_(std=.03)
    tokens = torch.randint(config.vocab_size, (2, config.sequence_len))
    model.eval()
    expected = model(tokens)
    path = tmp_path / "xsa.pt"
    torch.save(model.state_dict(), path)
    restored = build_model(config, "cpu", seed=17)
    restored.load_state_dict(torch.load(path, weights_only=True))
    restored.eval()
    torch.testing.assert_close(restored(tokens), expected, atol=0, rtol=0)


def test_projection_is_charged_in_both_directions_without_extra_parameters():
    config = debug_config("xsa")
    xsa = audit_graph(config)
    dense = audit_graph(replace(config, method="dense_full"))
    assert not xsa["unknown"]
    assert xsa["parameters"] == dense["parameters"]
    assert xsa["phases"]["optimizer"] == dense["phases"]["optimizer"]
    for phase in ("forward", "backward"):
        assert xsa["phases"][phase] > dense["phases"][phase]


def test_bfloat16_training_and_resume_match_continuous_updates(tmp_path, monkeypatch):
    from nexttransformer.models import backbone, lm
    from tests.test_train import losses, runner, weights

    # Synthetic rows and fake evaluators exercise the real optimizer, trainer and checkpoints.
    monkeypatch.setattr(lm, "COMPUTE_DTYPE", torch.bfloat16)
    monkeypatch.setattr(backbone, "COMPUTE_DTYPE", torch.bfloat16)
    run = runner("xsa", tmp_path, monkeypatch)
    continuous, resumed = tmp_path / "continuous", tmp_path / "resumed"
    run(continuous, "--smoke-steps", "3")
    run(resumed, "--smoke-steps", "3", "--max-wall-hours", "1e-12")
    assert json.loads((resumed / "run_status.json").read_text())["step"] == 1
    run(resumed, "--smoke-steps", "3")
    assert losses(continuous) == losses(resumed)
    actual, expected = weights(resumed, 3), weights(continuous, 3)
    for name in actual:
        torch.testing.assert_close(actual[name], expected[name], atol=0, rtol=0)
    status = json.loads((resumed / "run_status.json").read_text())
    assert status["phase"] == "smoke_complete" and status["mode"] == "research"


def test_full_causal_attention_is_required():
    with pytest.raises(ValueError, match="full causal attention"):
        replace(candidate_config("xsa"), window_pattern="SSSL")


@pytest.mark.skipif(not torch.cuda.is_available() or not HAS_FA3, reason="requires CUDA and FA3")
@pytest.mark.parametrize("length", [17, 2048])
def test_fa3_bfloat16_output_and_gradients_match_math_oracle(monkeypatch, length):
    monkeypatch.setattr(attention_backend, "USE_FA3", True)
    q, k, v = ((torch.randn(1, length, 2, 128, device="cuda") * .3).bfloat16().requires_grad_()
               for _ in range(3))
    qr, kr, vr = (a.detach().float().requires_grad_() for a in (q, k, v))
    actual = exclusive_projection(flash_attn.flash_attn_func(q, k, v, causal=True, window_size=(-1, 0)), v)
    expected = projector_oracle(attention_oracle(qr, kr, vr), vr)
    torch.testing.assert_close(actual.float(), expected, atol=3e-3, rtol=2e-2)
    weights = torch.randn_like(expected)
    actual_grads = torch.autograd.grad((actual.float() * weights).sum(), (q, k, v))
    expected_grads = torch.autograd.grad((expected * weights).sum(), (qr, kr, vr))
    for name, actual_grad, expected_grad in zip(("q", "k", "v"), actual_grads, expected_grads):
        # V receives two cancelling paths, each rounded to BF16 by autograd. Bound both
        # individual errors and the total relative error, including near-zero gradients.
        tolerance = 2e-2 if name == "v" else 5e-3
        torch.testing.assert_close(actual_grad.float(), expected_grad, atol=tolerance, rtol=2e-2)
        assert (actual_grad.float() - expected_grad).norm() / expected_grad.norm() < 1e-2


@pytest.mark.skipif(not torch.cuda.is_available() or not HAS_FA3, reason="requires CUDA and FA3")
def test_fa3_mixer_value_embeddings_and_cached_chunks(monkeypatch):
    monkeypatch.setattr(attention_backend, "USE_FA3", True)
    config = GPTConfig(n_embd=256, n_head=2, n_kv_head=2, n_layer=1, window_pattern="L")
    attn = ExclusiveSelfAttention(config, 0).cuda().eval()
    x = torch.randn(1, 17, 256, device="cuda", dtype=torch.bfloat16)
    ve = torch.randn_like(x)
    cos, sin = (a.cuda().bfloat16() for a in rotary_tables(17, 128, dtype=torch.float32))
    cache = KVCache(1, 2, 17, 128, 1, "cuda", x.dtype)
    with torch.no_grad():
        full = attn(x, ve, (cos, sin), (-1, 0), None)
        chunks = []
        for length in (5, 1, 4, 7):
            start = cache.get_pos()
            end = start + length
            chunks.append(attn(x[:, start:end], ve[:, start:end],
                               (cos[:, start:end], sin[:, start:end]), (-1, 0), cache))
        assert cache.get_pos() == 17
        torch.testing.assert_close(torch.cat(chunks, dim=1), full, atol=1e-2, rtol=2e-2)
