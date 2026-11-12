"""v2 full-attention Dense and Residual family: layouts, paper oracles, initialization, grouping."""
from dataclasses import replace
import json
from pathlib import Path

import pytest
import torch
from torch.nn import functional as F

from nexttransformer.models.backbone import GPT, GPTConfig, norm
from nexttransformer.models.lm import FROZEN_NATIVE, build_model
from nexttransformer.models.residual import HyperConnection, ManifoldHyperConnection, depth_attention, sinkhorn_knopp
from nanochat import flash_attention
from nexttransformer.models.config import METHODS, RESIDUAL, candidate_config, debug_config
from nexttransformer.protocol.spec import load_config
from nexttransformer.training.optimizer import parameter_groups

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(autouse=True)
def reproducible():
    torch.manual_seed(42)
    torch.set_num_threads(1)


def activate(model):
    with torch.no_grad():
        for block in model.transformer.h:
            block.attn.c_proj.weight.normal_(std=.03)
            block.mlp.c_proj.weight.normal_(std=.03)


def move_residual_parameters(model, scale=.05):
    with torch.no_grad():
        for name, p in model.named_parameters():
            if name.startswith(("residual_maps.", "depth_queries")):
                p.add_(torch.randn_like(p) * scale)


def batch(c, rows=2):
    x = torch.randint(c.vocab_size, (rows, c.sequence_len))
    return x, torch.randint(c.vocab_size, x.shape)


def logits_from(model, h):
    return 15 * torch.tanh(model.lm_head(norm(h))[..., :model.config.vocab_size].float() / 15)


def attention(model, x, idx, i):
    t = idx.shape[1]
    ve = model.value_embeds[str(i)](idx).to(x.dtype) if str(i) in model.value_embeds else None
    return model.transformer.h[i].attn(norm(x), ve, (model.cos[:, :t], model.sin[:, :t]), (t, 0), None)


def assert_matching_gradients(model, loss, oracle_loss):
    params = [p for p in model.parameters() if p.requires_grad]
    for actual, expected in zip(torch.autograd.grad(loss, params, retain_graph=True),
                                torch.autograd.grad(oracle_loss, params)):
        torch.testing.assert_close(actual, expected)


def test_pre_v2_configurations_keep_their_files_and_hashes():
    pre_v2 = set(METHODS[:METHODS.index("dense_full")])
    for path in sorted(p for p in (ROOT / "configs").glob("*/*.json") if p.parent.name not in ("native", "scaling")):
        raw = json.loads(path.read_text())
        config = load_config(path)
        if config.method in pre_v2:
            assert config.serialized() == raw["model"] and "residual_streams" not in raw["model"]
    with pytest.raises(ValueError, match="not a parameter"):
        replace(candidate_config("dense"), attnres_groups=4)
    with pytest.raises(ValueError, match="full causal"):
        replace(candidate_config("hc"), window_pattern="SSSL")


def test_dense_full_is_the_native_shell_with_full_attention_in_every_layer(monkeypatch):
    formal = build_model(candidate_config("dense_full"), "meta")
    n = formal.config.n_layer
    assert {formal._window(i, n) for i in range(n)} == {(2048, 0)}
    assert sum(w == (2048, 0) for w in build_model(candidate_config("dense"), "meta").window_sizes) == 6
    c = debug_config("dense_full")
    model = build_model(c)
    activate(model)
    native = GPT(GPTConfig(sequence_len=c.sequence_len, vocab_size=c.vocab_size, n_layer=c.layers, n_head=c.heads,
                           n_kv_head=c.heads, n_embd=c.width, window_pattern="L"))
    native.init_weights()
    native.load_state_dict(model.state_dict())
    seen, original = [], flash_attention.flash_attn.flash_attn_func

    def spy(q, k, v, causal=False, window_size=(-1, -1)):
        seen.append((causal, tuple(window_size)))
        return original(q, k, v, causal=causal, window_size=window_size)

    monkeypatch.setattr(flash_attention.flash_attn, "flash_attn_func", spy)
    x, y = batch(c)
    torch.testing.assert_close(model(x), native(x))
    assert seen[:c.layers] == [(True, (c.sequence_len, 0))] * c.layers
    assert [r["window"] for r in model.last_trace if r["kind"] == "block"] == [c.sequence_len] * c.layers
    model(x, y).backward()
    native(x, y).backward()
    for (name, a), b in zip(model.named_parameters(), native.parameters()):
        torch.testing.assert_close(a.grad, b.grad, msg=name)


def test_residual_standard_matches_native_blocks_and_ignores_native_scalars():
    c = debug_config("residual_standard")
    model = build_model(c)
    activate(model)
    x, y = batch(c)
    frozen = FROZEN_NATIVE["residual_standard"]
    assert all(not getattr(model, name).requires_grad for name in frozen)

    def oracle(idx):
        h, t = model._embed_reference(idx), idx.shape[1]
        for i, block in enumerate(model.transformer.h):
            ve = model.value_embeds[str(i)](idx).to(h.dtype) if str(i) in model.value_embeds else None
            h = block(h, ve, (model.cos[:, :t], model.sin[:, :t]), (t, 0), None)
        return logits_from(model, h)

    before = model(x)
    torch.testing.assert_close(before, oracle(x))
    with torch.no_grad():
        for name in frozen:
            getattr(model, name).add_(1.)
    torch.testing.assert_close(model(x), before)
    assert_matching_gradients(model, model(x, y), F.cross_entropy(oracle(x).flatten(0, 1), y.flatten()))
    owned = {name for group in parameter_groups(model) for name in group["parameter_names"]}
    assert not owned & set(frozen)


def hc_oracle(model, idx):
    """Zhu et al. Algorithms 2-3, written with the paper's tensor layout."""
    c = model.config
    hidden = model._embed_reference(idx)[:, :, None].repeat(1, 1, c.residual_streams, 1)
    for k, hc in enumerate(model.residual_maps):
        norm_h = F.rms_norm(hidden, (hidden.size(-1),))
        alpha = torch.tanh(norm_h @ hc.dynamic_alpha) * hc.alpha_scale + hc.static_alpha[None, None]
        beta = torch.tanh(norm_h @ hc.dynamic_beta[:, 0]) * hc.beta_scale + hc.static_beta[None, None]
        mix_h = alpha.transpose(-1, -2) @ hidden
        h = mix_h[..., 0, :]
        h = attention(model, h, idx, k // 2) if k % 2 == 0 else model.transformer.h[k // 2].mlp(norm(h))
        hidden = torch.einsum("blh,bln->blnh", h, beta) + mix_h[..., 1:, :]
    return logits_from(model, hidden.sum(-2))


def test_hc_matches_paper_algorithm_and_starts_as_prenorm_residual():
    c = debug_config("hc")
    model = build_model(c)
    activate(model)
    x, y = batch(c)
    standard = build_model(debug_config("residual_standard"))
    standard.load_state_dict(model.state_dict(), strict=False)
    torch.testing.assert_close(model(x), standard(x))
    for k, hc in enumerate(model.residual_maps):
        assert hc.static_alpha[:, 0].tolist() == [float(j == k % c.residual_streams) for j in range(c.residual_streams)]
    move_residual_parameters(model)
    torch.testing.assert_close(model(x), hc_oracle(model, x))
    assert_matching_gradients(model, model(x, y), F.cross_entropy(hc_oracle(model, x).flatten(0, 1), y.flatten()))


def mhc_oracle_maps(module, hidden):
    """Xie et al. Eqs. 7-9 without the column-maximum shift."""
    n, (b, t, _, d) = module.streams, hidden.shape
    raw = F.rms_norm(hidden.reshape(b, t, n * d), (n * d,)) @ module.projection
    pre = torch.sigmoid(module.gates[0] * raw[..., :n] + module.bias[:n])
    post = 2 * torch.sigmoid(module.gates[1] * raw[..., n:2 * n] + module.bias[n:2 * n])
    matrix = torch.exp((module.gates[2] * raw[..., 2 * n:] + module.bias[2 * n:]).reshape(b, t, n, n))
    for _ in range(module.iterations):
        matrix = matrix / matrix.sum(-2, keepdim=True)
        matrix = matrix / matrix.sum(-1, keepdim=True)
    return pre, post, matrix


def mhc_oracle(model, idx):
    c = model.config
    hidden = model._embed_reference(idx)[:, :, None].repeat(1, 1, c.residual_streams, 1)
    for k, module in enumerate(model.residual_maps):
        pre, post, res = mhc_oracle_maps(module, hidden)
        h = (pre[..., None] * hidden).sum(-2)
        h = attention(model, h, idx, k // 2) if k % 2 == 0 else model.transformer.h[k // 2].mlp(norm(h))
        hidden = res @ hidden + post[..., None] * h[:, :, None]
    return logits_from(model, hidden.sum(-2))


def test_mhc_matches_paper_equations_and_respects_the_manifold():
    c = debug_config("mhc")
    model = build_model(c)
    activate(model)
    move_residual_parameters(model, scale=.5)
    x, y = batch(c)
    torch.testing.assert_close(model(x), mhc_oracle(model, x))
    assert_matching_gradients(model, model(x, y), F.cross_entropy(mhc_oracle(model, x).flatten(0, 1), y.flatten()))
    streams = torch.randn(2, c.sequence_len, c.residual_streams, c.width)
    pre, post, res = model.residual_maps[3].maps(streams)
    assert (pre >= 0).all() and (pre <= 1).all() and (post >= 0).all() and (post <= 2).all() and (res >= 0).all()
    torch.testing.assert_close(res.sum(-1), torch.ones_like(res[..., 0]))
    # Twenty iterations are approximate: near-permutation logits leave a column-sum error.
    fresh = build_model(c)
    assert fresh.residual_maps[0].iterations == 20 and fresh.residual_maps[0].gates.tolist() == pytest.approx([.01] * 3)
    move_residual_parameters(fresh)
    _, _, res = fresh.residual_maps[3].maps(streams)
    assert (res.sum(-2) - 1).abs().max() < 1e-4


def test_sinkhorn_shift_invariance_and_convergence():
    logits = torch.randn(5, 4, 4, dtype=torch.float64) * 3
    plain = logits.exp()
    for _ in range(20):
        plain = plain / plain.sum(-2, keepdim=True)
        plain = plain / plain.sum(-1, keepdim=True)
    torch.testing.assert_close(sinkhorn_knopp(logits, 20), plain, rtol=1e-12, atol=1e-12)
    column_error = lambda m: float((m.sum(-2) - 1).abs().max())
    assert column_error(sinkhorn_knopp(logits, 20)) < column_error(sinkhorn_knopp(logits, 2))


def test_sinkhorn_stays_finite_where_fp32_exp_underflows():
    """The formal mHC run reached column logit spreads of 180, so whole rows underflowed under FP32 exp."""
    logits = torch.zeros(3, 4, 4, dtype=torch.float64)
    logits[:, 1] = torch.tensor([-200., -210., -190., -205.], dtype=torch.float64)
    logits[1:] += torch.randn(2, 4, 4, dtype=torch.float64)
    oracle = logits.exp()
    for _ in range(20):
        oracle = oracle / oracle.sum(-2, keepdim=True)
        oracle = oracle / oracle.sum(-1, keepdim=True)
    single = logits.float().requires_grad_()
    res = sinkhorn_knopp(single, 20)
    (res * torch.linspace(-1, 1, 16).view(4, 4)).sum().backward()
    assert torch.isfinite(res).all() and torch.isfinite(single.grad).all()
    torch.testing.assert_close(res.double(), oracle, rtol=1e-5, atol=1e-6)
    torch.testing.assert_close(res.sum(-1), torch.ones(3, 4))


def attnres_oracle(model, idx):
    """Kimi Team Fig. 2: block_attn_res before each sublayer, boundary before attention."""
    c = model.config
    starts = set(c.attnres_group_starts)

    def block_attn_res(blocks, partial, query):
        values = torch.stack(blocks + [partial])
        logits = torch.einsum("d,nbtd->nbt", query, F.rms_norm(values, (values.size(-1),)))
        return torch.einsum("nbt,nbtd->btd", logits.softmax(0), values)

    blocks, partial = [], model._embed_reference(idx)
    for i, block in enumerate(model.transformer.h):
        h = block_attn_res(blocks, partial, model.depth_queries[2 * i])
        if i in starts:
            blocks.append(partial)
            partial = None
        out = attention(model, h, idx, i)
        partial = out if partial is None else partial + out
        h = block_attn_res(blocks, partial, model.depth_queries[2 * i + 1])
        partial = partial + block.mlp(norm(h))
    return logits_from(model, block_attn_res(blocks, partial, model.depth_queries[-1]))


def test_attnres_matches_reference_pseudocode_boundaries_and_separate_queries():
    c = debug_config("attnres")
    assert c.attnres_group_sizes == (2, 1) and c.attnres_group_starts == (0, 2)
    formal = replace(candidate_config("attnres"), layers=23)
    assert formal.attnres_group_sizes == (3, 3, 3, 3, 3, 3, 3, 2)
    model = build_model(c)
    activate(model)
    x, y = batch(c)
    assert not model.depth_queries.any()
    torch.testing.assert_close(model(x), attnres_oracle(model, x))
    assert [r["sources"] for r in model.last_trace if r["kind"] == "depth_read"] == [1, 2, 2, 2, 2, 3, 3]
    model(x, y).backward()
    assert model.depth_queries.grad.shape == (2 * c.layers + 1, c.width)
    assert (model.depth_queries.grad.abs().sum(-1)[1:] > 0).all()
    model.zero_grad(set_to_none=True)
    move_residual_parameters(model, scale=.5)
    torch.testing.assert_close(model(x), attnres_oracle(model, x))
    assert_matching_gradients(model, model(x, y), F.cross_entropy(attnres_oracle(model, x).flatten(0, 1), y.flatten()))


@pytest.mark.parametrize("method", sorted(RESIDUAL))
def test_residual_parameters_use_auxiliary_adamw_and_native_scalars_stay_frozen(method):
    model = build_model(debug_config(method))
    model.set_training_stage(100)
    owner = {name: group for group in parameter_groups(model) for name in group["parameter_names"]}
    for name, p in model.named_parameters():
        if name.startswith(("residual_maps.", "depth_queries")):
            assert owner[name]["kind"] == "adamw" and owner[name]["weight_decay"] == 0
            assert owner[name]["betas"] == (.9, .95)
        if name in FROZEN_NATIVE[method]:
            assert name not in owner and not p.requires_grad


def test_residual_operators_match_fp64_paper_formulas_and_gradcheck():
    torch.manual_seed(0)
    b, t, n, d = 1, 3, 4, 6
    hc = HyperConnection(d, n, 2, .01).double()
    mhc = ManifoldHyperConnection(d, n, 3, .01, 20).double()
    for module in (hc, mhc):
        module.reset_parameters()
        with torch.no_grad():
            for p in module.parameters():
                p.add_(torch.randn_like(p) * .3)
    streams = torch.randn(b, t, n, d, dtype=torch.float64, requires_grad=True)
    output = torch.randn(b, t, d, dtype=torch.float64, requires_grad=True)
    normed = F.rms_norm(streams, (d,))
    alpha = torch.tanh(normed @ hc.dynamic_alpha) * hc.alpha_scale + hc.static_alpha
    beta = torch.tanh(normed @ hc.dynamic_beta[:, 0]) * hc.beta_scale + hc.static_beta
    mix = alpha.transpose(-1, -2) @ streams
    h, mixed, weights = hc.read(streams)
    torch.testing.assert_close(h, mix[..., 0, :], rtol=1e-12, atol=1e-12)
    torch.testing.assert_close(hc.write(mixed, output, weights), beta[..., None] * output[:, :, None] + mix[..., 1:, :],
                               rtol=1e-12, atol=1e-12)
    pre, post, res = mhc_oracle_maps(mhc, streams)
    h, mixed, weights = mhc.read(streams)
    torch.testing.assert_close(h, (pre[..., None] * streams).sum(-2), rtol=1e-12, atol=1e-12)
    torch.testing.assert_close(mixed, res @ streams, rtol=1e-12, atol=1e-12)
    torch.testing.assert_close(weights, post, rtol=1e-12, atol=1e-12)
    sources = [torch.randn(b, t, d, dtype=torch.float64, requires_grad=True) for _ in range(3)]
    query = torch.randn(d, dtype=torch.float64, requires_grad=True)
    attend = lambda q, *v: depth_attention(q, list(v), [F.rms_norm(x, (d,)) for x in v])
    values = torch.stack(sources)
    oracle = torch.einsum("nbt,nbtd->btd", torch.einsum("d,nbtd->nbt", query, F.rms_norm(values, (d,))).softmax(0), values)
    torch.testing.assert_close(attend(query, *sources), oracle, rtol=1e-12, atol=1e-12)
    assert torch.autograd.gradcheck(lambda s, o: hc.write(*hc.read(s)[1:2], o, hc.read(s)[2]) + hc.read(s)[0][:, :, None], (streams, output))
    assert torch.autograd.gradcheck(lambda s: torch.cat([x.flatten() for x in mhc.read(s)]), (streams,))
    assert torch.autograd.gradcheck(attend, (query, *sources))
