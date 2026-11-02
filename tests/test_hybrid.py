"""v2 hybrids H01/H02/H03: registered layouts, per-layer mixers, windows, costs and calibration."""
from dataclasses import replace
import json

import pytest
import torch

from nexttransformer.models.backbone import CausalSelfAttention
from nexttransformer.models.linear import LinearMixer
from nexttransformer.models.sparse import SparseMixer
from nexttransformer.models.others import MLAAttention
from nexttransformer.models.lm import build_model
from nexttransformer.protocol.calibrate import Auditor
from nexttransformer.protocol.compute import audit_graph, causal_edges, selected_edges
from nexttransformer.models.config import CHUNK_BACKENDS, HYBRID, LINEAR, MLA, SPARSE, METHODS, candidate_config, debug_config, hybrid_layout
from nexttransformer.protocol.spec import Protocol, load_config, write_config
from nexttransformer.training.optimizer import parameter_groups


@pytest.fixture(autouse=True)
def reproducible():
    torch.manual_seed(42)
    torch.set_num_threads(1)


def one_based(layout, kind):
    return [i+1 for i, k in enumerate(layout) if k == kind]


def activate(model):
    """Native init zeroes the output projections; give every branch a visible effect."""
    with torch.no_grad():
        for block in model.transformer.h:
            block.attn.c_proj.weight.normal_(std=.03)
            block.mlp.c_proj.weight.normal_(std=.03)
    return model


def test_layouts_follow_the_registration():
    h02, h03 = hybrid_layout("gdn_hybrid", 24), hybrid_layout("gdn_hybrid_swa", 24)
    # H02: GDN, GDN, GDN, attention repeated, so attention at one-based 4, 8, ..., N.
    assert one_based(h02, "attention") == list(range(4, 25, 4)) and h02.count("gdn") == 18
    # H03: GDN first and SWA second (lit_gpt gated_delta_per_layer=2); the last layer is SWA.
    assert one_based(h03, "swa") == list(range(2, 25, 2)) and one_based(h03, "gdn") == list(range(1, 24, 2))
    for method, bad in (("gdn_hybrid", 22), ("gdn_hybrid_swa", 23)):
        with pytest.raises(ValueError, match="multiple"):
            replace(candidate_config(method), layers=bad)
    with pytest.raises(ValueError, match="window_pattern L"):
        replace(candidate_config("gdn_hybrid"), window_pattern="SSSL")
    with pytest.raises(ValueError, match="not a parameter"):
        replace(candidate_config("gdn_hybrid"), sliding_window=512)
    with pytest.raises(ValueError, match="current token"):
        replace(candidate_config("gdn_hybrid_swa"), sliding_window=0)
    assert candidate_config("gdn_hybrid_swa").sliding_window == 2048


@pytest.mark.parametrize("layers", [4, *range(16, 33, 2)])
def test_h01_has_exactly_two_registered_full_layers(layers):
    c = replace(debug_config("deltanet_hybrid"), layers=layers)
    m = build_model(c)
    assert one_based(c.layer_mixers, "attention") == [2, layers//2+1]
    assert [i+1 for i, b in enumerate(m.transformer.h) if isinstance(b.attn, CausalSelfAttention)] == [2, layers//2+1]
    assert c.layer_mixers[-1] == "deltanet" and isinstance(m.transformer.h[-1].attn, LinearMixer)


@pytest.mark.parametrize("layers", [0, 2, 3, 17, 25, 31])
def test_h01_rejects_colliding_or_odd_layouts(layers):
    with pytest.raises(ValueError):
        replace(candidate_config("deltanet_hybrid"), layers=layers)


@pytest.mark.parametrize("method", sorted(HYBRID))
def test_serialized_layout_is_fingerprinted_and_verified(method, tmp_path):
    c = candidate_config(method)
    assert c.serialized()["layer_mixers"] == list(c.layer_mixers)
    path = tmp_path/"c.json"
    write_config(path, c)
    assert load_config(path) == c
    raw = json.loads(path.read_text())
    raw["model"]["layer_mixers"][0] = "attention"
    path.write_text(json.dumps(raw))
    with pytest.raises(ValueError, match="layout"):
        load_config(path)
    # Only hybrids carry a layout; existing rows serialize exactly as before.
    for other in METHODS[:METHODS.index("gdn_hybrid")]:
        assert "layer_mixers" not in candidate_config(other).serialized()
        assert "sliding_window" not in candidate_config(other).serialized()


@pytest.mark.parametrize("method", sorted(HYBRID))
def test_layers_use_the_registered_mixers_and_windows(method):
    m = build_model(candidate_config(method), "meta")
    c = m.config
    for i, (block, kind) in enumerate(zip(m.transformer.h, c.layer_mixers)):
        if kind in LINEAR:
            assert isinstance(block.attn, LinearMixer) and block.attn.kind == kind
            assert hasattr(block.attn, "decay_proj") == (kind == "gdn")
            assert block.attn.q_conv.fused
        elif kind in MLA:
            assert type(block.attn) is MLAAttention
            assert block.attn.rope == 0 and block.attn.ve_gate is None
        elif kind in SPARSE:
            assert type(block.attn) is SparseMixer and block.attn.kind == kind
        else:
            # B01's native attention: 12 heads of 128, QK norm, RoPE; VE where the shell places it.
            assert type(block.attn) is CausalSelfAttention
            assert (block.attn.n_head, block.attn.n_kv_head, block.attn.head_dim) == (12, 12, 128)
        assert m._window(i, c.n_layer) == ((2047, 0) if kind == "swa" else (2048, 0))
    assert sum(isinstance(b.attn, CausalSelfAttention) for b in m.transformer.h) == sum(k not in LINEAR|MLA for k in c.layer_mixers)


@pytest.mark.parametrize("method", sorted(HYBRID))
def test_trace_records_each_layer_mixer_window_and_ve(method):
    c = debug_config(method)
    m = build_model(c)
    m(torch.randint(c.vocab_size, (1, c.sequence_len)))
    blocks = [r for r in m.last_trace if r["kind"] == "block"]
    assert [r["mixer"] for r in blocks] == [k if k in LINEAR|MLA|SPARSE else "softmax" for k in c.layer_mixers]
    assert [r["window"] for r in blocks] == [c.sliding_window-1 if k == "swa" else c.sequence_len for k in c.layer_mixers]
    assert [r["ve"] for r in blocks] == [str(i) in m.value_embeds for i in range(c.n_layer)]


def test_sliding_window_sees_exactly_w_keys_including_the_current_token():
    c = debug_config("gdn_hybrid_swa")          # sliding_window 4 at T=12
    m = activate(build_model(c))
    layer = c.layer_mixers.index("swa")
    attn, window = m.transformer.h[layer].attn, m._window(layer, c.n_layer)
    assert window == (3, 0)
    t = c.sequence_len
    x = torch.randn(1, t, c.width)
    cos_sin = (m.cos[:, :t], m.sin[:, :t])
    with torch.no_grad():
        base = attn(x, None, cos_sin, window, None)
        far = x.clone(); far[:, :t-4] += 1.     # keys more than w-1 = 3 positions back
        near = x.clone(); near[:, t-4] += 1.    # the oldest key inside the window
        torch.testing.assert_close(attn(far, None, cos_sin, window, None)[:, -1], base[:, -1], rtol=0, atol=0)
        assert not torch.allclose(attn(near, None, cos_sin, window, None)[:, -1], base[:, -1])


def test_a_window_covering_the_context_is_full_causal_attention():
    # The registered H03 window (2048 keys) covers the whole 2K context: SWA is then global.
    covering = replace(debug_config("gdn_hybrid_swa"), sliding_window=12)
    wider = replace(covering, sliding_window=13)
    narrow = replace(covering, sliding_window=4)
    x = torch.randint(covering.vocab_size, (2, covering.sequence_len))
    out = {}
    for name, c in (("covering", covering), ("wider", wider), ("narrow", narrow)):
        model = activate(build_model(c)).eval()  # build_model reseeds: identical weights
        with torch.no_grad():
            out[name] = model(x)
    torch.testing.assert_close(out["covering"], out["wider"], rtol=1e-6, atol=1e-6)
    assert not torch.allclose(out["covering"], out["narrow"])


@pytest.mark.parametrize("method", sorted(HYBRID))
def test_cost_charges_attention_edges_only_at_attention_layers(method):
    c = debug_config(method)
    report = audit_graph(c)
    assert not report["unknown"] and report["pair_block"] == CHUNK_BACKENDS.get(c.backend)
    left = [c.sliding_window-1 if k == "swa" else c.sequence_len for k in c.layer_mixers if k not in LINEAR|SPARSE]
    edges = sum(c.heads*causal_edges(c.sequence_len, w) for w in left)
    assert report["operators"].get("forward/native_attention", {}).get("matrix", 0) == 2*edges*(c.qk_nope_dim+c.v_head_dim if method == "kda_hybrid" else 2*c.head_dim)


def test_optimizer_groups_own_gdn_auxiliaries_and_attention_matrices():
    m = build_model(debug_config("gdn_hybrid"))
    owner = {name: g["kind"] for g in parameter_groups(m) for name in g["parameter_names"]}
    for name in ("transformer.h.0.attn.decay_proj.weight", "transformer.h.0.attn.beta_proj.weight",
                 "transformer.h.0.attn.output_gate.weight", "transformer.h.0.attn.log_decay",
                 "transformer.h.0.attn.dt_bias"):
        assert owner[name] == "adamw", name
    for name in ("transformer.h.0.attn.c_q.weight", "transformer.h.3.attn.c_q.weight",
                 "transformer.h.3.attn.c_proj.weight"):
        assert owner[name] == "muon", name
    assert set(owner) == {n for n, p in m.named_parameters() if p.requires_grad}


def test_v2_grid_applies_the_protocol_tie_break_on_whole_repeating_units(tmp_path, monkeypatch):
    auditor = Auditor(tmp_path)
    # Synthetic cost: depth/23 of the target plus 1e-4 per FFN unit. At 24 layers the
    # nearest-width FFN would be 5632 (-0.77%); the tie-break prefers 5760 (+0.51%),
    # which is closer to 6144. 20 layers needs a far larger FFN change.
    cost = lambda c: c.layers/23+(c.mlp_width-6144)*1e-4
    monkeypatch.setattr(auditor, "aggregate", lambda c: {"flops_per_token": cost(c)})
    monkeypatch.setattr(auditor, "ffn_slope", lambda c: 1e-4)
    chosen, selected, table = auditor.v2_grid(candidate_config("gdn_hybrid"), 1.)
    assert [d["layers"] for d in table["depths"]] == [16, 20, 24, 28, 32]
    assert (chosen.layers, chosen.mlp_width) == (24, 5760)
    assert [(t["layers"], t["mlp_width"]) for t in table["candidates"][:2]] == [(24, 5760), (24, 5632)]
    assert table["candidates"][0]["audited_error"] == pytest.approx(selected["flops_per_token"]-1)
    _, _, swa = auditor.v2_grid(candidate_config("gdn_hybrid_swa"), 1.)
    assert [d["layers"] for d in swa["depths"]] == list(range(16, 33, 2))
    _, _, h01 = auditor.v2_grid(candidate_config("deltanet_hybrid"), 1.)
    assert [d["layers"] for d in h01["depths"]] == list(range(16, 33, 2))
    for method in ("kda_hybrid", "kda_no_ve", "mla_nope", "mha_no_ve"):
        _, _, table = auditor.v2_grid(candidate_config(method), 1.)
        assert [d["layers"] for d in table["depths"]] == list(range(16, 33, 4 if method == "kda_hybrid" else 1))


@pytest.mark.parametrize("backend", ["reference", "fla_chunk64"])
def test_h01_reuses_delta_parameters_state_boundaries_and_single_stage(backend):
    c = replace(debug_config("deltanet_hybrid", sequence_len=16), backend=backend)
    m = activate(build_model(c))
    mixer = m.transformer.h[-1].attn
    pure = LinearMixer(replace(c, method="deltanet"), c.layers-1)
    pure.load_state_dict(mixer.state_dict(), strict=True)
    x, ve = torch.randn(1, 16, c.width), torch.randn(1, 16, c.width)
    x[:, 8] = 0.  # An internal boundary token is an input vector, not a state-reset instruction.
    actual = mixer(x, ve, None, None, None)
    torch.testing.assert_close(actual, pure(x, ve, None, None, None), rtol=0, atol=0)
    assert not torch.allclose(actual[:, 8:], mixer(x[:, 8:], ve[:, 8:], None, None, None))
    owner = {name: g["kind"] for g in parameter_groups(m) for name in g["parameter_names"]}
    assert all(v == "adamw" for n, v in owner.items() if ".beta_proj." in n or "_conv." in n or ".output_gain" in n)
    for step in (0, 511, 512, 46874):
        assert m.set_training_stage(step) == "joint"
    ids = torch.randint(c.vocab_size, (2, c.sequence_len))
    loss = m(ids, ids)
    assert m.last_aux_loss.item() == 0
    torch.testing.assert_close(loss, m.last_lm_loss, rtol=0, atol=0)


def test_h01_full_mha_has_early_history_and_no_future_visibility():
    m = activate(build_model(debug_config("deltanet_hybrid", sequence_len=16)))
    x = torch.randn(1, 16, m.config.width, requires_grad=True)
    attn = m.transformer.h[1].attn
    rotary = (m.cos[:, :16], m.sin[:, :16])
    full = attn(x, None, rotary, m._window(1, m.config.layers), None)
    unlimited = attn(x, None, rotary, (-1, 0), None)
    torch.testing.assert_close(full, unlimited, rtol=0, atol=0)
    grad = torch.autograd.grad(full[:, 10].square().sum(), x)[0]
    assert grad[:, 0].abs().sum() > 0 and torch.count_nonzero(grad[:, 11:]) == 0


@pytest.mark.parametrize("layers", range(16, 33))
def test_h06_keeps_exactly_the_last_three_full_layers(layers):
    c = replace(candidate_config("moba_hybrid"), layers=layers)
    assert c.layer_mixers == ("moba",)*(layers-3) + ("attention",)*3
    for bad in (0, 1, 2, 3):
        with pytest.raises(ValueError):
            hybrid_layout("moba_hybrid", bad)


@pytest.mark.parametrize("layers", [4, 16, 20, 24, 28, 32])
def test_h05_reuses_qsa_only_at_each_fourth_layer(layers):
    c = replace(debug_config("qsa_hybrid"), layers=layers)
    assert c.layer_mixers == ("gdn", "gdn", "gdn", "qsa")*(layers//4)
    model = build_model(c, "meta")
    assert [i+1 for i, b in enumerate(model.transformer.h) if isinstance(b.attn, SparseMixer)] == list(range(4, layers+1, 4))
    assert c.has_sparse and c.has_sparse_burnin and c.value_embeddings


@pytest.mark.parametrize("method,kind", [("qsa_hybrid", "qsa"), ("moba_hybrid", "moba")])
@pytest.mark.parametrize("step", [0, 511, 512, 46874])
def test_sparse_hybrid_stage_edges_auxiliary_and_cost(method, kind, step, tmp_path):
    c = debug_config(method, sequence_len=17)
    model = activate(build_model(c))
    model.set_step(step)
    ids = torch.randint(c.vocab_size, (1, c.sequence_len))
    loss = model(ids, ids)
    sparse = [b.attn for b in model.transformer.h if isinstance(b.attn, SparseMixer)]
    stage = "burnin" if c.has_sparse_burnin and step < c.burnin_steps else "joint"
    edges = selected_edges(c, 0, c.sequence_len, stage, kind=kind)*c.heads
    for mixer in sparse:
        assert mixer.kind == kind and mixer.last_stats["selected_edges"] == edges
        assert mixer.last_stats["burnin"] == (stage == "burnin")
        assert mixer.last_stats["teacher_dense_score_edges"] == (c.heads*c.sequence_len**2 if kind == "qsa" else 0)
    torch.testing.assert_close(loss, model.last_lm_loss + sum(m.last_aux for m in sparse))
    loss.backward()
    if kind == "qsa":
        assert all(m.index_q.weight.grad.abs().sum() > 0 for m in sparse)
    else:
        assert model.last_aux_loss.item() == 0
    report = audit_graph(c, stage=stage)
    assert not report["unknown"]
    assert report["operators"]["forward/indexed_attention"]["matrix"] == len(sparse)*edges*4*c.head_dim
    stages = Auditor(tmp_path).stages(c)
    assert stages == ([("burnin", None, 512), ("joint", None, 46363)] if kind == "qsa" else [("joint", None, 46875)])
    assert sum(n for _, _, n in stages)*Protocol().global_batch_tokens == 49_152_000_000
    owner = [n for g in parameter_groups(model) for n in g["parameter_names"]]
    assert len(owner) == len(set(owner)) == len(list(model.parameters()))
    model.eval()
    model(ids, ids)
    assert model.last_aux_loss.item() == 0 and not any(m.last_stats["burnin"] for m in sparse)


@pytest.mark.parametrize("method,kind", [("qsa_hybrid", "qsa"), ("moba_hybrid", "moba")])
def test_sparse_hybrid_layer_reuses_pure_module_outputs_and_gradients(method, kind):
    c = debug_config(method, sequence_len=17)
    model = activate(build_model(c)).double()
    i = c.layer_mixers.index(kind)
    mixer = model.transformer.h[i].attn
    pure = SparseMixer(replace(c, method=kind, backend="reference"), i).double()
    pure.load_state_dict(mixer.state_dict())
    rotary = (model.cos[:, :17].double(), model.sin[:, :17].double())
    x = torch.randn(2, 17, c.width, dtype=torch.float64)
    ve = torch.randn_like(x) if str(i) in model.value_embeds else None
    for burnin in (True, False):
        results = []
        for module in (mixer, pure):
            module.in_burnin = burnin
            module.zero_grad(set_to_none=True)
            inp = x.clone().requires_grad_()
            out = module(inp, ve, rotary, (17, 0), None)
            (out.square().sum() + module.last_aux).backward()
            results.append((out, inp.grad, {n:p.grad for n,p in module.named_parameters()}))
        for actual, expected in zip(results[0][:2], results[1][:2]):
            torch.testing.assert_close(actual, expected, rtol=0, atol=0)
        for n, grad in results[0][2].items():
            torch.testing.assert_close(grad, results[1][2][n], rtol=0, atol=0)
