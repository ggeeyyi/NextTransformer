"""H04 and its VE-off controls; MLA oracles reused from xxy/others-attention 8cba0a3."""
from dataclasses import replace
import copy
import pytest
import torch
from nexttransformer.models.config import MLA, NO_VE, candidate_config, debug_config
from nexttransformer.models.lm import build_model
from nexttransformer.models.linear import LinearMixer
from nexttransformer.models.others import MLAAttention
from nexttransformer.training.optimizer import parameter_groups
from nexttransformer.models.backbone import GPT, GPTConfig


@pytest.mark.parametrize("layers", [4,16,20,24,28,32])
def test_h04_registered_layout_and_ve_off(layers):
    c=replace(candidate_config("kda_hybrid"),layers=layers)
    m=build_model(c,"meta")
    assert c.layer_mixers==("kda","kda","kda","mla_nope")*(layers//4)
    assert not m.value_embeds
    for i,block in enumerate(m.transformer.h):
        assert block.attn.ve_gate is None
        assert isinstance(block.attn,MLAAttention if i%4==3 else LinearMixer)
    a=m.transformer.h[3].attn
    assert {n:tuple(p.shape) for n,p in a.named_parameters()}=={
        "c_q.weight":(768,1536),"q_up.weight":(1536,768),"c_k.weight":(256,1536),
        "c_v.weight":(3072,256),"c_proj.weight":(1536,1536)}


@pytest.mark.parametrize("changes", [dict(layers=23),dict(qk_rope_dim=64),dict(q_rank=512)])
def test_h04_rejects_unregistered_structure(changes):
    with pytest.raises(ValueError):replace(candidate_config("kda_hybrid"),**changes)


@pytest.mark.parametrize("method", sorted(NO_VE))
def test_controls_have_no_ve_and_keep_shared_optimizer_and_stage(method):
    c=debug_config(method);m=build_model(c)
    assert not m.value_embeds and all(b.attn.ve_gate is None for b in m.transformer.h)
    names=[n for group in parameter_groups(m) for n in group["parameter_names"]]
    assert len(names)==len(set(names))==sum(p.requires_grad for p in m.parameters())
    assert not any("value_embeds" in n or "ve_gate" in n for n in names)
    for step in (0,512,46874):assert m.set_training_stage(step)=="joint"
    ids=torch.randint(c.vocab_size,(1,c.sequence_len));loss=m(ids,ids)
    assert m.last_aux_loss==0
    torch.testing.assert_close(loss,m.last_lm_loss,rtol=0,atol=0)


@pytest.mark.parametrize("backend", ["reference","fla_chunk64"])
def test_h04_reuses_kda_and_has_row_local_state(backend):
    c=replace(debug_config("kda_hybrid",sequence_len=16),backend=backend)
    m=build_model(c);activate(m)
    pure=build_model(replace(debug_config("kda_no_ve",sequence_len=16),backend=backend))
    a,b=m.transformer.h[0].attn,pure.transformer.h[0].attn
    b.load_state_dict(a.state_dict(),strict=True)
    x=torch.randn(2,16,c.width);x[:,8]=0
    y=a(x,None,None,None,None)
    torch.testing.assert_close(y,b(x,None,None,None,None),rtol=0,atol=0)
    torch.testing.assert_close(y[:1],a(x[:1],None,None,None,None))
    assert not torch.allclose(y[:,8:],a(x[:,8:],None,None,None,None))
    a(torch.randn_like(x),None,None,None,None)
    torch.testing.assert_close(y,a(x,None,None,None,None),rtol=0,atol=0)


def test_mla_full_history_without_position_encoding():
    m=build_model(debug_config("kda_hybrid"));activate(m);a=m.transformer.h[3].attn
    x=torch.randn(1,12,m.config.width,requires_grad=True)
    y=a(x,None,None,(12,0),None)
    torch.testing.assert_close(y,a(x,None,(torch.randn(1),torch.randn(1)),(-1,0),None))
    g=torch.autograd.grad(y[:,10].square().sum(),x)[0]
    assert g[:,0].abs().sum()>0 and torch.count_nonzero(g[:,11:])==0


def test_mha_control_is_native_full_attention_without_ve():
    c=debug_config("mha_no_ve");m=build_model(c);activate(m)
    native=GPT(GPTConfig(sequence_len=c.sequence_len,vocab_size=c.vocab_size,n_layer=c.layers,
                         n_head=c.heads,n_kv_head=c.heads,n_embd=c.width,window_pattern="L"))
    native.value_embeds=torch.nn.ModuleDict()
    for block in native.transformer.h:block.attn.ve_gate=None
    native.init_weights();native.load_state_dict(m.state_dict(),strict=True)
    ids=torch.randint(c.vocab_size,(2,c.sequence_len))
    torch.testing.assert_close(m(ids),native(ids))
    m(ids,ids).backward();native(ids,ids).backward()
    for (name,p),q in zip(m.named_parameters(),native.parameters()):
        torch.testing.assert_close(p.grad,q.grad,msg=name)


@pytest.fixture(autouse=True)
def reproducible():
    torch.manual_seed(42)
    torch.set_num_threads(1)


def activate(model):
    """Zero-initialized output projections would hide every attention gradient."""
    with torch.no_grad():
        for block in model.transformer.h:
            block.attn.c_proj.weight.normal_(std=.03)
            block.mlp.c_proj.weight.normal_(std=.03)


def rms(z):
    return z * torch.rsqrt(z.square().mean(-1, keepdim=True) + torch.finfo(z.dtype).eps)


def assert_same_gradients(actual, expected, params):
    g = torch.randn_like(actual)
    for a, b in zip(torch.autograd.grad(actual, params, g, retain_graph=True), torch.autograd.grad(expected, params, g)):
        torch.testing.assert_close(a, b)


def explicit_mla(attn, c, x, cos, sin):
    """DeepSeek-V2 MLA (2405.04434v5 eqs. 9-19) written per head from the module's weights."""
    b, t, _ = x.shape
    h, nope, rope, vd, r = c.heads, c.qk_nope_dim, c.qk_rope_dim, c.v_head_dim, c.kv_rank
    c_q = rms(x @ attn.c_q.weight.T)
    q = torch.einsum("btr,hdr->bthd", c_q, attn.q_up.weight.view(h, nope + rope, c.q_rank))
    compressed = x @ attn.c_k.weight.T
    c_kv = rms(compressed[..., :r])
    up = attn.c_v.weight.view(h, nope + vd, r)
    k_c = torch.einsum("btr,hdr->bthd", c_kv, up[:, :nope])
    v = torch.einsum("btr,hdr->bthd", c_kv, up[:, nope:])
    scores = torch.einsum("bihd,bjhd->bhij", q[..., :nope], k_c)
    if rope:
        def rotate(z):  # rotate-half pairs (i, i + rope/2) by -theta, the native convention
            z1, z2 = z[..., :rope // 2], z[..., rope // 2:]
            return torch.cat((z1 * cos + z2 * sin, -z1 * sin + z2 * cos), -1)
        q_r = rotate(q[..., nope:])
        k_r = rotate(compressed[..., r:][:, :, None, :])
        scores = scores + torch.einsum("bihd,bjd->bhij", q_r, k_r[:, :, 0])
    visible = torch.ones(t, t, dtype=torch.bool, device=x.device).tril()
    weights = (scores * (nope + rope)**-.5).masked_fill(~visible, float("-inf")).softmax(-1)
    heads = torch.einsum("bhij,bjhd->bihd", weights, v)
    return heads.reshape(b, t, h * vd) @ attn.c_proj.weight.T, (q, c_kv, up, weights)


@pytest.mark.parametrize("method", sorted(MLA))
def test_mla_matches_explicit_latent_reconstruction(method):
    c = debug_config(method)
    model = build_model(c).double()
    activate(model)
    attn = model.transformer.h[0].attn
    t = c.sequence_len
    cos, sin = (model.rope_cos[:, :t], model.rope_sin[:, :t]) if c.qk_rope_dim else (None, None)
    x = torch.randn(2, t, c.width, dtype=torch.float64, requires_grad=True)
    out = attn(x, None, (cos, sin), (t, 0), None)
    expected, _ = explicit_mla(attn, c, x, cos, sin)
    torch.testing.assert_close(out, expected)
    assert_same_gradients(out, expected, [x, *attn.parameters()])
    with torch.no_grad():
        changed = x.clone()
        changed[:, 5:] += torch.randn_like(changed[:, 5:])
        torch.testing.assert_close(attn(changed, None, (cos, sin), (t, 0), None)[:, :5], out[:, :5])
    with pytest.raises(ValueError, match="Value Embedding"):
        attn(x, x, (cos, sin), (t, 0), None)


@pytest.mark.parametrize("method", sorted(MLA))
def test_mla_projection_absorption_matches_reconstructed_keys_and_values(method):
    """The key up-projection folds into the query and the value up-projection into the output."""
    c = debug_config(method)
    model = build_model(c).double()
    activate(model)
    attn = model.transformer.h[0].attn
    t, nope, vd = c.sequence_len, c.qk_nope_dim, c.v_head_dim
    cos, sin = (model.rope_cos[:, :t], model.rope_sin[:, :t]) if c.qk_rope_dim else (None, None)
    x = torch.randn(2, t, c.width, dtype=torch.float64)
    reconstructed, (q, c_kv, up, weights) = explicit_mla(attn, c, x, cos, sin)
    k_c = torch.einsum("btr,hdr->bthd", c_kv, up[:, :nope])
    q_latent = torch.einsum("bthd,hdr->bthr", q[..., :nope], up[:, :nope])
    torch.testing.assert_close(torch.einsum("bihr,bjr->bhij", q_latent, c_kv),
                               torch.einsum("bihd,bjhd->bhij", q[..., :nope], k_c))
    latent_out = torch.einsum("bhij,bjr->bihr", weights, c_kv)
    absorbed_o = torch.einsum("ohd,hdr->ohr", attn.c_proj.weight.view(c.width, c.heads, vd), up[:, nope:])
    torch.testing.assert_close(torch.einsum("bihr,ohr->bio", latent_out, absorbed_o), reconstructed)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA backend numerical check")
@pytest.mark.parametrize("length", [65, 256])
def test_mla_cuda_bf16_matches_fp64_outputs_and_all_gradients(length, record_property):
    c=candidate_config("mla_nope")
    a=MLAAttention(c,0).cuda()
    with torch.no_grad():
        for p in (a.c_q,a.c_k):p.weight.uniform_(-(3/c.width)**.5,(3/c.width)**.5)
        a.init_weights();a.c_proj.weight.normal_(std=.03)
    a=a.bfloat16();oracle=copy.deepcopy(a).double()
    x=torch.randn(1,length,c.width,device="cuda",dtype=torch.bfloat16,requires_grad=True)
    ref=x.detach().double().requires_grad_()
    out=a(x,None,None,(length,0),None);expected,_=explicit_mla(oracle,c,ref,None,None)
    weight=torch.randn_like(out)
    got=torch.autograd.grad((out*weight).sum(),[x,*a.parameters()])
    wanted=torch.autograd.grad((expected*weight.double()).sum(),[ref,*oracle.parameters()])
    relative=lambda x,y:float(((x.double()-y).norm()/y.norm().clamp_min(1e-30)).detach())
    output_error=relative(out,expected);gradient_error=max(relative(x,y) for x,y in zip(got,wanted))
    record_property("output_relative_l2",output_error)
    record_property("max_gradient_relative_l2",gradient_error)
    # The shared FLA checker uses these same BF16-vs-FP64 tolerances.
    assert output_error<=1e-2 and gradient_error<=3e-2
