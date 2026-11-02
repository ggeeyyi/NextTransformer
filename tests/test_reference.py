"""Mechanism tests deliberately use nonzero branches to expose causal bugs."""
from dataclasses import replace
import json
import math

import pytest
import torch
from torch.nn import functional as F

from nexttransformer.models.backbone import GPT, GPTConfig
from nexttransformer.models.config import METHODS, candidate_config, debug_config
from nexttransformer.protocol.spec import Protocol, load_config, write_config
from nexttransformer.models.lm import build_model
from nexttransformer.models.linear import (CausalDepthwise, ELU_QK_TEMPERATURE, LinearMixer, _ELUChunkFP32,
                                     chunk_attention, chunk_reference, elu_feature_map, fused_gated_rms_norm,
                                     init_gdn_gate, recurrent_attention)
from nexttransformer.models.looped import act_weights, exit_distribution, huginn_depth, ouro_gate_loss
from nexttransformer.models.sparse import indexed_attention, safe_softmax


@pytest.fixture(autouse=True)
def reproducible():
    torch.manual_seed(42)
    torch.set_num_threads(1)


def activate_branches(model):
    with torch.no_grad():
        for block in model.transformer.h:
            block.attn.c_proj.weight.normal_(std=.03)
            block.mlp.c_proj.weight.normal_(std=.03)


def test_protocol_is_fixed_and_rejects_overrides(tmp_path):
    p=Protocol()
    assert p.steps==46875 and len(p.validation_steps)==189 and len(p.core_steps)==6
    assert p.accumulation(8)==8 and p.accumulation(16)==4
    with pytest.raises(ValueError):p.accumulation(3)
    with pytest.raises(ValueError):p.accumulation(8,world_size=4)
    path=tmp_path/"config.json"
    write_config(path,candidate_config("fixed_recursive"))
    assert load_config(path).method=="fixed_recursive"
    raw=json.loads(path.read_text());raw["protocol"]["total_tokens"]//=2
    path.write_text(json.dumps(raw))
    with pytest.raises(ValueError,match="controlled"):load_config(path)


@pytest.mark.parametrize("method",METHODS)
def test_nonzero_model_backward_and_causal_prefix(method):
    c=debug_config(method)
    m=build_model(c);activate_branches(m)
    m.set_step(c.burnin_steps+1)  # test sparse path, not only dense burn-in
    idx=torch.randint(c.vocab_size,(2,c.sequence_len))
    labels=torch.randint(c.vocab_size,idx.shape)
    loss=m(idx,labels);loss.backward()
    assert torch.isfinite(loss)
    for name,p in m.named_parameters():
        if p.requires_grad:
            assert p.grad is not None,name
            assert torch.isfinite(p.grad).all(),name
    m.eval()
    with torch.no_grad():
        changed=idx.clone();changed[:,6:]=(changed[:,6:]+7)%c.vocab_size
        torch.testing.assert_close(m(idx)[:,:6],m(changed)[:,:6],atol=2e-6,rtol=2e-6)
        # BPB is CE and excludes auxiliary penalties, even in training mode.
        logits=m(idx)
        expected=F.cross_entropy(logits.flatten(0,1),labels.flatten(),reduction="none")
        torch.testing.assert_close(m(idx,labels,loss_reduction="none"),expected)


def test_dense_delegates_logits_and_gradients():
    c=debug_config("dense")
    a=build_model(c)
    b=GPT(GPTConfig(sequence_len=c.sequence_len,vocab_size=c.vocab_size,n_layer=c.layers,
                    n_head=c.heads,n_kv_head=c.heads,n_embd=c.width))
    b.init_weights();b.load_state_dict(a.state_dict())
    x=torch.randint(c.vocab_size,(2,c.sequence_len));y=torch.randint(c.vocab_size,x.shape)
    torch.testing.assert_close(a(x),b(x),rtol=0,atol=0)
    a(x,y).backward();b(x,y).backward()
    for pa,pb in zip(a.parameters(),b.parameters()):
        torch.testing.assert_close(pa.grad,pb.grad,rtol=0,atol=0)


def test_act_remainder_mass_and_gradient():
    prob=torch.tensor([[.2,.7]],requires_grad=True)
    cum=torch.zeros_like(prob);alive=torch.ones_like(prob,dtype=torch.bool)
    masses=[]
    for i in range(3):
        mass,cum,alive,rem=act_weights(prob,cum,alive,final=i==2)
        masses.append(mass)
    torch.testing.assert_close(torch.stack(masses).sum(0),torch.ones_like(prob))
    loss=sum((i+1)*m.sum() for i,m in enumerate(masses));loss.backward()
    assert (prob.grad.abs()>0).all()


def test_huginn_detach_and_reproducible_depth():
    c=replace(debug_config("huginn"),huginn_backprop_loops=1,max_loops=4)
    assert [huginn_depth(c,i) for i in range(40)]==[huginn_depth(c,i) for i in range(40)]
    m=build_model(c);activate_branches(m)
    step=next(i for i in range(100) if huginn_depth(c,i)>1)
    m.set_step(step)
    x=torch.randint(c.vocab_size,(2,c.sequence_len));m(x,x).backward()
    records=[r for r in m.last_trace if r["kind"]=="state_adapter"]
    assert sum(r["backward"] for r in records)==1
    assert m.transformer.h[0].attn.c_q.weight.grad.abs().sum()>0
    assert m.state_adapter.weight.grad.abs().sum()>0


def test_ouro_gate_signal_alignment_and_backbone_freeze():
    gates=[torch.zeros(1,2,requires_grad=True),torch.zeros(1,2,requires_grad=True)]
    losses=[torch.tensor([[2.,2.]]),torch.tensor([[1.,2.]]),torch.tensor([[1.,1.]])]
    loss=ouro_gate_loss(gates,losses).sum();loss.backward()
    assert gates[0].grad[0,0]>0 and gates[0].grad[0,1]<0
    assert gates[1].grad[0,0]<0 and gates[1].grad[0,1]>0
    torch.testing.assert_close(exit_distribution(gates).sum(-1),torch.ones(1,2))
    c=debug_config("ouro");m=build_model(c)
    m.set_training_stage(Protocol().steps-c.ouro_gate_steps)
    x=torch.randint(c.vocab_size,(2,c.sequence_len));m(x,x).backward()
    assert all(p.grad is None for n,p in m.named_parameters() if not n.startswith("halter."))
    assert m.halter.weight.grad is not None


def test_indexed_attention_matches_dense_and_gradients():
    b,h,t,d=1,2,5,4
    q,k,v=(torch.randn(b,h,t,d,dtype=torch.float64,requires_grad=True) for _ in range(3))
    indices=torch.arange(t)[None,None,None].expand(b,h,t,t)
    valid=torch.ones(t,t,dtype=torch.bool).tril()[None,None].expand_as(indices)
    y=indexed_attention(q,k,v,indices,valid)
    oracle=safe_softmax(q@k.transpose(-2,-1)/d**.5,valid)@v
    torch.testing.assert_close(y,oracle,rtol=1e-10,atol=1e-10)
    grads=torch.autograd.grad(y.square().sum(),(q,k,v),retain_graph=True)
    expected=torch.autograd.grad(oracle.square().sum(),(q,k,v))
    for actual,want in zip(grads,expected):torch.testing.assert_close(actual,want,rtol=1e-10,atol=1e-10)
    assert torch.autograd.gradcheck(lambda a,b,c:indexed_attention(a,b,c,indices,valid),(q,k,v))


@pytest.mark.parametrize("kind",["deltanet","gla","kda","retnet","elu_linear","elu_norm","elu_qknorm","gdn"])
def test_linear_scan_matches_independent_history_oracle(kind):
    b,t,h,d=1,5,2,3
    q,k,v=(torch.randn(b,t,h,d,dtype=torch.float64,requires_grad=True) for _ in range(3))
    if kind in ("elu_linear","elu_norm","elu_qknorm"):q,k=F.elu(q)+1,F.elu(k)+1
    if kind in ("deltanet","kda","gdn"):q,k=F.normalize(q,dim=-1),F.normalize(k,dim=-1)
    beta=torch.full((b,t,h),.4,dtype=q.dtype) if kind in ("deltanet","kda","gdn") else None
    decay=torch.full_like(q,.8) if kind in ("gla","kda","retnet") else None
    if kind=="gdn":decay=torch.rand(b,t,h,1,dtype=q.dtype)*.5+.5  # one gate per head and token
    actual,state=recurrent_attention(q,k,v,kind=kind,beta=beta,decay=decay)
    expected=[]
    for i in range(t):
        if kind in ("deltanet","kda","gdn"):
            state_i=torch.zeros(b,h,d,d,dtype=q.dtype)
            eye=torch.eye(d,dtype=q.dtype)
            for j in range(i+1):
                transition=eye-beta[:,j,:,None,None]*(k[:,j,:,:,None]*k[:,j,:,None,:])
                if decay is not None:transition=transition*decay[:,j,:,None,:]
                state_i=transition@state_i+beta[:,j,:,None,None]*k[:,j,:,:,None]*v[:,j,:,None,:]
            yi=(q[:,i,:,:,None]*state_i).sum(-2)
        else:
            qs=q[:,i,None]
            if decay is None:coef=(qs*k[:,:i+1]).sum(-1)
            else:
                coef=torch.stack([(q[:,i]*k[:,j]*decay[:,j+1:i+1].prod(1)).sum(-1) for j in range(i+1)],1)
            yi=(coef[...,None]*v[:,:i+1]).sum(1)
            if kind=="elu_linear":yi=yi/(coef.sum(1)[...,None]+1e-6)
        expected.append(yi)
    expected=torch.stack(expected,1)
    torch.testing.assert_close(actual,expected,rtol=1e-9,atol=1e-9)
    ga=torch.autograd.grad(actual.square().sum(),(q,k,v),retain_graph=True)
    ge=torch.autograd.grad(expected.square().sum(),(q,k,v),retain_graph=True)
    for a,e in zip(ga,ge):torch.testing.assert_close(a,e,rtol=1e-8,atol=1e-8)
    first,s=recurrent_attention(q[:,:2],k[:,:2],v[:,:2],kind=kind,
        beta=beta[:,:2] if beta is not None else None,decay=decay[:,:2] if decay is not None else None)
    second,_=recurrent_attention(q[:,2:],k[:,2:],v[:,2:],kind=kind,initial_state=s,
        beta=beta[:,2:] if beta is not None else None,decay=decay[:,2:] if decay is not None else None)
    torch.testing.assert_close(torch.cat((first,second),1),actual,rtol=1e-10,atol=1e-10)


def test_deeploop_scales_all_residual_matrices_and_native_value_embeddings():
    c=debug_config("deeploop")
    m=build_model(c)
    beta=(8*c.maximum_depth)**-.5
    K=len(m.transformer.h)
    def spectral(proj,residual):
        n_out,n_in=proj.weight.shape
        std=(.5/n_in**.5)*min(1.,(n_out/n_in)**.5)
        return std/K if residual else std
    for block in m.transformer.h:
        assert block.alpha==(2*c.maximum_depth)**.5
        # Official LoopDeepNorm: spectral base init, residual outputs / K, then beta on V/O/FFN matrices.
        for proj,residual in ((block.attn.c_v,False),(block.attn.c_proj,True),(block.mlp.c_fc,False),(block.mlp.c_proj,True)):
            expected=beta*spectral(proj,residual)
            assert abs(float(proj.weight.detach().std())/expected-1)<.15
        # Q/K keep the spectral base init without beta.
        for proj in (block.attn.c_q,block.attn.c_k):
            assert abs(float(proj.weight.detach().std())/spectral(proj,False)-1)<.15
    assert m.value_embeds
    for embedding in m.value_embeds.values():
        assert abs(float(embedding.weight.detach().float().std())*c.width**.5/beta-1)<.15
    assert not m.resid_lambdas.requires_grad and not m.x0_lambdas.requires_grad


def _chunk_case(kind,t,b=2,h=2,d=3,dv=5,strength=1.):
    """Inputs as LinearMixer passes them, FP64, all differentiable: leaves, recurrent-oracle
    inputs, chunk_attention inputs, oracle kwargs, chunk kwargs. GDN hands chunk_attention its
    raw q/k and the q scale (the L2 norm happens there); the oracle gets normalized q/k."""
    raw=[torch.randn(b,t,h,n,dtype=torch.float64,requires_grad=True) for n in (d,d,dv)]
    gate=torch.randn(b,t,h,*((d,) if kind=="kda" else ()),dtype=torch.float64,requires_grad=True)
    logit=torch.randn(b,t,h,dtype=torch.float64,requires_grad=True)
    q,k,v=raw
    if kind in ("deltanet","kda","gdn"):
        q,k=F.normalize(q,dim=-1)*d**-.5,F.normalize(k,dim=-1)
        g=-F.softplus(gate-1)*strength;beta=torch.sigmoid(logit)
        if kind=="deltanet":return (*raw,logit),(q,k,v),(q,k,v),dict(beta=beta),dict(beta=beta)
        leaves=(raw[0],raw[1],raw[2],gate,logit)
        if kind=="kda":return leaves,(q,k,v),(q,k,v),dict(decay=g.exp(),beta=beta),dict(g=g,beta=beta)
        return leaves,(q,k,v),tuple(raw),dict(decay=g.exp()[...,None],beta=beta),dict(g=g,beta=beta,qk_l2norm_scale=d**-.5)
    if kind=="retnet":
        g=(1-2.**(-2-torch.arange(h,dtype=torch.float64))).log()  # stronger than formal decay
        return tuple(raw),(q,k,v),(q,k,v),dict(decay=g.exp()[None,None,:,None].expand(b,t,-1,1)),dict(g=g)
    q,k=F.elu(q)+1,F.elu(k)+1
    return tuple(raw),(q,k,v),(q,k,v),{},{}


@pytest.mark.parametrize("kind",["deltanet","kda","retnet","elu_linear","elu_norm","elu_qknorm","gdn"])
@pytest.mark.parametrize("t",[1,63,65,2048])
def test_chunk_form_matches_recurrent_oracle(kind,t):
    leaves,oracle,inputs,scan,chunked=_chunk_case(kind,t)
    expected,_=recurrent_attention(*oracle,kind=kind,**scan)
    actual=chunk_attention(*inputs,kind=kind,chunk=64,**chunked)
    torch.testing.assert_close(actual,expected,rtol=1e-9,atol=1e-9)
    weight=torch.randn_like(expected)
    ga=torch.autograd.grad((actual*weight).sum(),leaves,retain_graph=True)
    ge=torch.autograd.grad((expected*weight).sum(),leaves)
    for a,e in zip(ga,ge):torch.testing.assert_close(a,e,rtol=1e-8,atol=1e-8)


def test_gdn_chunk_form_is_the_kda_algebra_with_a_shared_gate():
    """GDN's chunk form equals the registered KDA chunk form with the head gate on every feature."""
    leaves,(q,k,v),_,_,chunked=_chunk_case("gdn",130)
    actual=chunk_reference(q,k,v,kind="gdn",g=chunked["g"],beta=chunked["beta"],chunk=64)
    shared=chunked["g"][...,None].expand(*q.shape)
    expected=chunk_reference(q,k,v,kind="kda",g=shared,beta=chunked["beta"],chunk=64)
    torch.testing.assert_close(actual,expected,rtol=1e-12,atol=1e-12)
    weight=torch.randn_like(expected)
    ga=torch.autograd.grad((actual*weight).sum(),leaves,retain_graph=True)
    ge=torch.autograd.grad((expected*weight).sum(),leaves)
    for a,e in zip(ga,ge):torch.testing.assert_close(a,e,rtol=1e-10,atol=1e-10)


def test_gdn_chunk_form_stays_exact_for_strong_gates():
    """Pairwise exp(b_i-b_j), i >= j: no exponent is positive, unlike the exp(b)exp(-b) KDA form
    (record 014). Here the within-chunk cumulative log decay reaches thousands."""
    leaves,oracle,raw,scan,chunked=_chunk_case("gdn",200,strength=60.)
    assert float(chunked["g"].detach()[:,:64].sum(1).min())<-1000
    expected,_=recurrent_attention(*oracle,kind="gdn",**scan)
    fp32=[x.detach().float().requires_grad_() for x in (*raw,chunked["g"],chunked["beta"])]
    actual=chunk_attention(*fp32[:3],kind="gdn",g=fp32[3],beta=fp32[4],chunk=64,qk_l2norm_scale=chunked["qk_l2norm_scale"])
    weight=torch.randn_like(expected)
    grads=torch.autograd.grad((actual.double()*weight).sum(),fp32)
    wanted=torch.autograd.grad((expected*weight).sum(),(*raw,chunked["g"],chunked["beta"]))
    assert actual.isfinite().all() and all(g.isfinite().all() for g in grads)
    assert (actual.double()-expected).norm()/expected.norm()<1e-5
    for a,e in zip(grads,wanted):assert (a.double()-e).norm()/e.norm()<1e-4


@pytest.mark.parametrize("method",["deltanet","deltanet_hybrid","kda","retnet","elu_linear","elu_norm","elu_qknorm","gdn"])
def test_chunked_backend_matches_scan_model_and_is_causal(method):
    scan=replace(debug_config(method,sequence_len=70),layers=4 if method=="deltanet_hybrid" else 1,backend="reference")
    chunked=replace(scan,backend="fla_chunk64")
    a=build_model(scan);activate_branches(a)
    b=build_model(chunked);b.load_state_dict(a.state_dict())
    idx=torch.randint(scan.vocab_size,(3,scan.sequence_len));labels=torch.randint(scan.vocab_size,idx.shape)
    torch.testing.assert_close(b(idx),a(idx),rtol=2e-5,atol=2e-5)
    a(idx,labels).backward();b(idx,labels).backward()
    for (name,pa),pb in zip(a.named_parameters(),b.parameters()):
        torch.testing.assert_close(pb.grad,pa.grad,rtol=2e-4,atol=2e-5,msg=name)
    b.eval()
    with torch.no_grad():
        changed=idx.clone();changed[:,40:]=(changed[:,40:]+7)%scan.vocab_size
        torch.testing.assert_close(b(idx)[:,:40],b(changed)[:,:40],rtol=0,atol=0)
        # Each packed row starts from a zero state: other rows never leak in.
        torch.testing.assert_close(b(idx[1:2]),b(idx)[1:2],rtol=1e-5,atol=1e-5)


def test_gdn_gate_is_per_head_with_the_fla_initialization():
    c=debug_config("gdn");m=build_model(c)
    values=[]
    for block in m.transformer.h:
        mixer=block.attn
        assert mixer.decay_proj.weight.shape==(c.heads,c.width)
        assert mixer.log_decay.shape==mixer.dt_bias.shape==(c.heads,)
        assert not hasattr(mixer,"gate_down") and hasattr(mixer,"output_gate")
        values.append(torch.cat((mixer.log_decay,mixer.dt_bias)).detach())
    assert len({tuple(v.tolist()) for v in values})==len(values)  # drawn, not constant
    kda=build_model(debug_config("kda")).transformer.h[0].attn  # KDA keeps its registered constants
    assert (kda.log_decay==0).all() and (kda.dt_bias==-3).all()
    log_a,dt_bias=torch.empty(20000),torch.empty(20000)
    init_gdn_gate(log_a,dt_bias)
    a,dt=log_a.exp(),F.softplus(dt_bias)
    assert 0<a.min() and a.max()<=16 and abs(float(a.mean())-8)<.2
    assert 1e-3*(1-1e-5)<=dt.min() and dt.max()<=.1*(1+1e-5)
    assert abs(float(dt.log().mean())-(math.log(1e-3)+math.log(.1))/2)<.05  # log-uniform
    seen=[]
    original=chunk_attention.__globals__["chunk_attention"]
    def capture(q,k,v,**kwargs):
        seen.append(kwargs);return original(q,k,v,**kwargs)
    chunk_attention.__globals__["chunk_attention"]=capture
    try:m(torch.randint(c.vocab_size,(2,c.sequence_len)))
    finally:chunk_attention.__globals__["chunk_attention"]=original
    assert all(s["kind"]=="gdn" and s["g"].shape==(2,c.sequence_len,c.heads) and s["g"].dtype==torch.float32
               and (s["g"]<0).all() and s["qk_l2norm_scale"]==c.head_dim**-.5 for s in seen)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="the FLA GDN path (chunk_kda) is CUDA-only")
@pytest.mark.parametrize("strength",[1.,60.])
def test_gdn_fla_kernel_matches_chunk_reference(strength):
    """Training dtypes (BF16 q/k/v/beta, FP32 log decay) against the FP64 chunk form."""
    torch.manual_seed(0)
    _,_,raw,_,chunked=_chunk_case("gdn",300,b=2,h=4,d=64,dv=64,strength=strength)
    ref=[x.detach().cuda().requires_grad_() for x in (*raw,chunked["g"],chunked["beta"])]
    low=[x.detach().to(torch.float32 if i==3 else torch.bfloat16).requires_grad_() for i,x in enumerate(ref)]
    scale=chunked["qk_l2norm_scale"]
    expected=chunk_attention(*ref[:3],kind="gdn",g=ref[3],beta=ref[4],chunk=64,implementation="torch",qk_l2norm_scale=scale)
    actual=chunk_attention(*low[:3],kind="gdn",g=low[3],beta=low[4],chunk=64,qk_l2norm_scale=scale)
    w=torch.randn_like(expected)
    ge=torch.autograd.grad((expected*w).sum(),ref);ga=torch.autograd.grad((actual.double()*w).sum(),low)
    assert (actual.double()-expected).norm()/expected.norm()<1e-2
    for a,e in zip(ga,ge):assert (a.double()-e).norm()/e.norm()<3e-2


def test_framework_rms_norm_default_epsilon_is_the_fp32_one():
    """fused_gated_rms_norm reproduces norm(): F.rms_norm(eps=None) on BF16 uses FP32's epsilon."""
    y=(torch.randn(8,128)*.03).bfloat16()
    manual=lambda eps:y.float()*torch.rsqrt(y.float().pow(2).mean(-1,keepdim=True)+eps)
    framework=F.rms_norm(y,(128,)).float()
    torch.testing.assert_close(framework,manual(torch.finfo(torch.float32).eps),rtol=1e-2,atol=1e-2)
    assert (framework-manual(torch.finfo(torch.bfloat16).eps)).abs().max()>.5


@pytest.mark.skipif(not torch.cuda.is_available(), reason="FLA's fused gated RMSNorm is CUDA-only")
def test_fused_gated_rms_norm_matches_the_framework_ops():
    torch.manual_seed(0)
    y,z=(torch.randn(2,300,4,64,device="cuda",dtype=torch.float64) for _ in range(2))
    gain=torch.empty(64,device="cuda").uniform_(.8,1.2)
    ref=[y.clone().requires_grad_(),z.clone().requires_grad_(),gain.double().requires_grad_()]
    low=[y.bfloat16().requires_grad_(),z.bfloat16().requires_grad_(),gain.clone().requires_grad_()]
    # The framework norm() of BF16 activations: F.rms_norm's default epsilon is FP32's, not BF16's.
    expected=F.rms_norm(ref[0],(64,),eps=torch.finfo(torch.float32).eps)*ref[2]*F.silu(ref[1])
    actual=fused_gated_rms_norm(*low)
    w=torch.randn_like(expected)
    ge=torch.autograd.grad((expected*w).sum(),ref);ga=torch.autograd.grad((actual.double()*w).sum(),low)
    assert actual.dtype==torch.bfloat16 and (actual.double()-expected).norm()/expected.norm()<1e-2
    for a,e in zip(ga,ge):assert (a.double()-e).norm()/e.norm()<2e-2


@pytest.mark.skipif(not torch.cuda.is_available(), reason="KDA output fusion is CUDA-only")
@pytest.mark.parametrize("length", [256, 2048])
@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_kda_gated_output_preserves_native_norm_and_gradients(length, dtype):
    torch.manual_seed(42)
    y, gate = [torch.randn(1, length, 12, 128, device="cuda", dtype=dtype, requires_grad=True) for _ in range(2)]
    gain = torch.empty(128, device="cuda").uniform_(.8, 1.2).requires_grad_()
    expected = (F.rms_norm(y, (128,))*gain.to(dtype))*F.silu(gate)
    actual = fused_gated_rms_norm(y, gate, gain.to(dtype), preserve_rounding=True)
    weight = torch.randn_like(y)
    old = torch.autograd.grad(expected, (y, gate, gain), weight)
    new = torch.autograd.grad(actual, (y, gate, gain), weight)
    relative = lambda a, b: float((a.float()-b.float()).norm()/b.float().norm().clamp_min(1e-20))
    assert relative(actual, expected) < (1e-4 if dtype == torch.float32 else .01)
    for a, b in zip(new, old):
        assert relative(a, b) < (1e-4 if dtype == torch.float32 else .03)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="the production FLA mixer path is CUDA-only")
@pytest.mark.parametrize("method",["gdn","deltanet","kda"])
def test_linear_mixer_cuda_path_matches_the_fp64_audited_path(method):
    """Production BF16 mixer, including fused convolution and gated RMSNorm,
    against the audited torch chunk path of the same weights in FP64 on CPU."""
    torch.manual_seed(0)
    c=replace(debug_config(method,width=128,sequence_len=130),backend="fla_chunk64")
    ref=LinearMixer(c,0).double()
    with torch.no_grad():ref.output_gain.uniform_(.8,1.2);ref.c_proj.weight.normal_(std=.05)
    cuda=LinearMixer(c,0).cuda();cuda.load_state_dict(ref.state_dict())
    x=torch.randn(2,c.sequence_len,c.width,dtype=torch.float64)
    xr,xc=x.clone().requires_grad_(),x.cuda().bfloat16().requires_grad_()
    # Same semantics in both: norm() of BF16 activations uses FP32's epsilon (F.rms_norm default);
    # an FP64 input would use FP64's, which matters for these small debug-scale outputs.
    module=LinearMixer.forward.__globals__;original=module["norm"]
    module["norm"]=lambda z:F.rms_norm(z,(z.size(-1),),eps=torch.finfo(torch.float32).eps)
    try:yr=ref(xr,None,None,None,None)
    finally:module["norm"]=original
    yc=cuda(xc,None,None,None,None)
    w=torch.randn_like(yr)
    gr=torch.autograd.grad((yr*w).sum(),[xr,*ref.parameters()])
    gc=torch.autograd.grad((yc.double()*w.cuda()).sum(),[xc,*cuda.parameters()])
    assert (yc.double().cpu()-yr).norm()/yr.norm()<2e-2
    names=["input",*(n for n,_ in ref.named_parameters())]
    for name,a,e in zip(names,gc,gr):assert (a.double().cpu()-e).norm()/e.norm()<5e-2,name


@pytest.mark.skipif(not torch.cuda.is_available(), reason="FLA causal_conv1d is CUDA-only")
def test_fused_causal_conv_matches_reference_path():
    torch.manual_seed(0)
    plain=CausalDepthwise(256,4).cuda();fused=CausalDepthwise(256,4,fused=True).cuda()
    fused.load_state_dict(plain.state_dict())
    x=torch.randn(2,300,256,device="cuda",requires_grad=True);w=torch.randn(2,300,256,device="cuda")
    a,b=plain(x),fused(x)
    torch.testing.assert_close(b,a,rtol=1e-4,atol=1e-5)
    ga=torch.autograd.grad((a*w).sum(),(x,plain.weight));gb=torch.autograd.grad((b*w).sum(),(x,fused.weight))
    for u,v in zip(gb,ga):torch.testing.assert_close(u,v,rtol=1e-4,atol=1e-4)


def _elu_inputs(device, dtype, t=130, scale=1.):
    torch.manual_seed(0)
    q,k,v=(torch.randn(2,t,3,16,device=device,dtype=dtype) for _ in range(3))
    return [(F.elu(x*scale)+1).detach().requires_grad_(True) for x in (q,k)]+[v.requires_grad_(True)]


@pytest.mark.parametrize("t",[1,63,65,130,2048])
@pytest.mark.parametrize("kind",["elu_linear","elu_norm","elu_qknorm"])
def test_elu_fp32_function_matches_registered_chunk_path(t,kind):
    """The CUDA ELU+1 path (batched chunk loop) is the audited chunk algebra."""
    leaves=_elu_inputs("cpu",torch.float64,t=t)
    expected=chunk_reference(*leaves,kind=kind,chunk=64)
    if kind=="elu_linear":
        expected=expected/((leaves[0]*leaves[1].cumsum(1)).sum(-1,keepdim=True)+1e-6)
    actual=_ELUChunkFP32.apply(*leaves,64,kind)
    torch.testing.assert_close(actual,expected,rtol=1e-12,atol=1e-12)
    w=torch.randn_like(expected)
    for a,e in zip(torch.autograd.grad((actual*w).sum(),leaves),torch.autograd.grad((expected*w).sum(),leaves)):
        torch.testing.assert_close(a,e,rtol=1e-10,atol=1e-10)


@pytest.mark.parametrize("dtype",[torch.bfloat16,torch.float16,torch.float32,torch.float64])
def test_elu_feature_negative_tail_and_unused_exp_branch(dtype):
    x=torch.tensor([-80.,-20.,-10.,-7.,-6.25,-6.,0.,1.,100.,1000.],dtype=dtype,requires_grad=True)
    y=elu_feature_map(x)
    assert y.dtype==(torch.float64 if dtype==torch.float64 else torch.float32)
    assert (y>0).all() and y.isfinite().all()
    expected=torch.cat((x[:6].double().exp(),x[6:].double()+1))
    torch.testing.assert_close(y.double(),expected,rtol=1e-7,atol=0)
    grad,=torch.autograd.grad(y.sum(),x)
    expected_grad=torch.cat((x[:6].double().exp(),torch.ones(4,dtype=torch.float64))).to(dtype)
    torch.testing.assert_close(grad,expected_grad,rtol=1e-6,atol=0)
    assert grad.isfinite().all()  # positive exp must never overflow in the unused branch


@pytest.mark.parametrize("kind",["elu_linear","elu_norm","elu_qknorm"])
@pytest.mark.parametrize("backend",["reference","fla_chunk64"])
def test_elu_mixer_bf16_full_path_matches_dense_fp64(kind,backend):
    """Exercise the production feature/attention/norm/gain/cast boundaries."""
    c=replace(debug_config(kind,sequence_len=65),backend=backend)
    mixer=LinearMixer(c,0)
    with torch.no_grad():
        mixer.c_q.weight.copy_(torch.eye(c.width)*3)
        mixer.c_k.weight.copy_(torch.eye(c.width)*2)
        if kind!="elu_linear":mixer.output_gain.uniform_(.8,1.2)
    x=(torch.randn(2,c.sequence_len,c.width)*4-2).bfloat16().requires_grad_()
    captured={}
    def projection(name):
        def hook(module,args,out):captured[name]=out
        return hook
    handles=[getattr(mixer,"c_"+name).register_forward_hook(projection(name)) for name in ("q","k","v")]
    def before_projection(module,args):captured["output"]=args[0]
    handles.append(mixer.c_proj.register_forward_pre_hook(before_projection))
    try:mixer(x,None,None,None,None)
    finally:
        for h in handles:h.remove()
    actual=captured["output"]
    assert actual.dtype==torch.bfloat16
    leaves=[captured[n] for n in ("q","k","v")]
    oracle=[z.detach().double().reshape(2,c.sequence_len,c.heads,c.head_dim).requires_grad_() for z in leaves]
    q,k,v=oracle
    # Independent dense causal oracle, including raw projection feature maps.
    if kind=="elu_qknorm":
        q,k=(F.rms_norm(z,(c.head_dim,),eps=torch.finfo(torch.float32).eps)*ELU_QK_TEMPERATURE for z in (q,k))
    q,k=(torch.where(z>0,z+1,z.clamp_max(0).exp()).transpose(1,2) for z in (q,k))
    scores=(q@k.transpose(-1,-2)).tril()
    expected=(scores@v.transpose(1,2)).transpose(1,2)
    if kind=="elu_linear":expected=expected/(scores.sum(-1).transpose(1,2).unsqueeze(-1)+1e-6)
    else:
        expected=F.rms_norm(expected,(c.head_dim,),eps=torch.finfo(x.dtype).eps)
        expected=expected*mixer.output_gain.detach().to(x.dtype).double()
    expected=expected.flatten(2)
    w=torch.randn_like(actual)
    ga=torch.autograd.grad(actual,leaves,w)
    ge=torch.autograd.grad(expected,oracle,w.double())
    assert (actual.double()-expected).norm()/expected.norm()<.01
    for a,e in zip(ga,ge):
        assert (a.double()-e.reshape_as(a)).norm()/e.norm()<.03


@pytest.mark.parametrize("kind",["elu_norm","elu_qknorm"])
def test_elu_qknorm_removes_the_qk_projection_scale(kind):
    """A06'' pins phi's input scale; A06' changes its output when q/k projections grow (record 021)."""
    c=replace(debug_config(kind,sequence_len=65),backend="fla_chunk64")
    torch.manual_seed(0)
    mixer=LinearMixer(c,0)
    x=torch.randn(2,c.sequence_len,c.width)
    before=mixer(x,None,None,None,None)
    with torch.no_grad():
        mixer.c_q.weight.mul_(7.);mixer.c_k.weight.mul_(.3)
    after=mixer(x,None,None,None,None)
    if kind=="elu_qknorm":torch.testing.assert_close(after,before,rtol=1e-5,atol=1e-5)
    else:assert (after-before).abs().max()>1e-2


def test_elu_qknorm_features_have_the_registered_scale():
    c=replace(debug_config("elu_qknorm",sequence_len=65),backend="fla_chunk64")
    mixer=LinearMixer(c,0);seen=[]
    original=chunk_attention.__globals__["chunk_attention"]
    def capture(q,k,v,**kwargs):
        seen.append((q,k));return original(q,k,v,**kwargs)
    chunk_attention.__globals__["chunk_attention"]=capture
    try:mixer(torch.randn(2,c.sequence_len,c.width).bfloat16()*50,None,None,None,None)
    finally:chunk_attention.__globals__["chunk_attention"]=original
    (q,k),=seen
    assert q.dtype==k.dtype==torch.float32
    for phi in (q,k):
        # phi is monotone, so invert it to recover the per-token, per-head RMS.
        pre=torch.where(phi>1,phi-1,phi.log())
        torch.testing.assert_close(pre.pow(2).mean(-1).sqrt(),torch.full(phi.shape[:-1],ELU_QK_TEMPERATURE),rtol=1e-4,atol=0)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="IEEE-FP32 ELU+1 path is the CUDA backend")
def test_elu_cuda_backend_keeps_q_gradient_with_large_denominators():
    torch.backends.cuda.matmul.allow_tf32=True  # as the training runtime sets it
    ref=_elu_inputs("cuda",torch.float64,t=2048,scale=6.)
    got=[x.detach().float().requires_grad_(True) for x in ref]
    y64=chunk_attention(*ref,kind="elu_linear",chunk=64,implementation="torch")
    y32=chunk_attention(*got,kind="elu_linear",chunk=64)
    w=torch.randn_like(y64)
    g64=torch.autograd.grad((y64*w).sum(),ref);g32=torch.autograd.grad((y32.double()*w).sum(),got)
    rel=[((a.double()-e).norm()/e.norm()).item() for a,e in zip(g32,g64)]
    assert max(rel)<1e-3,rel
    assert torch.backends.cuda.matmul.allow_tf32  # the runtime setting is restored


def test_chunked_backend_is_registered_only_for_the_chunked_linear_methods():
    assert replace(debug_config("kda"),backend="fla_chunk64").backend=="fla_chunk64"
    assert candidate_config("elu_norm").backend=="fla_chunk64"
    assert candidate_config("elu_qknorm").backend=="fla_chunk64"
    assert candidate_config("gdn").backend=="fla_chunk64"
    for method in ("dense","gla","moba"):
        with pytest.raises(ValueError):replace(debug_config(method),backend="fla_chunk64")


@pytest.mark.skipif(not torch.cuda.is_available(), reason="DeltaNet FLA is CUDA-only")
@pytest.mark.parametrize("length", [65, 2048])
def test_deltanet_fla_preserves_causality_and_independent_rows(length):
    torch.manual_seed(42)
    q,k,v=[torch.randn(2,length,2,128,device="cuda",dtype=torch.float32) for _ in range(3)]
    q,k=F.normalize(q,dim=-1)*128**-.5,F.normalize(k,dim=-1)
    q,k,v=[x.bfloat16() for x in (q,k,v)]
    beta=torch.rand(2,length,2,device="cuda",dtype=torch.bfloat16)
    with torch.no_grad():
        original=chunk_attention(q,k,v,kind="deltanet",beta=beta,chunk=64)
        changed=[x.clone() for x in (q,k,v)]
        for x in changed:x[:,length//2:]*=-1
        future=chunk_attention(*changed,kind="deltanet",beta=beta,chunk=64)
        row=chunk_attention(q[:1],k[:1],v[:1],kind="deltanet",beta=beta[:1],chunk=64)
    assert torch.equal(future[:,:length//2],original[:,:length//2])
    assert torch.equal(row,original[:1])
