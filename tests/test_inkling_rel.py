"""REL indexing, causality, trained factors, shared shell and kernel gradients."""
from dataclasses import replace
import pytest
import torch
from nexttransformer.methods.inkling_rel import RelativeSelfAttention
from nexttransformer.methods.inkling_rel.reference import dense_oracle, relative_attention_reference
from nexttransformer.models.config import debug_config, candidate_config
from nexttransformer.models.lm import build_model
from nexttransformer.protocol.compute import audit_graph
from nexttransformer.training.optimizer import parameter_groups


@pytest.fixture(autouse=True)
def seed():
    torch.manual_seed(42)
    torch.set_num_threads(1)


def case(t=5, dtype=torch.float64, device='cpu', heads=2, dim=16):
    q,k,v=[torch.randn(2,t,heads,dim,dtype=dtype,device=device,requires_grad=True) for _ in range(3)]
    r=torch.randn(2,t,heads,4,dtype=dtype,device=device,requires_grad=True)
    table=(torch.randn(4,max(t,8),dtype=dtype,device=device)*.1).requires_grad_()
    return q,k,v,r,table


def test_causal_reference_and_all_five_gradients_match_independent_dense_oracle():
    inputs=case()
    actual=relative_attention_reference(*inputs)
    expected=dense_oracle(*inputs)
    torch.testing.assert_close(actual,expected,rtol=1e-12,atol=1e-12)
    grad=torch.randn_like(actual)
    a=torch.autograd.grad(actual,inputs,grad,retain_graph=True)
    b=torch.autograd.grad(expected,inputs,grad)
    for x,y in zip(a,b):torch.testing.assert_close(x,y,rtol=1e-11,atol=1e-11)
    assert torch.autograd.gradcheck(relative_attention_reference,inputs,fast_mode=True)


def test_backward_distance_includes_self_and_last_distance():
    q=k=torch.zeros(1,4,1,1,dtype=torch.float64)
    v=torch.arange(4,dtype=torch.float64).reshape(1,4,1,1)
    r=torch.ones(1,4,1,1,dtype=torch.float64)
    table=torch.tensor([[0.,1.,2.,3.]],dtype=torch.float64)
    actual=relative_attention_reference(q,k,v,r,table)
    for i in range(4):
        weights=torch.arange(i,-1,-1,dtype=torch.float64).softmax(0)
        expected=(weights*torch.arange(i+1,dtype=torch.float64)).sum()
        torch.testing.assert_close(actual[0,i,0,0],expected)
    table[0,-1]+=10
    changed=relative_attention_reference(q,k,v,r,table)
    torch.testing.assert_close(actual[:,:3],changed[:,:3],rtol=0,atol=0)
    assert changed[0,-1,0,0]<actual[0,-1,0,0]


def test_future_tokens_do_not_change_prefix_and_bias_is_content_dependent():
    inputs=case(t=7)
    baseline=relative_attention_reference(*inputs)
    changed=[x.detach().clone() for x in inputs]
    for x in changed[:4]:x[:,4:]+=10
    future=relative_attention_reference(*changed)
    torch.testing.assert_close(baseline[:,:4],future[:,:4],rtol=0,atol=0)
    changed=[x.detach().clone() for x in inputs];changed[3][:,3]*=-5
    content=relative_attention_reference(*changed)
    assert not torch.allclose(baseline[:,3],content[:,3])
    torch.testing.assert_close(baseline[:,[0,1,2,4,5,6]],content[:,[0,1,2,4,5,6]],rtol=0,atol=0)


def test_zero_bias_is_dense_causal_attention_and_rope_is_unused():
    q,k,v,r,table=case();table=torch.zeros_like(table)
    expected=torch.nn.functional.scaled_dot_product_attention(q.transpose(1,2),k.transpose(1,2),v.transpose(1,2),is_causal=True).transpose(1,2)
    torch.testing.assert_close(relative_attention_reference(q,k,v,r,table),expected,rtol=1e-12,atol=1e-12)
    c=debug_config('inkling_rel');attn=RelativeSelfAttention(c,0).double()
    x=torch.randn(2,7,c.width,dtype=torch.float64)
    a=attn(x,None,None,(-1,0),None)
    b=attn(x,None,(torch.full((1,),float('nan')),)*2,(-1,0),None)
    torch.testing.assert_close(a,b,rtol=0,atol=0)


def test_full_2k_coverage_and_invalid_options():
    c=candidate_config('inkling_rel')
    assert c.option('d_rel')==16 and c.option('rel_extent')==c.sequence_len==2048
    with pytest.raises(ValueError,match='cover'):
        replace(c,options=dict(c.options,rel_extent=2047))
    with pytest.raises(ValueError,match='full causal'):
        replace(c,window_pattern='SSSL')


def test_model_gradients_initialization_optimizer_and_checkpoint(tmp_path):
    c=debug_config('inkling_rel');model=build_model(c,'cpu',seed=42)
    for block in model.transformer.h:
        attn=block.attn
        assert attn.rel_logits_proj.proj.shape==(c.option('d_rel'),c.option('rel_extent'))
        assert attn.c_r.weight.abs().sum()>0 and attn.rel_logits_proj.proj.abs().sum()>0
        with torch.no_grad():attn.c_proj.weight.normal_(std=.03)
    tokens=torch.randint(c.vocab_size,(2,c.sequence_len))
    model(tokens,tokens).backward()
    assert all(p.grad is not None for p in model.parameters() if p.requires_grad)
    assert all(torch.isfinite(p.grad).all() for p in model.parameters() if p.requires_grad)
    for block in model.transformer.h:
        assert block.attn.c_r.weight.grad.abs().sum()>0
        assert block.attn.rel_logits_proj.proj.grad.abs().sum()>0
    # Added ordinary matrices follow the shared Muon rule; no special optimizer override.
    groups=parameter_groups(model)
    owners={name:g['kind'] for g in groups for name in g['parameter_names']}
    assert all(kind=='muon' for name,kind in owners.items() if '.c_r.' in name or '.rel_logits_proj.' in name)
    path=tmp_path/'rel.pt';torch.save(model.state_dict(),path)
    restored=build_model(c,'cpu',seed=17);restored.load_state_dict(torch.load(path,weights_only=True))
    torch.testing.assert_close(model(tokens),restored(tokens),rtol=0,atol=0)


def test_cost_covers_rel_projection_and_all_trainable_parameters():
    c=debug_config('inkling_rel');cost=audit_graph(c)
    baseline=audit_graph(replace(c,method='dense_full',options=()))
    assert not cost['unknown']
    extra=c.layers*(c.width*c.heads*c.option('d_rel')+c.option('d_rel')*c.option('rel_extent'))
    assert cost['parameters']-baseline['parameters']==extra
    assert cost['phases']['optimizer']>baseline['phases']['optimizer']


@pytest.mark.skipif(not torch.cuda.is_available(),reason='CUDA is required')
@pytest.mark.parametrize('length',[1,17,65,257,2048])
def test_cuda_bf16_forward_and_all_gradients(length):
    from nexttransformer.methods.inkling_rel.kernel import relative_attention
    inputs=case(length,dtype=torch.bfloat16,device='cuda',dim=128)
    refs=[a.detach().float().requires_grad_() for a in inputs]
    actual=relative_attention(*inputs);expected=dense_oracle(*refs)
    grad=torch.randn_like(actual)
    a=torch.autograd.grad(actual,inputs,grad);b=torch.autograd.grad(expected,refs,grad.float())
    for value,ref in zip((actual,*a),(expected,*b)):
        assert torch.isfinite(value).all()
        relative=(value.float()-ref).norm()/ref.norm().clamp_min(1e-7)
        absolute=(value.float()-ref).abs().max()
        assert relative<.012 or absolute<.006


@pytest.mark.skipif(not torch.cuda.is_available(),reason='CUDA is required')
def test_cuda_noncontiguous_inputs_and_unused_distance_gradients():
    from nexttransformer.methods.inkling_rel.kernel import relative_attention
    q,k,v,r,table=case(17,dtype=torch.float32,device='cuda',dim=32)
    q,k,v=[a.transpose(1,2).contiguous().transpose(1,2).detach().requires_grad_() for a in (q,k,v)]
    table=torch.randn(4,32,device='cuda',requires_grad=True)
    out=relative_attention(q,k,v,r,table)
    expected=dense_oracle(q,k,v,r,table)
    torch.testing.assert_close(out,expected,atol=2e-5,rtol=2e-5)
    out.sum().backward()
    torch.testing.assert_close(table.grad[:,17:],torch.zeros_like(table.grad[:,17:]),rtol=0,atol=0)
    for grad in (q.grad,k.grad,r.grad):assert torch.isfinite(grad).all()
