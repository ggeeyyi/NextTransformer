import copy
from dataclasses import replace
import json
import math
from pathlib import Path

import pytest
import torch
import pyarrow as pa
import pyarrow.parquet as pq

from nexttransformer.models.backbone import GPT, GPTConfig
from nexttransformer.models.native.config import NativeModelConfig, NativeTrainConfig, load_config
from nexttransformer.models.native.model import NativeExperiment, build_model, theoretical_training_flops
from nexttransformer.runtime.data import BestFitStream
from nexttransformer.native.optimizer import build_optimizer, set_schedule
from nexttransformer.models.native.budget import SurvivalBudget
from nexttransformer.runtime.checkpoint import save_checkpoint, load_checkpoint


def tiny(architecture='dense',**kw):
    return NativeModelConfig(architecture=architecture,vocab_size=128,width=32,heads=2,mlp_width=128,
        sequence_len=16,dense_layers=6,pre_layers=1,core_layers=2,post_layers=1,
        fourier_dim=16,circle_max_steps=6,circle_target_steps=4.,circle_random_warmup=0,
        circle_route_transition=0,**kw)


def test_dense_is_upstream_forward_gradients_and_parameter_count():
    c=tiny();a=build_model(c,torch.device('cpu'))
    b=GPT(GPTConfig(sequence_len=16,vocab_size=128,n_layer=6,n_head=2,n_kv_head=2,n_embd=32))
    b.init_weights();b.load_state_dict(a.state_dict(),strict=True)
    x=torch.randint(128,(2,16));y=x.roll(-1,1)
    torch.testing.assert_close(a(x),b(x),rtol=0,atol=0)
    a(x,y)['loss'].backward();b(x,y).backward()
    for p,q in zip(a.parameters(),b.parameters()):torch.testing.assert_close(p.grad,q.grad,atol=0,rtol=0)
    with torch.device('meta'): full=NativeExperiment(NativeModelConfig())
    assert sum(p.numel() for p in full.parameters())==1384122122
    assert theoretical_training_flops(full.config)==pytest.approx(GPT.estimate_flops(full))


@pytest.mark.parametrize('architecture',['dense','mor','loopformer','circle'])
def test_causality_batch_independence_and_train_eval_route_parity(architecture):
    torch.manual_seed(42)
    c=tiny(architecture);m=build_model(c,torch.device('cpu'))
    # Nonzero residual outputs expose attention/selection bugs hidden by native init.
    for block in m.transformer.h:
        torch.nn.init.normal_(block.attn.c_proj.weight,std=.02)
        torch.nn.init.normal_(block.mlp.c_proj.weight,std=.02)
    if architecture=='circle':
        m.halter.depth_bias.data.fill_(.1)
        m.halter.delta_scale.data.fill_(.03)
    x=torch.randint(128,(2,16));changed=x.clone();changed[:,9:]=torch.randint(128,(2,7))
    m.train();train=m(x);counts=m.last_stats['active_tokens']
    m.eval();evaluation=m(x)
    torch.testing.assert_close(train,evaluation,rtol=1e-5,atol=1e-6)
    assert counts==m.last_stats['active_tokens']
    torch.testing.assert_close(evaluation[:,:9],m(changed)[:,:9],rtol=1e-5,atol=1e-6)
    torch.testing.assert_close(evaluation[:1],m(x[:1]),rtol=1e-5,atol=1e-6)
    m.train();out=m(x,x.roll(-1,1));out['loss'].backward()
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in m.parameters())
    optimizer=build_optimizer(m,NativeTrainConfig())
    set_schedule(optimizer,0,NativeTrainConfig());optimizer.step()


def test_mor_no_reentry_budget_gradients_and_global_update():
    c=tiny('mor');m=build_model(c,torch.device('cpu'))
    for r in m.routers:r.weight.data.zero_();r.bias.data.fill_(2.)
    m.routers[0].bias.data.fill_(-2.)
    x=torch.randint(128,(2,16));out=m(x,x)
    assert m.last_stats['active_tokens']==[32,0,0]
    out['router_loss'].backward()
    # Underspending must encourage the first closed gate to open.
    assert m.routers[0].bias.grad.item()<0
    old=m.survival_budget.dual.clone()
    m.update_route_budget([32,0,0],32,0)
    assert (m.survival_budget.dual<old).all()
    with pytest.raises(ValueError):m.update_route_budget([32,0,0],32,0)
    m.eval();old=copy.deepcopy(m.state_dict());m(x)
    for key,value in old.items():assert torch.equal(value,m.state_dict()[key])


def test_full_configuration_token_and_flop_budgets():
    tc=NativeTrainConfig()
    assert tc.steps==46875 and len(tc.validation_steps())==189
    assert tc.global_batch_tokens==1048576
    for name in ('dense','mor','loopformer','circle'):
        _, configured = load_config(Path(__file__).resolve().parents[1]/'configs/native'/f'{name}.json')
        assert configured.core_max_per_task == -1
        assert configured.core_steps() == [8000,16000,24000,32000,40000,46875]
        assert configured.core_steps(include_final=False) == [8000,16000,24000,32000,40000]
    assert replace(tc, core_every=0).core_steps() == [46875]
    values=[]
    for name in ['dense','mor','loopformer','circle']:
        c=NativeModelConfig(architecture=name)
        assert [tc.accumulation(c,8,b) for b in [1,2,4,8,16,32,64]]==[64,32,16,8,4,2,1]
        values.append(theoretical_training_flops(c))
        with torch.device('meta'):model=NativeExperiment(c)
        assert model.estimate_flops()==theoretical_training_flops(c)
    assert max(values)/min(values)<1.001
    with pytest.raises(ValueError,match='per-block exits'):
        NativeModelConfig(architecture='circle',circle_exit_interval=2)
    with pytest.raises(ValueError,match='mean_survival'):
        NativeModelConfig(architecture='circle',circle_budget_mode='cost')


class TinyTokenizer:
    def get_vocab_size(self):return 128
    def get_bos_token_id(self):return 127
    def encode(self, texts, prepend=None, num_threads=4):
        return [[prepend]+[ord(x)%100 for x in s] for s in texts]


def prepare_data(tmp_path):
    docs=['abcde'*(1+i%17) for i in range(150)]
    for name in ['shard_00000.parquet','shard_06542.parquet']:
        pq.write_table(pa.table({'text':docs}),tmp_path/name,row_group_size=25)


@pytest.mark.parametrize('steps,options,expected', [
    (6,[],[2,4,6]),  # final step is also an interval boundary: evaluate only once
    (5,[],[2,4,5]),
    (6,['--skip-final-core'],[2,4]),
    (6,['--smoke-steps','3'],[]),
])
def test_training_core_schedule_has_one_full_final(tmp_path,monkeypatch,steps,options,expected):
    """Exercise the training entry, with real tiny CPU updates and stubbed evals."""
    from contextlib import nullcontext
    from types import SimpleNamespace
    import sys
    import nexttransformer.native.train as train
    prepare_data(tmp_path)
    tc=NativeTrainConfig(total_tokens=32*steps,global_batch_tokens=32,val_tokens=32,
                        final_bpb_tokens=32,val_every=2,core_every=2,checkpoint_every=3)
    mc=tiny(); output=tmp_path/'run'; events=[]; calls=[]
    monkeypatch.setattr(train,'load_config',lambda path:(mc,tc))
    monkeypatch.setattr(train,'initialize',lambda seed:(0,1,torch.device('cpu')))
    monkeypatch.setattr(train,'verify_resources',lambda *a:({'tokenizer_sha256':'test'},'manifest'))
    monkeypatch.setattr(train.RustBPETokenizer,'from_directory',lambda path:TinyTokenizer())
    monkeypatch.setattr(train,'start_tracker',lambda *a:nullcontext(SimpleNamespace(log=events.append)))
    def bpb(model,tokenizer,data_dir,resources,tokens,batch,rank,world,device,split='val'):
        return {'val_bpb' if split=='val' else 'train_bpb':1.,'forward_flops_estimate':0.,'seconds':0.}
    def core(model,tokenizer,resources,device,max_per_task=-1,*,core_max_tokens=None):
        calls.append(max_per_task)
        return {'core_metric':.1,'full_core':max_per_task<=0,'entries':22,'seconds':0.}
    monkeypatch.setattr(train,'evaluate_bpb',bpb)
    monkeypatch.setattr(train,'evaluate_core',core)
    monkeypatch.setattr(sys,'argv',['native_train','--config','unused.json','--runtime-dir',str(tmp_path/'runtime'),
        '--resources',str(tmp_path),'--data-dir',str(tmp_path),'--output',str(output),'--micro-batch','2',
        '--wandb-mode','disabled',*options])
    train.main()
    assert [e['step'] for e in events if e['event']=='core']==expected
    assert calls==[-1]*len(expected)
    assert sorted(int(p.stem.split('_')[1]) for p in output.glob('core_[0-9]*.json'))==[s for s in expected if s<steps]
    assert (output/'core_final.json').exists()==(steps in expected)


def test_bestfit_matches_upstream_and_exact_resume_across_microbatch(tmp_path,monkeypatch):
    prepare_data(tmp_path)
    import nanochat.dataloader as upstream
    monkeypatch.setattr(upstream,'list_parquet_files',lambda **kw:sorted(str(p) for p in tmp_path.glob('*.parquet')))
    monkeypatch.setattr(upstream,'get_dist_info',lambda:(True,1,1,2))
    tok=TinyTokenizer()
    a=BestFitStream(tmp_path,tok,'train',1,2,16,buffer_size=10,tokenizer_batch_size=7)
    reference=upstream.tokenizing_distributed_data_loader_bos_bestfit(tok,3,16,'train',device='cpu',buffer_size=10,tokenizer_batch_size=7)
    for _ in range(4):
        actual=a.batch(3,torch.device('cpu'));expected=next(reference)
        for x,y in zip(actual,expected):assert torch.equal(x,y)
    state=a.state_dict()
    expected=a.batch(6,torch.device('cpu'))
    b=BestFitStream(tmp_path,tok,'train',1,2,16,buffer_size=10,tokenizer_batch_size=7);b.load_state_dict(state)
    batches=[b.batch(2,torch.device('cpu')) for _ in range(3)]
    for i in range(2):assert torch.equal(expected[i],torch.cat([batch[i] for batch in batches]))


@pytest.mark.parametrize('architecture',['dense','mor','loopformer','circle'])
def test_checkpoint_restores_router_budget_rng_optimizer_and_data(tmp_path,architecture):
    prepare_data(tmp_path)
    c=tiny(architecture)
    if architecture=='circle':c=replace(c,circle_random_warmup=5)
    tc=NativeTrainConfig();m=build_model(c,torch.device('cpu'));opt=build_optimizer(m,tc)
    stream=BestFitStream(tmp_path,TinyTokenizer(),'train',sequence_len=16,buffer_size=10)
    def update(model,optimizer,data,step):
        model.set_circle_step(step)
        x,y=data.batch(2,torch.device('cpu'));optimizer.zero_grad(set_to_none=True)
        model(x,y)['loss'].backward();optimizer.step()
        model.update_route_budget(model.last_stats['active_tokens'],x.numel(),step)
    update(m,opt,stream,0)
    saved=save_checkpoint(tmp_path/'run',m,opt,stream,c,tc,{'step':1},0,1)
    update(m,opt,stream,1)
    n=build_model(c,torch.device('cpu'));other=build_optimizer(n,tc)
    data=BestFitStream(tmp_path,TinyTokenizer(),'train',sequence_len=16,buffer_size=10)
    load_checkpoint(saved,n,other,data)
    update(n,other,data,1)
    for key,value in m.state_dict().items():torch.testing.assert_close(value,n.state_dict()[key],rtol=0,atol=0)
    assert stream.state_dict()==data.state_dict()


def test_bpb_is_upstream_on_identical_packed_rows(tmp_path,monkeypatch):
    prepare_data(tmp_path)
    import nanochat.dataloader as upstream
    from nanochat.loss_eval import evaluate_bpb as reference_bpb
    from nexttransformer.evaluation.native import evaluate_bpb
    monkeypatch.setattr(upstream,'list_parquet_files',lambda **kw:sorted(str(p) for p in tmp_path.glob('*.parquet')))
    monkeypatch.setattr(upstream,'get_dist_info',lambda:(False,0,0,1))
    tok=TinyTokenizer();device=torch.device('cpu');model=build_model(tiny(),device).eval()
    (tmp_path/'tokenizer').mkdir();table=torch.ones(128,dtype=torch.int32);table[-1]=0
    torch.save(table,tmp_path/'tokenizer/token_bytes.pt')
    batches=upstream.tokenizing_distributed_data_loader_bos_bestfit(tok,2,16,'val',device='cpu')
    expected=reference_bpb(model,batches,3,table)
    result=evaluate_bpb(model,tok,tmp_path,tmp_path,96,2,0,1,device)
    assert result['val_bpb']==expected
    assert result['input_tokens']==96


def test_native_optimizer_groups_match_upstream_and_first_update():
    c=tiny();tc=NativeTrainConfig();a=build_model(c,torch.device('cpu'))
    b=GPT(GPTConfig(sequence_len=16,vocab_size=128,n_layer=6,n_head=2,n_kv_head=2,n_embd=32))
    b.init_weights();b.load_state_dict(a.state_dict())
    scale=math.sqrt(tc.global_batch_tokens/524288)
    from nexttransformer.native.optimizer import D12_SCALING_PARAMS,D24_SCALING_PARAMS
    reference=b.setup_optimizer(unembedding_lr=tc.unembedding_lr*scale,embedding_lr=tc.embedding_lr*scale,
        scalar_lr=tc.scalar_lr*scale,matrix_lr=tc.muon_lr*scale,
        weight_decay=tc.weight_decay*scale*D12_SCALING_PARAMS/D24_SCALING_PARAMS)
    actual=build_optimizer(a,tc)
    for left,right in zip(actual.param_groups,reference.param_groups):
        for key in ('kind','lr','betas','eps','weight_decay','momentum','ns_steps','beta2'):
            assert left.get(key)==right.get(key)
        assert [p.shape for p in left['params']]==[p.shape for p in right['params']]
    assert len(actual.param_groups)==len(reference.param_groups)
    x=torch.randint(128,(2,16));y=x.roll(-1,1)
    a(x,y)['loss'].backward();b(x,y).backward();actual.step();reference.step()
    for left,right in zip(a.parameters(),b.parameters()):torch.testing.assert_close(left,right,atol=0,rtol=0)


def test_native_mor_budget_is_visible_in_default_tracking():
    from nexttransformer.runtime.tracking import metric_payload
    event={'event':'train','step':1,'budget_survival':[.64,.32],
           'budget_error':[0,0],'budget_dual':[0,0],'mean_recursions':1.96,'budget_max_abs_error':0.}
    values=metric_payload(event,tiny('mor'))
    assert values['train/budget_max_abs_error']==0.
    assert not any('budget_survival' in key or 'budget_dual' in key for key in values)
    assert values['train/mean_recursions']==1.96


def test_native_resource_guard_rejects_substituted_data(tmp_path):
    from nexttransformer.runtime.data import verify_resources,data_inventory,inventory_fingerprint,tokenizer_fingerprint
    from nexttransformer.runtime.files import DATA_REVISION,sha256
    data=tmp_path/'base_data_climbmix';data.mkdir();prepare_data(data)
    (tmp_path/'tokenizer').mkdir()
    for name in ('tokenizer.pkl','token_bytes.pt'):(tmp_path/'tokenizer'/name).write_bytes(b'test')
    record={'data_revision':DATA_REVISION,'tokenizer_sha256':tokenizer_fingerprint(tmp_path),
            'data_inventory_sha256':inventory_fingerprint(data_inventory(data)),
            'validation_sha256':sha256(data/'shard_06542.parquet'),'upstream_sources':{}}
    (tmp_path/'native_manifest.json').write_text(json.dumps(record))
    verify_resources(tmp_path)
    with (data/'shard_00000.parquet').open('ab') as f:f.write(b'changed')
    with pytest.raises(ValueError,match='inventory'):verify_resources(tmp_path)
