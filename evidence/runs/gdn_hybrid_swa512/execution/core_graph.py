"""Original-shape CUDA graphs for CORE; no changes to tokens or scoring."""
import os,json,time
from collections import defaultdict
from pathlib import Path
from contextlib import contextmanager
import inspect,textwrap
import torch
import torch.distributed as dist
from nanochat import core_eval
from core_inputs import inputs,score
r=Path(__file__).resolve().parents[1]


@contextmanager
def capture_safe_stats():
 from nexttransformer.models.sparse import SparseMixer
 original=SparseMixer.forward
 source=textwrap.dedent(inspect.getsource(original))
 assert source.count('int(valid.sum())')==1 and source.count('int(lv.sum())')==1
 source=source.replace('int(valid.sum())','valid.sum()').replace('int(lv.sum())','lv.sum()')
 namespace={};exec(compile(source,'<CORE deferred diagnostic edge counts>','exec'),original.__globals__,namespace)
 SparseMixer.forward=namespace['forward']
 try:yield
 finally:SparseMixer.forward=original

def score_tensor(e,losses,pred,ids):
 if e['task_type']=='language_modeling':
  a,b=e['start_idxs'][0],e['end_idxs'][0]
  return torch.all(pred[0,a-1:b-1]==ids[0,a:b]).float()
 means=torch.stack([losses[i,a-1:b-1].mean() for i,(a,b) in enumerate(zip(e['start_idxs'],e['end_idxs']))])
 # Python min keeps a first NaN, but ignores a later NaN; argmin alone does not.
 choice=torch.where(torch.isnan(means[0]),torch.zeros((),dtype=torch.long,device=means.device),torch.where(torch.isnan(means),torch.full_like(means,float('inf')),means).argmin())
 return (choice==e['gold']).float()

@torch.no_grad()
def evaluate_task(model,tokenizer,data,device,task_meta):
 assert model.config.method in ('gdn_hybrid_swa','qsa_hybrid')
 rank=dist.get_rank();saved=inputs(task_meta,data,rank)
 byshape=defaultdict(list)
 for e in saved['entries']:
  e['input_ids']=e['input_ids'].pin_memory();byshape[tuple(e['input_ids'].shape)].append(e)
 correct=torch.zeros(len(data),dtype=torch.float32,device=device)
 begin=time.monotonic();count=0;verified=0
 for shape,entries in sorted(byshape.items(),key=lambda x:(-len(x[1]),x[0])):
  first=entries[0];ids=first['input_ids'].to(device)
  stream=torch.cuda.Stream(device=device);stream.wait_stream(torch.cuda.current_stream())
  with torch.cuda.stream(stream):
   baseline=core_eval.forward_model(model,ids)
   expected=(baseline[0].clone(),baseline[1].clone())
  torch.cuda.current_stream().wait_stream(stream)
  g=torch.cuda.CUDAGraph()
  with capture_safe_stats(), torch.cuda.graph(g,stream=stream,capture_error_mode='relaxed'):
   losses,pred=core_eval.forward_model(model,ids)
  # First replay validates the entire original forward, including all unscored positions.
  replay_start=time.monotonic();g.replay();torch.cuda.synchronize(device);replay_seconds=time.monotonic()-replay_start
  if not (torch.equal(losses[:,:-1],expected[0][:,:-1]) and torch.equal(pred,expected[1])):
   diff=(losses[:,:-1]-expected[0][:,:-1]).abs()
   raise RuntimeError(f'CUDA graph output not bitwise equal: shape={shape}, max_loss_delta={diff.max().item()}, predictions={(pred!=expected[1]).sum().item()}')
  verified+=1;del baseline,expected
  queue_limit=max(1,min(32,int(.04/max(replay_seconds,.0001))))
  assert bool(score_tensor(first,losses,pred,ids).item())==score(first,losses,pred,ids)
  mark=torch.cuda.Event(enable_timing=True);mark.record()
  for j,e in enumerate(entries):
   if j:ids.copy_(e['input_ids'],non_blocking=True);g.replay()
   correct[e['idx']]=score_tensor(e,losses,pred,ids);count+=1
   if (j+1)%queue_limit==0 or j+1==len(entries):
    end=torch.cuda.Event(enable_timing=True);end.record();end.synchronize()
    sample=json.loads((r/'gpu-fast.json').read_text())['util'][rank]
    if sample>=70:
     duty=.70 if sample>88 else .86
     time.sleep(min(.05,mark.elapsed_time(end)/1000*(1/duty-1)))
    mark.record()
  del g,losses,pred,ids,stream
  p=r/'evidence'/f'core-graph-progress-{os.environ["T2M_H1_GPU_LEASE"]}-{rank}.json';tmp=p.with_suffix('.tmp');tmp.write_text(json.dumps(dict(at=time.time(),label=saved['label'],examples=count,total=len(saved['entries']),shapes_verified=verified,shape_count=len(byshape),seconds=time.monotonic()-begin)));tmp.replace(p)
 dist.barrier();dist.all_reduce(correct,op=dist.ReduceOp.SUM)
 return correct.mean().item()

def install():
 import nanochat.base_eval as base
 base.evaluate_task=evaluate_task
