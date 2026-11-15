"""Cache exact CPU inputs produced by the untouched official evaluate_example."""
import os,sys,json,random,hashlib,time
from pathlib import Path
os.environ['CUDA_VISIBLE_DEVICES']=''
import torch,yaml
from nexttransformer.runtime.site import bootstrap_runtime
bootstrap_runtime()
from nanochat import core_eval
from nanochat.tokenizer import RustBPETokenizer
r=Path(__file__).resolve().parents[1]; rank=int(sys.argv[1]); dest=r/'core-input-cache';dest.mkdir(exist_ok=True)
tokenizer=RustBPETokenizer.from_directory(r/'resources/tokenizer')
class Captured(Exception):pass
class Shell:max_seq_len=2048
current={}
def capture(model,input_ids):
 f=sys._getframe(1);d=f.f_locals
 current.update(input_ids=input_ids.clone(),start_idxs=d['start_idxs'],end_idxs=d['end_idxs'],gold=d['item'].get('gold'),task_type=d['task_type'],idx=d['idx'])
 raise Captured()
core_eval.forward_model=capture
config=yaml.safe_load((r/'resources/eval_bundle/core.yaml').read_text())
for task in config['icl_tasks']:
 meta=dict(task_type=task['icl_task_type'],dataset_uri=task['dataset_uri'],num_fewshot=task['num_fewshot'][0],continuation_delimiter=task.get('continuation_delimiter',' '))
 key=hashlib.sha256(json.dumps(meta,sort_keys=True).encode()).hexdigest()
 out=dest/f'{key}-rank{rank}.pt'
 if out.exists():continue
 source=r/'resources/eval_bundle/eval_data'/meta['dataset_uri'];raw=source.read_bytes()
 data=[json.loads(l.strip()) for l in raw.decode().splitlines()];random.Random(1337).shuffle(data)
 entries=[]
 for idx in range(rank,len(data),8):
  current={}
  try:core_eval.evaluate_example(idx,Shell(),tokenizer,data,'cpu',meta)
  except Captured:entries.append(current)
  else:raise AssertionError('Official input capture did not run')
 record=dict(label=task['label'],task_meta=meta,data_sha=hashlib.sha256(raw).hexdigest(),count=len(data),rank=rank,world=8,entries=entries,official_source_sha=hashlib.sha256(Path(core_eval.__file__).read_bytes()).hexdigest())
 tmp=out.with_suffix('.tmp');torch.save(record,tmp);tmp.replace(out)
 print(json.dumps(dict(at=time.time(),rank=rank,label=task['label'],examples=len(entries),rows=sum(e['input_ids'].shape[0] for e in entries))),flush=True)
(dest/f'rank{rank}.complete').write_text('22 official tasks prepared; no CUDA execution\n')
