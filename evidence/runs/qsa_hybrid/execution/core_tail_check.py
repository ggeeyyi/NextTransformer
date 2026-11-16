"""Targeted H05 final-tail resource check; never a full CORE result."""
import os,sys,json,random,time,hashlib
from pathlib import Path
for name in ('TRITON_CACHE_DIR','TORCHINDUCTOR_CACHE_DIR'):
 if os.environ.get(name):os.environ[name]=os.path.join(os.environ[name],'rank'+os.environ['LOCAL_RANK'])
import torch,yaml
import torch.distributed as dist
from nexttransformer.runtime.site import bootstrap_runtime
bootstrap_runtime()
from nexttransformer.runtime.process import initialize
from nexttransformer.models.lm import build_model
from nexttransformer.protocol.admission import admit_config
from nexttransformer.evaluation.formal import EvalRows
from nanochat.tokenizer import RustBPETokenizer
from nanochat import flash_attention
from timing import install
import core_graph
r=Path(__file__).resolve().parents[1];method,label=sys.argv[1:]
assert method=='qsa_hybrid'
rank,world,device=initialize(42);assert world==8 and not flash_attention.USE_FA3
torch.backends.cuda.enable_flash_sdp(False);torch.use_deterministic_algorithms(True)
install()
config,_=admit_config(r/'configs/hybrid'/f'{method}.json')
model=build_model(config,device,seed=42)
cp=max((r/'formal'/method/'training').glob('checkpoint_*/complete.json'),key=lambda p:p.parent.name).parent
saved=torch.load(cp/'model.pt',map_location='cpu',weights_only=False,mmap=True)
model.load_state_dict(saved['model'],strict=True);model.set_step(saved['state']['step']);model.set_sparse_backend('reference');model.eval();del saved
tokenizer=RustBPETokenizer.from_directory(r/'resources/tokenizer')
task=next(t for t in yaml.safe_load((r/'resources/eval_bundle/core.yaml').read_text())['icl_tasks'] if t['label']=='bigbench_language_identification')
meta=dict(task_type=task['icl_task_type'],dataset_uri=task['dataset_uri'],num_fewshot=task['num_fewshot'][0],continuation_delimiter=task.get('continuation_delimiter',' '))
data=[json.loads(l) for l in (r/'resources/eval_bundle/eval_data'/meta['dataset_uri']).read_text().splitlines()]
random.Random(1337).shuffle(data)
original_inputs=core_graph.inputs
def tail_inputs(*args):
 saved=original_inputs(*args);groups={}
 for e in saved['entries']:groups.setdefault(tuple(e['input_ids'].shape),[]).append(e)
 ordered=sorted(groups,key=lambda shape:(-len(groups[shape]),shape));chosen=set(ordered[-64:])
 entries=[e for e in saved['entries'] if tuple(e['input_ids'].shape) in chosen]
 assert len(chosen)==64
 (r/'evidence'/f'{label}-inputs-{rank}.json').write_text(json.dumps(dict(task=task['label'],rank=rank,shapes=sorted(chosen),indices=[e['idx'] for e in entries],scope='targeted final64 shapes only; not a benchmark result')))
 return dict(saved,entries=entries)
core_graph.inputs=tail_inputs
started=time.time();core_graph.evaluate_task(EvalRows(model),tokenizer,data,device,meta)
if rank==0:
 out=dict(status='passed',scope='targeted final64 shape groups per rank with staged synchronization; not full CORE',full_core=False,method=method,checkpoint=str(cp),optimizer_updates=0,seconds=time.time()-started,task=task['label'],shapes_per_rank=64)
 (r/'evidence'/f'{label}.json').write_text(json.dumps(out,indent=2));print(json.dumps(out),flush=True)
dist.destroy_process_group()

