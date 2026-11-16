import os,sys,json
from pathlib import Path
for name in ('TRITON_CACHE_DIR','TORCHINDUCTOR_CACHE_DIR'):
 if os.environ.get(name):os.environ[name]=os.path.join(os.environ[name],'rank'+os.environ['LOCAL_RANK'])
import torch
import torch.distributed as dist
from nexttransformer.runtime.site import bootstrap_runtime
bootstrap_runtime()
from nexttransformer.runtime.process import initialize
from nexttransformer.models.lm import build_model
from nexttransformer.protocol.admission import admit_config
from nexttransformer.evaluation.formal import evaluate_core
from nanochat.tokenizer import RustBPETokenizer
from nanochat import flash_attention
from timing import install
import core_graph
r=Path(__file__).resolve().parents[1];method,label=sys.argv[1:]
rank,world,device=initialize(42);assert world==8 and not flash_attention.USE_FA3
torch.backends.cuda.enable_flash_sdp(False);torch.use_deterministic_algorithms(True)
install()
config,_=admit_config(r/'configs/hybrid'/f'{method}.json')
model=build_model(config,device,seed=42)
cp=max((r/'formal'/method/'training').glob('checkpoint_*/complete.json'),key=lambda p:p.parent.name).parent
saved=torch.load(cp/'model.pt',map_location='cpu',weights_only=False,mmap=True)
model.load_state_dict(saved['model'],strict=True);model.set_step(saved['state']['step']);model.set_sparse_backend('reference');model.eval();del saved
tokenizer=RustBPETokenizer.from_directory(r/'resources/tokenizer');core_graph.install()
result=evaluate_core(model,tokenizer,r/'resources',device)
if rank==0:(r/'evidence'/f'{label}.json').write_text(json.dumps(dict(status='passed',method=method,checkpoint=str(cp),optimizer_updates=0,result=result),indent=2))
dist.destroy_process_group()
