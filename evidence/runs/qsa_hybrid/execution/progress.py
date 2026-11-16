"""Report actual TorchDynamo code loading and the standalone DDP final checksum."""
import json
import os
from pathlib import Path
import sys
import time
import torch

root=Path(__file__).resolve().parents[1]
token=os.environ['T2M_H1_GPU_LEASE']
rank=os.environ.get('LOCAL_RANK','0')
out=root/'evidence'/f'work-phase-{token}-{rank}.json'
state={'rank':rank,'compiling':False,'final_checksum':False,'exit_checkpoint':False}
def write():
    state['at']=time.time()
    p=out.with_suffix('.tmp');p.write_text(json.dumps(state));p.replace(out)
    with out.with_suffix('.jsonl').open('a') as stream:stream.write(json.dumps(state)+'\n')
def entered(args):
    state.update(compiling=True,compile_id=str(args.compile_id),compile_started=time.time());write()
def left(args):
    state.update(compiling=False,compile_ended=time.time());write()
def install(digest=None):
    torch._dynamo.on_compile_start(entered)
    torch._dynamo.on_compile_end(left)
    write()
    if digest is None:
        from nexttransformer.runtime.checkpoint import save_checkpoint
        target={'split':1,'continuous':12,'resume':12,'training':46875}.get(os.environ.get('NEXTT_STAGE'))
        if target is None:return
        tool=next(i for i in range(6) if sys.monitoring.get_tool(i) is None)
        sys.monitoring.use_tool_id(tool,'NextTransformer checkpoint progress')
        def checkpoint(code,offset):
            frame=sys._getframe(1)
            if frame.f_code is code:
                step=frame.f_locals.get('state',{}).get('step')
                if step==target:
                    state.update(exit_checkpoint=True,checkpoint_step=step)
                    write()
        sys.monitoring.register_callback(tool,sys.monitoring.events.PY_START,checkpoint)
        sys.monitoring.set_local_events(tool,save_checkpoint.__code__,sys.monitoring.events.PY_START)
        return
    starts=0
    def on_digest(code,offset):
        nonlocal starts
        starts+=1
        state.update(checksum_number=starts,final_checksum=starts==2)
        write()
    tool=next(i for i in range(6) if sys.monitoring.get_tool(i) is None)
    sys.monitoring.use_tool_id(tool,'NextTransformer checksum progress')
    sys.monitoring.register_callback(tool,sys.monitoring.events.PY_START,on_digest)
    sys.monitoring.set_local_events(tool,digest.__code__,sys.monitoring.events.PY_START)
