"""Bounded 8-GPU native smoke, fresh-process resume and all-task CORE acceptance.

Source experiment_env.sh first. Does not start formal pretraining or publish W&B.
One persistent process owns the sequence; status changes do not require polling by an agent.
"""
import argparse
import json
import math
import os
from pathlib import Path
import signal
import subprocess
import sys
import time


def write(path,value):
    tmp=path.with_suffix('.tmp');tmp.write_text(json.dumps(value,indent=2)+'\n');tmp.replace(path)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--models',nargs='+',default=['dense','mor','loopformer','circle'])
    p.add_argument('--timeout',type=int,default=3600)
    p.add_argument('--mor-pilot-steps',type=int,default=64)
    p.add_argument('--loopformer-pilot-steps',type=int,default=256)
    args=p.parse_args()
    root=Path(__file__).resolve().parents[1]
    hardware=json.loads((root/'configs/native/hardware.json').read_text())['a800']['micro_batch']
    if set(args.models)-set(hardware):p.error('unknown model')
    active=subprocess.check_output(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader,nounits'],text=True).strip()
    if active:raise RuntimeError('GPU compute processes already exist')
    args.output.mkdir(parents=True,exist_ok=False)
    status=args.output/'status.json';completed=[]
    env={**os.environ,'WANDB_MODE':'disabled','TORCH_NCCL_ASYNC_ERROR_HANDLING':'1'}

    def run(name,module,options):
        write(status,{'phase':'running','stage':name,'completed':completed})
        command=[sys.executable,'-m','torch.distributed.run','--standalone','--nproc_per_node=8',
                 '-m',module,*map(str,options)]
        start=time.time()
        with (args.output/f'{name}.log').open('w') as log:
            child=subprocess.Popen(command,cwd=root,env=env,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
            try:code=child.wait(timeout=args.timeout)
            except subprocess.TimeoutExpired:
                os.killpg(child.pid,signal.SIGTERM)
                try:child.wait(timeout=20)
                except subprocess.TimeoutExpired:os.killpg(child.pid,signal.SIGKILL);child.wait()
                raise RuntimeError(f'{name} timed out')
        if code:raise RuntimeError(f'{name} failed with code {code}; inspect its log')
        completed.append({'stage':name,'seconds':time.time()-start,'command':command})
        print(json.dumps({'stage':name,'status':'passed'}),flush=True)

    try:
        reports={}
        for model in args.models:
            directory=args.output/model;config=root/'configs/native'/f'{model}.json'
            batch=hardware[model]
            shared=['--config',config,'--output',directory,'--micro-batch',batch,'--log-every',1,'--wandb-mode','disabled']
            if model=='circle':shared+=['--circle-stage-step',10000]
            first_steps=(args.mor_pilot_steps if model=='mor' else
                         args.loopformer_pilot_steps if model=='loopformer' else 2)
            if first_steps<2:raise ValueError('pilot must include at least two updates')
            run(model+'_train','nexttransformer.native.train',shared+['--smoke-steps',first_steps])
            checkpoint=directory/f'checkpoint_{first_steps:08d}'
            initial=json.loads((directory/'run_status.json').read_text())
            run(model+'_resume','nexttransformer.native.train',shared+['--smoke-steps',first_steps+1,'--resume',checkpoint])
            final=json.loads((directory/'run_status.json').read_text())
            assert initial['step']==first_steps and final['step']==first_steps+1
            assert final['input_tokens']==(first_steps+1)*1048576 and final['smoke']
            last=directory/f'checkpoint_{first_steps+1:08d}'
            manifest=json.loads((last/'complete.json').read_text())
            assert len(manifest['files'])==9 and manifest['world_size']==8
            output=directory/'evaluation_smoke.json'
            run(model+'_eval','nexttransformer.evaluate',['--checkpoint',last,'--output',output,
                '--micro-batch',batch,'--split-tokens',batch*8*2048,'--max-per-task',64])
            evaluation=json.loads(output.read_text())
            assert len(evaluation['core']['results'])==22 and not evaluation['core']['full_core']
            assert math.isfinite(evaluation['core']['core_metric'])
            for split in ('train','val'):
                assert math.isfinite(evaluation['bpb'][split][split+'_bpb'])
            events=[json.loads(line) for line in (directory/'metrics.jsonl').read_text().splitlines()]
            train=[row for row in events if row['event']=='train']
            assert all(math.isfinite(row['lm_loss']) and math.isfinite(row['grad_norm']) for row in train)
            reports[model]={'initial_status':initial,'resumed_status':final,'last_train':train[-1],
                            'checkpoint_manifest':manifest,'evaluation_smoke':evaluation}
            write(args.output/'evidence.json',{'scope':'short acceptance, not quality or convergence evidence','models':reports})
        write(status,{'phase':'complete','completed':completed,'models':list(reports)})
    except BaseException as exc:
        write(status,{'phase':'failed','error':repr(exc),'completed':completed});raise


if __name__=='__main__':main()
