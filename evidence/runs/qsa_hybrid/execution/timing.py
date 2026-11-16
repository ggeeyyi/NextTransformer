"""Bounded stream pacing at real model layer boundaries; hooks return no tensors."""
import json
import os
import time
import weakref
from pathlib import Path
import torch
import core_timing

root=Path(__file__).resolve().parents[1]
tracked={'Block','FlexibleMLP','LinearMixer','SparseMixer'}
registered=weakref.WeakSet()
last=None
last_flush=0.
calls=0
sleep_total=0.
duty=.74

def boundary(tensor,requested_duty=None):
    global last,last_flush,calls,sleep_total
    if not isinstance(tensor,torch.Tensor) or not tensor.is_cuda:return
    stream=torch.cuda.current_stream(tensor.device)
    now=torch.cuda.Event(enable_timing=True)
    now.record(stream)
    now.synchronize()
    if last is not None:
        seconds=last.elapsed_time(now)/1000
        effective=duty if requested_duty is None else requested_duty
        pause=min(.008,max(0.,seconds)*(1/effective-1))
        if pause:time.sleep(pause)
        sleep_total+=pause
    last=torch.cuda.Event(enable_timing=True)
    last.record(stream)
    calls+=1
    if time.monotonic()-last_flush>=2:
        rank=os.environ.get('NEXTT_PACING_ID',os.environ.get('LOCAL_RANK','0'))
        p=root/'evidence'/f'pacing-{os.environ["T2M_H1_GPU_LEASE"]}-{rank}.json'
        tmp=p.with_suffix('.tmp')
        tmp.write_text(json.dumps(dict(at=time.time(),calls=calls,sleep_seconds=sleep_total,duty=duty if requested_duty is None else requested_duty)))
        tmp.replace(p);last_flush=time.monotonic()

def forward(module,inputs,out):
    if torch.cuda.is_current_stream_capturing():return
    if not isinstance(out,torch.Tensor) or not out.is_cuda:return
    if core_timing.active and not torch.is_grad_enabled():
        requested=core_timing.pacing()
        if requested is not None:boundary(out,requested_duty=requested)
        return
    boundary(out,requested_duty=None if torch.is_grad_enabled() else .88)
    if torch.is_grad_enabled() and out.requires_grad:out.register_hook(boundary)

def registered_child(parent,name,child):
    if type(child).__module__.startswith('nexttransformer.models.') and type(child).__name__ in tracked and child not in registered:
        registered.add(child)
        child.register_forward_hook(forward)

def install():
    assert os.environ.get('T2M_H1_GPU_LEASE')
    core_timing.install()
    torch.nn.modules.module.register_module_module_registration_hook(registered_child)
