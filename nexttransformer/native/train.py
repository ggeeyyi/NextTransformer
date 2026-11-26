"""Train the pinned native nanochat d24 comparison; use --dry-run before launching GPUs."""
from nexttransformer.runtime.site import add_runtime_arguments, bootstrap_runtime
if __name__ == '__main__':
    bootstrap_runtime()

import argparse
from contextlib import nullcontext
from dataclasses import asdict
import json
import os
from pathlib import Path
import sys
import time

import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP

from nexttransformer.runtime.tracking import start_tracker
from nexttransformer.runtime.tracking_auth import preflight
from nexttransformer.models.native.config import load_config
from nexttransformer.runtime.files import atomic_json
from nexttransformer.runtime.data import BestFitStream, verify_resources
from nanochat.tokenizer import RustBPETokenizer
from nexttransformer.evaluation.context import add_core_max_tokens, check_full_context, resume_core_limit
from nexttransformer.evaluation.native import evaluate_bpb, evaluate_core
from nexttransformer.models.native.model import NativeExperiment, build_model, theoretical_training_flops
from nexttransformer.models.native.halter import feature_forward_ops
from nexttransformer.runtime.statistics import training_stats, accumulate_stats
from nexttransformer.native.optimizer import build_optimizer, set_schedule, clip_halter_gradients, clip_condition_gradients
from nexttransformer.runtime.checkpoint import save_checkpoint, load_checkpoint
from nexttransformer.runtime.process import (initialize,autocast,synchronize,source_manifest,append_event,initial_state,reduce_sum)


def main():
    parser=argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    add_runtime_arguments(parser)
    add_core_max_tokens(parser)
    parser.add_argument('--config',required=True,type=Path)
    parser.add_argument('--output',type=Path)
    parser.add_argument('--world-size',type=int,default=8,help='dry-run planning only; runtime uses torchrun WORLD_SIZE')
    parser.add_argument('--dry-run',action='store_true')
    parser.add_argument('--resume',type=Path)
    parser.add_argument('--smoke-steps',type=int,default=0,help='explicit short test, never a completed pretraining run')
    parser.add_argument('--smoke-val-tokens',type=int,default=0,help='0 uses one complete distributed evaluation batch')
    parser.add_argument('--log-every',type=int,default=10)
    parser.add_argument('--skip-final-core',action='store_true',help='run nexttransformer.evaluate --eval core separately')
    parser.add_argument('--circle-stage-step',type=int,default=0,help='smoke-only curriculum offset for testing learned routing')
    args=parser.parse_args()
    if args.circle_stage_step and not args.smoke_steps: parser.error('curriculum offset is smoke-only')
    args.micro_batch = args.micro_batch or 4
    args.output = args.output or args.runtime_dir/'runs'/(args.run_name or 'native_'+args.config.stem+'_seed42')
    if args.log_every < 1: parser.error('--log-every must be positive')
    mc,tc=load_config(args.config)
    core_steps = tc.core_steps(include_final=not args.skip_final_core)
    if args.dry_run:
        with torch.device('meta'): model=NativeExperiment(mc)
        print(json.dumps({'model':asdict(mc),'train':asdict(tc),'parameters':sum(p.numel() for p in model.parameters()),
            'accumulation':tc.accumulation(mc,args.world_size,args.micro_batch),'steps':tc.steps,
            'halter_extra_forward_ops_per_input_token':feature_forward_ops(mc),
            'validation_events':len(tc.validation_steps()),
            'core_steps':[] if args.smoke_steps else core_steps,
            'core_events':0 if args.smoke_steps else len(core_steps),
            'train_GFLOPs_per_input_token':theoretical_training_flops(mc)/1e9},indent=2))
        return
    if args.resume and args.output.exists():
        resume_step=json.loads((args.resume/'complete.json').read_text())['step']
        if any(int(path.name.removeprefix('checkpoint_'))>resume_step
               for path in args.output.glob('checkpoint_[0-9]*') if path.is_dir()):
            raise ValueError('output contains later checkpoints; resume into a new --output directory')
    # A missing container credential must fail before rank zero owns an NCCL group.
    if int(os.environ.get('RANK', '0')) == 0:
        preflight(args.wandb_mode)
    rank,world,device=initialize(tc.seed)
    accumulation=tc.accumulation(mc,world,args.micro_batch)
    # Rank zero owns a run lock. Existence alone does not imply a live training run.
    import fcntl
    run_lock=None
    if rank==0:
        args.output.mkdir(parents=True,exist_ok=True)
        run_lock=(args.output/'run.lock').open('a')
        fcntl.flock(run_lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        if (args.output/'provenance.json').exists() and args.resume is None:
            raise ValueError('output already contains a run; supply --resume or a new output directory')
    if world>1: dist.barrier()
    record, manifest_hash=verify_resources(args.resources,args.data_dir)
    os.environ['NANOCHAT_BASE_DIR']=str(args.resources)
    tokenizer=RustBPETokenizer.from_directory(args.resources/'tokenizer')
    if tokenizer.get_vocab_size()!=mc.vocab_size: raise ValueError('model/tokenizer vocabulary mismatch')
    data_dir=args.data_dir or args.resources/'base_data_climbmix'
    train=BestFitStream(data_dir,tokenizer,'train',rank,world,mc.sequence_len)
    stop_step=min(tc.steps,args.smoke_steps) if args.smoke_steps else tc.steps
    val_tokens=(args.smoke_val_tokens or args.micro_batch*world*mc.sequence_len) if args.smoke_steps else tc.val_tokens
    if val_tokens % (args.micro_batch*world*mc.sequence_len):
        raise ValueError('validation tokens must divide the distributed evaluation batch')
    model=build_model(mc,device)
    wrapped=DDP(model,device_ids=[device.index] if device.type=='cuda' else None,broadcast_buffers=False) if world>1 else model
    optimizer=build_optimizer(model,tc)
    state=initial_state(manifest_hash)
    if args.resume:
        saved=load_checkpoint(args.resume,model,optimizer,train,rank,world)
        if saved['model_config']!=asdict(mc) or saved['train_config']!=asdict(tc):
            raise ValueError('resume configuration differs from checkpoint')
        state=saved['state']
        if state['data_manifest_sha256']!=manifest_hash:
            raise ValueError('resume data manifest differs from checkpoint')
        if state['input_tokens']!=state['step']*tc.global_batch_tokens or state['next_global_sequence']!=state['input_tokens']//mc.sequence_len:
            raise ValueError('checkpoint token cursor is inconsistent')
        if train.rows*world*mc.sequence_len != state['input_tokens']:
            raise ValueError('exact rank-local data cursor differs from checkpoint token count')
        del saved
        args.core_max_tokens = resume_core_limit(state, args.core_max_tokens, mc.sequence_len)
    state['core_max_tokens'] = args.core_max_tokens
    if not args.smoke_steps:
        check_full_context(mc, args.core_max_tokens)
    source=source_manifest()
    if args.resume and state.get('source_sha256') != source['sha256']:
        raise ValueError('resume source files differ from checkpoint; preserve the original implementation')
    if args.resume and state.get('smoke',False) != bool(args.smoke_steps):
        raise ValueError('smoke checkpoints cannot be promoted to formal pretraining runs')
    state['source_sha256']=source['sha256']
    if args.resume and state.get('circle_stage_offset',0)!=args.circle_stage_step:
        raise ValueError('resume curriculum offset differs')
    state['circle_stage_offset']=args.circle_stage_step
    state['smoke']=bool(args.smoke_steps)
    if rank==0:
        if args.resume:
            # Archive events beyond the selected checkpoint before deterministic replay.
            from nexttransformer.runtime.process import rewind_events
            rewind_events(args.output,state['step'])
        atomic_json(args.output/('provenance.json' if not args.resume else f'resume_{state["step"]:08d}.json'),{'model':asdict(mc),'train':asdict(tc),'source':source,
            'core_max_tokens':args.core_max_tokens,'command':sys.argv,'torch':torch.__version__,'cuda':torch.version.cuda,'world_size':world,
            'device':torch.cuda.get_device_name(device) if device.type=='cuda' else str(device),
            'micro_batch':args.micro_batch,'gradient_accumulation':accumulation,
            'data_manifest_sha256':manifest_hash,'tokenizer_sha256':record['tokenizer_sha256'],
            'optimizer':'native nanochat groups/schedules; DDP averages gradients; BF16 lookup embeddings as upstream',
            'smoke':bool(args.smoke_steps),'parameters':sum(p.numel() for p in model.parameters()),
            'halter_extra_forward_ops_per_input_token':feature_forward_ops(mc),
            'theoretical_GFLOPs_per_token':theoretical_training_flops(mc)/1e9})
    with start_tracker(args,rank,mc,tc,state,source['sha256'],device) as tracker:
        model.train()
        def validation(step):
            if step in state['completed_val_steps']: return
            result=evaluate_bpb(model,tokenizer,data_dir,args.resources,val_tokens,args.micro_batch,rank,world,device)
            state['completed_val_steps'].append(step)
            state['validation_forward_flops']+=result['forward_flops_estimate']
            state['validation_seconds']+=result['seconds']
            if rank==0:
                event={'event':'validation','step':step,**result}
                tracker.log(event)
                append_event(args.output,event)
                atomic_json(args.output/f'val_{step:08d}.json',event)
                print(json.dumps(event),flush=True)
        if state['step']==0: validation(0)
        global_sequences=tc.global_batch_tokens//mc.sequence_len
        last_checkpoint=None
        for step in range(state['step'],stop_step):
            model.set_circle_step(step+args.circle_stage_step)
            synchronize(device); start=time.perf_counter(); data_seconds=0.0
            if device.type=='cuda': torch.cuda.reset_peak_memory_stats(device)
            optimizer.zero_grad(set_to_none=True)
            scale=set_schedule(optimizer,step,tc)
            stats=training_stats(mc,device)
            for micro in range(accumulation):
                t=time.perf_counter()
                x,y=train.batch(args.micro_batch,device)
                data_seconds+=time.perf_counter()-t
                sync=wrapped.no_sync() if world>1 and micro<accumulation-1 else nullcontext()
                with sync,autocast(device):
                    out=wrapped(x,y)
                    (out['loss']/accumulation).backward()
                s=model.last_stats
                accumulate_stats(stats,out,s,accumulation,world,mc)
            grad_norm=torch.nn.utils.clip_grad_norm_(model.parameters(),tc.grad_clip or float("inf"),error_if_nonfinite=True)
            halter_grad_norm=clip_halter_gradients(model)
            condition_grad_norm=clip_condition_gradients(model)
            synchronize(device); optimizer_start=time.perf_counter()
            optimizer.step()
            synchronize(device); optimizer_seconds=time.perf_counter()-optimizer_start
            reduce_sum(stats)
            values=stats.tolist()
            if int(values[4])!=tc.global_batch_tokens: raise RuntimeError('training input-token budget drift')
            budget = model.update_route_budget(values[6:6+mc.routing_steps],values[4],step+args.circle_stage_step)
            # Include the new controller in measured update time and GPU-hours.
            synchronize(device); duration=time.perf_counter()-start
            state['step']=step+1; state['input_tokens']+=tc.global_batch_tokens
            state['next_global_sequence']=state['input_tokens']//mc.sequence_len
            state['training_flops_estimate']+=values[2]; state['training_seconds']+=duration
            if rank==0 and ((step+1)%args.log_every==0 or step==0 or step+1==stop_step):
                event={'event':'train','step':step+1,'input_tokens':state['input_tokens'],'lm_loss':values[0],
                    'router_aux_loss':values[1],'loss':values[6+mc.routing_steps],
                    'objective':values[9+mc.routing_steps],'compute_loss':values[8+mc.routing_steps],
                    'logical_sparse_forward_flops':values[7+mc.routing_steps],
                    'training_flops_estimate':state['training_flops_estimate'],'step_flops_estimate':values[2],
                    'recompute_flops_estimate':values[3],'mean_active_depth':values[5]/values[4],
                    'active_ratios':[v/values[4] for v in values[6:6+mc.routing_steps]],'tokens_per_second':tc.global_batch_tokens/duration,
                    'step_seconds':duration,'optimizer_seconds':optimizer_seconds,'data_seconds':data_seconds,
                    'grad_norm':float(grad_norm),'lr_multiplier':scale,'gpu_hours':world*state['training_seconds']/3600,
                    'peak_memory_bytes':torch.cuda.max_memory_allocated(device) if device.type=='cuda' else 0}
                if mc.architecture=='circle':
                    ratios=event['active_ratios']+[0]
                    event.update(circle_stage=model.circle_stage(),circle_method=mc.circle_method,
                                 circle_budget_mode=mc.circle_budget_mode, **budget,
                                 mean_ring_steps=sum(event['active_ratios']),cap_ratio=ratios[-2],
                                 exit_ratios=[ratios[i]-ratios[i+1] for i in range(mc.routing_steps)])
                    event.update(halter_delta_scale=float(model.halter.delta_scale.detach()),
                                 halter_depth_bias=model.halter.depth_bias.detach().tolist(),
                                 halter_grad_norm=halter_grad_norm,learned_route_fraction=model.learned_route_fraction(),
                                 circle_depth_loss=event['compute_loss'],
                                 circle_shape_loss=event['router_aux_loss']-event['compute_loss'])
                if mc.architecture=='mor':
                    event.update(mean_recursions=sum(event['active_ratios']),
                                 budget_max_abs_error=max(abs(v) for v in budget['budget_error']),**budget)
                if mc.architecture=='loopformer':
                    event.update(loop_condition_grad_norm=condition_grad_norm,**model.conditioner.diagnostics())
                tracker.log(event)
                append_event(args.output,event); print(json.dumps(event),flush=True)
            if state['step']%tc.val_every==0 or state['step']==stop_step: validation(state['step'])
            if not args.smoke_steps and state['step'] < tc.steps and state['step'] in core_steps:
                result=evaluate_core(model,tokenizer,args.resources,device,tc.core_max_per_task,
                                     core_max_tokens=args.core_max_tokens)
                if rank==0:
                    event={'event':'core','step':state['step'],**result}
                    atomic_json(args.output/f"core_{state['step']:08d}.json",event)
                    tracker.log(event)
            if state['step']%tc.checkpoint_every==0 or state['step']==stop_step:
                last_checkpoint=save_checkpoint(args.output,model,optimizer,train,mc,tc,state,rank,world)
        if not args.smoke_steps and state['step']==tc.steps:
            final_bpb={split:evaluate_bpb(model,tokenizer,data_dir,args.resources,tc.final_bpb_tokens,
                        args.micro_batch,rank,world,device,split) for split in ('train','val')}
            if rank==0:
                atomic_json(args.output/'bpb_final.json',final_bpb)
                event={'event':'final_bpb','step':state['step'],
                       **{split+'_bpb':result[split+'_bpb'] for split,result in final_bpb.items()}}
                tracker.log(event)
                append_event(args.output,event)
        if not args.smoke_steps and state['step']==tc.steps and not args.skip_final_core:
            result=evaluate_core(model,tokenizer,args.resources,device,core_max_tokens=args.core_max_tokens)
            if rank==0:
                atomic_json(args.output/'core_final.json',result)
                tracker.log({'event':'core','step':state['step'],**result})
        if rank==0:
            atomic_json(args.output/'run_status.json',{'phase':'smoke_complete' if args.smoke_steps else 'training_complete',
                **state,'checkpoint':str(last_checkpoint) if last_checkpoint else str(args.resume),
                'final_core_complete':(args.output/'core_final.json').exists()})
        if world>1: dist.destroy_process_group()

if __name__=='__main__':
    main()
