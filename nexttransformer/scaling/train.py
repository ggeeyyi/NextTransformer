"""Dense scaling-stage pretraining on nodes without a shared filesystem.

--dry-run prints the plan, --benchmark times updates, --smoke-steps runs a bounded test. Every node needs the
same source files, pinned data inventory and tokenizer; the run compares them across nodes before training.
Each node writes its own checkpoints and status file; rank zero writes the event history and W&B run.
"""
from nexttransformer.runtime.site import add_runtime_arguments, bootstrap_runtime
if __name__ == '__main__':
    bootstrap_runtime()

import argparse
from contextlib import nullcontext
from dataclasses import asdict, replace
import fcntl
from functools import partial
import hashlib
import json
import math
import os
from pathlib import Path
import signal
import socket
import statistics
import sys
import time

import torch
import torch.distributed as dist
from torch.distributed.algorithms.ddp_comm_hooks import default_hooks
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.checkpoint import checkpoint

from nanochat.flash_attention import USE_FA3
from nanochat.tokenizer import RustBPETokenizer
from nexttransformer.evaluation.native import evaluate_bpb, evaluate_core
from nexttransformer.models.native.model import NativeExperiment, build_model, theoretical_training_flops
from nexttransformer.native.optimizer import set_schedule
from nexttransformer.runtime.data import BestFitStream, verify_resources
from nexttransformer.runtime.files import atomic_json
from nexttransformer.runtime.process import (initialize, autocast, synchronize, source_manifest, append_event,
                                             initial_state, reduce_sum, rewind_events, rng_state, restore_rng)
from nexttransformer.runtime.statistics import training_stats, accumulate_stats
from nexttransformer.runtime.tracking import start_tracker
from nexttransformer.runtime.tracking_auth import preflight
from nexttransformer.scaling.checkpoint import (topology, gather, agree, common_step, complete_steps, save_checkpoint,
                                                load_checkpoint, set_aside_after, prune)
from nexttransformer.scaling.config import load_config, config_hash
from nexttransformer.scaling.optimizer import build_optimizer, muon_weight_decay, scaling_params

# Dense BF16 tensor-core peaks, used only for the reported MFU; unknown devices report none.
# Ordered so that a name is matched before any of its substrings ("H200" before "H20").
PEAK_FLOPS = (("B200", 2250e12), ("H200", 989e12), ("H100", 989e12), ("H800", 989e12), ("H20", 148e12),
              ("A100", 312e12), ("A800", 312e12), ("L40S", 362e12))
# Exit code and node status phase for a non-finite loss or gradient: restarting would replay the same update.
NONFINITE_EXIT = 3
stop_signal = []


def peak_flops(device):
    if device.type != 'cuda':
        return None
    name = torch.cuda.get_device_name(device)
    return next((peak for key, peak in PEAK_FLOPS if key in name), None)


def data_identity(record):
    keys = ('data_revision', 'data_inventory_sha256', 'validation_sha256', 'tokenizer_sha256')
    return hashlib.sha256(json.dumps({k: record[k] for k in keys}, sort_keys=True).encode()).hexdigest()


def check_nodes(identity):
    """Every rank runs the same sources, software, configuration, data and tokenizer; returns {node: host}."""
    node, _, _ = topology()
    entries = gather({'node': node, 'host': socket.gethostname(), **identity})
    for key in identity:
        values = {}
        for entry in entries:
            values.setdefault(json.dumps(entry[key]), set()).add(entry['host'])
        if len(values) > 1:
            raise ValueError(f'nodes disagree on {key}: '+'; '.join(f'{sorted(h)} {v}' for v, h in values.items()))
    return {f"{e['node']:03d}": e['host'] for e in entries}


def request_stop(number, frame):
    # The first signal stops after the current update with a checkpoint; a second one terminates at once.
    stop_signal.append(number)
    signal.signal(number, signal.SIG_DFL)


def archive_records(output):
    """Move the records of a run that never reached a checkpoint aside before it restarts from step zero."""
    archive = Path(output)/'restarts'/str(time.time_ns())
    for path in Path(output).iterdir():
        if path.is_file() and path.suffix in ('.json', '.jsonl') and not path.name.startswith('status.node'):
            archive.mkdir(parents=True, exist_ok=True)
            try:
                path.rename(archive/path.name)
            except FileNotFoundError:
                pass  # a node sharing this filesystem moved it first


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    add_runtime_arguments(parser)
    parser.add_argument('--config', required=True, type=Path)
    parser.add_argument('--output', type=Path)
    parser.add_argument('--world-size', type=int, default=32, help='dry-run planning only; runtime uses torchrun WORLD_SIZE')
    parser.add_argument('--dry-run', action='store_true')
    parser.add_argument('--resume', default='never',
                        help='never, auto (newest checkpoint complete on every node; fresh start if none) or a step')
    parser.add_argument('--allow-source-change', action='store_true', help='resume with changed source files (recorded)')
    parser.add_argument('--smoke-steps', type=int, default=0, help='explicit short test, never a completed pretraining run')
    parser.add_argument('--smoke-val-tokens', type=int, default=0, help='0 uses one complete distributed evaluation batch')
    parser.add_argument('--stop-after', type=int, default=0, help='checkpoint and stop after this many updates of this launch')
    parser.add_argument('--benchmark', type=int, default=0, metavar='STEPS',
                        help='time STEPS updates; no evaluation, checkpoints, tracking or run records')
    parser.add_argument('--benchmark-batch-tokens', type=int, default=0, help='benchmark-only global batch')
    parser.add_argument('--log-every', type=int, default=10)
    parser.add_argument('--compile', action='store_true', help='torch.compile each Transformer block')
    parser.add_argument('--recompute-layers', type=int, default=0,
                        help='recompute the activations of the first N blocks during backward (-1: every block)')
    parser.add_argument('--no-bucket-view', dest='bucket_view', action='store_false',
                        help='keep gradients outside the DDP buckets (one more gradient-sized buffer)')
    parser.add_argument('--bf16-grad-reduce', action='store_true', help='average gradients in BF16 (DDP compression hook)')
    parser.add_argument('--keep-local', type=int, default=3, help='rolling checkpoints kept on each node besides milestones')
    args = parser.parse_args(argv)
    args.micro_batch = int(args.micro_batch or 2)
    if args.log_every < 1: parser.error('--log-every must be positive')
    if args.resume not in ('never', 'auto') and not args.resume.isdigit():
        parser.error('--resume takes never, auto or a step')
    if args.benchmark and (args.smoke_steps or args.stop_after or args.resume != 'never'):
        parser.error('--benchmark cannot be combined with smoke, stop or resume options')
    if args.benchmark_batch_tokens and not args.benchmark: parser.error('--benchmark-batch-tokens needs --benchmark')
    if args.keep_local < 2: parser.error('--keep-local must keep at least two checkpoints')
    mc, tc = load_config(args.config)
    if args.benchmark_batch_tokens:
        tc = replace(tc, global_batch_tokens=args.benchmark_batch_tokens, total_tokens=args.benchmark_batch_tokens*tc.steps)
    core_steps = tc.core_steps()
    if args.dry_run:
        with torch.device('meta'): model = NativeExperiment(mc)
        flops = theoretical_training_flops(mc)
        print(json.dumps({'model': asdict(mc), 'train': asdict(tc), 'config_sha256': config_hash(mc, tc),
            'parameters': sum(p.numel() for p in model.parameters()), 'scaling_parameters': scaling_params(model),
            'muon_lr': tc.muon_lr*tc.batch_scale, 'muon_weight_decay': muon_weight_decay(model, tc), 'steps': tc.steps,
            'accumulation': tc.accumulation(mc, args.world_size, args.micro_batch),
            'validation_events': len(tc.validation_steps()), 'core_steps': core_steps,
            'train_GFLOPs_per_input_token': flops/1e9, 'total_training_flops': flops*tc.total_tokens}, indent=2))
        return
    benchmark = bool(args.benchmark)
    smoke = benchmark or bool(args.smoke_steps)
    output = args.output or args.runtime_dir/'runs'/(args.run_name or 'scaling_'+args.config.stem)
    for number in (signal.SIGTERM, signal.SIGINT):
        signal.signal(number, request_stop)
    # A missing container credential must fail before rank zero owns an NCCL group.
    if int(os.environ.get('RANK', '0')) == 0 and not benchmark:
        preflight(args.wandb_mode)
    rank, world, device = initialize(tc.seed)
    node, local_rank, local_world = topology()
    accumulation = tc.accumulation(mc, world, args.micro_batch)
    run_lock = None
    if not benchmark and local_rank == 0:
        output.mkdir(parents=True, exist_ok=True)
        run_lock = (output/f'run.node{node:03d}.lock').open('a')
        fcntl.flock(run_lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    if world > 1: dist.barrier()
    record, manifest_hash = verify_resources(args.resources, args.data_dir)
    os.environ['NANOCHAT_BASE_DIR'] = str(args.resources)
    tokenizer = RustBPETokenizer.from_directory(args.resources/'tokenizer')
    if tokenizer.get_vocab_size() != mc.vocab_size: raise ValueError('model/tokenizer vocabulary mismatch')
    source = source_manifest()
    identity = dict(source_sha256=source['sha256'], data=data_identity(record), config_sha256=config_hash(mc, tc),
                    torch=torch.__version__, cuda=torch.version.cuda, attention='fa3' if USE_FA3 else 'sdpa')
    hosts = check_nodes(identity)
    resumed = None
    if not benchmark:
        if args.resume == 'never':
            if not agree(not (output/'provenance.json').exists(), device):
                raise ValueError('output already contains a run; use --resume auto or a new --output directory')
        else:
            resumed = common_step(output) if args.resume == 'auto' else int(args.resume)
            if resumed is not None and not agree(resumed in complete_steps(output, node), device):
                raise ValueError(f'checkpoint {resumed} is not complete on every node')
            if local_rank == 0:
                set_aside_after(output, -1 if resumed is None else resumed, node)
                if resumed is None: archive_records(output)
        if world > 1: dist.barrier()
    data_dir = args.data_dir or args.resources/'base_data_climbmix'
    train = BestFitStream(data_dir, tokenizer, 'train', rank, world, mc.sequence_len)
    stop_step = min(tc.steps, args.benchmark or args.smoke_steps) if smoke else tc.steps
    val_tokens = (args.smoke_val_tokens or args.micro_batch*world*mc.sequence_len) if smoke else tc.val_tokens
    if val_tokens % (args.micro_batch*world*mc.sequence_len):
        raise ValueError('validation tokens must divide the distributed evaluation batch')
    model = build_model(mc, device)
    recompute = mc.dense_layers if args.recompute_layers < 0 else min(args.recompute_layers, mc.dense_layers)
    for block in model.transformer.h[:recompute]:
        block.forward = partial(checkpoint, block.forward, use_reentrant=False)
    if args.compile:
        for block in model.transformer.h: block.compile(dynamic=False)
    wrapped = DDP(model, device_ids=[device.index] if device.type == 'cuda' else None, broadcast_buffers=False,
                  gradient_as_bucket_view=args.bucket_view) if world > 1 else model
    if args.bf16_grad_reduce and world > 1:
        wrapped.register_comm_hook(None, default_hooks.bf16_compress_hook)
    optimizer = build_optimizer(model, tc)
    state = initial_state(identity['data'])
    if resumed is not None:
        saved = load_checkpoint(output, resumed, model, optimizer, train, device)
        if saved['model_config'] != asdict(mc) or saved['train_config'] != asdict(tc):
            raise ValueError('resume configuration differs from checkpoint')
        state = saved['state']
        if state['data_manifest_sha256'] != identity['data']:
            raise ValueError('resume data or tokenizer differs from checkpoint')
        if state['input_tokens'] != state['step']*tc.global_batch_tokens or \
                state['next_global_sequence'] != state['input_tokens']//mc.sequence_len:
            raise ValueError('checkpoint token cursor is inconsistent')
        if train.rows*world*mc.sequence_len != state['input_tokens']:
            raise ValueError('exact rank-local data cursor differs from checkpoint token count')
        if state.get('smoke', False) != smoke:
            raise ValueError('smoke checkpoints cannot be promoted to pretraining runs')
        if state['source_sha256'] != source['sha256'] and not args.allow_source_change:
            raise ValueError('resume source files differ from checkpoint; preserve them or pass --allow-source-change')
        del saved
    previous_source = state.get('source_sha256')
    state['source_sha256'], state['smoke'] = source['sha256'], smoke
    if not benchmark and local_rank == 0:
        if rank == 0 and resumed is not None:
            # Archive events beyond the selected checkpoint before deterministic replay.
            rewind_events(output, state['step'])
        atomic_json(output/('provenance.json' if resumed is None else f'resume_{state["step"]:08d}.json'), {
            'model': asdict(mc), 'train': asdict(tc), **identity, 'source': source, 'previous_source_sha256': previous_source,
            'command': sys.argv, 'world_size': world, 'nodes': hosts,
            'device': torch.cuda.get_device_name(device) if device.type == 'cuda' else str(device),
            'micro_batch': args.micro_batch, 'gradient_accumulation': accumulation, 'compile': args.compile,
            'recompute_layers': recompute, 'bucket_view': args.bucket_view, 'bf16_grad_reduce': args.bf16_grad_reduce,
            'native_manifest_sha256': manifest_hash, 'smoke': smoke,
            'parameters': sum(p.numel() for p in model.parameters()), 'scaling_parameters': scaling_params(model),
            'muon_weight_decay': muon_weight_decay(model, tc),
            'theoretical_GFLOPs_per_token': theoretical_training_flops(mc)/1e9})
    peak = peak_flops(device)
    tracker_args = argparse.Namespace(**{**vars(args), 'output': output, 'resume': resumed is not None,
                                         'wandb_mode': 'disabled' if benchmark else args.wandb_mode})
    with start_tracker(tracker_args, rank, mc, tc, state, source['sha256'], device) as tracker:
        model.train()

        def record(event):
            if rank == 0:
                tracker.log(event)
                append_event(output, event)
                print(json.dumps(event), flush=True)

        def validation(step):
            if benchmark or step in state['completed_val_steps']: return
            result = evaluate_bpb(model, tokenizer, data_dir, args.resources, val_tokens, args.micro_batch, rank, world, device)
            state['completed_val_steps'].append(step)
            state['validation_forward_flops'] += result['forward_flops_estimate']
            state['validation_seconds'] += result['seconds']
            record({'event': 'validation', 'step': step, **result})
            if rank == 0: atomic_json(output/f'val_{step:08d}.json', {'event': 'validation', 'step': step, **result})

        def warm_up():
            # DDP rebuilds its gradient buckets after the first backward. One discarded backward on a fixed
            # batch gives every update of every launch the same bucket layout, so NCCL's ring all-reduce
            # rounds identically and a resumed run matches a continuous one. No data or RNG state changes.
            if world == 1: return
            rngs, routing = rng_state(), model.routing_rng_state()
            dummy = torch.zeros(args.micro_batch, mc.sequence_len, dtype=torch.long, device=device)
            with autocast(device):
                wrapped(dummy, dummy)['loss'].backward()
            optimizer.zero_grad(set_to_none=not args.bucket_view)
            restore_rng(rngs); model.restore_routing_rng(routing)

        if state['step'] == 0: validation(0)
        warm_up()
        phase, launched, timings, last_checkpoint = None, state['step'], [], None
        for step in range(state['step'], stop_step):
            synchronize(device); start = time.perf_counter(); data_seconds = 0.
            if device.type == 'cuda': torch.cuda.reset_peak_memory_stats(device)
            optimizer.zero_grad(set_to_none=not args.bucket_view)
            scale = set_schedule(optimizer, step, tc)
            stats = training_stats(mc, device)
            for micro in range(accumulation):
                t = time.perf_counter()
                x, y = train.batch(args.micro_batch, device)
                data_seconds += time.perf_counter()-t
                sync = wrapped.no_sync() if world > 1 and micro < accumulation-1 else nullcontext()
                with sync, autocast(device):
                    out = wrapped(x, y)
                    (out['loss']/accumulation).backward()
                accumulate_stats(stats, out, model.last_stats, accumulation, world, mc)
            grad_norm = float(torch.nn.utils.clip_grad_norm_(model.parameters(), tc.grad_clip or float('inf')))
            values = reduce_sum(stats).tolist()
            if int(values[4]) != tc.global_batch_tokens: raise RuntimeError('training input-token budget drift')
            # DDP gives every rank the same gradients and the reduced loss, so all ranks stop together.
            if not (math.isfinite(grad_norm) and math.isfinite(values[0])):
                record({'event': 'nonfinite', 'step': step+1, 'lm_loss': values[0], 'grad_norm': grad_norm})
                phase = 'nonfinite'
                break
            synchronize(device); optimizer_start = time.perf_counter()
            optimizer.step()
            synchronize(device); optimizer_seconds = time.perf_counter()-optimizer_start
            duration = time.perf_counter()-start
            state['step'] = step+1; state['input_tokens'] += tc.global_batch_tokens
            state['next_global_sequence'] = state['input_tokens']//mc.sequence_len
            state['training_flops_estimate'] += values[2]; state['training_seconds'] += duration
            memory = torch.cuda.max_memory_allocated(device) if device.type == 'cuda' else 0
            timings.append((duration, data_seconds, optimizer_seconds, memory, values[2]))
            if not benchmark and ((step+1) % args.log_every == 0 or step == launched or step+1 == stop_step):
                record({'event': 'train', 'step': step+1, 'input_tokens': state['input_tokens'], 'lm_loss': values[0],
                        'loss': values[6+mc.routing_steps], 'grad_norm': grad_norm, 'lr_multiplier': scale,
                        'training_flops_estimate': state['training_flops_estimate'], 'step_flops_estimate': values[2],
                        'tokens_per_second': tc.global_batch_tokens/duration,
                        'mfu': values[2]/(duration*world*peak) if peak else None, 'step_seconds': duration,
                        'data_seconds': data_seconds, 'optimizer_seconds': optimizer_seconds,
                        'gpu_hours': world*state['training_seconds']/3600, 'peak_memory_bytes': memory})
            if state['step'] % tc.val_every == 0 or state['step'] == stop_step: validation(state['step'])
            if not smoke and state['step'] < tc.steps and state['step'] in core_steps:
                result = evaluate_core(model, tokenizer, args.resources, device, tc.core_max_per_task)
                record({'event': 'core', 'step': state['step'], **result})
                if rank == 0: atomic_json(output/f"core_{state['step']:08d}.json", {'event': 'core', 'step': state['step'], **result})
            requested = bool(stop_signal) or (rank == 0 and not benchmark and (output/'STOP').exists())
            stop = not agree(not requested, device) or (args.stop_after and state['step']-launched >= args.stop_after)
            if not benchmark and (state['step'] % tc.checkpoint_every == 0 or state['step'] == stop_step or stop):
                last_checkpoint = save_checkpoint(output, model, optimizer, train, mc, tc, state,
                                                  milestone=not smoke and tc.is_milestone(state['step']))
                if local_rank == 0: prune(output, node, args.keep_local)
            if stop and state['step'] < stop_step:
                if rank == 0 and (output/'STOP').exists(): (output/'STOP').rename(output/f'STOP.consumed-{state["step"]:08d}')
                phase = 'stopped'
                break
        complete = phase is None and not smoke and state['step'] == tc.steps
        if complete:
            # The final checkpoint precedes these evaluations; a restart repeats only the unfinished ones.
            final_bpb = {split: evaluate_bpb(model, tokenizer, data_dir, args.resources, tc.final_bpb_tokens,
                                             args.micro_batch, rank, world, device, split) for split in ('train', 'val')}
            if rank == 0: atomic_json(output/'bpb_final.json', final_bpb)
            record({'event': 'final_bpb', 'step': state['step'],
                    **{split+'_bpb': result[split+'_bpb'] for split, result in final_bpb.items()}})
            result = evaluate_core(model, tokenizer, args.resources, device)
            if rank == 0: atomic_json(output/'core_final.json', result)
            record({'event': 'core', 'step': state['step'], **result})
        phase = phase or ('smoke_complete' if smoke else 'training_complete' if complete else 'stopped')
        if benchmark and timings:
            measured = timings[min(3, len(timings)-1):]
            memory = torch.tensor(float(max(t[3] for t in measured)), device=device)
            if world > 1: dist.all_reduce(memory, op=dist.ReduceOp.MAX)
            seconds = statistics.median(t[0] for t in measured)
            summary = {'event': 'benchmark', 'world_size': world, 'micro_batch': args.micro_batch,
                       'gradient_accumulation': accumulation, 'global_batch_tokens': tc.global_batch_tokens,
                       'compile': args.compile, 'recompute_layers': recompute, 'bucket_view': args.bucket_view,
                       'bf16_grad_reduce': args.bf16_grad_reduce,
                       'attention': identity['attention'], 'measured_steps': len(measured), 'step_seconds': seconds,
                       'data_seconds': statistics.median(t[1] for t in measured),
                       'optimizer_seconds': statistics.median(t[2] for t in measured),
                       'tokens_per_second_per_gpu': tc.global_batch_tokens/seconds/world,
                       'mfu': statistics.median(t[4] for t in measured)/(seconds*world*peak) if peak else None,
                       'peak_memory_gib': memory.item()/2**30}
            if rank == 0:
                print(json.dumps(summary), flush=True)
                if args.output: atomic_json(args.output/'benchmark.json', summary)
        elif not benchmark and rank == 0:
            atomic_json(output/'run_status.json', {'phase': phase, **state,
                'checkpoint': str(last_checkpoint) if last_checkpoint else None,
                'final_core_complete': (output/'core_final.json').exists()})
        if not benchmark and local_rank == 0:
            atomic_json(output/f'status.node{node:03d}.json', {'phase': phase, 'step': state['step'], 'time': time.time()})
    if world > 1: dist.destroy_process_group()
    if phase == 'nonfinite': sys.exit(NONFINITE_EXIT)


if __name__ == '__main__':
    main()
