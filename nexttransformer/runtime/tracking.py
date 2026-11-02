"""Optional rank-zero W&B logging. Local metrics/checkpoints remain authoritative."""
from dataclasses import asdict
import json
import uuid

import torch
import torch.distributed as dist

from nexttransformer.runtime.files import atomic_json
from nexttransformer.runtime.process import rng_state, restore_rng


def metrics_schema(model_config):
    return 'native-v3'


def native_metric_names(prefix, architecture):
    """One comparison schema for all four native models; details stay local."""
    if prefix == 'train':
        names = {'lm_loss', 'loss', 'input_tokens', 'training_flops_estimate',
                 'tokens_per_second', 'grad_norm', 'lr_multiplier', 'gpu_hours', 'peak_memory_bytes'}
        if architecture == 'mor':
            names |= {'router_aux_loss', 'mean_active_depth', 'mean_recursions', 'budget_max_abs_error'}
        elif architecture == 'loopformer':
            names |= {'loop_condition_grad_norm','loop_modulation_saturated_fraction'}
        elif architecture == 'circle':
            names |= {'mean_ring_steps', 'cap_ratio', 'budget_target_steps',
                      'budget_error_ema', 'budget_weight_used','router_aux_loss','budget_shape_rmse'}
        return names
    if prefix == 'validation':
        extra = {'circle': {'mean_ring_steps', 'cap_ratio'}, 'mor': {'mean_recursions'}}
        return {'val_bpb'} | extra.get(architecture, set())
    if prefix == 'core':
        return {'core_metric', 'results', 'full_core', 'entries'}
    if prefix == 'final_bpb':
        return {'train_bpb', 'val_bpb'}
    return set()


def metric_payload(event, model_config, profile='core'):
    """Select cloud metrics without mutating the full local event."""
    if profile not in ('core', 'full'):
        raise ValueError('wandb_metrics must be core or full')
    prefix = event['event']
    values = {'step':event['step']}
    if profile == 'full':
        names = event.keys() - {'step', 'event'}
    else:
        names = native_metric_names(prefix, model_config.architecture)

    def flatten(name, value):
        if isinstance(value, (int, float)):
            values[name] = value
        elif isinstance(value, list):
            for index, item in enumerate(value): flatten(f'{name}/{index+1}', item)
        elif isinstance(value, dict):
            for key, item in value.items(): flatten(f'{name}/{key}', item)
    for name in sorted(names):
        if name in event:
            flatten(f'{prefix}/{name}', event[name])
    return values


def start_tracker(args, rank, model_config, train_config, state, source_hash, device):
    """All ranks observe rank-zero init failure before entering validation collectives."""
    tracker, error = None, None
    try:
        tracker = RunTracker(args, rank, model_config, train_config, state, source_hash)
    except Exception as exc:
        error = exc
    failed = torch.tensor(int(error is not None), device=device)
    if dist.is_initialized():
        dist.all_reduce(failed, op=dist.ReduceOp.MAX)
    if failed.item():
        raise RuntimeError('W&B tracker initialization failed; all ranks stopped before validation') from error
    return tracker


class RunTracker:
    def __init__(self, args, rank, model_config, train_config, state, source_hash):
        self.run = None
        self.path = args.output/'tracking.json'
        self.record = None
        self.model_config = model_config
        self.profile = getattr(args, 'wandb_metrics', 'core')
        if rank != 0 or args.wandb_mode == 'disabled':
            return
        try:
            import wandb
        except ImportError as exc:
            raise RuntimeError('W&B requested: install with setup_environment.sh --tracking') from exc
        previous = json.loads(self.path.read_text()) if self.path.exists() else None
        project = args.wandb_project or (previous['project'] if previous and args.resume else 'nextt')
        entity = args.wandb_entity or (previous.get('entity') if previous and args.resume else None)
        same_run = bool(args.resume and previous and previous['mode'] == 'online' and
                        args.wandb_mode == 'online' and state['step'] >= previous['last_step'])
        if same_run and (project != previous['project'] or entity != previous.get('entity')):
            raise ValueError('W&B resume project/entity changed; keep the original destination or use a new output directory')
        directory = args.wandb_dir or args.output/'wandb'
        directory.mkdir(parents=True, exist_ok=True)
        group = args.wandb_group or (previous.get('group') if previous and args.resume else None)
        saved_rng = rng_state()
        try:
            self.run = wandb.init(
                project=project, entity=entity, group=group,
                name=args.run_name or args.output.name, mode=args.wandb_mode, dir=str(directory),
                id=previous['id'] if same_run else uuid.uuid4().hex[:8],
                resume='must' if same_run else 'never',
                # Do not implicitly capture/upload source files or console output.
                settings=wandb.Settings(disable_code=True, disable_git=True, console='off', init_timeout=60),
                config={'model':asdict(model_config),'train':asdict(train_config),
                        'metrics_schema':metrics_schema(model_config), 'metrics_profile':self.profile,
                        'source_sha256':source_hash,'data_manifest_sha256':state['data_manifest_sha256'],
                        'resume_step':state['step'], 'smoke':state.get('smoke',False),
                        'parent_run_id':previous['id'] if previous and args.resume and not same_run else None})
            self.run.define_metric('step')
            self.run.define_metric('*', step_metric='step')
        finally:
            restore_rng(saved_rng)
        self.record = {'id':self.run.id,'project':self.run.project,'entity':self.run.entity,
                       'mode':args.wandb_mode,'group':group,'last_step':state['step'],'metrics_profile':self.profile,
                       'metrics_schema':metrics_schema(model_config),
                       'parent_run_id':previous['id'] if previous and args.resume and not same_run else None}
        if previous and not same_run:
            atomic_json(args.output/'tracking_history'/f'{previous["id"]}.json', previous)
        atomic_json(self.path,self.record)

    def log(self, event):
        if self.run is None:
            return
        self.run.log(metric_payload(event, self.model_config, self.profile))
        self.record['last_step'] = max(self.record['last_step'],event['step'])
        atomic_json(self.path,self.record)

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, traceback):
        if self.run is not None:
            self.run.finish(exit_code=1 if exc_type else 0)
