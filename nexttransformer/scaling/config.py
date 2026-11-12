"""Scientific configuration of a scaling-stage run.

The model section is the native backbone configuration. The training section replaces the frozen
reference protocol: it names its own protocol version, token budget, batch and evaluation cadence, and
keeps the native optimizer recipe with the transfer rules in nexttransformer.scaling.optimizer.
"""
from dataclasses import dataclass, asdict
import hashlib
import json
import math
from pathlib import Path

from nexttransformer.models.native.config import NativeModelConfig

# nanochat tuned its learning rates at this many tokens per update; larger batches scale them by sqrt(B/B0).
REFERENCE_BATCH_TOKENS = 524288


@dataclass(frozen=True)
class ScalingTrainConfig:
    protocol: str = 'nextt-scale-200b-4k-v1'
    seed: int = 42
    total_tokens: int = 199_999_094_784
    global_batch_tokens: int = 2_097_152
    val_every: int = 1000
    val_tokens: int = 80*524288
    final_bpb_tokens: int = 40*524288
    core_every: int = 20_000
    core_max_per_task: int = -1
    checkpoint_every: int = 1000
    milestone_every: int = 10_000
    warmup_steps: int = 40
    warmdown_ratio: float = .65
    final_lr_fraction: float = .05
    muon_lr: float = .02
    adamw_lr: float = .0006
    embedding_lr: float = .3
    unembedding_lr: float = .008
    scalar_lr: float = .5
    weight_decay: float = .28
    grad_clip: float = 0.

    def __post_init__(self):
        if not self.protocol or not self.protocol.strip():
            raise ValueError('a scaling run names its protocol version')
        if min(self.total_tokens, self.global_batch_tokens, self.val_every, self.val_tokens, self.final_bpb_tokens,
               self.checkpoint_every, self.milestone_every) <= 0:
            raise ValueError('budgets and intervals must be positive')
        if self.milestone_every % self.checkpoint_every:
            raise ValueError('milestones must fall on checkpoint steps')
        if not 0 <= self.warmdown_ratio <= 1 or not 0 <= self.final_lr_fraction <= 1 or self.warmup_steps < 0:
            raise ValueError('invalid learning-rate schedule')
        if self.grad_clip < 0 or not all(math.isfinite(v) and v > 0 for v in (
                self.muon_lr, self.adamw_lr, self.embedding_lr, self.unembedding_lr, self.scalar_lr)):
            raise ValueError('learning rates must be positive and clipping nonnegative')
        self.steps

    @property
    def steps(self):
        if self.total_tokens % self.global_batch_tokens:
            raise ValueError('total_tokens must be a multiple of global_batch_tokens')
        return self.total_tokens//self.global_batch_tokens

    @property
    def batch_scale(self):
        return math.sqrt(self.global_batch_tokens/REFERENCE_BATCH_TOKENS)

    def accumulation(self, model, world_size, micro_batch):
        denominator = world_size*micro_batch*model.sequence_len
        if denominator <= 0 or self.global_batch_tokens % denominator:
            raise ValueError('global batch must be divisible by world_size * micro_batch * sequence_len')
        return self.global_batch_tokens//denominator

    def validation_steps(self):
        return sorted({0, self.steps, *range(self.val_every, self.steps, self.val_every)})

    def core_steps(self, include_final=True):
        steps = list(range(self.core_every, self.steps, self.core_every)) if self.core_every > 0 else []
        return steps+([self.steps] if include_final else [])

    def is_milestone(self, step):
        return step % self.milestone_every == 0 or step == self.steps


def load_config(path):
    raw = json.loads(Path(path).read_text())
    if set(raw) != {'model', 'train'}:
        raise ValueError('configuration requires model and train sections')
    mc, tc = NativeModelConfig(**raw['model']), ScalingTrainConfig(**raw['train'])
    if mc.architecture != 'dense':
        raise ValueError('scaling runs support the dense backbone')
    for name in ('global_batch_tokens', 'val_tokens', 'final_bpb_tokens'):
        if getattr(tc, name) % mc.sequence_len:
            raise ValueError(f'{name} must contain whole sequences')
    return mc, tc


def config_hash(mc, tc):
    return hashlib.sha256(json.dumps({'model': asdict(mc), 'train': asdict(tc)}, sort_keys=True).encode()).hexdigest()
