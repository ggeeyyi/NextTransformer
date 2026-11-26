"""Call the pinned upstream BPB and CORE implementations without redefinitions."""
import os
import time
from pathlib import Path
import torch

from nanochat.loss_eval import evaluate_bpb as upstream_bpb
from nexttransformer.evaluation.core import score_core
from nexttransformer.evaluation.metrics import Metrics, verify_core
from nexttransformer.runtime.process import synchronize
from nexttransformer.runtime.data import BestFitStream


class CountedModel(torch.nn.Module):
    def __init__(self, model, metrics):
        super().__init__()
        self.model, self.metrics = model, metrics
        self.config = model.config
        self.max_seq_len = model.max_seq_len

    def get_device(self): return self.model.get_device()

    def forward(self, *args, **kwargs):
        result = self.model(*args, **kwargs)
        self.metrics.add(self.model.last_stats)
        return result


@torch.no_grad()
def evaluate_bpb(model, tokenizer, data_dir, resources, tokens, batch, rank, world, device, split='val'):
    denominator = batch*world*model.config.sequence_len
    if tokens % denominator: raise ValueError('BPB token budget must divide the distributed evaluation batch')
    stream = BestFitStream(data_dir, tokenizer, split, rank, world, model.config.sequence_len)
    token_bytes = torch.load(Path(resources)/'tokenizer/token_bytes.pt', map_location=device, weights_only=True)
    metrics = Metrics(model.config.routing_steps, model.config.architecture == 'circle')
    was_training = model.training
    model.eval(); synchronize(device); start = time.perf_counter()
    try:
        value = upstream_bpb(CountedModel(model, metrics), stream.batches(batch, device), tokens//denominator, token_bytes)
        result = metrics.result(device)
        synchronize(device)
        if result['input_tokens'] != tokens: raise RuntimeError('BPB input token count drift')
        return {**result, 'val_bpb' if split == 'val' else 'train_bpb':value,
                'seconds':time.perf_counter()-start, 'split':split, 'protocol':'nanochat_bos_bestfit'}
    finally:
        model.train(was_training)


@torch.no_grad()
def evaluate_core(model, tokenizer, resources, device, max_per_task=-1, *, core_max_tokens=None):
    verify_core(resources)
    if Path(os.environ['NANOCHAT_BASE_DIR']).resolve() != Path(resources).resolve():
        raise ValueError('NANOCHAT_BASE_DIR must identify the verified native resources')
    metrics = Metrics(model.config.routing_steps, model.config.architecture == 'circle')
    was_training = model.training
    model.eval(); synchronize(device); start = time.perf_counter()
    try:
        result = score_core(model, lambda m: CountedModel(m, metrics), tokenizer, device,
                            max_per_task=max_per_task, max_tokens=core_max_tokens)
        result.update(metrics.result(device))
        synchronize(device)
        return {**result, 'seconds':time.perf_counter()-start, 'full_core':max_per_task <= 0, 'entries':22}
    finally:
        model.train(was_training)
