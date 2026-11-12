"""Native Muon/AdamW groups with nanochat's weight-decay rule applied to the model actually trained.

The native comparison pins Muon weight decay to the d24 model. nanochat scales it inversely with the
matrix parameter count (blocks plus output head), which is the rule used here; for d24 the two agree.
"""
from nexttransformer.native.optimizer import D12_SCALING_PARAMS, build_optimizer as native_optimizer


def scaling_params(model):
    return sum(p.numel() for p in model.transformer.h.parameters())+model.lm_head.weight.numel()


def muon_weight_decay(model, config):
    return config.weight_decay*config.batch_scale*D12_SCALING_PARAMS/scaling_params(model)


def build_optimizer(model, config):
    optimizer = native_optimizer(model, config)
    decay = muon_weight_decay(model, config)
    for group in optimizer.param_groups:
        if group['kind'] == 'muon':
            group['weight_decay'] = group['initial_weight_decay'] = decay
    return optimizer
