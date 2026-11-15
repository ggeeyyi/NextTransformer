"""Native Muon/AdamW mathematics with explicit MoE gradient ownership.

The initial adapter keeps optimizer states local (unsharded). It does not reuse
an optimizer that assumes every parameter is replicated on the world group.
Do not also wrap this model in DDP: this optimizer owns gradient averaging.
"""
from collections import defaultdict
from contextlib import nullcontext
import math

import torch
import torch.distributed as dist
from nanochat.optim import MuonAdamW
from nexttransformer.native.optimizer import D12_SCALING_PARAMS


class MoEOptimizer(MuonAdamW):
    def __init__(self, groups, pg_collection=None):
        super().__init__(groups)
        self.pg_collection = pg_collection

    @torch.no_grad()
    def synchronize_gradients(self):
        distributed = dist.is_initialized() and dist.get_world_size() > 1
        if distributed and self.pg_collection is None:
            raise RuntimeError("multi-rank MoE requires explicit dense/expert process groups")
        for group in self.param_groups:
            for p in group["params"]:
                if p.grad is None:
                    raise RuntimeError("missing gradient in the MoE graph")
                if distributed:
                    pg = self.pg_collection.expt_dp if group["expert"] else self.pg_collection.dp_cp
                    if pg.size() > 1:
                        dist.all_reduce(p.grad, group=pg)
                    # EP dispatch backward sums contributions from all EP sample owners.
                    divisor = pg.size() * (self.pg_collection.ep.size() if group["expert"] else 1)
                    p.grad.div_(divisor)

    @torch.no_grad()
    def step(self):
        self.synchronize_gradients()
        # Reuse the pinned native update kernels with their single-owner path.
        # CPU experiments execute eagerly; CUDA retains the native compiled math.
        cpu = all(p.device.type == "cpu" for g in self.param_groups for p in g["params"])
        stance = torch.compiler.set_stance("force_eager") if cpu else nullcontext()
        with stance:
            gathers = []
            for group in self.param_groups:
                if group["kind"] == "adamw":
                    info = self._reduce_adamw(group, 1)
                    self._compute_adamw(group, info, gathers, 0, 1)
                else:
                    info = self._reduce_muon(group, 1)
                    self._compute_muon(group, info, gathers, 0)
            self._finish_gathers(gathers)


def build_optimizer(model, config, *, pg_collection=None):
    """Keep the native groups; route only router matrices to AdamW explicitly."""
    recipe = model.moe_recipe
    bscale = math.sqrt(config.global_batch_tokens / 524288)
    dscale = (recipe.width / 768) ** -0.5
    counts = recipe.counts()
    # Logical global count, not the local EP shard size.
    denominator = counts["block_total"] + counts["embedding_and_head"] // 2
    decay = config.weight_decay * bscale * D12_SCALING_PARAMS / denominator
    groups = []

    def adam(params, lr, betas, wd):
        params = [p for p in params if p.requires_grad]
        if params:
            groups.append(dict(kind="adamw", expert=False, params=params, lr=lr, initial_lr=lr,
                               betas=betas, eps=1e-10, weight_decay=wd))

    adam(model.lm_head.parameters(), config.unembedding_lr * bscale * dscale, (.8, .96), .01)
    adam(model.transformer.wte.parameters(), config.embedding_lr * bscale * dscale, (.8, .995), .001)
    adam([model.resid_lambdas], config.scalar_lr * bscale * .01, (.8, .95), .05)
    adam([model.x0_lambdas], config.scalar_lr * bscale, (.96, .95), 0.)
    adam([model.smear_gate.weight, model.smear_lambda, model.backout_lambda], .2, (.8, .95), 0.)
    routers = [p for name, p in model.named_parameters() if ".router." in name]
    adam(routers, config.adamw_lr * bscale, (.9, .95), 0.)
    allocated = {id(p) for g in groups for p in g["params"]}
    matrices = defaultdict(list)
    scalars = []
    for p in model.parameters():
        if not p.requires_grad or id(p) in allocated:
            continue
        if p.ndim == 2:
            expert = not getattr(p, "allreduce", True)
            matrices[(expert, tuple(p.shape))].append(p)
        else:
            scalars.append(p)
    adam(scalars, config.adamw_lr * bscale, (.9, .95), 0.)
    for (expert, shape), params in sorted(matrices.items()):
        groups.append(dict(kind="muon", expert=expert, params=params, lr=config.muon_lr * bscale,
                           initial_lr=config.muon_lr * bscale, momentum=.95, ns_steps=5, beta2=.9,
                           weight_decay=decay, initial_weight_decay=decay))
    names = {id(p): n for n, p in model.named_parameters() if p.requires_grad}
    ids = [id(p) for g in groups for p in g["params"]]
    if len(ids) != len(set(ids)) or set(ids) != set(names):
        raise RuntimeError("optimizer must cover each trainable parameter exactly once")
    for g in groups:
        g["parameter_names"] = [names[id(p)] for p in g["params"]]
    return MoEOptimizer(groups, pg_collection)
