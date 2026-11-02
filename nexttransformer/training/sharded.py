"""Muon/AdamW state sharding for gradients already averaged by DDP."""
import math
import torch
import torch._inductor.config
import torch.distributed as dist
from nanochat.optim import MuonAdamW

class ExperimentOptimizer(MuonAdamW):
    def step(self):
        # The extra conditioner/halter AdamW tensors add a finite set of shapes
        # to upstream's shape-specialized fused kernel. Its default limit of 8
        # is too small for LoopFormer. Scope the higher cap to optimizer.step;
        # keep upstream kernels and the rest of the process configuration intact.
        #
        # Every Inductor kernel of the fused steps keeps its single heuristic
        # launch configuration. By default each process benchmarks candidates
        # for some reductions, and near-equal timings pick different ones: after
        # a four-node resume one Muon reduction ran with R0_BLOCK 1024 instead
        # of 2048 (EXPERIMENTS record 031). The candidates come from pointwise
        # autotuning, from the persistent alternative that coalescing analysis
        # adds to looped reductions, and from scaling R0_BLOCK down at run time
        # by register use; all three are off. The compiled math is unchanged.
        with torch._dynamo.config.patch(recompile_limit=64), torch._inductor.config.patch(
                {"triton.autotune_pointwise": False, "triton.coalesce_tiling_analysis": False,
                 "dynamic_scale_rblock": False}):
            return super().step()

    # DDP performs reduction before clipping. Preserve nanochat's state sharding
    # and parameter gathers, without a second gradient reduction.
    def _reduce_adamw(self, group, world_size):
        rank = dist.get_rank() if world_size > 1 else 0
        infos = {}
        for p in group['params']:
            if p.grad is None:
                raise RuntimeError('missing gradient in the fixed training graph')
            replicated = world_size == 1 or p.numel() < 1024 or p.shape[0] % world_size != 0
            size = p.shape[0] // world_size if not replicated else 0
            grad = p.grad if replicated else p.grad[rank*size:(rank+1)*size].contiguous()
            infos[p] = dict(future=None, grad_slice=grad, is_small=replicated)
        return dict(param_infos=infos)

    def _reduce_muon(self, group, world_size):
        if world_size == 1:
            return super()._reduce_muon(group, world_size)
        params = group['params']
        rank = dist.get_rank()
        chunk_size = math.ceil(len(params)/world_size)
        p = params[0]
        chunk = torch.zeros(chunk_size, *p.shape, device=p.device, dtype=p.dtype)
        for i, param in enumerate(params[rank*chunk_size:(rank+1)*chunk_size]):
            chunk[i].copy_(param.grad)
        stack = torch.empty(chunk_size*world_size, *p.shape, device=p.device, dtype=p.dtype)
        return dict(future=None, grad_chunk=chunk, stacked_grads=stack, chunk_size=chunk_size)
