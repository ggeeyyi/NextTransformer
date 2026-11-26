"""Training-data-free synthetic checks before checkpoint CORE evaluation."""
from nexttransformer.runtime.site import bootstrap_runtime
bootstrap_runtime()
import json
import torch
import torch.distributed as dist
from torch.nn import functional as F
from nexttransformer.runtime.process import initialize
from nexttransformer.methods.inkling_rel.kernel import relative_attention
from nanochat.flash_attention import USE_FA3

rank, world, device = initialize(42)
assert USE_FA3 and world == 32
results = []
with torch.no_grad():
    for t in (2048, 2049, 5835):
        q, k, v = [torch.randn(1, t, 1, 128, device=device, dtype=torch.bfloat16) for _ in range(3)]
        r = torch.randn(1, t, 1, 4, device=device, dtype=torch.bfloat16) * .1
        table = torch.randn(4, 2048, device=device, dtype=torch.bfloat16) * .1
        effective = F.pad(table, (0, max(0, t - 2048)))
        actual = relative_attention(q, k, v, r, effective)
        distance = torch.arange(t, device=device)[:, None] - torch.arange(t, device=device)[None, :]
        rel = (r.float() @ table.float()).transpose(1, 2)
        bias = rel.gather(-1, distance.clamp(0, 2047)[None, None])
        bias.masked_fill_((distance < 0) | (distance >= 2048), 0.)
        score = q.transpose(1, 2).float() @ k.transpose(1, 2).float().transpose(-1, -2) * 128**-.5 + bias
        expected = (score.masked_fill(distance < 0, -torch.inf).softmax(-1) @ v.transpose(1, 2).float()).transpose(1, 2)
        relative = (actual.float() - expected).norm() / expected.norm()
        assert torch.isfinite(actual).all() and relative < .012
        results.append(dict(length=t,relative_error=float(relative)))
        del q, k, v, r, table, effective, actual, expected, distance, rel, bias, score
dist.barrier()
if rank == 0:
    print(json.dumps(dict(status='passed',world=world,checks=results)), flush=True)
dist.destroy_process_group()
