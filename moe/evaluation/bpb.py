"""Validation loss and bits per byte on the six reserved Dolma 3 validation sources (macro average)."""
import math

import torch
import torch.distributed as dist

from moe.data import Rows
from moe.spec import PROTOCOL, VALIDATION_SOURCES


@torch.no_grad()
def evaluate(model, val_dir, token_bytes, device, *, rows_per_source=PROTOCOL.val_rows_per_source, batch=4):
    """rows_per_source=None scores every complete row (the final validation)."""
    rank = dist.get_rank() if dist.is_initialized() else 0
    world = dist.get_world_size() if dist.is_initialized() else 1
    token_bytes = token_bytes.to(device)
    was_training = model.training
    model.eval()
    per_source = {}
    try:
        for source in VALIDATION_SOURCES:
            rows = Rows(f"{val_dir}/{source}.bin", model.config.sequence_len)
            count = len(rows) if rows_per_source is None else min(rows_per_source, len(rows))
            sums = torch.zeros(3, dtype=torch.float64, device=device)  # nats, scored tokens, bytes
            for start in range(rank * batch, count, world * batch):
                block = torch.from_numpy(rows.get(start, min(start + batch, count))).to(device)
                x, y = block[:, :-1], block[:, 1:]
                loss = model(x, y, reduction="none").view(-1).double()
                nbytes = token_bytes[y.reshape(-1)].double()
                sums += torch.stack((loss.sum(), torch.tensor(float(loss.numel()), device=device, dtype=torch.float64),
                                     nbytes.sum()))
                # bits per byte: nats of byte-carrying tokens only (special tokens have zero bytes)
                sums[0] -= (loss * (nbytes == 0)).sum()
            if world > 1:
                dist.all_reduce(sums)
            per_source[source] = dict(rows=count, bpb=(sums[0] / (math.log(2) * sums[2])).item())
    finally:
        model.train(was_training)
    return dict(sources=per_source, macro_bpb=sum(v["bpb"] for v in per_source.values()) / len(per_source))
