"""A stand-in for the official pipeline's rank function (tests only)."""

import json
import os
from pathlib import Path


def ranks(rank, world, request, output):
    import torch
    import torch.distributed as dist

    # Like official.train_rank: the global rank is written into LOCAL_RANK, and a file rendezvous is asked for.
    os.environ.update(RANK=str(rank), LOCAL_RANK=str(rank), WORLD_SIZE=str(world))
    dist.init_process_group(
        "gloo",
        init_method="file:///nonexistent/rendezvous",
        rank=rank,
        world_size=world,
    )
    value = torch.tensor([rank + 1.0])
    dist.all_reduce(value)
    Path(output, f"rank-{rank}.json").write_text(
        json.dumps(
            {
                "rank": dist.get_rank(),
                "sum": value.item(),
                "local_rank": os.environ["LOCAL_RANK"],
            }
        )
    )
    dist.destroy_process_group()
