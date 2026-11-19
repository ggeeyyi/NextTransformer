"""Multi-node research: one Gym worker per node, in a process group the operator's executor describes.

`ResearchParallel` (research.py) spans nodes when the worker runs inside an initialized one-process-per-node
`torch.distributed` group: the update's global rows split over every node's devices exactly as on one machine with
that many devices. An executor starts, on each node, with that node's share of the job's GPU group visible
(CUDA_VISIBLE_DEVICES) and NEXTTRANSFORMER_GYM_MESH set to

    {"node": k, "nodes": n, "master": "<node 0 address>", "port": p, "timeout": seconds, "network": "deny"|"allow"}

* `python -m nexttransformer_gym.mesh check`: forms the group and prints one JSON line with the measured all-reduce
  bandwidth (a trusted check, before the job is reported ready);
* `python -m nexttransformer_gym.mesh worker`: forms the group on cuda:0 with every connection made eagerly, runs
  each collective the research worker uses once, then (network `deny`) refuses new IPv4/IPv6 sockets in every
  thread, and only then reads the worker request from stdin and loads candidate code. The worker keeps the
  connections to its own job's nodes and opens no others.

Every node runs the same request; node 0 alone writes the checkpoint (worker.py). The executor gathers node 0's
events and stops the other nodes' workers when a node fails.
"""

import datetime
import json
import os
import sys
import time

ENV = "NEXTTRANSFORMER_GYM_MESH"
# Connect every NCCL algorithm when the group forms, so nothing connects after the socket filter; NCCL's RAS service
# keeps sockets of its own, and torch's heartbeat monitor and timeout dumps are not needed.
CONNECT_EAGERLY = {
    "NCCL_RUNTIME_CONNECT": "0",
    "NCCL_RAS_ENABLE": "0",
    "TORCH_NCCL_ENABLE_MONITORING": "0",
    "TORCH_NCCL_DUMP_ON_TIMEOUT": "0",
}
FIELDS = {"node", "nodes", "master", "port", "timeout", "network"}


def spec(value):
    """A mesh description, checked."""
    value = json.loads(value) if isinstance(value, str) else dict(value)
    backend = value.pop("backend", "nccl")  # gloo serves CPU tests
    if set(value) != FIELDS:
        raise ValueError(f"a mesh needs exactly {sorted(FIELDS)}")
    nodes, node = int(value["nodes"]), int(value["node"])
    if not (nodes > 1 and 0 <= node < nodes and int(value["port"]) > 0):
        raise ValueError("invalid mesh shape")
    if value["network"] not in ("deny", "allow") or backend not in ("nccl", "gloo"):
        raise ValueError("invalid mesh network or backend")
    return dict(value, node=node, nodes=nodes, backend=backend)


def join(mesh):
    """Join the job's one-process-per-node group (on this node's first GPU for NCCL)."""
    os.environ.update(CONNECT_EAGERLY)
    import torch
    import torch.distributed as dist

    options = {}
    if mesh["backend"] == "nccl":
        torch.cuda.set_device(0)
        options["device_id"] = torch.device("cuda", 0)
    dist.init_process_group(
        mesh["backend"],
        init_method=f"tcp://{mesh['master']}:{int(mesh['port'])}",
        rank=mesh["node"],
        world_size=mesh["nodes"],
        timeout=datetime.timedelta(seconds=int(mesh["timeout"])),
        **options,
    )
    return torch, dist


def warm(torch, dist, sizes=(1, 1 << 10, 1 << 20, 1 << 26)):
    """Run each collective the research worker uses, small and large, so every connection exists."""
    world = dist.get_world_size()
    cuda = dist.get_backend() == "nccl"
    device = "cuda" if cuda else "cpu"
    dtypes = (torch.float32, torch.float64, torch.int32) + (
        (torch.bfloat16,) if cuda else ()
    )
    for numel in sizes:
        for dtype in dtypes:
            value = torch.ones(numel, dtype=dtype, device=device)
            dist.all_reduce(value)
            dist.all_reduce(value, op=dist.ReduceOp.MAX)
            dist.broadcast(value, src=0)
            dist.all_gather([torch.empty_like(value) for _ in range(world)], value)
            del value
    gathered = [None] * world
    if cuda:
        with torch.cuda.device(0):
            dist.all_gather_object(gathered, {"node": dist.get_rank()})
    else:
        dist.all_gather_object(gathered, {"node": dist.get_rank()})
    dist.barrier()
    if cuda:
        torch.cuda.synchronize()
        torch.cuda.empty_cache()


def prepare(mesh, sizes=None):
    """Join, warm up and (network `deny`) refuse new network sockets in every thread."""
    from .netdeny import AF_INET, AF_INET6, deny_sockets

    torch, dist = join(mesh)
    if sizes is None and mesh["backend"] != "nccl":
        sizes = (1, 1 << 10)  # a CPU group (tests) needs no large buffers
    warm(torch, dist, **({"sizes": sizes} if sizes else {}))
    if mesh["network"] == "deny":
        deny_sockets([AF_INET, AF_INET6], all_threads=True)
    return torch, dist


def check(mesh):
    """Form the group and print the measured all-reduce bus bandwidth as one JSON line."""
    torch, dist = join(mesh)
    cuda = mesh["backend"] == "nccl"
    warm(torch, dist, **({} if cuda else {"sizes": (1, 1 << 10)}))
    value = torch.ones(
        1 << 28 if cuda else 1 << 16,
        dtype=torch.float32,
        device="cuda" if cuda else "cpu",
    )
    seconds = []
    for _ in range(3):
        if cuda:
            torch.cuda.synchronize()
        start = time.perf_counter()
        dist.all_reduce(value)
        if cuda:
            torch.cuda.synchronize()
        seconds.append(time.perf_counter() - start)
    best, world = min(seconds), dist.get_world_size()
    gib = value.numel() * 4 / 1024**3
    print(
        json.dumps(
            {
                "node": dist.get_rank(),
                "nodes": world,
                "all_reduce_gib": gib,
                "seconds": round(best, 4),
                "bus_gbit_per_s": round(2 * (world - 1) / world * 8 * gib / best, 1),
            }
        ),
        flush=True,
    )
    dist.destroy_process_group()
    return 0


def worker(mesh):
    """One node's research worker: the group first, then the network filter, then the request and candidate code."""
    import traceback

    prepare(mesh)
    from .worker import main

    code = 0
    try:
        main()
    except BaseException:
        traceback.print_exc()
        code = 1
    # Leave without tearing the group down: a peer that already left would make that wait.
    sys.stdout.flush()
    sys.stderr.flush()
    os._exit(code)


def main(argv=None):
    argv = list(sys.argv[1:] if argv is None else argv)
    if argv[:1] not in (["check"], ["worker"]):
        print("usage: python -m nexttransformer_gym.mesh check|worker", file=sys.stderr)
        return 2
    mesh = spec(os.environ.pop(ENV))
    return check(mesh) if argv[0] == "check" else worker(mesh)


if __name__ == "__main__":
    sys.exit(main())
