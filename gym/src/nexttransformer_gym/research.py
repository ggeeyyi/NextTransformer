"""Data parallelism with a fixed global optimizer-update batch.

One machine holds local replicas and reduces them with CUDA copies. Several machines (an operator-run
`torch.distributed` group with one process per node) each hold their local replicas: the global rows are split
over every node's devices in the same order as on one machine with that many devices, each node sums its
replicas' gradients locally, and the node sums are all-reduced across nodes before the optimizer step. Every node
then applies the same update; a fingerprint check after each update restores node 0's state if they ever differ.
"""

import copy
import math
from concurrent.futures import ThreadPoolExecutor
from contextlib import nullcontext

import torch

BUCKET_BYTES = 256 * 1024**2


def mesh():
    """(node, nodes) of an initialized one-process-per-node group, or (0, 1) on one machine."""
    import torch.distributed as dist

    if dist.is_available() and dist.is_initialized() and dist.get_world_size() > 1:
        return dist.get_rank(), dist.get_world_size()
    return 0, 1


def all_reduce_tensors(tensors):
    """Sum tensors across nodes in place, in buckets of one dtype and device."""
    import torch.distributed as dist
    from torch._utils import _flatten_dense_tensors, _unflatten_dense_tensors

    buckets, current, size = [], [], 0
    for tensor in tensors:
        nbytes = tensor.numel() * tensor.element_size()
        if current and (
            size + nbytes > BUCKET_BYTES
            or tensor.dtype != current[0].dtype
            or tensor.device != current[0].device
        ):
            buckets.append(current)
            current, size = [], 0
        current.append(tensor)
        size += nbytes
    if current:
        buckets.append(current)
    for bucket in buckets:
        flat = _flatten_dense_tensors(bucket)
        dist.all_reduce(flat)
        for target, source in zip(bucket, _unflatten_dense_tensors(flat, bucket)):
            target.copy_(source)


def microbatches(batch_size, microbatch_size, device_count):
    """Global row slices; the last slice need not fill every device."""
    capacity = microbatch_size * device_count
    for start in range(0, batch_size, capacity):
        yield start, min(start + capacity, batch_size)


def replica_slices(start, end, device_count):
    chunk = math.ceil((end - start) / device_count)
    for offset in range(start, end, chunk):
        yield offset, min(offset + chunk, end)


class ResearchParallel:
    """Persistent local replicas, using CUDA copies rather than NCCL collectives on one machine.

    Each replica accumulates its share of the global mean. Only the primary
    model owns optimizer state. Explicit copies work inside the existing network
    isolation and also support machines without GPU peer-to-peer access. Across
    machines, only the nodes' primaries meet, in the operator's process group (`mesh`).
    """

    def __init__(self, model, device_count):
        self.models = [model]
        for index in range(1, device_count):
            self.models.append(copy.deepcopy(model).to(f"cuda:{index}"))
        self.node, self.nodes = mesh()
        self.total = device_count * self.nodes
        self.resyncs = 0

    def mine(self, start, end):
        """This node's replica slices of global rows [start, end), as on one machine with `total` devices."""
        first = self.node * len(self.models)
        return list(replica_slices(start, end, self.total))[
            first : first + len(self.models)
        ]

    def backward(self, x, y, row_offset, microbatch_size, step=None):
        batch, count = x.shape[0], len(self.models)
        for model in self.models:
            model.train()
            model.zero_grad(set_to_none=True)
            if step is not None:
                model.set_step(step)
        if self.nodes == 1 and count == 1 and microbatch_size >= batch:
            self.models[0].set_row_offset(row_offset)
            loss = self.models[0](x, y)
            loss.backward()
            return float(loss.detach())

        def run(index, start, end):
            model = self.models[index]
            device = next(model.parameters()).device
            context = (
                torch.cuda.device(device) if device.type == "cuda" else nullcontext()
            )
            with context:
                model.set_row_offset(row_offset + start)
                loss = model(x[start:end].to(device), y[start:end].to(device))
                loss = loss * ((end - start) / batch)
                loss.backward()
                return float(loss.detach())

        value, active = 0.0, set()
        with ThreadPoolExecutor(max_workers=count) as pool:
            for start, end in microbatches(batch, microbatch_size, self.total):
                slices = self.mine(start, end)
                active.update(range(len(slices)))
                # Complete one round before advancing a replica's row identity.
                futures = [pool.submit(run, i, a, b) for i, (a, b) in enumerate(slices)]
                value += sum(f.result() for f in futures)
        with torch.no_grad():
            primary = list(self.models[0].parameters())
            for index in sorted(active - {0}):
                for target, source in zip(primary, self.models[index].parameters()):
                    if not target.requires_grad:
                        continue
                    if target.grad is None or source.grad is None:
                        from .common import GymError

                        raise GymError("numerical_failure", "Missing replica gradient")
                    target.grad.add_(source.grad.to(target.device))
            if self.nodes > 1:
                value = self.reduce_nodes(value)
        for model in self.models:
            model.set_row_offset(row_offset)
        return value

    def reduce_nodes(self, value):
        """Sum the nodes' gradients and losses; a parameter without a gradient on every node keeps none."""
        import torch.distributed as dist

        params = [p for p in self.models[0].parameters() if p.requires_grad]
        device = params[0].device
        present = torch.tensor(
            [p.grad is not None for p in params], dtype=torch.int32, device=device
        )
        dist.all_reduce(present, op=dist.ReduceOp.MAX)
        grads = []
        for param, has in zip(params, present.tolist()):
            if not has:
                continue
            if param.grad is None:  # this node computed no rows this update
                param.grad = torch.zeros_like(param)
            grads.append(param.grad)
        all_reduce_tensors(grads)
        loss = torch.tensor([value], dtype=torch.float64, device=device)
        dist.all_reduce(loss)
        return float(loss[0])

    def synchronize(self, optimizer=None):
        """Publish the one optimizer update before any replica's next forward."""
        if self.nodes > 1:
            self.agree(optimizer)
        state = self.models[0].state_dict()
        for replica in self.models[1:]:
            replica.load_state_dict(state)

    def agree(self, optimizer=None):
        """Every node applies the same update to the same state; if a fingerprint differs, take node 0's."""
        import torch.distributed as dist

        model = self.models[0]
        tensors = [
            t
            for t in model.state_dict().values()
            if torch.is_tensor(t) and t.is_floating_point()
        ]
        device = next(model.parameters()).device
        with torch.no_grad():
            local = torch.stack(
                [t.sum(dtype=torch.float64).to(device) for t in tensors]
                + [t.abs().sum(dtype=torch.float64).to(device) for t in tensors]
            )
            gathered = [torch.empty_like(local) for _ in range(self.nodes)]
            dist.all_gather(gathered, local)
            if all(torch.equal(gathered[0], other) for other in gathered[1:]):
                return
            self.resyncs += 1
            for value in model.state_dict().values():
                if torch.is_tensor(value):
                    broadcast(value, device)
            if optimizer is not None:
                for group in optimizer.param_groups:
                    for param in group["params"]:
                        for value in optimizer.state.get(param, {}).values():
                            if torch.is_tensor(value):
                                broadcast(value, device)

    def evaluate_batch(self, x, labels, sizes, row_offset, microbatch_size):
        """Evaluate one global dev batch across local replicas (and nodes) in parallel."""
        batch, count = x.shape[0], len(self.models)
        for model in self.models:
            model.eval()

        def score(index, start, end):
            model = self.models[index]
            device = next(model.parameters()).device
            context = (
                torch.cuda.device(device) if device.type == "cuda" else nullcontext()
            )
            with context, torch.no_grad():
                model.set_row_offset(row_offset + start)
                logits = model(x[start:end].to(device))
                targets = labels[start:end].to(device).reshape(-1)
                local_sizes = sizes.to(device)
                losses = torch.nn.functional.cross_entropy(
                    logits.float().reshape(-1, logits.size(-1)),
                    targets,
                    reduction="none",
                )
                valid = local_sizes[targets] > 0
                return (
                    float((losses * valid).sum()),
                    int(local_sizes[targets][valid].sum()),
                    int(targets.numel()),
                )

        if self.nodes == 1 and count == 1 and microbatch_size >= batch:
            result = score(0, 0, batch)
            self.models[0].set_row_offset(row_offset)
            return result

        nats, byte_count, tokens = 0.0, 0, 0
        with ThreadPoolExecutor(max_workers=count) as pool:
            for start, end in microbatches(batch, microbatch_size, self.total):
                slices = self.mine(start, end)
                futures = [
                    pool.submit(score, i, a, b) for i, (a, b) in enumerate(slices)
                ]
                for future in futures:
                    part_nats, part_bytes, part_tokens = future.result()
                    nats += part_nats
                    byte_count += part_bytes
                    tokens += part_tokens
        for model in self.models:
            model.set_row_offset(row_offset)
        if self.nodes > 1:
            import torch.distributed as dist

            device = next(self.models[0].parameters()).device
            totals = torch.tensor(
                [nats, byte_count, tokens], dtype=torch.float64, device=device
            )
            dist.all_reduce(totals)
            nats, byte_count, tokens = (
                float(totals[0]),
                int(totals[1]),
                int(totals[2]),
            )
        return nats, byte_count, tokens

    def routing_states(self):
        """Every replica's routing RNG state, in global replica order (a collective across nodes)."""
        local = [model.routing_rng_state() for model in self.models]
        if self.nodes == 1:
            return local
        import torch.distributed as dist

        # Object collectives stage their bytes on the calling thread's current device, which must be the group's.
        device = next(self.models[0].parameters()).device
        gathered = [None] * self.nodes
        with torch.cuda.device(device) if device.type == "cuda" else nullcontext():
            dist.all_gather_object(gathered, local)
        return [state for part in gathered for state in part]

    def restore_routing_states(self, states):
        if len(states) != self.total:
            raise ValueError("Checkpoint replica count differs")
        first = self.node * len(self.models)
        for model, state in zip(self.models, states[first : first + len(self.models)]):
            model.restore_routing_rng(state)


def broadcast(tensor, device):
    """Node 0's value of a tensor, in place; tensors off the group's device travel through it."""
    import torch.distributed as dist

    if tensor.device == device:
        dist.broadcast(tensor, src=0)
        return
    carrier = tensor.to(device)
    dist.broadcast(carrier, src=0)
    tensor.copy_(carrier)


def training_backward(model, x, y, row_offset, microbatch_size, device_count):
    """Convenience entry point for a single update; workers retain their replicas."""
    return ResearchParallel(model, device_count).backward(
        x, y, row_offset, microbatch_size
    )
