"""Single-machine data parallelism with a fixed global optimizer-update batch."""

import copy
import math
from concurrent.futures import ThreadPoolExecutor
from contextlib import nullcontext

import torch


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
    """Persistent local replicas, using CUDA copies rather than NCCL collectives.

    Each replica accumulates its share of the global mean. Only the primary
    model owns optimizer state. Explicit copies work inside the existing network
    isolation and also support machines without GPU peer-to-peer access.
    """

    def __init__(self, model, device_count):
        self.models = [model]
        for index in range(1, device_count):
            self.models.append(copy.deepcopy(model).to(f"cuda:{index}"))

    def backward(self, x, y, row_offset, microbatch_size, step=None):
        batch, count = x.shape[0], len(self.models)
        for model in self.models:
            model.train()
            model.zero_grad(set_to_none=True)
            if step is not None:
                model.set_step(step)
        if count == 1 and microbatch_size >= batch:
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
            for start, end in microbatches(batch, microbatch_size, count):
                slices = list(replica_slices(start, end, count))
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
        for model in self.models:
            model.set_row_offset(row_offset)
        return value

    def synchronize(self):
        """Publish the one optimizer update before any replica's next forward."""
        state = self.models[0].state_dict()
        for replica in self.models[1:]:
            replica.load_state_dict(state)

    def routing_states(self):
        return [model.routing_rng_state() for model in self.models]

    def restore_routing_states(self, states):
        if len(states) != len(self.models):
            raise ValueError("Checkpoint replica count differs")
        for model, state in zip(self.models, states):
            model.restore_routing_rng(state)


def training_backward(model, x, y, row_offset, microbatch_size, device_count):
    """Convenience entry point for a single update; workers retain their replicas."""
    return ResearchParallel(model, device_count).backward(
        x, y, row_offset, microbatch_size
    )
