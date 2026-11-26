"""Optional host storage for large floating-point tensors saved by autograd."""
from contextlib import nullcontext

import torch


def _pack(tensor):
    if not (tensor.is_cuda and tensor.dtype in (torch.bfloat16, torch.float32) and tensor.nbytes >= 16 * 2**20):
        return tensor.detach()
    # empty_like preserves non-overlapping dense strides, including transposed
    # projection weights and attention heads. Keep other views on the GPU.
    host = torch.empty_like(tensor, device="cpu", pin_memory=True)
    if host.stride() != tensor.stride():
        return tensor.detach()
    host.copy_(tensor, non_blocking=True)
    ready = torch.cuda.Event()
    ready.record(torch.cuda.current_stream(tensor.device))
    return tensor.device, host, ready


def _unpack(saved):
    if isinstance(saved, torch.Tensor):
        return saved
    device, host, ready = saved
    torch.cuda.current_stream(device).wait_event(ready)
    return host.to(device, non_blocking=True)


def activation_storage(offload):
    """Keep saved values and strides intact, with no forward recomputation."""
    if offload:
        return torch.autograd.graph.saved_tensors_hooks(_pack, _unpack)
    return nullcontext()
