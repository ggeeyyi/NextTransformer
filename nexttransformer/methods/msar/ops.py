"""Auditable MSAR arithmetic; values use [batch, time, stream, hidden] layout."""

import torch
from torch import nn
from torch.nn import functional as F


def _cuda_kernel_supported(value):
    # Keep CPU, FakeTensor cost audits and non-BF16 execution on the reference.
    return type(value) is torch.Tensor and value.is_cuda and value.dtype == torch.bfloat16


def coefficients(value):
    return value.to(torch.promote_types(value.dtype, torch.float32))


class TemporalAugmentation(nn.Module):
    """Independent zero-initialized causal K4/8/12 depthwise taps, plus the raw source."""

    def __init__(self, width):
        super().__init__()
        self.taps = nn.ParameterList(nn.Parameter(torch.empty(width, 1, k)) for k in (4, 8, 12))
        self.reset_parameters()

    @torch.no_grad()
    def reset_parameters(self):
        for weight in self.taps:
            nn.init.zeros_(weight)

    def forward(self, raw):
        if _cuda_kernel_supported(raw):
            from .kernels import temporal
            return temporal(raw, self.taps)
        channels = raw.transpose(1, 2)
        branches = [raw]
        for weight in self.taps:
            # Left padding produces exactly T outputs; no discarded right-tail computation.
            padded = F.pad(channels, (weight.shape[-1] - 1, 0))
            branches.append(F.conv1d(padded, weight.to(raw.dtype), groups=raw.shape[-1]).transpose(1, 2))
        return torch.stack(branches, dim=2)


class DepthRead(nn.Module):
    """Four pseudo-queries and scalar-per-stream HPre, followed by the shell pre-norm."""

    def __init__(self, width):
        super().__init__()
        self.query = nn.Parameter(torch.empty(4, width))
        self.proj_pre = nn.Parameter(torch.empty(4 * width, 4))
        self.alpha_pre = nn.Parameter(torch.empty(1))
        self.bias_pre = nn.Parameter(torch.empty(4))
        self.reset_parameters()

    @torch.no_grad()
    def reset_parameters(self):
        nn.init.zeros_(self.query)
        nn.init.normal_(self.proj_pre, std=0.006)
        nn.init.constant_(self.alpha_pre, 0.01)
        nn.init.zeros_(self.bias_pre)

    def forward(self, streams):
        flat = streams.flatten(-2)
        mean_square = flat.square().mean(-1, keepdim=True)
        # Substitute BEFORE rsqrt: masking an infinite rsqrt afterwards gives NaN gradients at zero.
        safe_square = torch.where(mean_square > 0, mean_square, torch.ones_like(mean_square))
        logits = (flat @ self.proj_pre.to(flat.dtype)) * safe_square.rsqrt()
        gates = torch.sigmoid(logits * self.alpha_pre.to(flat.dtype) + self.bias_pre.to(flat.dtype))
        return (gates.unsqueeze(-1) * streams).sum(-2)


def source_key(source):
    key = coefficients(source)
    return key * torch.rsqrt(key.square().mean(-1, keepdim=True) + 1e-6)


def completed_read(queries, sources, keys):
    """Batch all consumers in a block; no stacked copy of the historical hidden values."""
    stacked_keys = torch.stack(keys)
    with torch.autocast(device_type=stacked_keys.device.type, enabled=False):
        logits = torch.einsum("qnk,cbtnk->qcbtn", queries.to(stacked_keys.dtype), stacked_keys)
    # Expanded logsumexp stays inside the public FLOPs ledger's registered operators.
    maximum = logits.amax(1, keepdim=True).detach()
    exponents = (logits - maximum).exp()
    denominator = exponents.sum(1, keepdim=True)
    weights = (exponents / denominator).to(sources[0].dtype)
    lse = (maximum + denominator.log()).squeeze(1)
    if _cuda_kernel_supported(sources[0]) and all(s.is_contiguous() for s in sources):
        from .kernels import completed_values
        return completed_values(weights, sources), lse
    value = weights[:, 0].unsqueeze(-1) * sources[0]
    for index, source in enumerate(sources[1:], 1):
        value = value + weights[:, index].unsqueeze(-1) * source
    return value, lse


def merge_partial(completed, completed_lse, partial, partial_key, query):
    with torch.autocast(device_type=partial_key.device.type, enabled=False):
        logit = torch.einsum("nk,btnk->btn", query.to(partial_key.dtype), partial_key)
    weights = torch.stack((completed_lse, logit)).softmax(0).to(partial.dtype)
    return weights[0].unsqueeze(-1) * completed + weights[1].unsqueeze(-1) * partial
