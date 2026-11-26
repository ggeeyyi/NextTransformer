"""Identity Hyper-Connections with the Hy4 read, write and output gates.

A flat, RMS-normalized token state produces one read and one write scalar per stream.
Each sublayer reads all streams, then adds its gated update to the original streams.
The last block merges them with a separate learned read before the reference output norm.
"""

import math

import torch
from torch import nn
import torch.nn.functional as F

from moe.model import MoELM


class StreamMapping(nn.Module):
    """Project the full token state directly to read logits and, for sublayers, write logits."""

    def __init__(self, config, *, read_only=False):
        super().__init__()
        self.streams = config.option("streams")
        self.maps = 1 if read_only else 2
        self.norm_eps = config.norm_eps
        self.gate_eps = config.option("gate_eps")
        self.init_std = config.option("init_std")
        self.gate_init = config.option("gate_init")
        self.projection = nn.Parameter(torch.empty(self.streams * config.width, self.maps * self.streams))
        self.scales = nn.Parameter(torch.empty(self.maps))
        self.bias = nn.Parameter(torch.empty(self.maps * self.streams))

    @torch.no_grad()
    def reset_parameters(self):
        nn.init.normal_(self.projection, std=self.init_std)
        self.scales.fill_(self.gate_init)
        self.bias.zero_()
        self.bias[:self.streams].fill_(-math.log(self.streams - 1))

    def forward(self, streams):
        flat = streams.flatten(-2)
        normed = F.rms_norm(flat, (flat.size(-1),), eps=self.norm_eps)
        logits = normed @ self.projection.to(normed.dtype)
        # Keep the small coefficient calculations precise under BF16 autocast.
        logits = logits.to(torch.promote_types(logits.dtype, torch.float32))
        logits = logits.unflatten(-1, (self.maps, self.streams))
        gates = torch.sigmoid(logits * self.scales[:, None] + self.bias.view(self.maps, self.streams))
        pre = gates[..., 0, :] + self.gate_eps
        x = torch.einsum("...s,...sd->...d", pre.to(streams.dtype), streams)
        post = 2 * gates[..., 1, :] + self.gate_eps if self.maps == 2 else None
        return x, post


def write(streams, update, post):
    return torch.addcmul(streams, post.to(update.dtype).unsqueeze(-1), update.unsqueeze(-2))


class IdentityHyperConnectedBlock(nn.Module):
    """Wrap attention and MoE separately so the FFN gates see the updated streams."""

    def __init__(self, block, config, layer):
        super().__init__()
        self.block = block
        self.streams = config.option("streams")
        self.first = layer == 0
        self.attn_mapping = StreamMapping(config)
        self.ffn_mapping = StreamMapping(config)
        self.head_mapping = StreamMapping(config, read_only=True) if layer == config.layers - 1 else None

    def forward(self, streams, cos_sin, masks):
        if self.first:
            streams = streams.unsqueeze(-2).expand(-1, -1, self.streams, -1)
        block = self.block
        x, post = self.attn_mapping(streams)
        update = block.attn(block.attn_gated(block.attn_norm(x)), cos_sin, masks)
        streams = write(streams, update, post)
        x, post = self.ffn_mapping(streams)
        update = block.ffn(block.mlp_gated(block.mlp_norm(x)))
        streams = write(streams, update, post)
        if self.head_mapping is not None:
            x, _ = self.head_mapping(streams)
            return x
        return streams


block = IdentityHyperConnectedBlock
# Measure the complete multi-stream path, including replication and the learned final read.
model = MoELM


def init(lm):
    for mapping in lm.modules():
        if isinstance(mapping, StreamMapping):
            mapping.reset_parameters()


def check(config):
    streams = config.option("streams")
    if type(streams) is not int or streams < 2:
        raise ValueError("streams must be an integer of at least two")
    for name in ("gate_eps", "init_std", "gate_init"):
        value = config.option(name)
        if type(value) not in (int, float) or not math.isfinite(value) or value <= 0:
            raise ValueError(f"{name} must be finite and positive")
