"""Full AttnRes with Canon layers.

Parent: attnres (att_000025). attnres_groups equals the layer count, and the Canon-A/C layers of attnres_canon.
"""
import torch
from torch import nn
from torch.nn import functional as F

from nexttransformer.models.backbone import norm
from nexttransformer.models.lm import ReferenceLM
from nexttransformer.models.residual import depth_attention
from nextt_gym.upstream_model import model_config, set_stage, stage_for_step  # noqa: F401


class Canon(nn.Module):
    """Causal depthwise convolution over the last `size` positions, added to its input (Canon layer).

    Allen-Zhu, "Physics of Language Models: Part 4.1" (2025), Canon-A/C placement: on the input of every attention
    and MLP sublayer. The weights start at zero, so the layer starts as the identity.
    """

    def __init__(self, width, size=4):
        super().__init__()
        self.weight = nn.Parameter(torch.empty(width, 1, size))
        self.size = size

    def forward(self, x):
        y = F.conv1d(F.pad(x.transpose(1, 2), (self.size - 1, 0)), self.weight.to(x.dtype), groups=x.shape[-1])
        return x + y.transpose(1, 2)


class CanonAttnResLM(ReferenceLM):
    def __init__(self, config):
        super().__init__(config)
        self.canon = nn.ModuleList(Canon(config.width) for _ in range(2 * config.n_layer))

    @torch.no_grad()
    def init_weights(self):
        super().init_weights()
        for layer in self.canon:
            nn.init.zeros_(layer.weight)

    def _residual_hidden(self, idx):
        """Block AttnRes trunk of ReferenceLM, with a Canon layer on each sublayer input."""
        c = self.config
        x = self._embed_reference(idx)
        starts = set(c.attnres_group_starts)
        blocks, keys, partial = [], [], x
        for k in range(2 * c.n_layer):
            key = norm(partial)
            h = depth_attention(self.depth_queries[k], blocks + [partial], keys + [key])
            self._record("depth_read", sublayer=k, sources=len(blocks) + 1, tokens=idx.numel())
            if k % 2 == 0 and k // 2 in starts:
                blocks.append(partial); keys.append(key); partial = None
            h = self.canon[k](h)
            out = self._attention_branch(h, idx, k // 2) if k % 2 == 0 else self.transformer.h[k // 2].mlp(norm(h))
            partial = out if partial is None else partial + out
        h = depth_attention(self.depth_queries[-1], blocks + [partial], keys + [norm(partial)])
        self._record("depth_read", sublayer=2 * c.n_layer, sources=len(blocks) + 1, tokens=idx.numel())
        return h


def build(config):
    """As nexttransformer.models.lm.build_model, for this subclass; the caller has seeded the generator."""
    c = model_config(config)
    device = torch.get_default_device()
    with torch.device("meta"):
        model = CanonAttnResLM(c)
    if str(device) == "meta":
        return model
    model.to_empty(device=device)
    for module in model.modules():
        if isinstance(module, nn.Linear): module.reset_parameters()
    model.init_weights()
    return model
