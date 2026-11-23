import torch
from torch import nn
from torch.nn import functional as F

from nexttransformer.models.backbone import norm
from nexttransformer.models.lm import ReferenceLM
from nexttransformer.models.residual import depth_attention

from .memory import NgramMemory


class Canon(nn.Module):
    """Causal depthwise convolution over the last `size` positions, added to its input; starts as the identity."""

    def __init__(self, width, size=4):
        super().__init__()
        self.weight = nn.Parameter(torch.empty(width, 1, size))
        self.size = size

    def forward(self, x):
        y = F.conv1d(F.pad(x.transpose(1, 2), (self.size - 1, 0)), self.weight.to(x.dtype), groups=x.shape[-1])
        return x + y.transpose(1, 2)


def _drawn_by_init_weights():
    """Stands in for nn.Embedding.reset_parameters; init_weights draws the embeddings."""


class HydraLM(ReferenceLM):
    """Full AttnRes trunk (one group per block) with Canon-A/C on every sublayer input."""

    def __init__(self, config):
        super().__init__(config)
        self.depth_queries = nn.Parameter(torch.empty(2 * config.n_layer + 1, config.width))
        self.canon = nn.ModuleList(Canon(config.width) for _ in range(2 * config.n_layer))
        self.ngram_inject = config.option("ngram_inject")
        if config.option("value_embeds") == "none":
            # Drop the shell's per-layer value-embedding tables and gates (their parameters go to the memory).
            for key in [k for k in self.value_embeds.keys() if k.isdigit()]:
                del self.value_embeds[key]
            for block in self.transformer.h:
                block.attn.ve_gate = None
        self.memory = NgramMemory(config) if len(config.option("ngram_orders")) else None
        if self.memory is not None:
            for key, table in self.memory.make_tables().items():
                self.value_embeds[key] = table
        for module in self.modules():
            if isinstance(module, nn.Embedding):
                module.reset_parameters = _drawn_by_init_weights

    def _ordinary_hidden(self, idx):
        c = self.config
        x = self._embed_reference(idx)
        blocks, keys, partial = [], [], x
        if self.memory is not None:
            mem = self.memory(idx, self.value_embeds, x.dtype)
            if self.ngram_inject == "embed":
                partial = x + mem
            else:
                blocks.append(mem); keys.append(norm(mem))
        for k in range(2 * c.n_layer):
            key = norm(partial)
            h = depth_attention(self.depth_queries[k], blocks + [partial], keys + [key])
            self._record("depth_read", sublayer=k, sources=len(blocks) + 1, tokens=idx.numel())
            if k % 2 == 0:
                blocks.append(partial); keys.append(key); partial = None
            h = self.canon[k](h)
            if k % 2 == 0:
                out = self._attention_branch(h, idx, k // 2)
            else:
                out = self.transformer.h[k // 2].mlp(norm(h))
            partial = out if partial is None else partial + out
        h = depth_attention(self.depth_queries[-1], blocks + [partial], keys + [norm(partial)])
        self._record("depth_read", sublayer=2 * c.n_layer, sources=len(blocks) + 1, tokens=idx.numel())
        return h, None
