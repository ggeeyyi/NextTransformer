"""Inkling REL replacing RoPE in the unchanged shared Dense shell.

For each query/head, r = x W_r and bias(i,j) = sum_a r(i,h,a) E(a,i-j).
E is shared across heads within a layer; distance zero includes self attention.
All causal distances within the configured context are covered. Q/K RMS norm,
1.2 gain, 1/sqrt(head_dim) scaling, value embeddings and output projection are
inherited from the shared shell. REL bias is added after content-score scaling.
"""
import math
import torch
from torch import nn
from nexttransformer.models.backbone import CausalSelfAttention, Linear, norm
from .reference import relative_attention_reference


class RelativeDistanceTable(nn.Module):
    def __init__(self, d_rel, extent, std):
        super().__init__()
        self.std = std
        self.proj = nn.Parameter(torch.empty(d_rel, extent))
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.normal_(self.proj, std=self.std)


class RelativeSelfAttention(CausalSelfAttention):
    def __init__(self, config, layer):
        super().__init__(config, layer)
        self.d_rel = config.option("d_rel")
        self.rel_extent = config.option("rel_extent")
        self.rel_init_std = config.option("rel_init_std")
        self.c_r = Linear(self.n_embd, self.n_head * self.d_rel, bias=False)
        self.rel_logits_proj = RelativeDistanceTable(self.d_rel, self.rel_extent, self.rel_init_std)

    def forward(self, x, ve, cos_sin, window_size, kv_cache):
        if kv_cache is not None:
            raise ValueError("REL currently supports the protocol's full-sequence training/evaluation only")
        b, t, _ = x.shape
        if t > self.rel_extent:
            raise ValueError("REL table does not cover the requested context")
        q = self.c_q(x).view(b, t, self.n_head, self.head_dim)
        k = self.c_k(x).view(b, t, self.n_kv_head, self.head_dim)
        v = self.c_v(x).view(b, t, self.n_kv_head, self.head_dim)
        if ve is not None:
            gate = 3 * torch.sigmoid(self.ve_gate(x[..., :self.ve_gate_channels]))
            v = v + gate.unsqueeze(-1) * ve.view_as(v)
        q, k = norm(q) * 1.2, norm(k) * 1.2
        r = self.c_r(x).view(b, t, self.n_head, self.d_rel)
        table = self.rel_logits_proj.proj.to(r.dtype)
        if q.is_cuda:
            from .kernel import relative_attention
            y = relative_attention(q, k, v, r, table)
        else:
            y = relative_attention_reference(q, k, v, r, table)
        return self.c_proj(y.contiguous().view(b, t, self.n_embd))


def mixer(config, layer):
    return RelativeSelfAttention(config, layer)


@torch.no_grad()
def init(model):
    # HF's released Inkling initializer uses std=0.02 for both added matrices.
    # This is explicit initialization for the REL adaptation, not a claim about
    # Inkling's unpublished pretraining optimizer/initialization recipe.
    for block in model.transformer.h:
        nn.init.normal_(block.attn.c_r.weight, std=block.attn.rel_init_std)
        block.attn.rel_logits_proj.reset_parameters()


def check(config):
    if config.window_pattern != "L":
        raise ValueError("REL requires full causal attention in every layer")
    if type(config.option("d_rel")) is not int or config.option("d_rel") <= 0:
        raise ValueError("d_rel must be a positive integer")
    if type(config.option("rel_extent")) is not int or config.option("rel_extent") < config.sequence_len:
        raise ValueError("rel_extent must cover every causal distance in the context")
    if not math.isfinite(config.option("rel_init_std")) or config.option("rel_init_std") <= 0:
        raise ValueError("rel_init_std must be positive")
