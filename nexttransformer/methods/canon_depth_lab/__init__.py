"""Causal Canon layers with independently selectable depth and feature mixing.

The control exactly follows the served AttnRes-Canon computation. Extensions keep
all training behavior in the native ReferenceLM and use ordinary PyTorch ops.
"""
import torch
from torch import nn
from torch.nn import functional as F
from nanochat.flash_attention import flash_attn
from nexttransformer.models.backbone import CausalSelfAttention, Linear, apply_rotary_emb, norm
from nexttransformer.models.lm import ReferenceLM
from nexttransformer.models.residual import depth_attention


class Canon(nn.Module):
    def __init__(self, width, size):
        super().__init__()
        self.size = size
        self.weight = nn.Parameter(torch.empty(width, 1, size))

    def forward(self, x):
        y = F.conv1d(F.pad(x.transpose(1, 2), (self.size - 1, 0)),
                     self.weight.to(x.dtype), groups=x.shape[-1])
        return x + y.transpose(1, 2)


class CanonAttention(CausalSelfAttention):
    def __init__(self, config, layer):
        super().__init__(config, layer)
        self.canon_qkv = nn.ModuleList(Canon(self.n_embd, config.option('canon_size')) for _ in range(3))
        self.output_gate = Linear(config.width, config.heads, bias=False) if config.option('head_gate') else None

    def forward(self, x, ve, cos_sin, window_size, kv_cache):
        if kv_cache is not None:
            raise NotImplementedError('training and full-sequence evaluation only')
        b, t, c = x.size()
        q, k, v = (layer(z) for layer, z in zip(self.canon_qkv, (self.c_q(x), self.c_k(x), self.c_v(x))))
        q = q.view(b, t, self.n_head, self.head_dim)
        k = k.view(b, t, self.n_kv_head, self.head_dim)
        v = v.view(b, t, self.n_kv_head, self.head_dim)
        if ve is not None:
            ve = ve.view(b, t, self.n_kv_head, self.head_dim)
            v = v + 3 * torch.sigmoid(self.ve_gate(x[..., :self.ve_gate_channels])).unsqueeze(-1) * ve
        cos, sin = cos_sin
        q, k = apply_rotary_emb(q, cos, sin), apply_rotary_emb(k, cos, sin)
        q, k = norm(q) * 1.2, norm(k) * 1.2
        y = flash_attn.flash_attn_func(q, k, v, causal=True, window_size=window_size)
        if self.output_gate is not None:
            y = y * self.output_gate(x).sigmoid().unsqueeze(-1)
        return self.c_proj(y.contiguous().view(b, t, -1))


class CanonMLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.kind = config.option('ffn_kind')
        hidden = config.mlp_width if self.kind == 'relu2' else (2 * config.mlp_width // 3 // 128) * 128
        projected = hidden if self.kind == 'relu2' else 2 * hidden
        self.c_fc = Linear(config.width, projected, bias=False)
        self.c_proj = Linear(hidden, config.width, bias=False)
        self.canon = Canon(projected, config.option('canon_size'))

    def forward(self, x):
        z = self.canon(self.c_fc(x))
        if self.kind == 'relu2':
            y = F.relu(z).square()
        else:
            gate, value = z.chunk(2, dim=-1)
            y = F.silu(gate) * value
        return self.c_proj(y)


def _drawn_by_init_weights():
    """The shared init_weights initializes every embedding once."""


class CanonDepthLM(ReferenceLM):
    def __init__(self, config):
        super().__init__(config)
        self.depth_heads = config.option('depth_heads')
        self.granularity = config.option('source_granularity')
        self.depth_queries = nn.Parameter(torch.empty(2 * config.n_layer + 1, config.width))
        self.canon = nn.ModuleList(Canon(config.width, config.option('canon_size')) for _ in range(2 * config.n_layer))
        for module in self.modules():
            if isinstance(module, nn.Embedding):
                module.reset_parameters = _drawn_by_init_weights

    def _depth_read(self, query, sources, keys):
        if self.depth_heads == 1:
            return depth_attention(query, sources, keys)
        heads = self.depth_heads
        width = self.config.width // heads
        q = query.to(keys[0].dtype).reshape(heads, width)
        logits = torch.stack([(key.reshape(*key.shape[:-1], heads, width) * q).sum(-1) for key in keys]).float()
        weights = logits.softmax(0).to(sources[0].dtype)
        out = weights[0, ..., None] * sources[0].reshape(*sources[0].shape[:-1], heads, width)
        for weight, value in zip(weights[1:], sources[1:]):
            out = out + weight[..., None] * value.reshape(*value.shape[:-1], heads, width)
        return out.flatten(-2)

    def _ordinary_hidden(self, idx):
        x = self._embed_reference(idx)
        blocks, keys, partial = [], [], x
        for k in range(2 * self.config.n_layer):
            key = norm(partial)
            h = self._depth_read(self.depth_queries[k], blocks + [partial], keys + [key])
            self._record('depth_read', sublayer=k, sources=len(blocks) + 1, tokens=idx.numel())
            if k % self.granularity == 0:
                blocks.append(partial)
                keys.append(key)
                partial = None
            h = self.canon[k](h)
            out = self._attention_branch(h, idx, k // 2) if k % 2 == 0 else self.transformer.h[k // 2].mlp(norm(h))
            partial = out if partial is None else partial + out
        h = self._depth_read(self.depth_queries[-1], blocks + [partial], keys + [norm(partial)])
        self._record('depth_read', sublayer=2 * self.config.n_layer, sources=len(blocks) + 1, tokens=idx.numel())
        return h, None


def mixer(config, layer):
    return CanonAttention(config, layer)


def ffn(config, layer):
    return CanonMLP(config)


model = CanonDepthLM
frozen = ('resid_lambdas', 'x0_lambdas', 'backout_lambda')


@torch.no_grad()
def init(model):
    for module in model.modules():
        if isinstance(module, Canon):
            nn.init.zeros_(module.weight)
        if isinstance(module, CanonAttention) and module.output_gate is not None:
            s = (3 / model.config.width) ** 0.5
            nn.init.uniform_(module.output_gate.weight, -s, s)


def check(config):
    if config.option('ffn_kind') not in ('relu2', 'swiglu'):
        raise ValueError('ffn_kind must be relu2 or swiglu')
    if config.option('canon_size') not in (2, 4, 8, 12):
        raise ValueError('canon_size must be 2, 4, 8, or 12')
    if config.option('depth_heads') not in (1, 2, 4, 8, 12) or config.width % config.option('depth_heads'):
        raise ValueError('invalid depth head count')
    if config.option('source_granularity') not in (1, 2, 4, 6):
        raise ValueError('invalid source granularity')
    if not isinstance(config.option('head_gate'), bool):
        raise ValueError('head_gate must be boolean')
