"""AttnRes with Canon layers at all four positions.

Parent: attnres (att_000025). Canon-A/C on sublayer inputs as in attnres_canon, plus Canon-B on q, k, v and Canon-D on the MLP hidden.
"""
import torch
from torch import nn
from torch.nn import functional as F

from nanochat.flash_attention import flash_attn
from nexttransformer.models.backbone import CausalSelfAttention, GPTConfig, Linear, apply_rotary_emb, norm
from nexttransformer.models.lm import ReferenceLM
from nexttransformer.models.residual import depth_attention
from nextt_gym.upstream_model import model_config, set_stage, stage_for_step  # noqa: F401


class Canon(nn.Module):
    """Causal depthwise convolution over the last `size` positions, added to its input (Canon layer).

    Allen-Zhu, "Physics of Language Models: Part 4.1" (2025). The weights start at zero, so the layer starts as the
    identity.
    """

    def __init__(self, width, size=4):
        super().__init__()
        self.weight = nn.Parameter(torch.empty(width, 1, size))
        self.size = size

    def forward(self, x):
        y = F.conv1d(F.pad(x.transpose(1, 2), (self.size - 1, 0)), self.weight.to(x.dtype), groups=x.shape[-1])
        return x + y.transpose(1, 2)


class VariantAttention(CausalSelfAttention):
    """B01 attention with an optional sigmoid output gate and optional Canon layers on q, k and v (Canon-B).

    The gate (Gated Attention, Qiu et al., 2025) is computed from the attention input and multiplies the attention
    output before the output projection: one gate per head ("head") or per channel ("element"). Its projection
    starts at zero, so every gate starts at 0.5; the output projection itself starts at zero (B01).
    """

    def __init__(self, config, layer_idx, gate, canon_qkv):
        super().__init__(config, layer_idx)
        self.gate = gate
        if gate: self.c_gate = Linear(self.n_embd, self.n_head if gate == "head" else self.n_embd, bias=False)
        self.canon_qkv = nn.ModuleList(Canon(self.n_embd) for _ in range(3)) if canon_qkv else None

    def forward(self, x, ve, cos_sin, window_size, kv_cache):
        if kv_cache is not None:
            raise NotImplementedError("training and full-sequence evaluation only")
        B, T, C = x.size()
        q, k, v = self.c_q(x), self.c_k(x), self.c_v(x)
        if self.canon_qkv is not None:
            q, k, v = (layer(t) for layer, t in zip(self.canon_qkv, (q, k, v)))
        q = q.view(B, T, self.n_head, self.head_dim)
        k = k.view(B, T, self.n_kv_head, self.head_dim)
        v = v.view(B, T, self.n_kv_head, self.head_dim)
        if ve is not None:
            ve = ve.view(B, T, self.n_kv_head, self.head_dim)
            v = v + 3 * torch.sigmoid(self.ve_gate(x[..., :self.ve_gate_channels])).unsqueeze(-1) * ve
        cos, sin = cos_sin
        q, k = apply_rotary_emb(q, cos, sin), apply_rotary_emb(k, cos, sin)
        q, k = norm(q) * 1.2, norm(k) * 1.2
        y = flash_attn.flash_attn_func(q, k, v, causal=True, window_size=window_size)
        if self.gate:
            y = y * torch.sigmoid(self.c_gate(x)).view(B, T, self.n_head, -1).to(y.dtype)
        return self.c_proj(y.contiguous().view(B, T, -1))


class CanonMLP(nn.Module):
    """The squared-ReLU MLP with a Canon layer on its hidden activations, before the nonlinearity (Canon-D)."""

    def __init__(self, c):
        super().__init__()
        self.c_fc = Linear(c.width, c.mlp_width, bias=False)
        self.c_proj = Linear(c.mlp_width, c.width, bias=False)
        self.canon = Canon(c.mlp_width)

    def forward(self, x):
        return self.c_proj(F.relu(self.canon(self.c_fc(x))).square())


class VariantLM(ReferenceLM):
    """Block AttnRes (ReferenceLM, method attnres) with the components switched on by the class constants."""

    GATE = None            # None, "head" or "element"
    CANON_INPUTS = False   # Canon-A/C: on the input of every attention and MLP sublayer
    CANON_QKV = False      # Canon-B: on q, k and v
    CANON_MLP = False      # Canon-D: on the MLP hidden activations

    def __init__(self, config):
        super().__init__(config)
        native = GPTConfig(sequence_len=config.sequence_len, vocab_size=config.vocab_size, n_layer=config.n_layer,
                           n_embd=config.width, n_head=config.heads, n_kv_head=config.heads,
                           window_pattern=config.window_pattern)
        for i, block in enumerate(self.transformer.h):
            if self.GATE or self.CANON_QKV: block.attn = VariantAttention(native, i, self.GATE, self.CANON_QKV)
            if self.CANON_MLP: block.mlp = CanonMLP(config)
        if self.CANON_INPUTS: self.canon = nn.ModuleList(Canon(config.width) for _ in range(2 * config.n_layer))

    @torch.no_grad()
    def init_weights(self):
        super().init_weights()
        for module in self.modules():
            if isinstance(module, Canon): nn.init.zeros_(module.weight)
            if isinstance(module, VariantAttention) and module.gate: nn.init.zeros_(module.c_gate.weight)

    def _residual_hidden(self, idx):
        """Block AttnRes trunk of ReferenceLM, with a Canon layer on each sublayer input if CANON_INPUTS."""
        if not self.CANON_INPUTS: return super()._residual_hidden(idx)
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


class CandidateLM(VariantLM):
    CANON_INPUTS = True
    CANON_QKV = True
    CANON_MLP = True


def build(config):
    """As nexttransformer.models.lm.build_model, for this subclass; the caller has seeded the generator."""
    c = model_config(config)
    device = torch.get_default_device()
    with torch.device("meta"):
        model = CandidateLM(c)
    if str(device) == "meta":
        return model
    model.to_empty(device=device)
    for module in model.modules():
        if isinstance(module, nn.Linear): module.reset_parameters()
    model.init_weights()
    return model
